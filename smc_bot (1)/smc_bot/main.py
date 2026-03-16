"""
main.py  –  Entry point del bot SMC.
Integra: market data, struttura, AI filter, risk, esecuzione,
         monitor posizioni aperte, trailing SL, breakeven, alerting.
"""

import asyncio
import signal
import sys
from datetime import datetime, timedelta

from config.settings import Settings
from core.market_data import MarketDataFeed
from core.data_processor import DataProcessor
from core.market_structure import MarketStructureEngine
from core.pattern_detector import PatternDetector
from ai.filter import AIFilter
from ai.trainer import ModelTrainer
from risk.manager import RiskManager
from broker.executor import TradeExecutor
from monitoring.dashboard import Dashboard
from monitoring.alerts import get_alerter
from monitoring.trade_monitor import TradeMonitor
from utils.logger import get_logger

log = get_logger("main")


class SMCBot:

    def __init__(self):
        self.s        = Settings()
        self._running = False

        # Moduli core
        self.feed     = MarketDataFeed(self.s)
        self.proc     = DataProcessor(self.s)
        self.struct   = MarketStructureEngine(self.s)
        self.pattern  = PatternDetector(self.s)
        self.ai       = AIFilter(self.s)
        self.risk     = RiskManager(self.s)
        self.executor = TradeExecutor(self.s)

        # Monitoring
        self.dash     = Dashboard(self.s)
        self.alerter  = get_alerter()
        self.monitor  = TradeMonitor(self.s)
        self.trainer  = ModelTrainer(self.s)

        # Heartbeat
        self._last_heartbeat = datetime.now()
        self._HEARTBEAT_INTERVAL = timedelta(hours=1)

        log.info("SMC Bot inizializzato.")

    # ── Avvio ────────────────────────────────────────────────────────────────

    async def run(self):
        self._running = True
        self.feed.connect()
        self.executor.connect()

        # Aggiorna balance reale dal broker
        balance = self.executor.get_balance()
        if balance > 0:
            self.risk.balance = balance

        self.alerter.bot_started(self.s.SYMBOLS)
        log.info(
            "Bot avviato | Balance %.2f | Loop %.0fs | Simboli: %s",
            self.risk.balance, self.s.LOOP_INTERVAL, ", ".join(self.s.SYMBOLS)
        )

        while self._running:
            try:
                await self._loop()
                await asyncio.sleep(self.s.LOOP_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Errore loop: {e}", exc_info=True)
                self.alerter.critical_error(str(e))
                await asyncio.sleep(10)

        self._shutdown()

    async def _loop(self):
        """Un ciclo completo: analisi + gestione posizioni aperte."""

        # ── 1. Controlla posizioni aperte ────────────────────────────────────
        await self._check_open_positions()

        # ── 2. Cerca nuovi setup ─────────────────────────────────────────────
        for sym in self.s.SYMBOLS:
            try:
                await self._process_symbol(sym)
            except Exception as e:
                log.error(f"Errore su {sym}: {e}", exc_info=True)

        # ── 3. Aggiorna dashboard ─────────────────────────────────────────────
        self.dash.update(self.risk.stats())

        # ── 4. Heartbeat Telegram ────────────────────────────────────────────
        if datetime.now() - self._last_heartbeat > self._HEARTBEAT_INTERVAL:
            self.alerter.heartbeat(self.risk.stats())
            self._last_heartbeat = datetime.now()

    # ── Analisi singolo simbolo ──────────────────────────────────────────────

    async def _process_symbol(self, sym: str):
        s = self.s

        # Dati multi-timeframe
        df_h = self.feed.get_ohlcv(sym, s.TF_HTF, s.CANDLES_HTF)
        df_m = self.feed.get_ohlcv(sym, s.TF_MTF, s.CANDLES_MTF)
        df_l = self.feed.get_ohlcv(sym, s.TF_LTF, s.CANDLES_LTF)

        if df_h is None or df_m is None or df_l is None:
            log.warning(f"{sym}: dati non disponibili.")
            return

        # Pipeline analisi
        htf   = self.struct.analyze(self.proc.process(df_h), sym, s.TF_HTF)
        mtf   = self.struct.analyze(self.proc.process(df_m), sym, s.TF_MTF)
        ltf_p = self.proc.process(df_l)
        ltf   = self.struct.analyze(ltf_p, sym, s.TF_LTF)

        setup = self.pattern.detect(sym, htf, mtf, ltf, ltf_p)
        if setup is None:
            log.debug(f"{sym}: nessun setup.")
            return

        # Filtro AI
        setup.ai_prob = self.ai.evaluate(setup)
        if setup.ai_prob < s.AI_THRESHOLD:
            log.info(f"{sym}: AI rifiuta setup (prob={setup.ai_prob:.0%})")
            return

        log.info(f"{sym}: setup {setup.direction.value} | AI={setup.ai_prob:.0%} | R:R={setup.rr:.2f}")

        # Risk management
        if not self.risk.can_open(setup):
            return

        setup.lot_size = self.risk.lot_size(setup)

        # Esecuzione
        trade = self.executor.open_trade(setup)
        if trade:
            self.risk.register(trade)
            self.dash.log_trade(trade)
            self.alerter.trade_opened(trade)

    # ── Monitoraggio posizioni aperte ────────────────────────────────────────

    async def _check_open_positions(self):
        """Controlla SL/TP hit, trailing SL, breakeven su tutte le posizioni."""
        if not self.risk.open_trades:
            return

        prices = self.monitor.get_current_prices(
            list({t.symbol for t in self.risk.open_trades.values()}),
            self.feed
        )

        to_close = self.monitor.check_all(self.risk.open_trades, prices)

        for trade in to_close:
            # Chiude sul broker
            close_price = trade.close_price or prices.get(trade.symbol, trade.entry)
            self.executor.close_trade(trade, close_price)

            # Aggiorna risk manager
            self.risk.close_trade(trade)

            # Raccoglie dati per training AI
            self.trainer.append_trade(trade, trade.outcome)

            log.info(
                f"Trade chiuso [{trade.trade_id}] {trade.symbol} "
                f"{trade.outcome.upper()} | PnL {trade.pnl_usd:+.2f} USD"
            )

    # ── Shutdown ─────────────────────────────────────────────────────────────

    def _shutdown(self):
        log.info("Shutdown in corso…")
        self.executor.disconnect()
        self.dash.save_report()
        final_stats = self.risk.stats()
        self.alerter.bot_stopped(final_stats)
        log.info(
            "Bot fermato | Balance %.2f | PnL %.2f | Trade %d | WR %.1f%%",
            final_stats["balance"],
            final_stats.get("total_pnl", 0),
            final_stats["total_trades"],
            final_stats["win_rate"] * 100,
        )

    def stop(self):
        self._running = False


# ── Signal handler ───────────────────────────────────────────────────────────

_bot: SMCBot | None = None


def _handle_signal(sig, frame):
    log.info(f"Segnale {sig} ricevuto → arresto...")
    if _bot:
        _bot.stop()
    sys.exit(0)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    _bot = SMCBot()
    asyncio.run(_bot.run())
