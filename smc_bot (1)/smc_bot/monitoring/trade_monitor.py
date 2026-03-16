"""
monitoring/trade_monitor.py
============================
Monitora le posizioni aperte e gestisce:
- Rilevamento SL / TP hit (per paper trading e verifica MT5)
- Trailing Stop Loss (opzionale)
- Breakeven automatico (sposta SL a entry quando +1R)
- Notifiche via Telegram per eventi di trade

Viene chiamato nel loop principale del bot.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import Settings
from core.models import Trade, Direction, TradeStatus
from utils.pip_utils import pip_size, to_pips
from monitoring.alerts import get_alerter

log = logging.getLogger("monitor")


class TradeMonitor:

    def __init__(self, settings: Settings):
        self.s        = settings
        self.alerter  = get_alerter()
        self._be_done : set = set()   # Trade già spostati a breakeven

    # ── Entry point chiamato ogni loop ───────────────────────────────────────

    def check_all(
        self,
        open_trades: Dict[str, Trade],
        current_prices: Dict[str, float],
    ) -> List[Trade]:
        """
        Controlla tutte le posizioni aperte.
        
        Args:
            open_trades:    dict {trade_id → Trade}
            current_prices: dict {symbol → prezzo corrente}
        
        Returns:
            Lista dei Trade che devono essere chiusi (SL o TP colpito).
        """
        to_close: List[Trade] = []

        for tid, trade in list(open_trades.items()):
            price = current_prices.get(trade.symbol)
            if price is None:
                continue

            # Controlla SL / TP hit
            hit = self._check_sl_tp(trade, price)
            if hit:
                to_close.append(trade)
                continue

            # Trailing SL (se abilitato)
            if self.s.TRAILING_SL_ENABLED:
                self._apply_trailing_sl(trade, price)

            # Breakeven automatico
            if self.s.BREAKEVEN_ENABLED:
                self._apply_breakeven(trade, price)

        return to_close

    # ── SL / TP detection ────────────────────────────────────────────────────

    def _check_sl_tp(self, trade: Trade, price: float) -> bool:
        """
        Ritorna True se SL o TP sono stati colpiti dal prezzo corrente.
        In live MT5 questo avviene automaticamente; qui è per paper trading
        e come doppio controllo.
        """
        if trade.direction == Direction.LONG:
            if price <= trade.sl:
                log.info(f"[{trade.trade_id}] SL colpito @ {price:.5f}")
                trade.outcome    = "loss"
                trade.close_price = price
                trade.close_time  = datetime.now()
                trade.status      = TradeStatus.CLOSED
                self._calc_pnl(trade)
                self.alerter.trade_closed(trade)
                return True
            if price >= trade.tp:
                log.info(f"[{trade.trade_id}] TP colpito @ {price:.5f}")
                trade.outcome    = "win"
                trade.close_price = price
                trade.close_time  = datetime.now()
                trade.status      = TradeStatus.CLOSED
                self._calc_pnl(trade)
                self.alerter.trade_closed(trade)
                return True

        else:  # SHORT
            if price >= trade.sl:
                log.info(f"[{trade.trade_id}] SL colpito @ {price:.5f}")
                trade.outcome    = "loss"
                trade.close_price = price
                trade.close_time  = datetime.now()
                trade.status      = TradeStatus.CLOSED
                self._calc_pnl(trade)
                self.alerter.trade_closed(trade)
                return True
            if price <= trade.tp:
                log.info(f"[{trade.trade_id}] TP colpito @ {price:.5f}")
                trade.outcome    = "win"
                trade.close_price = price
                trade.close_time  = datetime.now()
                trade.status      = TradeStatus.CLOSED
                self._calc_pnl(trade)
                self.alerter.trade_closed(trade)
                return True

        return False

    # ── Trailing SL ──────────────────────────────────────────────────────────

    def _apply_trailing_sl(self, trade: Trade, price: float):
        """
        Trailing Stop Loss: sposta lo SL seguendo il prezzo a distanza fissa.
        
        Distanza = TRAILING_SL_PIPS pips dallo SL attuale.
        Lo SL si sposta solo quando il prezzo avanza (mai indietro).
        """
        ps      = pip_size(trade.symbol, price)
        trail   = self.s.TRAILING_SL_PIPS * ps

        if trade.direction == Direction.LONG:
            new_sl = price - trail
            if new_sl > trade.sl:
                old = trade.sl
                trade.sl = new_sl
                log.debug(f"[{trade.trade_id}] Trailing SL: {old:.5f} → {new_sl:.5f}")

        else:  # SHORT
            new_sl = price + trail
            if new_sl < trade.sl:
                old = trade.sl
                trade.sl = new_sl
                log.debug(f"[{trade.trade_id}] Trailing SL: {old:.5f} → {new_sl:.5f}")

    # ── Breakeven ────────────────────────────────────────────────────────────

    def _apply_breakeven(self, trade: Trade, price: float):
        """
        Sposta lo SL a breakeven (entry) quando il profitto raggiunge 1R.
        
        1R = la distanza originale SL–entry.
        Si attiva una sola volta per trade.
        """
        if trade.trade_id in self._be_done:
            return

        ps       = pip_size(trade.symbol, price)
        risk     = abs(trade.entry - trade.sl)   # distanza originale
        be_level = self.s.BREAKEVEN_TRIGGER_R    # default 1.0 (1R)

        if trade.direction == Direction.LONG:
            target = trade.entry + risk * be_level
            if price >= target:
                trade.sl = trade.entry + ps     # SL a entry + 1 pip (small profit)
                self._be_done.add(trade.trade_id)
                log.info(f"[{trade.trade_id}] Breakeven attivato. SL → {trade.sl:.5f}")

        else:  # SHORT
            target = trade.entry - risk * be_level
            if price <= target:
                trade.sl = trade.entry - ps
                self._be_done.add(trade.trade_id)
                log.info(f"[{trade.trade_id}] Breakeven attivato. SL → {trade.sl:.5f}")

    # ── PnL calculation ──────────────────────────────────────────────────────

    def _calc_pnl(self, trade: Trade):
        """Calcola PnL in USD e % del capitale."""
        from utils.pip_utils import pip_value_per_lot
        if trade.close_price is None:
            return

        ps      = pip_size(trade.symbol, trade.entry)
        pv      = pip_value_per_lot(trade.symbol, trade.entry)

        if trade.direction == Direction.LONG:
            pips = (trade.close_price - trade.entry) / ps
        else:
            pips = (trade.entry - trade.close_price) / ps

        trade.pnl_usd = pips * pv * trade.lot_size
        trade.pnl_pct = trade.pnl_usd / self.s.INITIAL_BALANCE

    # ── Utility ──────────────────────────────────────────────────────────────

    def get_current_prices(self, symbols: list, feed) -> Dict[str, float]:
        """
        Ottiene i prezzi correnti per tutti i simboli.
        Compatibile con MarketDataFeed (live e simulation).
        """
        prices = {}
        for sym in symbols:
            try:
                df = feed.get_ohlcv(sym, "1M", 2)
                if df is not None and len(df) > 0:
                    prices[sym] = float(df["close"].iloc[-1])
            except Exception as e:
                log.debug(f"Prezzo {sym} non disponibile: {e}")
        return prices
