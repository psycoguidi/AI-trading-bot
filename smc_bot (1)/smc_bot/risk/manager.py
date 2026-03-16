"""
risk/manager.py
===============
Gestione rischio:
  - Position sizing (1% capitale)
  - Max trade aperti (3)
  - Stop giornaliero (-3%)
  - R:R minimo
  - No duplicati per simbolo
"""

import logging
from datetime import datetime, date
from typing import Dict, List

from config.settings import Settings
from core.models import Trade, TradeSetup, TradeStatus
from utils.pip_utils import pip_size, to_pips, pip_value_per_lot

log = logging.getLogger("risk")


class RiskManager:

    def __init__(self, settings: Settings):
        self.s          = settings
        self.balance    = settings.INITIAL_BALANCE
        self.open_trades: Dict[str, Trade]  = {}
        self.closed_trades: List[Trade]     = []
        self.daily_pnl  = 0.0
        self._day       = datetime.now().date()

    # ── Gate pre-trade ───────────────────────────────────────────────────────

    def can_open(self, setup: TradeSetup) -> bool:
        self._maybe_reset_daily()

        if len(self.open_trades) >= self.s.MAX_OPEN_TRADES:
            log.warning(f"Max trade aperti ({self.s.MAX_OPEN_TRADES}). Bloccato.")
            return False

        if self.daily_pnl <= -(self.balance * self.s.MAX_DAILY_LOSS):
            log.warning(f"Daily loss limit raggiunto ({self.daily_pnl:.2f}). Stop trading.")
            return False

        for t in self.open_trades.values():
            if t.symbol == setup.symbol:
                log.debug(f"Trade già aperto su {setup.symbol}.")
                return False

        if setup.rr < self.s.MIN_RR:
            log.debug(f"R:R {setup.rr:.2f} insufficiente.")
            return False

        return True

    # ── Position sizing ──────────────────────────────────────────────────────

    def lot_size(self, setup: TradeSetup) -> float:
        """
        Lotto = (balance × risk%) / (SL_pips × pip_value_per_lot)
        """
        risk_usd = self.balance * self.s.RISK_PER_TRADE
        sl_pips  = to_pips(abs(setup.entry - setup.sl), setup.symbol, setup.entry)
        pv       = pip_value_per_lot(setup.symbol, setup.entry)

        if sl_pips <= 0 or pv <= 0:
            return 0.01

        lots = risk_usd / (sl_pips * pv)
        lots = round(lots, 2)
        lots = max(0.01, min(lots, 10.0))

        log.debug(
            f"{setup.symbol}: lot={lots} | risk={risk_usd:.2f} USD | "
            f"sl={sl_pips:.1f}p | pv={pv}"
        )
        return lots

    # ── Registro ─────────────────────────────────────────────────────────────

    def register(self, trade: Trade):
        self.open_trades[trade.trade_id] = trade

    def close(self, trade_id: str, close_price: float):
        if trade_id not in self.open_trades:
            return
        trade = self.open_trades.pop(trade_id)
        trade.close_price = close_price
        trade.close_time  = datetime.now()
        trade.status      = TradeStatus.CLOSED

        ps = pip_size(trade.symbol, trade.entry)
        pv = pip_value_per_lot(trade.symbol, trade.entry)

        if trade.direction.value == "LONG":
            pips = (close_price - trade.entry) / ps
        else:
            pips = (trade.entry - close_price) / ps

        trade.pnl_usd = pips * pv * trade.lot_size
        trade.pnl_pct = trade.pnl_usd / self.balance
        trade.outcome = "win" if trade.pnl_usd > 0 else "loss"

        self.daily_pnl  += trade.pnl_usd
        self.balance    += trade.pnl_usd
        self.closed_trades.append(trade)

        sign = "✅" if trade.pnl_usd > 0 else "❌"
        log.info(f"{sign} Chiuso {trade_id}: {trade.pnl_usd:+.2f} USD ({trade.pnl_pct:+.2%})")

    # ── Statistiche ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = len(self.closed_trades)
        wins  = sum(1 for t in self.closed_trades if t.pnl_usd > 0)
        return {
            "balance":      self.balance,
            "daily_pnl":    self.daily_pnl,
            "open_trades":  len(self.open_trades),
            "total_trades": total,
            "wins":         wins,
            "losses":       total - wins,
            "win_rate":     wins / total if total else 0.0,
            "total_pnl":    sum(t.pnl_usd for t in self.closed_trades),
        }

    def _maybe_reset_daily(self):
        today = datetime.now().date()
        if today != self._day:
            self.daily_pnl = 0.0
            self._day      = today
