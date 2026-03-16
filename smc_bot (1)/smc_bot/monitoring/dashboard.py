"""monitoring/dashboard.py – Pannello statistiche + salvataggio CSV."""

import csv
import logging
import os
from datetime import datetime
from typing import List

from config.settings import Settings
from core.models import Trade

log = logging.getLogger("dashboard")


class Dashboard:

    def __init__(self, settings: Settings):
        self.s       = settings
        self.log_dir = settings.__class__.__dataclass_fields__ and "logs" or "logs"
        self.log_dir = "logs"
        os.makedirs(self.log_dir, exist_ok=True)
        self._records: List[dict] = []
        self._last_print = datetime.now()

    def update(self, stats: dict):
        now = datetime.now()
        if (now - self._last_print).seconds >= 60:
            self._print(stats)
            self._last_print = now

    def _print(self, s: dict):
        w = s.get("win_rate", 0)
        print(
            f"\n{'─'*52}\n"
            f"  SMC BOT  {datetime.now():%H:%M:%S}\n"
            f"{'─'*52}\n"
            f"  💰 Balance      ${s.get('balance',0):>12,.2f}\n"
            f"  📊 Daily PnL    ${s.get('daily_pnl',0):>+12.2f}\n"
            f"  📈 Total PnL    ${s.get('total_pnl',0):>+12.2f}\n"
            f"  🔄 Open         {s.get('open_trades',0):>12}\n"
            f"  📋 Total trades {s.get('total_trades',0):>12}\n"
            f"  🎯 Win rate     {w:>11.1%}\n"
            f"{'─'*52}"
        )

    def log_trade(self, trade: Trade):
        self._records.append({
            "id":        trade.trade_id,
            "symbol":    trade.symbol,
            "direction": trade.direction.value,
            "entry":     trade.entry,
            "sl":        trade.sl,
            "tp":        trade.tp,
            "lot_size":  trade.lot_size,
            "ai_prob":   f"{trade.setup.ai_prob:.2%}",
            "rr":        f"{trade.setup.rr:.2f}",
            "open_time": datetime.now().isoformat(),
        })

    def save_report(self):
        if not self._records:
            return
        path = os.path.join(self.log_dir,
                             f"trades_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self._records[0].keys())
            w.writeheader()
            w.writerows(self._records)
        log.info(f"Report salvato: {path}")
