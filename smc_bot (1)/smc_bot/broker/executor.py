"""
broker/executor.py
==================
Apre e chiude ordini tramite MT5 o paper trading.
"""

import logging
import uuid
from datetime import datetime
from typing import Optional

from config.settings import Settings
from core.models import Direction, Trade, TradeSetup, TradeStatus

log = logging.getLogger("executor")


class TradeExecutor:

    def __init__(self, settings: Settings):
        self.s        = settings
        self._mt5     = None
        self._paper   : dict = {}

    # ── Connessione ──────────────────────────────────────────────────────────

    def connect(self):
        if self.s.BROKER == "mt5":
            self._init_mt5()
        else:
            log.info("Paper trading attivo.")

    def _init_mt5(self):
        try:
            import MetaTrader5 as mt5
            ok = mt5.initialize(
                login=self.s.MT5_LOGIN,
                password=self.s.MT5_PASSWORD,
                server=self.s.MT5_SERVER,
            )
            if ok:
                self._mt5 = mt5
                log.info("MT5 executor connesso.")
            else:
                log.error(f"MT5 errore: {mt5.last_error()} → paper mode")
        except ImportError:
            log.warning("MetaTrader5 non installato → paper mode")

    def disconnect(self):
        if self._mt5:
            self._mt5.shutdown()

    # ── Apertura ─────────────────────────────────────────────────────────────

    def open_trade(self, setup: TradeSetup) -> Optional[Trade]:
        tid = str(uuid.uuid4())[:8].upper()
        trade = Trade(
            trade_id=tid,
            symbol=setup.symbol,
            direction=setup.direction,
            entry=setup.entry,
            sl=setup.sl,
            tp=setup.tp,
            lot_size=setup.lot_size,
            setup=setup,
        )

        ok = self._send_order(trade)
        if ok:
            trade.status    = TradeStatus.OPEN
            trade.open_time = datetime.now()
            log.info(f"Trade aperto: {trade}")
            return trade

        trade.status = TradeStatus.REJECTED
        log.error(f"Trade rifiutato: {tid}")
        return None

    def _send_order(self, trade: Trade) -> bool:
        if self._mt5:
            return self._mt5_order(trade)
        return self._paper_order(trade)

    def _mt5_order(self, trade: Trade) -> bool:
        mt5  = self._mt5
        ot   = mt5.ORDER_TYPE_BUY if trade.direction == Direction.LONG else mt5.ORDER_TYPE_SELL
        tick = mt5.symbol_info_tick(trade.symbol)
        if not tick:
            return False
        price = tick.ask if trade.direction == Direction.LONG else tick.bid

        req = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      trade.symbol,
            "volume":      trade.lot_size,
            "type":        ot,
            "price":       price,
            "sl":          trade.sl,
            "tp":          trade.tp,
            "deviation":   20,
            "magic":       20240101,
            "comment":     f"SMC_{trade.trade_id}",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(req)
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        log.error(f"MT5 retcode {res.retcode}: {res.comment}")
        return False

    def _paper_order(self, trade: Trade) -> bool:
        self._paper[trade.trade_id] = trade
        log.info(f"[PAPER] {trade}")
        return True

    # ── Chiusura ─────────────────────────────────────────────────────────────

    def close_trade(self, trade: Trade, price: float) -> bool:
        if self._mt5:
            return self._mt5_close(trade, price)
        if trade.trade_id in self._paper:
            del self._paper[trade.trade_id]
            return True
        return False

    def _mt5_close(self, trade: Trade, price: float) -> bool:
        mt5 = self._mt5
        ot  = mt5.ORDER_TYPE_SELL if trade.direction == Direction.LONG else mt5.ORDER_TYPE_BUY
        tick = mt5.symbol_info_tick(trade.symbol)
        cp   = (tick.bid if trade.direction == Direction.LONG else tick.ask) if tick else price
        req  = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": trade.symbol,
            "volume": trade.lot_size, "type": ot, "price": cp,
            "deviation": 20, "magic": 20240101,
            "comment": f"CLOSE_{trade.trade_id}",
        }
        res = mt5.order_send(req)
        return res.retcode == mt5.TRADE_RETCODE_DONE

    def get_balance(self) -> float:
        if self._mt5:
            info = self._mt5.account_info()
            return info.balance if info else 0.0
        return self.s.INITIAL_BALANCE
