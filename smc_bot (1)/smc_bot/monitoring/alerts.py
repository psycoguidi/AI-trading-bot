"""
monitoring/alerts.py
=====================
Sistema di notifiche via Telegram.

SETUP:
1. Crea un bot con @BotFather → ottieni TELEGRAM_TOKEN
2. Invia /start al bot, poi vai su:
   https://api.telegram.org/bot<TOKEN>/getUpdates
   per trovare il tuo CHAT_ID
3. Imposta le variabili d'ambiente:
   TELEGRAM_TOKEN=xxxxx
   TELEGRAM_CHAT_ID=123456

Le notifiche vengono inviate per:
- Trade aperto  (entry, SL, TP, AI prob, R:R)
- Trade chiuso  (outcome, PnL)
- Stop giornaliero raggiunto
- Errore critico del bot
- Heartbeat ogni ora (il bot è vivo)
"""

import logging
import os
import time
from datetime import datetime
from typing import Optional

from core.models import Trade, Direction

log = logging.getLogger("alerts")

# ── Emoji per leggibilità rapida su mobile ────────────────────────────────────
_DIR  = {Direction.LONG: "📈 LONG", Direction.SHORT: "📉 SHORT"}
_WIN  = "✅"
_LOSS = "❌"
_WARN = "⚠️"
_INFO = "ℹ️"
_HEART = "💓"


class TelegramAlerter:
    """
    Invia messaggi Telegram in modo sincrono (requests) o asincrono (httpx).
    Usa rate limiting per evitare flood (max 1 msg/secondo).
    """

    def __init__(self):
        self.token   = os.getenv("TELEGRAM_TOKEN", "")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
        self.enabled = bool(self.token and self.chat_id)
        self._last_sent = 0.0
        self._min_interval = 1.0   # secondi tra messaggi

        if self.enabled:
            log.info("Telegram alerting attivo.")
        else:
            log.info("Telegram non configurato (TELEGRAM_TOKEN / TELEGRAM_CHAT_ID mancanti).")

    # ── API pubblica ─────────────────────────────────────────────────────────

    def trade_opened(self, trade: Trade):
        """Notifica apertura trade."""
        rr  = trade.setup.rr
        sym = trade.symbol
        msg = (
            f"🚀 *TRADE APERTO*\n"
            f"{'─' * 25}\n"
            f"*Simbolo* : `{sym}`\n"
            f"*Direzione*: {_DIR[trade.direction]}\n"
            f"*Entry*   : `{trade.entry:.5f}`\n"
            f"*SL*      : `{trade.sl:.5f}`\n"
            f"*TP*      : `{trade.tp:.5f}`\n"
            f"*R:R*     : `{rr:.2f}`\n"
            f"*AI Prob* : `{trade.setup.ai_prob:.0%}`\n"
            f"*Lotti*   : `{trade.lot_size:.2f}`\n"
            f"*Ora*     : `{datetime.now().strftime('%H:%M:%S')}`"
        )
        self._send(msg)

    def trade_closed(self, trade: Trade):
        """Notifica chiusura trade con PnL."""
        icon = _WIN if trade.pnl_usd > 0 else _LOSS
        msg = (
            f"{icon} *TRADE CHIUSO*\n"
            f"{'─' * 25}\n"
            f"*Simbolo*  : `{trade.symbol}`\n"
            f"*Outcome*  : `{trade.outcome.upper()}`\n"
            f"*PnL*      : `{trade.pnl_usd:+.2f} USD`\n"
            f"*PnL%*     : `{trade.pnl_pct:+.2%}`\n"
            f"*Durata*   : `{self._duration(trade)}`\n"
            f"*Ora*      : `{datetime.now().strftime('%H:%M:%S')}`"
        )
        self._send(msg)

    def daily_stop(self, pnl: float, balance: float):
        """Notifica stop giornaliero raggiunto."""
        msg = (
            f"{_WARN} *STOP GIORNALIERO*\n"
            f"{'─' * 25}\n"
            f"Perdita giornaliera: `{pnl:+.2f} USD`\n"
            f"Balance attuale    : `{balance:,.2f} USD`\n"
            f"*Il bot ha smesso di tradare per oggi.*"
        )
        self._send(msg)

    def critical_error(self, error: str):
        """Notifica errore critico."""
        msg = (
            f"🔴 *ERRORE CRITICO*\n"
            f"{'─' * 25}\n"
            f"`{error[:300]}`\n"
            f"Ora: `{datetime.now().strftime('%H:%M:%S')}`"
        )
        self._send(msg)

    def heartbeat(self, stats: dict):
        """Heartbeat orario con statistiche."""
        msg = (
            f"{_HEART} *BOT ATTIVO*  `{datetime.now().strftime('%H:%M')}`\n"
            f"{'─' * 25}\n"
            f"💰 Balance : `{stats.get('balance', 0):,.2f} USD`\n"
            f"📊 PnL oggi: `{stats.get('daily_pnl', 0):+.2f} USD`\n"
            f"🔄 Trade aperti : `{stats.get('open_trades', 0)}`\n"
            f"🎯 Win Rate     : `{stats.get('win_rate', 0):.1%}`\n"
            f"📋 Totale trade : `{stats.get('total_trades', 0)}`"
        )
        self._send(msg)

    def bot_started(self, symbols: list):
        """Notifica avvio bot."""
        msg = (
            f"🟢 *BOT AVVIATO*\n"
            f"{'─' * 25}\n"
            f"*Simboli*: `{', '.join(symbols)}`\n"
            f"*Ora*    : `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        self._send(msg)

    def bot_stopped(self, stats: dict):
        """Notifica stop bot."""
        msg = (
            f"🔴 *BOT FERMATO*\n"
            f"{'─' * 25}\n"
            f"💰 Balance finale : `{stats.get('balance', 0):,.2f} USD`\n"
            f"📊 PnL sessione   : `{stats.get('total_pnl', 0):+.2f} USD`\n"
            f"📋 Trade eseguiti : `{stats.get('total_trades', 0)}`\n"
            f"🎯 Win Rate       : `{stats.get('win_rate', 0):.1%}`"
        )
        self._send(msg)

    # ── Invio ────────────────────────────────────────────────────────────────

    def _send(self, text: str):
        if not self.enabled:
            log.debug(f"[ALERT simulato] {text[:80]}...")
            return

        # Rate limit
        now = time.time()
        wait = self._min_interval - (now - self._last_sent)
        if wait > 0:
            time.sleep(wait)

        try:
            import requests
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id":    self.chat_id,
                "text":       text,
                "parse_mode": "Markdown",
            }, timeout=10)
            if resp.status_code == 200:
                log.debug("Alert Telegram inviato.")
            else:
                log.warning(f"Telegram errore {resp.status_code}: {resp.text[:100]}")
        except ImportError:
            log.warning("'requests' non installato. pip install requests")
        except Exception as e:
            log.warning(f"Telegram send fallito: {e}")

        self._last_sent = time.time()

    @staticmethod
    def _duration(trade: Trade) -> str:
        if trade.open_time and trade.close_time:
            delta = trade.close_time - trade.open_time
            h, m = divmod(int(delta.total_seconds()) // 60, 60)
            return f"{h}h {m}m" if h else f"{m}m"
        return "N/A"


# ── Singleton per uso globale ─────────────────────────────────────────────────

_alerter: Optional[TelegramAlerter] = None


def get_alerter() -> TelegramAlerter:
    global _alerter
    if _alerter is None:
        _alerter = TelegramAlerter()
    return _alerter
