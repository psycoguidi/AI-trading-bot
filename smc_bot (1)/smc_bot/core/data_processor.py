"""
core/data_processor.py
=======================
Arricchisce il DataFrame OHLCV con ATR, momentum e swing points.
"""

import logging

import numpy as np
import pandas as pd

from config.settings import Settings

log = logging.getLogger("processor")


class DataProcessor:

    def __init__(self, settings: Settings):
        self.s = settings

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = _add_atr(df)
        df = _add_momentum(df)
        df = self._add_swings(df)
        return df

    def _add_swings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Swing High: high[i] == max su finestra [i-L … i+R]
        Swing Low:  low[i]  == min su finestra [i-L … i+R]
        """
        L, R = self.s.SWING_LEFT, self.s.SWING_RIGHT
        n = len(df)
        sh = np.zeros(n, dtype=bool)
        sl = np.zeros(n, dtype=bool)

        highs = df["high"].values
        lows  = df["low"].values

        for i in range(L, n - R):
            win_h = highs[i - L: i + R + 1]
            win_l = lows [i - L: i + R + 1]
            if highs[i] == win_h.max():
                sh[i] = True
            if lows[i] == win_l.min():
                sl[i] = True

        df["swing_high"] = sh
        df["swing_low"]  = sl
        return df


# ── indicatori puri (funzioni libere, facilmente testabili) ─────────────────

def _add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c  = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(span=period, adjust=False).mean()
    return df


def _add_momentum(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    df["momentum"] = df["close"].pct_change(periods=period).fillna(0)
    return df
