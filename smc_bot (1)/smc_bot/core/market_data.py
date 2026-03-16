"""
core/market_data.py
===================
Fornisce dati OHLCV multi-timeframe.
Supporta MetaTrader 5 (live) e simulazione (sviluppo/backtest).
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings

log = logging.getLogger("market_data")

_BASE_PRICE = {
    "EURUSD": 1.0850, "GBPUSD": 1.2700,
    "XAUUSD": 2350.0, "US30": 39_000.0, "NAS100": 18_000.0,
}

_TF_MINUTES = {"1M": 1, "5M": 5, "15M": 15, "1H": 60, "4H": 240, "1D": 1440}

_MT5_TF = {
    "1M": 1, "5M": 5, "15M": 15,
    "1H": 16385, "4H": 16388, "1D": 16408,
}


class MarketDataFeed:

    def __init__(self, settings: Settings):
        self.settings = settings
        self._mt5 = None
        self._use_mt5 = (settings.BROKER == "mt5")

    # ── public ──────────────────────────────────────────────────────────────

    def connect(self):
        if self._use_mt5:
            self._init_mt5()

    def get_ohlcv(self, symbol: str, timeframe: str, n_bars: int) -> pd.DataFrame:
        if self._use_mt5 and self._mt5 is not None:
            df = self._fetch_mt5(symbol, timeframe, n_bars)
            if df is not None and len(df) >= 20:
                return df
            log.warning(f"MT5 fetch fallito per {symbol} {timeframe}. Uso simulazione.")
        return self._simulate(symbol, timeframe, n_bars)

    # ── MT5 ─────────────────────────────────────────────────────────────────

    def _init_mt5(self):
        try:
            import MetaTrader5 as mt5
            ok = mt5.initialize(
                login=self.settings.MT5_LOGIN,
                password=self.settings.MT5_PASSWORD,
                server=self.settings.MT5_SERVER,
            )
            if ok:
                self._mt5 = mt5
                log.info("MT5 connesso.")
            else:
                log.error(f"MT5 init fallito: {mt5.last_error()}")
        except ImportError:
            log.warning("MetaTrader5 non installato.")

    def _fetch_mt5(self, symbol: str, timeframe: str, n: int) -> Optional[pd.DataFrame]:
        mt5 = self._mt5
        tf  = _MT5_TF.get(timeframe)
        if tf is None:
            return None
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        df = df[["time", "open", "high", "low", "close", "volume"]].set_index("time")
        return df

    # ── Simulazione realistica ───────────────────────────────────────────────

    def _simulate(self, symbol: str, timeframe: str, n: int) -> pd.DataFrame:
        """
        Genera OHLCV simulati con:
        - Random walk con drift
        - Volatilità variabile (regime switch)
        - Struttura multi-swing realistica
        """
        rng  = np.random.default_rng(hash(symbol + timeframe) % (2**31))
        base = _BASE_PRICE.get(symbol, 1.0)
        vol  = base * 0.0004   # volatilità base per barra

        closes = [base]
        for i in range(1, n):
            # Alterna regime: trend / sideways
            regime_vol = vol * (1.5 if (i // 30) % 2 == 0 else 0.6)
            drift = rng.normal(0, regime_vol)
            closes.append(max(closes[-1] + drift, base * 0.5))

        closes = np.array(closes)
        noise  = np.abs(rng.normal(0, vol * 0.7, n))

        opens  = closes - rng.normal(0, vol * 0.3, n)
        highs  = np.maximum(closes, opens) + noise
        lows   = np.minimum(closes, opens) - noise
        vols   = rng.integers(200, 3000, n).astype(float)

        minutes = _TF_MINUTES.get(timeframe, 1)
        freq    = f"{minutes}min"
        idx     = pd.date_range(end=datetime.now(), periods=n, freq=freq)

        df = pd.DataFrame(
            {"open": opens, "high": highs, "low": lows,
             "close": closes, "volume": vols},
            index=idx,
        )
        df.index.name = "time"
        return df
