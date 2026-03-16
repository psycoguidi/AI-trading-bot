"""
utils/synthetic_data.py
=======================
Genera dati OHLCV sintetici con struttura SMC realistica:
- Trend con Higher Highs / Higher Lows (o Lower)
- Liquidity sweep prima delle inversioni
- Fair Value Gap autentici
- CHoCH chiari

Usato per backtest e sviluppo senza broker connesso.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple


def generate_smc_data(
    n_bars: int = 500,
    base_price: float = 1.0850,
    pip: float = 0.0001,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Produce un DataFrame OHLCV con struttura SMC chiara.

    Il mercato alterna fasi:
    - Bullish: HH + HL
    - Sweep sopra il massimo
    - CHoCH ribassista
    - Bearish: LH + LL
    - Sweep sotto il minimo
    - CHoCH rialzista
    """
    rng = np.random.default_rng(seed)

    opens, highs, lows, closes, vols = [], [], [], [], []
    price = base_price
    phase = "bullish"
    phase_bars = 0
    phase_len = rng.integers(30, 60)
    trend_step = pip * rng.uniform(2, 5)

    for i in range(n_bars):
        phase_bars += 1

        # Fase: avanza nella direzione del trend
        if phase == "bullish":
            mu = trend_step
        elif phase == "bearish":
            mu = -trend_step
        elif phase == "sweep_high":
            # Spike violento sopra il massimo (fakeout)
            mu = trend_step * 4 if phase_bars <= 2 else -trend_step * 3
        elif phase == "sweep_low":
            mu = -trend_step * 4 if phase_bars <= 2 else trend_step * 3
        else:
            mu = 0

        noise = rng.normal(0, pip * 3)
        close = price + mu + noise
        close = max(close, pip)

        spread = pip * rng.uniform(8, 20)
        wick_up   = abs(rng.normal(0, pip * 4))
        wick_down = abs(rng.normal(0, pip * 4))

        # FVG sintetico: ogni ~15 barre in trend forte lascia un gap
        if phase in ("bullish", "bearish") and i % 15 == 7:
            if phase == "bullish":
                wick_down = 0        # Candela esplosiva → possibile gap
                spread *= 2
            else:
                wick_up = 0
                spread *= 2

        o = price
        c = close
        h = max(o, c) + wick_up
        l = min(o, c) - wick_down
        h = max(h, o, c)
        l = min(l, o, c)

        opens.append(o)
        highs.append(h)
        lows.append(l)
        closes.append(c)
        vols.append(int(rng.uniform(500, 3000)))

        price = c

        # Cambio di fase
        if phase_bars >= phase_len:
            phase_bars = 0
            phase_len  = rng.integers(25, 55)
            if phase == "bullish":
                phase = "sweep_high"
                phase_len = rng.integers(2, 5)
            elif phase == "sweep_high":
                phase = "bearish"
            elif phase == "bearish":
                phase = "sweep_low"
                phase_len = rng.integers(2, 5)
            elif phase == "sweep_low":
                phase = "bullish"
            trend_step = pip * rng.uniform(2, 5)

    freq_map = {0.0001: "1min", 0.1: "5min", 1.0: "1h"}
    freq = "1min"   # Default

    idx = pd.date_range(end=datetime.now(), periods=n_bars, freq=freq)

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": vols,
    }, index=idx)
    df.index.name = "time"
    return df


def make_multi_tf(
    symbol: str = "EURUSD",
    n_ltf: int = 600,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Genera tre DataFrame coerenti per HTF / MTF / LTF.
    I TF superiori sono ricampionati dall'LTF.
    """
    pip = 0.0001
    base = {"EURUSD": 1.0850, "GBPUSD": 1.2700,
            "XAUUSD": 2350.0, "US30": 39000.0, "NAS100": 18000.0}.get(symbol, 1.0)

    if base > 1000:
        pip = 1.0
    elif base > 100:
        pip = 0.1

    # LTF = 1M
    ltf = generate_smc_data(n_ltf, base, pip, seed)

    # MTF = 5M → ricampiona 5 candele LTF in 1
    mtf = _resample(ltf, 5)

    # HTF = 1H → ricampiona 60 candele LTF in 1
    htf = _resample(ltf, 60)

    return htf, mtf, ltf


def _resample(df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """Ricampiona un DataFrame OHLCV accorpando `factor` barre."""
    result = []
    for i in range(0, len(df) - factor, factor):
        chunk = df.iloc[i: i + factor]
        result.append({
            "open":   chunk["open"].iloc[0],
            "high":   chunk["high"].max(),
            "low":    chunk["low"].min(),
            "close":  chunk["close"].iloc[-1],
            "volume": chunk["volume"].sum(),
        })
    out = pd.DataFrame(result, index=df.index[factor::factor][:len(result)])
    out.index.name = "time"
    return out
