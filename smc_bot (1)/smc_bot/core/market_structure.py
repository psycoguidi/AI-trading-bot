"""
core/market_structure.py
=========================
Analisi struttura mercato SMC:
  - Swing H/L
  - Trend (HH/HL vs LH/LL)
  - CHoCH (Change of Character)
  - Fair Value Gap
  - Liquidity Sweep
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple

import pandas as pd

from config.settings import Settings
from core.models import (
    CHoCH, FVG, LiqSweep, MarketStructure,
    SwingPoint, Direction,
)
from utils.pip_utils import pip_size, to_pips

log = logging.getLogger("structure")


class MarketStructureEngine:

    def __init__(self, settings: Settings):
        self.s = settings

    # ── Entry point ─────────────────────────────────────────────────────────

    def analyze(self, df: pd.DataFrame, symbol: str = "",
                timeframe: str = "") -> MarketStructure:
        ms = MarketStructure(
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            atr=float(df["atr"].iloc[-1]) if "atr" in df.columns else 0.0,
            momentum=float(df["momentum"].iloc[-1]) if "momentum" in df.columns else 0.0,
        )

        ms.swing_highs = self._collect_swings(df, "high")
        ms.swing_lows  = self._collect_swings(df, "low")
        ms.trend       = self._detect_trend(ms.swing_highs, ms.swing_lows)
        ms.choch       = self._detect_choch(df, ms.swing_highs, ms.swing_lows, ms.trend)
        ms.fvgs        = self._detect_fvgs(df, symbol)
        ms.sweep       = self._detect_sweep(df, ms.swing_highs, ms.swing_lows, symbol)

        return ms

    # ── Swing collection ────────────────────────────────────────────────────

    def _collect_swings(self, df: pd.DataFrame, kind: str) -> List[SwingPoint]:
        col   = "swing_high" if kind == "high" else "swing_low"
        price = "high"       if kind == "high" else "low"
        if col not in df.columns:
            return []

        mask = df[col]
        pts  = []
        for i, (ts, row) in enumerate(df[mask].iterrows()):
            pts.append(SwingPoint(
                bar_index=int(df.index.get_loc(ts)),
                price=float(row[price]),
                timestamp=ts if isinstance(ts, datetime) else datetime.now(),
                kind=kind,
            ))
        return pts[-30:]   # ultimi 30

    # ── Trend ────────────────────────────────────────────────────────────────

    def _detect_trend(self, highs: List[SwingPoint],
                      lows: List[SwingPoint]) -> str:
        n = self.s.MIN_SWINGS
        if len(highs) < n or len(lows) < n:
            return "ranging"

        last_h = [p.price for p in highs[-n:]]
        last_l = [p.price for p in lows[-n:]]

        hh = all(last_h[i] < last_h[i+1] for i in range(n-1))
        hl = all(last_l[i] < last_l[i+1] for i in range(n-1))
        lh = all(last_h[i] > last_h[i+1] for i in range(n-1))
        ll = all(last_l[i] > last_l[i+1] for i in range(n-1))

        if hh and hl:  return "bullish"
        if lh and ll:  return "bearish"
        return "ranging"

    # ── CHoCH ────────────────────────────────────────────────────────────────

    def _detect_choch(self, df: pd.DataFrame,
                      highs: List[SwingPoint], lows: List[SwingPoint],
                      trend: str) -> Optional[CHoCH]:
        """
        CHoCH rialzista: struttura passata bear (almeno 2 LH+LL)
                         + prezzo attuale > qualsiasi swing high recente

        CHoCH ribassista: struttura passata bull (almeno 2 HH+HL)
                          + prezzo attuale < qualsiasi swing low recente

        CORREZIONE CHIAVE: il trend passato viene valutato sulle prime N-1
        swing, mentre la rottura è verificata sull'ultima candela disponibile.
        Questo evita il paradosso per cui classificare il trend "bullish"
        implica già che nessun low è stato rotto.
        """
        if len(highs) < 2 or len(lows) < 2:
            return None

        current = float(df["close"].iloc[-1])
        ts      = datetime.now()

        # ── Trend struttura PASSATA (escludi ultimo swing) ──────────────────
        past_highs = highs[:-1]
        past_lows  = lows[:-1]
        past_trend = self._detect_trend(past_highs, past_lows)

        # ── CHoCH LONG: struttura passata bear, close sopra ultimo SH ───────
        if past_trend == "bearish" and highs:
            ref = highs[-1]          # Ultimo Lower High da rompere
            if current > ref.price:
                log.debug(f"CHoCH LONG  @{current:.5f} > LH {ref.price:.5f}")
                return CHoCH(Direction.LONG, current, ref, ts, confirmed=True)

        # ── CHoCH SHORT: struttura passata bull, close sotto ultimo SL ───────
        if past_trend == "bullish" and lows:
            ref = lows[-1]           # Ultimo Higher Low da rompere
            if current < ref.price:
                log.debug(f"CHoCH SHORT @{current:.5f} < HL {ref.price:.5f}")
                return CHoCH(Direction.SHORT, current, ref, ts, confirmed=True)

        # ── Fallback: cerca rottura strutturale anche in ranging ─────────────
        # Se il prezzo rompe il minimo assoluto degli ultimi swing high
        # o il massimo degli swing low → CHoCH implicito
        if trend == "ranging" and len(highs) >= 3 and len(lows) >= 3:
            # Verifica se gli ultimi 3 highs erano LH (bear implicito)
            h3 = [p.price for p in highs[-3:]]
            l3 = [p.price for p in lows[-3:]]
            if all(h3[i] > h3[i+1] for i in range(2)):    # LH LH LH
                ref = highs[-1]
                if current > ref.price:
                    log.debug(f"CHoCH LONG (ranging) @{current:.5f}")
                    return CHoCH(Direction.LONG, current, ref, ts, confirmed=True)
            if all(l3[i] < l3[i+1] for i in range(2)):    # HL HL HL
                ref = lows[-1]
                if current < ref.price:
                    log.debug(f"CHoCH SHORT (ranging) @{current:.5f}")
                    return CHoCH(Direction.SHORT, current, ref, ts, confirmed=True)

        return None

    # ── FVG ──────────────────────────────────────────────────────────────────

    def _detect_fvgs(self, df: pd.DataFrame, symbol: str) -> List[FVG]:
        """
        FVG rialzista:  high[i-1] < low[i+1]   zona = [high[i-1], low[i+1]]
        FVG ribassista: low[i-1]  > high[i+1]  zona = [high[i+1], low[i-1]]
        """
        fvgs    = []
        n       = len(df)
        px      = float(df["close"].iloc[-1])
        ps      = pip_size(symbol, px)
        min_sz  = self.s.FVG_MIN_PIPS * ps
        max_age = self.s.FVG_MAX_AGE

        for offset in range(1, min(max_age + 1, n - 1)):
            i = n - 1 - offset
            if i < 1 or i >= n - 1:
                continue
            c1, c3 = df.iloc[i - 1], df.iloc[i + 1]
            ts = df.index[i]
            ts = ts if isinstance(ts, datetime) else datetime.now()

            # Rialzista
            if c1["high"] < c3["low"]:
                sz = c3["low"] - c1["high"]
                if sz >= min_sz:
                    fvgs.append(FVG(
                        direction=Direction.LONG,
                        fvg_high=float(c3["low"]),
                        fvg_low=float(c1["high"]),
                        bar_index=i,
                        timestamp=ts,
                        size_pips=sz / ps,
                    ))
            # Ribassista
            elif c1["low"] > c3["high"]:
                sz = c1["low"] - c3["high"]
                if sz >= min_sz:
                    fvgs.append(FVG(
                        direction=Direction.SHORT,
                        fvg_high=float(c1["low"]),
                        fvg_low=float(c3["high"]),
                        bar_index=i,
                        timestamp=ts,
                        size_pips=sz / ps,
                    ))

        # Ordina: più recente prima
        fvgs.sort(key=lambda f: f.bar_index, reverse=True)

        # Assegna priorità
        for idx, fvg in enumerate(fvgs):
            fvg.priority = idx + 1

        return fvgs

    # ── Liquidity Sweep ──────────────────────────────────────────────────────

    def _detect_sweep(self, df: pd.DataFrame,
                      highs: List[SwingPoint], lows: List[SwingPoint],
                      symbol: str) -> Optional[LiqSweep]:
        """
        Sweep sopra un high: candela recente sfora ma chiude sotto.
        Sweep sotto un low:  candela recente sfora ma chiude sopra.
        Segnala che la liquidità è stata assorbita → probabile inversione.
        """
        if len(df) < 3:
            return None

        px   = float(df["close"].iloc[-1])
        ps   = pip_size(symbol, px)
        tol  = self.s.SWEEP_TOLERANCE_PIPS * ps
        rec  = df.iloc[-3:]
        hi   = float(rec["high"].max())
        lo   = float(rec["low"].min())
        ts   = datetime.now()

        if highs:
            ref = highs[-1].price
            if hi > ref + tol and px < ref:
                log.debug(f"LiqSweep ABOVE {ref:.5f}")
                return LiqSweep("above", ref, hi, lo, ts)

        if lows:
            ref = lows[-1].price
            if lo < ref - tol and px > ref:
                log.debug(f"LiqSweep BELOW {ref:.5f}")
                return LiqSweep("below", ref, hi, lo, ts)

        return None
