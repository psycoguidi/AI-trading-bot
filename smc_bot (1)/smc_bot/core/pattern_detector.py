"""
core/pattern_detector.py
=========================
Verifica la sequenza SMC completa e costruisce il TradeSetup.

SEQUENZA RICHIESTA:
  1. Liquidity Sweep  (HTF o MTF)
  2. CHoCH            su MTF
  3. FVG valida       su MTF (formata DOPO il CHoCH)
  4. Prezzo nella FVG su LTF
  5. Candela di conferma su LTF
  6. Allineamento bias HTF
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import Settings
from core.models import (
    CHoCH, FVG, Direction, MarketStructure, TradeSetup,
)
from utils.pip_utils import pip_size, to_pips

log = logging.getLogger("pattern")


class PatternDetector:

    def __init__(self, settings: Settings):
        self.s = settings

    def detect(
        self,
        symbol: str,
        htf: MarketStructure,
        mtf: MarketStructure,
        ltf: MarketStructure,
        df_ltf: pd.DataFrame,
    ) -> Optional[TradeSetup]:

        # ── 1. CHoCH obbligatorio su MTF ──────────────────────────────────
        if mtf.choch is None or not mtf.choch.confirmed:
            return None
        choch = mtf.choch
        direction = choch.direction

        # ── 2. Allineamento HTF ───────────────────────────────────────────
        htf_ok = _htf_aligned(direction, htf.trend)
        if not htf_ok:
            log.debug(f"{symbol}: HTF {htf.trend} non allineato con {direction.value}")
            return None

        # ── 3. FVG valida dopo il CHoCH ───────────────────────────────────
        fvg = self._best_fvg(direction, mtf)
        if fvg is None:
            log.debug(f"{symbol}: nessuna FVG valida su MTF")
            return None

        # ── 4. Prezzo nel ritracciamento (nella FVG) su LTF ──────────────
        # Controlla se una delle ultime N candele LTF ha toccato la zona FVG
        # (low <= fvg_high e high >= fvg_low) — ritracciamento reale.
        price = float(df_ltf["close"].iloc[-1])
        lookback_ltf = min(10, len(df_ltf))
        recent_ltf   = df_ltf.iloc[-lookback_ltf:]
        touched_fvg  = any(
            float(row["low"])  <= fvg.fvg_high and
            float(row["high"]) >= fvg.fvg_low
            for _, row in recent_ltf.iterrows()
        )
        if not touched_fvg:
            log.debug(
                f"{symbol}: nessuna candela LTF recente ha toccato FVG "
                f"[{fvg.fvg_low:.5f}–{fvg.fvg_high:.5f}]"
            )
            return None

        # ── 5. Candela di conferma su LTF ─────────────────────────────────
        # Almeno una delle ultime 3 candele LTF deve confermare la direzione
        if not _ltf_confirmed(direction, df_ltf, lookback=3):
            log.debug(f"{symbol}: nessuna conferma su LTF (ultime 3 candele)")
            return None

        # ── 6. SL / TP ────────────────────────────────────────────────────
        ps = pip_size(symbol, price)
        sl, tp = self._calc_sl_tp(direction, fvg, mtf, ps)
        if sl is None or tp is None:
            log.debug(f"{symbol}: SL/TP non calcolabili")
            return None

        # ── Verifica R:R minimo ────────────────────────────────────────────
        risk   = abs(price - sl)
        reward = abs(tp - price)
        rr     = reward / risk if risk > 0 else 0.0
        if rr < self.s.MIN_RR:
            log.debug(f"{symbol}: R:R {rr:.2f} < min {self.s.MIN_RR}")
            return None

        # ── Costruzione setup ─────────────────────────────────────────────
        sweep = mtf.sweep or htf.sweep

        setup = TradeSetup(
            symbol=symbol,
            direction=direction,
            entry=price,
            sl=sl,
            tp=tp,
            fvg=fvg,
            choch=choch,
            sweep=sweep,
            fvg_pips=fvg.size_pips,
            dist_choch_pips=to_pips(abs(price - choch.break_price), symbol, price),
            atr_pips=to_pips(mtf.atr, symbol, price),
            momentum=abs(mtf.momentum),
            dist_liq_pips=to_pips(abs(price - sweep.swept_price), symbol, price) if sweep else 0.0,
            htf_aligned=htf_ok,
            structure_score=_structure_score(mtf),
            rr=rr,
        )

        log.info(
            f"✅ Setup {direction.value} {symbol} | "
            f"entry={price:.5f} sl={sl:.5f} tp={tp:.5f} | "
            f"R:R={rr:.2f} | FVG={fvg.size_pips:.1f}p"
        )
        return setup

    # ── helpers ──────────────────────────────────────────────────────────────

    def _best_fvg(self, direction: Direction,
                  mtf: MarketStructure) -> Optional[FVG]:
        """Prima FVG nella direzione del CHoCH, formata dopo il CHoCH."""
        choch_bar = mtf.choch.swing_ref.bar_index if mtf.choch else -1
        candidates = [
            f for f in mtf.fvgs
            if f.direction == direction
            and f.bar_index > choch_bar
            and not f.filled
        ]
        if not candidates:
            return None

        # Priorità 1 sempre; se strong momentum accetta anche priorità 2
        if self.s.USE_SECOND_FVG and abs(mtf.momentum) > 0.004 \
                and len(candidates) >= 2:
            return candidates[0]   # ancora la prima, ma potremmo fare logica avanzata
        return candidates[0]

    def _calc_sl_tp(self, direction: Direction, fvg: FVG,
                    mtf: MarketStructure, ps: float):
        """
        SL: oltre il bordo della FVG + buffer
        TP: prima della struttura opposta (penultimo swing, non quello appena rotto dal CHoCH)
        """
        buf_sl = self.s.SL_BUFFER_PIPS * ps
        buf_tp = self.s.TP_BUFFER_PIPS * ps

        if direction == Direction.LONG:
            sl = fvg.fvg_low - buf_sl
            # TP = sotto l'ultimo swing high (struttura opposta)
            # Prende il swing high più recente che sia SOPRA la FVG
            sh_candidates = [h for h in mtf.swing_highs if h.price > fvg.fvg_high]
            if not sh_candidates:
                return None, None
            tp = sh_candidates[-1].price - buf_tp
            if tp <= fvg.fvg_high:
                return None, None

        else:  # SHORT
            sl = fvg.fvg_high + buf_sl
            # TP = sopra l'ultimo swing low (struttura opposta)
            # Prende il swing low più recente che sia SOTTO la FVG
            sl_candidates = [l for l in mtf.swing_lows if l.price < fvg.fvg_low]
            if not sl_candidates:
                return None, None
            tp = sl_candidates[-1].price + buf_tp
            if tp >= fvg.fvg_low:
                return None, None

        return sl, tp


# ── funzioni pure ─────────────────────────────────────────────────────────────

def _htf_aligned(direction: Direction, htf_trend: str) -> bool:
    if htf_trend == "ranging":
        return True
    return (direction == Direction.LONG  and htf_trend == "bullish") or \
           (direction == Direction.SHORT and htf_trend == "bearish")


def _ltf_confirmed(direction: Direction, df: pd.DataFrame, lookback: int = 3) -> bool:
    """
    Verifica che almeno una delle ultime `lookback` candele LTF
    sia nella direzione del trade (chiusura > apertura per LONG, viceversa SHORT).
    """
    if len(df) < 1:
        return False
    recent = df.iloc[-lookback:]
    for _, row in recent.iterrows():
        c, o = float(row["close"]), float(row["open"])
        if direction == Direction.LONG  and c > o:
            return True
        if direction == Direction.SHORT and c < o:
            return True
    return False


def _structure_score(mtf: MarketStructure) -> float:
    score = 0.0
    swings = len(mtf.swing_highs) + len(mtf.swing_lows)
    score += min(swings / 20, 1.0) * 0.4
    if mtf.trend != "ranging":
        score += 0.3
    if mtf.sweep is not None:
        score += 0.3
    return min(score, 1.0)
