"""
core/models.py  –  Dataclass condivise da tutti i moduli.
Nessuna dipendenza da altri moduli interni → zero import circolari.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


# ── Enumerazioni ────────────────────────────────────────────────────────────

class Direction(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    PENDING  = "PENDING"
    OPEN     = "OPEN"
    CLOSED   = "CLOSED"
    REJECTED = "REJECTED"


# ── Elementi strutturali ────────────────────────────────────────────────────

@dataclass
class SwingPoint:
    bar_index: int          # Posizione nel DataFrame
    price:     float        # high se kind="high", low se kind="low"
    timestamp: datetime
    kind:      str          # "high" | "low"


@dataclass
class FVG:
    """Fair Value Gap – zona di inefficienza tra tre candele."""
    direction:   Direction
    fvg_high:    float       # Bordo superiore della zona
    fvg_low:     float       # Bordo inferiore della zona
    bar_index:   int         # Indice della candela centrale
    timestamp:   datetime
    size_pips:   float = 0.0
    priority:    int   = 1   # 1 = prima FVG (più alta priorità)
    filled:      bool  = False


@dataclass
class CHoCH:
    """Change of Character – cambio di struttura."""
    direction:   Direction   # Direzione del NUOVO trend
    break_price: float       # Prezzo che ha rotto la struttura
    swing_ref:   SwingPoint  # Swing violato
    timestamp:   datetime
    confirmed:   bool = False


@dataclass
class LiqSweep:
    """Liquidity Sweep su swing H/L o equal levels."""
    side:          str    # "above" | "below"
    swept_price:   float  # Livello di liquidità sweep-ato
    candle_high:   float
    candle_low:    float
    timestamp:     datetime


# ── Output analisi struttura ────────────────────────────────────────────────

@dataclass
class MarketStructure:
    symbol:     str
    timeframe:  str
    timestamp:  datetime
    trend:      str = "ranging"        # "bullish" | "bearish" | "ranging"
    swing_highs: List[SwingPoint] = field(default_factory=list)
    swing_lows:  List[SwingPoint] = field(default_factory=list)
    choch:       Optional[CHoCH]  = None
    sweep:       Optional[LiqSweep] = None
    fvgs:        List[FVG]        = field(default_factory=list)
    atr:         float = 0.0
    momentum:    float = 0.0


# ── Setup operativo ─────────────────────────────────────────────────────────

@dataclass
class TradeSetup:
    symbol:       str
    direction:    Direction
    entry:        float
    sl:           float
    tp:           float
    fvg:          FVG
    choch:        CHoCH
    sweep:        Optional[LiqSweep] = None

    # Feature per AI filter
    fvg_pips:          float = 0.0
    dist_choch_pips:   float = 0.0
    atr_pips:          float = 0.0
    momentum:          float = 0.0
    dist_liq_pips:     float = 0.0
    htf_aligned:       bool  = False
    structure_score:   float = 0.0
    rr:                float = 0.0

    # Riempiti dopo
    ai_prob:       float    = 0.0
    lot_size:      float    = 0.0
    timestamp:     datetime = field(default_factory=datetime.now)

    @property
    def risk_pips(self) -> float:
        return abs(self.entry - self.sl)

    @property
    def reward_pips(self) -> float:
        return abs(self.tp - self.entry)


# ── Trade ───────────────────────────────────────────────────────────────────

@dataclass
class Trade:
    trade_id:    str
    symbol:      str
    direction:   Direction
    entry:       float
    sl:          float
    tp:          float
    lot_size:    float
    setup:       TradeSetup
    status:      TradeStatus = TradeStatus.PENDING
    open_time:   Optional[datetime] = None
    close_time:  Optional[datetime] = None
    close_price: Optional[float]    = None
    pnl_usd:     float = 0.0
    pnl_pct:     float = 0.0
    outcome:     str   = ""   # "win" | "loss" | "timeout"

    def __str__(self) -> str:
        return (
            f"[{self.trade_id}] {self.direction.value} {self.symbol} "
            f"entry={self.entry:.5f} sl={self.sl:.5f} tp={self.tp:.5f} "
            f"lot={self.lot_size:.2f} AI={self.setup.ai_prob:.0%}"
        )
