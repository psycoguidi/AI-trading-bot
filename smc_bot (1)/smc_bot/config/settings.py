"""
config/settings.py  –  Configurazione centralizzata del bot SMC.
Ogni parametro ha un commento che spiega il suo effetto.
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Settings:

    # ── BROKER ─────────────────────────────────────────────────────────────
    BROKER: str = "simulation"           # "mt5" | "simulation"
    MT5_LOGIN:    int = int(os.getenv("MT5_LOGIN",    "0"))
    MT5_PASSWORD: str = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER:   str = os.getenv("MT5_SERVER",   "")

    # ── SIMBOLI ────────────────────────────────────────────────────────────
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "XAUUSD", "US30", "NAS100"
    ])

    # ── TIMEFRAME ──────────────────────────────────────────────────────────
    TF_HTF: str = "1H"     # Bias direzionale
    TF_MTF: str = "5M"     # Struttura + CHOCH + FVG
    TF_LTF: str = "1M"     # Timing di entrata

    # Quante candele caricare per ogni TF
    CANDLES_HTF: int = 200
    CANDLES_MTF: int = 300
    CANDLES_LTF: int = 150

    # ── AI FILTER ──────────────────────────────────────────────────────────
    AI_THRESHOLD: float = 0.55           # Probabilità minima per aprire
    AI_MODEL_PATH: str = "models/smc_model.pkl"

    # ── RISK MANAGEMENT ────────────────────────────────────────────────────
    RISK_PER_TRADE:  float = 0.01        # 1% del capitale per trade
    MAX_OPEN_TRADES: int   = 3
    MAX_DAILY_LOSS:  float = 0.03        # Stop se -3% giornaliero
    MIN_RR:          float = 1.5         # R:R minimo accettato
    SL_BUFFER_PIPS:  float = 3.0         # Pips extra sotto/sopra FVG per SL
    TP_BUFFER_PIPS:  float = 5.0         # Pips prima della struttura opposta

    # ── FVG ────────────────────────────────────────────────────────────────
    FVG_MIN_PIPS:   float = 1.5          # Dimensione minima FVG in pips
    FVG_MAX_AGE:    int   = 30           # Età massima FVG in candele
    USE_SECOND_FVG: bool  = True         # Considera anche la 2ª FVG

    # ── SWING / CHOCH ──────────────────────────────────────────────────────
    SWING_LEFT:  int = 5                 # Candele a sinistra per swing
    SWING_RIGHT: int = 5                 # Candele a destra  per swing
    MIN_SWINGS:  int = 3                 # Swing minimi per analisi trend

    # ── LIQUIDITY SWEEP ────────────────────────────────────────────────────
    SWEEP_TOLERANCE_PIPS: float = 2.0    # Tolleranza per equal H/L
    SWEEP_LOOKBACK:       int   = 60     # Candele di lookback

    # ── LOOP ───────────────────────────────────────────────────────────────
    LOOP_INTERVAL: float = 30.0          # Secondi tra ogni ciclo

    # ── ACCOUNT (paper trading default) ────────────────────────────────────
    INITIAL_BALANCE: float = 10_000.0

    # ── TRAILING SL ────────────────────────────────────────────────────────
    TRAILING_SL_ENABLED:  bool  = False      # Abilita trailing SL
    TRAILING_SL_PIPS:     float = 10.0       # Distanza trailing in pips

    # ── BREAKEVEN ──────────────────────────────────────────────────────────
    BREAKEVEN_ENABLED:    bool  = True       # Sposta SL a entry quando +1R
    BREAKEVEN_TRIGGER_R:  float = 1.0        # Attiva dopo 1R di profitto
