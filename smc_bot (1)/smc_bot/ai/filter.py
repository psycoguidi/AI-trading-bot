"""
ai/filter.py
============
Valuta la probabilità di successo di un TradeSetup SMC.

MODALITÀ:
  1. Modello ML sklearn (GradientBoosting) se esiste models/smc_model.pkl
  2. Modello euristico pesato (sempre disponibile come fallback)

FEATURE (8 dimensioni, tutte in [0, 1]):
  0  fvg_norm          dimensione FVG normalizzata
  1  dist_choch_norm   distanza dal CHoCH (vicino = buono)
  2  atr_norm          volatilità (troppa = rischio)
  3  momentum_norm     forza momentum
  4  dist_liq_norm     distanza dalla liquidità
  5  htf_aligned       allineamento HTF (0/1)
  6  structure_score   qualità struttura [0,1]
  7  rr_norm           Risk:Reward normalizzato
"""

import logging
import os
from typing import Optional

import numpy as np

from config.settings import Settings
from core.models import TradeSetup

log = logging.getLogger("ai_filter")

# Pesi euristici derivati dalla logica SMC
_WEIGHTS = np.array([
    0.10,   # fvg_norm
    0.08,   # dist_choch_norm  (inverso: vicino = meglio)
    0.07,   # atr_norm         (inverso: basso = meglio)
    0.12,   # momentum_norm
    0.08,   # dist_liq_norm
    0.25,   # htf_aligned      (fattore principale)
    0.15,   # structure_score
    0.15,   # rr_norm
], dtype=np.float32)


class AIFilter:

    def __init__(self, settings: Settings):
        self.s = settings
        self._model = _load_model(settings.AI_MODEL_PATH)

    def evaluate(self, setup: TradeSetup) -> float:
        feat = _extract(setup)
        if self._model is not None:
            prob = _predict_ml(self._model, feat)
        else:
            prob = _predict_heuristic(feat, setup)
        log.debug(f"AI prob {setup.symbol}: {prob:.2%}")
        return float(np.clip(prob, 0.0, 1.0))


# ── Feature extraction ───────────────────────────────────────────────────────

def _extract(s: TradeSetup) -> np.ndarray:
    return np.array([
        min(s.fvg_pips       / 15.0, 1.0),
        1.0 - min(s.dist_choch_pips / 80.0, 1.0),   # vicino = alto score
        1.0 - min(s.atr_pips        / 40.0, 1.0),   # bassa vol = alto score
        min(s.momentum              / 0.015, 1.0),
        min(s.dist_liq_pips         / 150.0, 1.0),
        float(s.htf_aligned),
        s.structure_score,
        min(s.rr                    / 4.0, 1.0),
    ], dtype=np.float32)


# ── Predizione euristica ─────────────────────────────────────────────────────

def _predict_heuristic(feat: np.ndarray, setup: TradeSetup) -> float:
    score = float(np.dot(feat, _WEIGHTS))

    # Bonus se sweep confermato
    if setup.sweep is not None:
        score += 0.07

    # Bonus se FVG priorità 1
    if setup.fvg.priority == 1:
        score += 0.03

    # Penalità se R:R < 1.8
    if setup.rr < 1.8:
        score -= 0.06

    return float(np.clip(score, 0.0, 1.0))


# ── ML ───────────────────────────────────────────────────────────────────────

def _load_model(path: str):
    if not os.path.exists(path):
        log.info("Nessun modello AI trovato → uso euristica.")
        return None
    try:
        import joblib
        model = joblib.load(path)
        log.info(f"Modello AI caricato: {path}")
        return model
    except Exception as e:
        log.warning(f"Errore caricamento modello: {e} → uso euristica.")
        return None


def _predict_ml(model, feat: np.ndarray) -> float:
    try:
        return float(model.predict_proba(feat.reshape(1, -1))[0][1])
    except Exception as e:
        log.error(f"ML predict error: {e}")
        return 0.0


# ── Training ─────────────────────────────────────────────────────────────────

class ModelTrainer:
    """
    Addestra il modello su un CSV storico di trade.

    CSV richiesto (colonne):
      fvg_pips, dist_choch_pips, atr_pips, momentum,
      dist_liq_pips, htf_aligned, structure_score, rr, outcome

    outcome: 1 = win, 0 = loss
    """

    def __init__(self, settings: Settings):
        self.s = settings

    def train(self, csv_path: str):
        try:
            import pandas as pd
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import classification_report
            import joblib

            df = pd.read_csv(csv_path)
            cols = ["fvg_pips", "dist_choch_pips", "atr_pips", "momentum",
                    "dist_liq_pips", "htf_aligned", "structure_score", "rr"]

            X = df[cols].values
            y = df["outcome"].values

            # Normalizza allo stesso modo di _extract
            norms = np.array([15, 80, 40, 0.015, 150, 1, 1, 4], dtype=float)
            X_n   = np.clip(X / norms, 0, 1)
            X_n[:, 1] = 1 - X_n[:, 1]   # dist_choch: inverso
            X_n[:, 2] = 1 - X_n[:, 2]   # atr: inverso

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_n, y, test_size=0.2, random_state=42, stratify=y)

            model = GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.04,
                max_depth=4, subsample=0.8, random_state=42)
            model.fit(X_tr, y_tr)

            scores = cross_val_score(model, X_n, y, cv=5, scoring="roc_auc")
            print(f"CV AUC: {scores.mean():.3f} ± {scores.std():.3f}")
            print(classification_report(y_te, model.predict(X_te)))

            os.makedirs(os.path.dirname(self.s.AI_MODEL_PATH), exist_ok=True)
            joblib.dump(model, self.s.AI_MODEL_PATH)
            log.info(f"Modello salvato: {self.s.AI_MODEL_PATH}")

        except ImportError as e:
            log.error(f"Librerie mancanti: {e}")
