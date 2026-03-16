"""
ai/trainer.py
=============
Raccolta dati e addestramento del modello AI.

FLUSSO:
1. Il bot raccoglie trade con outcome reale → CSV in data/
2. Quando hai abbastanza dati (min 200 trade), esegui il trainer
3. Il modello viene salvato in models/smc_model.pkl
4. Il bot carica automaticamente il modello al prossimo avvio

UTILIZZO:
    python -m ai.trainer                    # addestra con dati in data/
    python -m ai.trainer --min-trades 100   # abbassa soglia minima
    python -m ai.trainer --evaluate         # solo valuta modello esistente

FEATURE (stesse di ai/filter.py):
    fvg_pips, dist_choch_pips, atr_pips, momentum, dist_liq_pips,
    htf_aligned, structure_score, rr  →  outcome (1=win, 0=loss)
"""

import argparse
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import Settings
from utils.logger import get_logger

log = get_logger("trainer")

FEATURE_COLS = [
    "fvg_pips",
    "dist_choch_pips",
    "atr_pips",
    "momentum",
    "dist_liq_pips",
    "htf_aligned",
    "structure_score",
    "rr",
]
LABEL_COL = "outcome_bin"   # 1 = win, 0 = loss/timeout


class ModelTrainer:

    def __init__(self, settings: Settings):
        self.s          = settings
        self.model_path = settings.AI_MODEL_PATH

    # ── Raccolta dati ────────────────────────────────────────────────────────

    def append_trade(self, trade, outcome: str):
        """
        Aggiunge un trade completato al dataset di training.
        Chiamato automaticamente dal bot dopo ogni trade chiuso.
        
        Args:
            trade:   oggetto Trade con setup popolato
            outcome: "win" | "loss" | "timeout"
        """
        row = {
            "timestamp":       datetime.now().isoformat(),
            "symbol":          trade.symbol,
            "direction":       trade.direction.value,
            "fvg_pips":        trade.setup.fvg_pips,
            "dist_choch_pips": trade.setup.dist_choch_pips,
            "atr_pips":        trade.setup.atr_pips,
            "momentum":        trade.setup.momentum,
            "dist_liq_pips":   trade.setup.dist_liq_pips,
            "htf_aligned":     int(trade.setup.htf_aligned),
            "structure_score": trade.setup.structure_score,
            "rr":              trade.setup.rr,
            "ai_prob":         trade.setup.ai_prob,
            "outcome":         outcome,
            "outcome_bin":     1 if outcome == "win" else 0,
            "pnl_usd":         trade.pnl_usd,
        }

        path = "data/training_data.csv"
        os.makedirs("data", exist_ok=True)
        write_header = not os.path.exists(path)

        with open(path, "a", newline="") as f:
            import csv
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        log.debug(f"Trade {outcome} salvato nel dataset.")

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, csv_path: str = "data/training_data.csv",
              min_trades: int = 200) -> bool:
        """
        Addestra il modello GradientBoosting sul CSV di training.
        
        Returns:
            True se il training è andato a buon fine, False altrimenti.
        """
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            from sklearn.model_selection import train_test_split, cross_val_score
            from sklearn.metrics import classification_report, confusion_matrix
            from sklearn.preprocessing import StandardScaler
            from sklearn.pipeline import Pipeline
            import joblib
        except ImportError:
            log.error("scikit-learn non installato. pip install scikit-learn joblib")
            return False

        # Carica dati
        if not os.path.exists(csv_path):
            log.error(f"Dataset non trovato: {csv_path}")
            return False

        df = pd.read_csv(csv_path)
        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL])
        df = df[df["outcome"] != "timeout"]   # Esclude timeout (ambigui)

        if len(df) < min_trades:
            log.warning(
                f"Dataset troppo piccolo: {len(df)} trade "
                f"(minimo {min_trades}). Continua a raccogliere dati."
            )
            return False

        log.info(f"Training su {len(df)} trade.")

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(int)

        # Split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pipeline con scaling + GBM
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            ))
        ])

        pipe.fit(X_tr, y_tr)

        # Valutazione
        y_pred = pipe.predict(X_te)
        y_prob = pipe.predict_proba(X_te)[:, 1]

        log.info("\n" + classification_report(y_te, y_pred,
                 target_names=["loss", "win"]))

        # Cross-validation
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
        log.info(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Feature importance
        model = pipe.named_steps["model"]
        importances = sorted(
            zip(FEATURE_COLS, model.feature_importances_),
            key=lambda x: -x[1]
        )
        log.info("Feature importance:")
        for feat, imp in importances:
            bar = "█" * int(imp * 40)
            log.info(f"  {feat:<20s} {imp:.3f}  {bar}")

        # Salva
        os.makedirs(os.path.dirname(self.model_path) or ".", exist_ok=True)
        joblib.dump(pipe, self.model_path)
        log.info(f"Modello salvato: {self.model_path}")

        # Salva report
        self._save_report(df, cv_scores, importances)
        return True

    # ── Valutazione modello esistente ────────────────────────────────────────

    def evaluate(self, csv_path: str = "data/training_data.csv"):
        """Valuta le performance del modello esistente su dati recenti."""
        try:
            import joblib
            from sklearn.metrics import classification_report, roc_auc_score
        except ImportError:
            log.error("scikit-learn non installato.")
            return

        if not os.path.exists(self.model_path):
            log.error(f"Modello non trovato: {self.model_path}")
            return

        if not os.path.exists(csv_path):
            log.error(f"Dataset non trovato: {csv_path}")
            return

        pipe = joblib.load(self.model_path)
        df   = pd.read_csv(csv_path).dropna(subset=FEATURE_COLS + [LABEL_COL])
        df   = df[df["outcome"] != "timeout"]

        X = df[FEATURE_COLS].values.astype(np.float32)
        y = df[LABEL_COL].values.astype(int)

        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]

        print("\n" + "═" * 50)
        print("  VALUTAZIONE MODELLO AI")
        print("═" * 50)
        print(classification_report(y, y_pred, target_names=["loss", "win"]))
        print(f"  ROC-AUC: {roc_auc_score(y, y_prob):.3f}")
        print(f"  Threshold attuale: {self.s.AI_THRESHOLD:.0%}")
        print("═" * 50)

        # Analisi threshold
        print("\n  Impatto threshold sul trade rate:")
        print(f"  {'Threshold':>10} | {'Trade %':>8} | {'Win Rate':>9}")
        print(f"  {'-'*33}")
        for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
            mask     = y_prob >= thr
            n_taken  = mask.sum()
            pct      = n_taken / len(y) if len(y) > 0 else 0
            wr       = y[mask].mean() if n_taken > 0 else 0
            marker   = " ← attuale" if abs(thr - self.s.AI_THRESHOLD) < 0.01 else ""
            print(f"  {thr:>10.0%} | {pct:>7.1%}  | {wr:>8.1%}{marker}")

    # ── Report ───────────────────────────────────────────────────────────────

    def _save_report(self, df, cv_scores, importances):
        path = f"data/training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(path, "w") as f:
            f.write(f"SMC Bot – Training Report\n")
            f.write(f"{'='*50}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trade totali: {len(df)}\n")
            f.write(f"Wins: {df[LABEL_COL].sum()} ({df[LABEL_COL].mean():.1%})\n")
            f.write(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}\n\n")
            f.write("Feature Importance:\n")
            for feat, imp in importances:
                f.write(f"  {feat:<22s} {imp:.4f}\n")
        log.info(f"Report salvato: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMC AI Trainer")
    parser.add_argument("--data",       default="data/training_data.csv")
    parser.add_argument("--min-trades", type=int, default=200)
    parser.add_argument("--evaluate",   action="store_true",
                        help="Solo valuta il modello esistente")
    args = parser.parse_args()

    s = Settings()
    trainer = ModelTrainer(s)

    if args.evaluate:
        trainer.evaluate(args.data)
    else:
        ok = trainer.train(args.data, args.min_trades)
        if ok:
            trainer.evaluate(args.data)


if __name__ == "__main__":
    main()
