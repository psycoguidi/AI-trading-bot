"""
backtester.py
=============
Testa la strategia SMC su dati storici simulati o reali.

Esecuzione:
    python backtester.py
    python backtester.py --symbol XAUUSD --bars 800

OUTPUT:
    - Statistiche complete a schermo
    - logs/backtest_<timestamp>.csv  (trade log)
"""

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from config.settings import Settings
from core.data_processor import DataProcessor
from core.market_structure import MarketStructureEngine
from core.pattern_detector import PatternDetector
from ai.filter import AIFilter
from core.models import Direction, TradeSetup
from utils.synthetic_data import generate_smc_data, _resample
from utils.logger import get_logger

log = get_logger("backtester")


# ─────────────────────────────────────────────────────────────────────────────
class Backtester:

    def __init__(self, settings: Settings):
        self.s       = settings
        self.proc    = DataProcessor(settings)
        self.struct  = MarketStructureEngine(settings)
        self.pattern = PatternDetector(settings)
        self.ai      = AIFilter(settings)

    # ── Run ──────────────────────────────────────────────────────────────────

    def run(
        self,
        symbol: str,
        df_htf: pd.DataFrame,
        df_mtf: pd.DataFrame,
        df_ltf: pd.DataFrame,
    ) -> dict:

        balance = self.s.INITIAL_BALANCE
        equity  = [balance]
        records: List[dict] = []
        WINDOW  = max(self.s.SWING_LEFT * 6, 40)   # Lookback minimo adattivo

        n = len(df_mtf)
        if n < WINDOW + 10:
            log.error(f"Dati insufficienti: {n} barre MTF (servono almeno {WINDOW + 10})")
            return _empty_results(balance)

        log.info(f"Backtest {symbol}: {n} barre MTF | lookback={WINDOW}")

        for i in range(WINDOW, n):
            sl_h = df_htf.iloc[: max(1, i // 5)]   # HTF ha meno barre
            sl_m = df_mtf.iloc[:i]
            sl_l = df_ltf.iloc[: i * 5]             # LTF ha più barre

            if len(sl_h) < 10 or len(sl_l) < 20:
                equity.append(balance)
                continue

            try:
                htf = self.struct.analyze(self.proc.process(sl_h), symbol, self.s.TF_HTF)
                mtf = self.struct.analyze(self.proc.process(sl_m), symbol, self.s.TF_MTF)
                ltf_p = self.proc.process(sl_l)
                ltf = self.struct.analyze(ltf_p, symbol, self.s.TF_LTF)

                setup = self.pattern.detect(symbol, htf, mtf, ltf, ltf_p)
            except Exception as e:
                log.debug(f"Errore barra {i}: {e}")
                equity.append(balance)
                continue

            if setup is None:
                equity.append(balance)
                continue

            prob = self.ai.evaluate(setup)
            if prob < self.s.AI_THRESHOLD:
                equity.append(balance)
                continue

            # Simula esito con candele future
            future = df_mtf.iloc[i: i + 60]
            outcome, pnl_pct = _simulate_outcome(setup, future, self.s.RISK_PER_TRADE)

            pnl     = balance * pnl_pct
            balance = max(balance + pnl, 0.01)
            equity.append(balance)

            records.append({
                "bar":       i,
                "timestamp": str(df_mtf.index[i]),
                "direction": setup.direction.value,
                "entry":     round(setup.entry, 5),
                "sl":        round(setup.sl, 5),
                "tp":        round(setup.tp, 5),
                "rr":        round(setup.rr, 2),
                "ai_prob":   round(prob, 3),
                "fvg_pips":  round(setup.fvg_pips, 1),
                "htf_trend": htf.trend,
                "outcome":   outcome,
                "pnl_usd":   round(pnl, 2),
                "balance":   round(balance, 2),
            })
            log.info(
                f"  Trade {len(records):>3}: {setup.direction.value:<5} | "
                f"AI={prob:.0%} | RR={setup.rr:.1f} | {outcome.upper()}"
            )

        return _compile(records, equity, self.s.INITIAL_BALANCE)

    # ── Report ───────────────────────────────────────────────────────────────

    def print_report(self, r: dict):
        sep = "═" * 54
        print(f"\n{sep}")
        print(f"  BACKTEST REPORT  –  {r.get('symbol', 'N/A')}")
        print(sep)
        if r.get("total", 0) == 0:
            print("  ⚠️  Nessun trade generato.")
            print("      Prova con più barre o parametri più permissivi.")
            print(sep)
            return
        print(f"  Barre analizzate  : {r['bars_analyzed']:>8}")
        print(f"  Trade totali      : {r['total']:>8}")
        print(f"  Wins / Losses     : {r['wins']:>4} / {r['losses']:<4}")
        print(f"  Timeouts          : {r['timeouts']:>8}")
        print(f"  Win Rate          : {r['win_rate']:>7.1%}")
        print(f"  Profit Factor     : {r['profit_factor']:>8.2f}")
        print(f"  Avg Win           : ${r['avg_win']:>+9.2f}")
        print(f"  Avg Loss          : ${r['avg_loss']:>+9.2f}")
        print(f"  Max Drawdown      : {r['max_drawdown']:>7.1%}")
        print(f"  Net PnL           : ${r['net_pnl']:>+9.2f}")
        print(f"  Balance finale    : ${r['final_balance']:>9,.2f}")
        print(sep)

    def save_csv(self, r: dict, path: str):
        df = r.get("trades_df")
        if df is None or len(df) == 0:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        df.to_csv(path, index=False)
        log.info(f"Trade log salvato: {path}")


# ── Simulazione esito ─────────────────────────────────────────────────────────

def _simulate_outcome(
    setup: TradeSetup,
    future: pd.DataFrame,
    risk_pct: float,
) -> Tuple[str, float]:
    if len(future) == 0:
        return "timeout", 0.0

    for _, row in future.iterrows():
        h = float(row["high"])
        l = float(row["low"])

        if setup.direction == Direction.LONG:
            if l <= setup.sl:
                return "loss", -risk_pct
            if h >= setup.tp:
                return "win", risk_pct * setup.rr
        else:
            if h >= setup.sl:
                return "loss", -risk_pct
            if l <= setup.tp:
                return "win", risk_pct * setup.rr

    return "timeout", 0.0


# ── Statistiche ───────────────────────────────────────────────────────────────

def _compile(records: List[dict], equity: List[float], initial: float) -> dict:
    if not records:
        r = _empty_results(initial)
        r["bars_analyzed"] = len(equity)
        return r

    df     = pd.DataFrame(records)
    wins   = df[df["outcome"] == "win"]
    losses = df[df["outcome"] == "loss"]

    gross_win  = wins["pnl_usd"].sum()   if len(wins)   else 0.0
    gross_loss = losses["pnl_usd"].sum() if len(losses) else 0.0
    pf = abs(gross_win / gross_loss) if gross_loss != 0 else float("inf")

    eq   = np.array(equity, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = ((eq - peak) / np.where(peak > 0, peak, 1)).min()

    return {
        "bars_analyzed": len(equity),
        "total":         len(df),
        "wins":          len(wins),
        "losses":        len(losses),
        "timeouts":      int((df["outcome"] == "timeout").sum()),
        "win_rate":      len(wins) / len(df),
        "profit_factor": pf,
        "avg_win":       wins["pnl_usd"].mean()   if len(wins)   else 0.0,
        "avg_loss":      losses["pnl_usd"].mean() if len(losses) else 0.0,
        "max_drawdown":  float(dd),
        "net_pnl":       float(eq[-1] - initial),
        "final_balance": float(eq[-1]),
        "equity_curve":  equity,
        "trades_df":     df,
    }


def _empty_results(initial: float) -> dict:
    return {
        "bars_analyzed": 0, "total": 0, "wins": 0, "losses": 0, "timeouts": 0,
        "win_rate": 0.0, "profit_factor": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
        "max_drawdown": 0.0, "net_pnl": 0.0, "final_balance": initial,
        "equity_curve": [initial], "trades_df": pd.DataFrame(), "symbol": "",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SMC Backtester")
    parser.add_argument("--symbol", default="EURUSD")
    parser.add_argument("--bars",   type=int, default=800,
                        help="Numero barre LTF da generare (default 800)")
    parser.add_argument("--seed",   type=int, default=42)
    args = parser.parse_args()

    s = Settings()
    s.BROKER       = "simulation"
    s.SWING_LEFT   = 3     # Più sensibile per backtest
    s.SWING_RIGHT  = 3
    s.AI_THRESHOLD = 0.45  # Soglia leggermente abbassata per vedere più trade

    sym  = args.symbol
    n    = args.bars

    log.info(f"Generazione dati sintetici SMC per {sym} ({n} barre LTF)…")

    base = {"EURUSD": 1.0850, "GBPUSD": 1.2700,
            "XAUUSD": 2350.0, "US30": 39000.0, "NAS100": 18000.0}.get(sym, 1.0850)
    pip  = 0.0001 if base < 100 else (0.1 if base < 1000 else 1.0)

    df_ltf = generate_smc_data(n, base, pip, args.seed)
    df_mtf = _resample(df_ltf, 5)
    df_htf = _resample(df_ltf, 60)

    log.info(f"Barre  →  HTF: {len(df_htf)} | MTF: {len(df_mtf)} | LTF: {len(df_ltf)}")

    bt = Backtester(s)
    results = bt.run(sym, df_htf, df_mtf, df_ltf)
    results["symbol"] = sym
    bt.print_report(results)

    if results["total"] > 0:
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = f"logs/backtest_{sym}_{ts}.csv"
        os.makedirs("logs", exist_ok=True)
        bt.save_csv(results, out)
        print(f"  📁 Trade log → {out}\n")


if __name__ == "__main__":
    main()
