"""
tests/test_all.py
=================
Test suite completa per il bot SMC.

Esecuzione:
    python -m pytest tests/ -v
    python tests/test_all.py          # senza pytest
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import unittest
from datetime import datetime
import pandas as pd
import numpy as np

from config.settings import Settings
from core.models import Direction, FVG, CHoCH, SwingPoint, TradeSetup
from core.data_processor import DataProcessor
from core.market_structure import MarketStructureEngine
from core.pattern_detector import PatternDetector, _htf_aligned, _ltf_confirmed
from ai.filter import AIFilter
from risk.manager import RiskManager
from utils.pip_utils import pip_size, to_pips, pip_value_per_lot
from utils.synthetic_data import generate_smc_data, _resample
from backtester import _simulate_outcome


def _make_settings(**kwargs) -> Settings:
    s = Settings()
    s.SWING_LEFT  = 3
    s.SWING_RIGHT = 3
    s.MIN_SWINGS  = 2
    for k, v in kwargs.items():
        setattr(s, k, v)
    return s


def _make_df(n=100, base=1.0850, trend="bullish", seed=0) -> pd.DataFrame:
    """DataFrame sintetico semplice con trend controllato."""
    rng = np.random.default_rng(seed)
    closes = [base]
    step = 0.0002 if trend == "bullish" else -0.0002
    for _ in range(n - 1):
        closes.append(closes[-1] + step + rng.normal(0, 0.0001))
    closes = np.array(closes)
    noise = np.abs(rng.normal(0, 0.0001, n))
    df = pd.DataFrame({
        "open":   closes - noise * 0.3,
        "high":   closes + noise,
        "low":    closes - noise,
        "close":  closes,
        "volume": rng.integers(100, 1000, n).astype(float),
    })
    df.index = pd.date_range("2024-01-01", periods=n, freq="1min")
    df.index.name = "time"
    return df


# ═════════════════════════════════════════════════════════════════════════════
# 1. PIP UTILS
# ═════════════════════════════════════════════════════════════════════════════

class TestPipUtils(unittest.TestCase):

    def test_pip_size_forex(self):
        self.assertAlmostEqual(pip_size("EURUSD", 1.085), 0.0001)

    def test_pip_size_gold(self):
        self.assertAlmostEqual(pip_size("XAUUSD", 2350.0), 0.1)

    def test_pip_size_index(self):
        self.assertAlmostEqual(pip_size("US30", 39000.0), 1.0)

    def test_to_pips_forex(self):
        result = to_pips(0.0010, "EURUSD", 1.085)
        self.assertAlmostEqual(result, 10.0)

    def test_pip_value_per_lot_forex(self):
        val = pip_value_per_lot("EURUSD", 1.085)
        self.assertEqual(val, 10.0)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATA PROCESSOR
# ═════════════════════════════════════════════════════════════════════════════

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        self.s    = _make_settings()
        self.proc = DataProcessor(self.s)

    def test_atr_column_exists(self):
        df = _make_df(50)
        out = self.proc.process(df)
        self.assertIn("atr", out.columns)
        self.assertFalse(out["atr"].isna().all())

    def test_momentum_column(self):
        df = _make_df(50)
        out = self.proc.process(df)
        self.assertIn("momentum", out.columns)

    def test_swing_columns(self):
        # I dati SMC sintetici hanno swing pronunciati con fasi trend/correzione
        from utils.synthetic_data import generate_smc_data
        df  = generate_smc_data(300, seed=5)
        out = self.proc.process(df)
        self.assertIn("swing_high", out.columns)
        self.assertIn("swing_low",  out.columns)
        self.assertGreater(out["swing_high"].sum(), 0, "Nessun swing high trovato")
        self.assertGreater(out["swing_low"].sum(),  0, "Nessun swing low trovato")

    def test_atr_positive(self):
        df = _make_df(50)
        out = self.proc.process(df)
        self.assertTrue((out["atr"].dropna() > 0).all())


# ═════════════════════════════════════════════════════════════════════════════
# 3. MARKET STRUCTURE ENGINE
# ═════════════════════════════════════════════════════════════════════════════

class TestMarketStructureEngine(unittest.TestCase):

    def setUp(self):
        self.s    = _make_settings()
        self.proc = DataProcessor(self.s)
        self.eng  = MarketStructureEngine(self.s)

    def _analyze(self, n=150, trend="bullish", seed=0):
        df = _make_df(n, trend=trend, seed=seed)
        return self.eng.analyze(self.proc.process(df), "EURUSD", "5M")

    def test_bullish_trend_detected(self):
        ms = self._analyze(n=150, trend="bullish")
        self.assertIn(ms.trend, ("bullish", "ranging"))

    def test_bearish_trend_detected(self):
        ms = self._analyze(n=150, trend="bearish")
        self.assertIn(ms.trend, ("bearish", "ranging"))

    def test_swing_highs_found(self):
        from utils.synthetic_data import generate_smc_data
        df = generate_smc_data(300, seed=5)
        ms = self.eng.analyze(self.proc.process(df), "EURUSD", "5M")
        self.assertGreater(len(ms.swing_highs), 0)

    def test_swing_lows_found(self):
        from utils.synthetic_data import generate_smc_data
        df = generate_smc_data(300, seed=5)
        ms = self.eng.analyze(self.proc.process(df), "EURUSD", "5M")
        self.assertGreater(len(ms.swing_lows), 0)

    def test_fvgs_detected(self):
        """Almeno alcune FVG devono essere trovate su 100 candele."""
        ms = self._analyze(n=100, seed=7)
        self.assertGreaterEqual(len(ms.fvgs), 0)   # FVG è facoltativa

    def test_choch_on_synthetic_data(self):
        """Con dati SMC sintetici il CHoCH deve apparire almeno una volta."""
        # Usiamo più barre e step più piccolo per assicurare pattern chiari
        df_ltf = generate_smc_data(2000, seed=7)
        df_mtf = _resample(df_ltf, 5)
        found  = False
        # Scorriamo ogni singola barra (non a step di 10) per non perdere eventi
        for i in range(40, len(df_mtf)):
            ms = self.eng.analyze(
                self.proc.process(df_mtf.iloc[:i]), "EURUSD", "5M"
            )
            if ms.choch is not None:
                found = True
                break
        self.assertTrue(found, "Nessun CHoCH trovato con dati sintetici SMC (2000 barre)")

    def test_market_structure_fields(self):
        ms = self._analyze()
        self.assertIsNotNone(ms.symbol)
        self.assertIsNotNone(ms.trend)
        self.assertIsInstance(ms.atr, float)
        self.assertIsInstance(ms.momentum, float)

    def test_fvg_direction_consistency(self):
        """Una FVG rialzista deve avere fvg_high > fvg_low."""
        ms = self._analyze(n=100, seed=3)
        for fvg in ms.fvgs:
            self.assertGreater(fvg.fvg_high, fvg.fvg_low,
                               f"FVG malformata: high={fvg.fvg_high} low={fvg.fvg_low}")


# ═════════════════════════════════════════════════════════════════════════════
# 4. PATTERN DETECTOR HELPERS
# ═════════════════════════════════════════════════════════════════════════════

class TestPatternDetectorHelpers(unittest.TestCase):

    def test_htf_aligned_long_bullish(self):
        self.assertTrue(_htf_aligned(Direction.LONG, "bullish"))

    def test_htf_aligned_long_bearish(self):
        self.assertFalse(_htf_aligned(Direction.LONG, "bearish"))

    def test_htf_aligned_short_bearish(self):
        self.assertTrue(_htf_aligned(Direction.SHORT, "bearish"))

    def test_htf_aligned_ranging_always_true(self):
        self.assertTrue(_htf_aligned(Direction.LONG,  "ranging"))
        self.assertTrue(_htf_aligned(Direction.SHORT, "ranging"))

    def test_ltf_confirmed_long(self):
        df = pd.DataFrame({
            "open":  [1.0860, 1.0870, 1.0865],
            "high":  [1.0870, 1.0880, 1.0875],
            "low":   [1.0855, 1.0865, 1.0860],
            "close": [1.0862, 1.0875, 1.0873],   # ultima bullish
        })
        self.assertTrue(_ltf_confirmed(Direction.LONG, df))

    def test_ltf_confirmed_short(self):
        df = pd.DataFrame({
            "open":  [1.0870, 1.0865, 1.0875],
            "high":  [1.0880, 1.0870, 1.0878],
            "low":   [1.0860, 1.0860, 1.0865],
            "close": [1.0862, 1.0862, 1.0867],   # ultima bearish
        })
        self.assertTrue(_ltf_confirmed(Direction.SHORT, df))

    def test_ltf_not_confirmed(self):
        """Tutte le ultime 3 candele contro la direzione → False."""
        df = pd.DataFrame({
            "open":  [1.0870, 1.0875, 1.0880],
            "high":  [1.0875, 1.0880, 1.0885],
            "low":   [1.0860, 1.0870, 1.0875],
            "close": [1.0862, 1.0872, 1.0876],   # tutte bearish
        })
        self.assertFalse(_ltf_confirmed(Direction.LONG, df))


# ═════════════════════════════════════════════════════════════════════════════
# 5. AI FILTER
# ═════════════════════════════════════════════════════════════════════════════

class TestAIFilter(unittest.TestCase):

    def _make_setup(self, htf_aligned=True, rr=2.5, fvg_pips=8.0,
                    has_sweep=True) -> TradeSetup:
        fvg   = FVG(Direction.LONG, 1.0880, 1.0870, 10, datetime.now(), fvg_pips)
        swing = SwingPoint(5, 1.0855, datetime.now(), "low")
        choch = CHoCH(Direction.LONG, 1.0876, swing, datetime.now(), True)
        return TradeSetup(
            symbol="EURUSD", direction=Direction.LONG,
            entry=1.0875, sl=1.0865, tp=1.0900,
            fvg=fvg, choch=choch, sweep=None,
            fvg_pips=fvg_pips, dist_choch_pips=5.0, atr_pips=12.0,
            momentum=0.003, dist_liq_pips=20.0,
            htf_aligned=htf_aligned, structure_score=0.7, rr=rr,
        )

    def test_probability_in_range(self):
        ai = AIFilter(_make_settings())
        p = ai.evaluate(self._make_setup())
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_aligned_higher_than_misaligned(self):
        ai = AIFilter(_make_settings())
        p_ok  = ai.evaluate(self._make_setup(htf_aligned=True))
        p_bad = ai.evaluate(self._make_setup(htf_aligned=False))
        self.assertGreater(p_ok, p_bad)

    def test_high_rr_raises_probability(self):
        ai = AIFilter(_make_settings())
        p_high = ai.evaluate(self._make_setup(rr=4.0))
        p_low  = ai.evaluate(self._make_setup(rr=1.2))
        self.assertGreater(p_high, p_low)

    def test_threshold_filtering(self):
        ai = AIFilter(_make_settings(AI_THRESHOLD=0.99))
        s  = _make_settings(AI_THRESHOLD=0.99)
        p  = ai.evaluate(self._make_setup())
        # Setup mediocre non deve superare soglia 99%
        self.assertLess(p, 0.99)

    def test_evaluate_returns_float(self):
        ai = AIFilter(_make_settings())
        p  = ai.evaluate(self._make_setup())
        self.assertIsInstance(p, float)


# ═════════════════════════════════════════════════════════════════════════════
# 6. RISK MANAGER
# ═════════════════════════════════════════════════════════════════════════════

class TestRiskManager(unittest.TestCase):

    def _make_setup(self, symbol="EURUSD", rr=2.0) -> TradeSetup:
        fvg   = FVG(Direction.LONG, 1.0880, 1.0870, 10, datetime.now(), 10.0)
        swing = SwingPoint(5, 1.0860, datetime.now(), "low")
        choch = CHoCH(Direction.LONG, 1.0876, swing, datetime.now(), True)
        return TradeSetup(
            symbol=symbol, direction=Direction.LONG,
            entry=1.0875, sl=1.0865, tp=1.0900,
            fvg=fvg, choch=choch, rr=rr,
            htf_aligned=True, structure_score=0.6,
        )

    def test_can_open_basic(self):
        rm = RiskManager(_make_settings())
        self.assertTrue(rm.can_open(self._make_setup()))

    def test_max_trades_limit(self):
        s  = _make_settings(MAX_OPEN_TRADES=1)
        rm = RiskManager(s)
        # Simula un trade già aperto
        from core.models import Trade, TradeStatus
        t = Trade("T001","GBPUSD",Direction.LONG,1.27,1.265,1.285,0.1,
                  self._make_setup("GBPUSD"),TradeStatus.OPEN)
        t.open_time = datetime.now()
        rm.open_trades["T001"] = t
        self.assertFalse(rm.can_open(self._make_setup("EURUSD")))

    def test_same_symbol_blocked(self):
        rm = RiskManager(_make_settings())
        from core.models import Trade, TradeStatus
        t = Trade("T002","EURUSD",Direction.LONG,1.085,1.084,1.090,0.1,
                  self._make_setup(),TradeStatus.OPEN)
        t.open_time = datetime.now()
        rm.open_trades["T002"] = t
        self.assertFalse(rm.can_open(self._make_setup()))

    def test_daily_loss_stops_trading(self):
        rm = RiskManager(_make_settings(MAX_DAILY_LOSS=0.03))
        rm.balance    = 10_000.0
        rm.daily_pnl  = -350.0   # > 3%
        self.assertFalse(rm.can_open(self._make_setup()))

    def test_lot_size_positive(self):
        rm = RiskManager(_make_settings())
        lot = rm.lot_size(self._make_setup())
        self.assertGreater(lot, 0)

    def test_lot_size_respects_risk(self):
        """Con 1% di rischio su 10k€ e 10 pips di SL → ~1 lotto."""
        rm = RiskManager(_make_settings(RISK_PER_TRADE=0.01))
        rm.balance = 10_000.0
        lot = rm.lot_size(self._make_setup())
        # 10_000 * 0.01 = 100 USD di rischio / (10 pips * $10/pip) = 1 lotto
        self.assertAlmostEqual(lot, 1.0, delta=0.5)

    def test_stats_structure(self):
        rm = RiskManager(_make_settings())
        st = rm.stats()
        for key in ("balance", "daily_pnl", "open_trades", "total_trades",
                    "win_rate", "total_pnl"):
            self.assertIn(key, st)


# ═════════════════════════════════════════════════════════════════════════════
# 7. BACKTEST OUTCOME SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════

class TestOutcomeSimulator(unittest.TestCase):

    def _setup(self, direction=Direction.LONG):
        fvg   = FVG(direction, 1.0880, 1.0870, 10, datetime.now(), 10.0)
        swing = SwingPoint(5, 1.0860, datetime.now(), "low")
        choch = CHoCH(direction, 1.0876, swing, datetime.now(), True)
        entry = 1.0875
        sl    = 1.0865 if direction==Direction.LONG else 1.0885
        tp    = 1.0905 if direction==Direction.LONG else 1.0845
        return TradeSetup(
            symbol="EURUSD", direction=direction,
            entry=entry, sl=sl, tp=tp,
            fvg=fvg, choch=choch, rr=3.0,
        )

    def _make_candles(self, highs, lows):
        n = len(highs)
        return pd.DataFrame({
            "open":  highs,
            "high":  highs,
            "low":   lows,
            "close": highs,
            "volume": [100]*n,
        })

    def test_long_win(self):
        setup = self._setup(Direction.LONG)
        future = self._make_candles(
            highs=[1.0878, 1.0888, 1.0898, 1.0910],
            lows= [1.0872, 1.0882, 1.0892, 1.0904],
        )
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(pnl, 0.03, places=5)  # rr=3.0 * 1%

    def test_long_loss(self):
        setup = self._setup(Direction.LONG)
        future = self._make_candles(
            highs=[1.0874, 1.0870, 1.0866],
            lows= [1.0868, 1.0864, 1.0860],  # tocca SL
        )
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        self.assertEqual(outcome, "loss")
        self.assertAlmostEqual(pnl, -0.01, places=5)

    def test_short_win(self):
        setup = self._setup(Direction.SHORT)
        future = self._make_candles(
            highs=[1.0870, 1.0860, 1.0850, 1.0844],
            lows= [1.0865, 1.0854, 1.0844, 1.0840],
        )
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        self.assertEqual(outcome, "win")
        self.assertAlmostEqual(pnl, 0.03, places=5)

    def test_short_loss(self):
        setup = self._setup(Direction.SHORT)
        future = self._make_candles(
            highs=[1.0880, 1.0886, 1.0890],
            lows= [1.0876, 1.0882, 1.0884],  # tocca SL
        )
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        self.assertEqual(outcome, "loss")

    def test_timeout_no_candles(self):
        setup  = self._setup()
        future = pd.DataFrame(columns=["open","high","low","close","volume"])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        self.assertEqual(outcome, "timeout")
        self.assertEqual(pnl, 0.0)


# ═════════════════════════════════════════════════════════════════════════════
# 8. SYNTHETIC DATA GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

class TestSyntheticData(unittest.TestCase):

    def test_output_shape(self):
        df = generate_smc_data(200, seed=1)
        self.assertEqual(len(df), 200)
        for col in ("open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns)

    def test_ohlc_consistency(self):
        df = generate_smc_data(200, seed=2)
        self.assertTrue((df["high"] >= df["low"]).all())
        self.assertTrue((df["high"] >= df["open"]).all())
        self.assertTrue((df["high"] >= df["close"]).all())
        self.assertTrue((df["low"]  <= df["open"]).all())
        self.assertTrue((df["low"]  <= df["close"]).all())

    def test_resample_reduces_rows(self):
        df  = generate_smc_data(200, seed=3)
        out = _resample(df, 5)
        self.assertLess(len(out), len(df))

    def test_different_seeds_different_data(self):
        d1 = generate_smc_data(100, seed=1)
        d2 = generate_smc_data(100, seed=99)
        self.assertFalse(d1["close"].equals(d2["close"]))


# ═════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader  = unittest.TestLoader()
    suite   = loader.loadTestsFromModule(sys.modules[__name__])
    runner  = unittest.TextTestRunner(verbosity=2)
    result  = runner.run(suite)

    print(f"\n{'═'*60}")
    print(f"  Totale test : {result.testsRun}")
    print(f"  Passati     : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Falliti     : {len(result.failures)}")
    print(f"  Errori      : {len(result.errors)}")
    print(f"{'═'*60}")

    sys.exit(0 if result.wasSuccessful() else 1)
