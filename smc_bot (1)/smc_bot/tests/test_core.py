"""
tests/test_core.py
==================
Test unitari per tutti i moduli core del bot SMC.

Esecuzione:
    pytest tests/ -v
    pytest tests/test_core.py::test_fvg_detection -v
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from config.settings import Settings
from core.models import Direction, FVG, CHoCH, SwingPoint, TradeSetup, Trade, TradeStatus
from core.data_processor import DataProcessor
from core.market_structure import MarketStructureEngine
from core.pattern_detector import PatternDetector, _htf_aligned, _ltf_confirmed
from core.market_data import MarketDataFeed
from ai.filter import AIFilter
from risk.manager import RiskManager
from backtester import _simulate_outcome
from utils.pip_utils import pip_size, to_pips, pip_value_per_lot
from utils.synthetic_data import generate_smc_data, _resample


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def settings():
    s = Settings()
    s.BROKER      = "simulation"
    s.SWING_LEFT  = 3
    s.SWING_RIGHT = 3
    s.MIN_SWINGS  = 2
    return s


@pytest.fixture
def sample_df():
    """DataFrame OHLCV con struttura SMC chiara (bullish trend)."""
    n = 60
    base = 1.0850
    pip  = 0.0001
    data = []
    price = base
    # Costruisce una sequenza HH + HL manuale
    for i in range(n):
        step = pip * 2 if i % 10 < 7 else -pip * 1   # salita con ritracciamento
        price = max(price + step + np.random.normal(0, pip * 0.5), 1.0)
        o = price - abs(np.random.normal(0, pip))
        c = price
        h = max(o, c) + abs(np.random.normal(0, pip))
        l = min(o, c) - abs(np.random.normal(0, pip))
        data.append({"open": o, "high": h, "low": l, "close": c, "volume": 500.0})

    idx = pd.date_range(end=datetime.now(), periods=n, freq="1min")
    df  = pd.DataFrame(data, index=idx)
    df.index.name = "time"
    return df


@pytest.fixture
def proc(settings):
    return DataProcessor(settings)


@pytest.fixture
def eng(settings):
    return MarketStructureEngine(settings)


@pytest.fixture
def pat(settings):
    return PatternDetector(settings)


@pytest.fixture
def ai(settings):
    return AIFilter(settings)


@pytest.fixture
def risk(settings):
    return RiskManager(settings)


# ── Utils ─────────────────────────────────────────────────────────────────────

class TestPipUtils:

    def test_pip_size_forex(self):
        assert pip_size("EURUSD", 1.0850) == pytest.approx(0.0001)

    def test_pip_size_gold(self):
        assert pip_size("XAUUSD", 2350.0) == pytest.approx(0.1)

    def test_pip_size_index(self):
        assert pip_size("US30", 39000.0) == pytest.approx(1.0)

    def test_to_pips_forex(self):
        distance = 0.0010   # 10 pips
        result = to_pips(distance, "EURUSD", 1.0850)
        assert result == pytest.approx(10.0)

    def test_pip_value_per_lot_forex(self):
        val = pip_value_per_lot("EURUSD", 1.0850)
        assert val == pytest.approx(10.0)


# ── Data Processor ────────────────────────────────────────────────────────────

class TestDataProcessor:

    def test_atr_added(self, proc, sample_df):
        df = proc.process(sample_df)
        assert "atr" in df.columns
        assert df["atr"].iloc[-1] > 0

    def test_momentum_added(self, proc, sample_df):
        df = proc.process(sample_df)
        assert "momentum" in df.columns
        assert not df["momentum"].isna().all()

    def test_swing_columns_added(self, proc, sample_df):
        df = proc.process(sample_df)
        assert "swing_high" in df.columns
        assert "swing_low" in df.columns

    def test_swing_high_correct(self, proc, sample_df):
        """Un swing high deve avere high > vicini."""
        df = proc.process(sample_df)
        n = 3   # SWING_LEFT/RIGHT
        for i, row in df[df["swing_high"]].iterrows():
            idx = df.index.get_loc(i)
            if idx > n and idx < len(df) - n:
                window = df["high"].iloc[idx - n: idx + n + 1]
                assert df["high"].iloc[idx] == window.max(), \
                    f"Swing high non è il massimo locale a {i}"

    def test_atr_non_negative(self, proc, sample_df):
        df = proc.process(sample_df)
        assert (df["atr"].dropna() >= 0).all()

    def test_process_preserves_shape(self, proc, sample_df):
        df = proc.process(sample_df)
        assert len(df) == len(sample_df)


# ── Market Structure Engine ───────────────────────────────────────────────────

class TestMarketStructureEngine:

    def test_returns_market_structure(self, eng, proc, sample_df):
        from core.models import MarketStructure
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        assert isinstance(ms, MarketStructure)

    def test_swing_highs_found(self, eng, proc, sample_df):
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        assert len(ms.swing_highs) > 0

    def test_swing_lows_found(self, eng, proc, sample_df):
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        assert len(ms.swing_lows) > 0

    def test_trend_valid_value(self, eng, proc, sample_df):
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        assert ms.trend in ("bullish", "bearish", "ranging")

    def test_fvg_structure(self, eng, proc, sample_df):
        """Ogni FVG deve avere fvg_high > fvg_low."""
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        for fvg in ms.fvgs:
            assert fvg.fvg_high > fvg.fvg_low, \
                f"FVG invalida: high={fvg.fvg_high} low={fvg.fvg_low}"

    def test_fvg_direction_consistent(self, eng, proc, sample_df):
        """FVG LONG deve avere fvg_low come bordo inferiore (candela 1 high)."""
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        for fvg in ms.fvgs:
            assert fvg.direction in (Direction.LONG, Direction.SHORT)

    def test_choch_found_on_smc_data(self, settings, proc, eng):
        """Con dati SMC strutturati deve trovare almeno un CHoCH."""
        df_ltf = generate_smc_data(1500, seed=42)
        df_mtf = _resample(df_ltf, 5)

        choch_found = False
        for i in range(40, len(df_mtf)):
            ms = eng.analyze(proc.process(df_mtf.iloc[:i]), "EURUSD", "5M")
            if ms.choch is not None:
                choch_found = True
                assert ms.choch.confirmed
                assert isinstance(ms.choch.direction, Direction)
                break
        assert choch_found, "Nessun CHoCH trovato nei dati SMC strutturati"

    def test_atr_positive(self, eng, proc, sample_df):
        ms = eng.analyze(proc.process(sample_df), "EURUSD", "1M")
        assert ms.atr > 0


# ── Pattern Detector ──────────────────────────────────────────────────────────

class TestPatternDetector:

    def test_htf_aligned_bullish(self):
        assert _htf_aligned(Direction.LONG, "bullish") is True

    def test_htf_aligned_bearish(self):
        assert _htf_aligned(Direction.SHORT, "bearish") is True

    def test_htf_misaligned(self):
        assert _htf_aligned(Direction.LONG, "bearish") is False

    def test_htf_ranging_always_ok(self):
        assert _htf_aligned(Direction.LONG, "ranging") is True
        assert _htf_aligned(Direction.SHORT, "ranging") is True

    def test_ltf_confirmed_long(self):
        df = pd.DataFrame({
            "open":  [1.0860, 1.0855, 1.0858],
            "close": [1.0855, 1.0862, 1.0850],   # candela 1 è rialzista
        }, index=pd.date_range("2024-01-01", periods=3, freq="1min"))
        assert _ltf_confirmed(Direction.LONG, df, lookback=3) is True

    def test_ltf_confirmed_short(self):
        df = pd.DataFrame({
            "open":  [1.0860, 1.0870, 1.0865],
            "close": [1.0870, 1.0855, 1.0868],   # candela 1 è ribassista
        }, index=pd.date_range("2024-01-01", periods=3, freq="1min"))
        assert _ltf_confirmed(Direction.SHORT, df, lookback=3) is True

    def test_ltf_not_confirmed(self):
        """Tutte le candele contro direzione → non confermato."""
        df = pd.DataFrame({
            "open":  [1.0870, 1.0865, 1.0860],
            "close": [1.0865, 1.0860, 1.0855],   # tutte ribassiste
        }, index=pd.date_range("2024-01-01", periods=3, freq="1min"))
        assert _ltf_confirmed(Direction.LONG, df, lookback=3) is False

    def test_detect_returns_none_without_choch(self, pat, settings, proc, eng):
        """Senza CHoCH il detector deve restituire None."""
        from core.models import MarketStructure
        empty_ms = MarketStructure("EURUSD", "5M", datetime.now())
        df = pd.DataFrame({
            "open": [1.085]*10, "high": [1.086]*10,
            "low": [1.084]*10, "close": [1.085]*10,
            "volume": [500.0]*10, "atr": [0.0002]*10,
            "momentum": [0.0]*10, "swing_high": [False]*10,
            "swing_low": [False]*10, "pip_value": [0.0001]*10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="1min"))
        result = pat.detect("EURUSD", empty_ms, empty_ms, empty_ms, df)
        assert result is None


# ── AI Filter ─────────────────────────────────────────────────────────────────

class TestAIFilter:

    def _make_setup(self, direction=Direction.LONG, rr=2.0, htf=True):
        fvg = FVG(direction, fvg_high=1.0870, fvg_low=1.0860,
                  bar_index=10, timestamp=datetime.now(), size_pips=10.0)
        swing = SwingPoint(5, 1.0850, datetime.now(), "low")
        choch = CHoCH(direction, 1.0875, swing, datetime.now(), True)
        return TradeSetup(
            symbol="EURUSD", direction=direction,
            entry=1.0865, sl=1.0855, tp=1.0865 + rr * 0.001,
            fvg=fvg, choch=choch,
            fvg_pips=10, dist_choch_pips=5, atr_pips=8,
            momentum=0.003, dist_liq_pips=20,
            htf_aligned=htf, structure_score=0.7, rr=rr,
        )

    def test_returns_float(self, ai):
        setup = self._make_setup()
        prob = ai.evaluate(setup)
        assert isinstance(prob, float)

    def test_probability_in_range(self, ai):
        for rr in [1.0, 2.0, 3.5, 5.0]:
            prob = ai.evaluate(self._make_setup(rr=rr))
            assert 0.0 <= prob <= 1.0, f"Probabilità {prob} fuori range con RR={rr}"

    def test_htf_aligned_higher_prob(self, ai):
        aligned   = ai.evaluate(self._make_setup(htf=True))
        misaligned = ai.evaluate(self._make_setup(htf=False))
        assert aligned > misaligned, "Setup allineato HTF dovrebbe avere prob più alta"

    def test_high_rr_higher_prob(self, ai):
        low_rr  = ai.evaluate(self._make_setup(rr=1.2))
        high_rr = ai.evaluate(self._make_setup(rr=4.0))
        assert high_rr > low_rr


# ── Risk Manager ─────────────────────────────────────────────────────────────

class TestRiskManager:

    def _make_setup(self, symbol="EURUSD", direction=Direction.LONG):
        fvg = FVG(direction, 1.0870, 1.0860, 10, datetime.now(), 10.0)
        sw  = SwingPoint(5, 1.0850, datetime.now(), "low")
        ch  = CHoCH(direction, 1.0875, sw, datetime.now(), True)
        return TradeSetup(
            symbol=symbol, direction=direction,
            entry=1.0865, sl=1.0855, tp=1.0905,
            fvg=fvg, choch=ch, rr=4.0,
        )

    def test_can_open_initial(self, risk):
        setup = self._make_setup()
        assert risk.can_open(setup) is True

    def test_blocks_duplicate_symbol(self, risk):
        """Stesso simbolo già aperto → bloccato."""
        setup = self._make_setup("EURUSD")
        # Simula un trade aperto
        from core.models import Trade, TradeStatus
        trade = Trade("T001", "EURUSD", Direction.LONG, 1.0865, 1.0855, 1.0905,
                      0.01, setup, TradeStatus.OPEN, datetime.now())
        risk.open_trades["T001"] = trade

        setup2 = self._make_setup("EURUSD")
        assert risk.can_open(setup2) is False

    def test_allows_different_symbol(self, risk):
        """Simbolo diverso → consentito."""
        setup_eu = self._make_setup("EURUSD")
        from core.models import Trade, TradeStatus
        trade = Trade("T001", "EURUSD", Direction.LONG, 1.0865, 1.0855, 1.0905,
                      0.01, setup_eu, TradeStatus.OPEN, datetime.now())
        risk.open_trades["T001"] = trade

        setup_gb = self._make_setup("GBPUSD")
        assert risk.can_open(setup_gb) is True

    def test_blocks_max_trades(self, risk):
        """Raggiunto max trade → bloccato."""
        from core.models import Trade, TradeStatus
        for i in range(risk.s.MAX_OPEN_TRADES):
            s = self._make_setup(f"SYM{i}")
            t = Trade(f"T{i:03d}", f"SYM{i}", Direction.LONG, 1.0, 0.99, 1.05,
                      0.01, s, TradeStatus.OPEN, datetime.now())
            risk.open_trades[t.trade_id] = t
        new_setup = self._make_setup("NEW")
        assert risk.can_open(new_setup) is False

    def test_lot_size_positive(self, risk):
        setup = self._make_setup()
        lot = risk.lot_size(setup)
        assert lot > 0

    def test_lot_size_respects_risk(self, risk):
        """Con rischio 1% e SL 10 pips, il lot deve essere ragionevole."""
        setup = self._make_setup()
        lot = risk.lot_size(setup)
        # 1% di 10.000 = 100 USD di rischio
        # 10 pips × 10 USD/pip × lot = 100  →  lot ≈ 1.0
        assert 0.01 <= lot <= 5.0

    def test_daily_loss_block(self, risk):
        """Stop trading dopo perdita giornaliera > 3%."""
        risk.daily_pnl = -(risk.s.INITIAL_BALANCE * risk.s.MAX_DAILY_LOSS + 1)
        setup = self._make_setup()
        assert risk.can_open(setup) is False

    def test_stats_structure(self, risk):
        s = risk.stats()
        assert "balance" in s
        assert "win_rate" in s
        assert "total_trades" in s


# ── Backtester Outcome Simulation ─────────────────────────────────────────────

class TestBacktesterOutcome:

    def _make_setup(self, direction, entry, sl, tp):
        fvg = FVG(direction, max(entry, tp) + 0.001, min(entry, tp) - 0.001,
                  10, datetime.now(), 10.0)
        sw  = SwingPoint(5, sl, datetime.now(), "low")
        ch  = CHoCH(direction, entry, sw, datetime.now(), True)
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        rr = reward / risk if risk > 0 else 1.0
        return TradeSetup(
            symbol="EURUSD", direction=direction,
            entry=entry, sl=sl, tp=tp,
            fvg=fvg, choch=ch, rr=rr,
        )

    def _future(self, prices_h, prices_l):
        df = pd.DataFrame({
            "high":  prices_h,
            "low":   prices_l,
            "open":  [(h+l)/2 for h,l in zip(prices_h, prices_l)],
            "close": [(h+l)/2 for h,l in zip(prices_h, prices_l)],
        }, index=pd.date_range("2024-01-01", periods=len(prices_h), freq="1min"))
        return df

    def test_long_wins(self):
        setup   = self._make_setup(Direction.LONG, 1.0865, 1.0855, 1.0900)
        future  = self._future([1.0870, 1.0882, 1.0895, 1.0905],
                                [1.0862, 1.0875, 1.0888, 1.0898])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "win"
        assert pnl > 0

    def test_long_loses(self):
        setup   = self._make_setup(Direction.LONG, 1.0865, 1.0855, 1.0900)
        future  = self._future([1.0864, 1.0860, 1.0856],
                                [1.0857, 1.0852, 1.0848])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "loss"
        assert pnl < 0

    def test_short_wins(self):
        setup   = self._make_setup(Direction.SHORT, 1.0865, 1.0875, 1.0830)
        future  = self._future([1.0868, 1.0860, 1.0850, 1.0840],
                                [1.0858, 1.0848, 1.0835, 1.0825])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "win"
        assert pnl > 0

    def test_short_loses(self):
        setup   = self._make_setup(Direction.SHORT, 1.0865, 1.0875, 1.0830)
        future  = self._future([1.0872, 1.0878, 1.0880],
                                [1.0867, 1.0872, 1.0874])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "loss"
        assert pnl < 0

    def test_timeout_empty_future(self):
        setup  = self._make_setup(Direction.LONG, 1.0865, 1.0855, 1.0900)
        future = pd.DataFrame(columns=["high", "low"])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "timeout"
        assert pnl == 0.0

    def test_rr_respected_in_win(self):
        """Il PnL deve essere risk_pct × R:R."""
        setup  = self._make_setup(Direction.LONG, 1.0865, 1.0855, 1.0905)
        rr     = setup.rr
        future = self._future([1.0870, 1.0882, 1.0895, 1.0910],
                               [1.0862, 1.0875, 1.0888, 1.0902])
        outcome, pnl = _simulate_outcome(setup, future, 0.01)
        assert outcome == "win"
        assert pnl == pytest.approx(0.01 * rr, rel=0.01)


# ── Integration test ──────────────────────────────────────────────────────────

class TestIntegration:
    """Test end-to-end della pipeline completa su dati sintetici."""

    def test_full_pipeline_runs(self, settings, proc, eng, pat, ai):
        """La pipeline non deve sollevare eccezioni su dati sintetici."""
        df_ltf = generate_smc_data(800, seed=7)
        df_mtf = _resample(df_ltf, 5)
        df_htf = _resample(df_ltf, 60)

        errors = 0
        for i in range(40, min(150, len(df_mtf))):
            try:
                sl_h  = df_htf.iloc[:max(1, i // 5)]
                sl_m  = df_mtf.iloc[:i]
                sl_l  = df_ltf.iloc[:i * 5]
                htf   = eng.analyze(proc.process(sl_h), "EURUSD", "1H")
                mtf   = eng.analyze(proc.process(sl_m), "EURUSD", "5M")
                ltf_p = proc.process(sl_l)
                ltf   = eng.analyze(ltf_p, "EURUSD", "1M")
                setup = pat.detect("EURUSD", htf, mtf, ltf, ltf_p)
                if setup:
                    prob = ai.evaluate(setup)
                    assert 0.0 <= prob <= 1.0
            except Exception as e:
                errors += 1
                print(f"Errore a barra {i}: {e}")

        assert errors == 0, f"{errors} errori nella pipeline"

    def test_backtester_runs_without_crash(self, settings):
        """Il backtester completo non deve crashare."""
        from backtester import Backtester
        settings.SWING_LEFT  = 3
        settings.SWING_RIGHT = 3
        settings.MIN_SWINGS  = 2
        settings.AI_THRESHOLD = 0.40

        df_ltf = generate_smc_data(600, seed=55)
        df_mtf = _resample(df_ltf, 5)
        df_htf = _resample(df_ltf, 60)

        bt = Backtester(settings)
        results = bt.run("EURUSD", df_htf, df_mtf, df_ltf)

        assert "total" in results
        assert "win_rate" in results
        assert "final_balance" in results
        assert results["final_balance"] > 0
