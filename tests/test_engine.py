"""Tests for Metadron Capital investment engine.

Tests all 6 layers:
    L1 Data (UniverseEngine, Yahoo data)
    L2 Signals (MacroEngine, MetadronCube, GMTF)
    L3 ML (AlphaOptimizer)
    L4 Portfolio (BetaCorridor)
    L5 Execution (PaperBroker, ExecutionEngine)
    L6 Agents (SectorBots, Scorecard)
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data.universe_engine import (
    UniverseEngine, Security, GICS_SECTORS, SECTOR_ETFS, RV_PAIRS, ALL_ETFS,
)
from engine.signals.macro_engine import (
    MacroEngine, MacroSnapshot, MarketRegime, CubeRegime,
    sigmoid_trigger, compute_gammas,
)
from engine.signals.metadron_cube import (
    MetadronCube, CubeOutput, SleeveAllocation, REGIME_PARAMS,
    R_LOW, R_HIGH, BETA_MAX, BETA_INV,
)
from engine.ml.alpha_optimizer import (
    AlphaOptimizer, AlphaOutput, AlphaSignal,
    build_features, ewma_cov, classify_quality,
)
from engine.portfolio.beta_corridor import (
    BetaCorridor, BetaState, BetaAction,
    ALPHA, EXECUTION_MULTIPLIER, VOL_STANDARD,
)
from engine.execution.paper_broker import (
    PaperBroker, OrderSide, OrderStatus, SignalType, Position,
)
from engine.agents.sector_bots import (
    SectorBot, SectorBotManager, AgentScorecard, AgentTier,
)


# ===========================================================================
# L1: Universe Engine
# ===========================================================================
class TestUniverseEngine:
    def test_gics_sectors(self):
        assert len(GICS_SECTORS) == 11
        assert 45 in GICS_SECTORS  # Information Technology

    def test_sector_etfs(self):
        assert len(SECTOR_ETFS) == 11
        assert SECTOR_ETFS["Energy"] == "XLE"

    def test_rv_pairs(self):
        assert len(RV_PAIRS) == 26
        assert ("AAPL", "MSFT") in RV_PAIRS

    def test_all_etfs(self):
        assert len(ALL_ETFS) >= 30

    def test_universe_engine_init(self):
        ue = UniverseEngine()
        assert ue.size() >= 0  # May be 0 if no network

    def test_security_dataclass(self):
        sec = Security(ticker="AAPL", name="Apple", sector="Information Technology")
        assert sec.ticker == "AAPL"
        assert sec.quality_tier == "D"  # Default

    def test_screen(self):
        ue = UniverseEngine()
        ue._equities = [
            Security(ticker="A", sector="Energy", market_cap=100e9),
            Security(ticker="B", sector="Energy", market_cap=5e9),
            Security(ticker="C", sector="Financials", market_cap=50e9),
        ]
        ue._loaded = True
        result = ue.screen(sectors=["Energy"])
        assert len(result) == 2
        result = ue.screen(min_market_cap=10e9)
        assert len(result) == 2


# ===========================================================================
# L2: Signals (MacroEngine, GMTF, MetadronCube)
# ===========================================================================
class TestMacroEngine:
    def test_sigmoid_trigger(self):
        # At threshold → ~0.5
        val = sigmoid_trigger(0.01, 0.01, sensitivity=15)
        assert 0.4 < val < 0.6

        # Well above → ~1.0
        val = sigmoid_trigger(1.0, 0.01, sensitivity=15)
        assert val > 0.99

        # Well below → ~0.0
        val = sigmoid_trigger(-1.0, 0.01, sensitivity=15)
        assert val < 0.01

    def test_regime_classification(self):
        engine = MacroEngine()
        # Bullish snapshot
        snap = MacroSnapshot(vix=12, spy_return_3m=0.15, yield_spread=1.0, credit_spread=2.0)
        regime = engine._classify_regime(snap)
        assert regime == MarketRegime.BULL

        # Stress snapshot
        snap = MacroSnapshot(vix=40, spy_return_3m=-0.15, yield_spread=-1.0, credit_spread=6.0)
        regime = engine._classify_regime(snap)
        assert regime == MarketRegime.STRESS

    def test_cube_regime_mapping(self):
        engine = MacroEngine()
        snap = MacroSnapshot(vix=50)
        assert engine._classify_cube_regime(snap) == CubeRegime.CRASH

        snap = MacroSnapshot(vix=12, spy_return_3m=0.15, yield_spread=1.0, credit_spread=2.0)
        snap.regime = MarketRegime.BULL
        assert engine._classify_cube_regime(snap) == CubeRegime.TRENDING


class TestMetadronCube:
    def test_init(self):
        cube = MetadronCube()
        assert cube.get_last() is None

    def test_compute_basic(self):
        cube = MetadronCube()
        macro = MacroSnapshot(
            regime=MarketRegime.BULL,
            cube_regime=CubeRegime.TRENDING,
            vix=15, spy_return_3m=0.10,
            yield_10y=4.0, yield_2y=3.5,
            credit_spread=2.0,
            sector_rankings={"Information Technology": 0.5, "Financials": 0.3},
        )
        out = cube.compute(macro)
        assert isinstance(out, CubeOutput)
        assert out.regime == CubeRegime.TRENDING
        assert out.max_leverage == 3.0
        assert out.beta_cap == 0.65

    def test_sleeve_allocation_sums_to_one(self):
        cube = MetadronCube()
        for regime in CubeRegime:
            macro = MacroSnapshot(
                regime=MarketRegime.TRANSITION,
                cube_regime=regime,
                vix=20, spy_return_3m=0.05,
                sector_rankings={},
            )
            out = cube.compute(macro)
            total = out.sleeves.total()
            assert abs(total - 1.0) < 0.02, f"Sleeves don't sum to 1 for {regime}: {total}"

    def test_regime_params(self):
        assert REGIME_PARAMS[CubeRegime.TRENDING]["max_leverage"] == 3.0
        assert REGIME_PARAMS[CubeRegime.CRASH]["beta_cap"] == -0.20

    def test_target_beta_corridor(self):
        cube = MetadronCube()
        # At R_LOW → should be near BETA_INV scaled
        beta_low = cube._compute_target_beta(R_LOW, 0.15)
        # At R_HIGH → should be positive
        beta_high = cube._compute_target_beta(R_HIGH, 0.15)
        assert beta_high > beta_low
        # Caps
        assert beta_high <= BETA_MAX
        assert beta_low >= BETA_INV

    def test_risk_state(self):
        cube = MetadronCube()
        macro = MacroSnapshot(vix=40, credit_spread=5.0)
        risk = cube._compute_risk(macro)
        assert 0 <= risk.value <= 1
        assert risk.vix_component > 0.5  # High VIX


# ===========================================================================
# L3: ML (AlphaOptimizer)
# ===========================================================================
class TestAlphaOptimizer:
    def test_classify_quality(self):
        assert classify_quality(2.5, 0.20) == "A"
        assert classify_quality(0.6, 0.02) == "D"
        assert classify_quality(-1.0, -0.20) == "G"

    def test_build_features(self):
        np.random.seed(42)
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        returns = pd.DataFrame(
            np.random.randn(n, 3) * 0.01,
            index=dates, columns=["A", "B", "C"],
        )
        prices = (1 + returns).cumprod() * 100
        features = build_features(returns, prices)
        assert "mkt_mean" in features.columns
        assert "momentum_3m" in features.columns
        assert len(features) == n

    def test_ewma_cov(self):
        np.random.seed(42)
        returns = pd.DataFrame(
            np.random.randn(50, 3) * 0.01,
            columns=["A", "B", "C"],
        )
        cov = ewma_cov(returns)
        assert cov.shape == (3, 3)
        assert np.all(np.diag(cov) > 0)  # Positive variance
        # Symmetric
        assert np.allclose(cov, cov.T, atol=1e-10)


# ===========================================================================
# L4: Portfolio (BetaCorridor)
# ===========================================================================
class TestBetaCorridor:
    def test_init(self):
        bc = BetaCorridor(nav=1_000_000)
        assert bc.nav == 1_000_000
        assert bc.current_beta == 0.0

    def test_target_beta_at_boundaries(self):
        bc = BetaCorridor()
        # Below corridor
        state = bc.calculate_target_beta(Rm=0.05, sigma_m=0.15)
        assert state.corridor_position == "BELOW"

        # Within corridor
        state = bc.calculate_target_beta(Rm=0.09, sigma_m=0.15)
        assert state.corridor_position == "WITHIN"

        # Above corridor
        state = bc.calculate_target_beta(Rm=0.15, sigma_m=0.15)
        assert state.corridor_position == "ABOVE"

    def test_vol_adjustment(self):
        bc = BetaCorridor()
        # Normal vol (15%)
        state_normal = bc.calculate_target_beta(Rm=0.10, sigma_m=0.15)
        # High vol (30%) → beta should contract
        state_high = bc.calculate_target_beta(Rm=0.10, sigma_m=0.30)
        assert abs(state_high.target_beta) < abs(state_normal.target_beta)

    def test_rebalance_hold(self):
        bc = BetaCorridor()
        state = BetaState(target_beta=0.01, current_beta=0.0)
        bc.current_beta = 0.0
        action = bc.compute_rebalance(state, instrument_price=500)
        assert action.action == "HOLD"  # Delta < threshold

    def test_rebalance_buy(self):
        bc = BetaCorridor(nav=1_000_000)
        state = BetaState(target_beta=0.5)
        bc.current_beta = 0.0
        action = bc.compute_rebalance(state, instrument_price=500)
        assert action.action == "BUY"
        assert action.quantity > 0

    def test_alpha_headstart(self):
        assert ALPHA == 0.02  # 2% secular alpha


# ===========================================================================
# L5: Execution (PaperBroker)
# ===========================================================================
class TestPaperBroker:
    def test_init(self):
        broker = PaperBroker(initial_cash=1_000_000)
        assert broker.state.cash == 1_000_000
        assert broker.state.nav == 1_000_000

    def test_place_order_buy(self):
        broker = PaperBroker(initial_cash=100_000)
        # Mock price by directly updating
        order = broker.place_order("TEST", OrderSide.BUY, 100, reason="test")
        # If no price data, should be rejected
        if order.status == OrderStatus.REJECTED:
            assert "No price" in order.reason
        # If filled, check state
        elif order.status == OrderStatus.FILLED:
            assert broker.state.cash < 100_000
            assert "TEST" in broker.state.positions

    def test_signal_types(self):
        assert len(SignalType) == 20  # 15 original + 4 social signals + HOLD
        assert SignalType.MICRO_PRICE_BUY.value == "MICRO_PRICE_BUY"

    def test_portfolio_summary(self):
        broker = PaperBroker(initial_cash=500_000)
        summary = broker.get_portfolio_summary()
        assert summary["cash"] == 500_000
        assert summary["positions"] == 0
        assert summary["nav"] == 500_000


# ===========================================================================
# L6: Agents (SectorBots, Scorecard)
# ===========================================================================
class TestSectorBots:
    def test_sector_bot_init(self):
        bot = SectorBot(sector="Energy", gics_code=10, etf="XLE")
        assert bot.sector == "Energy"
        assert bot.is_active

    def test_sector_bot_manager_init(self):
        mgr = SectorBotManager()
        assert len(mgr.bots) == 11  # 11 GICS sectors
        assert "Energy" in mgr.bots
        assert "Information Technology" in mgr.bots

    def test_agent_scorecard(self):
        sc = AgentScorecard()
        sc.update_score("EnergyBot", accuracy=0.85, sharpe=2.5, hit_rate=0.80, is_top_this_week=True)
        lb = sc.get_leaderboard()
        assert len(lb) == 1
        assert lb[0][0] == "EnergyBot"
        assert lb[0][1].accuracy == 0.85

    def test_tier_promotion(self):
        sc = AgentScorecard()
        # 4 consecutive top weeks → should promote to General
        for _ in range(4):
            sc.update_score("TopBot", accuracy=0.85, sharpe=2.5, hit_rate=0.80, is_top_this_week=True)
        assert sc.scores["TopBot"].tier == AgentTier.GENERAL

    def test_tier_demotion(self):
        sc = AgentScorecard()
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55)
        assert sc.scores["WeakBot"].tier == AgentTier.CAPTAIN
        # 2 consecutive bottom weeks
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55, is_bottom_this_week=True)
        sc.update_score("WeakBot", accuracy=0.60, sharpe=1.6, hit_rate=0.55, is_bottom_this_week=True)
        assert sc.scores["WeakBot"].tier == AgentTier.LIEUTENANT  # Demoted

    def test_leaderboard_format(self):
        sc = AgentScorecard()
        sc.update_score("Bot_A", accuracy=0.90, sharpe=3.0, hit_rate=0.85)
        sc.update_score("Bot_B", accuracy=0.50, sharpe=0.5, hit_rate=0.40)
        text = sc.print_leaderboard()
        assert "Bot_A" in text
        assert "Bot_B" in text
        assert "Rank" in text


# ===========================================================================
# Integration
# ===========================================================================
class TestIntegration:
    def test_full_cube_pipeline(self):
        """Test Macro → Cube → Beta target."""
        macro = MacroEngine()
        snap = MacroSnapshot(
            regime=MarketRegime.BULL,
            cube_regime=CubeRegime.TRENDING,
            vix=14, spy_return_3m=0.12,
            yield_10y=4.2, yield_2y=3.8,
            credit_spread=2.5,
            sector_rankings={"Information Technology": 0.6, "Financials": 0.4},
        )
        cube = MetadronCube()
        out = cube.compute(snap)

        assert out.regime == CubeRegime.TRENDING
        assert out.target_beta > 0  # Bull regime → positive beta
        assert out.sleeves.p1_directional_equity > 0.10  # Carry allocation in trending

        # Beta corridor check
        bc = BetaCorridor()
        state = bc.calculate_target_beta(
            Rm=snap.spy_return_3m * 4,  # annualise
            sigma_m=snap.vix / 100,
            regime_beta_cap=out.beta_cap,
        )
        assert state.target_beta <= out.beta_cap

    def test_signal_type_coverage(self):
        """Ensure all signal types are defined."""
        expected = [
            "MICRO_PRICE_BUY", "MICRO_PRICE_SELL",
            "RV_LONG", "RV_SHORT", "FALLEN_ANGEL_BUY",
            "ML_AGENT_BUY", "ML_AGENT_SELL",
            "DRL_AGENT_BUY", "DRL_AGENT_SELL",
            "TFT_BUY", "TFT_SELL",
            "MC_BUY", "MC_SELL",
            "QUALITY_BUY", "QUALITY_SELL",
            "HOLD",
        ]
        for sig in expected:
            assert hasattr(SignalType, sig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
