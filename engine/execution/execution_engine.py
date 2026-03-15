"""ExecutionEngine — Full signal pipeline orchestrator.

Signal pipeline (execution order):
    UniverseEngine → MacroEngine → MetadronCube → AlphaOptimizer → ExecutionEngine

ML Vote Ensemble (5 tiers, each votes ±1):
    Tier-1  Pure-numpy 2-layer net
    Tier-2  Momentum/mean-reversion voter
    Tier-3  Volatility regime voter
    Tier-4  Monte Carlo voter (ARIMA-like + noise)
    Tier-5  Quality tier voter (top-down + bottom-up)

effective_min_edge = 2.0 + max(0, -vote_score) bps
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from ..data.universe_engine import UniverseEngine, get_engine, SECTOR_ETFS
from ..data.yahoo_data import get_returns, get_adj_close, get_market_stats
from ..signals.macro_engine import MacroEngine, MacroSnapshot, MarketRegime
from ..signals.metadron_cube import MetadronCube, CubeOutput
from ..ml.alpha_optimizer import AlphaOptimizer, AlphaOutput, AlphaSignal
from ..portfolio.beta_corridor import BetaCorridor, BetaState, BetaAction
from .paper_broker import (
    PaperBroker, OrderSide, SignalType, Position,
)


# ---------------------------------------------------------------------------
# ML Vote Ensemble (pure-numpy, no external ML frameworks required)
# ---------------------------------------------------------------------------
@dataclass
class VoteResult:
    ticker: str
    score: float = 0.0        # Aggregate vote score [-5, +5]
    votes: dict = field(default_factory=dict)  # tier → vote
    signal: SignalType = SignalType.HOLD
    edge_bps: float = 0.0


class MLVoteEnsemble:
    """5-tier vote ensemble. Each tier votes ±1. Pure numpy."""

    def vote(self, ticker: str, returns: pd.Series, alpha_signal: Optional[AlphaSignal] = None) -> VoteResult:
        result = VoteResult(ticker=ticker)
        if returns.empty or len(returns) < 21:
            return result

        r = returns.values

        # Tier 1: Simple neural net (2-layer, pure numpy)
        result.votes["T1_neural"] = self._tier1_neural(r)

        # Tier 2: Momentum/mean-reversion
        result.votes["T2_momentum"] = self._tier2_momentum(r)

        # Tier 3: Volatility regime
        result.votes["T3_vol_regime"] = self._tier3_vol(r)

        # Tier 4: Monte Carlo
        result.votes["T4_monte_carlo"] = self._tier4_mc(r)

        # Tier 5: Quality tier
        result.votes["T5_quality"] = self._tier5_quality(alpha_signal)

        # Aggregate
        result.score = sum(result.votes.values())
        result.edge_bps = 2.0 + max(0, -result.score)

        # Signal assignment
        if result.score >= 3:
            result.signal = SignalType.ML_AGENT_BUY
        elif result.score >= 1:
            result.signal = SignalType.QUALITY_BUY
        elif result.score <= -3:
            result.signal = SignalType.ML_AGENT_SELL
        elif result.score <= -1:
            result.signal = SignalType.QUALITY_SELL
        else:
            result.signal = SignalType.HOLD

        return result

    def _tier1_neural(self, r: np.ndarray) -> int:
        """Pure-numpy 2-layer net."""
        np.random.seed(abs(hash(r.tobytes())) % (2**31))
        x = np.array([r[-5:].mean(), r[-20:].mean(), r.std()])
        w1 = np.random.randn(3, 4) * 0.1
        b1 = np.zeros(4)
        w2 = np.random.randn(4, 1) * 0.1
        h = np.tanh(x @ w1 + b1)
        out = float(h @ w2)
        # Use the actual return momentum as the real signal
        momentum = r[-5:].mean()
        combined = out * 0.3 + momentum * 100 * 0.7
        return 1 if combined > 0 else -1

    def _tier2_momentum(self, r: np.ndarray) -> int:
        """Momentum voter: 20d momentum vs 60d."""
        if len(r) < 60:
            return 0
        mom_short = r[-20:].sum()
        mom_long = r[-60:].sum()
        if mom_short > 0 and mom_long > 0:
            return 1
        elif mom_short < 0 and mom_long < 0:
            return -1
        return 0

    def _tier3_vol(self, r: np.ndarray) -> int:
        """Volatility regime: low vol = bullish, high vol = bearish."""
        if len(r) < 60:
            return 0
        vol_recent = r[-20:].std()
        vol_long = r[-60:].std()
        ratio = vol_recent / vol_long if vol_long > 0 else 1.0
        if ratio < 0.8:
            return 1   # Vol compression = bullish
        elif ratio > 1.3:
            return -1  # Vol expansion = bearish
        return 0

    def _tier4_mc(self, r: np.ndarray) -> int:
        """Monte Carlo voter: simulate 100 paths, count positive."""
        if len(r) < 20:
            return 0
        mu = r[-20:].mean()
        sigma = r[-20:].std()
        if sigma == 0:
            return 0
        np.random.seed(42)
        paths = mu + sigma * np.random.randn(100)
        pct_positive = (paths > 0).mean()
        if pct_positive > 0.55:
            return 1
        elif pct_positive < 0.45:
            return -1
        return 0

    def _tier5_quality(self, alpha_signal: Optional[AlphaSignal]) -> int:
        """Quality tier voter."""
        if alpha_signal is None:
            return 0
        tier = alpha_signal.quality_tier
        if tier in ("A", "B"):
            return 1
        elif tier in ("F", "G"):
            return -1
        return 0


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """Full pipeline orchestrator: Universe → Macro → Cube → Alpha → Execute.

    Runs the complete Metadron Capital investment engine.
    """

    def __init__(
        self,
        initial_nav: float = 1_000_000.0,
        top_n_per_sector: int = 5,
    ):
        self.universe = get_engine()
        self.macro = MacroEngine()
        self.cube = MetadronCube()
        self.alpha = AlphaOptimizer()
        self.beta = BetaCorridor(nav=initial_nav)
        self.broker = PaperBroker(initial_cash=initial_nav)
        self.ensemble = MLVoteEnsemble()
        self.top_n = top_n_per_sector
        self._last_run: Optional[dict] = None

    def run_pipeline(self) -> dict:
        """Execute the full signal pipeline.

        Returns dict with all stage outputs.
        """
        result = {"timestamp": datetime.now().isoformat(), "stages": {}}

        # Stage 1: Universe
        self.universe.load()
        result["stages"]["universe"] = {
            "total_equities": self.universe.size(),
            "sectors": self.universe.get_sectors(),
        }

        # Stage 2: Macro analysis
        macro_snap = self.macro.analyze()
        result["stages"]["macro"] = {
            "regime": macro_snap.regime.value,
            "vix": macro_snap.vix,
            "spy_1m": macro_snap.spy_return_1m,
            "spy_3m": macro_snap.spy_return_3m,
            "sector_rankings": macro_snap.sector_rankings,
        }

        # Stage 3: MetadronCube
        cube_out = self.cube.compute(macro_snap)
        result["stages"]["cube"] = {
            "regime": cube_out.regime.value,
            "target_beta": cube_out.target_beta,
            "beta_cap": cube_out.beta_cap,
            "max_leverage": cube_out.max_leverage,
            "risk_budget": cube_out.risk_budget_pct,
            "sleeves": cube_out.sleeves.as_dict(),
            "liquidity": cube_out.liquidity.value,
            "risk": cube_out.risk.value,
        }

        # Stage 4: Select top names from leading sectors
        leader_sectors = cube_out.flow.leader_sectors
        if not leader_sectors:
            leader_sectors = list(macro_snap.sector_rankings.keys())[:4]

        selected_tickers = []
        for sector in leader_sectors:
            secs = self.universe.get_by_sector(sector)
            secs_sorted = sorted(secs, key=lambda s: s.market_cap, reverse=True)
            selected_tickers.extend([s.ticker for s in secs_sorted[:self.top_n]])

        # Add ETF proxies for diversification
        for sector in leader_sectors:
            etf = SECTOR_ETFS.get(sector)
            if etf and etf not in selected_tickers:
                selected_tickers.append(etf)

        if not selected_tickers:
            selected_tickers = ["SPY", "QQQ", "IWM", "XLK", "XLF"]

        result["stages"]["selection"] = {
            "leader_sectors": leader_sectors,
            "selected_tickers": selected_tickers[:30],  # Cap at 30
        }

        # Stage 5: Alpha optimisation
        alpha_out = self.alpha.optimize(selected_tickers[:20])
        alpha_map = {s.ticker: s for s in alpha_out.signals}
        result["stages"]["alpha"] = {
            "expected_return": alpha_out.expected_annual_return,
            "volatility": alpha_out.annual_volatility,
            "sharpe": alpha_out.sharpe_ratio,
            "weights": alpha_out.optimal_weights,
            "top_signals": [
                {"ticker": s.ticker, "tier": s.quality_tier, "alpha": s.alpha_pred}
                for s in alpha_out.signals[:10]
            ],
        }

        # Stage 6: Beta corridor
        beta_state, beta_action = self.beta.run_cycle(
            regime_beta_cap=cube_out.beta_cap,
        )
        result["stages"]["beta"] = {
            "target_beta": beta_state.target_beta,
            "current_beta": beta_state.current_beta,
            "Rm": beta_state.Rm,
            "sigma_m": beta_state.sigma_m,
            "corridor": beta_state.corridor_position,
            "action": beta_action.action,
        }

        # Stage 7: ML vote ensemble + execution
        trades = []
        nav = self.broker.compute_nav()
        equity_budget = nav * cube_out.sleeves.p1_directional_equity

        for ticker, weight in alpha_out.optimal_weights.items():
            if weight < 0.01:
                continue

            # Get returns for voting
            try:
                rets = get_returns(ticker, start=(
                    pd.Timestamp.now() - pd.Timedelta(days=300)
                ).strftime("%Y-%m-%d"))
                if isinstance(rets, pd.DataFrame) and not rets.empty:
                    ticker_rets = rets.iloc[:, 0]
                else:
                    continue
            except Exception:
                continue

            # Vote
            vote = self.ensemble.vote(
                ticker, ticker_rets,
                alpha_signal=alpha_map.get(ticker),
            )

            # Execute if vote is actionable
            if vote.signal in (SignalType.ML_AGENT_BUY, SignalType.QUALITY_BUY):
                target_value = equity_budget * weight
                price = self.broker._get_current_price(ticker)
                if price > 0:
                    qty = max(1, int(target_value / price))
                    order = self.broker.place_order(
                        ticker=ticker,
                        side=OrderSide.BUY,
                        quantity=qty,
                        signal_type=vote.signal,
                        reason=f"Vote={vote.score:.1f} Alpha={alpha_map.get(ticker, AlphaSignal(ticker=ticker)).alpha_pred:.4f}",
                    )
                    trades.append({
                        "ticker": ticker,
                        "side": "BUY",
                        "qty": qty,
                        "price": price,
                        "vote_score": vote.score,
                        "signal": vote.signal.value,
                    })

            elif vote.signal in (SignalType.ML_AGENT_SELL, SignalType.QUALITY_SELL):
                pos = self.broker.get_position(ticker)
                if pos and pos.quantity > 0:
                    order = self.broker.place_order(
                        ticker=ticker,
                        side=OrderSide.SELL,
                        quantity=pos.quantity,
                        signal_type=vote.signal,
                        reason=f"Vote={vote.score:.1f} SELL signal",
                    )
                    trades.append({
                        "ticker": ticker,
                        "side": "SELL",
                        "qty": pos.quantity,
                        "vote_score": vote.score,
                        "signal": vote.signal.value,
                    })

        # Beta rebalance via SPY
        if beta_action.action != "HOLD":
            side = OrderSide.BUY if beta_action.action == "BUY" else OrderSide.SELL
            self.broker.place_order(
                ticker="SPY",
                side=side,
                quantity=beta_action.quantity,
                signal_type=SignalType.MICRO_PRICE_BUY if side == OrderSide.BUY else SignalType.MICRO_PRICE_SELL,
                reason=beta_action.reason,
            )

        result["stages"]["execution"] = {
            "trades": trades,
            "portfolio": self.broker.get_portfolio_summary(),
        }

        self._last_run = result
        return result

    def get_portfolio_summary(self) -> dict:
        return self.broker.get_portfolio_summary()

    def get_positions(self) -> dict:
        return {k: {"qty": v.quantity, "price": v.current_price, "pnl": v.unrealized_pnl}
                for k, v in self.broker.get_all_positions().items()}
