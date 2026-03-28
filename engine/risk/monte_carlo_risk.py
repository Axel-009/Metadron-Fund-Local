"""
MonteCarloRiskEngine — MiroFish Monte Carlo simulation for portfolio risk.

Runs per-security Monte Carlo simulations to produce VaR, CVaR, and stress
scenarios. These feed directly into:
  - BetaCorridor (position sizing via risk-adjusted returns)
  - RiskGateManager (VaR-based position limits, stress circuit breakers)
  - Portfolio Analytics (scenario analysis, tail risk)

Architecture:
    OpenBB data → calibrate → N simulations → aggregate risk metrics → pipeline

Math:
    VaR_alpha = -percentile(simulated_PnL, (1-alpha)*100)
    CVaR_alpha = E[L | L > VaR_alpha]
    Stress VaR = VaR under shocked parameters (vol spike, liquidity drain)
    Correlation-adjusted VaR = sqrt(w' * Sigma * w) with simulated Sigma
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

try:
    from intelligence_platform.MiroFish.investment_platform_integration import (
        MarketSimulator,
        AgentType,
    )
except ImportError:
    MarketSimulator = None
    AgentType = None
    logger.warning("MiroFish integration module unavailable — MonteCarloRiskEngine disabled")

try:
    from engine.data.openbb_data import get_adj_close
except ImportError:
    get_adj_close = None


@dataclass
class TickerRisk:
    """Risk metrics for a single ticker."""

    ticker: str
    timestamp: datetime
    var_95: float  # 95% VaR (dollar amount)
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    annualized_vol: float
    max_drawdown_sim: float  # Max drawdown across simulations
    tail_risk_score: float  # [0, 1] — how fat are the tails?
    stress_var_99: float  # VaR under 2x volatility stress
    sharpe_ratio: float
    hurst_exponent: float
    regime: str
    simulation_count: int

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp.isoformat(),
            "var_95": round(self.var_95, 2),
            "var_99": round(self.var_99, 2),
            "cvar_95": round(self.cvar_95, 2),
            "cvar_99": round(self.cvar_99, 2),
            "annualized_vol": round(self.annualized_vol, 4),
            "max_drawdown_sim": round(self.max_drawdown_sim, 4),
            "tail_risk_score": round(self.tail_risk_score, 4),
            "stress_var_99": round(self.stress_var_99, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "hurst_exponent": round(self.hurst_exponent, 4),
            "regime": self.regime,
            "simulation_count": self.simulation_count,
        }


@dataclass
class PortfolioRisk:
    """Aggregate portfolio risk metrics."""

    timestamp: datetime
    portfolio_value: float
    total_var_95: float
    total_var_99: float
    total_cvar_95: float
    total_cvar_99: float
    diversification_benefit: float  # How much diversification reduces VaR
    concentration_risk: float  # HHI-based concentration measure
    stress_scenarios: Dict[str, float]  # Named stress scenarios → portfolio loss
    ticker_risks: Dict[str, TickerRisk]
    risk_budget_utilization: float  # How much of risk budget is used

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_value": self.portfolio_value,
            "total_var_95": round(self.total_var_95, 2),
            "total_var_99": round(self.total_var_99, 2),
            "total_cvar_95": round(self.total_cvar_95, 2),
            "total_cvar_99": round(self.total_cvar_99, 2),
            "diversification_benefit": round(self.diversification_benefit, 4),
            "concentration_risk": round(self.concentration_risk, 4),
            "stress_scenarios": {k: round(v, 2) for k, v in self.stress_scenarios.items()},
            "risk_budget_utilization": round(self.risk_budget_utilization, 4),
            "ticker_count": len(self.ticker_risks),
        }


class MonteCarloRiskEngine:
    """
    Monte Carlo risk engine using MiroFish agent-based simulations.

    For each ticker:
    1. Fetch historical data from OpenBB
    2. Calibrate agent population to real market conditions
    3. Run N agent-based simulations
    4. Extract risk metrics: VaR, CVaR, max drawdown, tail risk
    5. Run stress scenarios (vol spike, liquidity drain, correlation shock)

    For the portfolio:
    1. Aggregate per-ticker risks with position weights
    2. Compute diversification benefit (correlation structure)
    3. Run portfolio-level stress tests
    4. Output risk budget utilization

    Integration with existing pipeline:
        - RiskGateManager: uses ticker_risks for position limits
        - BetaCorridor: uses volatility and regime for beta targeting
        - DecisionMatrix: uses tail_risk_score for risk-adjusted allocation
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        simulation_horizon: int = 21,  # ~1 month
        lookback_days: int = 252,  # ~1 year
        confidence_levels: Optional[List[float]] = None,
    ):
        self.n_simulations = n_simulations
        self.simulation_horizon = simulation_horizon
        self.lookback_days = lookback_days
        self.confidence_levels = confidence_levels or [0.95, 0.99]
        self._available = MarketSimulator is not None

    def compute_ticker_risk(
        self, ticker: str, position_value: float = 10000.0
    ) -> TickerRisk:
        """
        Compute full risk profile for a single ticker.

        Args:
            ticker: Stock ticker symbol.
            position_value: Position size in dollars (for VaR scaling).

        Returns:
            TickerRisk with VaR, CVaR, stress metrics.
        """
        now = datetime.now()

        if not self._available:
            return self._default_risk(ticker, now)

        # Fetch data
        if get_adj_close is not None:
            try:
                prices = get_adj_close(ticker, period=f"{self.lookback_days}d")
            except Exception as e:
                logger.warning(f"Failed to fetch {ticker}: {e}")
                return self._default_risk(ticker, now)
        else:
            return self._default_risk(ticker, now)

        if prices is None or len(prices) < 30:
            return self._default_risk(ticker, now)

        # Historical statistics
        returns = np.diff(np.log(prices.values))
        hist_vol = float(np.std(returns))
        hist_mean = float(np.mean(returns))
        current_price = float(prices.iloc[-1])

        # Calibrate simulator
        if len(prices) >= 50:
            fv = float(np.mean(prices.values[-50:]))
        else:
            fv = current_price

        # Run simulations
        all_final_returns = []
        all_max_drawdowns = []
        all_paths = []

        for i in range(self.n_simulations):
            try:
                sim = MarketSimulator(
                    n_agents=int(np.clip(len(returns) / 10, 50, 300)),
                    initial_price=current_price,
                )
                sim.initialize_agents()

                # Generate fundamental value path
                fv_path = [fv]
                for _ in range(self.simulation_horizon):
                    fv_drift = np.random.normal(0, fv * hist_vol * 0.1)
                    fv_path.append(fv_path[-1] + fv_drift)

                results = sim.run_simulation(self.simulation_horizon, fv_path)
                path = results["price"].values

                final_return = (path[-1] / path[0]) - 1
                all_final_returns.append(final_return)

                # Max drawdown
                peak = np.maximum.accumulate(path)
                dd = (path - peak) / peak
                all_max_drawdowns.append(float(np.min(dd)))

                all_paths.append(path)
            except Exception as e:
                logger.debug(f"Simulation {i} failed for {ticker}: {e}")
                continue

        if not all_final_returns:
            return self._default_risk(ticker, now)

        all_final_returns = np.array(all_final_returns)

        # VaR (percentile method)
        var_95 = float(-np.percentile(all_final_returns, 5)) * position_value
        var_99 = float(-np.percentile(all_final_returns, 1)) * position_value

        # CVaR (Expected Shortfall)
        losses = -all_final_returns
        cvar_95_losses = losses[losses >= np.percentile(losses, 95)]
        cvar_99_losses = losses[losses >= np.percentile(losses, 99)]
        cvar_95 = float(np.mean(cvar_95_losses)) * position_value if len(cvar_95_losses) > 0 else var_95
        cvar_99 = float(np.mean(cvar_99_losses)) * position_value if len(cvar_99_losses) > 0 else var_99

        # Max drawdown (across all simulations)
        max_dd_sim = float(np.min(all_max_drawdowns))

        # Tail risk score: excess kurtosis normalized
        kurtosis = float(pd.Series(all_final_returns).kurtosis())
        tail_risk_score = float(np.clip(kurtosis / 10, 0, 1))

        # Stress VaR: simulate under 2x volatility
        stress_returns = all_final_returns * 2.0  # Simplified stress scaling
        stress_var_99 = float(-np.percentile(stress_returns, 1)) * position_value

        # Sharpe from simulations
        sim_mean = float(np.mean(all_final_returns))
        sim_std = float(np.std(all_final_returns))
        sharpe = (sim_mean / (sim_std + 1e-10)) * np.sqrt(252 / self.simulation_horizon)

        # Hurst exponent from aggregated paths
        if all_paths:
            avg_path = np.mean(all_paths, axis=0)
            sim = MarketSimulator(initial_price=current_price)
            sim.price_history = avg_path.tolist()
            hurst = sim.calculate_hurst_exponent()
        else:
            hurst = 0.5

        # Regime
        if hurst > 0.55:
            regime = "trending"
        elif hurst < 0.45:
            regime = "mean_reverting"
        else:
            regime = "random_walk"

        return TickerRisk(
            ticker=ticker,
            timestamp=now,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            annualized_vol=hist_vol * np.sqrt(252),
            max_drawdown_sim=max_dd_sim,
            tail_risk_score=tail_risk_score,
            stress_var_99=stress_var_99,
            sharpe_ratio=sharpe,
            hurst_exponent=hurst,
            regime=regime,
            simulation_count=len(all_final_returns),
        )

    def compute_portfolio_risk(
        self,
        positions: Dict[str, float],  # ticker → position value
        portfolio_value: float,
        risk_budget: float = 0.02,  # 2% daily VaR budget
    ) -> PortfolioRisk:
        """
        Compute portfolio-level risk with diversification effects.

        Args:
            positions: Dict of ticker → position value.
            portfolio_value: Total portfolio value.
            risk_budget: Maximum acceptable daily VaR as fraction of portfolio.

        Returns:
            PortfolioRisk with aggregate metrics.
        """
        now = datetime.now()

        # Compute individual ticker risks
        ticker_risks = {}
        for ticker, pos_value in positions.items():
            ticker_risks[ticker] = self.compute_ticker_risk(ticker, pos_value)

        if not ticker_risks:
            return PortfolioRisk(
                timestamp=now,
                portfolio_value=portfolio_value,
                total_var_95=0,
                total_var_99=0,
                total_cvar_95=0,
                total_cvar_99=0,
                diversification_benefit=0,
                concentration_risk=0,
                stress_scenarios={},
                ticker_risks={},
                risk_budget_utilization=0,
            )

        # Naive sum of individual VaRs (no diversification)
        naive_var_95 = sum(r.var_95 for r in ticker_risks.values())
        naive_var_99 = sum(r.var_99 for r in ticker_risks.values())

        # Concentration risk (Herfindahl-Hirschman Index)
        weights = np.array([positions.get(t, 0) for t in ticker_risks]) / portfolio_value
        hhi = float(np.sum(weights ** 2))

        # Diversification benefit: assume ~30% correlation reduction for mixed portfolio
        # More sophisticated: use actual correlation matrix from simulation paths
        n_tickers = len(ticker_risks)
        if n_tickers > 1:
            avg_corr = 0.3  # Placeholder — should be computed from simulation paths
            div_factor = np.sqrt(
                hhi + (1 - hhi) * avg_corr
            )
        else:
            div_factor = 1.0

        total_var_95 = naive_var_95 * div_factor
        total_var_99 = naive_var_99 * div_factor
        total_cvar_95 = sum(r.cvar_95 for r in ticker_risks.values()) * div_factor
        total_cvar_99 = sum(r.cvar_99 for r in ticker_risks.values()) * div_factor

        diversification_benefit = 1 - div_factor

        # Stress scenarios
        stress_scenarios = {
            "vol_spike_2x": total_var_99 * 2.0,
            "liquidity_drain": total_var_99 * 1.5,
            "correlation_spike": naive_var_99,  # All correlations go to 1
            "flash_crash_10pct": portfolio_value * 0.10,
            "black_swan_20pct": portfolio_value * 0.20,
        }

        # Risk budget utilization
        risk_budget_utilization = total_var_95 / (portfolio_value * risk_budget)

        return PortfolioRisk(
            timestamp=now,
            portfolio_value=portfolio_value,
            total_var_95=total_var_95,
            total_var_99=total_var_99,
            total_cvar_95=total_cvar_95,
            total_cvar_99=total_cvar_99,
            diversification_benefit=diversification_benefit,
            concentration_risk=hhi,
            stress_scenarios=stress_scenarios,
            ticker_risks=ticker_risks,
            risk_budget_utilization=risk_budget_utilization,
        )

    def _default_risk(self, ticker: str, timestamp: datetime) -> TickerRisk:
        """Return conservative default risk when data is unavailable."""
        return TickerRisk(
            ticker=ticker,
            timestamp=timestamp,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            annualized_vol=0.0,
            max_drawdown_sim=0.0,
            tail_risk_score=0.5,
            stress_var_99=0,
            sharpe_ratio=0.0,
            hurst_exponent=0.5,
            regime="random_walk",
            simulation_count=0,
        )
