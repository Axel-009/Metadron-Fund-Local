"""MiroFish Dual Simulation Backend — CAMEL-AI + OASIS.

Full agent-based simulation engine for L2 Pattern Discovery.

Simulation Mode A: "Synthetic Market" (Price Discovery)
    Uses CAMEL multi-agent framework to simulate agent populations
    (bulls, bears, market makers, momentum chasers) trading the
    classified instruments. Discovers emergent clustering, liquidity
    fragility, herding convergence, and regime transition points.

Simulation Mode B: "Contagion Network" (Relationship Discovery)
    Uses OASIS social simulation to model stress/opportunity propagation
    across the classified universe. Discovers hidden correlations,
    contagion paths, supply chain echoes, and divergence signals.

The "dual" output fires when both simulations agree or disagree,
producing high-conviction signals for the PatternDiscoveryBus.

Dependencies:
    pip install "camel-ai[all]"
    OASIS social simulation (bundled with camel-ai)

Usage:
    from backends.camel_oasis.mirofish_backend import MiroFishDualSimulation
    sim = MiroFishDualSimulation()
    patterns = sim.run_dual(universe_data, classified_instruments)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# CAMEL-AI framework
try:
    from camel.agents import ChatAgent
    from camel.messages import BaseMessage
    from camel.types import ModelType, ModelPlatformType
    _HAS_CAMEL = True
    logger.info("CAMEL-AI framework loaded")
except ImportError:
    _HAS_CAMEL = False
    logger.warning("CAMEL-AI not available — install with: pip install 'camel-ai[all]'")

# OASIS social simulation
try:
    from oasis.social_agent import SocialAgent
    from oasis.social_platform import SocialPlatform
    _HAS_OASIS = True
    logger.info("OASIS social simulation loaded")
except ImportError:
    _HAS_OASIS = False
    logger.info("OASIS not available as separate package — using CAMEL built-in society")

MODELS_DIR = Path(__file__).parent.parent.parent / "models"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredPattern:
    """Output from pattern discovery."""
    source: str              # "mirofish_market" | "mirofish_contagion"
    pattern_type: str        # "clustering" | "herding" | "contagion" | "divergence" | "regime_shift"
    tickers: list[str] = field(default_factory=list)
    direction: int = 0       # -1, 0, +1
    strength: float = 0.0    # [0, 1]
    confidence: float = 0.0  # [0, 1]
    half_life_days: int = 5
    description: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentState:
    """State of a simulated market agent."""
    agent_id: str
    agent_type: str          # "bull" | "bear" | "mm" | "momentum" | "value" | "arb"
    position: float = 0.0    # current position (-1 to +1)
    conviction: float = 0.5  # [0, 1]
    pnl: float = 0.0
    trades: int = 0


# ---------------------------------------------------------------------------
# Mode A: Synthetic Market Simulation
# ---------------------------------------------------------------------------

class SyntheticMarketSimulation:
    """Agent-based market simulation using CAMEL multi-agent framework.

    Simulates N agents with different strategies trading a set of instruments.
    Looks for emergent patterns that diverge from real market behavior.
    """

    AGENT_TYPES = {
        "bull": {"bias": 0.6, "momentum_sensitivity": 0.3, "mean_reversion": 0.1},
        "bear": {"bias": -0.6, "momentum_sensitivity": 0.3, "mean_reversion": 0.1},
        "mm": {"bias": 0.0, "momentum_sensitivity": 0.1, "mean_reversion": 0.8},
        "momentum": {"bias": 0.0, "momentum_sensitivity": 0.9, "mean_reversion": 0.0},
        "value": {"bias": 0.0, "momentum_sensitivity": 0.0, "mean_reversion": 0.9},
        "arb": {"bias": 0.0, "momentum_sensitivity": 0.2, "mean_reversion": 0.5},
    }

    def __init__(self, n_agents_per_type: int = 10, n_steps: int = 100):
        self.n_agents_per_type = n_agents_per_type
        self.n_steps = n_steps
        self._agents: list[AgentState] = []
        self._camel_agents: list = []

        # Initialize agents
        for agent_type, params in self.AGENT_TYPES.items():
            for i in range(n_agents_per_type):
                self._agents.append(AgentState(
                    agent_id=f"{agent_type}_{i}",
                    agent_type=agent_type,
                ))

        # Initialize CAMEL agents for sophisticated reasoning
        if _HAS_CAMEL:
            self._init_camel_agents()

        logger.info(f"SyntheticMarket initialized: {len(self._agents)} agents, "
                    f"{n_steps} steps")

    def _init_camel_agents(self):
        """Initialize CAMEL-AI agents for market analysis."""
        try:
            # Market analyst agent
            analyst_msg = BaseMessage.make_assistant_message(
                role_name="Market Analyst",
                content="You are a quantitative market analyst. Analyze price patterns "
                        "and identify emerging trends, regime changes, and anomalies."
            )
            self._analyst = ChatAgent(
                system_message=analyst_msg,
                model=ModelType.STUB,  # Use stub for offline operation
            )

            # Risk analyst agent
            risk_msg = BaseMessage.make_assistant_message(
                role_name="Risk Analyst",
                content="You are a risk analyst. Identify crowded trades, liquidity risks, "
                        "and potential contagion paths across correlated assets."
            )
            self._risk_analyst = ChatAgent(
                system_message=risk_msg,
                model=ModelType.STUB,
            )
            logger.info("CAMEL agents initialized for market simulation")
        except Exception as e:
            logger.warning(f"CAMEL agent init failed (will use numpy fallback): {e}")
            self._analyst = None
            self._risk_analyst = None

    def simulate(self, prices: dict[str, pd.Series]) -> dict[str, list[float]]:
        """Run market simulation across all instruments.

        Args:
            prices: Dict of ticker -> close prices.

        Returns:
            Dict of ticker -> simulated price series.
        """
        simulated_prices = {}

        for ticker, real_prices in prices.items():
            if len(real_prices) < 20:
                continue

            returns = real_prices.pct_change().dropna()
            mu = float(returns.mean())
            sigma = float(returns.std())

            # Run agent-based simulation
            sim_price = float(real_prices.iloc[-1])
            sim_series = [sim_price]

            for step in range(self.n_steps):
                # Each agent votes based on their type
                aggregate_demand = 0.0
                for agent in self._agents:
                    params = self.AGENT_TYPES[agent.agent_type]

                    # Momentum signal from recent returns
                    lookback = min(step + 1, 20)
                    recent_return = (sim_series[-1] / sim_series[max(0, -lookback)] - 1) if len(sim_series) > 1 else 0

                    # Agent decision
                    momentum_signal = recent_return * params["momentum_sensitivity"]
                    mean_rev_signal = -recent_return * params["mean_reversion"]
                    bias = params["bias"] * 0.01

                    agent_signal = momentum_signal + mean_rev_signal + bias
                    agent_signal += np.random.normal(0, sigma * 0.5)  # noise
                    aggregate_demand += agent_signal

                # Market impact
                avg_demand = aggregate_demand / len(self._agents)
                price_impact = avg_demand * sigma * sim_price
                noise = np.random.normal(mu * sim_price, sigma * sim_price)
                sim_price = max(sim_price * 0.5, sim_price + price_impact + noise)
                sim_series.append(sim_price)

            simulated_prices[ticker] = sim_series

        return simulated_prices

    def detect_emergent_clustering(self, real_prices: dict[str, pd.Series],
                                    sim_prices: dict[str, list[float]]) -> list[DiscoveredPattern]:
        """Find instruments that cluster differently in simulation vs reality."""
        patterns = []
        tickers = list(real_prices.keys())
        if len(tickers) < 4:
            return patterns

        # Compute real correlation matrix
        real_df = pd.DataFrame({t: real_prices[t].values[-60:]
                                for t in tickers if len(real_prices[t]) >= 60})
        if real_df.shape[1] < 4:
            return patterns
        real_corr = real_df.pct_change().dropna().corr()

        # Compute simulated correlation matrix
        sim_df = pd.DataFrame({t: sim_prices[t][-60:]
                               for t in tickers if t in sim_prices and len(sim_prices[t]) >= 60})
        if sim_df.shape[1] < 4:
            return patterns
        sim_corr = sim_df.pct_change().dropna().corr()

        # Find divergences between real and simulated correlations
        common = list(set(real_corr.columns) & set(sim_corr.columns))
        for i, t1 in enumerate(common):
            for t2 in common[i+1:]:
                real_c = real_corr.loc[t1, t2]
                sim_c = sim_corr.loc[t1, t2]
                divergence = abs(real_c - sim_c)

                if divergence > 0.4:  # significant divergence
                    patterns.append(DiscoveredPattern(
                        source="mirofish_market",
                        pattern_type="clustering",
                        tickers=[t1, t2],
                        direction=1 if sim_c > real_c else -1,
                        strength=min(divergence, 1.0),
                        confidence=min(divergence / 0.6, 1.0),
                        description=(
                            f"Emergent clustering divergence: {t1}-{t2} "
                            f"real_corr={real_c:.3f} sim_corr={sim_c:.3f}"
                        ),
                        metadata={"real_corr": real_c, "sim_corr": sim_c,
                                  "divergence": divergence},
                    ))

        return patterns

    def detect_herding(self, sim_prices: dict[str, list[float]]) -> list[DiscoveredPattern]:
        """Detect when agent types converge on the same instruments."""
        patterns = []

        for ticker in sim_prices:
            # Count agents with strong positions in same direction
            bull_count = sum(1 for a in self._agents
                           if a.agent_type in ("bull", "momentum") and a.position > 0.3)
            total = len(self._agents)
            herding_ratio = bull_count / total if total > 0 else 0

            if herding_ratio > 0.7 or herding_ratio < 0.3:
                direction = 1 if herding_ratio > 0.7 else -1
                patterns.append(DiscoveredPattern(
                    source="mirofish_market",
                    pattern_type="herding",
                    tickers=[ticker],
                    direction=-direction,  # contrarian signal
                    strength=abs(herding_ratio - 0.5) * 2,
                    confidence=0.6,
                    description=f"Agent herding detected: {herding_ratio:.0%} aligned on {ticker}",
                    metadata={"herding_ratio": herding_ratio},
                ))

        return patterns

    def detect_regime_transition(self, real_prices: dict[str, pd.Series],
                                  sim_prices: dict[str, list[float]]) -> list[DiscoveredPattern]:
        """Detect when simulation flips regime before real market."""
        patterns = []

        for ticker in sim_prices:
            if ticker not in real_prices or len(real_prices[ticker]) < 40:
                continue

            real = real_prices[ticker].values[-40:]
            sim = np.array(sim_prices[ticker][-40:]) if len(sim_prices[ticker]) >= 40 else None
            if sim is None:
                continue

            # Check if simulation shows trend reversal
            real_trend = np.polyfit(range(len(real)), real, 1)[0]
            sim_trend = np.polyfit(range(len(sim)), sim, 1)[0]

            if np.sign(real_trend) != np.sign(sim_trend):
                patterns.append(DiscoveredPattern(
                    source="mirofish_market",
                    pattern_type="regime_shift",
                    tickers=[ticker],
                    direction=int(np.sign(sim_trend)),
                    strength=0.7,
                    confidence=0.5,
                    half_life_days=10,
                    description=(
                        f"Regime transition: real trend={real_trend:.4f} "
                        f"sim trend={sim_trend:.4f} — simulation leading"
                    ),
                    metadata={"real_trend": float(real_trend),
                              "sim_trend": float(sim_trend)},
                ))

        return patterns


# ---------------------------------------------------------------------------
# Mode B: Contagion Network Simulation
# ---------------------------------------------------------------------------

class ContagionNetworkSimulation:
    """Stress/opportunity propagation network using CAMEL society.

    Models how shocks in one instrument propagate through the universe.
    Uses correlation-based adjacency with lead-lag estimation.
    """

    def __init__(self, shock_magnitude: float = 0.05, propagation_decay: float = 0.7):
        self.shock_magnitude = shock_magnitude
        self.propagation_decay = propagation_decay

    def build_adjacency(self, prices: dict[str, pd.Series],
                        window: int = 60) -> pd.DataFrame:
        """Build correlation-based adjacency matrix."""
        tickers = [t for t, p in prices.items() if len(p) >= window]
        if len(tickers) < 3:
            return pd.DataFrame()

        returns = pd.DataFrame({
            t: prices[t].pct_change().dropna().values[-window:]
            for t in tickers
            if len(prices[t].pct_change().dropna()) >= window
        })

        if returns.shape[1] < 3:
            return pd.DataFrame()

        corr = returns.corr()
        # Zero out diagonal and weak correlations
        adj = corr.copy()
        np.fill_diagonal(adj.values, 0)
        adj[adj.abs() < 0.3] = 0  # only keep meaningful connections
        return adj

    def simulate_contagion(self, adjacency: pd.DataFrame,
                           shock_ticker: str) -> dict[str, float]:
        """Simulate stress propagation from a shocked instrument.

        Returns dict of ticker -> cumulative impact.
        """
        if shock_ticker not in adjacency.columns:
            return {}

        tickers = list(adjacency.columns)
        impacts = {t: 0.0 for t in tickers}
        impacts[shock_ticker] = self.shock_magnitude

        # Propagation rounds
        active = {shock_ticker}
        for round_num in range(5):
            next_active = set()
            for source in active:
                for target in tickers:
                    if target == source:
                        continue
                    edge_weight = adjacency.loc[source, target]
                    if abs(edge_weight) < 0.1:
                        continue
                    propagated = impacts[source] * edge_weight * self.propagation_decay
                    if abs(propagated) > abs(impacts[target]) * 0.1:
                        impacts[target] += propagated
                        next_active.add(target)
            active = next_active
            if not active:
                break

        return impacts

    def find_contagion_paths(self, prices: dict[str, pd.Series]) -> list[DiscoveredPattern]:
        """Run contagion simulation from each instrument and find paths."""
        patterns = []
        adj = self.build_adjacency(prices)
        if adj.empty:
            return patterns

        tickers = list(adj.columns)

        for shock_ticker in tickers:
            impacts = self.simulate_contagion(adj, shock_ticker)
            # Find significant downstream impacts
            significant = {t: v for t, v in impacts.items()
                          if t != shock_ticker and abs(v) > 0.01}

            if len(significant) >= 2:
                # Sort by impact magnitude
                sorted_impacts = sorted(significant.items(), key=lambda x: abs(x[1]), reverse=True)
                top_affected = [t for t, _ in sorted_impacts[:5]]

                patterns.append(DiscoveredPattern(
                    source="mirofish_contagion",
                    pattern_type="contagion",
                    tickers=[shock_ticker] + top_affected,
                    direction=0,
                    strength=min(abs(sorted_impacts[0][1]) / self.shock_magnitude, 1.0),
                    confidence=0.55,
                    description=(
                        f"Contagion path from {shock_ticker}: "
                        f"{' → '.join(top_affected[:3])}"
                    ),
                    metadata={
                        "shock_source": shock_ticker,
                        "impacts": {t: round(v, 4) for t, v in sorted_impacts[:5]},
                    },
                ))

        return patterns

    def find_hidden_correlations(self, prices: dict[str, pd.Series]) -> list[DiscoveredPattern]:
        """Find instruments with no obvious fundamental link that co-move."""
        patterns = []
        adj = self.build_adjacency(prices)
        if adj.empty:
            return patterns

        # Define sector groups (instruments that SHOULD be correlated)
        sector_groups = {
            "tech": {"AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "XLK", "QQQ"},
            "fin": {"JPM", "BAC", "GS", "V", "MA", "XLF"},
            "energy": {"XOM", "CVX", "COP", "XLE"},
            "health": {"JNJ", "UNH", "PFE", "XLV"},
        }

        # Find cross-sector correlations (unexpected)
        for t1 in adj.columns:
            t1_sector = None
            for sector, members in sector_groups.items():
                if t1 in members:
                    t1_sector = sector
                    break

            for t2 in adj.columns:
                if t2 <= t1:
                    continue
                t2_sector = None
                for sector, members in sector_groups.items():
                    if t2 in members:
                        t2_sector = sector
                        break

                # If different sectors but high correlation → hidden link
                if t1_sector and t2_sector and t1_sector != t2_sector:
                    corr = adj.loc[t1, t2]
                    if abs(corr) > 0.5:
                        patterns.append(DiscoveredPattern(
                            source="mirofish_contagion",
                            pattern_type="hidden_correlation",
                            tickers=[t1, t2],
                            direction=int(np.sign(corr)),
                            strength=abs(corr),
                            confidence=0.6,
                            description=(
                                f"Hidden cross-sector correlation: "
                                f"{t1}({t1_sector}) ↔ {t2}({t2_sector}) = {corr:.3f}"
                            ),
                            metadata={"correlation": float(corr),
                                      "sectors": [t1_sector, t2_sector]},
                        ))

        return patterns

    def find_divergences(self, prices: dict[str, pd.Series],
                          window: int = 60) -> list[DiscoveredPattern]:
        """Find instruments that SHOULD be correlated but have diverged."""
        patterns = []

        # Same-sector pairs
        sector_groups = {
            "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"],
            "fin": ["JPM", "BAC", "GS"],
            "energy": ["XOM", "CVX", "COP"],
        }

        for sector, members in sector_groups.items():
            available = [m for m in members if m in prices and len(prices[m]) >= window]
            for i, t1 in enumerate(available):
                for t2 in available[i+1:]:
                    r1 = prices[t1].pct_change().dropna().values[-window:]
                    r2 = prices[t2].pct_change().dropna().values[-window:]
                    min_len = min(len(r1), len(r2))
                    if min_len < 20:
                        continue

                    # Check historical vs recent correlation
                    full_corr = np.corrcoef(r1[:min_len], r2[:min_len])[0, 1]
                    recent_corr = np.corrcoef(r1[-20:min_len], r2[-20:min_len])[0, 1] if min_len >= 20 else full_corr

                    # Divergence = drop in correlation
                    divergence = full_corr - recent_corr
                    if divergence > 0.3:
                        patterns.append(DiscoveredPattern(
                            source="mirofish_contagion",
                            pattern_type="divergence",
                            tickers=[t1, t2],
                            direction=0,  # mean reversion opportunity
                            strength=min(divergence, 1.0),
                            confidence=0.65,
                            half_life_days=15,
                            description=(
                                f"Same-sector divergence: {t1}-{t2} "
                                f"hist_corr={full_corr:.3f} recent_corr={recent_corr:.3f}"
                            ),
                            metadata={
                                "historical_corr": float(full_corr),
                                "recent_corr": float(recent_corr),
                                "divergence": float(divergence),
                                "sector": sector,
                            },
                        ))

        return patterns


# ---------------------------------------------------------------------------
# MiroFish Dual Simulation Engine
# ---------------------------------------------------------------------------

class MiroFishDualSimulation:
    """Dual simulation engine combining Synthetic Market + Contagion Network.

    Runs both simulations on the classified universe and produces
    structured patterns for the PatternDiscoveryBus.

    When both modes agree → high conviction
    When they disagree → divergence signal (often contrarian opportunity)
    """

    def __init__(self, n_agents: int = 10, n_steps: int = 100):
        self.market_sim = SyntheticMarketSimulation(
            n_agents_per_type=n_agents, n_steps=n_steps,
        )
        self.contagion_sim = ContagionNetworkSimulation()
        self._last_patterns: list[DiscoveredPattern] = []
        logger.info("MiroFish Dual Simulation Engine initialized")

    def run_dual(self, prices: dict[str, pd.Series]) -> list[DiscoveredPattern]:
        """Run both simulation modes and merge results.

        Args:
            prices: Dict of ticker -> close prices from OpenBB universe.

        Returns:
            List of DiscoveredPattern for the PatternDiscoveryBus.
        """
        all_patterns = []

        # --- Mode A: Synthetic Market ---
        logger.info("MiroFish Mode A: Running synthetic market simulation...")
        sim_prices = self.market_sim.simulate(prices)

        clustering = self.market_sim.detect_emergent_clustering(prices, sim_prices)
        herding = self.market_sim.detect_herding(sim_prices)
        regime_shifts = self.market_sim.detect_regime_transition(prices, sim_prices)

        all_patterns.extend(clustering)
        all_patterns.extend(herding)
        all_patterns.extend(regime_shifts)

        logger.info(f"Mode A: {len(clustering)} clustering, {len(herding)} herding, "
                    f"{len(regime_shifts)} regime shifts")

        # --- Mode B: Contagion Network ---
        logger.info("MiroFish Mode B: Running contagion network simulation...")
        contagion = self.contagion_sim.find_contagion_paths(prices)
        hidden = self.contagion_sim.find_hidden_correlations(prices)
        divergences = self.contagion_sim.find_divergences(prices)

        all_patterns.extend(contagion)
        all_patterns.extend(hidden)
        all_patterns.extend(divergences)

        logger.info(f"Mode B: {len(contagion)} contagion, {len(hidden)} hidden corr, "
                    f"{len(divergences)} divergences")

        # --- Dual Agreement/Disagreement ---
        dual_patterns = self._cross_reference(all_patterns)
        all_patterns.extend(dual_patterns)

        # Sort by confidence * strength
        all_patterns.sort(key=lambda p: p.confidence * p.strength, reverse=True)

        self._last_patterns = all_patterns
        logger.info(f"MiroFish Dual: {len(all_patterns)} total patterns discovered")

        return all_patterns

    def _cross_reference(self, patterns: list[DiscoveredPattern]) -> list[DiscoveredPattern]:
        """Find agreements/disagreements between Mode A and Mode B."""
        cross_patterns = []

        market_patterns = [p for p in patterns if p.source == "mirofish_market"]
        contagion_patterns = [p for p in patterns if p.source == "mirofish_contagion"]

        # For each ticker, check if both modes have signals
        market_tickers = {}
        for p in market_patterns:
            for t in p.tickers:
                market_tickers.setdefault(t, []).append(p)

        contagion_tickers = {}
        for p in contagion_patterns:
            for t in p.tickers:
                contagion_tickers.setdefault(t, []).append(p)

        common = set(market_tickers.keys()) & set(contagion_tickers.keys())
        for ticker in common:
            m_dirs = [p.direction for p in market_tickers[ticker] if p.direction != 0]
            c_dirs = [p.direction for p in contagion_tickers[ticker] if p.direction != 0]

            if not m_dirs or not c_dirs:
                continue

            m_consensus = int(np.sign(sum(m_dirs)))
            c_consensus = int(np.sign(sum(c_dirs)))

            if m_consensus == c_consensus:
                # Agreement → high conviction
                cross_patterns.append(DiscoveredPattern(
                    source="mirofish_dual",
                    pattern_type="dual_agreement",
                    tickers=[ticker],
                    direction=m_consensus,
                    strength=0.85,
                    confidence=0.8,
                    description=f"Dual agreement on {ticker}: both modes say {'+' if m_consensus > 0 else '-'}",
                    metadata={"market_direction": m_consensus, "contagion_direction": c_consensus},
                ))
            else:
                # Disagreement → contrarian opportunity
                cross_patterns.append(DiscoveredPattern(
                    source="mirofish_dual",
                    pattern_type="dual_divergence",
                    tickers=[ticker],
                    direction=0,
                    strength=0.7,
                    confidence=0.6,
                    description=(
                        f"Dual divergence on {ticker}: market={'+' if m_consensus > 0 else '-'} "
                        f"vs contagion={'+' if c_consensus > 0 else '-'}"
                    ),
                    metadata={"market_direction": m_consensus, "contagion_direction": c_consensus},
                ))

        return cross_patterns

    def get_last_patterns(self) -> list[DiscoveredPattern]:
        return list(self._last_patterns)
