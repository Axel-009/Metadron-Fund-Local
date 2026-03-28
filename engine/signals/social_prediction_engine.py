"""
SocialPredictionEngine — Agent-based market simulation signal source.

Replaces the previous MiroFish social media simulation with agent-based
market microstructure simulation. Uses Kyle's Lambda, Heterogeneous Agent
Models (HAM), and order book dynamics to generate trading signals.

Pipeline position: L2 → feeds AlphaOptimizer, MLVoteEnsemble (Tier-6)

Data flow:
    Market data (OpenBB) → Calibrate agent population → Simulate price paths
        → Extract signal (direction, confidence, regime) → SocialSnapshot

No social media. Pure market microstructure simulation.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)

# Import agent sim engine (the actual simulation)
try:
    from .agent_sim_engine import AgentSimEngine, AgentSimSignal
except ImportError:
    AgentSimEngine = None
    AgentSimSignal = None
    logger.warning("AgentSimEngine unavailable — SocialPredictionEngine disabled")


@dataclass
class SocialSnapshot:
    """
    Snapshot output consumed by MLVoteEnsemble (Tier-6) and AlphaOptimizer.
    
    Despite the name (kept for backward compat), this now represents
    agent-based MARKET SIMULATION signals, not social media sentiment.
    """
    ticker: str = ""
    timestamp: str = ""
    sentiment_score: float = 0.0      # [-1, +1] — agent consensus
    engagement: float = 0.0           # [0, 1] — simulation confidence
    momentum: float = 0.0             # Predicted return
    consensus_strength: float = 0.0   # Agent agreement level
    regime: str = "random_walk"       # trending/mean_reverting/random_walk
    hurst_exponent: float = 0.5
    social_signal: str = "HOLD"       # BUY/SELL/HOLD (mapped from direction)
    n_simulations: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "sentiment_score": round(self.sentiment_score, 4),
            "engagement": round(self.engagement, 4),
            "momentum": round(self.momentum, 6),
            "consensus_strength": round(self.consensus_strength, 4),
            "regime": self.regime,
            "hurst_exponent": round(self.hurst_exponent, 4),
            "social_signal": self.social_signal,
            "n_simulations": self.n_simulations,
        }


class SocialPredictionEngine:
    """
    Agent-based market simulation engine.
    
    Runs Monte Carlo agent simulations per security to produce
    directional signals, regime classification, and confidence scores.
    
    Uses MiroFish's MarketSimulator (Kyle's Lambda, HAM, order book)
    calibrated to real market data from OpenBB.
    
    No social media. Pure market microstructure.
    """
    
    def __init__(self, n_simulations: int = 100, simulation_horizon: int = 20):
        if AgentSimEngine is None:
            raise ImportError("AgentSimEngine required for SocialPredictionEngine")
        
        self._engine = AgentSimEngine(
            n_simulations=n_simulations,
            simulation_horizon=simulation_horizon,
        )
        logger.info("SocialPredictionEngine initialized (agent sim, %d paths)", n_simulations)
    
    def analyze(self, tickers: Optional[List[str]] = None) -> Dict[str, SocialSnapshot]:
        """
        Run agent simulation for tickers and return snapshots.
        
        Args:
            tickers: List of tickers to analyze. If None, returns empty.
        
        Returns:
            Dict mapping ticker → SocialSnapshot
        """
        if not tickers:
            return {}
        
        results = {}
        for ticker in tickers:
            try:
                signal = self._engine.simulate_ticker(ticker)
                snap = SocialSnapshot(
                    ticker=signal.ticker,
                    timestamp=signal.timestamp.isoformat(),
                    sentiment_score=signal.agent_consensus,
                    engagement=signal.confidence,
                    momentum=signal.predicted_return,
                    consensus_strength=signal.confidence,
                    regime=signal.regime,
                    hurst_exponent=signal.hurst_exponent,
                    social_signal=signal.direction,
                    n_simulations=signal.simulated_paths,
                )
                results[ticker] = snap
            except Exception as e:
                logger.debug("Agent sim failed for %s: %s", ticker, e)
                results[ticker] = SocialSnapshot(ticker=ticker, social_signal="HOLD")
        
        return results
    
    def get_signal(self, ticker: str) -> SocialSnapshot:
        """Get signal for a single ticker."""
        results = self.analyze([ticker])
        return results.get(ticker, SocialSnapshot(ticker=ticker))
    
    def get_all_signals(self) -> Dict[str, SocialSnapshot]:
        """Get signals for all tickers (called by pipeline)."""
        # This is called without tickers — return empty
        # The pipeline calls analyze(tickers=...) directly
        return {}
