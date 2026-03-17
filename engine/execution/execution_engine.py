"""ExecutionEngine — Full signal pipeline orchestrator.

Signal pipeline (execution order):
    UniverseEngine → MacroEngine → MetadronCube → AlphaOptimizer → ExecutionEngine

ML Vote Ensemble (10 tiers, each votes ±1):
    Tier-1   Pure-numpy 2-layer net
    Tier-2   Momentum/mean-reversion voter
    Tier-3   Volatility regime voter
    Tier-4   Monte Carlo voter (ARIMA-like + noise)
    Tier-5   Quality tier voter (top-down + bottom-up)
    Tier-6   Social sentiment voter (MiroFish prediction engine)
    Tier-7   Distressed asset voter
    Tier-8   Event-driven voter
    Tier-9   CVR voter
    Tier-10  Credit quality voter (UniverseClassifier)

effective_min_edge = 2.0 + max(0, -vote_score) bps

Deep Trading Features:
    - Micro-price estimation (bid/ask midpoint proxy)
    - Order flow imbalance detection
    - Cross-asset correlation signals
    - Intraday momentum decomposition

Risk Gate Manager:
    - Pre-trade risk validation (8 gates)
    - Position-level limits
    - Portfolio-level limits
    - Drawdown circuit breakers
"""

import logging
import numpy as np
import pandas as pd
from collections import deque
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

# Social prediction engine (MiroFish integration)
try:
    from ..signals.social_prediction_engine import SocialPredictionEngine, SocialSnapshot
    from ..ml.social_features import SocialFeatureBuilder
    _HAS_SOCIAL = True
except ImportError:
    _HAS_SOCIAL = False

# Distressed asset engine
try:
    from ..signals.distressed_asset_engine import DistressedAssetEngine
    _HAS_DISTRESS = True
except ImportError:
    _HAS_DISTRESS = False

# CVR engine
try:
    from ..signals.cvr_engine import CVREngine
    _HAS_CVR = True
except ImportError:
    _HAS_CVR = False

# Event-driven engine
try:
    from ..signals.event_driven_engine import EventDrivenEngine
    _HAS_EVENT = True
except ImportError:
    _HAS_EVENT = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Deep Trading Features
# ---------------------------------------------------------------------------
@dataclass
class MicroPriceEstimate:
    """Micro-price estimation from recent price action."""
    ticker: str = ""
    mid_price: float = 0.0
    micro_price: float = 0.0
    bid_proxy: float = 0.0
    ask_proxy: float = 0.0
    spread_bps: float = 0.0
    imbalance: float = 0.0        # [-1, +1] order flow imbalance
    urgency_score: float = 0.0    # [0, 1] how urgent to trade


class MicroPriceEngine:
    """Estimate micro-prices from daily OHLCV data (OpenBB).

    In paper broker mode, we estimate bid/ask from high/low range
    and compute order flow imbalance from close position within range.
    """

    def __init__(self, spread_multiplier: float = 1.5):
        self.spread_multiplier = spread_multiplier
        self._cache: dict[str, MicroPriceEstimate] = {}

    def estimate(self, ticker: str, prices: pd.DataFrame) -> MicroPriceEstimate:
        """Estimate micro-price from OHLCV data."""
        est = MicroPriceEstimate(ticker=ticker)

        if prices is None or prices.empty or len(prices) < 5:
            return est

        try:
            # Get latest bar
            last = prices.iloc[-1]
            prev = prices.iloc[-2]

            # Use column names or positional
            close = float(last.get("Close", last.get("Adj Close", last.iloc[-1])))
            high = float(last.get("High", close * 1.005))
            low = float(last.get("Low", close * 0.995))
            volume = float(last.get("Volume", 1e6))

            prev_close = float(prev.get("Close", prev.get("Adj Close", prev.iloc[-1])))
            prev_volume = float(prev.get("Volume", 1e6))

            est.mid_price = (high + low) / 2

            # Spread estimation from daily range
            daily_range = high - low
            if est.mid_price > 0:
                est.spread_bps = (daily_range / est.mid_price) * 10000 * 0.1  # ~10% of range

            # Bid/ask proxy
            half_spread = daily_range * 0.05 * self.spread_multiplier
            est.bid_proxy = est.mid_price - half_spread
            est.ask_proxy = est.mid_price + half_spread

            # Micro-price: weighted by close position in range
            if daily_range > 0:
                close_position = (close - low) / daily_range
                est.micro_price = est.bid_proxy + close_position * (est.ask_proxy - est.bid_proxy)
            else:
                est.micro_price = close

            # Order flow imbalance: close relative to open + volume change
            open_price = float(last.get("Open", prev_close))
            if daily_range > 0:
                price_imbalance = (close - open_price) / daily_range
            else:
                price_imbalance = 0.0

            volume_imbalance = 0.0
            if prev_volume > 0:
                vol_ratio = volume / prev_volume
                volume_imbalance = np.clip((vol_ratio - 1.0) / 2.0, -1, 1)

            # Combined imbalance
            est.imbalance = float(np.clip(
                0.6 * price_imbalance + 0.4 * volume_imbalance * np.sign(price_imbalance),
                -1, 1,
            ))

            # Urgency: higher when price is near extreme of range with volume
            if daily_range > 0:
                extremity = abs(close_position - 0.5) * 2  # 0 at midpoint, 1 at edges
                vol_activity = min(volume / max(prev_volume, 1), 3.0) / 3.0
                est.urgency_score = float(np.clip(extremity * 0.5 + vol_activity * 0.5, 0, 1))

            self._cache[ticker] = est

        except Exception as e:
            logger.debug(f"MicroPrice estimate failed for {ticker}: {e}")

        return est


# ---------------------------------------------------------------------------
# Cross-Asset Correlation Monitor
# ---------------------------------------------------------------------------
class CrossAssetMonitor:
    """Track cross-asset correlations for signal confirmation."""

    BENCHMARKS = {
        "equity": "SPY",
        "bonds": "TLT",
        "gold": "GLD",
        "oil": "USO",
        "dollar": "UUP",
        "volatility": "VXX",
    }

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self._corr_matrix: Optional[pd.DataFrame] = None
        self._regime_correlations: dict[str, float] = {}

    def compute_correlations(self) -> dict:
        """Compute correlation matrix for cross-asset benchmarks."""
        try:
            start = (pd.Timestamp.now() - pd.Timedelta(days=self.lookback + 30)).strftime("%Y-%m-%d")
            tickers = list(self.BENCHMARKS.values())
            prices = get_adj_close(tickers, start=start)

            if prices.empty or len(prices) < 20:
                return {}

            returns = prices.pct_change().dropna()
            self._corr_matrix = returns.corr()

            # Key relationships
            result = {}
            if "SPY" in returns.columns and "TLT" in returns.columns:
                result["stock_bond_corr"] = float(returns["SPY"].corr(returns["TLT"]))
            if "SPY" in returns.columns and "GLD" in returns.columns:
                result["stock_gold_corr"] = float(returns["SPY"].corr(returns["GLD"]))
            if "SPY" in returns.columns and "UUP" in returns.columns:
                result["stock_dollar_corr"] = float(returns["SPY"].corr(returns["UUP"]))

            # Risk-on/risk-off signal
            risk_on_score = 0.0
            if result.get("stock_bond_corr", 0) < -0.2:
                risk_on_score += 0.3  # Normal inverse relationship = healthy
            if result.get("stock_gold_corr", 0) < 0:
                risk_on_score += 0.3  # Gold not rallying = risk-on
            if result.get("stock_dollar_corr", 0) > -0.3:
                risk_on_score += 0.4

            result["risk_on_score"] = float(np.clip(risk_on_score, 0, 1))
            self._regime_correlations = result
            return result

        except Exception as e:
            logger.debug(f"Cross-asset correlation failed: {e}")
            return {}


# ---------------------------------------------------------------------------
# Risk Gate Manager
# ---------------------------------------------------------------------------
@dataclass
class RiskGateResult:
    """Result of risk gate evaluation."""
    passed: bool = True
    gate_name: str = ""
    reason: str = ""
    limit_value: float = 0.0
    current_value: float = 0.0


class RiskGateManager:
    """Pre-trade risk validation with 8 independent gates.

    Each gate must pass for a trade to execute.
    Gates can be soft (warning) or hard (block).
    """

    def __init__(
        self,
        max_position_pct: float = 0.10,       # 10% max single position
        max_sector_pct: float = 0.30,          # 30% max sector concentration
        max_daily_loss_pct: float = 0.03,      # 3% max daily loss
        max_gross_exposure: float = 2.5,       # 250% max gross exposure
        max_net_exposure: float = 1.5,         # 150% max net exposure
        max_trade_count_daily: int = 100,      # Max trades per day
        min_liquidity_adv: float = 100_000,    # Min average daily volume (dollars)
        max_drawdown_pct: float = 0.10,        # 10% max drawdown before halt
    ):
        self.max_position_pct = max_position_pct
        self.max_sector_pct = max_sector_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_gross_exposure = max_gross_exposure
        self.max_net_exposure = max_net_exposure
        self.max_trade_count_daily = max_trade_count_daily
        self.min_liquidity_adv = min_liquidity_adv
        self.max_drawdown_pct = max_drawdown_pct
        self._daily_trade_count = 0
        self._daily_pnl = 0.0
        self._peak_nav = 0.0
        self._trade_date = ""

    def evaluate_all(
        self,
        ticker: str,
        side: OrderSide,
        quantity: int,
        price: float,
        broker: PaperBroker,
    ) -> list[RiskGateResult]:
        """Run all 8 risk gates. Returns list of results."""
        results = []
        nav = broker.compute_nav()
        if nav <= 0:
            nav = 1_000_000  # Fallback

        # Track peak NAV for drawdown
        if nav > self._peak_nav:
            self._peak_nav = nav

        # Reset daily counter if new day
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._trade_date:
            self._daily_trade_count = 0
            self._daily_pnl = 0.0
            self._trade_date = today

        trade_value = quantity * price

        # Gate 1: Position size limit
        position_pct = trade_value / nav
        results.append(RiskGateResult(
            passed=position_pct <= self.max_position_pct,
            gate_name="G1_POSITION_SIZE",
            reason=f"Position {position_pct:.1%} vs limit {self.max_position_pct:.1%}",
            limit_value=self.max_position_pct,
            current_value=position_pct,
        ))

        # Gate 2: Sector concentration
        sector_exposure = self._compute_sector_exposure(ticker, trade_value, broker)
        results.append(RiskGateResult(
            passed=sector_exposure <= self.max_sector_pct,
            gate_name="G2_SECTOR_CONCENTRATION",
            reason=f"Sector {sector_exposure:.1%} vs limit {self.max_sector_pct:.1%}",
            limit_value=self.max_sector_pct,
            current_value=sector_exposure,
        ))

        # Gate 3: Daily loss limit
        daily_loss_pct = abs(min(self._daily_pnl, 0)) / nav
        results.append(RiskGateResult(
            passed=daily_loss_pct <= self.max_daily_loss_pct,
            gate_name="G3_DAILY_LOSS",
            reason=f"Daily loss {daily_loss_pct:.1%} vs limit {self.max_daily_loss_pct:.1%}",
            limit_value=self.max_daily_loss_pct,
            current_value=daily_loss_pct,
        ))

        # Gate 4: Gross exposure
        exposures = broker.compute_exposures()
        gross = exposures.get("gross", 0)
        results.append(RiskGateResult(
            passed=gross <= self.max_gross_exposure,
            gate_name="G4_GROSS_EXPOSURE",
            reason=f"Gross {gross:.1%} vs limit {self.max_gross_exposure:.1%}",
            limit_value=self.max_gross_exposure,
            current_value=gross,
        ))

        # Gate 5: Net exposure
        net = exposures.get("net", 0)
        results.append(RiskGateResult(
            passed=abs(net) <= self.max_net_exposure,
            gate_name="G5_NET_EXPOSURE",
            reason=f"Net {net:.1%} vs limit {self.max_net_exposure:.1%}",
            limit_value=self.max_net_exposure,
            current_value=abs(net),
        ))

        # Gate 6: Trade count throttle
        results.append(RiskGateResult(
            passed=self._daily_trade_count < self.max_trade_count_daily,
            gate_name="G6_TRADE_COUNT",
            reason=f"Trades {self._daily_trade_count} vs limit {self.max_trade_count_daily}",
            limit_value=float(self.max_trade_count_daily),
            current_value=float(self._daily_trade_count),
        ))

        # Gate 7: Drawdown circuit breaker
        drawdown = 1 - (nav / self._peak_nav) if self._peak_nav > 0 else 0
        results.append(RiskGateResult(
            passed=drawdown <= self.max_drawdown_pct,
            gate_name="G7_DRAWDOWN",
            reason=f"Drawdown {drawdown:.1%} vs limit {self.max_drawdown_pct:.1%}",
            limit_value=self.max_drawdown_pct,
            current_value=drawdown,
        ))

        # Gate 8: Cash sufficiency (for buys)
        if side in (OrderSide.BUY, OrderSide.COVER):
            cash_sufficient = broker.state.cash >= trade_value * 0.95
            results.append(RiskGateResult(
                passed=cash_sufficient,
                gate_name="G8_CASH_SUFFICIENCY",
                reason=f"Cash ${broker.state.cash:,.0f} vs trade ${trade_value:,.0f}",
                limit_value=trade_value,
                current_value=broker.state.cash,
            ))
        else:
            results.append(RiskGateResult(
                passed=True,
                gate_name="G8_CASH_SUFFICIENCY",
                reason="Sell order — no cash check needed",
            ))

        return results

    def all_passed(self, results: list[RiskGateResult]) -> bool:
        return all(r.passed for r in results)

    def record_trade(self, pnl: float = 0.0):
        """Record a completed trade for daily tracking."""
        self._daily_trade_count += 1
        self._daily_pnl += pnl

    def _compute_sector_exposure(
        self, ticker: str, trade_value: float, broker: PaperBroker,
    ) -> float:
        """Compute sector exposure including proposed trade."""
        nav = broker.state.nav
        if nav <= 0:
            return 0.0
        # Simplified: treat each position as its own sector for now
        existing = sum(
            abs(p.market_value) for p in broker.state.positions.values()
            if p.sector == ticker  # Will match sector-level ETFs
        )
        return (existing + trade_value) / nav

    def get_summary(self) -> dict:
        """Risk gate status summary."""
        return {
            "daily_trades": self._daily_trade_count,
            "daily_pnl": self._daily_pnl,
            "peak_nav": self._peak_nav,
            "max_position_pct": self.max_position_pct,
            "max_sector_pct": self.max_sector_pct,
            "max_daily_loss_pct": self.max_daily_loss_pct,
        }


# ---------------------------------------------------------------------------
# Deep Trading Feature Builder
# ---------------------------------------------------------------------------
class DeepTradingFeatures:
    """Build deep trading features from price data.

    Combines micro-price, cross-asset, and momentum features
    into a unified feature vector for ML voting.
    """

    def __init__(self):
        self._micro_engine = MicroPriceEngine()
        self._cross_asset = CrossAssetMonitor()
        self._feature_history: dict[str, deque] = {}

    def build_features(self, ticker: str, returns: pd.Series) -> dict:
        """Build comprehensive feature set for a single ticker."""
        features = {"ticker": ticker}

        if returns is None or returns.empty or len(returns) < 21:
            return features

        r = returns.values

        # Momentum features (multiple horizons)
        features["mom_5d"] = float(r[-5:].sum()) if len(r) >= 5 else 0
        features["mom_10d"] = float(r[-10:].sum()) if len(r) >= 10 else 0
        features["mom_21d"] = float(r[-21:].sum()) if len(r) >= 21 else 0
        features["mom_63d"] = float(r[-63:].sum()) if len(r) >= 63 else 0
        features["mom_126d"] = float(r[-126:].sum()) if len(r) >= 126 else 0
        features["mom_252d"] = float(r[-252:].sum()) if len(r) >= 252 else 0

        # Momentum acceleration (rate of change of momentum)
        if len(r) >= 42:
            mom_recent = r[-21:].sum()
            mom_prev = r[-42:-21].sum()
            features["mom_acceleration"] = mom_recent - mom_prev
        else:
            features["mom_acceleration"] = 0

        # Volatility features
        features["vol_5d"] = float(r[-5:].std() * np.sqrt(252)) if len(r) >= 5 else 0
        features["vol_21d"] = float(r[-21:].std() * np.sqrt(252)) if len(r) >= 21 else 0
        features["vol_63d"] = float(r[-63:].std() * np.sqrt(252)) if len(r) >= 63 else 0

        # Vol regime (recent vs long-term)
        if len(r) >= 63:
            vol_ratio = r[-21:].std() / max(r[-63:].std(), 1e-8)
            features["vol_regime"] = float(vol_ratio)
        else:
            features["vol_regime"] = 1.0

        # Skewness and kurtosis
        if len(r) >= 21:
            features["skew_21d"] = float(pd.Series(r[-21:]).skew())
            features["kurt_21d"] = float(pd.Series(r[-21:]).kurtosis())
        else:
            features["skew_21d"] = 0
            features["kurt_21d"] = 0

        # Mean reversion signal
        if len(r) >= 63:
            z_score = (r[-1] - r[-63:].mean()) / max(r[-63:].std(), 1e-8)
            features["zscore_63d"] = float(np.clip(z_score, -4, 4))
        else:
            features["zscore_63d"] = 0

        # Drawdown from peak
        if len(r) >= 21:
            cum = np.cumsum(r[-63:]) if len(r) >= 63 else np.cumsum(r[-21:])
            peak = np.maximum.accumulate(cum)
            dd = cum - peak
            features["max_drawdown"] = float(dd.min())
            features["current_drawdown"] = float(dd[-1])
        else:
            features["max_drawdown"] = 0
            features["current_drawdown"] = 0

        # Autocorrelation (serial correlation)
        if len(r) >= 22:
            features["autocorr_1d"] = float(np.corrcoef(r[-22:-1], r[-21:])[0, 1])
        else:
            features["autocorr_1d"] = 0

        # RSI proxy (14-period)
        if len(r) >= 14:
            gains = r[-14:].copy()
            losses = r[-14:].copy()
            gains[gains < 0] = 0
            losses[losses > 0] = 0
            avg_gain = gains.mean()
            avg_loss = abs(losses.mean())
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features["rsi_14"] = 100 - (100 / (1 + rs))
            else:
                features["rsi_14"] = 100
        else:
            features["rsi_14"] = 50

        # MACD proxy (12/26)
        if len(r) >= 26:
            ema_12 = float(pd.Series(r).ewm(span=12).mean().iloc[-1])
            ema_26 = float(pd.Series(r).ewm(span=26).mean().iloc[-1])
            features["macd_signal"] = ema_12 - ema_26
        else:
            features["macd_signal"] = 0

        # Store in history
        if ticker not in self._feature_history:
            self._feature_history[ticker] = deque(maxlen=252)
        self._feature_history[ticker].append(features)

        return features


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
    confidence: float = 0.0
    features: dict = field(default_factory=dict)


class MLVoteEnsemble:
    """5-tier vote ensemble. Each tier votes ±1. Pure numpy.

    Enhanced with deep trading features and adaptive weighting.
    """

    # Tier weights (can be adjusted based on historical performance)
    TIER_WEIGHTS = {
        "T1_neural": 1.0,
        "T2_momentum": 1.2,
        "T3_vol_regime": 0.8,
        "T4_monte_carlo": 0.9,
        "T5_quality": 1.1,
        "T6_social": 1.0,
        "T7_distress": 0.9,
        "T8_event": 1.0,
        "T9_cvr": 0.7,
        "T10_credit_quality": 0.9,
    }

    def __init__(self):
        self._feature_builder = DeepTradingFeatures()
        self._vote_history: dict[str, list] = {}
        self._social_snapshot: Optional[dict] = None
        self._social_feature_builder = SocialFeatureBuilder() if _HAS_SOCIAL else None
        self._distress_signals: Optional[dict] = None
        self._event_signals: Optional[dict] = None
        self._cvr_signals: Optional[dict] = None
        self._credit_scores: Optional[dict] = None

    def set_social_snapshot(self, snapshot_dict: dict) -> None:
        """Inject social prediction snapshot for Tier-6 voting."""
        self._social_snapshot = snapshot_dict
        if self._social_feature_builder:
            self._social_feature_builder.add_snapshot(snapshot_dict)

    def set_distress_signals(self, signals: dict) -> None:
        """Inject distressed asset signals for Tier-7 voting."""
        self._distress_signals = signals

    def set_event_signals(self, signals: dict) -> None:
        """Inject event-driven signals for Tier-8 voting."""
        self._event_signals = signals

    def set_cvr_signals(self, signals: dict) -> None:
        """Inject CVR signals for Tier-9 voting."""
        self._cvr_signals = signals

    def set_credit_scores(self, scores: dict) -> None:
        """Inject credit quality scores for Tier-10 voting."""
        self._credit_scores = scores

    def vote(self, ticker: str, returns: pd.Series, alpha_signal: Optional[AlphaSignal] = None) -> VoteResult:
        result = VoteResult(ticker=ticker)
        if returns.empty or len(returns) < 21:
            return result

        r = returns.values

        # Build deep features
        result.features = self._feature_builder.build_features(ticker, returns)

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

        # Tier 6: Social sentiment (MiroFish)
        result.votes["T6_social"] = self._tier6_social(ticker)

        # Tier 7: Distressed asset signal
        result.votes["T7_distress"] = self._tier7_distress(ticker)

        # Tier 8: Event-driven signal
        result.votes["T8_event"] = self._tier8_event(ticker)

        # Tier 9: CVR signal
        result.votes["T9_cvr"] = self._tier9_cvr(ticker)

        # Tier 10: Credit quality
        result.votes["T10_credit_quality"] = self._tier10_credit_quality(ticker)

        # Weighted aggregate
        weighted_score = sum(
            vote * self.TIER_WEIGHTS.get(tier, 1.0)
            for tier, vote in result.votes.items()
        )
        total_weight = sum(self.TIER_WEIGHTS.get(t, 1.0) for t in result.votes)
        n_tiers = len(result.votes)
        result.score = weighted_score / max(total_weight / n_tiers, 1)  # Normalize to [-N, +N] range

        result.edge_bps = 2.0 + max(0, -result.score)

        # Confidence based on vote agreement
        non_zero_votes = [v for v in result.votes.values() if v != 0]
        if non_zero_votes:
            agreement = abs(sum(non_zero_votes)) / len(non_zero_votes)
            result.confidence = agreement
        else:
            result.confidence = 0.0

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

        # Track vote history
        if ticker not in self._vote_history:
            self._vote_history[ticker] = []
        self._vote_history[ticker].append({
            "timestamp": datetime.now().isoformat(),
            "score": result.score,
            "signal": result.signal.value,
        })

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

    def _tier6_social(self, ticker: str) -> int:
        """Social sentiment voter (MiroFish prediction engine).

        Uses social simulation data to vote on ticker direction.
        Checks both ticker-specific sentiment and overall market sentiment.
        """
        if not _HAS_SOCIAL or self._social_snapshot is None:
            return 0

        # Get ticker-specific signal
        ticker_signals = self._social_snapshot.get("ticker_signals", {})
        ticker_sentiment = ticker_signals.get(ticker, 0.0)

        # If we have a strong ticker-specific signal, use it
        if abs(ticker_sentiment) > 0.3:
            return 1 if ticker_sentiment > 0 else -1

        # Fall back to aggregate social vote
        vote_score = self._social_snapshot.get("vote_score", 0)
        signal_strength = self._social_snapshot.get("signal_strength", 0.0)

        # Only vote if signal is strong enough
        if signal_strength > 0.4:
            return vote_score

        return 0

    def _tier7_distress(self, ticker: str) -> int:
        """Distressed asset voter.

        Uses DistressedAssetEngine signals to vote:
        - Fallen angel with positive Kelly → +1 (contrarian buy)
        - Critical distress → -1 (avoid)
        - Otherwise neutral
        """
        if not self._distress_signals:
            return 0
        sig = self._distress_signals.get(ticker, {})
        if not sig:
            return 0
        if sig.get("is_fallen_angel") and sig.get("kelly_fraction", 0) > 0.02:
            return 1  # Fallen angel opportunity
        level = sig.get("level", "SAFE")
        if level in ("CRITICAL", "DISTRESSED"):
            return -1  # Avoid distressed names
        return 0

    def _tier8_event(self, ticker: str) -> int:
        """Event-driven voter.

        Uses EventDrivenEngine signals:
        - LONG with >50bps alpha → +1
        - SHORT with < -50bps alpha → -1
        - Otherwise neutral
        """
        if not self._event_signals:
            return 0
        sig = self._event_signals.get(ticker, {})
        if not sig:
            return 0
        alpha = sig.get("expected_alpha_bps", 0)
        signal = sig.get("signal", "HOLD")
        if signal == "LONG" and alpha > 50:
            return 1
        elif signal == "SHORT" and alpha < -50:
            return -1
        return 0

    def _tier9_cvr(self, ticker: str) -> int:
        """CVR voter.

        Uses CVREngine signals:
        - STRONG_BUY or BUY → +1
        - SELL or AVOID → -1
        """
        if not self._cvr_signals:
            return 0
        sig = self._cvr_signals.get(ticker, {})
        if not sig:
            return 0
        signal = sig.get("signal", "HOLD")
        if signal in ("STRONG_BUY", "BUY"):
            return 1
        elif signal in ("SELL", "AVOID"):
            return -1
        return 0

    def _tier10_credit_quality(self, ticker: str) -> int:
        """Credit quality voter.

        Uses CreditQualityClassifier scores:
        - Strong credit (score > 0.7) → +1 (bullish)
        - Weak credit (score < 0.3) → -1 (bearish)
        - Otherwise neutral
        """
        if not self._credit_scores:
            return 0
        score_data = self._credit_scores.get(ticker, {})
        if not score_data:
            return 0
        credit_score = score_data.get("credit_quality_score", 0.5)
        if credit_score > 0.7:
            return 1  # Strong credit quality
        elif credit_score < 0.3:
            return -1  # Weak credit quality
        return 0

    def get_vote_history(self, ticker: str) -> list:
        return self._vote_history.get(ticker, [])


# ---------------------------------------------------------------------------
# Trade Allocation Engine
# ---------------------------------------------------------------------------
class TradeAllocator:
    """Convert signals and weights into concrete trade sizes.

    Respects sleeve allocations from Gate-Z and position limits.
    """

    def __init__(self, min_trade_value: float = 1000.0):
        self.min_trade_value = min_trade_value

    def allocate(
        self,
        ticker: str,
        weight: float,
        vote: VoteResult,
        nav: float,
        equity_budget: float,
        current_position: Optional[Position] = None,
        price: float = 0.0,
    ) -> dict:
        """Compute trade allocation for a single ticker.

        Returns dict with side, quantity, target_value, etc.
        """
        result = {
            "ticker": ticker,
            "side": None,
            "quantity": 0,
            "target_value": 0.0,
            "price": price,
            "signal": vote.signal,
            "vote_score": vote.score,
            "weight": weight,
        }

        if price <= 0 or weight < 0.005:
            return result

        if vote.signal in (SignalType.ML_AGENT_BUY, SignalType.QUALITY_BUY):
            # Target value based on weight and confidence
            confidence_adj = 0.5 + 0.5 * vote.confidence
            target_value = equity_budget * weight * confidence_adj

            # Account for existing position
            current_value = 0
            if current_position and current_position.quantity > 0:
                current_value = current_position.quantity * price

            incremental_value = target_value - current_value
            if incremental_value > self.min_trade_value:
                qty = max(1, int(incremental_value / price))
                result["side"] = OrderSide.BUY
                result["quantity"] = qty
                result["target_value"] = target_value

        elif vote.signal in (SignalType.ML_AGENT_SELL, SignalType.QUALITY_SELL):
            if current_position and current_position.quantity > 0:
                # Sell based on conviction
                sell_pct = 0.5 if vote.score >= -3 else 1.0
                qty = max(1, int(current_position.quantity * sell_pct))
                result["side"] = OrderSide.SELL
                result["quantity"] = qty
                result["target_value"] = qty * price

        return result


# ---------------------------------------------------------------------------
# Pipeline Stage Tracker
# ---------------------------------------------------------------------------
class PipelineTracker:
    """Track execution timing and results for each pipeline stage."""

    def __init__(self):
        self._stages: list[dict] = []
        self._start_time: Optional[datetime] = None

    def start_pipeline(self):
        self._stages = []
        self._start_time = datetime.now()

    def record_stage(self, name: str, duration_ms: float, result: dict):
        self._stages.append({
            "name": name,
            "duration_ms": duration_ms,
            "result_keys": list(result.keys()),
            "timestamp": datetime.now().isoformat(),
        })

    def get_summary(self) -> dict:
        total_ms = sum(s["duration_ms"] for s in self._stages)
        return {
            "total_duration_ms": total_ms,
            "stages": self._stages,
            "start_time": self._start_time.isoformat() if self._start_time else "",
        }


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------
class ExecutionEngine:
    """Full pipeline orchestrator: Universe → Macro → Cube → Alpha → Execute.

    Runs the complete Metadron Capital investment engine with:
    - Deep trading features (micro-price, cross-asset)
    - 8-gate risk management
    - Smart trade allocation
    - Pipeline performance tracking
    """

    def __init__(
        self,
        initial_nav: float = 1_000_000.0,
        top_n_per_sector: int = 5,
        enable_risk_gates: bool = True,
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

        # Enhanced components
        self.micro_price = MicroPriceEngine()
        self.cross_asset = CrossAssetMonitor()
        self.risk_gates = RiskGateManager() if enable_risk_gates else None
        self.allocator = TradeAllocator()
        self.tracker = PipelineTracker()
        self._run_count = 0
        self._trade_log: list[dict] = []

        # Social prediction engine (MiroFish bridge)
        self.social: Optional[SocialPredictionEngine] = None
        if _HAS_SOCIAL:
            try:
                self.social = SocialPredictionEngine()
            except Exception as e:
                logger.warning(f"SocialPredictionEngine init failed: {e}")

        # Distressed asset engine
        self.distress = None
        if _HAS_DISTRESS:
            try:
                self.distress = DistressedAssetEngine()
            except Exception as e:
                logger.warning(f"DistressedAssetEngine init failed: {e}")

        # CVR engine
        self.cvr = None
        if _HAS_CVR:
            try:
                self.cvr = CVREngine()
            except Exception as e:
                logger.warning(f"CVREngine init failed: {e}")

        # Event-driven engine
        self.event = None
        if _HAS_EVENT:
            try:
                self.event = EventDrivenEngine()
            except Exception as e:
                logger.warning(f"EventDrivenEngine init failed: {e}")

    def run_pipeline(self) -> dict:
        """Execute the full signal pipeline.

        Returns dict with all stage outputs.
        """
        self.tracker.start_pipeline()
        self._run_count += 1
        result = {
            "timestamp": datetime.now().isoformat(),
            "run_number": self._run_count,
            "stages": {},
        }

        # Stage 1: Universe
        t0 = datetime.now()
        self.universe.load()
        result["stages"]["universe"] = {
            "total_equities": self.universe.size(),
            "sectors": self.universe.get_sectors(),
        }
        self.tracker.record_stage("universe", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["universe"])

        # Stage 2: Macro analysis
        t0 = datetime.now()
        macro_snap = self.macro.analyze()
        result["stages"]["macro"] = {
            "regime": macro_snap.regime.value,
            "vix": macro_snap.vix,
            "spy_1m": macro_snap.spy_return_1m,
            "spy_3m": macro_snap.spy_return_3m,
            "sector_rankings": macro_snap.sector_rankings,
        }
        self.tracker.record_stage("macro", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["macro"])

        # Stage 3: MetadronCube
        t0 = datetime.now()
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
        self.tracker.record_stage("cube", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["cube"])

        # Stage 3.5: Cross-asset correlation check
        t0 = datetime.now()
        cross_asset_data = self.cross_asset.compute_correlations()
        result["stages"]["cross_asset"] = cross_asset_data
        self.tracker.record_stage("cross_asset", (datetime.now() - t0).total_seconds() * 1000, cross_asset_data)

        # Stage 3.7: Social prediction (MiroFish agent simulation)
        t0 = datetime.now()
        social_data = {}
        if self.social:
            try:
                social_snap = self.social.analyze()
                social_data = self.social.as_dict()
                # Feed social snapshot to ML ensemble for Tier-6 voting
                self.ensemble.set_social_snapshot(social_data)
                logger.info(
                    f"Social prediction: signal={social_snap.social_signal} "
                    f"sentiment={social_snap.overall_sentiment:.3f} "
                    f"confidence={social_snap.sentiment_confidence:.3f} "
                    f"agents={social_snap.total_agents}"
                )
            except Exception as e:
                social_data = {"error": str(e)}
                logger.warning(f"Social prediction stage failed: {e}")
        else:
            social_data = {"status": "mirofish_not_available"}
        result["stages"]["social_prediction"] = social_data
        self.tracker.record_stage("social_prediction", (datetime.now() - t0).total_seconds() * 1000, social_data)

        # Stage 3.8: Distressed asset analysis
        t0 = datetime.now()
        distress_data = {}
        if self.distress:
            try:
                self.distress.analyze()
                distress_signals = self.distress.get_distress_signals()
                self.ensemble.set_distress_signals(distress_signals)
                fallen_angels = self.distress.get_fallen_angels()
                critical = self.distress.get_critical()
                distress_data = {
                    "total_analyzed": len(distress_signals),
                    "fallen_angels": [s.ticker for s in fallen_angels],
                    "critical_names": [s.ticker for s in critical],
                    "opportunities": len(self.distress.get_opportunities()),
                }
                logger.info(f"Distress analysis: {len(distress_signals)} names, "
                            f"{len(fallen_angels)} fallen angels, {len(critical)} critical")
            except Exception as e:
                distress_data = {"error": str(e)}
                logger.warning(f"Distressed asset stage failed: {e}")
        else:
            distress_data = {"status": "distress_engine_not_available"}
        result["stages"]["distressed_assets"] = distress_data
        self.tracker.record_stage("distressed_assets", (datetime.now() - t0).total_seconds() * 1000, distress_data)

        # Stage 3.85: CVR analysis
        t0 = datetime.now()
        cvr_data = {}
        if self.cvr:
            try:
                self.cvr.analyze()
                cvr_signals = self.cvr.get_trading_signals()
                self.ensemble.set_cvr_signals(cvr_signals)
                buy_signals = self.cvr.get_buy_signals()
                cvr_data = {
                    "total_instruments": len(cvr_signals),
                    "buy_signals": [v.ticker for v in buy_signals],
                    "signals": {k: v["signal"] for k, v in cvr_signals.items()},
                }
                logger.info(f"CVR analysis: {len(cvr_signals)} instruments, "
                            f"{len(buy_signals)} buy signals")
            except Exception as e:
                cvr_data = {"error": str(e)}
                logger.warning(f"CVR stage failed: {e}")
        else:
            cvr_data = {"status": "cvr_engine_not_available"}
        result["stages"]["cvr"] = cvr_data
        self.tracker.record_stage("cvr", (datetime.now() - t0).total_seconds() * 1000, cvr_data)

        # Stage 3.9: Event-driven analysis
        t0 = datetime.now()
        event_data = {}
        if self.event:
            try:
                # Pass regime multiplier based on cube regime
                regime_mult = {"TRENDING": 1.0, "RANGE": 0.8, "STRESS": 0.5, "CRASH": 0.3}
                mult = regime_mult.get(cube_out.regime.value, 0.8)
                event_result = self.event.analyze(regime_multiplier=mult)
                event_signals = self.event.get_trading_signals()
                self.ensemble.set_event_signals(event_signals)
                top_ideas = self.event.get_top_ideas(min_alpha_bps=50)
                event_data = {
                    "total_events": event_result.total_events,
                    "positions": len(event_result.positions),
                    "weighted_alpha_bps": event_result.weighted_expected_alpha_bps,
                    "top_ideas": [{"ticker": p.ticker, "type": p.event_type.value,
                                   "alpha_bps": p.expected_alpha_bps} for p in top_ideas[:5]],
                    "event_counts": event_result.event_counts,
                }
                logger.info(f"Event analysis: {event_result.total_events} events, "
                            f"wtd alpha={event_result.weighted_expected_alpha_bps:+.0f}bps")
            except Exception as e:
                event_data = {"error": str(e)}
                logger.warning(f"Event-driven stage failed: {e}")
        else:
            event_data = {"status": "event_engine_not_available"}
        result["stages"]["event_driven"] = event_data
        self.tracker.record_stage("event_driven", (datetime.now() - t0).total_seconds() * 1000, event_data)

        # Stage 3.95: Credit quality classification (UniverseClassifier)
        t0 = datetime.now()
        credit_data = {}
        try:
            from ..ml.universe_classifier import UniverseClassifier
            classifier = UniverseClassifier()
            # Build fundamentals from return data for universe
            all_tickers = [s.ticker for s in self.universe.get_all()[:50]]
            returns_df = get_returns(all_tickers, start="2024-01-01")
            if not returns_df.empty:
                for ticker in returns_df.columns:
                    if ticker in returns_df.columns:
                        classifier.store.compute_from_returns(ticker, returns_df[ticker].dropna())
                classifier.train()
                classifications = classifier.classify_universe()
                credit_scores = classifier.get_credit_scores()
                self.ensemble.set_credit_scores(credit_scores)
                # Run reconciliation
                classifier.reconcile_universe()
                flagged = classifier.get_flagged()
                credit_data = {
                    "classified": len(classifications),
                    "ml_available": classifier.is_ml_available,
                    "flagged_divergences": len(flagged),
                    "rising_stars": len(classifier.get_rising_stars()),
                    "fallen_angels": len(classifier.get_fallen_angels()),
                }
                logger.info(f"Credit quality: {len(classifications)} classified, "
                            f"{len(flagged)} flagged divergences")
            else:
                credit_data = {"status": "no_return_data"}
        except Exception as e:
            credit_data = {"error": str(e)}
            logger.warning(f"Credit quality stage failed: {e}")
        result["stages"]["credit_quality"] = credit_data
        self.tracker.record_stage("credit_quality", (datetime.now() - t0).total_seconds() * 1000, credit_data)

        # Stage 4: Select top names from leading sectors
        t0 = datetime.now()
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
        self.tracker.record_stage("selection", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["selection"])

        # Stage 5: Alpha optimisation
        t0 = datetime.now()
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
        self.tracker.record_stage("alpha", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["alpha"])

        # Stage 5.5: Decision Matrix (evaluate all alpha signals through gates)
        t0 = datetime.now()
        try:
            from .decision_matrix import DecisionMatrix, AlphaBetaUnleashed
            decision_mx = DecisionMatrix()
            proposals = []
            for ticker, weight in alpha_out.optimal_weights.items():
                if weight < 0.01:
                    continue
                sig = alpha_map.get(ticker)
                # Include credit quality score if available
                cq_score = 0.5
                if hasattr(self.ensemble, '_credit_scores') and self.ensemble._credit_scores:
                    cq_data = self.ensemble._credit_scores.get(ticker, {})
                    cq_score = cq_data.get("credit_quality_score", 0.5)
                proposals.append({
                    "ticker": ticker,
                    "weight": weight,
                    "alpha": sig.alpha_pred if sig else 0.0,
                    "quality_tier": sig.quality_tier if sig else "D",
                    "regime": cube_out.regime.value,
                    "conviction": weight * 2,
                    "adv_ratio": 0.01,
                    "credit_quality_score": cq_score,
                })
            dm_results = decision_mx.evaluate_batch(proposals)
            approved = [r for r in dm_results if r.get("approved", False)]
            result["stages"]["decision_matrix"] = {
                "total_proposals": len(proposals),
                "approved": len(approved),
                "rejected": len(proposals) - len(approved),
                "top_scores": [
                    {"ticker": r["ticker"], "score": round(r.get("composite_score", 0), 4)}
                    for r in dm_results[:5]
                ],
            }
            # Override alpha weights with decision matrix approved set
            dm_approved_tickers = {r["ticker"] for r in approved}
            alpha_out.optimal_weights = {
                k: v for k, v in alpha_out.optimal_weights.items()
                if k in dm_approved_tickers
            }
        except Exception as e:
            result["stages"]["decision_matrix"] = {"error": str(e)}
        self.tracker.record_stage("decision_matrix", (datetime.now() - t0).total_seconds() * 1000,
                                  result["stages"].get("decision_matrix", {}))

        # Stage 6: Beta corridor + AlphaBetaUnleashed
        t0 = datetime.now()
        beta_state, beta_action = self.beta.run_cycle(
            regime_beta_cap=cube_out.beta_cap,
        )
        # AlphaBetaUnleashed: 1-min cadence beta management
        abu_data = {}
        try:
            from .decision_matrix import AlphaBetaUnleashed
            abu = AlphaBetaUnleashed()
            sleeve_beta = alpha_out.sharpe_ratio * 0.1 if alpha_out.sharpe_ratio else 0.0
            rm_adj = macro_snap.spy_return_3m * 4 + getattr(macro_snap, 'rm_adjustment', 0)
            abu_target = abu.compute_target_beta(
                rm_realized=macro_snap.spy_return_3m * 4,
                rm_adjustment=getattr(macro_snap, 'rm_adjustment', 0),
                vol=macro_snap.vix / 100,
            )
            abu_hedge = abu.compute_hedge_requirement(abu_target, sleeve_beta)
            abu_data = {
                "abu_target_beta": abu_target,
                "sleeve_beta": sleeve_beta,
                "hedge_requirement": abu_hedge,
            }
        except Exception:
            pass
        result["stages"]["beta"] = {
            "target_beta": beta_state.target_beta,
            "current_beta": beta_state.current_beta,
            "Rm": beta_state.Rm,
            "sigma_m": beta_state.sigma_m,
            "corridor": beta_state.corridor_position,
            "action": beta_action.action,
            **abu_data,
        }
        self.tracker.record_stage("beta", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["beta"])

        # Stage 7: ML vote ensemble + execution
        t0 = datetime.now()
        trades = []
        blocked_trades = []
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

            # Get allocation
            current_pos = self.broker.get_position(ticker)
            price = self.broker._get_current_price(ticker)
            if price <= 0:
                continue

            allocation = self.allocator.allocate(
                ticker=ticker,
                weight=weight,
                vote=vote,
                nav=nav,
                equity_budget=equity_budget,
                current_position=current_pos,
                price=price,
            )

            if allocation["side"] is None or allocation["quantity"] <= 0:
                continue

            # Risk gate check
            if self.risk_gates:
                gate_results = self.risk_gates.evaluate_all(
                    ticker=ticker,
                    side=allocation["side"],
                    quantity=allocation["quantity"],
                    price=price,
                    broker=self.broker,
                )
                if not self.risk_gates.all_passed(gate_results):
                    failed_gates = [g.gate_name for g in gate_results if not g.passed]
                    blocked_trades.append({
                        "ticker": ticker,
                        "reason": f"Risk gates failed: {', '.join(failed_gates)}",
                        "vote_score": vote.score,
                    })
                    continue

            # Execute
            order = self.broker.place_order(
                ticker=ticker,
                side=allocation["side"],
                quantity=allocation["quantity"],
                signal_type=vote.signal,
                reason=f"Vote={vote.score:.1f} Alpha={alpha_map.get(ticker, AlphaSignal(ticker=ticker)).alpha_pred:.4f} Conf={vote.confidence:.2f}",
            )

            if self.risk_gates:
                self.risk_gates.record_trade()

            trade_record = {
                "ticker": ticker,
                "side": allocation["side"].value,
                "qty": allocation["quantity"],
                "price": price,
                "vote_score": vote.score,
                "confidence": vote.confidence,
                "signal": vote.signal.value,
                "order_id": order.id,
            }
            trades.append(trade_record)
            self._trade_log.append(trade_record)

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
            "blocked_trades": blocked_trades,
            "portfolio": self.broker.get_portfolio_summary(),
        }
        self.tracker.record_stage("execution", (datetime.now() - t0).total_seconds() * 1000, result["stages"]["execution"])

        # Pipeline summary
        result["pipeline"] = self.tracker.get_summary()
        if self.risk_gates:
            result["risk_summary"] = self.risk_gates.get_summary()

        self._last_run = result
        return result

    def get_portfolio_summary(self) -> dict:
        return self.broker.get_portfolio_summary()

    def get_positions(self) -> dict:
        return {k: {"qty": v.quantity, "price": v.current_price, "pnl": v.unrealized_pnl}
                for k, v in self.broker.get_all_positions().items()}

    def get_last_run(self) -> Optional[dict]:
        return self._last_run

    def get_trade_log(self) -> list[dict]:
        return list(self._trade_log)

    def get_risk_gate_summary(self) -> dict:
        if self.risk_gates:
            return self.risk_gates.get_summary()
        return {}

    def get_pipeline_timing(self) -> dict:
        return self.tracker.get_summary()

    def format_execution_report(self) -> str:
        """Generate ASCII execution report from last run."""
        if not self._last_run:
            return "No pipeline run yet."

        r = self._last_run
        lines = [
            "=" * 70,
            f"EXECUTION REPORT — Run #{r.get('run_number', '?')}",
            f"Timestamp: {r['timestamp']}",
            "=" * 70,
        ]

        # Regime
        cube = r["stages"].get("cube", {})
        lines.append(f"\nREGIME: {cube.get('regime', 'N/A')}")
        lines.append(f"  Target Beta: {cube.get('target_beta', 0):.3f}  |  "
                      f"Beta Cap: {cube.get('beta_cap', 0):.3f}  |  "
                      f"Max Leverage: {cube.get('max_leverage', 0):.1f}x")

        # Sleeves
        sleeves = cube.get("sleeves", {})
        if sleeves:
            lines.append("\n  SLEEVE ALLOCATION:")
            for k, v in sleeves.items():
                lines.append(f"    {k}: {v:.1%}")

        # Social Prediction
        social = r["stages"].get("social_prediction", {})
        if social.get("total_agents", 0) > 0:
            lines.append(f"\nSOCIAL PREDICTION (MiroFish):")
            lines.append(f"  Signal: {social.get('social_signal', 'N/A')}  |  "
                          f"Strength: {social.get('signal_strength', 0):.3f}  |  "
                          f"Vote: {social.get('vote_score', 0):+d}")
            lines.append(f"  Sentiment: {social.get('overall_sentiment', 0):+.3f}  |  "
                          f"Confidence: {social.get('sentiment_confidence', 0):.3f}  |  "
                          f"Trend: {social.get('sentiment_trend', 0):+.3f}")
            lines.append(f"  Agents: {social.get('total_agents', 0)} "
                          f"(Bull={social.get('agents_bullish', 0)} / "
                          f"Bear={social.get('agents_bearish', 0)} / "
                          f"Neutral={social.get('agents_neutral', 0)})")
            topics = social.get("top_topics", [])
            if topics:
                lines.append(f"  Top Topics:")
                for t in topics[:5]:
                    lines.append(f"    {t['topic']}: sentiment={t['sentiment']:+.3f} engagement={t['engagement']}")

        # Distressed Assets
        distress = r["stages"].get("distressed_assets", {})
        if distress.get("total_analyzed", 0) > 0:
            lines.append(f"\nDISTRESSED ASSETS:")
            lines.append(f"  Analyzed: {distress.get('total_analyzed', 0)}  |  "
                          f"Fallen Angels: {len(distress.get('fallen_angels', []))}  |  "
                          f"Critical: {len(distress.get('critical_names', []))}  |  "
                          f"Opportunities: {distress.get('opportunities', 0)}")
            if distress.get("fallen_angels"):
                lines.append(f"  Fallen Angels: {', '.join(distress['fallen_angels'])}")
            if distress.get("critical_names"):
                lines.append(f"  Critical: {', '.join(distress['critical_names'])}")

        # CVR
        cvr = r["stages"].get("cvr", {})
        if cvr.get("total_instruments", 0) > 0:
            lines.append(f"\nCVR ANALYSIS:")
            lines.append(f"  Instruments: {cvr.get('total_instruments', 0)}  |  "
                          f"Buy Signals: {len(cvr.get('buy_signals', []))}")
            if cvr.get("buy_signals"):
                lines.append(f"  Buy: {', '.join(cvr['buy_signals'])}")

        # Event-Driven
        event = r["stages"].get("event_driven", {})
        if event.get("total_events", 0) > 0:
            lines.append(f"\nEVENT-DRIVEN:")
            lines.append(f"  Events: {event.get('total_events', 0)}  |  "
                          f"Positions: {event.get('positions', 0)}  |  "
                          f"Wtd Alpha: {event.get('weighted_alpha_bps', 0):+.0f}bps")
            top = event.get("top_ideas", [])
            for idea in top[:3]:
                lines.append(f"    {idea['ticker']}: {idea['type']} → {idea['alpha_bps']:+.0f}bps")

        # Alpha
        alpha = r["stages"].get("alpha", {})
        lines.append(f"\nALPHA: E[R]={alpha.get('expected_return', 0):.1%}  "
                      f"Vol={alpha.get('volatility', 0):.1%}  "
                      f"Sharpe={alpha.get('sharpe', 0):.2f}")

        # Trades
        execution = r["stages"].get("execution", {})
        trades = execution.get("trades", [])
        blocked = execution.get("blocked_trades", [])

        lines.append(f"\nTRADES EXECUTED: {len(trades)}")
        if trades:
            lines.append(f"  {'Ticker':<8} {'Side':<6} {'Qty':>6} {'Price':>10} {'Vote':>6} {'Signal':<18}")
            lines.append("  " + "-" * 56)
            for t in trades[:15]:
                lines.append(
                    f"  {t['ticker']:<8} {t['side']:<6} {t['qty']:>6} "
                    f"${t['price']:>9,.2f} {t['vote_score']:>+5.1f} {t['signal']:<18}"
                )

        if blocked:
            lines.append(f"\nBLOCKED TRADES: {len(blocked)}")
            for b in blocked[:5]:
                lines.append(f"  {b['ticker']}: {b['reason']}")

        # Portfolio
        port = execution.get("portfolio", {})
        lines.extend([
            f"\nPORTFOLIO:",
            f"  NAV: ${port.get('nav', 0):,.0f}  |  Cash: ${port.get('cash', 0):,.0f}",
            f"  P&L: ${port.get('total_pnl', 0):,.0f}  |  Win Rate: {port.get('win_rate', 0):.1%}",
            f"  Positions: {port.get('positions', 0)}  |  Trades: {port.get('total_trades', 0)}",
            f"  Gross Exp: {port.get('gross_exposure', 0):.1%}  |  Net Exp: {port.get('net_exposure', 0):.1%}",
        ])

        # Pipeline timing
        pipeline = r.get("pipeline", {})
        lines.append(f"\nPIPELINE: {pipeline.get('total_duration_ms', 0):.0f}ms total")
        for stage in pipeline.get("stages", []):
            lines.append(f"  {stage['name']:<15} {stage['duration_ms']:>8.1f}ms")

        lines.append("=" * 70)
        return "\n".join(lines)
