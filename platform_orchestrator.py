# ============================================================
# SOURCE: https://github.com/Axel-009/Metadron-Capital
# LAYER:  platform (root)
# ROLE:   Master orchestrator — coordinates all subsystems into
#         a synchronized daily investment pipeline
# ============================================================
"""
Investment Platform Orchestrator for Metadron Capital.

Central orchestration layer that coordinates all subsystems into a
synchronized investment strategy. Manages the daily workflow from data
ingestion through analysis, strategy generation, execution, reporting,
and ML learning feedback loops.

Usage:
    from platform_orchestrator import InvestmentPlatformOrchestrator
    orchestrator = InvestmentPlatformOrchestrator()
    orchestrator.daily_open_routine()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine module imports — the orchestrator USES engine modules, not re-creates
# ---------------------------------------------------------------------------
_ENGINE_AVAILABLE = True
_ENGINE_IMPORT_ERRORS = []

# L1 Data
try:
    from engine.data.universe_engine import UniverseEngine, get_engine, GICS_SECTORS
except ImportError as e:
    UniverseEngine = None
    get_engine = None
    GICS_SECTORS = {}
    _ENGINE_IMPORT_ERRORS.append(f"data.universe_engine: {e}")

try:
    from engine.data.openbb_data import (
        get_adj_close, get_returns, get_prices, get_fundamentals,
        get_macro_data, get_market_stats, get_fred_series, get_options_chains,
    )
except ImportError as e:
    _ENGINE_IMPORT_ERRORS.append(f"data.openbb_data: {e}")
    try:
        from engine.data.yahoo_data import get_adj_close, get_returns, get_prices
    except ImportError:
        pass

# L2 Signal Engines
try:
    from engine.signals.macro_engine import MacroEngine, MacroSnapshot, MarketRegime as MacroRegime, CubeRegime
except ImportError as e:
    MacroEngine = None
    MacroSnapshot = None
    MacroRegime = None
    CubeRegime = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.macro_engine: {e}")

try:
    from engine.signals.metadron_cube import MetadronCube, CubeOutput
except ImportError as e:
    MetadronCube = None
    CubeOutput = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.metadron_cube: {e}")

try:
    from engine.signals.security_analysis_engine import SecurityAnalysisEngine
except ImportError as e:
    SecurityAnalysisEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.security_analysis_engine: {e}")

try:
    from engine.signals.contagion_engine import ContagionEngine
except ImportError as e:
    ContagionEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.contagion_engine: {e}")

try:
    from engine.signals.stat_arb_engine import StatArbEngine
except ImportError as e:
    StatArbEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.stat_arb_engine: {e}")

try:
    from engine.signals.social_prediction_engine import SocialPredictionEngine
except ImportError as e:
    SocialPredictionEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.social_prediction_engine: {e}")

try:
    from engine.signals.distressed_asset_engine import DistressedAssetEngine
except ImportError as e:
    DistressedAssetEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.distressed_asset_engine: {e}")

try:
    from engine.signals.cvr_engine import CVREngine
except ImportError as e:
    CVREngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.cvr_engine: {e}")

try:
    from engine.signals.event_driven_engine import EventDrivenEngine
except ImportError as e:
    EventDrivenEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.event_driven_engine: {e}")

try:
    from engine.signals.fed_liquidity_plumbing import FedLiquidityPlumbing
except ImportError as e:
    FedLiquidityPlumbing = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.fed_liquidity_plumbing: {e}")

try:
    from engine.signals.agent_sim_engine import AgentSimEngine
except ImportError as e:
    AgentSimEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"signals.agent_sim_engine: {e}")

# L3 ML Engines
try:
    from engine.ml.alpha_optimizer import AlphaOptimizer, AlphaOutput, AlphaSignal
except ImportError as e:
    AlphaOptimizer = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.alpha_optimizer: {e}")

try:
    from engine.ml.universe_classifier import UniverseClassifier
except ImportError as e:
    UniverseClassifier = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.universe_classifier: {e}")

try:
    from engine.ml.pattern_recognition import PatternRecognitionEngine
except ImportError as e:
    PatternRecognitionEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.pattern_recognition: {e}")

try:
    from engine.ml.deep_learning_engine import DeepLearningEngine
except ImportError as e:
    DeepLearningEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.deep_learning_engine: {e}")

try:
    from engine.ml.model_evaluator import ModelEvaluator
except ImportError as e:
    ModelEvaluator = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.model_evaluator: {e}")

try:
    from engine.ml.model_store import ModelStore
except ImportError as e:
    ModelStore = None
    _ENGINE_IMPORT_ERRORS.append(f"ml.model_store: {e}")

# L4 Portfolio
try:
    from engine.portfolio.beta_corridor import BetaCorridor, BetaState, BetaAction
except ImportError as e:
    BetaCorridor = None
    _ENGINE_IMPORT_ERRORS.append(f"portfolio.beta_corridor: {e}")

# L5 Execution
try:
    from engine.execution.execution_engine import ExecutionEngine
except ImportError as e:
    ExecutionEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"execution.execution_engine: {e}")

try:
    from engine.execution.options_engine import OptionsEngine, BlackScholesModel
except ImportError as e:
    OptionsEngine = None
    BlackScholesModel = None
    _ENGINE_IMPORT_ERRORS.append(f"execution.options_engine: {e}")

# L4 Risk
try:
    from engine.risk.monte_carlo_risk import MonteCarloRiskEngine
except ImportError as e:
    MonteCarloRiskEngine = None
    _ENGINE_IMPORT_ERRORS.append(f"risk.monte_carlo_risk: {e}")

# Monitoring
try:
    from engine.monitoring.learning_loop import LearningLoop
except ImportError as e:
    LearningLoop = None
    _ENGINE_IMPORT_ERRORS.append(f"monitoring.learning_loop: {e}")

if _ENGINE_IMPORT_ERRORS:
    logger.warning("Engine import warnings: %s", "; ".join(_ENGINE_IMPORT_ERRORS))
else:
    logger.info("All engine modules loaded successfully")


# ---------------------------------------------------------------------------
# Enumerations and configuration
# ---------------------------------------------------------------------------


class TradingHorizon(str, Enum):
    """Multi-horizon strategy time frames."""

    HFT = "hft"                      # Intraday alpha capture
    SWING = "swing"                  # 1-5 day momentum
    MEDIUM_TERM = "medium_term"      # 1-6 month positions (separate book)
    LONG_TERM = "long_term"          # 6+ month macro trades (separate book)


class SignalType(str, Enum):
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    MACRO = "macro"
    SENTIMENT = "sentiment"
    DISTRESS = "distress"
    QUANTITATIVE = "quantitative"


class ExecutionMethod(str, Enum):
    ALGORITHMIC = "algorithmic"
    VOICE = "voice"
    DMA = "dma"
    DARK_POOL = "dark_pool"


class MarketRegime(str, Enum):
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    TRANSITION = "transition"
    CRISIS = "crisis"
    EUPHORIA = "euphoria"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class AnalysisSignal:
    """A single analysis signal from any subsystem."""

    ticker: str
    signal_type: SignalType
    direction: str  # "long", "short", "neutral"
    strength: float  # -1.0 to 1.0
    confidence: float  # 0 to 1
    horizon: TradingHorizon
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TradeThesis:
    """Complete trade thesis with rationale and sizing."""

    ticker: str
    direction: str
    horizon: TradingHorizon
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    notional: float
    signals: list[AnalysisSignal] = field(default_factory=list)
    composite_score: float = 0.0
    risk_reward_ratio: float = 0.0
    thesis_text: str = ""
    execution_method: ExecutionMethod = ExecutionMethod.ALGORITHMIC
    urgency: str = "normal"  # normal, high, immediate
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ExecutionResult:
    """Result of a trade execution."""

    trade_id: str
    ticker: str
    direction: str
    requested_size: float
    filled_size: float
    avg_price: float
    slippage_bps: float
    commission: float
    execution_time_ms: float
    venue: str
    status: str  # filled, partial, rejected, cancelled
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class PortfolioPosition:
    """Current portfolio position."""

    ticker: str
    direction: str
    quantity: float
    avg_entry: float
    current_price: float
    unrealised_pnl: float
    realised_pnl: float
    horizon: TradingHorizon
    book: str  # hft_book, swing_book, medium_book, long_book
    entry_date: str
    thesis: Optional[TradeThesis] = None


@dataclass
class AnalysisResults:
    """Aggregated results from the analysis pipeline."""

    technical_signals: list[AnalysisSignal] = field(default_factory=list)
    fundamental_signals: list[AnalysisSignal] = field(default_factory=list)
    macro_signals: list[AnalysisSignal] = field(default_factory=list)
    sentiment_signals: list[AnalysisSignal] = field(default_factory=list)
    distress_signals: list[AnalysisSignal] = field(default_factory=list)
    regime: MarketRegime = MarketRegime.RISK_ON
    factor_exposures: dict = field(default_factory=dict)
    sector_scores: dict = field(default_factory=dict)
    composite_opportunities: list[dict] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DailyReport:
    """End-of-day performance and activity report."""

    date: str
    total_pnl: float
    pnl_by_book: dict[str, float] = field(default_factory=dict)
    pnl_by_sector: dict[str, float] = field(default_factory=dict)
    trades_executed: int = 0
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    sharpe_estimate: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    missed_opportunities: list[dict] = field(default_factory=list)
    new_recommendations: list[TradeThesis] = field(default_factory=list)
    attribution: dict = field(default_factory=dict)


@dataclass
class WeeklyReport:
    """Weekly review aggregating daily reports."""

    week_ending: str
    daily_reports: list[DailyReport] = field(default_factory=list)
    weekly_pnl: float = 0.0
    best_trades: list[dict] = field(default_factory=list)
    worst_trades: list[dict] = field(default_factory=list)
    strategy_performance: dict[str, float] = field(default_factory=dict)
    model_accuracy: dict[str, float] = field(default_factory=dict)
    regime_changes: list[dict] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


@dataclass
class RiskLimits:
    """Risk management limits."""

    max_gross_exposure: float = 2.0  # 2x NAV
    max_net_exposure: float = 0.5   # 50% NAV
    max_single_name_pct: float = 0.05  # 5%
    max_sector_pct: float = 0.25  # 25%
    max_daily_loss_pct: float = 0.02  # 2%
    max_drawdown_pct: float = 0.10  # 10%
    max_hft_notional: float = 10_000_000
    max_correlation: float = 0.70
    min_liquidity_days: float = 3.0


# ---------------------------------------------------------------------------
# Subsystem interfaces
# ---------------------------------------------------------------------------


class _TechnicalAnalyzer:
    """Technical analysis across all asset classes."""

    def scan(self, universe: list, price_cache: dict[str, pd.DataFrame]) -> list[AnalysisSignal]:
        """
        Run technical scan on universe.

        Indicators: RSI(14), MACD(12,26,9), Bollinger(20,2), ATR(14),
        Volume profile, Support/Resistance levels, Ichimoku Cloud,
        Moving averages (20/50/200 SMA/EMA crossovers).
        """
        signals: list[AnalysisSignal] = []

        for security in universe:
            ticker = security.ticker if hasattr(security, "ticker") else str(security)
            prices = price_cache.get(ticker)
            if prices is None or len(prices) < 200:
                continue

            close = prices["Close"].values if "Close" in prices.columns else prices.iloc[:, 3].values

            # RSI(14)
            rsi = self._calculate_rsi(close, 14)

            # MACD(12, 26, 9)
            macd_line, signal_line, histogram = self._calculate_macd(close)

            # Bollinger Bands(20, 2)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger(close, 20, 2)

            # Moving average crossovers
            sma_20 = self._sma(close, 20)
            sma_50 = self._sma(close, 50)
            sma_200 = self._sma(close, 200)

            # Composite signal
            score = 0.0
            n_signals = 0

            # RSI signal
            if rsi is not None:
                if rsi < 30:
                    score += 0.5  # oversold -> bullish
                elif rsi > 70:
                    score -= 0.5  # overbought -> bearish
                n_signals += 1

            # MACD crossover
            if macd_line is not None and signal_line is not None:
                if macd_line > signal_line:
                    score += 0.3
                else:
                    score -= 0.3
                n_signals += 1

            # Bollinger position
            current = close[-1]
            if bb_lower is not None and current < bb_lower:
                score += 0.4  # below lower band -> bullish
            elif bb_upper is not None and current > bb_upper:
                score -= 0.4  # above upper band -> bearish
            n_signals += 1

            # MA trend
            if sma_20 is not None and sma_50 is not None and sma_200 is not None:
                if sma_20 > sma_50 > sma_200:
                    score += 0.5  # strong uptrend
                elif sma_20 < sma_50 < sma_200:
                    score -= 0.5  # strong downtrend
                n_signals += 1

            if n_signals > 0:
                avg_score = score / n_signals
                direction = "long" if avg_score > 0.1 else ("short" if avg_score < -0.1 else "neutral")
                confidence = min(1.0, abs(avg_score) * 2)

                # Determine horizon based on signal persistence
                horizon = TradingHorizon.SWING
                if abs(avg_score) > 0.4:
                    horizon = TradingHorizon.MEDIUM_TERM

                signals.append(AnalysisSignal(
                    ticker=ticker,
                    signal_type=SignalType.TECHNICAL,
                    direction=direction,
                    strength=round(avg_score, 4),
                    confidence=round(confidence, 4),
                    horizon=horizon,
                    metadata={"rsi": rsi, "macd": macd_line, "sma_20": sma_20},
                ))

        logger.info("Technical scan: %d signals from %d securities", len(signals), len(universe))
        return signals

    @staticmethod
    def _sma(data: np.ndarray, period: int) -> Optional[float]:
        if len(data) < period:
            return None
        return float(np.mean(data[-period:]))

    @staticmethod
    def _ema(data: np.ndarray, period: int) -> Optional[np.ndarray]:
        if len(data) < period:
            return None
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(data))
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> Optional[float]:
        if len(close) < period + 1:
            return None
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return round(100.0 - (100.0 / (1.0 + rs)), 2)

    def _calculate_macd(
        self, close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        ema_fast = self._ema(close, fast)
        ema_slow = self._ema(close, slow)
        if ema_fast is None or ema_slow is None:
            return None, None, None

        macd_line = ema_fast - ema_slow
        signal_ema = self._ema(macd_line, signal)
        if signal_ema is None:
            return float(macd_line[-1]), None, None

        histogram = macd_line - signal_ema
        return float(macd_line[-1]), float(signal_ema[-1]), float(histogram[-1])

    @staticmethod
    def _calculate_bollinger(
        close: np.ndarray, period: int = 20, std_mult: float = 2.0,
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if len(close) < period:
            return None, None, None
        window = close[-period:]
        middle = float(np.mean(window))
        std = float(np.std(window, ddof=1))
        return middle + std_mult * std, middle, middle - std_mult * std


class _FundamentalAnalyzer:
    """Fundamental analysis for equities and bonds."""

    def scan(
        self,
        universe: list,
        fundamentals_cache: dict[str, dict],
    ) -> list[AnalysisSignal]:
        """
        Score universe on fundamental metrics: P/E, P/B, ROE, FCF yield,
        earnings growth, debt/equity, dividend yield, Piotroski F-score.
        """
        signals: list[AnalysisSignal] = []

        for security in universe:
            ticker = security.ticker if hasattr(security, "ticker") else str(security)
            data = fundamentals_cache.get(ticker, {})
            ratios = data.get("ratios")
            if ratios is None or (hasattr(ratios, "empty") and ratios.empty):
                continue

            try:
                row = ratios.iloc[0] if hasattr(ratios, "iloc") else ratios
                pe = float(row.get("pe_ratio", row.get("peRatio", 0)) or 0)
                pb = float(row.get("pb_ratio", row.get("priceToBook", 0)) or 0)
                roe = float(row.get("roe", row.get("returnOnEquity", 0)) or 0)
                div_yield = float(row.get("dividend_yield", row.get("dividendYield", 0)) or 0)

                # Value scoring
                score = 0.0
                if 0 < pe < 15:
                    score += 0.3
                elif pe > 30:
                    score -= 0.2
                if 0 < pb < 1.5:
                    score += 0.2
                if roe > 0.15:
                    score += 0.3
                elif roe < 0:
                    score -= 0.3
                if div_yield > 0.03:
                    score += 0.2

                direction = "long" if score > 0.2 else ("short" if score < -0.2 else "neutral")
                confidence = min(1.0, abs(score) * 1.5)

                signals.append(AnalysisSignal(
                    ticker=ticker,
                    signal_type=SignalType.FUNDAMENTAL,
                    direction=direction,
                    strength=round(score, 4),
                    confidence=round(confidence, 4),
                    horizon=TradingHorizon.MEDIUM_TERM,
                    metadata={"pe": pe, "pb": pb, "roe": roe, "div_yield": div_yield},
                ))
            except Exception as exc:
                logger.debug("Fundamental scan failed for %s: %s", ticker, exc)

        logger.info("Fundamental scan: %d signals", len(signals))
        return signals


class _MacroAnalyzer:
    """Macro regime detection and factor analysis."""

    def detect_regime(self, macro_data: dict) -> MarketRegime:
        """
        Detect current market regime based on macro indicators.

        Inputs: VIX, yield curve slope, credit spreads, PMI, leading indicators.
        """
        vix = macro_data.get("vix", 20)
        yield_slope = macro_data.get("yield_curve_slope", 0.5)  # 10Y - 2Y
        credit_spread = macro_data.get("credit_spread_bps", 150)
        pmi = macro_data.get("pmi", 52)

        # Regime scoring
        risk_score = 0

        # VIX
        if vix < 15:
            risk_score += 2  # complacency / risk-on
        elif vix < 20:
            risk_score += 1
        elif vix < 30:
            risk_score -= 1
        else:
            risk_score -= 3  # fear

        # Yield curve
        if yield_slope > 1.0:
            risk_score += 1
        elif yield_slope < 0:
            risk_score -= 2  # inversion = recession signal

        # Credit spreads
        if credit_spread < 100:
            risk_score += 1
        elif credit_spread > 300:
            risk_score -= 2

        # PMI
        if pmi > 55:
            risk_score += 1
        elif pmi < 48:
            risk_score -= 2

        if risk_score >= 4:
            return MarketRegime.EUPHORIA
        elif risk_score >= 1:
            return MarketRegime.RISK_ON
        elif risk_score >= -2:
            return MarketRegime.TRANSITION
        elif risk_score >= -4:
            return MarketRegime.RISK_OFF
        else:
            return MarketRegime.CRISIS

    def calculate_factor_exposures(
        self, portfolio: list[PortfolioPosition], factor_data: dict,
    ) -> dict[str, float]:
        """
        Calculate portfolio factor exposures.

        Factors: market beta, size (SMB), value (HML), momentum (UMD),
        quality (QMJ), low_vol, credit, duration, curve.
        """
        exposures: dict[str, float] = {
            "market_beta": 0.0,
            "size_smb": 0.0,
            "value_hml": 0.0,
            "momentum_umd": 0.0,
            "quality_qmj": 0.0,
            "low_vol": 0.0,
            "credit": 0.0,
            "duration": 0.0,
        }

        if not portfolio:
            return exposures

        total_notional = sum(abs(p.quantity * p.current_price) for p in portfolio)
        if total_notional == 0:
            return exposures

        for pos in portfolio:
            weight = (pos.quantity * pos.current_price) / total_notional
            betas = factor_data.get(pos.ticker, {})

            for factor in exposures:
                factor_beta = betas.get(factor, 0.0)
                exposures[factor] += weight * factor_beta

        return {k: round(v, 4) for k, v in exposures.items()}

    def generate_macro_signals(self, regime: MarketRegime, factor_data: dict) -> list[AnalysisSignal]:
        """Generate macro-level trading signals based on regime."""
        signals: list[AnalysisSignal] = []

        regime_biases = {
            MarketRegime.RISK_ON: {"equity": 0.5, "credit": 0.3, "duration": -0.2, "commodity": 0.3},
            MarketRegime.RISK_OFF: {"equity": -0.5, "credit": -0.3, "duration": 0.5, "commodity": -0.3},
            MarketRegime.CRISIS: {"equity": -0.8, "credit": -0.6, "duration": 0.8, "gold": 0.7},
            MarketRegime.EUPHORIA: {"equity": 0.3, "credit": 0.2, "duration": -0.3, "commodity": 0.4},
            MarketRegime.TRANSITION: {"equity": 0.0, "credit": 0.0, "duration": 0.0, "commodity": 0.0},
        }

        biases = regime_biases.get(regime, {})
        macro_proxies = {
            "equity": "SPY", "credit": "HYG", "duration": "TLT",
            "commodity": "DBC", "gold": "GLD",
        }

        for asset_type, bias in biases.items():
            if abs(bias) < 0.1:
                continue
            proxy = macro_proxies.get(asset_type, "SPY")
            direction = "long" if bias > 0 else "short"
            signals.append(AnalysisSignal(
                ticker=proxy,
                signal_type=SignalType.MACRO,
                direction=direction,
                strength=bias,
                confidence=0.6,
                horizon=TradingHorizon.MEDIUM_TERM,
                metadata={"regime": regime.value, "asset_type": asset_type},
            ))

        return signals


class _SentimentAnalyzer:
    """News and sentiment analysis."""

    def analyze(self, news_data: list[dict], universe: list) -> list[AnalysisSignal]:
        """
        Analyze news sentiment for the universe.

        Uses keyword-based scoring as a baseline, with provision for
        ML model integration.
        """
        signals: list[AnalysisSignal] = []
        ticker_set = {
            s.ticker if hasattr(s, "ticker") else str(s) for s in universe
        }

        # Aggregate sentiment by ticker
        ticker_sentiment: dict[str, list[float]] = {}

        positive_words = {
            "beat", "exceeds", "upgrade", "bullish", "growth", "record",
            "breakout", "profit", "revenue beat", "strong", "outperform",
            "raised guidance", "buy", "momentum", "recovery", "turnaround",
        }
        negative_words = {
            "miss", "downgrade", "bearish", "decline", "loss", "bankruptcy",
            "default", "fraud", "investigation", "warning", "cut", "layoff",
            "sell", "weak", "underperform", "lowered guidance", "restructuring",
        }

        for item in news_data:
            ticker = item.get("ticker", "")
            if ticker not in ticker_set:
                continue

            text = (item.get("headline", "") + " " + item.get("body", "")).lower()

            pos_count = sum(1 for w in positive_words if w in text)
            neg_count = sum(1 for w in negative_words if w in text)

            total = pos_count + neg_count
            if total == 0:
                continue

            score = (pos_count - neg_count) / total

            if ticker not in ticker_sentiment:
                ticker_sentiment[ticker] = []
            ticker_sentiment[ticker].append(score)

        for ticker, scores in ticker_sentiment.items():
            avg = float(np.mean(scores))
            direction = "long" if avg > 0.1 else ("short" if avg < -0.1 else "neutral")
            confidence = min(1.0, len(scores) / 10.0 * abs(avg))

            signals.append(AnalysisSignal(
                ticker=ticker,
                signal_type=SignalType.SENTIMENT,
                direction=direction,
                strength=round(avg, 4),
                confidence=round(confidence, 4),
                horizon=TradingHorizon.SWING,
                metadata={"news_count": len(scores), "avg_sentiment": round(avg, 4)},
            ))

        logger.info("Sentiment scan: %d signals from %d news items", len(signals), len(news_data))
        return signals


class _RiskManager:
    """Portfolio-level risk management."""

    def __init__(self, limits: RiskLimits) -> None:
        self.limits = limits

    def check_trade(
        self,
        trade: TradeThesis,
        portfolio: list[PortfolioPosition],
        nav: float,
    ) -> tuple[bool, str]:
        """
        Validate a proposed trade against risk limits.

        Returns (approved, reason).
        """
        trade_notional = abs(trade.notional)

        # Single name concentration
        existing = sum(
            abs(p.quantity * p.current_price)
            for p in portfolio if p.ticker == trade.ticker
        )
        combined_pct = (existing + trade_notional) / nav if nav > 0 else 1.0
        if combined_pct > self.limits.max_single_name_pct:
            return False, f"Single name limit exceeded: {combined_pct:.1%} > {self.limits.max_single_name_pct:.1%}"

        # Gross exposure
        gross = sum(abs(p.quantity * p.current_price) for p in portfolio) + trade_notional
        if gross / nav > self.limits.max_gross_exposure:
            return False, f"Gross exposure limit: {gross/nav:.2f}x > {self.limits.max_gross_exposure:.2f}x"

        # Net exposure
        long_val = sum(
            p.quantity * p.current_price for p in portfolio if p.direction == "long"
        )
        short_val = sum(
            abs(p.quantity * p.current_price) for p in portfolio if p.direction == "short"
        )
        if trade.direction == "long":
            net = (long_val + trade_notional - short_val) / nav
        else:
            net = (long_val - short_val - trade_notional) / nav
        if abs(net) > self.limits.max_net_exposure:
            return False, f"Net exposure limit: {abs(net):.1%} > {self.limits.max_net_exposure:.1%}"

        # HFT book limit
        if trade.horizon == TradingHorizon.HFT:
            hft_notional = sum(
                abs(p.quantity * p.current_price)
                for p in portfolio if p.book == "hft_book"
            )
            if hft_notional + trade_notional > self.limits.max_hft_notional:
                return False, "HFT notional limit exceeded"

        return True, "Approved"

    def position_size(
        self,
        signal_strength: float,
        confidence: float,
        volatility: float,
        nav: float,
        max_loss_pct: float = 0.01,
    ) -> float:
        """
        Calculate position size using volatility-adjusted sizing.

        Size = (NAV * max_loss_pct * confidence * |signal|) / volatility
        Capped at single-name limit.
        """
        if volatility <= 0:
            return 0.0

        raw_size = (nav * max_loss_pct * confidence * abs(signal_strength)) / volatility
        max_size = nav * self.limits.max_single_name_pct
        return min(raw_size, max_size)

    def calculate_portfolio_var(
        self,
        portfolio: list[PortfolioPosition],
        returns_data: dict[str, np.ndarray],
        confidence: float = 0.99,
        horizon_days: int = 1,
    ) -> float:
        """
        Calculate portfolio Value-at-Risk using parametric method.

        VaR = Z * sigma_portfolio * sqrt(T) * portfolio_value
        """
        if not portfolio:
            return 0.0

        tickers = [p.ticker for p in portfolio]
        weights = np.array([p.quantity * p.current_price for p in portfolio])
        total = np.sum(np.abs(weights))
        if total == 0:
            return 0.0
        weights = weights / total

        # Build correlation matrix from returns
        n = len(tickers)
        returns_matrix = []
        for t in tickers:
            r = returns_data.get(t, np.zeros(252))
            returns_matrix.append(r[-252:])

        if returns_matrix:
            ret_arr = np.array(returns_matrix)
            cov_matrix = np.cov(ret_arr)
            port_var = float(np.dot(weights, np.dot(cov_matrix, weights)))
            port_std = np.sqrt(port_var) * np.sqrt(horizon_days)

            from scipy import stats as sp_stats
            z_score = sp_stats.norm.ppf(confidence)
            var = z_score * port_std * total
            return round(var, 2)

        return 0.0


class _CubeRotation:
    """
    Composite opportunity scoring: "Cube Rotation" algorithm.

    Aggregates signals across all dimensions (technical, fundamental,
    macro, sentiment, distress) into a single composite score per security.
    """

    def __init__(
        self,
        weights: Optional[dict[SignalType, float]] = None,
    ) -> None:
        self.weights = weights or {
            SignalType.TECHNICAL: 0.25,
            SignalType.FUNDAMENTAL: 0.25,
            SignalType.MACRO: 0.20,
            SignalType.SENTIMENT: 0.15,
            SignalType.DISTRESS: 0.15,
        }

    def score_universe(
        self,
        all_signals: list[AnalysisSignal],
    ) -> list[dict]:
        """
        Produce ranked opportunity list from all signals.

        Returns list of dicts: {ticker, composite_score, direction, signals, horizon}.
        Sorted by absolute composite score descending.
        """
        # Group signals by ticker
        by_ticker: dict[str, list[AnalysisSignal]] = {}
        for sig in all_signals:
            if sig.ticker not in by_ticker:
                by_ticker[sig.ticker] = []
            by_ticker[sig.ticker].append(sig)

        opportunities: list[dict] = []

        for ticker, signals in by_ticker.items():
            weighted_score = 0.0
            total_weight = 0.0

            for sig in signals:
                w = self.weights.get(sig.signal_type, 0.1)
                direction_mult = 1.0 if sig.direction == "long" else (-1.0 if sig.direction == "short" else 0.0)
                weighted_score += w * sig.strength * sig.confidence * direction_mult
                total_weight += w

            if total_weight > 0:
                composite = weighted_score / total_weight
            else:
                composite = 0.0

            direction = "long" if composite > 0 else ("short" if composite < 0 else "neutral")

            # Determine primary horizon
            horizon_votes: dict[TradingHorizon, float] = {}
            for sig in signals:
                horizon_votes[sig.horizon] = horizon_votes.get(sig.horizon, 0) + abs(sig.strength)
            primary_horizon = max(horizon_votes, key=horizon_votes.get) if horizon_votes else TradingHorizon.SWING

            opportunities.append({
                "ticker": ticker,
                "composite_score": round(composite, 6),
                "direction": direction,
                "signals": signals,
                "horizon": primary_horizon,
                "signal_count": len(signals),
            })

        opportunities.sort(key=lambda x: abs(x["composite_score"]), reverse=True)
        return opportunities


class _ExecutionEngine:
    """Trade execution management."""

    def __init__(self) -> None:
        self._trade_counter = 0

    def execute(
        self,
        trade: TradeThesis,
        method: Optional[ExecutionMethod] = None,
    ) -> ExecutionResult:
        """
        Execute a trade thesis.

        In production, this routes to the appropriate execution venue.
        """
        self._trade_counter += 1
        trade_id = f"TRD-{datetime.utcnow().strftime('%Y%m%d')}-{self._trade_counter:06d}"

        exec_method = method or trade.execution_method

        # Simulated execution (replace with broker API in production)
        start_time = time.monotonic()

        # Estimate slippage based on method and urgency
        base_slippage = {
            ExecutionMethod.ALGORITHMIC: 1.0,
            ExecutionMethod.DMA: 2.0,
            ExecutionMethod.DARK_POOL: 0.5,
            ExecutionMethod.VOICE: 3.0,
        }.get(exec_method, 2.0)

        if trade.urgency == "immediate":
            base_slippage *= 2.0
        elif trade.urgency == "high":
            base_slippage *= 1.5

        # Random slippage component (in production: actual fill data)
        slippage_bps = base_slippage + np.random.uniform(-0.5, 1.0)
        slippage_mult = 1.0 + slippage_bps / 10_000 * (1 if trade.direction == "long" else -1)

        avg_price = trade.entry_price * slippage_mult
        filled_size = trade.position_size  # assume full fill

        elapsed = (time.monotonic() - start_time) * 1000

        # Commission estimate (bps)
        commission_rate = {
            ExecutionMethod.ALGORITHMIC: 0.5,
            ExecutionMethod.DMA: 1.0,
            ExecutionMethod.DARK_POOL: 0.3,
            ExecutionMethod.VOICE: 2.0,
        }.get(exec_method, 1.0)

        commission = abs(filled_size * avg_price * commission_rate / 10_000)

        venue_map = {
            ExecutionMethod.ALGORITHMIC: "ALGO-TWAP",
            ExecutionMethod.DMA: "NYSE-DMA",
            ExecutionMethod.DARK_POOL: "IEX-DARK",
            ExecutionMethod.VOICE: "VOICE-DESK",
        }

        return ExecutionResult(
            trade_id=trade_id,
            ticker=trade.ticker,
            direction=trade.direction,
            requested_size=trade.position_size,
            filled_size=filled_size,
            avg_price=round(avg_price, 4),
            slippage_bps=round(slippage_bps, 2),
            commission=round(commission, 2),
            execution_time_ms=round(elapsed, 2),
            venue=venue_map.get(exec_method, "UNKNOWN"),
            status="filled",
        )


class _MLLearner:
    """Machine learning feedback loop for strategy improvement."""

    def __init__(self) -> None:
        self._prediction_history: list[dict] = []
        self._strategy_weights: dict[str, float] = {}
        self._missed_opportunities: list[dict] = []

    def record_prediction(
        self,
        ticker: str,
        predicted_direction: str,
        predicted_return: float,
        actual_return: Optional[float] = None,
        signal_type: Optional[SignalType] = None,
    ) -> None:
        self._prediction_history.append({
            "ticker": ticker,
            "predicted_direction": predicted_direction,
            "predicted_return": predicted_return,
            "actual_return": actual_return,
            "signal_type": signal_type.value if signal_type else None,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def record_missed_opportunity(
        self,
        ticker: str,
        direction: str,
        potential_return: float,
        reason_missed: str,
    ) -> None:
        self._missed_opportunities.append({
            "ticker": ticker,
            "direction": direction,
            "potential_return": potential_return,
            "reason_missed": reason_missed,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def update_strategy_weights(
        self,
        daily_results: list[dict],
        lookback_days: int = 30,
    ) -> dict[str, float]:
        """
        Recalibrate strategy weights based on recent performance.

        Uses exponential decay weighting of historical accuracy.
        """
        if not self._prediction_history:
            return self._strategy_weights

        recent = [
            p for p in self._prediction_history
            if p.get("actual_return") is not None
        ][-lookback_days * 50:]

        # Group by signal type
        type_accuracy: dict[str, list[float]] = {}
        for pred in recent:
            stype = pred.get("signal_type", "unknown")
            actual = pred["actual_return"]
            predicted = pred["predicted_return"]
            if stype not in type_accuracy:
                type_accuracy[stype] = []
            # Accuracy: did we get the direction right?
            correct = 1.0 if (actual > 0 and predicted > 0) or (actual < 0 and predicted < 0) else 0.0
            type_accuracy[stype].append(correct)

        # Update weights proportional to accuracy
        total_accuracy = 0.0
        for stype, accuracies in type_accuracy.items():
            if accuracies:
                avg_acc = float(np.mean(accuracies))
                self._strategy_weights[stype] = avg_acc
                total_accuracy += avg_acc

        # Normalise
        if total_accuracy > 0:
            for k in self._strategy_weights:
                self._strategy_weights[k] /= total_accuracy

        return dict(self._strategy_weights)

    def get_missed_opportunities_report(self, n: int = 20) -> list[dict]:
        return self._missed_opportunities[-n:]


# ---------------------------------------------------------------------------
# InvestmentPlatformOrchestrator
# ---------------------------------------------------------------------------


class InvestmentPlatformOrchestrator:
    """
    Central orchestration layer for Metadron Capital's investment platform.
    Coordinates all subsystems into a synchronized investment strategy.

    Daily Workflow:
    1. Data Ingestion (OpenBB Universe) -> Full universe pull at market open
    2. Classification (GICS + Asset Class) -> Categorize all securities
    3. Analysis Pipeline:
       a. Technical scan (all asset classes) -> signals
       b. Fundamental scan (equities, bonds) -> value signals
       c. Macro analysis -> regime detection, factor exposure
       d. Sentiment/News -> catalyst scoring
       e. Distress scan -> special situations
    4. Cube Rotation -> composite opportunity scoring
    5. Multi-Horizon Strategy:
       a. HFT Engine -> intraday alpha capture
       b. Swing trades -> 1-5 day momentum
       c. Medium-term theses -> 1-6 month positions (separate book)
       d. Long-term strategic -> 6+ month macro trades (separate book)
    6. Execution:
       a. HFT orders -> continuous execution
       b. Medium/long-term -> voice execution with written thesis
    7. Reporting:
       a. Real-time P&L dashboard
       b. EOD daily P&L report with attribution
       c. Missed opportunities report
       d. New thesis recommendations (medium/long term)
    8. ML Learning:
       a. Update prediction models with new data
       b. Recalibrate strategy weights based on performance
       c. Store and learn from missed opportunities
    """

    def __init__(
        self,
        nav: Optional[float] = None,
        risk_limits: Optional[RiskLimits] = None,
        cube_weights: Optional[dict[SignalType, float]] = None,
    ) -> None:
        # Dynamic NAV: pull from Alpaca if available
        if nav is None:
            try:
                from engine.execution.alpaca_broker import AlpacaBroker
                import os
                api_key = os.environ.get("ALPACA_API_KEY", "")
                secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
                if api_key and secret_key:
                    broker = AlpacaBroker(initial_cash=0, api_key=api_key, secret_key=secret_key, paper=True)
                    nav = broker.get_nav()
                    logger.info("Orchestrator NAV from Alpaca: $%,.2f", nav)
                else:
                    nav = 1_000_000.0
            except Exception:
                nav = 1_000_000.0
        self.nav = nav
        self.risk_limits = risk_limits or RiskLimits()

        # --- Internal fallback subsystems (used when engine modules unavailable) ---
        self._technical = _TechnicalAnalyzer()
        self._fundamental = _FundamentalAnalyzer()
        self._macro = _MacroAnalyzer()
        self._sentiment = _SentimentAnalyzer()
        self._risk = _RiskManager(self.risk_limits)
        self._cube = _CubeRotation(cube_weights)
        self._execution_internal = _ExecutionEngine()
        self._ml = _MLLearner()

        # --- Engine modules (institutional-grade, the real pipeline) ---
        self.macro_engine = MacroEngine() if MacroEngine else None
        self.metadron_cube = MetadronCube() if MetadronCube else None
        self.security_analyzer = SecurityAnalysisEngine() if SecurityAnalysisEngine else None
        self.contagion_engine = ContagionEngine() if ContagionEngine else None
        self.stat_arb_engine = StatArbEngine() if StatArbEngine else None
        self.social_engine = SocialPredictionEngine() if SocialPredictionEngine else None
        self.distress_engine = DistressedAssetEngine() if DistressedAssetEngine else None
        self.cvr_engine = CVREngine() if CVREngine else None
        self.event_engine = EventDrivenEngine() if EventDrivenEngine else None
        self.fed_plumbing = FedLiquidityPlumbing() if FedLiquidityPlumbing else None
        self.agent_sim = AgentSimEngine() if AgentSimEngine else None

        # ML engines
        self.alpha_optimizer = AlphaOptimizer() if AlphaOptimizer else None
        self.universe_classifier = UniverseClassifier() if UniverseClassifier else None
        self.pattern_engine = PatternRecognitionEngine() if PatternRecognitionEngine else None
        self.deep_learning = DeepLearningEngine() if DeepLearningEngine else None
        self.model_evaluator = ModelEvaluator() if ModelEvaluator else None
        self.model_store = ModelStore() if ModelStore else None

        # Portfolio & Risk
        self.beta_corridor = BetaCorridor() if BetaCorridor else None
        self.monte_carlo_risk = MonteCarloRiskEngine() if MonteCarloRiskEngine else None

        # Execution
        self.execution_engine = ExecutionEngine() if ExecutionEngine else None
        self.options_engine = OptionsEngine() if OptionsEngine else None

        # Learning
        self.learning_loop = LearningLoop() if LearningLoop else None

        # Data
        self.universe_engine = get_engine() if get_engine else None

        # State
        self._universe: list = []
        self._portfolio: list[PortfolioPosition] = []
        self._daily_trades: list[ExecutionResult] = []
        self._analysis_results: Optional[AnalysisResults] = None
        self._current_regime = MarketRegime.RISK_ON
        self._daily_reports: list[DailyReport] = []

        engine_count = sum(1 for x in [
            self.macro_engine, self.metadron_cube, self.security_analyzer,
            self.alpha_optimizer, self.execution_engine, self.options_engine,
            self.monte_carlo_risk, self.learning_loop,
        ] if x is not None)
        logger.info(
            "Platform orchestrator initialized: NAV=$%.2fM, engines=%d/8",
            nav / 1_000_000, engine_count,
        )

    # -------------------------------------------------------------------
    # Daily workflow
    # -------------------------------------------------------------------

    def daily_open_routine(self) -> None:
        """
        Execute the complete daily open routine using the full engine pipeline:

        Pipeline: UniverseEngine → MacroEngine → MetadronCube → SecurityAnalysis
                  → PatternDiscovery → AlphaOptimizer → BetaCorridor
                  → DecisionMatrix → ExecutionEngine → AlpacaBroker

        Fallback: Uses internal analyzers if engine modules are unavailable.
        """
        logger.info("=== DAILY OPEN ROUTINE START ===")
        start = time.monotonic()

        # Step 1: Data ingestion — UniverseEngine
        logger.info("Step 1: Loading universe via UniverseEngine...")
        if self.universe_engine:
            try:
                self._universe = self.universe_engine.get_all_tickers()
                logger.info("Universe loaded: %d securities via UniverseEngine", len(self._universe))
            except Exception as e:
                logger.warning("UniverseEngine failed: %s — falling back", e)
                self._universe = []
        if not self._universe:
            try:
                from openbb_universe import get_full_universe
                snapshot = get_full_universe()
                self._universe = snapshot.securities
                logger.info("Universe loaded: %d securities via openbb_universe", len(self._universe))
            except ImportError:
                logger.warning("No universe source available, using cached")
                self._universe = []

        # Step 2: Macro regime — MacroEngine
        logger.info("Step 2: Macro regime analysis via MacroEngine...")
        macro_snapshot = None
        if self.macro_engine:
            try:
                macro_snapshot = self.macro_engine.classify_regime()
                regime_map = {
                    "BULL": MarketRegime.RISK_ON, "BEAR": MarketRegime.RISK_OFF,
                    "TRANSITION": MarketRegime.TRANSITION, "STRESS": MarketRegime.CRISIS,
                    "CRASH": MarketRegime.CRISIS,
                }
                if hasattr(macro_snapshot, 'regime'):
                    self._current_regime = regime_map.get(str(macro_snapshot.regime), MarketRegime.TRANSITION)
                logger.info("Macro regime: %s", self._current_regime)
            except Exception as e:
                logger.warning("MacroEngine failed: %s — using fallback", e)

        if macro_snapshot is None:
            # Fallback to internal macro analyzer
            macro_data = {}
            self._current_regime = self._macro.detect_regime(macro_data)
            logger.info("Macro regime (fallback): %s", self._current_regime)

        # Step 3: MetadronCube intelligence tensor
        cube_output = None
        if self.metadron_cube and macro_snapshot:
            try:
                cube_output = self.metadron_cube.compute(macro_snapshot)
                logger.info("MetadronCube: regime=%s", getattr(cube_output, 'regime', 'unknown'))
            except Exception as e:
                logger.warning("MetadronCube failed: %s", e)

        # Step 4: Security analysis — Graham-Dodd-Klarman
        security_results = None
        if self.security_analyzer:
            try:
                security_results = self.security_analyzer.analyze(
                    tickers=self._universe[:50],  # top 50 for speed
                    macro_snapshot=macro_snapshot,
                    cube_output=cube_output,
                )
                logger.info("SecurityAnalysis: %d results", len(security_results) if security_results else 0)
            except Exception as e:
                logger.warning("SecurityAnalysis failed: %s", e)

        # Step 5: Additional signal engines
        signals = {}

        # Contagion risk
        if self.contagion_engine:
            try:
                contagion_result = self.contagion_engine.run_scenario("CORRELATION_SPIKE")
                signals['contagion'] = contagion_result
            except Exception as e:
                logger.debug("ContagionEngine: %s", e)

        # Stat arb pairs
        if self.stat_arb_engine:
            try:
                stat_arb_signals = self.stat_arb_engine.scan()
                signals['stat_arb'] = stat_arb_signals
            except Exception as e:
                logger.debug("StatArbEngine: %s", e)

        # Social prediction (MiroFish)
        if self.social_engine:
            try:
                social_snap = self.social_engine.analyze()
                signals['social'] = social_snap
            except Exception as e:
                logger.debug("SocialPrediction: %s", e)

        # Distressed assets
        if self.distress_engine:
            try:
                distress_results = self.distress_engine.analyze()
                signals['distress'] = distress_results
            except Exception as e:
                logger.debug("DistressedAsset: %s", e)

        # CVR analysis
        if self.cvr_engine:
            try:
                cvr_results = self.cvr_engine.analyze()
                signals['cvr'] = cvr_results
            except Exception as e:
                logger.debug("CVREngine: %s", e)

        # Event driven
        if self.event_engine:
            try:
                event_signals = self.event_engine.analyze()
                signals['event'] = event_signals
            except Exception as e:
                logger.debug("EventDriven: %s", e)

        # Fed plumbing
        if self.fed_plumbing:
            try:
                fed_state = self.fed_plumbing.compute_plumbing_state()
                signals['fed_plumbing'] = fed_state
            except Exception as e:
                logger.debug("FedPlumbing: %s", e)

        # Agent simulation
        if self.agent_sim:
            try:
                agent_signals = self.agent_sim.simulate(self._universe[:20])
                signals['agent_sim'] = agent_signals
            except Exception as e:
                logger.debug("AgentSim: %s", e)

        # Step 6: ML pattern recognition
        if self.pattern_engine:
            try:
                patterns = self.pattern_engine.scan(self._universe[:30])
                signals['patterns'] = patterns
            except Exception as e:
                logger.debug("PatternEngine: %s", e)

        # Step 7: Alpha optimization
        alpha_output = None
        if self.alpha_optimizer:
            try:
                alpha_output = self.alpha_optimizer.optimize(
                    tickers=self._universe,
                    signals=signals,
                    macro_snapshot=macro_snapshot,
                )
                logger.info("AlphaOptimizer: %d signals generated",
                           len(alpha_output) if alpha_output else 0)
            except Exception as e:
                logger.warning("AlphaOptimizer failed: %s", e)

        # Step 8: Beta corridor management
        if self.beta_corridor:
            try:
                beta_action = self.beta_corridor.rebalance()
                logger.info("BetaCorridor: action=%s", getattr(beta_action, 'action', 'HOLD'))
            except Exception as e:
                logger.debug("BetaCorridor: %s", e)

        # Step 9: Monte Carlo risk check
        if self.monte_carlo_risk:
            try:
                risk_report = self.monte_carlo_risk.simulate_portfolio(
                    tickers=self._universe[:30],
                    num_simulations=5000,
                )
                logger.info("MonteCarloRisk: completed %d sims", 5000)
                signals['monte_carlo_risk'] = risk_report
            except Exception as e:
                logger.debug("MonteCarloRisk: %s", e)

        # Step 10: Options pricing (if options engine available)
        if self.options_engine:
            try:
                options_signals = self.options_engine.generate_signals()
                signals['options'] = options_signals
                logger.info("OptionsEngine: signals generated")
            except Exception as e:
                logger.debug("OptionsEngine: %s", e)

        # Step 11: Also run internal analysis pipeline (for trade thesis generation)
        logger.info("Step 11: Running internal analysis pipeline...")
        analysis = self.run_analysis_pipeline(self._universe)
        self._analysis_results = analysis

        # Step 12: Generate trades
        logger.info("Step 12: Generating trade theses...")
        trades = self.generate_trades(analysis, self._portfolio)

        # Step 13: Execute HFT trades via ExecutionEngine
        hft_trades = [t for t in trades if t.horizon == TradingHorizon.HFT]
        if hft_trades:
            logger.info("Step 13: Executing %d HFT trades...", len(hft_trades))
            if self.execution_engine:
                try:
                    # Use the real execution engine
                    self.execution_engine.execute_trades(hft_trades)
                except Exception as e:
                    logger.warning("ExecutionEngine failed: %s — using internal", e)
                    results = self.execute_hft_trades(hft_trades)
                    self._daily_trades.extend(results)
            else:
                results = self.execute_hft_trades(hft_trades)
                self._daily_trades.extend(results)

        elapsed = time.monotonic() - start
        logger.info("=== DAILY OPEN ROUTINE COMPLETE (%.1fs) ===", elapsed)

    def run_analysis_pipeline(
        self,
        universe: list,
        price_cache: Optional[dict[str, pd.DataFrame]] = None,
        fundamentals_cache: Optional[dict[str, dict]] = None,
        macro_data: Optional[dict] = None,
        news_data: Optional[list[dict]] = None,
    ) -> AnalysisResults:
        """
        Run the full analysis pipeline across all dimensions.

        Parameters
        ----------
        universe : list
            Full security universe.
        price_cache : dict, optional
            Pre-fetched price data.
        fundamentals_cache : dict, optional
            Pre-fetched fundamentals.
        macro_data : dict, optional
            Macro indicators for regime detection.
        news_data : list[dict], optional
            News items for sentiment analysis.

        Returns
        -------
        AnalysisResults
        """
        if price_cache is None:
            price_cache = {}
        if fundamentals_cache is None:
            fundamentals_cache = {}
        if macro_data is None:
            macro_data = {}
        if news_data is None:
            news_data = []

        results = AnalysisResults()

        # 3a: Technical scan
        logger.info("  3a: Technical scan...")
        results.technical_signals = self._technical.scan(universe, price_cache)

        # 3b: Fundamental scan
        logger.info("  3b: Fundamental scan...")
        results.fundamental_signals = self._fundamental.scan(universe, fundamentals_cache)

        # 3c: Macro analysis
        logger.info("  3c: Macro analysis...")
        results.regime = self._macro.detect_regime(macro_data)
        self._current_regime = results.regime
        results.macro_signals = self._macro.generate_macro_signals(results.regime, {})
        results.factor_exposures = self._macro.calculate_factor_exposures(self._portfolio, {})

        # 3d: Sentiment
        logger.info("  3d: Sentiment analysis...")
        results.sentiment_signals = self._sentiment.analyze(news_data, universe)

        # 3e: Distress scan (uses separate scanner module)
        logger.info("  3e: Distress scan...")
        try:
            from distress_scanner import DistressScanner
            scanner = DistressScanner()
            equities = [s for s in universe if hasattr(s, "asset_class") and s.asset_class.value == "equity"]
            distressed = scanner.scan_distressed_equities(equities, fundamentals_cache)
            for d in distressed:
                results.distress_signals.append(AnalysisSignal(
                    ticker=d.ticker,
                    signal_type=SignalType.DISTRESS,
                    direction="short" if d.distress_composite > 70 else "long",
                    strength=d.distress_composite / 100,
                    confidence=0.6,
                    horizon=TradingHorizon.MEDIUM_TERM,
                    metadata={"z_score": d.z_score, "composite": d.distress_composite},
                ))
        except ImportError:
            logger.warning("distress_scanner not available, skipping")

        # 4: Cube rotation
        logger.info("  4: Cube rotation scoring...")
        all_signals = (
            results.technical_signals
            + results.fundamental_signals
            + results.macro_signals
            + results.sentiment_signals
            + results.distress_signals
        )
        results.composite_opportunities = self._cube.score_universe(all_signals)

        logger.info(
            "Analysis pipeline complete: %d opportunities scored, regime=%s",
            len(results.composite_opportunities),
            results.regime.value,
        )
        return results

    def generate_trades(
        self,
        analysis: AnalysisResults,
        current_portfolio: list[PortfolioPosition],
    ) -> list[TradeThesis]:
        """
        Convert analysis results into actionable trade theses.

        Applies position sizing, risk checks, and execution method selection.
        """
        trades: list[TradeThesis] = []

        for opp in analysis.composite_opportunities:
            if abs(opp["composite_score"]) < 0.05:
                continue  # skip weak signals

            ticker = opp["ticker"]
            direction = opp["direction"]
            horizon = opp["horizon"]
            score = opp["composite_score"]

            # Position sizing
            vol_estimate = 0.20  # default 20% annualised vol
            for sig in opp.get("signals", []):
                if "atr" in sig.metadata:
                    vol_estimate = sig.metadata["atr"]
                    break

            size = self._risk.position_size(
                signal_strength=abs(score),
                confidence=min(1.0, abs(score) * 3),
                volatility=vol_estimate,
                nav=self.nav,
            )

            if size < 1000:
                continue  # skip tiny trades

            # Entry / target / stop (heuristic)
            entry = 100.0  # placeholder; in production use live price
            if direction == "long":
                target = entry * (1 + abs(score) * 0.5)
                stop = entry * (1 - abs(score) * 0.2)
            else:
                target = entry * (1 - abs(score) * 0.5)
                stop = entry * (1 + abs(score) * 0.2)

            rr = abs(target - entry) / abs(stop - entry) if abs(stop - entry) > 0 else 0

            # Execution method based on horizon
            if horizon == TradingHorizon.HFT:
                exec_method = ExecutionMethod.ALGORITHMIC
                urgency = "immediate"
            elif horizon == TradingHorizon.SWING:
                exec_method = ExecutionMethod.ALGORITHMIC
                urgency = "high"
            elif horizon == TradingHorizon.MEDIUM_TERM:
                exec_method = ExecutionMethod.VOICE
                urgency = "normal"
            else:
                exec_method = ExecutionMethod.VOICE
                urgency = "normal"

            trade = TradeThesis(
                ticker=ticker,
                direction=direction,
                horizon=horizon,
                entry_price=entry,
                target_price=round(target, 4),
                stop_loss=round(stop, 4),
                position_size=round(size, 2),
                notional=round(size * entry, 2),
                signals=opp.get("signals", []),
                composite_score=score,
                risk_reward_ratio=round(rr, 2),
                thesis_text=self._generate_thesis_text(ticker, direction, opp),
                execution_method=exec_method,
                urgency=urgency,
            )

            # Risk check
            approved, reason = self._risk.check_trade(trade, current_portfolio, self.nav)
            if approved:
                trades.append(trade)
            else:
                logger.info("Trade rejected (%s): %s %s - %s", ticker, direction, horizon.value, reason)

        logger.info("Generated %d trade theses", len(trades))
        return trades

    def _generate_thesis_text(self, ticker: str, direction: str, opp: dict) -> str:
        """Generate a written thesis for medium/long-term trades."""
        signal_summary = []
        for sig in opp.get("signals", []):
            signal_summary.append(
                f"  - {sig.signal_type.value}: {sig.direction} "
                f"(strength={sig.strength:.3f}, confidence={sig.confidence:.3f})"
            )

        text = (
            f"Trade Thesis: {direction.upper()} {ticker}\n"
            f"Composite Score: {opp['composite_score']:.4f}\n"
            f"Horizon: {opp['horizon'].value}\n"
            f"Regime: {self._current_regime.value}\n"
            f"Signal Count: {opp['signal_count']}\n\n"
            f"Signal Breakdown:\n" + "\n".join(signal_summary)
        )
        return text

    def execute_hft_trades(self, trades: list[TradeThesis]) -> list[ExecutionResult]:
        """
        Execute HFT trades via algorithmic execution.

        Uses TWAP/VWAP algorithms for minimal market impact.
        """
        results: list[ExecutionResult] = []
        for trade in trades:
            result = self._execution.execute(trade, ExecutionMethod.ALGORITHMIC)
            results.append(result)

            # Record for ML
            self._ml.record_prediction(
                ticker=trade.ticker,
                predicted_direction=trade.direction,
                predicted_return=trade.composite_score,
                signal_type=SignalType.QUANTITATIVE,
            )

        logger.info("Executed %d HFT trades", len(results))
        return results

    def generate_separate_lt_recommendations(
        self,
        analysis: AnalysisResults,
    ) -> list[TradeThesis]:
        """
        Generate medium/long-term recommendations as a SEPARATE book.

        These trades require written theses and voice execution.
        Not mixed with HFT/swing books.
        """
        lt_trades: list[TradeThesis] = []

        for opp in analysis.composite_opportunities:
            horizon = opp["horizon"]
            if horizon not in (TradingHorizon.MEDIUM_TERM, TradingHorizon.LONG_TERM):
                continue

            if abs(opp["composite_score"]) < 0.15:
                continue  # higher threshold for long-term

            ticker = opp["ticker"]
            direction = opp["direction"]
            score = opp["composite_score"]

            # Larger position sizes for conviction long-term trades
            size = self._risk.position_size(
                signal_strength=abs(score),
                confidence=min(1.0, abs(score) * 2),
                volatility=0.25,
                nav=self.nav,
                max_loss_pct=0.02,
            )

            entry = 100.0
            if direction == "long":
                target = entry * (1 + abs(score))
                stop = entry * (1 - abs(score) * 0.3)
            else:
                target = entry * (1 - abs(score))
                stop = entry * (1 + abs(score) * 0.3)

            rr = abs(target - entry) / abs(stop - entry) if abs(stop - entry) > 0 else 0

            trade = TradeThesis(
                ticker=ticker,
                direction=direction,
                horizon=horizon,
                entry_price=entry,
                target_price=round(target, 4),
                stop_loss=round(stop, 4),
                position_size=round(size, 2),
                notional=round(size * entry, 2),
                signals=opp.get("signals", []),
                composite_score=score,
                risk_reward_ratio=round(rr, 2),
                thesis_text=self._generate_thesis_text(ticker, direction, opp),
                execution_method=ExecutionMethod.VOICE,
                urgency="normal",
            )
            lt_trades.append(trade)

        logger.info("Generated %d long-term recommendations", len(lt_trades))
        return lt_trades

    def end_of_day_routine(self) -> DailyReport:
        """
        Execute end-of-day routine:
        1. Calculate daily P&L with attribution
        2. Generate missed opportunities report
        3. Generate new thesis recommendations
        4. Produce the daily report
        """
        logger.info("=== END OF DAY ROUTINE ===")
        today = datetime.utcnow().strftime("%Y-%m-%d")

        # P&L calculation
        total_pnl = 0.0
        pnl_by_book: dict[str, float] = {
            "hft_book": 0.0,
            "swing_book": 0.0,
            "medium_book": 0.0,
            "long_book": 0.0,
        }
        pnl_by_sector: dict[str, float] = {}

        for pos in self._portfolio:
            pnl = pos.unrealised_pnl + pos.realised_pnl
            total_pnl += pnl
            pnl_by_book[pos.book] = pnl_by_book.get(pos.book, 0.0) + pnl

        # Trade statistics
        n_trades = len(self._daily_trades)
        filled = [t for t in self._daily_trades if t.status == "filled"]
        fill_rate = len(filled) / n_trades if n_trades > 0 else 0.0
        avg_slippage = (
            float(np.mean([t.slippage_bps for t in filled])) if filled else 0.0
        )

        # Win rate
        winning = [t for t in filled if t.slippage_bps < 5]  # simplified
        win_rate = len(winning) / len(filled) if filled else 0.0

        # Exposure
        gross = sum(abs(p.quantity * p.current_price) for p in self._portfolio)
        long_val = sum(p.quantity * p.current_price for p in self._portfolio if p.direction == "long")
        short_val = sum(abs(p.quantity * p.current_price) for p in self._portfolio if p.direction == "short")
        net = long_val - short_val

        # Missed opportunities
        missed = self._ml.get_missed_opportunities_report(20)

        # New long-term recommendations
        new_recs: list[TradeThesis] = []
        if self._analysis_results:
            new_recs = self.generate_separate_lt_recommendations(self._analysis_results)

        # Attribution
        attribution = {
            "technical": sum(
                pnl_by_book.get("hft_book", 0) + pnl_by_book.get("swing_book", 0)
                for _ in [1]
            ),
            "fundamental": pnl_by_book.get("medium_book", 0.0),
            "macro": pnl_by_book.get("long_book", 0.0),
        }

        report = DailyReport(
            date=today,
            total_pnl=round(total_pnl, 2),
            pnl_by_book={k: round(v, 2) for k, v in pnl_by_book.items()},
            pnl_by_sector={k: round(v, 2) for k, v in pnl_by_sector.items()},
            trades_executed=n_trades,
            fill_rate=round(fill_rate, 4),
            avg_slippage_bps=round(avg_slippage, 2),
            gross_exposure=round(gross, 2),
            net_exposure=round(net, 2),
            sharpe_estimate=0.0,
            max_drawdown=0.0,
            win_rate=round(win_rate, 4),
            missed_opportunities=missed,
            new_recommendations=new_recs,
            attribution=attribution,
        )

        self._daily_reports.append(report)
        logger.info(
            "EOD Report: P&L=$%.2f, Trades=%d, Fill=%.1f%%, Slippage=%.1fbps",
            total_pnl, n_trades, fill_rate * 100, avg_slippage,
        )
        return report

    def weekend_review(self) -> WeeklyReport:
        """
        Generate a comprehensive weekly review.

        Aggregates daily reports, identifies best/worst trades,
        evaluates strategy performance, and generates recommendations.
        """
        logger.info("=== WEEKEND REVIEW ===")
        week_ending = datetime.utcnow().strftime("%Y-%m-%d")

        # Get this week's daily reports
        recent_reports = self._daily_reports[-5:]

        weekly_pnl = sum(r.total_pnl for r in recent_reports)

        # Strategy performance by book
        strategy_perf: dict[str, float] = {}
        for r in recent_reports:
            for book, pnl in r.pnl_by_book.items():
                strategy_perf[book] = strategy_perf.get(book, 0.0) + pnl

        # Model accuracy from ML learner
        model_accuracy = self._ml.update_strategy_weights([])

        # Recommendations
        recommendations: list[str] = []
        if weekly_pnl < 0:
            recommendations.append("Review position sizing - negative week")
        if strategy_perf.get("hft_book", 0) < 0:
            recommendations.append("HFT strategy underperforming - review signal thresholds")

        for book, pnl in strategy_perf.items():
            if pnl < -self.nav * 0.005:
                recommendations.append(f"Consider reducing {book} exposure")

        report = WeeklyReport(
            week_ending=week_ending,
            daily_reports=recent_reports,
            weekly_pnl=round(weekly_pnl, 2),
            best_trades=[],
            worst_trades=[],
            strategy_performance={k: round(v, 2) for k, v in strategy_perf.items()},
            model_accuracy=model_accuracy,
            regime_changes=[],
            recommendations=recommendations,
        )

        logger.info("Weekly Review: P&L=$%.2f", weekly_pnl)
        return report

    def learn_from_today(
        self,
        results: list[ExecutionResult],
        missed: list[dict],
    ) -> None:
        """
        ML learning step: update models with today's results.

        1. Record actual returns for executed trades
        2. Record missed opportunities for future learning
        3. Recalibrate strategy weights
        """
        logger.info("ML learning step: %d results, %d missed", len(results), len(missed))

        for r in results:
            self._ml.record_prediction(
                ticker=r.ticker,
                predicted_direction=r.direction,
                predicted_return=0.0,
                actual_return=None,
                signal_type=None,
            )

        for m in missed:
            self._ml.record_missed_opportunity(
                ticker=m.get("ticker", ""),
                direction=m.get("direction", ""),
                potential_return=m.get("potential_return", 0.0),
                reason_missed=m.get("reason", "unknown"),
            )

        # Recalibrate
        new_weights = self._ml.update_strategy_weights([])
        if new_weights:
            logger.info("Updated strategy weights: %s", new_weights)

    # -------------------------------------------------------------------
    # Utility / status methods
    # -------------------------------------------------------------------

    @property
    def current_regime(self) -> MarketRegime:
        return self._current_regime

    @property
    def portfolio_summary(self) -> dict:
        gross = sum(abs(p.quantity * p.current_price) for p in self._portfolio)
        long_val = sum(p.quantity * p.current_price for p in self._portfolio if p.direction == "long")
        short_val = sum(abs(p.quantity * p.current_price) for p in self._portfolio if p.direction == "short")

        return {
            "positions": len(self._portfolio),
            "gross_exposure": round(gross, 2),
            "net_exposure": round(long_val - short_val, 2),
            "long_exposure": round(long_val, 2),
            "short_exposure": round(short_val, 2),
            "nav": self.nav,
            "regime": self._current_regime.value,
        }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    orchestrator = InvestmentPlatformOrchestrator(nav=100_000_000)
    print(f"Portfolio Summary: {orchestrator.portfolio_summary}")
    print(f"Current Regime: {orchestrator.current_regime.value}")

    # Demo: run analysis with empty caches
    analysis = orchestrator.run_analysis_pipeline(
        universe=[],
        macro_data={"vix": 18, "yield_curve_slope": 0.5, "credit_spread_bps": 140, "pmi": 53},
    )
    print(f"Regime detected: {analysis.regime.value}")
    print(f"Macro signals: {len(analysis.macro_signals)}")
    print(f"Composite opportunities: {len(analysis.composite_opportunities)}")
