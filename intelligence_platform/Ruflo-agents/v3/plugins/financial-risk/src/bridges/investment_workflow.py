"""
Investment Workflow Agent for Ruflo Agent Framework.

Orchestrates the investment workflow through coordinated agent swarms:
1. Universe data fetch agent (parallel across asset classes)
2. Analysis agents (technical, fundamental, macro, sentiment)
3. Strategy agents (HFT scanner, swing scanner, macro thesis)
4. Risk management agent (position sizing, exposure limits)
5. Execution agent (order routing, slippage control)
6. Reporting agent (P&L, attribution, missed opportunities)

Usage:
    from investment_workflow import InvestmentWorkflowAgent
    agent = InvestmentWorkflowAgent()
    report = await agent.run_daily_workflow()
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent task primitives
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class AgentType(str, Enum):
    DATA_FETCH = "data_fetch"
    TECHNICAL_ANALYSIS = "technical_analysis"
    FUNDAMENTAL_ANALYSIS = "fundamental_analysis"
    MACRO_ANALYSIS = "macro_analysis"
    SENTIMENT_ANALYSIS = "sentiment_analysis"
    DISTRESS_ANALYSIS = "distress_analysis"
    HFT_STRATEGY = "hft_strategy"
    SWING_STRATEGY = "swing_strategy"
    MACRO_STRATEGY = "macro_strategy"
    RISK_MANAGEMENT = "risk_management"
    EXECUTION = "execution"
    REPORTING = "reporting"
    ML_LEARNING = "ml_learning"


@dataclass
class AgentTask:
    """A single task to be executed by an agent."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    agent_type: AgentType = AgentType.DATA_FETCH
    name: str = ""
    description: str = ""
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    dependencies: list[str] = field(default_factory=list)
    input_data: dict = field(default_factory=dict)
    output_data: dict = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    execution_time_ms: float = 0.0
    retries: int = 0
    max_retries: int = 3


@dataclass
class TradeThesis:
    """A trade thesis produced by strategy agents."""

    ticker: str
    direction: str  # long, short
    horizon: str  # hft, swing, medium_term, long_term
    entry_price: float
    target_price: float
    stop_loss: float
    position_size: float
    notional: float
    composite_score: float
    risk_reward: float
    thesis_text: str = ""
    signals: list[dict] = field(default_factory=list)
    execution_method: str = "algorithmic"  # algorithmic, voice, dma, dark_pool
    urgency: str = "normal"


@dataclass
class ExecutionResult:
    """Result of a trade execution."""

    trade_id: str
    ticker: str
    direction: str
    filled_size: float
    avg_price: float
    slippage_bps: float
    commission: float
    venue: str
    status: str  # filled, partial, rejected
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class DailyReport:
    """Daily performance and activity report."""

    date: str
    total_pnl: float
    pnl_by_strategy: dict[str, float] = field(default_factory=dict)
    trades_executed: int = 0
    fill_rate: float = 0.0
    avg_slippage_bps: float = 0.0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0
    missed_opportunities: list[dict] = field(default_factory=list)
    new_recommendations: list[TradeThesis] = field(default_factory=list)
    agent_performance: dict[str, dict] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class SwarmResult:
    """Result from a group of agents running in parallel."""

    tasks: list[AgentTask]
    total_time_ms: float
    success_count: int
    failure_count: int
    aggregated_output: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Agent registry and base agent
# ---------------------------------------------------------------------------


class BaseAgent:
    """Base class for all investment workflow agents."""

    def __init__(self, agent_type: AgentType, name: str = "") -> None:
        self.agent_type = agent_type
        self.name = name or agent_type.value
        self._metrics: dict[str, Any] = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time_ms": 0.0,
        }

    async def execute(self, task: AgentTask) -> AgentTask:
        """Execute a task and return the updated task."""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()
        start = time.monotonic()

        try:
            result = await self._run(task)
            task.output_data = result
            task.status = TaskStatus.COMPLETED
            self._metrics["tasks_completed"] += 1
        except Exception as exc:
            task.error = str(exc)
            task.status = TaskStatus.FAILED
            self._metrics["tasks_failed"] += 1
            logger.error("Agent %s task %s failed: %s", self.name, task.task_id, exc)

            # Retry logic
            if task.retries < task.max_retries:
                task.retries += 1
                logger.info("Retrying task %s (attempt %d/%d)", task.task_id, task.retries, task.max_retries)
                return await self.execute(task)

        elapsed = (time.monotonic() - start) * 1000
        task.execution_time_ms = elapsed
        task.completed_at = datetime.utcnow()
        self._metrics["total_execution_time_ms"] += elapsed

        return task

    async def _run(self, task: AgentTask) -> dict:
        """Override in subclasses. Returns output data dict."""
        raise NotImplementedError

    @property
    def metrics(self) -> dict:
        return dict(self._metrics)


# ---------------------------------------------------------------------------
# Specialised agents
# ---------------------------------------------------------------------------


class DataFetchAgent(BaseAgent):
    """Fetches universe data from OpenBB in parallel across asset classes."""

    def __init__(self) -> None:
        super().__init__(AgentType.DATA_FETCH, "UniverseDataFetch")

    async def _run(self, task: AgentTask) -> dict:
        asset_class = task.input_data.get("asset_class", "equity")
        logger.info("DataFetch: fetching %s universe", asset_class)

        # In production: call openbb_data module
        try:
            if asset_class == "equity":
                from openbb_data import get_equity_universe
                securities = get_equity_universe()
            elif asset_class == "fixed_income":
                from openbb_data import get_bond_universe
                securities = get_bond_universe()
            elif asset_class == "commodity":
                from openbb_data import get_commodity_universe
                securities = get_commodity_universe()
            elif asset_class == "fx":
                from openbb_data import get_fx_universe
                securities = get_fx_universe()
            elif asset_class == "crypto":
                from openbb_data import get_crypto_universe
                securities = get_crypto_universe()
            else:
                from openbb_data import get_full_universe
                snapshot = get_full_universe()
                securities = snapshot.securities

            tickers = [s.ticker for s in securities]
            return {
                "asset_class": asset_class,
                "count": len(tickers),
                "tickers": tickers,
                "securities": securities,
            }
        except ImportError:
            logger.warning("openbb_data not available, returning empty universe")
            return {"asset_class": asset_class, "count": 0, "tickers": [], "securities": []}


class TechnicalAnalysisAgent(BaseAgent):
    """Runs technical analysis across the universe."""

    def __init__(self) -> None:
        super().__init__(AgentType.TECHNICAL_ANALYSIS, "TechnicalAnalysis")

    async def _run(self, task: AgentTask) -> dict:
        universe = task.input_data.get("universe", [])
        price_cache = task.input_data.get("price_cache", {})
        logger.info("TechAnalysis: scanning %d securities", len(universe))

        signals: list[dict] = []

        for security in universe:
            ticker = security.ticker if hasattr(security, "ticker") else str(security)
            prices = price_cache.get(ticker)
            if prices is None:
                continue

            try:
                import numpy as np
                close = prices["Close"].values if "Close" in prices.columns else prices.iloc[:, 3].values
                if len(close) < 50:
                    continue

                # RSI(14)
                deltas = np.diff(close)
                gains = np.where(deltas > 0, deltas, 0.0)
                losses = np.where(deltas < 0, -deltas, 0.0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss)) if avg_loss > 0 else 100.0

                # SMA crossover
                sma20 = float(np.mean(close[-20:]))
                sma50 = float(np.mean(close[-50:]))

                score = 0.0
                if rsi < 30:
                    score += 0.5
                elif rsi > 70:
                    score -= 0.5
                if sma20 > sma50:
                    score += 0.3
                else:
                    score -= 0.3

                direction = "long" if score > 0.1 else ("short" if score < -0.1 else "neutral")
                signals.append({
                    "ticker": ticker,
                    "type": "technical",
                    "direction": direction,
                    "strength": round(score, 4),
                    "rsi": round(rsi, 2),
                    "sma20": round(sma20, 4),
                    "sma50": round(sma50, 4),
                })
            except Exception as exc:
                logger.debug("Tech analysis failed for %s: %s", ticker, exc)

        return {"signals": signals, "count": len(signals)}


class FundamentalAnalysisAgent(BaseAgent):
    """Runs fundamental analysis on equities and bonds."""

    def __init__(self) -> None:
        super().__init__(AgentType.FUNDAMENTAL_ANALYSIS, "FundamentalAnalysis")

    async def _run(self, task: AgentTask) -> dict:
        universe = task.input_data.get("universe", [])
        fundamentals_cache = task.input_data.get("fundamentals_cache", {})
        logger.info("FundAnalysis: scanning %d securities", len(universe))

        signals: list[dict] = []

        for security in universe:
            ticker = security.ticker if hasattr(security, "ticker") else str(security)
            data = fundamentals_cache.get(ticker, {})
            ratios = data.get("ratios")
            if ratios is None:
                continue

            try:
                row = ratios.iloc[0] if hasattr(ratios, "iloc") else ratios
                pe = float(row.get("pe_ratio", row.get("peRatio", 0)) or 0)
                roe = float(row.get("roe", row.get("returnOnEquity", 0)) or 0)

                score = 0.0
                if 0 < pe < 15:
                    score += 0.4
                elif pe > 30:
                    score -= 0.3
                if roe > 0.15:
                    score += 0.3
                elif roe < 0:
                    score -= 0.4

                direction = "long" if score > 0.2 else ("short" if score < -0.2 else "neutral")
                signals.append({
                    "ticker": ticker,
                    "type": "fundamental",
                    "direction": direction,
                    "strength": round(score, 4),
                    "pe": pe,
                    "roe": roe,
                })
            except Exception:
                pass

        return {"signals": signals, "count": len(signals)}


class MacroAnalysisAgent(BaseAgent):
    """Macro regime detection and factor analysis."""

    def __init__(self) -> None:
        super().__init__(AgentType.MACRO_ANALYSIS, "MacroAnalysis")

    async def _run(self, task: AgentTask) -> dict:
        macro_data = task.input_data.get("macro_data", {})
        logger.info("MacroAnalysis: detecting regime")

        vix = macro_data.get("vix", 20)
        slope = macro_data.get("yield_curve_slope", 0.5)
        spread = macro_data.get("credit_spread_bps", 150)
        pmi = macro_data.get("pmi", 52)

        risk_score = 0
        if vix < 15:
            risk_score += 2
        elif vix > 30:
            risk_score -= 3
        if slope < 0:
            risk_score -= 2
        elif slope > 1:
            risk_score += 1
        if spread > 300:
            risk_score -= 2
        if pmi > 55:
            risk_score += 1
        elif pmi < 48:
            risk_score -= 2

        if risk_score >= 4:
            regime = "euphoria"
        elif risk_score >= 1:
            regime = "risk_on"
        elif risk_score >= -2:
            regime = "transition"
        elif risk_score >= -4:
            regime = "risk_off"
        else:
            regime = "crisis"

        # Generate macro signals
        regime_biases = {
            "risk_on": {"SPY": 0.5, "HYG": 0.3, "TLT": -0.2},
            "risk_off": {"SPY": -0.5, "TLT": 0.5, "GLD": 0.4},
            "crisis": {"SPY": -0.8, "TLT": 0.8, "GLD": 0.7},
            "euphoria": {"SPY": 0.3, "HYG": 0.2, "TLT": -0.3},
            "transition": {},
        }

        signals = []
        biases = regime_biases.get(regime, {})
        for ticker, bias in biases.items():
            signals.append({
                "ticker": ticker,
                "type": "macro",
                "direction": "long" if bias > 0 else "short",
                "strength": bias,
                "regime": regime,
            })

        return {"regime": regime, "risk_score": risk_score, "signals": signals}


class SentimentAnalysisAgent(BaseAgent):
    """News and sentiment analysis agent."""

    def __init__(self) -> None:
        super().__init__(AgentType.SENTIMENT_ANALYSIS, "SentimentAnalysis")

    async def _run(self, task: AgentTask) -> dict:
        news_data = task.input_data.get("news_data", [])
        logger.info("SentimentAnalysis: processing %d news items", len(news_data))

        positive = {"beat", "upgrade", "growth", "record", "outperform", "raised", "bullish"}
        negative = {"miss", "downgrade", "loss", "bankruptcy", "default", "warning", "bearish"}

        ticker_scores: dict[str, list[float]] = {}

        for item in news_data:
            ticker = item.get("ticker", "")
            text = (item.get("headline", "") + " " + item.get("body", "")).lower()

            pos = sum(1 for w in positive if w in text)
            neg = sum(1 for w in negative if w in text)
            total = pos + neg
            if total == 0 or not ticker:
                continue

            score = (pos - neg) / total
            if ticker not in ticker_scores:
                ticker_scores[ticker] = []
            ticker_scores[ticker].append(score)

        signals = []
        import numpy as np
        for ticker, scores in ticker_scores.items():
            avg = float(np.mean(scores))
            signals.append({
                "ticker": ticker,
                "type": "sentiment",
                "direction": "long" if avg > 0.1 else ("short" if avg < -0.1 else "neutral"),
                "strength": round(avg, 4),
                "news_count": len(scores),
            })

        return {"signals": signals, "count": len(signals)}


class DistressAnalysisAgent(BaseAgent):
    """Distress scanning agent using Z-score, Merton, F-score."""

    def __init__(self) -> None:
        super().__init__(AgentType.DISTRESS_ANALYSIS, "DistressAnalysis")

    async def _run(self, task: AgentTask) -> dict:
        universe = task.input_data.get("universe", [])
        fundamentals_cache = task.input_data.get("fundamentals_cache", {})
        logger.info("DistressAnalysis: scanning %d securities", len(universe))

        try:
            from distress_scanner import DistressScanner
            scanner = DistressScanner()
            distressed = scanner.scan_distressed_equities(universe, fundamentals_cache)
            return {
                "distressed_count": len(distressed),
                "distressed": [
                    {
                        "ticker": d.ticker,
                        "composite": d.distress_composite,
                        "z_score": d.z_score.z_score if d.z_score else None,
                        "merton_pd": d.merton.probability_of_default if d.merton else None,
                        "f_score": d.f_score.f_score if d.f_score else None,
                    }
                    for d in distressed
                ],
            }
        except ImportError:
            logger.warning("distress_scanner not available")
            return {"distressed_count": 0, "distressed": []}


class HFTStrategyAgent(BaseAgent):
    """HFT (intraday alpha capture) strategy agent."""

    def __init__(self) -> None:
        super().__init__(AgentType.HFT_STRATEGY, "HFTStrategy")

    async def _run(self, task: AgentTask) -> dict:
        signals = task.input_data.get("signals", [])
        logger.info("HFTStrategy: evaluating %d signals", len(signals))

        trades: list[dict] = []
        for sig in signals:
            if abs(sig.get("strength", 0)) < 0.3:
                continue
            if sig.get("type") != "technical":
                continue

            trades.append({
                "ticker": sig["ticker"],
                "direction": sig["direction"],
                "horizon": "hft",
                "strength": sig["strength"],
                "execution_method": "algorithmic",
                "urgency": "immediate",
            })

        return {"trades": trades, "count": len(trades)}


class SwingStrategyAgent(BaseAgent):
    """Swing trading (1-5 day momentum) strategy agent."""

    def __init__(self) -> None:
        super().__init__(AgentType.SWING_STRATEGY, "SwingStrategy")

    async def _run(self, task: AgentTask) -> dict:
        signals = task.input_data.get("signals", [])
        logger.info("SwingStrategy: evaluating %d signals", len(signals))

        # Group signals by ticker and score compositely
        by_ticker: dict[str, list[dict]] = {}
        for sig in signals:
            ticker = sig.get("ticker", "")
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(sig)

        trades: list[dict] = []
        for ticker, sigs in by_ticker.items():
            # Need at least technical + one other signal
            types = {s.get("type") for s in sigs}
            if "technical" not in types:
                continue
            if len(types) < 2:
                continue

            # Composite score
            total_strength = sum(s.get("strength", 0) for s in sigs)
            avg_strength = total_strength / len(sigs)

            if abs(avg_strength) < 0.15:
                continue

            direction = "long" if avg_strength > 0 else "short"
            trades.append({
                "ticker": ticker,
                "direction": direction,
                "horizon": "swing",
                "strength": round(avg_strength, 4),
                "signal_count": len(sigs),
                "execution_method": "algorithmic",
                "urgency": "high",
            })

        return {"trades": trades, "count": len(trades)}


class MacroStrategyAgent(BaseAgent):
    """Macro thesis (medium/long-term) strategy agent."""

    def __init__(self) -> None:
        super().__init__(AgentType.MACRO_STRATEGY, "MacroStrategy")

    async def _run(self, task: AgentTask) -> dict:
        signals = task.input_data.get("signals", [])
        regime = task.input_data.get("regime", "risk_on")
        logger.info("MacroStrategy: regime=%s, %d signals", regime, len(signals))

        trades: list[dict] = []
        for sig in signals:
            if sig.get("type") not in ("macro", "fundamental"):
                continue
            if abs(sig.get("strength", 0)) < 0.2:
                continue

            trades.append({
                "ticker": sig["ticker"],
                "direction": sig["direction"],
                "horizon": "medium_term" if abs(sig["strength"]) < 0.5 else "long_term",
                "strength": sig["strength"],
                "regime": regime,
                "execution_method": "voice",
                "urgency": "normal",
            })

        return {"trades": trades, "count": len(trades)}


class RiskManagementAgent(BaseAgent):
    """Risk management agent: position sizing, exposure limits."""

    def __init__(
        self,
        nav: float = 100_000_000,
        max_single_name_pct: float = 0.05,
        max_gross_leverage: float = 2.0,
        max_net_exposure: float = 0.5,
        max_daily_loss_pct: float = 0.02,
    ) -> None:
        super().__init__(AgentType.RISK_MANAGEMENT, "RiskManagement")
        self.nav = nav
        self.max_single_name_pct = max_single_name_pct
        self.max_gross_leverage = max_gross_leverage
        self.max_net_exposure = max_net_exposure
        self.max_daily_loss_pct = max_daily_loss_pct

    async def _run(self, task: AgentTask) -> dict:
        trades = task.input_data.get("trades", [])
        portfolio = task.input_data.get("portfolio", [])
        logger.info("RiskCheck: evaluating %d trades", len(trades))

        approved: list[dict] = []
        rejected: list[dict] = []

        current_gross = sum(
            abs(p.get("notional", 0)) for p in portfolio
        )
        current_long = sum(
            p.get("notional", 0) for p in portfolio if p.get("direction") == "long"
        )
        current_short = sum(
            abs(p.get("notional", 0)) for p in portfolio if p.get("direction") == "short"
        )

        for trade in trades:
            ticker = trade.get("ticker", "")
            strength = abs(trade.get("strength", 0))
            volatility = trade.get("volatility", 0.20)

            # Position sizing
            if volatility > 0:
                raw_size = (self.nav * self.max_daily_loss_pct * strength) / volatility
            else:
                raw_size = self.nav * 0.01 * strength

            max_size = self.nav * self.max_single_name_pct
            position_size = min(raw_size, max_size)
            notional = position_size

            # Gross exposure check
            if (current_gross + notional) / self.nav > self.max_gross_leverage:
                rejected.append({**trade, "reason": "gross exposure limit"})
                continue

            # Net exposure check
            if trade.get("direction") == "long":
                new_net = (current_long + notional - current_short) / self.nav
            else:
                new_net = (current_long - current_short - notional) / self.nav

            if abs(new_net) > self.max_net_exposure:
                rejected.append({**trade, "reason": "net exposure limit"})
                continue

            # Single name check
            existing = sum(
                abs(p.get("notional", 0))
                for p in portfolio if p.get("ticker") == ticker
            )
            if (existing + notional) / self.nav > self.max_single_name_pct:
                rejected.append({**trade, "reason": "single name limit"})
                continue

            approved_trade = {
                **trade,
                "position_size": round(position_size, 2),
                "notional": round(notional, 2),
            }
            approved.append(approved_trade)
            current_gross += notional
            if trade.get("direction") == "long":
                current_long += notional
            else:
                current_short += notional

        return {
            "approved": approved,
            "rejected": rejected,
            "approved_count": len(approved),
            "rejected_count": len(rejected),
        }


class ExecutionAgent(BaseAgent):
    """Execution agent: order routing, slippage control."""

    def __init__(self) -> None:
        super().__init__(AgentType.EXECUTION, "Execution")
        self._trade_counter = 0

    async def _run(self, task: AgentTask) -> dict:
        trades = task.input_data.get("trades", [])
        logger.info("Execution: routing %d trades", len(trades))

        results: list[dict] = []

        for trade in trades:
            self._trade_counter += 1
            trade_id = f"TRD-{datetime.utcnow().strftime('%Y%m%d')}-{self._trade_counter:06d}"

            method = trade.get("execution_method", "algorithmic")
            slippage_base = {
                "algorithmic": 1.0,
                "dma": 2.0,
                "dark_pool": 0.5,
                "voice": 3.0,
            }.get(method, 2.0)

            if trade.get("urgency") == "immediate":
                slippage_base *= 2.0

            import numpy as np
            slippage = slippage_base + np.random.uniform(-0.5, 1.0)

            entry_price = trade.get("entry_price", 100.0)
            position_size = trade.get("position_size", 10000)

            commission_rate = {"algorithmic": 0.5, "dma": 1.0, "dark_pool": 0.3, "voice": 2.0}.get(method, 1.0)
            commission = abs(position_size * entry_price * commission_rate / 10_000)

            venue = {"algorithmic": "ALGO-TWAP", "dma": "NYSE-DMA", "dark_pool": "IEX-DARK", "voice": "VOICE-DESK"}.get(method, "UNKNOWN")

            results.append({
                "trade_id": trade_id,
                "ticker": trade.get("ticker", ""),
                "direction": trade.get("direction", ""),
                "filled_size": position_size,
                "avg_price": round(entry_price * (1 + slippage / 10_000 * (1 if trade.get("direction") == "long" else -1)), 4),
                "slippage_bps": round(slippage, 2),
                "commission": round(commission, 2),
                "venue": venue,
                "status": "filled",
            })

        return {"executions": results, "count": len(results)}


class ReportingAgent(BaseAgent):
    """Reporting agent: P&L, attribution, missed opportunities."""

    def __init__(self) -> None:
        super().__init__(AgentType.REPORTING, "Reporting")

    async def _run(self, task: AgentTask) -> dict:
        executions = task.input_data.get("executions", [])
        portfolio = task.input_data.get("portfolio", [])
        missed = task.input_data.get("missed_opportunities", [])
        agent_metrics = task.input_data.get("agent_metrics", {})

        logger.info("Reporting: generating daily report")

        today = datetime.utcnow().strftime("%Y-%m-%d")

        total_pnl = sum(p.get("pnl", 0) for p in portfolio)
        n_trades = len(executions)
        filled = [e for e in executions if e.get("status") == "filled"]
        fill_rate = len(filled) / n_trades if n_trades > 0 else 0.0

        import numpy as np
        avg_slippage = float(np.mean([e.get("slippage_bps", 0) for e in filled])) if filled else 0.0

        report = {
            "date": today,
            "total_pnl": round(total_pnl, 2),
            "trades_executed": n_trades,
            "fill_rate": round(fill_rate, 4),
            "avg_slippage_bps": round(avg_slippage, 2),
            "missed_opportunities": missed[:20],
            "agent_performance": agent_metrics,
        }

        return {"report": report}


class MLLearningAgent(BaseAgent):
    """ML learning agent: model updates and weight recalibration."""

    def __init__(self) -> None:
        super().__init__(AgentType.ML_LEARNING, "MLLearning")

    async def _run(self, task: AgentTask) -> dict:
        results = task.input_data.get("execution_results", [])
        missed = task.input_data.get("missed_opportunities", [])

        logger.info("MLLearning: processing %d results, %d missed", len(results), len(missed))

        # Simplified learning: track prediction accuracy
        correct = 0
        total = 0
        for r in results:
            actual_return = r.get("actual_return")
            predicted_direction = r.get("direction")
            if actual_return is not None:
                total += 1
                if (actual_return > 0 and predicted_direction == "long") or \
                   (actual_return < 0 and predicted_direction == "short"):
                    correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": round(accuracy, 4),
            "predictions_evaluated": total,
            "missed_recorded": len(missed),
            "model_updated": True,
        }


# ---------------------------------------------------------------------------
# InvestmentWorkflowAgent (main orchestrator)
# ---------------------------------------------------------------------------


class InvestmentWorkflowAgent:
    """
    Ruflo agent that orchestrates the investment workflow:
    1. Universe data fetch agent (parallel across asset classes)
    2. Analysis agents (technical, fundamental, macro, sentiment)
    3. Strategy agents (HFT scanner, swing scanner, macro thesis)
    4. Risk management agent (position sizing, exposure limits)
    5. Execution agent (order routing, slippage control)
    6. Reporting agent (P&L, attribution, missed opportunities)
    """

    def __init__(
        self,
        nav: float = 100_000_000,
        max_single_name_pct: float = 0.05,
        max_gross_leverage: float = 2.0,
    ) -> None:
        self.nav = nav

        # Initialise all agents
        self._data_agent = DataFetchAgent()
        self._tech_agent = TechnicalAnalysisAgent()
        self._fund_agent = FundamentalAnalysisAgent()
        self._macro_agent = MacroAnalysisAgent()
        self._sentiment_agent = SentimentAnalysisAgent()
        self._distress_agent = DistressAnalysisAgent()
        self._hft_agent = HFTStrategyAgent()
        self._swing_agent = SwingStrategyAgent()
        self._macro_strategy_agent = MacroStrategyAgent()
        self._risk_agent = RiskManagementAgent(nav, max_single_name_pct, max_gross_leverage)
        self._execution_agent = ExecutionAgent()
        self._reporting_agent = ReportingAgent()
        self._ml_agent = MLLearningAgent()

        self._portfolio: list[dict] = []
        self._daily_results: list[dict] = []

        logger.info("InvestmentWorkflowAgent initialized: NAV=$%.2fM", nav / 1_000_000)

    # -------------------------------------------------------------------
    # Swarm coordination
    # -------------------------------------------------------------------

    async def _run_parallel(self, tasks: list[tuple[BaseAgent, AgentTask]]) -> list[AgentTask]:
        """Run multiple agent tasks in parallel."""
        coros = [agent.execute(task) for agent, task in tasks]
        results = await asyncio.gather(*coros, return_exceptions=True)

        completed: list[AgentTask] = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Parallel task failed: %s", r)
                task = AgentTask(status=TaskStatus.FAILED, error=str(r))
                completed.append(task)
            else:
                completed.append(r)

        return completed

    def create_analysis_swarm(
        self,
        universe: list,
        price_cache: Optional[dict] = None,
        fundamentals_cache: Optional[dict] = None,
        macro_data: Optional[dict] = None,
        news_data: Optional[list[dict]] = None,
    ) -> list[AgentTask]:
        """
        Create a swarm of analysis tasks to run in parallel.

        Returns list of AgentTask objects ready for parallel execution.
        """
        if price_cache is None:
            price_cache = {}
        if fundamentals_cache is None:
            fundamentals_cache = {}
        if macro_data is None:
            macro_data = {}
        if news_data is None:
            news_data = []

        tasks = [
            AgentTask(
                agent_type=AgentType.TECHNICAL_ANALYSIS,
                name="technical_scan",
                input_data={"universe": universe, "price_cache": price_cache},
                priority=TaskPriority.HIGH,
            ),
            AgentTask(
                agent_type=AgentType.FUNDAMENTAL_ANALYSIS,
                name="fundamental_scan",
                input_data={"universe": universe, "fundamentals_cache": fundamentals_cache},
                priority=TaskPriority.HIGH,
            ),
            AgentTask(
                agent_type=AgentType.MACRO_ANALYSIS,
                name="macro_regime",
                input_data={"macro_data": macro_data},
                priority=TaskPriority.HIGH,
            ),
            AgentTask(
                agent_type=AgentType.SENTIMENT_ANALYSIS,
                name="sentiment_scan",
                input_data={"news_data": news_data},
                priority=TaskPriority.NORMAL,
            ),
            AgentTask(
                agent_type=AgentType.DISTRESS_ANALYSIS,
                name="distress_scan",
                input_data={"universe": universe, "fundamentals_cache": fundamentals_cache},
                priority=TaskPriority.NORMAL,
            ),
        ]

        return tasks

    async def coordinate_strategy_agents(
        self,
        analysis_results: dict,
    ) -> list[TradeThesis]:
        """
        Coordinate strategy agents with analysis results.

        Runs HFT, swing, and macro strategy agents in parallel,
        then merges their outputs.
        """
        all_signals = []
        for key in ("technical", "fundamental", "macro", "sentiment", "distress"):
            signals = analysis_results.get(f"{key}_signals", [])
            all_signals.extend(signals)

        regime = analysis_results.get("regime", "risk_on")

        # Run strategy agents in parallel
        strategy_tasks = [
            (self._hft_agent, AgentTask(
                agent_type=AgentType.HFT_STRATEGY,
                name="hft_strategy",
                input_data={"signals": all_signals},
            )),
            (self._swing_agent, AgentTask(
                agent_type=AgentType.SWING_STRATEGY,
                name="swing_strategy",
                input_data={"signals": all_signals},
            )),
            (self._macro_strategy_agent, AgentTask(
                agent_type=AgentType.MACRO_STRATEGY,
                name="macro_strategy",
                input_data={"signals": all_signals, "regime": regime},
            )),
        ]

        results = await self._run_parallel(strategy_tasks)

        # Merge all trades
        all_trades: list[dict] = []
        for task in results:
            if task.status == TaskStatus.COMPLETED:
                trades = task.output_data.get("trades", [])
                all_trades.extend(trades)

        # Convert to TradeThesis objects
        theses: list[TradeThesis] = []
        for t in all_trades:
            theses.append(TradeThesis(
                ticker=t.get("ticker", ""),
                direction=t.get("direction", "neutral"),
                horizon=t.get("horizon", "swing"),
                entry_price=t.get("entry_price", 100.0),
                target_price=t.get("target_price", 105.0),
                stop_loss=t.get("stop_loss", 97.0),
                position_size=t.get("position_size", 0),
                notional=t.get("notional", 0),
                composite_score=t.get("strength", 0),
                risk_reward=0.0,
                signals=[t],
                execution_method=t.get("execution_method", "algorithmic"),
                urgency=t.get("urgency", "normal"),
            ))

        logger.info("Strategy agents produced %d trade theses", len(theses))
        return theses

    async def risk_check(
        self,
        trades: list[TradeThesis],
        portfolio: list[dict],
    ) -> list[TradeThesis]:
        """
        Run risk management checks on proposed trades.

        Returns only approved trades (filtered by position limits,
        exposure constraints, and concentration rules).
        """
        trade_dicts = [
            {
                "ticker": t.ticker,
                "direction": t.direction,
                "horizon": t.horizon,
                "strength": t.composite_score,
                "entry_price": t.entry_price,
                "execution_method": t.execution_method,
                "urgency": t.urgency,
            }
            for t in trades
        ]

        risk_task = AgentTask(
            agent_type=AgentType.RISK_MANAGEMENT,
            name="risk_check",
            input_data={"trades": trade_dicts, "portfolio": portfolio},
        )

        result = await self._risk_agent.execute(risk_task)

        if result.status != TaskStatus.COMPLETED:
            logger.error("Risk check failed: %s", result.error)
            return []

        approved = result.output_data.get("approved", [])
        rejected = result.output_data.get("rejected", [])

        logger.info(
            "Risk check: %d approved, %d rejected",
            len(approved), len(rejected),
        )

        # Map back to TradeThesis
        approved_tickers = {t["ticker"] for t in approved}
        filtered = [t for t in trades if t.ticker in approved_tickers]

        # Update position sizes from risk agent
        size_map = {t["ticker"]: t.get("position_size", 0) for t in approved}
        for t in filtered:
            if t.ticker in size_map:
                t.position_size = size_map[t.ticker]
                t.notional = t.position_size * t.entry_price

        return filtered

    async def execute_via_agents(
        self,
        approved_trades: list[TradeThesis],
    ) -> list[ExecutionResult]:
        """
        Execute approved trades through the execution agent.

        Routes HFT trades to algorithmic execution and
        medium/long-term trades to voice execution.
        """
        trade_dicts = [
            {
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_price": t.entry_price,
                "position_size": t.position_size,
                "execution_method": t.execution_method,
                "urgency": t.urgency,
            }
            for t in approved_trades
        ]

        exec_task = AgentTask(
            agent_type=AgentType.EXECUTION,
            name="execute_trades",
            input_data={"trades": trade_dicts},
        )

        result = await self._execution_agent.execute(exec_task)

        if result.status != TaskStatus.COMPLETED:
            logger.error("Execution failed: %s", result.error)
            return []

        exec_data = result.output_data.get("executions", [])

        results: list[ExecutionResult] = []
        for e in exec_data:
            results.append(ExecutionResult(
                trade_id=e["trade_id"],
                ticker=e["ticker"],
                direction=e["direction"],
                filled_size=e["filled_size"],
                avg_price=e["avg_price"],
                slippage_bps=e["slippage_bps"],
                commission=e["commission"],
                venue=e["venue"],
                status=e["status"],
            ))

        logger.info("Executed %d trades via agents", len(results))
        return results

    async def generate_reports_parallel(
        self,
        executions: Optional[list[dict]] = None,
    ) -> DailyReport:
        """
        Generate daily report and ML learning update in parallel.
        """
        if executions is None:
            executions = self._daily_results

        # Collect agent metrics
        agent_metrics = {}
        for agent in [
            self._data_agent, self._tech_agent, self._fund_agent,
            self._macro_agent, self._sentiment_agent, self._distress_agent,
            self._hft_agent, self._swing_agent, self._macro_strategy_agent,
            self._risk_agent, self._execution_agent,
        ]:
            agent_metrics[agent.name] = agent.metrics

        # Run reporting and ML learning in parallel
        report_task = AgentTask(
            agent_type=AgentType.REPORTING,
            name="daily_report",
            input_data={
                "executions": executions,
                "portfolio": self._portfolio,
                "missed_opportunities": [],
                "agent_metrics": agent_metrics,
            },
        )
        ml_task = AgentTask(
            agent_type=AgentType.ML_LEARNING,
            name="ml_update",
            input_data={
                "execution_results": executions,
                "missed_opportunities": [],
            },
        )

        results = await self._run_parallel([
            (self._reporting_agent, report_task),
            (self._ml_agent, ml_task),
        ])

        # Extract report
        report_data = {}
        for task in results:
            if task.agent_type == AgentType.REPORTING and task.status == TaskStatus.COMPLETED:
                report_data = task.output_data.get("report", {})

        report = DailyReport(
            date=report_data.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
            total_pnl=report_data.get("total_pnl", 0.0),
            trades_executed=report_data.get("trades_executed", 0),
            fill_rate=report_data.get("fill_rate", 0.0),
            avg_slippage_bps=report_data.get("avg_slippage_bps", 0.0),
            missed_opportunities=report_data.get("missed_opportunities", []),
            agent_performance=report_data.get("agent_performance", {}),
        )

        logger.info("Daily report generated: P&L=$%.2f", report.total_pnl)
        return report

    # -------------------------------------------------------------------
    # Full daily workflow
    # -------------------------------------------------------------------

    async def run_daily_workflow(
        self,
        price_cache: Optional[dict] = None,
        fundamentals_cache: Optional[dict] = None,
        macro_data: Optional[dict] = None,
        news_data: Optional[list[dict]] = None,
    ) -> DailyReport:
        """
        Execute the full daily investment workflow.

        Steps:
        1. Fetch universe data (parallel by asset class)
        2. Run analysis swarm (technical, fundamental, macro, sentiment, distress)
        3. Generate strategy trades (HFT, swing, macro thesis)
        4. Risk check all trades
        5. Execute approved trades
        6. Generate daily report with ML learning
        """
        logger.info("=== DAILY WORKFLOW START ===")
        start = time.monotonic()

        # Step 1: Fetch universe
        fetch_tasks = [
            (self._data_agent, AgentTask(
                agent_type=AgentType.DATA_FETCH,
                name=f"fetch_{ac}",
                input_data={"asset_class": ac},
            ))
            for ac in ["equity", "fixed_income", "commodity", "fx", "crypto"]
        ]
        fetch_results = await self._run_parallel(fetch_tasks)

        universe = []
        for task in fetch_results:
            if task.status == TaskStatus.COMPLETED:
                universe.extend(task.output_data.get("securities", []))

        logger.info("Step 1: Universe fetched - %d securities", len(universe))

        # Step 2: Analysis swarm
        analysis_tasks = self.create_analysis_swarm(
            universe, price_cache, fundamentals_cache, macro_data, news_data,
        )
        agent_map = {
            AgentType.TECHNICAL_ANALYSIS: self._tech_agent,
            AgentType.FUNDAMENTAL_ANALYSIS: self._fund_agent,
            AgentType.MACRO_ANALYSIS: self._macro_agent,
            AgentType.SENTIMENT_ANALYSIS: self._sentiment_agent,
            AgentType.DISTRESS_ANALYSIS: self._distress_agent,
        }
        parallel_analysis = [
            (agent_map[t.agent_type], t) for t in analysis_tasks
        ]
        analysis_results = await self._run_parallel(parallel_analysis)

        # Aggregate analysis outputs
        aggregated: dict[str, Any] = {"regime": "risk_on"}
        all_signals: list[dict] = []
        for task in analysis_results:
            if task.status == TaskStatus.COMPLETED:
                signals = task.output_data.get("signals", [])
                all_signals.extend(signals)
                if "regime" in task.output_data:
                    aggregated["regime"] = task.output_data["regime"]

        aggregated["all_signals"] = all_signals
        # Split by type for strategy agents
        for sig_type in ("technical", "fundamental", "macro", "sentiment", "distress"):
            aggregated[f"{sig_type}_signals"] = [
                s for s in all_signals if s.get("type") == sig_type
            ]

        logger.info("Step 2: Analysis complete - %d total signals", len(all_signals))

        # Step 3: Strategy agents
        theses = await self.coordinate_strategy_agents(aggregated)
        logger.info("Step 3: Strategies generated - %d theses", len(theses))

        # Step 4: Risk check
        approved = await self.risk_check(theses, self._portfolio)
        logger.info("Step 4: Risk check - %d approved", len(approved))

        # Step 5: Execute
        executions = await self.execute_via_agents(approved)
        self._daily_results = [
            {
                "trade_id": e.trade_id,
                "ticker": e.ticker,
                "direction": e.direction,
                "filled_size": e.filled_size,
                "avg_price": e.avg_price,
                "slippage_bps": e.slippage_bps,
                "status": e.status,
            }
            for e in executions
        ]
        logger.info("Step 5: Executed - %d trades", len(executions))

        # Step 6: Report
        report = await self.generate_reports_parallel()

        elapsed = (time.monotonic() - start) * 1000
        logger.info("=== DAILY WORKFLOW COMPLETE (%.1fms) ===", elapsed)

        return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    agent = InvestmentWorkflowAgent(nav=100_000_000)

    # Run with sample macro data
    report = asyncio.run(agent.run_daily_workflow(
        macro_data={"vix": 18, "yield_curve_slope": 0.5, "credit_spread_bps": 140, "pmi": 53},
    ))

    print(f"\nDaily Report: {report.date}")
    print(f"  P&L: ${report.total_pnl:,.2f}")
    print(f"  Trades: {report.trades_executed}")
    print(f"  Fill Rate: {report.fill_rate:.1%}")
    print(f"  Avg Slippage: {report.avg_slippage_bps:.1f}bps")
