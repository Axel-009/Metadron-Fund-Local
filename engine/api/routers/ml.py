"""
ML router — ML, ML MODELS, MC SIM, SIM, QUANT, STRAT tabs
Wraps: AlphaOptimizer, Backtester, PatternRecognition, UniverseClassifier,
       DeepLearningEngine, SocialFeatures
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.ml")
router = APIRouter()

_alpha = None
_backtester = None
_pattern_rec = None
_classifier = None
_deep = None


def _get_alpha():
    global _alpha
    if _alpha is None:
        from engine.ml.alpha_optimizer import AlphaOptimizer
        _alpha = AlphaOptimizer()
    return _alpha


def _get_backtester():
    global _backtester
    if _backtester is None:
        from engine.ml.backtester import Backtester
        _backtester = Backtester()
    return _backtester


def _get_pattern_rec():
    global _pattern_rec
    if _pattern_rec is None:
        from engine.ml.pattern_recognition import PatternRecognition
        _pattern_rec = PatternRecognition()
    return _pattern_rec


def _get_classifier():
    global _classifier
    if _classifier is None:
        from engine.ml.universe_classifier import UniverseClassifier
        _classifier = UniverseClassifier()
    return _classifier


def _get_deep():
    global _deep
    if _deep is None:
        from engine.ml.deep_learning_engine import DeepLearningEngine
        _deep = DeepLearningEngine()
    return _deep


# ─── ML tab ────────────────────────────────────────────────

@router.get("/alpha/last")
async def alpha_last():
    """Last alpha optimization result."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"status": "no_optimization_run", "timestamp": datetime.utcnow().isoformat()}

        return {
            "expected_return": getattr(last, "expected_annual_return", 0),
            "volatility": getattr(last, "annual_volatility", 0),
            "sharpe": getattr(last, "sharpe_ratio", 0),
            "max_drawdown": getattr(last, "max_drawdown", 0),
            "weights": getattr(last, "optimal_weights", {}),
            "signal_count": len(getattr(last, "signals", [])),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/alpha/last error: {e}")
        return {"error": str(e)}


@router.get("/alpha/signals")
async def alpha_signals():
    """Per-ticker alpha signals with quality tiers."""
    try:
        alpha = _get_alpha()
        last = alpha.get_last()
        if last is None:
            return {"signals": [], "status": "no_optimization_run"}

        result = []
        for s in (getattr(last, "signals", []) or []):
            result.append({
                "ticker": getattr(s, "ticker", ""),
                "alpha_pred": getattr(s, "alpha_pred", 0),
                "quality_tier": getattr(s, "quality_tier", "?"),
                "sharpe_estimate": getattr(s, "sharpe_estimate", 0),
                "momentum_3m": getattr(s, "momentum_3m", 0),
                "momentum_1m": getattr(s, "momentum_1m", 0),
                "vol": getattr(s, "vol", 0),
                "weight": getattr(s, "weight", 0),
            })
        return {"signals": result, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/alpha/signals error: {e}")
        return {"signals": [], "error": str(e)}


# ─── ML MODELS tab ─────────────────────────────────────────

@router.get("/models/status")
async def models_status():
    """Status of all engine modules across layers."""
    try:
        # Probe each engine module to check if it's importable and functional
        modules = []

        engine_map = [
            ("L1", "UniverseEngine", "engine.data.universe_engine"),
            ("L1", "OpenBBData", "engine.data.openbb_data"),
            ("L1", "IngestionOrchestrator", "engine.data.ingestion_orchestrator"),
            ("L2", "MacroEngine", "engine.signals.macro_engine"),
            ("L2", "MetadronCube", "engine.signals.metadron_cube"),
            ("L2", "StatArbEngine", "engine.signals.stat_arb_engine"),
            ("L2", "ContagionEngine", "engine.signals.contagion_engine"),
            ("L2", "SecurityAnalysisEngine", "engine.signals.security_analysis_engine"),
            ("L2", "PatternDiscoveryEngine", "engine.signals.pattern_discovery_engine"),
            ("L2", "SocialPredictionEngine", "engine.signals.social_prediction_engine"),
            ("L2", "DistressedAssetEngine", "engine.signals.distressed_asset_engine"),
            ("L2", "CVREngine", "engine.signals.cvr_engine"),
            ("L2", "EventDrivenEngine", "engine.signals.event_driven_engine"),
            ("L2", "FedLiquidityPlumbing", "engine.signals.fed_liquidity_plumbing"),
            ("L3", "AlphaOptimizer", "engine.ml.alpha_optimizer"),
            ("L3", "Backtester", "engine.ml.backtester"),
            ("L3", "PatternRecognition", "engine.ml.pattern_recognition"),
            ("L3", "UniverseClassifier", "engine.ml.universe_classifier"),
            ("L3", "DeepLearningEngine", "engine.ml.deep_learning_engine"),
            ("L3", "SocialFeatures", "engine.ml.social_features"),
            ("L4", "BetaCorridor", "engine.portfolio.beta_corridor"),
            ("L5", "ExecutionEngine", "engine.execution.execution_engine"),
            ("L5", "DecisionMatrix", "engine.execution.decision_matrix"),
            ("L5", "OptionsEngine", "engine.execution.options_engine"),
            ("L5", "PaperBroker", "engine.execution.paper_broker"),
            ("L5", "AlpacaBroker", "engine.execution.alpaca_broker"),
            ("L6", "ResearchBots", "engine.agents.research_bots"),
            ("L6", "SectorBots", "engine.agents.sector_bots"),
            ("L6", "AgentScorecard", "engine.agents.agent_scorecard"),
            ("L6", "InvestorPersonas", "engine.agents.investor_personas"),
            ("L7", "L7UnifiedExecution", "engine.execution.l7_unified_execution_surface"),
            ("L7", "WondertraderEngine", "engine.execution.wondertrader_engine"),
            ("L7", "ExchangeCoreEngine", "engine.execution.exchange_core_engine"),
            ("MON", "MarketWrap", "engine.monitoring.market_wrap"),
            ("MON", "PlatinumReport", "engine.monitoring.platinum_report"),
            ("MON", "DailyReport", "engine.monitoring.daily_report"),
            ("MON", "AnomalyDetector", "engine.monitoring.anomaly_detector"),
            ("MON", "LearningLoop", "engine.monitoring.learning_loop"),
            ("MON", "SectorTracker", "engine.monitoring.sector_tracker"),
            ("MON", "HeatmapEngine", "engine.monitoring.heatmap_engine"),
            ("MON", "PortfolioAnalytics", "engine.monitoring.portfolio_analytics"),
        ]

        import importlib
        for layer, name, module_path in engine_map:
            try:
                importlib.import_module(module_path)
                modules.append({"layer": layer, "name": name, "module": module_path, "status": "online"})
            except Exception as ex:
                modules.append({"layer": layer, "name": name, "module": module_path, "status": "offline", "error": str(ex)[:100]})

        online = sum(1 for m in modules if m["status"] == "online")
        return {
            "modules": modules,
            "total": len(modules),
            "online": online,
            "offline": len(modules) - online,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/models/status error: {e}")
        return {"modules": [], "error": str(e)}


# ─── MC SIM tab ────────────────────────────────────────────

@router.get("/monte-carlo")
async def monte_carlo(
    tickers: str = Query("SPY", description="Comma-separated tickers"),
    n_paths: int = Query(1000, ge=10, le=10000),
    horizon_days: int = Query(252, ge=5, le=756),
):
    """Run Monte Carlo simulation."""
    try:
        bt = _get_backtester()
        ticker_list = [t.strip() for t in tickers.split(",")]

        if hasattr(bt, "monte_carlo"):
            result = bt.monte_carlo(tickers=ticker_list, n_paths=n_paths, horizon=horizon_days)
        elif hasattr(bt, "run_monte_carlo"):
            result = bt.run_monte_carlo(tickers=ticker_list, n_paths=n_paths, horizon=horizon_days)
        else:
            return {"error": "Monte Carlo not available on backtester", "timestamp": datetime.utcnow().isoformat()}

        return {**(result if isinstance(result, dict) else {"data": str(result)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/monte-carlo error: {e}")
        return {"error": str(e)}


# ─── SIM tab (HMM + Black-Scholes) ────────────────────────

@router.get("/regime-history")
async def regime_history():
    """HMM regime classification history."""
    try:
        from engine.signals.metadron_cube import MetadronCube
        cube = MetadronCube()
        history = cube.get_history()

        entries = getattr(history, "entries", []) if hasattr(history, "entries") else (history if isinstance(history, list) else [])
        regimes = []
        for h in entries[-200:]:
            regimes.append({
                "regime": getattr(h, "regime", "UNKNOWN"),
                "target_beta": getattr(h, "target_beta", 0),
                "max_leverage": getattr(h, "max_leverage", 1.0),
            })

        return {"regimes": regimes, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/regime-history error: {e}")
        return {"regimes": [], "error": str(e)}


# ─── QUANT tab ─────────────────────────────────────────────

@router.get("/patterns")
async def pattern_recognition(ticker: str = Query("SPY")):
    """Technical pattern recognition for a ticker."""
    try:
        pr = _get_pattern_rec()
        if hasattr(pr, "analyze"):
            result = pr.analyze(ticker)
        elif hasattr(pr, "scan"):
            result = pr.scan(ticker)
        else:
            result = {}
        return {**(result if isinstance(result, dict) else {"data": str(result)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/patterns error: {e}")
        return {"error": str(e)}


@router.get("/classifier/tiers")
async def classifier_tiers():
    """Universe classifier quality tiers A-G."""
    try:
        clf = _get_classifier()
        if hasattr(clf, "get_tiers"):
            tiers = clf.get_tiers()
        elif hasattr(clf, "classify"):
            tiers = clf.classify()
        else:
            tiers = {}
        return {**(tiers if isinstance(tiers, dict) else {"data": str(tiers)}), "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"ml/classifier error: {e}")
        return {"error": str(e)}


# ─── STRAT tab ─────────────────────────────────────────────

@router.get("/strategy/config")
async def strategy_config():
    """Current pipeline strategy configuration."""
    try:
        return {
            "pipeline_stages": [
                "L1_Universe", "L2_Macro", "L2_Cube", "L2_Signals",
                "L3_Alpha", "L4_Beta", "L5_Decision", "L5_Execution",
                "L6_Agents", "L7_HFT",
            ],
            "regime_params": {
                "TRENDING": {"leverage": 3.0, "beta_cap": 0.65, "beta_burst": 0.70},
                "RANGE": {"leverage": 2.5, "beta_cap": 0.45, "beta_burst": 0.55},
                "STRESS": {"leverage": 1.5, "beta_cap": 0.15, "beta_burst": 0.20},
                "CRASH": {"leverage": 0.8, "beta_cap": -0.20, "beta_burst": -0.10},
            },
            "sleeve_allocation": {
                "p1_directional_equity": 0.40,
                "p2_factor_rotation": 0.10,
                "p3_commodities_macro": 0.10,
                "p4_options_convexity": 0.25,
                "p5_hedges_volatility": 0.10,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"ml/strategy/config error: {e}")
        return {"error": str(e)}
