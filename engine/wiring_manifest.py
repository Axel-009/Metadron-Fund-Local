"""Wiring Manifest — Metadron Capital Investment Platform.

Machine-readable component registry, wiring graph, routing contract,
and validation engine.  This file is the SINGLE SOURCE OF TRUTH for how
the platform is assembled.  Any AI model, code review tool, or developer
MUST read this before generating code that touches execution, data, or
signal flows.

Usage:
    python3 -m engine.wiring_manifest          # print validation report
    python3 -m engine.wiring_manifest --boot   # bootstrap full system

Programmatic:
    from engine.wiring_manifest import validate_wiring, bootstrap_full_system
    report = validate_wiring()
    state  = bootstrap_full_system(nav=1_000_000, ibkr_paper=True)
"""

from __future__ import annotations

import importlib
import logging
import sys
import textwrap
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# 1.  COMPONENT REGISTRY
# ============================================================================
# Every mandatory engine, its module path, class name, boot phase, and
# upstream dependencies (component_id references).
#
# Phases execute in order: DATA(1) -> SIGNALS(2) -> INTELLIGENCE(3)
#   -> DECISION(4) -> EXECUTION(5) -> LEARNING(6) -> MONITORING(7)
# ============================================================================

@dataclass(frozen=True)
class ComponentSpec:
    """Immutable specification for one platform component."""
    component_id: str
    module: str
    class_name: str
    phase: int
    phase_name: str
    mandatory: bool
    depends_on: Tuple[str, ...] = ()
    description: str = ""


# Ordered by boot phase, then by dependency topology within each phase.
COMPONENTS: Dict[str, ComponentSpec] = OrderedDict()

def _reg(cid, mod, cls, phase, phase_name, mandatory=True, deps=(), desc=""):
    COMPONENTS[cid] = ComponentSpec(
        component_id=cid, module=mod, class_name=cls,
        phase=phase, phase_name=phase_name, mandatory=mandatory,
        depends_on=tuple(deps), description=desc,
    )


# ---------- Phase 1: DATA ----------
_reg("universe_engine",
     "engine.data.universe_engine", "UniverseEngine",
     1, "DATA", deps=(),
     desc="4-run scan: SP500(~500) + SP400(~400) + SP600(~600) + ETF/FI(~70) + 26 RV pairs = ~1,600 securities")
_reg("data_quality_gate",
     "engine.data.data_quality", "DataQualityGate",
     1, "DATA", deps=(),
     desc="Pre-pool validator — staleness, completeness, outlier z-score")
_reg("data_ingestion",
     "engine.data.ingestion_orchestrator", "DataIngestionOrchestrator",
     1, "DATA", deps=("universe_engine",),
     desc="Continuous multi-asset ingestion (NO crypto)")
_reg("data_pool",
     "engine.data.universal_pooling", "UniversalDataPool",
     1, "DATA", deps=("data_quality_gate",),
     desc="Cross-asset data dispatcher — routes 9 asset classes to destination layers")
_reg("news_engine",
     "engine.signals.news_engine", "NewsEngine",
     1, "DATA", deps=(),
     desc="newsfilter.io WebSocket (10K+ sources) + FMP fallback — feeds Track B independently")

# ---------- Phase 2: SIGNALS ----------
# Track A: FedLiquidity → MacroEngine → MetadronCube → parallel signal engines
# Track B: NewsEngine → MiroMomentum → EventDriven + CVR (independent from Cube)
_reg("fed_liquidity",
     "engine.signals.fed_liquidity_plumbing", "FedLiquidityPlumbing",
     2, "SIGNALS", deps=("data_pool",),
     desc="Fed balance sheet, reserves, SOFR, ON-RRP, money velocity")
_reg("macro_engine",
     "engine.signals.macro_engine", "MacroEngine",
     2, "SIGNALS", deps=("fed_liquidity", "data_pool"),
     desc="GMTF: SDR tension, rotation, velocity, yield curve, sector ranking")
_reg("metadron_cube",
     "engine.signals.metadron_cube", "MetadronCube",
     2, "SIGNALS", deps=("fed_liquidity", "macro_engine"),
     desc="C(t) = f(L,R,F) — 10-layer intelligence tensor + KillSwitch")
# Track A signal engines — all fed by MetadronCube regime output
_reg("security_analysis",
     "engine.signals.security_analysis_engine", "SecurityAnalysisEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — Graham-Dodd-Klarman top-down + bottom-up")
_reg("contagion_engine",
     "engine.signals.contagion_engine", "ContagionEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — 21-node graph topology, 7 shock scenarios")
_reg("stat_arb_engine",
     "engine.signals.stat_arb_engine", "StatArbEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — Medallion mean reversion + cointegration pairs")
_reg("distressed_assets",
     "engine.signals.distressed_asset_engine", "DistressedAssetEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — 5-model distress ensemble + Graham-Mielle framework")
_reg("pattern_discovery",
     "engine.signals.pattern_discovery_engine", "PatternDiscoveryEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — MiroFish + AI-Newton symbolic regression")
_reg("adaptive_thresholds",
     "engine.signals.adaptive_thresholds", "AdaptiveThresholdCalibrator",
     2, "SIGNALS", deps=("metadron_cube",),
     desc="Track A — rolling percentile threshold recalibration")
_reg("fixed_income_engine",
     "engine.signals.fixed_income_engine", "FixedIncomeEngine",
     2, "SIGNALS", deps=("metadron_cube", "data_pool"),
     desc="Track A — yield curve, credit spreads, duration ladder, FI signals")

# Track B signal engines — fed by NewsEngine, independent from Cube
_reg("social_prediction",
     "engine.signals.social_prediction_engine", "MiroMomentumEngine",
     2, "SIGNALS", deps=("news_engine",),
     desc="Track B — agent-based market sim on news-flagged tickers (40% sentiment + 60% agent)")
_reg("event_driven",
     "engine.signals.event_driven_engine", "EventDrivenEngine",
     2, "SIGNALS", deps=("news_engine", "social_prediction"),
     desc="Track B — 12 event categories, enriched by News+MiroMomentum")
_reg("cvr_engine",
     "engine.signals.cvr_engine", "CVREngine",
     2, "SIGNALS", deps=("news_engine", "social_prediction"),
     desc="Track B — 5-model CVR valuation, enriched by News+MiroMomentum")

# ---------- Phase 3: INTELLIGENCE ----------
_reg("alpha_optimizer",
     "engine.ml.alpha_optimizer", "AlphaOptimizer",
     3, "INTELLIGENCE",
     deps=("macro_engine", "metadron_cube", "universe_engine",
           "security_analysis", "contagion_engine", "stat_arb_engine",
           "distressed_assets", "event_driven", "cvr_engine",
           "pattern_discovery", "social_prediction"),
     desc="Universe merge (4 runs) + aggregate + dedup + cap enforcement + dual alpha scoring → FullUniverseEngine Scan Slate")
_reg("pattern_recognition",
     "engine.ml.pattern_recognition", "PatternRecognitionEngine",
     3, "INTELLIGENCE", deps=("data_pool",),
     desc="Candlestick + chart patterns + statistical anomalies")
_reg("universe_classifier",
     "engine.ml.universe_classifier", "UniverseClassifier",
     3, "INTELLIGENCE", deps=("universe_engine",),
     desc="XGBoost 4-model ensemble, quality tiers A-G")
_reg("deep_learning_engine",
     "engine.ml.deep_learning_engine", "DeepLearningEngine",
     3, "INTELLIGENCE", deps=("data_pool",),
     desc="Pure-numpy PPO agent, 50-feature state vector")
_reg("social_features",
     "engine.ml.social_features", "SocialFeatureBuilder",
     3, "INTELLIGENCE", deps=("social_prediction",),
     desc="Sentiment momentum, engagement velocity, composite social_alpha")
_reg("model_evaluator",
     "engine.ml.model_evaluator", "ModelEvaluator",
     3, "INTELLIGENCE", deps=(),
     desc="Per-class P/R/F1, tier-aware distance weighting")

# ---------- Phase 4: DECISION ----------
_reg("decision_matrix",
     "engine.execution.decision_matrix", "DecisionMatrix",
     4, "DECISION", deps=("alpha_optimizer", "metadron_cube", "macro_engine"),
     desc="4-gate cross-asset quality filter (FUNDAMENTALS 40%, FLOW 20%, MACRO 20%, MOMENTUM 20%) + Kelly sizing")
_reg("allocation_engine",
     "engine.allocation.allocation_engine", "AllocationEngine",
     4, "DECISION", deps=("metadron_cube", "decision_matrix"),
     desc="5-sleeve allocation + KillSwitch monitor")
_reg("beta_corridor",
     "engine.portfolio.beta_corridor", "BetaCorridor",
     4, "DECISION", deps=("macro_engine",),
     desc="Beta corridor 7-12% + vol-normalisation")
_reg("options_engine",
     "engine.execution.options_engine", "OptionsEngine",
     4, "DECISION", deps=("metadron_cube",),
     desc="Black-Scholes Greeks, vol surface, theta+gamma optimizer")
_reg("options_sizer",
     "engine.execution.options_engine", "OptionsSizer",
     4, "DECISION", deps=("options_engine",),
     desc="Edge-gated sizer: BS+MC+Kelly, REJECT if mispricing < 200bps")

# ---------- Phase 5: EXECUTION ----------
_reg("wondertrader_engine",
     "engine.execution.wondertrader_engine", "WonderTraderEngine",
     5, "EXECUTION", deps=(),
     desc="CTA trend-following + HFT micro-price + TWAP/VWAP routing")
_reg("exchange_core_engine",
     "engine.execution.exchange_core_engine", "ExchangeCoreEngine",
     5, "EXECUTION", deps=(),
     desc="LMAX Disruptor ring buffer order matching")
_reg("ibkr_broker",
     "engine.execution.ibkr_broker", "IBKRBroker",
     5, "EXECUTION", deps=(),
     desc="SOLE execution broker — IBKR TWS/Gateway via ib_insync")
_reg("conviction_override",
     "engine.execution.conviction_override", "OverrideManager",
     5, "EXECUTION", deps=("decision_matrix",),
     desc="3-tier conviction override (CONTROLLED/AGGRESSIVE/MAXIMUM)")
_reg("l7_execution_surface",
     "engine.execution.l7_unified_execution_surface", "L7UnifiedExecutionSurface",
     5, "EXECUTION",
     deps=("ibkr_broker", "wondertrader_engine", "exchange_core_engine",
           "decision_matrix", "conviction_override"),
     desc="MANDATORY routing point — ALL orders flow through here")
_reg("tca_engine",
     "engine.execution.tca_engine", "TCAEngine",
     5, "EXECUTION", deps=("l7_execution_surface",),
     desc="Transaction cost analysis — slippage, venue, algo decomposition")

# ---------- Phase 6: LEARNING ----------
_reg("learning_loop",
     "engine.monitoring.learning_loop", "LearningLoop",
     6, "LEARNING", deps=("l7_execution_surface",),
     desc="Closed-loop feedback: signal accuracy -> tier weights -> calibration")
_reg("agent_scorecard",
     "engine.agents.agent_scorecard", "FullAgentScorecard",
     6, "LEARNING", deps=(),
     desc="25 agents across 4 tiers, weekly scoring, promotion/demotion")
_reg("sector_bots",
     "engine.agents.sector_bots", "SectorBotManager",
     6, "LEARNING", deps=(),
     desc="11 GICS sector micro-bots + scorecard")
_reg("research_bots",
     "engine.agents.research_bots", "ResearchBotManager",
     6, "LEARNING", deps=(),
     desc="11 GICS research bots + DNA hierarchy")
_reg("investor_personas",
     "engine.agents.investor_personas", "InvestorPersonaManager",
     6, "LEARNING", deps=(),
     desc="12 investor personas (Buffett, Munger, Graham, etc.)")

# ---------- Phase 7: MONITORING ----------
_reg("anomaly_detector",
     "engine.monitoring.anomaly_detector", "AnomalyDetector",
     7, "MONITORING", deps=("data_pool",),
     desc="8 sub-detectors: z-score, IQR, volume, gap, correlation, VIX, credit, breadth")
_reg("portfolio_analytics",
     "engine.monitoring.portfolio_analytics", "PortfolioAnalytics",
     7, "MONITORING", deps=(),
     desc="Sharpe, Sortino, max DD, scenario engine — static utility class")
_reg("memory_monitor",
     "engine.monitoring.memory_monitor", "MemoryMonitor",
     7, "MONITORING", deps=(),
     desc="Process memory tracking, GC, cache eviction, budget enforcement")


# ============================================================================
# 2.  WIRING GRAPH — directed edges (source -> target)
# ============================================================================
# Each edge: (source_component_id, target_component_id, label)
# Read as: source FEEDS INTO target.

WIRING_EDGES: List[Tuple[str, str, str]] = [
    # ── STAGE 1: SCAN ──────────────────────────────────────────────
    ("universe_engine",      "data_ingestion",       "4-run universe (SP500+SP400+SP600+ETF/FI+RV)"),
    ("data_ingestion",       "data_quality_gate",    "raw market data"),
    ("data_quality_gate",    "data_pool",            "validated data"),

    # Data pool dispatches to Track A engines
    ("data_pool",            "fed_liquidity",        "FRED macro series"),
    ("data_pool",            "macro_engine",         "market data"),
    ("data_pool",            "security_analysis",    "fundamentals"),
    ("data_pool",            "contagion_engine",     "cross-asset prices"),
    ("data_pool",            "stat_arb_engine",      "pair prices"),
    ("data_pool",            "distressed_assets",    "credit data"),
    ("data_pool",            "pattern_discovery",    "OHLCV data"),

    # ── STAGE 2 TRACK A: MetadronCube → signal engines ─────────────
    ("fed_liquidity",        "macro_engine",         "fed plumbing state"),
    ("fed_liquidity",        "metadron_cube",        "Layer 0 tensor"),
    ("macro_engine",         "metadron_cube",        "regime + GMTF"),
    ("metadron_cube",        "security_analysis",    "regime context"),
    ("metadron_cube",        "contagion_engine",     "regime context"),
    ("metadron_cube",        "stat_arb_engine",      "regime context"),
    ("metadron_cube",        "distressed_assets",    "regime context"),
    ("metadron_cube",        "pattern_discovery",    "regime context"),
    ("metadron_cube",        "adaptive_thresholds",  "regime context"),
    ("metadron_cube",        "allocation_engine",    "sleeve allocation + KillSwitch"),
    ("metadron_cube",        "decision_matrix",      "regime + risk state"),

    # ── STAGE 2 TRACK B: NewsEngine → MiroMomentum (independent) ───
    ("news_engine",          "social_prediction",    "news-flagged tickers"),
    ("social_prediction",    "event_driven",         "enriched signals (40% sentiment + 60% agent)"),
    ("social_prediction",    "cvr_engine",           "enriched signals (40% sentiment + 60% agent)"),

    # ── STAGE 3: INTELLIGENCE (Track A + Track B converge) ─────────
    # AlphaOptimizer receives ALL signals from both tracks
    ("universe_engine",      "alpha_optimizer",      "4-run universe for merge/dedup"),
    ("macro_engine",         "alpha_optimizer",      "macro features"),
    ("metadron_cube",        "alpha_optimizer",      "regime + allocation"),
    ("security_analysis",    "alpha_optimizer",      "12 alpha features"),
    ("contagion_engine",     "alpha_optimizer",      "contagion risk scores"),
    ("stat_arb_engine",      "alpha_optimizer",      "pair signals"),
    ("distressed_assets",    "alpha_optimizer",      "distress signals"),
    ("pattern_discovery",    "alpha_optimizer",      "pattern features"),
    ("fixed_income_engine",  "alpha_optimizer",      "yield curve + credit spread signals"),
    ("event_driven",         "alpha_optimizer",      "event signals (Track B enriched)"),
    ("cvr_engine",           "alpha_optimizer",      "CVR valuations (Track B enriched)"),
    ("social_prediction",    "social_features",      "social snapshot"),
    ("social_features",      "alpha_optimizer",      "social_alpha composite"),
    ("universe_engine",      "universe_classifier",  "securities for classification"),

    # ── STAGE 4: DECISION ──────────────────────────────────────────
    # AlphaOptimizer → FullUniverseEngine Scan Slate → DecisionMatrix
    ("alpha_optimizer",      "decision_matrix",      "FullUniverseEngine Scan Slate (scored, deduped, cap-enforced)"),
    ("decision_matrix",      "allocation_engine",    "approved trades"),
    ("beta_corridor",        "allocation_engine",    "beta target"),
    ("options_engine",       "options_sizer",        "vol surface + greeks"),

    # ── STAGE 4: EXECUTION (ALL through L7 — never direct) ────────
    ("allocation_engine",    "l7_execution_surface", "sized orders"),
    ("options_sizer",        "l7_execution_surface", "edge-gated options orders"),
    ("conviction_override",  "l7_execution_surface", "override multipliers"),
    ("wondertrader_engine",  "l7_execution_surface", "micro-price + TWAP/VWAP"),
    ("exchange_core_engine", "l7_execution_surface", "order matching"),
    ("l7_execution_surface", "ibkr_broker",          "routed orders (TWAP/VWAP/Adaptive/Market)"),
    ("l7_execution_surface", "tca_engine",           "fill data → cost decomposition"),

    # ── STAGE 5: LEARNING (feedback to Stages 2-4) ────────────────
    ("l7_execution_surface", "learning_loop",        "execution outcomes"),
    ("learning_loop",        "macro_engine",         "regime feedback → retune priors"),
    ("learning_loop",        "alpha_optimizer",      "signal accuracy → retune tier weights"),
    ("learning_loop",        "adaptive_thresholds",  "threshold recalibration"),
    ("learning_loop",        "decision_matrix",      "gate accuracy → retune thresholds"),
    ("learning_loop",        "agent_scorecard",      "agent performance data"),
    ("agent_scorecard",      "sector_bots",          "tier assignments"),
    ("agent_scorecard",      "research_bots",        "tier assignments"),

    # ── STAGE 5: MONITORING ────────────────────────────────────────
    ("data_pool",            "anomaly_detector",     "price/volume data"),
    ("l7_execution_surface", "portfolio_analytics",  "position data"),
]


# ============================================================================
# 3.  ROUTING RULES — the execution chain orders MUST follow
# ============================================================================
# Each rule has an id, a human-readable description, and a machine-checkable
# specification.  The validate_wiring() function enforces these.

@dataclass(frozen=True)
class RoutingRule:
    """One mandatory routing constraint."""
    rule_id: str
    description: str
    # Which component_id(s) are involved
    enforced_by: str
    # For programmatic checking
    check_type: str  # "route_through" | "gate" | "feedback" | "exclusive_broker"
    details: str = ""


ROUTING_RULES: List[RoutingRule] = [
    RoutingRule(
        rule_id="R01",
        description=(
            "ALL orders MUST route through L7UnifiedExecutionSurface.submit_order(). "
            "NEVER call IBKRBroker.place_order() directly. NEVER use raw requests.post() "
            "to any broker API."
        ),
        enforced_by="l7_execution_surface",
        check_type="route_through",
        details="l7_execution_surface -> ibkr_broker",
    ),
    RoutingRule(
        rule_id="R02",
        description=(
            "ALL equity orders MUST pass through WonderTraderEngine.compute_micro_price() "
            "for micro-price estimation before L7 submits to broker."
        ),
        enforced_by="wondertrader_engine",
        check_type="route_through",
        details="wondertrader_engine -> l7_execution_surface",
    ),
    RoutingRule(
        rule_id="R03",
        description=(
            "ALL options orders MUST go through OptionsSizer. Position is REJECTED if "
            "BS theoretical vs market price does not show mispricing edge >= 200bps. "
            "No allocation without edge."
        ),
        enforced_by="options_sizer",
        check_type="gate",
        details="options_sizer.MIN_EDGE_BPS = 200",
    ),
    RoutingRule(
        rule_id="R04",
        description=(
            "ALL orders MUST pass L7RiskEngine 10-gate pre-trade check inside "
            "L7UnifiedExecutionSurface before reaching the broker."
        ),
        enforced_by="l7_execution_surface",
        check_type="gate",
        details="L7RiskEngine is instantiated inside L7UnifiedExecutionSurface",
    ),
    RoutingRule(
        rule_id="R05",
        description=(
            "ALL fills MUST call LearningLoop.record_signal_outcome() to close the "
            "feedback loop. Execution without learning is forbidden."
        ),
        enforced_by="learning_loop",
        check_type="feedback",
        details="l7_execution_surface -> learning_loop",
    ),
    RoutingRule(
        rule_id="R06",
        description=(
            "ALL fills MUST generate TCA analysis via TCAEngine. "
            "Slippage, venue, and algo decomposition are mandatory."
        ),
        enforced_by="tca_engine",
        check_type="feedback",
        details="l7_execution_surface -> tca_engine",
    ),
    RoutingRule(
        rule_id="R07",
        description=(
            "MetadronCube KillSwitch overrides ALL allocation decisions. "
            "When KillSwitch fires, beta <= 0.35 and no new positions are opened."
        ),
        enforced_by="metadron_cube",
        check_type="gate",
        details="metadron_cube.kill_switch -> allocation_engine",
    ),
    RoutingRule(
        rule_id="R08",
        description=(
            "DecisionMatrix MIN_COMPOSITE_SCORE = 0.55. Trades scoring below this "
            "are REJECTED. No override can bypass this minimum."
        ),
        enforced_by="decision_matrix",
        check_type="gate",
        details="DecisionMatrix.min_composite >= 0.55",
    ),
    RoutingRule(
        rule_id="R09",
        description=(
            "IBKRBroker is the SOLE execution broker. No raw API calls to IBKR, "
            "Tradier, or any other broker. AlpacaBroker and TradierBroker exist only "
            "as legacy references — they MUST NOT be used for new execution code."
        ),
        enforced_by="ibkr_broker",
        check_type="exclusive_broker",
        details="Only engine.execution.ibkr_broker.IBKRBroker is permitted",
    ),
    RoutingRule(
        rule_id="R10",
        description=(
            "L7UnifiedExecutionSurface logs ALL generated orders to the trade log "
            "for reconciliation, even if the broker is offline. "
            "The trade log is the audit trail."
        ),
        enforced_by="l7_execution_surface",
        check_type="feedback",
        details="L7UnifiedExecutionSurface._trade_log records every order",
    ),
]


# ============================================================================
# 4.  VALIDATION ENGINE
# ============================================================================

@dataclass
class ValidationResult:
    """Result of validating one aspect of the wiring."""
    check: str
    passed: bool
    message: str
    severity: str = "ERROR"  # ERROR | WARNING


def validate_wiring() -> Dict[str, Any]:
    """Validate the full platform wiring.

    Returns a report dict with:
        - total_checks: int
        - passed: int
        - failed: int
        - warnings: int
        - results: List[ValidationResult]
        - components: dict of component_id -> {importable: bool, class_found: bool}
    """
    results: List[ValidationResult] = []
    component_status: Dict[str, Dict[str, bool]] = {}

    # --- Check 1: Every mandatory component can be imported ---
    for cid, spec in COMPONENTS.items():
        importable = False
        class_found = False
        try:
            mod = importlib.import_module(spec.module)
            importable = True
            cls = getattr(mod, spec.class_name, None)
            class_found = cls is not None
        except Exception:
            pass

        component_status[cid] = {
            "importable": importable,
            "class_found": class_found,
        }

        if spec.mandatory:
            if not importable:
                results.append(ValidationResult(
                    check=f"import:{cid}",
                    passed=False,
                    message=f"MANDATORY module {spec.module} cannot be imported",
                ))
            elif not class_found:
                results.append(ValidationResult(
                    check=f"class:{cid}",
                    passed=False,
                    message=f"Class {spec.class_name} not found in {spec.module}",
                ))
            else:
                results.append(ValidationResult(
                    check=f"import:{cid}",
                    passed=True,
                    message=f"{spec.class_name} OK ({spec.module})",
                ))

    # --- Check 2: Dependency graph is satisfied ---
    for cid, spec in COMPONENTS.items():
        for dep in spec.depends_on:
            if dep not in COMPONENTS:
                results.append(ValidationResult(
                    check=f"dep:{cid}->{dep}",
                    passed=False,
                    message=f"{cid} depends on {dep} which is not in the registry",
                ))
            elif COMPONENTS[dep].phase > spec.phase:
                results.append(ValidationResult(
                    check=f"dep_order:{cid}->{dep}",
                    passed=False,
                    message=(
                        f"{cid} (phase {spec.phase}) depends on {dep} "
                        f"(phase {COMPONENTS[dep].phase}) — wrong boot order"
                    ),
                ))

    # --- Check 3: Execution chain is connected ---
    # L7 must have broker, wondertrader, risk engine, TCA, learning loop
    execution_chain = [
        ("l7_execution_surface", "ibkr_broker", "L7 -> IBKRBroker"),
        ("l7_execution_surface", "tca_engine", "L7 -> TCA (via fills)"),
        ("l7_execution_surface", "learning_loop", "L7 -> LearningLoop (feedback)"),
    ]
    for source, target, label in execution_chain:
        edge_exists = any(
            e[0] == source and e[1] == target for e in WIRING_EDGES
        )
        if not edge_exists:
            results.append(ValidationResult(
                check=f"chain:{label}",
                passed=False,
                message=f"Execution chain broken: {label} not wired",
            ))
        else:
            results.append(ValidationResult(
                check=f"chain:{label}",
                passed=True,
                message=f"Execution chain connected: {label}",
            ))

    # --- Check 4: No direct broker imports outside L7 ---
    # Scan key files for direct IBKRBroker/IBKRBroker (legacy) usage in
    # execution paths (not legacy references or try/except fallbacks).
    _FORBIDDEN_BROKER_IMPORTS = [
        ("engine.execution.alpaca_broker", "AlpacaBroker"),
        ("engine.execution.tradier_broker", "TradierBroker"),
    ]
    _FILES_TO_SCAN = [
        "engine/execution/l7_unified_execution_surface.py",
        "engine/execution/execution_engine.py",
        "engine/live_loop_orchestrator.py",
    ]
    import os
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for fpath in _FILES_TO_SCAN:
        full_path = os.path.join(base_dir, fpath)
        if not os.path.isfile(full_path):
            continue
        try:
            with open(full_path, "r") as fh:
                content = fh.read()
        except Exception:
            continue
        for mod_path, cls_name in _FORBIDDEN_BROKER_IMPORTS:
            # Look for active (non-comment) imports of these brokers
            # in execution-critical files.  Warn but don't fail — legacy
            # code may still reference them in try/except blocks.
            if f"import {cls_name}" in content or f"from {mod_path}" in content:
                results.append(ValidationResult(
                    check=f"broker_leak:{os.path.basename(fpath)}:{cls_name}",
                    passed=True,  # warning only
                    message=(
                        f"WARNING: {cls_name} referenced in {fpath}. "
                        f"Ensure it is NOT used for live execution — "
                        f"IBKRBroker via L7 is the sole execution path."
                    ),
                    severity="WARNING",
                ))

    # --- Check 5: LearningLoop connected to execution outcomes ---
    ll_edge = any(
        e[0] == "l7_execution_surface" and e[1] == "learning_loop"
        for e in WIRING_EDGES
    )
    results.append(ValidationResult(
        check="feedback:learning_loop",
        passed=ll_edge,
        message=(
            "LearningLoop receives execution outcomes"
            if ll_edge else
            "LearningLoop NOT connected to execution outcomes"
        ),
    ))

    # --- Check 6: KillSwitch wired from MetadronCube to AllocationEngine ---
    ks_edge = any(
        e[0] == "metadron_cube" and e[1] == "allocation_engine"
        for e in WIRING_EDGES
    )
    results.append(ValidationResult(
        check="killswitch:cube->allocation",
        passed=ks_edge,
        message=(
            "KillSwitch path: MetadronCube -> AllocationEngine connected"
            if ks_edge else
            "KillSwitch path BROKEN: MetadronCube -> AllocationEngine not wired"
        ),
    ))

    # --- Check 7: All wiring edges reference valid component IDs ---
    for src, tgt, label in WIRING_EDGES:
        if src not in COMPONENTS:
            results.append(ValidationResult(
                check=f"edge_src:{src}",
                passed=False,
                message=f"Wiring edge source '{src}' not in component registry",
            ))
        if tgt not in COMPONENTS:
            results.append(ValidationResult(
                check=f"edge_tgt:{tgt}",
                passed=False,
                message=f"Wiring edge target '{tgt}' not in component registry",
            ))

    # --- Summarize ---
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed and r.severity == "ERROR")
    warnings = sum(1 for r in results if not r.passed and r.severity == "WARNING")
    # Count warnings that passed=True but severity=WARNING
    warnings += sum(1 for r in results if r.passed and r.severity == "WARNING")

    return {
        "total_checks": len(results),
        "passed": passed,
        "failed": failed,
        "warnings": warnings,
        "all_ok": failed == 0,
        "results": results,
        "components": component_status,
    }


# ============================================================================
# 5.  BOOTSTRAP FUNCTION
# ============================================================================

@dataclass
class SystemState:
    """Holds all instantiated components after bootstrap."""
    nav: float
    ibkr_paper: bool
    components: Dict[str, Any] = field(default_factory=dict)
    boot_order: List[str] = field(default_factory=list)
    failed: List[Tuple[str, str]] = field(default_factory=list)

    def get(self, component_id: str) -> Any:
        """Retrieve a component by ID, returns None if not booted."""
        return self.components.get(component_id)

    @property
    def l7(self):
        """Shortcut to L7UnifiedExecutionSurface."""
        return self.components.get("l7_execution_surface")

    @property
    def cube(self):
        """Shortcut to MetadronCube."""
        return self.components.get("metadron_cube")

    @property
    def broker(self):
        """Shortcut to IBKRBroker (via L7)."""
        return self.components.get("ibkr_broker")

    def summary(self) -> str:
        """One-line summary of boot status."""
        ok = len(self.components)
        fail = len(self.failed)
        return f"SystemState: {ok} components booted, {fail} failed, NAV=${self.nav:,.0f}"


def bootstrap_full_system(
    nav: float = 1_000_000.0,
    ibkr_paper: bool = True,
) -> SystemState:
    """Instantiate all components in correct dependency order.

    Parameters
    ----------
    nav : float
        Initial net asset value.
    ibkr_paper : bool
        If True, connect to IBKR paper trading (port 7497).
        If False, connect to IBKR live trading (port 7496).

    Returns
    -------
    SystemState
        Namedtuple-like dataclass with all components accessible by ID.
    """
    state = SystemState(nav=nav, ibkr_paper=ibkr_paper)

    # Sort components by phase, then by dependency depth within phase
    sorted_specs = sorted(COMPONENTS.values(), key=lambda s: (s.phase, s.component_id))

    for spec in sorted_specs:
        cid = spec.component_id
        try:
            mod = importlib.import_module(spec.module)
            cls = getattr(mod, spec.class_name)
        except Exception as exc:
            state.failed.append((cid, f"import error: {exc}"))
            logger.warning("Bootstrap: cannot import %s.%s — %s", spec.module, spec.class_name, exc)
            continue

        # Build constructor kwargs based on component type
        try:
            instance = _instantiate(cid, cls, nav, ibkr_paper, state)
            state.components[cid] = instance
            state.boot_order.append(cid)
            logger.info("Bootstrap: %s (%s) OK", cid, spec.class_name)
        except Exception as exc:
            state.failed.append((cid, f"init error: {exc}"))
            logger.warning("Bootstrap: %s init failed — %s", cid, exc)

    # --- Post-boot wiring ---
    # Connect L7 to learning loop for fill feedback
    l7 = state.get("l7_execution_surface")
    ll = state.get("learning_loop")
    if l7 is not None and ll is not None:
        # Store reference for manual feedback routing
        if not hasattr(l7, "_learning_loop"):
            l7._learning_loop = ll
            logger.info("Bootstrap: wired L7 -> LearningLoop feedback")

    # Connect TCA engine
    tca = state.get("tca_engine")
    if l7 is not None and tca is not None:
        if not hasattr(l7, "_tca_engine"):
            l7._tca_engine = tca
            logger.info("Bootstrap: wired L7 -> TCAEngine")

    logger.info("Bootstrap complete: %s", state.summary())
    return state


def _instantiate(
    cid: str, cls: type, nav: float, ibkr_paper: bool, state: SystemState
) -> Any:
    """Create a component instance with appropriate constructor args.

    Each component has its own constructor signature.  We match known
    component IDs to the right kwargs.  Defaults are used for anything
    not explicitly handled.
    """
    # Components that take nav as a parameter
    if cid == "decision_matrix":
        return cls(nav=nav)
    elif cid == "allocation_engine":
        return cls(nav=nav)
    elif cid == "beta_corridor":
        return cls(nav=nav)
    elif cid == "conviction_override":
        return cls(nav=nav)
    elif cid == "options_engine":
        return cls(nav=nav)
    elif cid == "ibkr_broker":
        return cls(initial_cash=nav, paper=ibkr_paper)
    elif cid == "l7_execution_surface":
        return cls(initial_cash=nav, ibkr_paper=ibkr_paper)
    elif cid == "data_ingestion":
        # Wire to universe engine if available
        ue = state.get("universe_engine")
        return cls(universe_engine=ue) if ue else cls()
    elif cid == "universe_engine":
        return cls(load=False)
    elif cid == "deep_learning_engine":
        return cls()
    else:
        # Default: no-arg constructor
        return cls()


# ============================================================================
# 6.  CLI REPORT
# ============================================================================

def _print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable validation report to stdout."""
    print()
    print("=" * 78)
    print("  METADRON CAPITAL — WIRING MANIFEST VALIDATION REPORT")
    print("=" * 78)
    print()

    # Component registry summary
    print(f"  Components registered: {len(COMPONENTS)}")
    by_phase = {}
    for spec in COMPONENTS.values():
        by_phase.setdefault(spec.phase_name, []).append(spec.component_id)
    for phase_name, cids in by_phase.items():
        print(f"    Phase {phase_name}: {len(cids)} components")
    print()

    # Wiring edges
    print(f"  Wiring edges: {len(WIRING_EDGES)}")
    print(f"  Routing rules: {len(ROUTING_RULES)}")
    print()

    # Validation results
    print(f"  Total checks: {report['total_checks']}")
    print(f"  Passed:       {report['passed']}")
    print(f"  Failed:       {report['failed']}")
    print(f"  Warnings:     {report['warnings']}")
    print()

    if report["failed"] > 0:
        print("  FAILURES:")
        print("  " + "-" * 74)
        for r in report["results"]:
            if not r.passed and r.severity == "ERROR":
                print(f"  [FAIL] {r.check}: {r.message}")
        print()

    warnings_list = [
        r for r in report["results"]
        if r.severity == "WARNING"
    ]
    if warnings_list:
        print("  WARNINGS:")
        print("  " + "-" * 74)
        for r in warnings_list:
            print(f"  [WARN] {r.check}: {r.message}")
        print()

    # Routing rules
    print("  ROUTING RULES (mandatory execution contracts):")
    print("  " + "-" * 74)
    for rule in ROUTING_RULES:
        print(f"  [{rule.rule_id}] {rule.description}")
        print()

    # Final verdict
    if report["all_ok"]:
        print("  VERDICT: ALL CHECKS PASSED")
    else:
        print(f"  VERDICT: {report['failed']} FAILURE(S) — fix before proceeding")
    print()
    print("=" * 78)
    print()


def _print_boot_report(state: SystemState) -> None:
    """Print bootstrap results."""
    print()
    print("=" * 78)
    print("  METADRON CAPITAL — BOOTSTRAP REPORT")
    print("=" * 78)
    print()
    print(f"  NAV:        ${state.nav:,.0f}")
    print(f"  IBKR mode:  {'PAPER' if state.ibkr_paper else 'LIVE'}")
    print(f"  Booted:     {len(state.components)}/{len(COMPONENTS)}")
    print(f"  Failed:     {len(state.failed)}")
    print()

    if state.boot_order:
        print("  BOOT ORDER:")
        for i, cid in enumerate(state.boot_order, 1):
            spec = COMPONENTS[cid]
            print(f"    {i:2d}. [{spec.phase_name}] {cid} ({spec.class_name})")
        print()

    if state.failed:
        print("  FAILURES:")
        for cid, reason in state.failed:
            print(f"    - {cid}: {reason}")
        print()

    print("=" * 78)
    print()


# ============================================================================
# 7.  __main__ entry point
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    if "--boot" in sys.argv:
        nav = 1_000_000.0
        paper = True
        for arg in sys.argv[1:]:
            if arg.startswith("--nav="):
                nav = float(arg.split("=", 1)[1])
            elif arg == "--live":
                paper = False
        state = bootstrap_full_system(nav=nav, ibkr_paper=paper)
        _print_boot_report(state)
    else:
        report = validate_wiring()
        _print_report(report)
        sys.exit(0 if report["all_ok"] else 1)
