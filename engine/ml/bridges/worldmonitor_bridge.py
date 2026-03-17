"""WorldMonitor Bridge — Global Event Feed → EventDrivenEngine + MacroEngine.

Connects the WorldMonitor real-time global monitoring platform to Metadron Capital's
event-driven and macro engines. WorldMonitor provides 30+ data categories including:
    - market: real-time market data & sentiment
    - economic: GDP, inflation, employment, central bank decisions
    - news: global news aggregation & classification
    - conflict/military: geopolitical risk events
    - supply-chain: supply chain disruption monitoring
    - trade: global trade flow & tariff events
    - cyber: cybersecurity threat intelligence
    - natural/climate/wildfire/seismology: natural disaster events
    - unrest: civil unrest & political instability

Integration Points:
    1. EventDrivenEngine (engine/signals/event_driven_engine.py)
       - Feeds REGULATORY, CREDIT_EVENT, MGMT_CHANGE categories
       - Geopolitical events → political beta adjustment
       - Supply chain disruptions → sector impact scoring

    2. MacroEngine (engine/signals/macro_engine.py)
       - Economic indicators → regime confirmation
       - Central bank decisions → GMTF gamma updates
       - Trade/tariff events → sector rotation signals

Source repo: repos/worldmonitor (https://github.com/koala73/worldmonitor)

Usage:
    from engine.ml.bridges.worldmonitor_bridge import WorldMonitorBridge
    bridge = WorldMonitorBridge()
    events = bridge.fetch_market_events()
    macro_signals = bridge.fetch_macro_signals()
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# WorldMonitor source location
WORLDMONITOR_DIR = Path(__file__).parent.parent.parent.parent / "repos" / "worldmonitor"

# Event categories from WorldMonitor API that map to Metadron engines
EVENT_CATEGORY_MAP = {
    # WorldMonitor category → EventDrivenEngine category
    "market": "CAPITAL_STRUCTURE",
    "economic": "REGULATORY",
    "conflict": "REGULATORY",
    "military": "REGULATORY",
    "cyber": "CREDIT_EVENT",
    "supply-chain": "RESTRUCTURING",
    "trade": "REGULATORY",
    "unrest": "REGULATORY",
    "news": "MGMT_CHANGE",
}

MACRO_CATEGORY_MAP = {
    # WorldMonitor category → MacroEngine signal type
    "economic": "regime_confirmation",
    "trade": "sector_rotation",
    "market": "sentiment_update",
    "eia": "energy_signal",
    "climate": "commodity_impact",
}

# Impact severity scaling
SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.75,
    "medium": 0.50,
    "low": 0.25,
    "info": 0.10,
}


@dataclass
class WorldMonitorEvent:
    """Normalized event from WorldMonitor feed."""
    category: str = ""
    title: str = ""
    severity: str = "medium"
    region: str = ""
    tickers: list[str] = field(default_factory=list)
    sectors: list[str] = field(default_factory=list)
    sentiment: float = 0.0       # [-1, 1]
    impact_score: float = 0.0    # [0, 1]
    timestamp: str = ""
    source_url: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class WorldMonitorMacroSignal:
    """Macro signal derived from WorldMonitor data."""
    signal_type: str = ""        # regime_confirmation, sector_rotation, etc.
    direction: int = 0           # -1, 0, +1
    strength: float = 0.0       # [0, 1]
    confidence: float = 0.0     # [0, 1]
    regime_impact: str = ""     # BULL/BEAR/STRESS/CRASH
    sector_weights: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class WorldMonitorBridge:
    """Bridge between WorldMonitor global event feed and Metadron Capital engines.

    Provides:
    - Event normalization for EventDrivenEngine
    - Macro signal extraction for MacroEngine
    - Geopolitical risk scoring for MetadronCube
    - Supply chain disruption impact for sector allocation
    """

    # Geopolitical risk keywords and their severity
    GEO_RISK_KEYWORDS = {
        "war": 1.0, "invasion": 1.0, "nuclear": 1.0, "sanctions": 0.8,
        "embargo": 0.8, "conflict": 0.7, "military": 0.6, "missile": 0.8,
        "attack": 0.7, "coup": 0.9, "revolution": 0.8, "blockade": 0.7,
        "tariff": 0.5, "trade war": 0.7, "cyber attack": 0.6,
    }

    # Sector impact map for geopolitical events by region
    REGION_SECTOR_IMPACT = {
        "middle_east": {"XLE": 0.9, "XLF": 0.3, "XLI": 0.4},
        "europe": {"XLF": 0.6, "XLI": 0.5, "XLE": 0.4},
        "asia_pacific": {"XLK": 0.7, "XLI": 0.6, "XLY": 0.4},
        "americas": {"XLF": 0.5, "XLK": 0.4, "XLI": 0.5},
    }

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url
        self._events_cache: list[WorldMonitorEvent] = []
        self._macro_cache: list[WorldMonitorMacroSignal] = []
        self._geo_risk_score: float = 0.0
        logger.info(f"WorldMonitor bridge initialized (source: {WORLDMONITOR_DIR})")

    def fetch_market_events(self, categories: Optional[list[str]] = None) -> list[WorldMonitorEvent]:
        """Fetch and normalize market-relevant events from WorldMonitor.

        Args:
            categories: Filter by WorldMonitor categories. Default: all market-relevant.

        Returns:
            List of normalized WorldMonitorEvent for EventDrivenEngine consumption.
        """
        target_categories = categories or list(EVENT_CATEGORY_MAP.keys())
        events = []

        for category in target_categories:
            category_events = self._fetch_category_events(category)
            events.extend(category_events)

        # Sort by impact score
        events.sort(key=lambda e: e.impact_score, reverse=True)
        self._events_cache = events
        logger.info(f"WorldMonitor: fetched {len(events)} market events")
        return events

    def fetch_macro_signals(self) -> list[WorldMonitorMacroSignal]:
        """Extract macro signals from WorldMonitor economic/trade data.

        Returns:
            List of macro signals for MacroEngine consumption.
        """
        signals = []

        for category, signal_type in MACRO_CATEGORY_MAP.items():
            category_events = self._fetch_category_events(category)
            if category_events:
                signal = self._aggregate_to_macro_signal(category_events, signal_type)
                signals.append(signal)

        self._macro_cache = signals
        logger.info(f"WorldMonitor: generated {len(signals)} macro signals")
        return signals

    def get_geopolitical_risk_score(self) -> float:
        """Compute aggregate geopolitical risk score [0, 1].

        Used by MetadronCube RiskStateModel to adjust R(t).
        """
        if not self._events_cache:
            self.fetch_market_events(["conflict", "military", "unrest", "cyber"])

        risk_events = [e for e in self._events_cache
                       if e.category in ("conflict", "military", "unrest", "cyber")]

        if not risk_events:
            self._geo_risk_score = 0.0
            return 0.0

        # Weighted sum of severity scores
        total_risk = sum(
            SEVERITY_WEIGHTS.get(e.severity, 0.25) * e.impact_score
            for e in risk_events
        )
        # Normalize to [0, 1] with sigmoid
        self._geo_risk_score = float(1.0 / (1.0 + np.exp(-2.0 * (total_risk - 1.0))))
        return self._geo_risk_score

    def get_supply_chain_disruption_score(self) -> dict[str, float]:
        """Compute per-sector supply chain disruption scores.

        Returns:
            Dict of sector ETF → disruption score [0, 1].
        """
        sc_events = [e for e in self._events_cache if e.category == "supply-chain"]
        if not sc_events:
            return {}

        sector_scores = {}
        for event in sc_events:
            for sector in event.sectors:
                current = sector_scores.get(sector, 0.0)
                sector_scores[sector] = min(current + event.impact_score * 0.3, 1.0)

        return sector_scores

    def to_event_engine_format(self, events: Optional[list[WorldMonitorEvent]] = None) -> list[dict]:
        """Convert WorldMonitor events to EventDrivenEngine event dict format.

        Returns list of dicts compatible with EventDrivenEngine.__init__(events=...).
        """
        events = events or self._events_cache
        formatted = []

        for event in events:
            mapped_category = EVENT_CATEGORY_MAP.get(event.category, "REGULATORY")
            formatted.append({
                "category": mapped_category,
                "ticker": event.tickers[0] if event.tickers else "",
                "description": event.title,
                "catalyst_date": event.timestamp,
                "current_price": 0,  # To be filled by data engine
                "sentiment": event.sentiment,
                "impact_score": event.impact_score,
                "source": f"worldmonitor:{event.category}",
                "metadata": event.metadata,
            })

        return formatted

    def to_macro_engine_format(self, signals: Optional[list[WorldMonitorMacroSignal]] = None) -> dict:
        """Convert WorldMonitor macro signals to MacroEngine-compatible format.

        Returns dict with regime hints and sector weight adjustments.
        """
        signals = signals or self._macro_cache

        result = {
            "geo_risk_score": self._geo_risk_score,
            "regime_hints": [],
            "sector_adjustments": {},
            "signal_count": len(signals),
        }

        for signal in signals:
            if signal.regime_impact:
                result["regime_hints"].append({
                    "regime": signal.regime_impact,
                    "confidence": signal.confidence,
                    "source": signal.signal_type,
                })
            for sector, weight in signal.sector_weights.items():
                current = result["sector_adjustments"].get(sector, 0.0)
                result["sector_adjustments"][sector] = current + weight * signal.direction

        return result

    # --- Private helpers ---

    def _fetch_category_events(self, category: str) -> list[WorldMonitorEvent]:
        """Fetch events for a specific WorldMonitor category.

        Override this method to connect to the actual WorldMonitor API.
        Default implementation returns empty list (offline mode).
        """
        # Placeholder — actual implementation connects to WorldMonitor API
        # e.g., GET {base_url}/api/{category}/v1/events
        logger.debug(f"WorldMonitor: fetch {category} (offline mode — no events)")
        return []

    def _aggregate_to_macro_signal(self, events: list[WorldMonitorEvent],
                                    signal_type: str) -> WorldMonitorMacroSignal:
        """Aggregate multiple events into a single macro signal."""
        if not events:
            return WorldMonitorMacroSignal(signal_type=signal_type)

        avg_sentiment = float(np.mean([e.sentiment for e in events]))
        avg_impact = float(np.mean([e.impact_score for e in events]))
        direction = int(np.sign(avg_sentiment)) if abs(avg_sentiment) > 0.1 else 0

        # Determine regime impact
        regime_impact = ""
        if avg_sentiment > 0.3 and avg_impact > 0.5:
            regime_impact = "BULL"
        elif avg_sentiment < -0.3 and avg_impact > 0.5:
            regime_impact = "STRESS"
        elif avg_sentiment < -0.6 and avg_impact > 0.7:
            regime_impact = "CRASH"

        return WorldMonitorMacroSignal(
            signal_type=signal_type,
            direction=direction,
            strength=avg_impact,
            confidence=min(len(events) / 10, 1.0),
            regime_impact=regime_impact,
            metadata={"event_count": len(events), "avg_sentiment": avg_sentiment},
        )
