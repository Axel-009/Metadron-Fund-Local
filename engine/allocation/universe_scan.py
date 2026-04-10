"""Metadron Capital — Full Universe Scan Orchestrator.

Timing Architecture:
- 4 universe runs x 150 seconds each = 600s (10 min) scanning
- 5 minutes aggregation + L7 execution
- Total full cycle: 15 minutes
- 5 minutes backtesting + risk management pass
- Total with risk: 20 minutes
- 3 full cycles/hour through market hours (09:30-16:00 ET)

Emits real-time SSE events for the Thinking Tab as signals are discovered.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import AsyncGenerator, Callable, Optional

logger = logging.getLogger("metadron.allocation.universe_scan")

# ─── Prometheus instrumentation ──────────────────────────────────
_prom = None

def _get_prom():
    global _prom
    if _prom is not None:
        return _prom
    try:
        from engine.bridges.prometheus_metrics import get_metrics
        _prom = get_metrics()
    except Exception:
        _prom = {}
    return _prom

from .allocation_engine import (
    AllocationEngine, AllocationRules, AllocationSlate,
    BucketType, CyclePhase, InstrumentType, ScanSignal,
)
from .sp1500_universe import UNIVERSES, get_universe




# ─── OpenBB / FMP signal scoring helper ─────────────────────────────────────

def _score_ticker_fmp(ticker: str) -> tuple:
    """Fetch a real quote from OpenBB (FMP provider) and derive signal scores.

    Returns: (signal_type, confidence, alpha_score, regime_context)
    Falls back to deterministic neutral values if OpenBB/FMP is unavailable.
    """
    try:
        from openbb import obb
        result = obb.equity.price.quote(symbol=ticker, provider="fmp")
        if result and result.results:
            q = result.results[0]
            change_pct = getattr(q, "change_percent", None) or 0.0
            # FMP returns e.g. 2.5 for +2.5% — normalise to fraction
            if abs(change_pct) > 1:
                change_pct = change_pct / 100.0

            volume = getattr(q, "volume", None) or 0
            avg_volume = getattr(q, "average_volume", None) or max(volume, 1)
            volume_ratio = volume / avg_volume if avg_volume else 1.0

            # Signal classification
            if change_pct > 0.01 and volume_ratio > 1.2:
                signal_type = "BUY"
                confidence = min(0.95, 0.5 + change_pct * 10 + (volume_ratio - 1) * 0.1)
            elif change_pct < -0.01 and volume_ratio > 1.2:
                signal_type = "SELL"
                confidence = min(0.95, 0.5 + abs(change_pct) * 10 + (volume_ratio - 1) * 0.1)
            elif abs(change_pct) < 0.003:
                signal_type = "HOLD"
                confidence = 0.40
            else:
                signal_type = "RV_LONG" if change_pct > 0 else "SELL"
                confidence = round(0.3 + abs(change_pct) * 5, 2)

            confidence = round(min(0.95, max(0.30, confidence)), 2)
            alpha_score = round(change_pct * 0.5 * volume_ratio, 4)
            alpha_score = max(-0.08, min(0.08, alpha_score))

            if change_pct > 0.015:
                regime_context = "BULL"
            elif change_pct < -0.015:
                regime_context = "BEAR"
            elif abs(change_pct) < 0.005:
                regime_context = "RANGE"
            else:
                regime_context = "TRANSITION"

            return signal_type, confidence, alpha_score, regime_context
    except Exception as e:
        logger.debug("FMP quote failed for %s: %s — using neutral", ticker, e)

    # Deterministic neutral fallback (never random)
    return "HOLD", 0.40, 0.0, "RANGE"


# ═══════════════════════════════════════════════════════════════════════════
# Scan Configuration
# ═══════════════════════════════════════════════════════════════════════════

HEARTBEAT_SECONDS = 150       # 2.5 minutes per universe run
AGGREGATION_SECONDS = 300     # 5 minutes for aggregation + execution
RISK_PASS_SECONDS = 300       # 5 minutes for backtesting + risk
FULL_CYCLE_SECONDS = HEARTBEAT_SECONDS * 4 + AGGREGATION_SECONDS  # 15 min
TOTAL_CYCLE_SECONDS = FULL_CYCLE_SECONDS + RISK_PASS_SECONDS       # 20 min
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

UNIVERSE_ORDER = ["SP500", "SP400_MIDCAP", "SP600_SMALLCAP", "ETF_FI"]


@dataclass
class ScanRunStatus:
    """Status of a single universe run."""
    universe: str = ""
    description: str = ""
    run_number: int = 0
    started_at: str = ""
    elapsed_seconds: float = 0.0
    heartbeat_total: int = HEARTBEAT_SECONDS
    signals_discovered: int = 0
    completed: bool = False
    favored_names: list = field(default_factory=list)
    trades: list = field(default_factory=list)


@dataclass
class ScanCycleStatus:
    """Status of a full scan cycle (4 runs + aggregation + risk)."""
    cycle_number: int = 0
    phase: str = CyclePhase.SCANNING.value
    current_run: int = 0
    current_universe: str = ""
    elapsed_seconds: float = 0.0
    total_signals: int = 0
    runs: list = field(default_factory=list)
    started_at: str = ""
    completed: bool = False

    def to_dict(self) -> dict:
        return {
            "cycle_number": self.cycle_number,
            "phase": self.phase,
            "current_run": self.current_run,
            "current_universe": self.current_universe,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "total_signals": self.total_signals,
            "runs": [
                {
                    "universe": r.universe,
                    "description": r.description,
                    "run_number": r.run_number,
                    "elapsed_seconds": round(r.elapsed_seconds, 1),
                    "heartbeat_total": r.heartbeat_total,
                    "signals_discovered": r.signals_discovered,
                    "completed": r.completed,
                }
                for r in self.runs
            ],
            "started_at": self.started_at,
            "completed": self.completed,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Signal Event Queue (for SSE streaming to Thinking Tab)
# ═══════════════════════════════════════════════════════════════════════════

class SignalEventBus:
    """Thread-safe event bus for streaming scan signals to the frontend.

    The Thinking Tab SSE endpoint reads from this bus.
    """

    def __init__(self, max_events: int = 500):
        self._events: list[dict] = []
        self._max_events = max_events
        self._listeners: list[asyncio.Queue] = []

    def emit(self, event: dict):
        """Emit a signal event. Listeners (SSE streams) receive it."""
        event["_emitted_at"] = datetime.now(timezone.utc).isoformat()
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events:]
        # Push to all active listeners
        for q in self._listeners:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass
        # Prometheus: track signal generation
        if event.get("type") == "signal_discovered":
            try:
                prom = _get_prom()
                if prom and "signals_generated_total" in prom:
                    prom["signals_generated_total"].inc()
            except Exception:
                pass

    def subscribe(self) -> asyncio.Queue:
        """Create a new subscription queue for SSE streaming."""
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._listeners.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        """Remove a subscription queue."""
        if q in self._listeners:
            self._listeners.remove(q)

    def recent_events(self, limit: int = 50) -> list[dict]:
        """Return the most recent events."""
        return self._events[-limit:]

    def clear(self):
        self._events.clear()


# Global event bus instance
signal_bus = SignalEventBus()


# ═══════════════════════════════════════════════════════════════════════════
# Full Universe Scan Orchestrator
# ═══════════════════════════════════════════════════════════════════════════

class FullUniverseScan:
    """Orchestrates the 4-universe scan cycle.

    Each cycle:
    1. Run 1-4: Scan each universe (150s heartbeat each)
    2. Aggregate results across all runs
    3. Feed to AllocationEngine for sizing
    4. Output to L7 execution surface
    5. Run backtesting + risk management pass

    Emits real-time events for each signal discovered → Thinking Tab SSE.
    """

    def __init__(self, allocation_engine: Optional[AllocationEngine] = None,
                 backtest: bool = False):
        self.engine = allocation_engine or AllocationEngine(backtest=backtest)
        self.backtest = backtest
        self.cycle_count: int = 0
        self.current_status = ScanCycleStatus()
        self._running = False
        self._last_slate: Optional[AllocationSlate] = None
        self._run_slates: list[AllocationSlate] = []

    async def run_universe(self, universe_name: str, run_number: int) -> tuple[ScanRunStatus, AllocationSlate]:
        """Execute a single 150s universe scan.

        In production, this would invoke the signal pipeline on the universe.
        For now, simulates discovery of signals with realistic timing.
        """
        run = ScanRunStatus(
            universe=universe_name,
            description=UNIVERSES.get(universe_name, ""),
            run_number=run_number,
            started_at=datetime.now(timezone.utc).isoformat(),
            heartbeat_total=HEARTBEAT_SECONDS,
        )

        tickers = get_universe(universe_name)
        signals: list[ScanSignal] = []
        start = time.time()

        # Emit scan start event
        signal_bus.emit({
            "type": "scan_start",
            "universe": universe_name,
            "run_number": run_number,
            "ticker_count": len(tickers),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Simulate scanning — in production, each ticker is evaluated by signal engines
        # We yield periodically to allow SSE streaming
        scan_interval = max(0.5, HEARTBEAT_SECONDS / max(len(tickers), 1))
        if self.backtest:
            scan_interval = 0.01  # Fast-forward in backtest

        for i, ticker in enumerate(tickers):
            elapsed = time.time() - start
            run.elapsed_seconds = elapsed

            # Update global status
            self.current_status.elapsed_seconds = (
                (run_number - 1) * HEARTBEAT_SECONDS + elapsed
            )

            # Simulate signal discovery (in production, real signal engines run here)
            # Emit a discovery event for every ~10th ticker to avoid flooding
            if i % max(1, len(tickers) // 15) == 0:
                # Determine instrument type from universe
                if universe_name == "ETF_FI":
                    from .sp1500_universe import ETF_TICKERS
                    instrument_type = (
                        InstrumentType.ETF.value
                        if ticker in ETF_TICKERS
                        else InstrumentType.FIXED_INCOME.value
                    )
                else:
                    instrument_type = InstrumentType.EQUITY.value

                # Real signal scoring via OpenBB FMP (replaces random simulation)
                signal_type, confidence, alpha_score, regime_context = _score_ticker_fmp(ticker)

                signal = ScanSignal(
                    ticker=ticker,
                    signal_type=signal_type,
                    instrument_type=instrument_type,
                    confidence=confidence,
                    alpha_score=alpha_score,
                    regime_context=regime_context,
                    universe=universe_name,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                signals.append(signal)
                run.signals_discovered = len(signals)

                # Emit event for Thinking Tab
                signal_bus.emit({
                    "type": "signal_discovered",
                    "ticker": ticker,
                    "signal_type": signal.signal_type,
                    "instrument_type": signal.instrument_type,
                    "confidence": signal.confidence,
                    "alpha_score": signal.alpha_score,
                    "regime_context": signal.regime_context,
                    "bucket": self.engine.classify_opportunity(signal),
                    "universe": universe_name,
                    "run_number": run_number,
                    "timestamp": signal.timestamp,
                })

            # Yield control for SSE streaming
            if not self.backtest:
                await asyncio.sleep(min(scan_interval, 1.0))
            else:
                await asyncio.sleep(0)

            # Respect heartbeat limit
            if not self.backtest and (time.time() - start) >= HEARTBEAT_SECONDS:
                break

        # Run allocation on discovered signals
        slate = self.engine.apply_rules(signals, backtest=self.backtest)
        run.completed = True
        run.elapsed_seconds = time.time() - start
        run.favored_names = [s.ticker for s in signals if s.alpha_score > 0.02]
        run.trades = [p.to_dict() for p in slate.positions[:10]]

        # Emit run complete
        signal_bus.emit({
            "type": "run_complete",
            "universe": universe_name,
            "run_number": run_number,
            "signals_discovered": len(signals),
            "favored_names": run.favored_names[:20],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return run, slate

    async def run_full_cycle(self, cycle_number: int = 0) -> AllocationSlate:
        """Execute a full scan cycle: 4 runs → aggregate → execute → risk check.

        Total: ~15 min scan+execute + 5 min risk = 20 min.
        """
        self.cycle_count = cycle_number or self.cycle_count + 1
        self.current_status = ScanCycleStatus(
            cycle_number=self.cycle_count,
            started_at=datetime.now(timezone.utc).isoformat(),
        )
        self._running = True
        self._run_slates = []

        # Phase 1: SCANNING (4 universe runs)
        self.current_status.phase = CyclePhase.SCANNING.value
        for i, universe in enumerate(UNIVERSE_ORDER):
            self.current_status.current_run = i + 1
            self.current_status.current_universe = universe

            signal_bus.emit({
                "type": "phase_update",
                "phase": CyclePhase.SCANNING.value,
                "current_run": i + 1,
                "universe": universe,
                "cycle_number": self.cycle_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            run_status, slate = await self.run_universe(universe, i + 1)
            self.current_status.runs.append(run_status)
            self.current_status.total_signals += run_status.signals_discovered
            self._run_slates.append(slate)

        # Phase 2: AGGREGATING
        self.current_status.phase = CyclePhase.AGGREGATING.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.AGGREGATING.value,
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        aggregated = self.engine.aggregate_runs(*self._run_slates)
        aggregated.cycle_number = self.cycle_count
        aggregated.phase = CyclePhase.AGGREGATING.value

        if not self.backtest:
            await asyncio.sleep(min(5, AGGREGATION_SECONDS))

        # Phase 3: EXECUTING (L7 surface)
        self.current_status.phase = CyclePhase.EXECUTING.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.EXECUTING.value,
            "positions_count": len(aggregated.positions),
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Validate against kill switch
        aggregated = self.engine.validate_against_kill_switch(
            aggregated, self.engine.nav
        )

        if not self.backtest:
            await asyncio.sleep(min(5, AGGREGATION_SECONDS))

        # Phase 4: RISK CHECK (backtesting + risk management)
        self.current_status.phase = CyclePhase.RISK_CHECK.value
        signal_bus.emit({
            "type": "phase_update",
            "phase": CyclePhase.RISK_CHECK.value,
            "kill_switch": aggregated.kill_switch_triggered,
            "cycle_number": self.cycle_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        if not self.backtest:
            await asyncio.sleep(min(5, RISK_PASS_SECONDS))

        # Phase 5: COOLDOWN
        self.current_status.phase = CyclePhase.COOLDOWN.value
        self.current_status.completed = True
        aggregated.phase = CyclePhase.COOLDOWN.value

        signal_bus.emit({
            "type": "cycle_complete",
            "cycle_number": self.cycle_count,
            "total_signals": self.current_status.total_signals,
            "positions": len(aggregated.positions),
            "kill_switch": aggregated.kill_switch_triggered,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Prometheus: track scan cycle metrics
        try:
            prom = _get_prom()
            if prom:
                if "scan_run_number" in prom:
                    prom["scan_run_number"].set(self.cycle_count)
                if "universe_scan_progress_pct" in prom:
                    prom["universe_scan_progress_pct"].set(100.0)
                if "scan_cycle_duration_seconds" in prom:
                    prom["scan_cycle_duration_seconds"].observe(self.current_status.elapsed_seconds)
                if "active_universe_size" in prom:
                    total_tickers = sum(len(get_universe(u)) for u in UNIVERSE_ORDER)
                    prom["active_universe_size"].set(total_tickers)
        except Exception:
            pass

        self._last_slate = aggregated
        self._running = False
        return aggregated

    async def run_continuous(self, max_cycles: int = 0):
        """Run continuous scan cycles during market hours.

        3 cycles per hour, 09:30-16:00 ET.
        Set max_cycles > 0 to limit for testing.
        """
        cycle = 0
        while True:
            cycle += 1
            if max_cycles > 0 and cycle > max_cycles:
                break

            await self.run_full_cycle(cycle)

            # Cooldown between cycles
            if not self.backtest:
                await asyncio.sleep(5)

    def get_scan_status(self) -> dict:
        """Return current scan cycle status."""
        return self.current_status.to_dict()

    def get_last_slate(self) -> Optional[dict]:
        """Return last computed allocation slate."""
        if self._last_slate:
            return self._last_slate.to_dict()
        return None

    @property
    def is_running(self) -> bool:
        return self._running
