# Metadron Capital — Production Readiness Fix Instructions

**For:** Claude Code / AI coding assistant
**Date:** 2026-03-26
**Source:** Full codebase audit of Metadron-Capital
**Scope:** Backend engine, pipeline, risk controls, ML, infrastructure. NO frontend work.

---

## Context

This is a quant fund platform targeting $1,000 → $100,000 in 100 days via automated signal pipeline, ML-driven alpha extraction, and multi-broker execution. The codebase is a sophisticated prototype with sound architecture but critical operational gaps that prevent deployment.

**Working directory:** `/root/.openclaw/workspace/fund`

**Test before and after each fix:**
```bash
python3 -m pytest tests/ -v
```

---

## PHASE 1: MAKE THE PLATFORM RUN (Priority 1)

### Task 1.1 — Install Dependencies

The platform has zero Python dependencies installed. It cannot run.

```bash
cd /root/.openclaw/workspace/fund
python3 -m pip install --upgrade pip
python3 -m pip install numpy pandas scipy scikit-learn pyyaml python-dotenv rich matplotlib seaborn plotly
python3 -m pip install openbb  # Data provider
python3 -m pip install pytest  # Testing
```

**DO NOT install** (dead dependencies, save 5GB+):
- `torch` — not used in engine code
- `tensorflow` — not used in engine code
- `langchain`, `langchain-anthropic`, `langgraph` — not used in engine code
- `fredapi` — OpenBB already provides FRED access

After installing, update `pyproject.toml` to remove the dead dependencies:
- Remove: `torch>=2.0`, `tensorflow>=2.15`, `langchain`, `langchain-anthropic`, `langchain-openai`, `langgraph`, `fredapi>=0.5`
- Keep: everything else

**Verification:** `python3 -c "import numpy, pandas, scipy, sklearn; print('OK')"`

### Task 1.2 — Create .env from Template

```bash
cp .env.example .env
```

Fill in placeholder values if real keys aren't available. The platform should start and run in degraded mode (empty DataFrames) without real API keys. This is by design — `openbb_data.py` has graceful fallbacks.

**Verification:** `python3 -c "from engine.data.openbb_data import get_adj_close; print('OK')"` (should not crash)

### Task 1.3 — Fix `run_hourly.py` Broken Import

**File:** `run_hourly.py`
**Problem:** Imports `generate_sector_heatmap` from `engine.monitoring.daily_report` but the function is defined in `engine.monitoring.heatmap_engine`

**Fix:**
```python
# Change:
from engine.monitoring.daily_report import generate_sector_heatmap
# To:
from engine.monitoring.heatmap_engine import generate_sector_heatmap
```

**If `generate_sector_heatmap` doesn't exist in `heatmap_engine.py`**, find where it actually lives with:
```bash
grep -rn "def generate_sector_heatmap" engine/
```
and fix the import to point there.

**Verification:** `python3 run_hourly.py` should not crash on import (may crash later on missing data, that's fine)

---

## PHASE 2: FIX EXECUTION SAFETY (Priority 2)

### Task 2.1 — Fix Risk Gate G2 (Sector Concentration)

**File:** `engine/execution/execution_engine.py` — look for `RiskGateManager` class, specifically the G2 gate method.

**Problem:** The sector concentration check compares `p.sector == ticker` which never matches. The sector field on positions contains the ticker symbol, not the GICS sector name.

**Fix:** The position object needs to carry the actual GICS sector. In the `Security` dataclass (likely in `engine/data/universe_engine.py` or `engine/data/cross_asset_universe.py`), ensure there's a `sector` field with the GICS sector name (e.g., "Information Technology", "Health Care").

Then in RiskGate G2:
```python
# BEFORE (broken):
for ticker, p in positions.items():
    if p.sector == sector:  # p.sector is a ticker, never matches sector name
        sector_weight += p.weight

# AFTER (fixed):
from engine.data.cross_asset_universe import GICS_SECTOR_MAP
for ticker, p in positions.items():
    ticker_sector = GICS_SECTOR_MAP.get(ticker, "Unknown")
    if ticker_sector == sector:
        sector_weight += p.weight
```

**Also add a test:**
```python
def test_g2_sector_concentration_gate():
    """G2 should block trades that exceed sector concentration limit."""
    # Create positions all in same sector
    # Verify gate rejects over-concentration
```

**Verification:** Run tests, verify G2 actually blocks sector concentration.

### Task 2.2 — Add Order Persistence to PaperBroker

**File:** `engine/execution/paper_broker.py`

**Problem:** All state is in-memory. Process crash = portfolio lost.

**Fix:** Add JSON serialization to disk:

```python
import json
from pathlib import Path

class PaperBroker:
    STATE_FILE = Path("data/paper_broker_state.json")
    
    def __init__(self, ...):
        # ... existing init ...
        self.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()
    
    def _save_state(self):
        """Persist portfolio state to disk after every state change."""
        state = {
            "cash": self.cash,
            "positions": {k: asdict(v) for k, v in self.positions.items()},
            "trade_log": self.trade_log,
            "timestamp": datetime.now().isoformat()
        }
        self.STATE_FILE.write_text(json.dumps(state, indent=2, default=str))
    
    def _load_state(self):
        """Resume from last known state on startup."""
        if self.STATE_FILE.exists():
            state = json.loads(self.STATE_FILE.read_text())
            self.cash = state.get("cash", self.cash)
            # Reconstruct positions from saved state
            # ... 
    
    def execute_order(self, order):
        # ... existing logic ...
        self._save_state()  # Save after every trade
```

**Add a test:**
```python
def test_paper_broker_persistence():
    """Broker state survives process restart."""
    broker = PaperBroker(cash=10000)
    broker.execute_order(...)
    # Create new broker instance
    broker2 = PaperBroker()
    assert broker2.cash == broker.cash  # Restored from disk
```

### Task 2.3 — Fix Top-Level Imports in execution_engine.py

**File:** `engine/execution/execution_engine.py`

**Problem:** Non-optional top-level imports for 12+ sub-engines. If any single module fails to import (missing dep, syntax error), the entire platform crashes.

**Fix:** Wrap ALL sub-engine imports in try/except:

```python
# BEFORE:
from ..signals.social_prediction_engine import SocialPredictionEngine
from ..signals.distressed_asset_engine import DistressedAssetEngine
from ..signals.cvr_engine import CVREngine
from ..signals.event_driven_engine import EventDrivenEngine

# AFTER:
try:
    from ..signals.social_prediction_engine import SocialPredictionEngine
except ImportError as e:
    SocialPredictionEngine = None
    logging.warning(f"SocialPredictionEngine unavailable: {e}")

try:
    from ..signals.distressed_asset_engine import DistressedAssetEngine
except ImportError as e:
    DistressedAssetEngine = None
    logging.warning(f"DistressedAssetEngine unavailable: {e}")

# ... repeat for ALL top-level imports that aren't core dependencies
```

Then in `run_pipeline()`, check `if EngineClass is not None` before using each engine.

**The only imports that should remain non-optional:** numpy, pandas, the core data types (Security, MacroSnapshot, etc.), and the engines that are essential to the pipeline (UniverseEngine, MacroEngine, AlphaOptimizer, PaperBroker/TradierBroker).

**Verification:** Temporarily rename one signal engine file, verify the platform still starts.

### Task 2.4 — Remove Hardcoded Entry Price in platform_orchestrator.py

**File:** `platform_orchestrator.py`

**Problem:** Uses `entry = 100.0` as hardcoded placeholder.

**Options:**
- **Option A:** Remove `platform_orchestrator.py` entirely (recommended — it's a dead parallel system, see Task 3.1)
- **Option B:** If keeping it, replace `entry = 100.0` with actual price fetch from OpenBB

**Recommendation:** Option A. Delete the file.

---

## PHASE 3: ARCHITECTURE CLEANUP (Priority 3)

### Task 3.1 — Remove Dead Code: platform_orchestrator.py

**File:** `platform_orchestrator.py` (~1,000 lines)

**Problem:** This is a COMPLETELY SEPARATE system from `engine/`. It has its own `_TechnicalAnalyzer`, `_FundamentalAnalyzer`, `_MacroAnalyzer`, `_SentimentAnalyzer`, `_RiskManager`, `_CubeRotation`, `_ExecutionEngine`, `_MLLearner` — none of which connect to the actual engine modules. It also imports `distress_scanner` which doesn't exist.

**Fix:**
```bash
# Archive it, don't delete (it might have useful patterns)
mkdir -p archive/
mv platform_orchestrator.py archive/
```

Update any references to it:
```bash
grep -rn "platform_orchestrator" . --include="*.py"
```

Remove or update any imports that reference it.

### Task 3.2 — Remove Dead Code: core/ Orphaned Modules

**Files:** `core/platform.py`, `core/signals.py`, `core/portfolio.py`

**Problem:** These are the "original" core platform modules. They load repos.yaml and do status reporting, but nothing in the engine imports or uses them. Only `bootstrap.py` calls them.

**Fix:**
```bash
# Check what imports core/ modules
grep -rn "from core" . --include="*.py"
grep -rn "import core" . --include="*.py"

# If only bootstrap.py uses them:
mkdir -p archive/
mv core/platform.py core/signals.py core/portfolio.py archive/
# Keep core/__init__.py for now
```

**DO NOT delete `bootstrap.py`** — it might be used for initial setup. Review it first.

### Task 3.3 — Fix repos.yaml

**File:** `config/repos.yaml`

**Problems:**
1. **Duplicate key:** `stock_prediction` appears twice with different paths
2. **Missing repos:** worldmonitor, markov-model, wondertrader, exchange-core, qstrader, get-shit-done

**Fix:**
```yaml
# Remove duplicate stock_prediction entry
# Keep only one:
stock_prediction:
  repo: https://github.com/...
  layer: L2
  role: Stock price prediction models

# Add missing repos:
worldmonitor:
  repo: https://github.com/koala73/worldmonitor
  layer: L2
  role: Global real-time event monitoring (30+ categories)

markov_model:
  repo: https://github.com/...
  layer: L3
  role: Hidden Markov Models for regime detection

wondertrader:
  repo: https://github.com/...
  layer: L7
  role: CTA trend-following, HFT micro-price, TWAP/VWAP routing

exchange_core:
  repo: https://github.com/...
  layer: L7
  role: Ultra-low-latency order matching engine

qstrader:
  repo: https://github.com/...
  layer: L4
  role: Backtesting framework

get_shit_done:
  repo: https://github.com/...
  layer: L6
  role: GSD plugin for learning loop
```

### Task 3.4 — Split execution_engine.py Monolith

**File:** `engine/execution/execution_engine.py` (~1,500 lines)

**Problem:** Contains 8+ classes in one file. Hard to maintain, hard to test, hard to debug.

**Fix:** Extract into separate files:

```
engine/execution/
├── execution_engine.py          # Main orchestrator (keep, slim down)
├── micro_price.py               # MicroPriceEngine class
├── cross_asset_monitor.py       # CrossAssetMonitor class
├── risk_gates.py                # RiskGateManager class (8 gates)
├── deep_features.py             # DeepTradingFeatures class
├── ml_vote.py                   # MLVoteEnsemble class
├── trade_allocator.py           # TradeAllocator class
├── pipeline_tracker.py          # PipelineTracker class
└── __init__.py                  # Re-export all for backward compat
```

In `__init__.py`:
```python
from .micro_price import MicroPriceEngine
from .risk_gates import RiskGateManager
from .ml_vote import MLVoteEnsemble
# ... etc
```

**Keep backward compat:** All existing imports like `from engine.execution.execution_engine import MicroPriceEngine` should still work via the re-exports.

---

## PHASE 4: ML PIPELINE (Priority 4)

### Task 4.1 — Implement Model Persistence

**Files:** `engine/ml/model_evaluator.py`, `engine/signals/` (AlphaOptimizer)

**Problem:** No models are saved. Training is a no-op.

**Fix:** Add a simple model persistence layer:

```python
# engine/ml/model_store.py
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class ModelStore:
    """Simple model persistence using numpy + JSON metadata."""
    
    MODELS_DIR = Path("data/models")
    
    def __init__(self):
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def save(self, model_name: str, weights: np.ndarray, metadata: dict = None):
        """Save model weights + metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.MODELS_DIR / model_name
        path.mkdir(exist_ok=True)
        
        np.save(path / f"{timestamp}_weights.npy", weights)
        
        meta = {
            "model_name": model_name,
            "timestamp": timestamp,
            "shape": list(weights.shape),
            **(metadata or {})
        }
        (path / f"{timestamp}_meta.json").write_text(json.dumps(meta, indent=2))
    
    def load_latest(self, model_name: str) -> tuple:
        """Load most recent model weights + metadata."""
        path = self.MODELS_DIR / model_name
        if not path.exists():
            return None, None
        
        npy_files = sorted(path.glob("*_weights.npy"))
        if not npy_files:
            return None, None
        
        latest_npy = npy_files[-1]
        latest_meta = path / latest_npy.name.replace("_weights.npy", "_meta.json")
        
        weights = np.load(latest_npy)
        meta = json.loads(latest_meta.read_text()) if latest_meta.exists() else {}
        return weights, meta
```

Then wire it into `MLVoteEnsemble`:
```python
class MLVoteEnsemble:
    def __init__(self):
        self.store = ModelStore()
        self._load_or_init_models()
    
    def _load_or_init_models(self):
        """Load persisted models or initialize fresh."""
        for tier in range(1, 5):
            weights, meta = self.store.load_latest(f"tier{tier}")
            if weights is not None:
                self.models[tier] = weights
            else:
                self.models[tier] = self._init_model(tier)
    
    def train(self, tier, features, labels):
        """Train and persist."""
        # ... training logic ...
        self.store.save(f"tier{tier}", trained_weights, {"sharpe": sharpe, "samples": len(labels)})
```

### Task 4.2 — Wire Walk-Forward Validation

**File:** `engine/ml/backtester.py` (if exists) or create `engine/ml/walk_forward.py`

**Problem:** CLAUDE.md describes walk-forward validation. The code doesn't implement it.

**Fix:** Create a walk-forward validator:

```python
# engine/ml/walk_forward.py
import numpy as np
from dataclasses import dataclass
from typing import List, Callable

@dataclass
class WalkForwardResult:
    train_sharpe: float
    test_sharpe: float
    train_period: tuple
    test_period: tuple
    weights: np.ndarray

class WalkForwardValidator:
    """
    Expanding window walk-forward validation.
    
    Train on [0:t], test on [t:t+gap], slide forward.
    """
    
    def __init__(self, train_window: int = 252, test_window: int = 21, step: int = 21):
        self.train_window = train_window  # ~1 year
        self.test_window = test_window    # ~1 month
        self.step = step                  # monthly steps
    
    def validate(self, features: np.ndarray, labels: np.ndarray, 
                 train_fn: Callable, predict_fn: Callable) -> List[WalkForwardResult]:
        """
        Run walk-forward validation.
        
        Args:
            features: (T, N) feature matrix
            labels: (T,) target returns
            train_fn: (X_train, y_train) -> model_weights
            predict_fn: (model_weights, X_test) -> predictions
        
        Returns:
            List of WalkForwardResult for each fold
        """
        results = []
        T = len(features)
        
        for start in range(0, T - self.train_window - self.test_window, self.step):
            train_end = start + self.train_window
            test_end = train_end + self.test_window
            
            X_train = features[start:train_end]
            y_train = labels[start:train_end]
            X_test = features[train_end:test_end]
            y_test = labels[train_end:test_end]
            
            weights = train_fn(X_train, y_train)
            train_pred = predict_fn(weights, X_train)
            test_pred = predict_fn(weights, X_test)
            
            results.append(WalkForwardResult(
                train_sharpe=self._sharpe(train_pred, y_train),
                test_sharpe=self._sharpe(test_pred, y_test),
                train_period=(start, train_end),
                test_period=(train_end, test_end),
                weights=weights
            ))
        
        return results
    
    def _sharpe(self, predictions, actuals):
        """Compute Sharpe ratio of predictions."""
        returns = predictions * actuals  # Simple return proxy
        if returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)
```

Wire it into `ExecutionEngine.run_pipeline()` after the AlphaOptimizer step:
```python
# After alpha optimization, run walk-forward validation
if self.walk_forward and len(historical_features) > 273:  # Need at least 1 year + 1 month
    wf_results = self.walk_forward.validate(historical_features, historical_labels, ...)
    self.logger.info(f"Walk-forward OOS Sharpe: {np.mean([r.test_sharpe for r in wf_results]):.2f}")
```

### Task 4.3 — Fix Tier-1 Neural Net

**File:** `engine/execution/execution_engine.py` — `MLVoteEnsemble._tier1_neural()` method

**Problem:** Re-seeds random weights every call. Not learning anything.

**Fix:** Either:
- **Option A:** Remove the tier-1 neural net entirely and replace with a simple trained model (e.g., logistic regression on the features)
- **Option B:** Make it actually train and persist weights

**Recommendation:** Option A for now. Replace with:
```python
def _tier1_neural(self, features, regime):
    """Tier 1: Simple logistic regression on features."""
    # Use persisted weights from ModelStore
    weights, meta = self.store.load_latest("tier1")
    if weights is None:
        return 0.0  # No model yet, abstain
    
    # Linear model: score = sigmoid(X @ w)
    score = 1 / (1 + np.exp(-features @ weights))
    return float(np.clip(score - 0.5, -1, 1))  # Center around 0
```

---

## PHASE 5: TESTS (Priority 5)

### Task 5.1 — Run Existing Tests

```bash
cd /root/.openclaw/workspace/fund
python3 -m pytest tests/ -v
```

Fix any failures. Expected failures: import errors (deps), missing modules.

### Task 5.2 — Add Missing Critical Tests

**File:** `tests/test_execution_pipeline.py` (new)

```python
"""Tests for the main execution pipeline."""

def test_pipeline_runs_without_crashing():
    """ExecutionEngine.run_pipeline() completes without error."""
    from engine.execution.execution_engine import ExecutionEngine
    engine = ExecutionEngine(broker_type="paper")
    result = engine.run_pipeline()
    assert result is not None
    assert hasattr(result, 'trades')

def test_pipeline_with_empty_data():
    """Pipeline handles empty market data gracefully."""
    # Should not crash, should return empty result

def test_pipeline_with_partial_engines():
    """Pipeline works when some sub-engines are unavailable."""
    # Simulate ImportError for one engine, verify pipeline still runs
```

**File:** `tests/test_ml_vote.py` (new)

```python
"""Tests for ML vote ensemble."""

def test_vote_deterministic():
    """Same input produces same vote."""
    from engine.execution.execution_engine import MLVoteEnsemble
    ensemble = MLVoteEnsemble()
    features = np.random.randn(10)
    vote1 = ensemble.vote(features, "BULL")
    vote2 = ensemble.vote(features, "BULL")
    assert vote1 == vote2

def test_vote_range():
    """Vote is always in [-1, 1]."""
    ensemble = MLVoteEnsemble()
    for _ in range(100):
        features = np.random.randn(10)
        vote = ensemble.vote(features, "BULL")
        assert -1 <= vote <= 1
```

**File:** `tests/test_monitoring_smoke.py` (new)

```python
"""Smoke tests for monitoring modules."""

def test_daily_report_generates():
    """Daily report produces non-empty output."""
    from engine.monitoring.daily_report import DailyReport
    report = DailyReport()
    output = report.generate(portfolio_state={})
    assert len(output) > 0

def test_learning_loop_records():
    """Learning loop accepts signal outcomes."""
    from engine.monitoring.learning_loop import LearningLoop
    loop = LearningLoop()
    loop.record_signal_outcome("ticker", "signal_type", 0.5, "BULL", True)
    assert len(loop.outcomes) > 0
```

---

## PHASE 6: CLEANUP & DOCUMENTATION

### Task 6.1 — Update CLAUDE.md

After all fixes, update CLAUDE.md to reflect:
- Current dependency list (removed torch/tensorflow/langchain)
- Updated architecture (no more platform_orchestrator.py)
- Current test status
- What's production-ready vs what's still in progress

### Task 6.2 — Add Run Scripts Validation

Add a simple smoke test script:

```bash
#!/bin/bash
# validate.sh — Pre-deployment validation
set -e

echo "=== Dependency Check ==="
python3 -c "import numpy, pandas, scipy, sklearn, yaml; print('Dependencies OK')"

echo "=== Import Check ==="
python3 -c "from engine.execution.execution_engine import ExecutionEngine; print('Engine imports OK')"

echo "=== Test Suite ==="
python3 -m pytest tests/ -v --tb=short

echo "=== Pipeline Smoke Test ==="
python3 -c "
from engine.execution.execution_engine import ExecutionEngine
engine = ExecutionEngine(broker_type='paper')
print('Engine initialized OK')
"

echo "=== ALL CHECKS PASSED ==="
```

---

## EXECUTION ORDER

1. **Phase 1** (Tasks 1.1–1.3) — Make it run. ~1 hour.
2. **Phase 2** (Tasks 2.1–2.4) — Fix execution safety. ~3-4 hours.
3. **Phase 3** (Tasks 3.1–3.4) — Clean architecture. ~2-3 hours.
4. **Phase 4** (Tasks 4.1–4.3) — ML pipeline. ~4-6 hours.
5. **Phase 5** (Tasks 5.1–5.2) — Tests. ~2-3 hours.
6. **Phase 6** (Tasks 6.1–6.2) — Cleanup. ~1 hour.

**Total estimated effort: 13-18 hours of coding work.**

---

## WHAT THIS DOCUMENT DOES NOT COVER

- ❌ Frontend/UI development (Phase 5 of the deployment plan)
- ❌ Pattern/edge identification (separate build)
- ❌ Live broker integration testing (Tradier sandbox needs API keys)
- ❌ Hetzner deployment automation (separate task)
- ❌ Performance optimization (only after fundamentals are solid)
- ❌ Adding new trading strategies (only fixing what exists)

---

*Generated by Bobby. Take this to Claude, get it done, bring it back clean.*
