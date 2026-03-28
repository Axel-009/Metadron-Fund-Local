# Bobby Build Instructions — Go-Live Readiness

**Date:** 2026-03-27 | **Target:** Live execution trials tomorrow | **Branch:** bobby-agent-session

---

## SESSION SUMMARY

20 commits, 4,957 lines added, 153/153 tests green, 50/55 modules import clean.
Bobby wired Alpaca broker, MC risk engine, agent sim, model persistence, Black-Scholes dedup, full orchestrator pipeline. Strong B+ work.

---

## PHASE 1: SECURITY BLOCKERS (2.5 hrs) — MUST FIX BEFORE LIVE

### 1.1 Model Store Unsafe Deserialization (30 min)
**File:** `engine/ml/model_store.py:77`
- `joblib.load()` executes arbitrary code from malicious .joblib files
- **Fix:** Add HMAC-SHA256 signing on save, verify before every load
- **Fix:** Validate `name` param — reject if contains `..` or `/`

### 1.2 Alpaca Credential Silent Failure (20 min)
**File:** `engine/execution/alpaca_broker.py:161-162`
- Empty API keys default to `""` — broker inits but all trades fail silently
- Falls back to PaperBroker without telling you → you think you're live but you're paper
- **Fix:** Raise `ValueError` if keys empty. Validate with `get_account()` call on init
- **Fix:** Remove silent PaperBroker fallback in execution_engine.py when broker_type="alpaca"

### 1.3 Paper Broker State Corruption (40 min)
**File:** `engine/execution/paper_broker.py:902-926`
- `_load_state()` applies JSON values with zero validation — negative cash, NaN nav, phantom positions
- `_save_state()` non-atomic — crash mid-write = corrupted file
- **Fix:** Atomic writes (write to .tmp, then rename)
- **Fix:** Validate cash >= 0, nav > 0, quantity > 0, price > 0 on load

### 1.4 Log Credential Leakage (15 min)
**File:** `engine/execution/alpaca_broker.py:265-270`
- Alpaca API errors may include auth headers in exception objects
- **Fix:** Log only `status_code` and `code`, never full exception

### 1.5 Orchestrator None Guards (30 min)
**File:** `platform_orchestrator.py`
- 30+ engine imports set to None on failure, but called without None checks
- **Fix:** Add `if Engine is not None:` guard before every engine call in `daily_open_routine()`

### 1.6 Test Everything (15 min)
- Run `python3 -m pytest tests/ -v` — must stay 153/153

---

## PHASE 2: CLOSE THE LEARNING LOOP (2 hrs) — CRITICAL FOR ALPHA

The learning loop **logs everything but feeds nothing back**. This is the #1 gap vs Medallion.

### 2.1 Active Tier Weight Feedback (45 min)
**File:** `engine/monitoring/learning_loop.py`
- `compute_tier_weight_adjustments()` computes new weights but they aren't persisted
- **Fix:** Save adjusted weights to JSON after each session
- **Fix:** Load saved weights on next `ExecutionEngine.__init__()` and apply to `MLVoteEnsemble`

### 2.2 Regime Calibration Feedback (30 min)
- `record_regime_feedback()` tracks predicted vs realized regimes
- **Fix:** When regime accuracy < 60% over 20 samples, log WARNING and bias MetadronCube toward conservative regime

### 2.3 Alpha Optimizer Retraining Trigger (45 min)
**File:** `engine/ml/alpha_optimizer.py`
- Walk-forward trains on `optimize()` call but never retrains on outcomes
- **Fix:** After 50 learning loop outcomes, trigger `walk_forward()` with latest data
- **Fix:** Use `ModelStore.save_sklearn()` to persist trained model; load on restart

---

## PHASE 3: WIRE GSD + PAUL INTO LIVE LOOP (1.5 hrs)

Code exists in `intelligence_platform/plugins/gsd_paul_plugin.py` but isn't called during execution.

### 3.1 GSD Gradient Tracking (45 min)
- **Fix:** In `live_loop_orchestrator.py:run_learning_phase()`, call `gsd_plugin.update_gradients()` with signal deltas
- Feed gradient-weighted confidence adjustments into next MLVoteEnsemble cycle

### 3.2 Paul Pattern Library (45 min)
- **Fix:** After each trade outcome, call `paul_plugin.record_pattern()` with full trade context
- Before each new trade, call `paul_plugin.match_patterns()` to get historical success rate
- Feed pattern match score as 11th input to DecisionMatrix

---

## PHASE 4: DAILY LOG FILE (30 min)

### 4.1 Create Daily Audit Log
**New file:** `engine/monitoring/daily_audit_log.py`

Generate a single JSON log file per day at `logs/daily/YYYY-MM-DD.json` containing:

```json
{
  "date": "2026-03-28",
  "session": {"start": "09:30", "end": "16:00", "heartbeats": 180},
  "pipeline": {"stages_run": 17, "stages_failed": 0, "errors": []},
  "signals": {"generated": 1044, "passed_gates": 47, "executed": 12},
  "portfolio": {"nav_open": 1000000, "nav_close": 1050000, "pnl": 50000, "positions": 15},
  "risk": {"max_drawdown": 0.02, "var_95": 30000, "sector_concentration_max": 0.35},
  "learning": {"outcomes_recorded": 12, "regime_accuracy": 0.75, "best_engine": "momentum", "worst_engine": "social"},
  "agents": {"promotions": 1, "demotions": 0, "elite_count": 3},
  "errors": [{"time": "10:15", "component": "MacroEngine", "error": "OpenBB timeout", "severity": "LOW"}],
  "system": {"memory_mb": 2400, "cpu_pct": 45, "latency_avg_ms": 850}
}
```

- **Wire it:** Call from `run_close.py` at EOD to generate the file
- **Wire it:** Call from `live_loop_orchestrator.py:_run_market_close()` as final step

---

## PHASE 5: PRODUCTION HARDENING (2 hrs)

### 5.1 Circuit Breaker (45 min)
- Add to `alpaca_broker.py`: trip after 5 consecutive failures, auto-reset after 60s cooldown
- States: CLOSED → OPEN → HALF_OPEN

### 5.2 Idempotent Orders (30 min)
- Generate deterministic order ID: `hash(ticker + signal + timestamp_minute)`
- Pass as `client_order_id` to Alpaca API — prevents duplicate fills on heartbeat race

### 5.3 Signal Validation Layer (30 min)
- Before DecisionMatrix: reject any signal with NaN score, stale timestamp (>5 min), or score outside [-1, 1]

### 5.4 Graceful Shutdown (15 min)
- Add SIGTERM/SIGINT handlers to `live_loop_orchestrator.py`
- On shutdown: cancel pending orders, save state, persist learning snapshot

---

## PHASE 6: PLATINUM REPORT FIX (15 min)

- `run_open.py:209` imports `platinum_report.py` which doesn't exist (only v2 exists)
- **Fix:** Change import to `from engine.monitoring.platinum_report_v2 import PlatinumReportGenerator`

---

## TIME BUDGET

| Phase | Task | Time |
|-------|------|------|
| **1** | Security blockers | 2.5 hrs |
| **2** | Close learning loop | 2.0 hrs |
| **3** | Wire GSD + Paul | 1.5 hrs |
| **4** | Daily audit log | 0.5 hrs |
| **5** | Production hardening | 2.0 hrs |
| **6** | Platinum report fix | 0.25 hrs |
| | **TOTAL** | **~9 hrs** |

**Priority order:** Phase 1 → Phase 4 → Phase 6 → Phase 2 → Phase 5 → Phase 3

Phase 1 + 4 + 6 (3.25 hrs) = minimum for live trials tomorrow.
All 6 phases = Medallion-grade autonomous system.

---

## SYSTEM STATUS DASHBOARD (Current)

```
LAYER           STATUS    MODULES  GRADE  NOTES
─────────────────────────────────────────────────────
L1 Data         PASS      5/5      A      1,044 securities loaded
L2 Signals      PASS      11/11    A-     All engines instantiate
L3 ML           PASS      7/8      B+     model_store on bobby branch only
L4 Portfolio    PASS      1/1      A-     Beta corridor functional
L5 Execution    PASS      6/7      B+     Alpaca on bobby branch only
L6 Agents       PASS      6/6      B      11 sector bots + 12 personas
L7 HFT          PASS      1/1      A-     Unified surface working
Risk            PARTIAL   0/1      —      MC risk on bobby branch only
Monitoring      PASS      12/13    A-     platinum_report.py missing (v2 exists)
Orchestrator    PASS      2/2      B-     Phase coordination incomplete
Core            PASS      3/3      A      Platform + signals + portfolio
Tests           GREEN     153/153  A      All passing
─────────────────────────────────────────────────────
OVERALL: 50/55 modules passing | Paper: GO | Live: BLOCKED (Phase 1)
```
