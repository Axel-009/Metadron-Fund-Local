# Metadron Capital — self-contained build (LEAN / QuantConnect + IBKR)

This repository is the clean, standalone extraction of the Metadron intelligence engine. It runs on
its own: every engine module, the research integrations it loads, the vendored upstream repos
(Kronos, LEAN, OpenJarvis), Jarvis, the docs and the tests are inside it.

Reference architecture: `docs/architecture_dna/ARCHITECTURE_DNA.md` (the original design — where it
disagrees with this file, THIS file wins; e.g. its corridor section was linear and is superseded).

## Layout

```
algorithm/              QuantConnect LEAN algorithm (main, corridor, risk_budget, rotation, market_map, macro_regime, sector_rotation)
agents/                 agent knowledge: market_reasoning.md (trader framework + scenario JSON schema)
engine/                 Intelligence engine (extracted from the legacy monorepo, package paths unchanged)
  data/ signals/ ml/    universe, macro (GMTF, HMM), cube, security analysis, distress, events, CVR, patterns, ML
  allocation/           AllocationEngine + universe scan — THE allocation reference
  execution/            decision matrix, options engine, protection, event tail hedges, conviction override, L7
  risk/                 Monte Carlo risk + pre-trade MC gate
  portfolio/            legacy beta corridor (superseded by algorithm/corridor.py)
  agents/ monitoring/   sector/research bots, scorecard, NanoClaw permission guard, learning loop, reports
  api/ db/ security/    FastAPI engine API (port 8001), SQLite, auth
  bridges/              Jarvis, Llama / Qwen / Air-LLM servers, LLM inference bridge, Prometheus
core/                   original platform core (cyclical vs secular decomposition, money velocity)
intelligence_platform/  research integrations the engine loads at runtime (MiroFish, QLIB, distress, qstrader, …)
vendor/                 Kronos, LEAN, OpenJarvis — full source copies, pinned in vendor/VENDOR.md
jarvis/                 Jarvis setup guide + dashboard page
governance/ config/     credit classification, repo/heartbeat config
docs/                   architecture DNA, corridor charts
tests/                  238 tests (legacy engine suite + new algorithm tests)
scripts/update_vendor.sh  re-sync vendored repos to upstream
```

## Commands

```bash
pip install -r requirements.txt
python3 -m pytest tests -q                    # full suite
python3 docs/charts/corridor_v2.py            # regenerate the corridor chart
pm2 start ecosystem.config.cjs                # engine API + Jarvis + model servers
scripts/update_vendor.sh                      # update Kronos / LEAN / OpenJarvis
```
QuantConnect: create a Python project, copy `algorithm/*.py` + `config.json`, backtest.

## Design rules

1. QuantConnect (LEAN) is the execution and backtest engine; IBKR is the broker (paper → live).
   QuantConnect data for signals, IBKR quotes only for cross-checks.
2. Every module stays pure Python where possible; all optional imports are guarded (`try/except`)
   and degrade gracefully — never crash on a missing library.
3. Hard caps live in code constants (`algorithm/corridor.py`, `algorithm/risk_budget.py`) and change
   only with the owner's approval; agents may propose, never apply.
4. Agents (and Jarvis) are read-only on the trading system: they research, explain and recommend.
5. Tests must pass before pushing. Vendored code is updated only through `scripts/update_vendor.sh`.
6. Agents use the Anthropic API (current default model: claude-opus-5-5).

### Smart Beta v2 — CURRENT definition (owner-approved; code: `algorithm/corridor.py`)

Chart: `docs/charts/corridor_v2.png` (script: `docs/charts/corridor_v2.py`).

```
L(Rm, σm) = k · s(Rm) · (Rm − rf) / σm²          long market-beta leverage, capped at 4.5x

Rm     annualised S&P 500 return (expected)          rf  risk-free rate = 4%
σm     current annualised market volatility           k   share of full Kelly = 1.0 (parameter)
t      = clip((Rm − 7%) / 5%, 0, 1)
s(Rm)  = 3t² − 2t³                                   corridor participation (smooth 0 → 1)
γ(Rm)  = ds/dRm = 6t(1 − t) / 5%                     corridor gamma — flattened bell,
                                                     peak 0.30 per 1% at 9.5%, area 1
(Rm − rf)/σm²                                        "gamma market": vol-adjusted full Kelly
```

| Zone | Rm (annualised S&P) | Long beta |
|------|---------------------|-----------|
| DEFENSIVE | < 7% (bear / weak market) | **0** — only shorts / hedges |
| DYNAMIC | 7% – 12% | smooth ramp, k · s · (Rm − 4%)/σ² |
| ROTATION | > 12% | full vol-adjusted Kelly + active sector rotation, up to the 4.5x cap |

Leverage (k = 1): σ 15% → 0.43 / 1.22 / 2.79 / 3.56 at Rm 8.5 / 9.5 / 11 / 12%, 4.44 at 14%, 4.5 cap
from ≈14.1%; σ 20% → 2.00 at 12%, 3.00 at 16%; σ 25% → 1.28 at 12%. Rising volatility shrinks
leverage by 1/σ², so the book de-risks itself in volatile tapes without exiting.
Rebalance only when |ΔL| > 0.05. Delivery: Micro E-mini S&P futures (MES) — 4.5x is not reachable with
Reg-T stock margin (≈2x); SPY is the ≤2x alternative. Extra leverage above 12% may come from long calls
when implied vol ≤ realised (delta efficiency Ω = Δ·S/premium), else futures / call spreads.

### Risk budget (owner-approved; code: `algorithm/risk_budget.py`) — replaces "VaR ≤ 1.5% NAV"

A single 1.5% 1-day VaR on the whole book caps equity leverage at 0.97x at σ 15% (0.72x at 20%) and
blocks most small/mid caps — superseded. Under vol-adjusted Kelly, beta VaR = 1.645·k·s·Sharpe/√252
(falls as σ rises), so VaR is a regime ceiling, not the sizing tool.

| Layer | Rule |
|-------|------|
| 1 Sizing | corridor k-Kelly leverage above |
| 2 Regime VaR ceiling on market-beta risk (1-day 95%) | TRENDING 7.0% · RANGE 4.5% · STRESS 2.25% · CRASH 1.0% → leverage ceiling = VaR / (1.645·σm/√252). TRENDING 7.0% = the VaR of 4.5x at σ 15%, so the owner's max is reachable |
| 3 Stock-specific (alpha) VaR budget, 1-day 95% | **2.5% of NAV** (parameter) on idiosyncratic risk after removing market beta; per-name ≤ 0.5% NAV (≈ 9.7% weight at 50% vol); sizing ∝ 1/vol; same-sector names correlated 0.3; long options count at premium (defined risk); trades scaled down to fit, not rejected |
| 4 Hard stops | gross ≤ 4.5x; CVaR95 ≤ 1.5 × regime ceiling; drawdown −4% trim (halve), −8% close all but hedges |
| No double sizing | When Rm > 12% (ROTATION) and the regime reads STRESS/CRASH, the regime VaR ceiling is NOT applied on top of Kelly — Kelly's 1/σ² already sizes for volatility; only the 4.5x cap remains. A crash *inside* a strong market is handled by theme rotation, not by cutting market beta twice. |

Diversified small/mid-cap books fit easily: 40 names × 2.5% at 45% vol → 0.74%; 20 × 5% → 1.04%;
the budget binds only on concentration (10 × 10% at 50% vol → 1.64%; 5 × 20% → 2.32%).

### Four levels of "crash" + theme rotation (owner-approved; code: `algorithm/rotation.py`)

| Level | Detected by | Response |
|-------|-------------|----------|
| MARKET | S&P regime + Rm (corridor) | sizes market beta (MES overlay) |
| THEME | equal-weight theme index ≥ 30% below its 63-day high AND ≥ 15 pts under SPY over 21d = CRASH (≥ 15% / ≥ 8 pts = STRESS) | rotation playbook below — market beta untouched |
| POSITION | one holding's own drawdown | position protection (stops / trims) |
| PORTFOLIO | NAV drawdown | hard stops (−4% trim, −8% close) |

Paradigm (Oct–Nov): SPY > 12% at max leverage while AI names with heavy debt / private-credit exposure
(e.g. CRWV, NBIS) and capex over-spenders (e.g. IREN, HIVE) crash. Playbook — all equity sizes pass the
2.5% stock-specific VaR budget; targets are idempotent (re-running daily never compounds):

1. **CUT** crashed-theme longs to 0 (CRASH) or ≤ 2% NAV per name (STRESS).
2. **ROTATE** freed weight into defensives: defense (e.g. LMT, NOC, RTX, GD) 60% when the geopolitical-
   escalation score ≥ 0.5, else 25%; the rest to healthcare (e.g. LLY, JNJ, MRK, ABBV).
3. **ESCALATION** posture (score ≥ 0.5): long oil calls (USO/XLE, 1% NAV premium, ~60 DTE) + refiners
   (e.g. VLO, MPC, ≥ 1.5% each). Score = mean of: oil +10%/21d, gold +5%/21d, defense beats SPY by 5 pts/21d,
   VIX ≥ 25; an events-agent flag overrides.
4. **HARVEST** the drawdown: LEAPS calls (≥ 270 DTE, 0.5% NAV premium) on crashed names ≥ 18% below their
   1-year high that pass the solvency check (Merton PD ≤ 5%, interest coverage ≥ 2; unknown = no LEAPS);
   call spreads instead when IV/RV > 1.3. Short-dated straddles (7–30 DTE, 0.3% NAV, max 3) where realised
   vol ≥ 1.1 × implied or an event is within 14 days.

Tickers are owner-supplied configuration (THEMES / DEFENSIVES in rotation.py), not recommendations.
Equity legs trade only with `rotation_trading=true` (default false: plan plotted + logged); option legs
are logged until the options engine is ported (build step 6). Min trade 0.5% NAV (no churn).

### Trader framework → sub-sector rotation (owner-approved; code: `algorithm/market_map.py`, `macro_regime.py`, `sector_rotation.py`; agent guide: `agents/market_reasoning.md`)

The theme-crash playbook above is ONE instance of a general framework that agents must understand:

1. **Market map** (`market_map.py`): 26 sub-sectors tagged CYCLICAL / SECULAR / DEFENSIVE, their role in
   value chains (AI: platforms → chip design → semi equipment → regional foundries → memory → networking →
   data centers → power/grid; debt-funded neoclouds and capex-heavy pivots; at-risk low-end SaaS and
   labor-intensive services), ETF proxies, owner example tickers, and sensitivities (priors, re-estimated
   weekly) to 12 factors: GROWTH, INFLATION, LABOR, WAGES, PRODUCTIVITY, RATES, LIQUIDITY, CREDIT,
   GEOPOLITICS, SENTIMENT, AI_CAPEX, COMMODITIES.
2. **Macro question tree** (`macro_regime.diagnose`): inflation within mandate? → economy hot (data vs
   expectation)? → labor tight? → gapping? → room for stimulus? → productivity? → real wages? (productivity ↑
   + wages flat + employment retained ⇒ AI_SUBSTITUTION) → liquidity? → corporate bonds (HY OAS +35 bp/1m ⇒
   CREDIT_STRESS)? → geopolitics (ESCALATION)? → sentiment (EUPHORIA / CAPITULATION)? Each tag moves named
   sub-sectors (IMPLICATION_EFFECTS). Also: stock–bond correlation regime (negative ⇒ Treasuries hedge;
   positive ⇒ commodities / gold / cash) and market bias BULLISH / NEUTRAL / BEARISH.
3. **Scenarios** (`sector_rotation`): base / up / down what-ifs with probabilities and factor shocks, written
   by the macro agent or the owner to ObjectStore `metadron/scenarios.json`; macro readings (FRED) to
   `metadron/macro_inputs.json`. Score = sensitivities × factors + implications + 0.5 × fundamentals
   (revisions, revenue, legal fights) + theme state (STRESS −0.3, CRASH −0.8). **Edge** = probability-weighted
   score; **downside** = worst scenario. Equity-sleeve mix by bias (cyclical / secular / defensive):
   BULLISH 40/45/15, NEUTRAL 30/40/30, BEARISH 15/30/55; within each, weight ∝ max(edge − 0.5·max(−downside,0), 0).
4. Output each day: plots (*Market view*, *Equity mix*) and ObjectStore `metadron/sector_views.json`
   (factors, findings, overweight / avoid with drivers, edge and downside) for agents, Jarvis and Obsidian.

### Smart Beta v1 — corridor derivation (superseded by v2, kept for reference)

Chart: `docs/charts/beta_gamma_smooth.png` (script: `docs/charts/beta_gamma_smooth.py`).

**Labels (use these names everywhere — code, docs, dashboards):**

| Symbol | Name | Meaning |
|--------|------|---------|
| Rm | expected market return | expected annual S&P 500 return; the corridor is Rm ∈ [7%, 12%] (S&P historical earnings range) |
| β(Rm) | corridor beta | target exposure identifier, a smooth S-curve in Rm |
| β_L, β_H | corridor floor / ceiling | β_L = −0.029 at Rm = 7%, β_H = 0.425 at Rm = 12% |
| Δβ | corridor span | β_H − β_L = 0.454 |
| γ(Rm) | corridor gamma | **γ = dβ/dRm**, the rate at which beta changes with Rm (as option gamma is the slope of delta). A flattened bell. |
| λ | risk aversion | the Merton/CRRA coefficient (NOT called gamma, to avoid the clash with corridor gamma) |
| h | hurdle | Rm at which β = 0 |
| σ_std | standard volatility | 15% thesis standard |
| M | execution multiplier | 4.7 — converts β into portfolio leverage |

**1. Endpoints (Merton, Brownian motion).** Under dS/S = μ dt + σ dW a CRRA investor holds
β = (Rm − h)/(λσ²). The two corridor endpoints pin λσ² and h:

```
λσ² = (12% − 7%) / (β_H − β_L) = 0.05 / 0.454 = 0.1101   → λ ≈ 4.89 at σ = 15%
h   = 7% − β_L · λσ²          = 7% + 0.029 × 0.1101 = 7.32%
check: β(7%) = (0.07 − 0.0732)/0.1101 = −0.029 ; β(12%) = (0.12 − 0.0732)/0.1101 = 0.425
```
Only the product λσ² is identified (λ = 0.1101/σ²: 7.65 at 12%, 4.89 at 15%, 3.40 at 18%,
2.75 at 20%). Equivalently β = 1/λ ≈ 20% of full Kelly at σ = 15%.

**2. Smooth beta curve (the shape).** Inside the corridor β follows a smoothstep S-curve; outside
it is flat:

```
t     = clip((Rm − 7%) / 5%, 0, 1)
β(Rm) = β_L + Δβ · (3t² − 2t³)
```
Values: β(7%) = −0.029, β(9.5%) = 0.198, β(12%) = 0.425; flat beyond both edges.

**3. Corridor gamma (the flattened bell).**

```
γ(Rm) = dβ/dRm = Δβ · 6t(1 − t) / 5%      for 7% < Rm < 12%,  0 outside
```
- zero at and beyond 7% and 12% (exposure stops changing at the corridor edges)
- peak at Rm = 9.5%: γ_max = 1.5 · Δβ / 5% = 13.62 per unit Rm = **0.136 per 1% of Rm**
- area under γ over the corridor = Δβ = **0.454**
- the old linear ramp has a flat-box gamma of 0.091 per 1% with jumps at 7% and 12% —
  superseded by the smooth curve.

**4. From beta to leverage.**

```
leverage(Rm) = β(Rm) × M × vol_scale(σ_m)      M = 4.7
             → −0.136 at Rm ≤ 7% (hedge floor, BETA_INV) … 2.00 at Rm ≥ 12% (BETA_MAX)
vol_scale    = σ_std / σ_m  (current, linear)   or  (σ_std / σ_m)²  (Merton-consistent) — OPEN
```
Then: regime cap (or burst cap when momentum confirms) → hard caps (VaR95 ≤ 1.5% NAV,
gross-leverage gate). M ≈ λ, so leverage ≈ full Kelly at σ = 15%; at higher true volatility a
fixed M = 4.7 is over-Kelly (full-Kelly leverage at 12%: 2.08 at σ 15%, 1.44 at 18%, 1.17 at 20%).

**5. Inputs each cycle.** Rm_adj = realised SPY drift + macro adjustment (GMTF tension, HMM regime
probabilities) + Kronos index forecast → β(Rm_adj), γ(Rm_adj) → leverage budget L* (gross, split
across sleeves by Kelly risk-adjusted alpha within sleeve caps) → SPY / index-option overlay closes
the residual net-beta gap. Rebalance only when |Δβ| > 0.05 (throttle).

**Known defects in the legacy engine (fixed in the LEAN build):** linear ramp instead of the smooth
curve; constants duplicated in 3 places; AlphaBetaUnleashed calls a non-existent method and uses a
fallback that saturates at 2.0 near Rm ≈ 7.4%, and its output is never executed; sleeve deployment
ignores the corridor; TRENDING cap 0.65 makes the top of the corridor inert; beta burst never applied.
ARCHITECTURE_DNA.md's "linear(0 → BETA_MAX)" description is wrong.

## Target Architecture — LEAN / QuantConnect migration (agreed design)

Stack: QuantConnect cloud (LEAN, dashboard, backtests, live node) → IBKR (paper now,
live later; QC data for signals, IBKR for execution + cross-check). Agents: Claude
Managed Agents (Anthropic Console = agent dashboard). Readable control room: Obsidian
vault in this repo (agents write notes; Dataview tables; graph view). Kept from the
engine: beta corridor, options engine, Markov HMM, GMTF, MetadronCube, pattern
recognition, security analysis, ML ensemble, decision gates, MC gate, protection,
event hedges. Added: Kronos (forecasting). Reference only: freqtrade (GPLv3 — copy
design, never code): FreqAI-style retraining, protections, dry-run first, chat control.

1. **Forecasting → reallocation.** Kronos + walk-forward ML give per-name return and
   vol forecasts → (a) alpha model expected return, (b) forecast vol in Kelly, vol
   scaling and the MC gate, (c) index forecast into Rm_adj → β*, L*. Rebalance only
   when a weight or the beta gap moves > 0.05 (gamma throttle).
2. **Fundamentals → selection.** SecurityAnalysisEngine (CAPE/ERP/max P/E top-down;
   Graham number, NCAV, MoS ≥ 33%, ROIC−WACC, DuPont, owner earnings, 8-test grade,
   5-method IV bottom-up) + ML quality tier + credit score → Fundamentals gate (40%,
   must pass). Data: QC Morningstar fundamentals + SEC filings.
3. **Universe data.** QC point-in-time S&P 1500 universe (daily refresh); prices/options
   from QC, cross-checked vs IBKR; macro from FRED. "Data health" chart: names scanned
   vs expected, stale/missing data, failed cross-checks, skipped chains → alerts.
4. **Position monitoring.** QC live dashboard; IBKR app/TWS (independent check); daily
   reconciliation by the risk-officer agent; Obsidian daily note per position (why
   held, gate scores, drawdown, protection, upcoming events/hedges); phone alerts; Jarvis.
5. **Profitability gaps.** (a) live paper P&L vs backtest expectation; (b) per-model
   logged forecasts scored (hit rate, IC) + weekly ablation backtests — zero
   contribution 4 weeks → down-weight then remove; (c) per-agent scorecard (accuracy,
   P&L of approved changes, cost, rejection rate) → DIRECTOR…RECRUIT, demote/pause.
6. **Dynamic tilt under hard caps.** Adaptive layer (gate/tier weights, corridor slope
   within band, Kelly vol adj, thresholds) retuned weekly by walk-forward optimisation
   inside fixed min/max, adopted only if it beats current out of sample; regime shifts it
   too. Hard caps (VaR95 1.5%, CVaR 2.5%, position 20%, regime leverage/beta caps,
   drawdown stops −6/−10% position, −4/−8% portfolio, daily-loss breaker) live in a
   locked config — owner-only changes; agents can only propose.
7. **ML models.** All free/open source, run inside the QC algorithm: walk-forward alpha
   (linear + XGBoost, core), 4-classifier quality tier (NB, GBM, RF, XGBoost), HMM,
   Kronos-small (PyTorch; fine-tuned monthly off-platform, weights in QC Object
   Store). The first ablation decides which survive.
8. **Macro ↔ micro.** Macro (GMTF, HMM, Fed liquidity, curve, credit) sets regime, Rm_adj,
   β*, leverage budget, sector and sleeve weights → micro (fundamentals, ML, Kronos,
   patterns, events) picks names inside that budget → decision matrix needs both
   (Fundamentals 40 / Flow 20 / Macro 20 / Momentum 20) → Kelly size capped by regime
   and VaR → corridor overlay closes the beta gap. Macro can shrink/veto; micro cannot
   override a macro cap.
9. **Capital structure.** New monitor for every held/universe name: HY/IG OAS level +
   change (FRED ICE BofA series), debt/EBITDA, interest coverage, 1–3y maturity wall,
   FCF vs capex burn, leases (filings), new facilities / private credit / secured debt
   (8-K, events agent), Merton equity-implied default probability and credit spread vs
   market spread. Deteriorating structure → Fundamentals-gate penalty or block, tighter
   protection, event tail hedge. Issuer-level bond/CDS data = paid vendor, only if the
   free signals prove out in backtests. (Current "credit_spread" is an HY−IG ETF
   volatility difference, not a spread — replace.)

## Allocation reference

Code: `engine/allocation/allocation_engine.py` (+ `universe_scan.py`). Sleeves:


| Sleeve | Allocation | Credit Tier | Notes |
|--------|-----------|-------------|-------|
| IG Equities | 40% | A/B (IG) | All caps — no mega cap restriction |
| Options | 25% | Mixed | 10% IG + 10% HY + 5% Distressed |
| Bond/Commodity ETFs | 10% | — | TLT, GLD, USO, HYG, LQD, etc. |
| HY Equities | 10% | C/D (HY) | BB-B rated, leveraged |
| Distressed Equity | 10% | E/F | Fallen angels, recovery plays |
| Cash (Dry Powder) | 5% | — | Never deployed, buying power reserve |

Deploy 95% of NAV + leverage. Only 5% remains as dry powder.

## MetadronCube

### 10-Layer Intelligence

| Layer | Name | Function |
|-------|------|----------|
| 0 | FedPlumbingLayer | SOFR, reserves, TGA, ON-RRP |
| 1 | LiquidityTensor | L(t) in [-1,+1] |
| 2 | ReserveFlowKernel | ΔReserves → ΔEquity/Credit |
| Risk | RiskStateModel | R(t) in [0,1] |
| Flow | CapitalFlowModel | F(t) sector rotation |
| 4 | RegimeEngine (HMM) | TRENDING/RANGE/STRESS/CRASH |
| Gate | GateZAllocator | 5-sleeve allocation |
| Entry | GateLogic | 4-gate entry scoring |
| Kill | KillSwitch | Auto-derisking triggers |
| Cal | FCLPLoop | Daily recalibration |

### Regime Parameters (95% Alpha Target)

| Regime | Leverage | Beta Cap | Beta Burst | Crash Floor |
|--------|----------|----------|------------|-------------|
| TRENDING | 3.0x | 0.65 | 0.70 | ≥+25% |
| RANGE | 2.5x | 0.45 | 0.55 | ≥+25% |
| STRESS | 1.5x | 0.15 | 0.20 | ≥+25% |
| CRASH | 0.8x | -0.20 | -0.10 | ≥+25% |

### MLVoteEnsemble (10 Tiers)

| Tier | Name | Weight | Source |
|------|------|--------|--------|
| T1 | Neural | 1.0 | DRL/FinRL bridge |
| T2 | Momentum | 1.2 | Technical signals |
| T3 | Vol Regime | 0.8 | Volatility regime |
| T4 | Monte Carlo | 0.9 | MC simulation |
| T5 | Quality | 1.1 | Quality ranker |
| T6 | Social | 1.0 | Agent sim consensus |
| T7 | Distress | 0.9 | Distressed asset engine |
| T8 | Event | 1.0 | Event-driven engine |
| T9 | CVR | 0.7 | CVR engine |
| T10 | Credit Quality | 0.9 | UniverseClassifier/CreditQualityClassifier |
