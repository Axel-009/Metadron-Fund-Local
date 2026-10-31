# Metadron Capital — self-contained build

The Metadron intelligence engine extracted into one standalone, runnable repository:
QuantConnect LEAN algorithm + IBKR execution, the full engine, research integrations, vendored
Kronos / LEAN / OpenJarvis, Jarvis, docs and tests. Nothing here depends on the old monorepo.

**Start with `CLAUDE.md`** — layout, commands, design rules, and the canonical maths
(smart-beta corridor v2, risk budget, allocation, target architecture).

## Quick start

```bash
pip install -r requirements.txt
python3 -m pytest tests -q          # 218 tests
```

## What is where

| Folder | Contents |
|--------|----------|
| `algorithm/` | QuantConnect algorithm: `main.py` (MES futures corridor overlay), `corridor.py` (v2), `risk_budget.py` |
| `engine/` | 168 engine modules: macro (GMTF, HMM), MetadronCube, security analysis, distress, events, CVR, patterns, ML, decision matrix, options, protection, tail hedges, allocation, risk, API, agents, monitoring |
| `core/` | Original platform core: cyclical vs secular decomposition, money velocity |
| `intelligence_platform/` | Research integrations the engine loads (MiroFish, QLIB, distress models, ai-hedgefund, quant-trading, qstrader, plugins, Air-LLM, …) |
| `vendor/` | Kronos, LEAN, OpenJarvis — full source, pinned in `vendor/VENDOR.md`, update with `scripts/update_vendor.sh` |
| `jarvis/` | Jarvis setup guide (`jarvis/README.md`) + dashboard page |
| `docs/` | `architecture_dna/` (reference design), `charts/` (corridor derivation charts + scripts) |
| `governance/`, `config/` | Credit classification, repo / heartbeat config |
| `tests/` | Engine suite + algorithm tests |

## Build roadmap

| Step | Scope | Status |
|------|-------|--------|
| 1–2a | Corridor v2 (rf 4%, 0 below 7%, Kelly ramp, 4.5x cap), split risk budget, MES delivery | done |
| — | Full extraction into this standalone repo | done |
| 2b | HMM regime + GMTF + forward Rm estimate wired into `algorithm/` | next |
| 3 | Universe + fundamentals (S&P 1500, Morningstar) + security analysis in the algorithm | |
| 4 | Alpha: walk-forward ML, quality tiers, pattern recognition, Kronos | |
| 5 | Decision matrix in the algorithm (4 gates, spread, MC gate, Kelly + 1-contract rule) | |
| 6 | Options engine, position protection, event tail hedges, capital-structure monitor | |
| 7 | Data-health, reconciliation, Managed Agents roster, Jarvis front desk, Obsidian vault | |

## Run on QuantConnect

1. Create a Python project on quantconnect.com (or `lean cloud push` with the LEAN CLI).
2. Copy `algorithm/main.py`, `corridor.py`, `risk_budget.py`, `config.json`.
3. Backtest — charts: *Rm vs corridor*, *Corridor*, *Leverage*, *Risk*.
4. Parameters: `kelly_fraction` (1.0), `beta_instrument` (`MES` or `SPY` ≤ 2x).

## Licenses

In-house code: proprietary (Metadron Capital). Vendored: Kronos (MIT), LEAN (Apache-2.0),
OpenJarvis (Apache-2.0) — each keeps its own LICENSE. freqtrade is a design reference only (GPL-3.0,
not included).
