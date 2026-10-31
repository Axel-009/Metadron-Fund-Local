# metadron_lean — clean LEAN / QuantConnect build

The new, minimal home of the Metadron intelligence engine. Every module here is pure Python
(no OpenBB, no sandbox services) so the same file runs on QuantConnect, locally and in tests.
The legacy `engine/` stays as the reference archive.

## Build steps

| Step | Module | Status |
|------|--------|--------|
| 1 | `corridor.py` — smart-beta corridor v1 + `main.py` SPY overlay | done (superseded by 2a) |
| 2a | corridor v2 (rf 4%, 0 below 7%, Kelly ramp, 4.5x cap) + `risk_budget.py` (regime VaR ceilings, 2.5% alpha budget, hard stops) + MES futures delivery | done, 18 tests |
| 2b | Macro: HMM regime + GMTF (FRED) + forward Rm estimate (replaces trailing drift and the placeholder volatility regime) | next |
| 3 | Universe + fundamentals (S&P 1500, Morningstar) + security analysis | |
| 4 | Alpha: walk-forward ML, quality tiers, pattern recognition, Kronos | |
| 5 | Decision matrix: 4 gates, spread, Monte Carlo gate, Kelly + 1-contract rule | |
| 6 | Options engine, position protection, event tail hedges, capital-structure monitor | |
| 7 | Data-health checks, reconciliation, agents (Managed Agents), Obsidian vault | |

The canonical maths is in the root `CLAUDE.md` → "Smart Beta — corridor derivation".

## Run the tests

```bash
python3 -m pytest metadron_lean/tests -q
```

## Run on QuantConnect

1. Create a Python project on quantconnect.com (or `lean cloud push` with the LEAN CLI).
2. Copy `main.py`, `corridor.py` and `risk_budget.py` into it (`config.json` carries the parameters).
3. Backtest. Charts appear on the backtest page: *Rm vs corridor*, *Corridor* (participation +
   gamma), *Leverage* (Kelly, cap, target, actual), *Risk* (volatility, beta VaR, drawdown).
4. Parameters: `kelly_fraction` (1.0), `beta_instrument` (`MES` futures, or `SPY` ≤ 2x).
