# metadron_lean — clean LEAN / QuantConnect build

The new, minimal home of the Metadron intelligence engine. Every module here is pure Python
(no OpenBB, no sandbox services) so the same file runs on QuantConnect, locally and in tests.
The legacy `engine/` stays as the reference archive.

## Build steps

| Step | Module | Status |
|------|--------|--------|
| 1 | `corridor.py` — smart-beta corridor (smooth β, γ = dβ/dRm bell, leverage) + `main.py` SPY overlay | done, 7 tests |
| 2 | Macro: HMM regime + GMTF (FRED) → Rm adjustment, regime caps, burst | next |
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
2. Copy `main.py` and `corridor.py` into it (`config.json` carries the parameters).
3. Backtest. Charts appear on the backtest page: *Rm vs corridor*, *Corridor beta*,
   *Corridor gamma*, *Leverage*, *Volatility*.
4. Change `vol_mode` (`linear` / `squared`) or `leverage_cap` in the project parameters and re-run
   to compare.
