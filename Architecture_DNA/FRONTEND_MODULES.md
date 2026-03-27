# Frontend Modules — Metadron Capital

> **Purpose:** Reference index of all frontend build UIs in the system.

## Active Frontend Modules (Build-System Backed)

| # | Module | Path | Tech Stack | Purpose |
|---|--------|------|------------|---------|
| 1 | AI Hedgefund | `intelligence_platform/ai-hedgefund/app/frontend/` | React 18 + TypeScript + Vite 5 + Tailwind + React Flow + Radix UI (shadcn/ui) | Visual strategy builder, node-based flow editor, backtest dashboard, real-time execution monitoring. Most complete standalone frontend — serves as the base for consolidated UI. |
| 2 | Ruvocal Chat UI | `intelligence_platform/Ruflo-agents/ruflo/src/ruvocal/` | SvelteKit 2 + Svelte 5 + Vite 6 + Tailwind + MongoDB | Chat interface with voice input, markdown rendering, file upload for LLM agent interaction. Full-featured chat application with database integration and authentication. |
| 3 | OpenBB Plotly Components | `intelligence_platform/open-bb/frontend-components/plotly/` | React 18 + Vite 4 + Plotly.js + Tailwind + TypeScript | Shared Plotly financial charting component library. Reusable — not a standalone app. Exports as single-file bundle via rollup. |
| 4 | OpenBB Table Components | `intelligence_platform/open-bb/frontend-components/tables/` | React 17 + Vite 4 + React Table 8 + React Virtual + Tailwind + TypeScript | Shared data table component with virtual scrolling. Reusable — not a standalone app. Exports as single-file bundle via rollup. |

## Non-Build Frontends (No npm build pipeline)

| # | Module | Path | Tech Stack | Status |
|---|--------|------|------------|--------|
| 5 | MiroFish Frontend | `mirofish/frontend/` | Vue 3 + Vite + D3.js (documented target) | **Stub only** — directory exists with empty `src/assets/`, no `package.json`. Not yet initialized. |
| 6 | Claude Flow Browser Dashboard | `intelligence_platform/Ruflo-agents/v2/examples/browser-dashboard/` | Vanilla JS + WebSocket + Node.js server | Proof-of-concept swarm monitoring dashboard. Has `package.json` but minimal framework usage. |
| 7 | Stock Forecasting JS | `intelligence_platform/Stock-prediction/stock-forecasting-js/` | Vanilla HTML/CSS/JS (D3, Plotly, TensorFlow.js, Materialize CSS) | In-browser LSTM stock prediction with chart visualization. No `package.json`, no build tool — plain HTML/JS files. |

## Technology Summary

- **Build tool:** Vite (all 4 primary modules)
- **Primary framework:** React (modules 1, 3, 4)
- **Secondary frameworks:** SvelteKit (module 2), Vue 3 (module 5 — planned)
- **Styling:** Tailwind CSS (all modules)
- **TypeScript:** Used in all build-backed modules

## Consolidation Notes

- **AI Hedgefund frontend** (module 1) is the most complete and serves as the base for a consolidated Metadron Capital UI.
- **Ruvocal Chat** (module 2) is intended to be integrated as a panel within the consolidated frontend for agent interaction.
- **OpenBB components** (modules 3 & 4) are reusable chart/table libraries available to any frontend that needs financial visualization.
- **MiroFish frontend** (module 5) needs initialization if social prediction visualization is required.
