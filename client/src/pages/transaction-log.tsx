import { useState, useEffect, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";

// ═══════════ SIMULATED TRANSACTION DATA ═══════════

interface Transaction {
  id: string;
  time: string;
  ticker: string;
  side: "BUY" | "SELL" | "SHORT" | "COVER";
  qty: number;
  price: number;
  notional: number;
  fillType: "FULL" | "PARTIAL" | "REJECTED";
  venue: string;
  signalType: string;
  latencyMs: number;
  slippageBps: number;
  // Fundamentals
  pe: number;
  eps: number;
  mktCap: string;
  sector: string;
  divYield: number;
  beta: number;
}

const SECTORS = ["Technology", "Financials", "Healthcare", "Consumer Disc.", "Energy", "Industrials", "Materials", "Utilities", "Real Estate", "Comm. Svcs."];
const VENUES = ["ARCA", "NYSE", "NASDAQ", "BATS", "IEX", "EDGX", "DARK"];
const SIGNALS = ["ML_AGENT", "MICRO_PRICE", "RV_PAIR", "SOCIAL", "CVR", "EVENT", "QUALITY", "DRL_AGENT", "MOMENTUM", "TFT_MODEL"];
const TICKERS_DATA: Record<string, { pe: number; eps: number; mktCap: string; sector: string; divYield: number; beta: number; basePrice: number }> = {
  AAPL: { pe: 28.4, eps: 6.67, mktCap: "2.94T", sector: "Technology", divYield: 0.52, beta: 1.24, basePrice: 189 },
  MSFT: { pe: 35.2, eps: 11.95, mktCap: "3.12T", sector: "Technology", divYield: 0.72, beta: 0.98, basePrice: 420 },
  NVDA: { pe: 62.8, eps: 13.93, mktCap: "2.15T", sector: "Technology", divYield: 0.02, beta: 1.71, basePrice: 875 },
  AMZN: { pe: 52.1, eps: 3.56, mktCap: "1.93T", sector: "Consumer Disc.", divYield: 0.00, beta: 1.15, basePrice: 185 },
  GOOGL: { pe: 24.8, eps: 6.29, mktCap: "1.94T", sector: "Comm. Svcs.", divYield: 0.00, beta: 1.05, basePrice: 155 },
  META: { pe: 22.5, eps: 22.48, mktCap: "1.30T", sector: "Comm. Svcs.", divYield: 0.00, beta: 1.22, basePrice: 505 },
  JPM: { pe: 11.8, eps: 16.81, mktCap: "572B", sector: "Financials", divYield: 2.35, beta: 1.12, basePrice: 198 },
  TSLA: { pe: 55.4, eps: 3.22, mktCap: "567B", sector: "Consumer Disc.", divYield: 0.00, beta: 2.08, basePrice: 178 },
  XOM: { pe: 12.1, eps: 9.52, mktCap: "460B", sector: "Energy", divYield: 3.28, beta: 0.87, basePrice: 115 },
  UNH: { pe: 21.3, eps: 23.57, mktCap: "462B", sector: "Healthcare", divYield: 1.42, beta: 0.72, basePrice: 502 },
  V: { pe: 30.1, eps: 9.40, mktCap: "580B", sector: "Financials", divYield: 0.76, beta: 0.95, basePrice: 282 },
  BAC: { pe: 10.5, eps: 3.42, mktCap: "287B", sector: "Financials", divYield: 2.52, beta: 1.35, basePrice: 35 },
  GS: { pe: 14.2, eps: 30.82, mktCap: "142B", sector: "Financials", divYield: 2.15, beta: 1.42, basePrice: 437 },
  HD: { pe: 24.8, eps: 15.11, mktCap: "374B", sector: "Consumer Disc.", divYield: 2.28, beta: 1.02, basePrice: 374 },
  LLY: { pe: 98.2, eps: 8.18, mktCap: "764B", sector: "Healthcare", divYield: 0.68, beta: 0.52, basePrice: 803 },
  AVGO: { pe: 38.5, eps: 35.24, mktCap: "677B", sector: "Technology", divYield: 1.52, beta: 1.15, basePrice: 1357 },
};

function generateTransactions(count: number): Transaction[] {
  const tickers = Object.keys(TICKERS_DATA);
  const txns: Transaction[] = [];
  const now = new Date();

  for (let i = 0; i < count; i++) {
    const ticker = tickers[Math.floor(Math.random() * tickers.length)];
    const d = TICKERS_DATA[ticker];
    const side = Math.random() > 0.5 ? (Math.random() > 0.3 ? "BUY" : "SHORT") : (Math.random() > 0.3 ? "SELL" : "COVER");
    const qty = Math.floor(50 + Math.random() * 950);
    const price = d.basePrice * (1 + (Math.random() - 0.5) * 0.02);
    const notional = qty * price;
    const t = new Date(now.getTime() - i * (15000 + Math.random() * 120000));

    txns.push({
      id: `TXN-${(100000 + count - i).toString(36).toUpperCase()}`,
      time: t.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + t.getMilliseconds().toString().padStart(3, "0"),
      ticker,
      side: side as Transaction["side"],
      qty,
      price: +price.toFixed(2),
      notional: +notional.toFixed(2),
      fillType: Math.random() > 0.05 ? (Math.random() > 0.1 ? "FULL" : "PARTIAL") : "REJECTED",
      venue: VENUES[Math.floor(Math.random() * VENUES.length)],
      signalType: SIGNALS[Math.floor(Math.random() * SIGNALS.length)],
      latencyMs: +(0.5 + Math.random() * 12).toFixed(1),
      slippageBps: +((Math.random() - 0.3) * 4).toFixed(2),
      pe: d.pe,
      eps: d.eps,
      mktCap: d.mktCap,
      sector: d.sector,
      divYield: d.divYield,
      beta: d.beta,
    });
  }
  return txns;
}

// ═══════════ SUMMARY CARDS ═══════════

function SummaryCards({ txns }: { txns: Transaction[] }) {
  const filled = txns.filter((t) => t.fillType !== "REJECTED");
  const totalNotional = filled.reduce((s, t) => s + t.notional, 0);
  const buys = filled.filter((t) => t.side === "BUY" || t.side === "COVER").length;
  const sells = filled.filter((t) => t.side === "SELL" || t.side === "SHORT").length;
  const avgLatency = filled.length ? filled.reduce((s, t) => s + t.latencyMs, 0) / filled.length : 0;
  const avgSlippage = filled.length ? filled.reduce((s, t) => s + t.slippageBps, 0) / filled.length : 0;
  const rejectRate = txns.length ? ((txns.length - filled.length) / txns.length * 100) : 0;

  const cards = [
    { label: "TOTAL EXECUTED", value: filled.length.toString(), sub: `of ${txns.length} orders` },
    { label: "NOTIONAL VOLUME", value: `$${(totalNotional / 1e6).toFixed(2)}M`, sub: "total traded" },
    { label: "BUY / SELL", value: `${buys} / ${sells}`, sub: `ratio ${buys > 0 ? (buys / Math.max(sells, 1)).toFixed(2) : "—"}` },
    { label: "AVG LATENCY", value: `${avgLatency.toFixed(1)}ms`, sub: "fill to confirm" },
    { label: "AVG SLIPPAGE", value: `${avgSlippage.toFixed(2)} bps`, sub: avgSlippage < 0 ? "favorable" : "adverse" },
    { label: "REJECT RATE", value: `${rejectRate.toFixed(1)}%`, sub: `${txns.length - filled.length} rejected` },
  ];

  return (
    <div className="grid grid-cols-6 gap-[2px]">
      {cards.map((c) => (
        <div key={c.label} className="terminal-panel p-2">
          <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">{c.label}</div>
          <div className="text-sm font-mono tabular-nums text-terminal-text-primary mt-0.5">{c.value}</div>
          <div className="text-[8px] text-terminal-text-faint mt-0.5">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ FILTER BAR ═══════════

function FilterBar({ filter, setFilter }: { filter: any; setFilter: (f: any) => void }) {
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 text-[10px] font-mono border-b border-terminal-border/50">
      <input
        type="text"
        placeholder="Search ticker..."
        value={filter.ticker}
        onChange={(e) => setFilter({ ...filter, ticker: e.target.value.toUpperCase() })}
        className="w-[100px] bg-terminal-bg border border-terminal-border/50 rounded px-2 py-0.5 text-terminal-text-primary placeholder:text-terminal-text-faint outline-none focus:border-terminal-accent/50"
      />
      <select
        value={filter.side}
        onChange={(e) => setFilter({ ...filter, side: e.target.value })}
        className="bg-terminal-bg border border-terminal-border/50 rounded px-1 py-0.5 text-terminal-text-primary outline-none"
      >
        <option value="">All Sides</option>
        <option value="BUY">BUY</option>
        <option value="SELL">SELL</option>
        <option value="SHORT">SHORT</option>
        <option value="COVER">COVER</option>
      </select>
      <select
        value={filter.fill}
        onChange={(e) => setFilter({ ...filter, fill: e.target.value })}
        className="bg-terminal-bg border border-terminal-border/50 rounded px-1 py-0.5 text-terminal-text-primary outline-none"
      >
        <option value="">All Status</option>
        <option value="FULL">FULL</option>
        <option value="PARTIAL">PARTIAL</option>
        <option value="REJECTED">REJECTED</option>
      </select>
      <select
        value={filter.sector}
        onChange={(e) => setFilter({ ...filter, sector: e.target.value })}
        className="bg-terminal-bg border border-terminal-border/50 rounded px-1 py-0.5 text-terminal-text-primary outline-none"
      >
        <option value="">All Sectors</option>
        {SECTORS.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <div className="flex-1" />
      <span className="text-terminal-text-faint text-[8px]">Auto-refresh: 5s</span>
      <div className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
    </div>
  );
}

// ═══════════ MAIN COMPONENT ═══════════

export default function TransactionLog() {
  const [allTxns, setAllTxns] = useState<Transaction[]>(() => generateTransactions(200));
  const [filter, setFilter] = useState({ ticker: "", side: "", fill: "", sector: "" });
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Simulate new transactions arriving
  useEffect(() => {
    const interval = setInterval(() => {
      setAllTxns((prev) => {
        const newTxns = generateTransactions(1);
        return [...newTxns, ...prev].slice(0, 500);
      });
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  const filteredTxns = useMemo(() => {
    return allTxns.filter((t) => {
      if (filter.ticker && !t.ticker.includes(filter.ticker)) return false;
      if (filter.side && t.side !== filter.side) return false;
      if (filter.fill && t.fillType !== filter.fill) return false;
      if (filter.sector && t.sector !== filter.sector) return false;
      return true;
    });
  }, [allTxns, filter]);

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px]" data-testid="transaction-log">
      {/* Summary cards */}
      <SummaryCards txns={filteredTxns} />

      {/* Main log table */}
      <DashboardPanel
        title="LIVE TRANSACTION LOG"
        className="flex-1"
        headerRight={
          <div className="flex items-center gap-2">
            <span className="text-[8px] text-terminal-text-faint font-mono tabular-nums">{filteredTxns.length} transactions</span>
            <button className="px-2 py-0.5 text-[8px] bg-terminal-surface-2 text-terminal-text-muted rounded hover:text-terminal-text-primary">
              CSV
            </button>
          </div>
        }
        noPadding
      >
        <div className="h-full flex flex-col">
          <FilterBar filter={filter} setFilter={setFilter} />

          {/* Table header */}
          <div className="flex items-center px-2 py-1.5 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50 font-mono flex-shrink-0">
            <span className="w-[100px]">Time</span>
            <span className="w-[55px]">Ticker</span>
            <span className="w-[50px]">Side</span>
            <span className="w-[55px] text-right">Qty</span>
            <span className="w-[75px] text-right">Price</span>
            <span className="w-[90px] text-right">Notional</span>
            <span className="w-[55px] text-center">Status</span>
            <span className="w-[50px]">Venue</span>
            <span className="w-[85px]">Signal</span>
            <span className="w-[55px] text-right">Lat ms</span>
            <span className="w-[55px] text-right">Slip bps</span>
            <span className="w-px bg-terminal-border/30 h-3 mx-1" />
            <span className="w-[40px] text-right">P/E</span>
            <span className="w-[50px] text-right">EPS</span>
            <span className="w-[55px] text-right">Mkt Cap</span>
            <span className="flex-1 text-right">Sector</span>
          </div>

          {/* Scrollable rows */}
          <div className="flex-1 overflow-auto">
            {filteredTxns.map((t) => (
              <div key={t.id}>
                <div
                  onClick={() => setExpandedId(expandedId === t.id ? null : t.id)}
                  className={`flex items-center px-2 py-1 text-[10px] font-mono tabular-nums border-b border-terminal-border/10 hover:bg-white/[0.02] cursor-pointer ${
                    t.fillType === "REJECTED" ? "opacity-50" : ""
                  } ${expandedId === t.id ? "bg-terminal-accent/5" : ""}`}
                >
                  <span className="w-[100px] text-terminal-text-muted">{t.time}</span>
                  <span className="w-[55px] text-terminal-accent font-semibold">{t.ticker}</span>
                  <span className={`w-[50px] font-semibold ${
                    t.side === "BUY" || t.side === "COVER" ? "text-terminal-positive" : "text-terminal-negative"
                  }`}>
                    {t.side}
                  </span>
                  <span className="w-[55px] text-right">{t.qty.toLocaleString()}</span>
                  <span className="w-[75px] text-right">${t.price.toFixed(2)}</span>
                  <span className="w-[90px] text-right">${t.notional >= 1e6 ? (t.notional / 1e6).toFixed(2) + "M" : t.notional.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                  <span className={`w-[55px] text-center text-[8px] font-semibold ${
                    t.fillType === "FULL" ? "text-terminal-positive" : t.fillType === "PARTIAL" ? "text-terminal-warning" : "text-terminal-negative"
                  }`}>
                    {t.fillType}
                  </span>
                  <span className="w-[50px] text-terminal-text-faint">{t.venue}</span>
                  <span className="w-[85px] text-terminal-text-muted truncate">{t.signalType}</span>
                  <span className={`w-[55px] text-right ${t.latencyMs > 8 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>
                    {t.latencyMs}
                  </span>
                  <span className={`w-[55px] text-right ${t.slippageBps > 1 ? "text-terminal-negative" : t.slippageBps < -0.5 ? "text-terminal-positive" : "text-terminal-text-muted"}`}>
                    {t.slippageBps > 0 ? "+" : ""}{t.slippageBps}
                  </span>
                  <span className="w-px bg-terminal-border/30 h-3 mx-1" />
                  <span className="w-[40px] text-right text-terminal-text-faint">{t.pe > 0 ? t.pe.toFixed(1) : "—"}</span>
                  <span className="w-[50px] text-right text-terminal-text-faint">{t.eps.toFixed(2)}</span>
                  <span className="w-[55px] text-right text-terminal-text-faint">{t.mktCap}</span>
                  <span className="flex-1 text-right text-terminal-text-faint truncate">{t.sector}</span>
                </div>

                {/* Expanded detail row */}
                {expandedId === t.id && (
                  <div className="px-4 py-2 bg-terminal-surface-2/30 border-b border-terminal-border/20 text-[9px] font-mono">
                    <div className="grid grid-cols-6 gap-4">
                      <div>
                        <div className="text-terminal-text-faint">Order ID</div>
                        <div className="text-terminal-text-primary">{t.id}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Signal Source</div>
                        <div className="text-terminal-accent">{t.signalType}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Div Yield</div>
                        <div className="text-terminal-text-primary">{t.divYield.toFixed(2)}%</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Beta</div>
                        <div className="text-terminal-text-primary">{t.beta.toFixed(2)}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Cost Basis</div>
                        <div className="text-terminal-text-primary">${(t.qty * t.price).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Execution Quality</div>
                        <div className={t.slippageBps < 0 ? "text-terminal-positive" : t.slippageBps > 2 ? "text-terminal-negative" : "text-terminal-text-primary"}>
                          {t.slippageBps < 0 ? "GOOD" : t.slippageBps > 2 ? "POOR" : "FAIR"}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </DashboardPanel>
    </div>
  );
}
