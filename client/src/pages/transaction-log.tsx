import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface OrderRecord {
  id: string;
  ticker: string;
  side: string;
  qty: number;
  price: number;
  notional: number;
  fill_type: string;
  order_type: string;
  status: string;
  submitted_at: string;
  filled_at: string;
  signal_type: string;
}

interface BlotterRecord {
  ticker: string;
  side: string;
  quantity: number;
  fill_price: number;
  arrival_price: number;
  slippage_bps: number;
  implementation_shortfall: number;
  market_impact_bps: number;
  signal_type: string;
  product_type: string;
  routing_strategy: string;
  urgency: string;
  timestamp: string;
  tca_snapshot: Record<string, unknown>;
}

interface Transaction {
  id: string;
  time: string;
  ticker: string;
  side: string;
  qty: number;
  price: number;
  notional: number;
  fillType: string;
  venue: string;
  signalType: string;
  latencyMs: number;
  slippageBps: number;
  orderType: string;
  routingStrategy: string;
  urgency: string;
  productType: string;
}

// ═══════════ SUMMARY CARDS ═══════════

function SummaryCards({ txns }: { txns: Transaction[] }) {
  const filled = txns.filter((t) => t.fillType !== "REJECTED" && t.fillType !== "PENDING");
  const totalNotional = filled.reduce((s, t) => s + t.notional, 0);
  const buys = filled.filter((t) => t.side === "BUY" || t.side === "COVER").length;
  const sells = filled.filter((t) => t.side === "SELL" || t.side === "SHORT").length;
  const avgLatency = filled.length ? filled.reduce((s, t) => s + t.latencyMs, 0) / filled.length : 0;
  const avgSlippage = filled.length ? filled.reduce((s, t) => s + t.slippageBps, 0) / filled.length : 0;
  const rejectRate = txns.length ? ((txns.length - filled.length) / txns.length * 100) : 0;

  const cards = [
    { label: "TOTAL EXECUTED", value: filled.length.toString(), sub: `of ${txns.length} orders` },
    { label: "NOTIONAL VOLUME", value: totalNotional >= 1e6 ? `$${(totalNotional / 1e6).toFixed(2)}M` : `$${totalNotional.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: "total traded" },
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

function FilterBar({
  filter,
  setFilter,
  sides,
  statuses,
  source,
}: {
  filter: { ticker: string; side: string; fill: string; signal: string };
  setFilter: (f: { ticker: string; side: string; fill: string; signal: string }) => void;
  sides: string[];
  statuses: string[];
  source: string;
}) {
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
        {sides.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <select
        value={filter.fill}
        onChange={(e) => setFilter({ ...filter, fill: e.target.value })}
        className="bg-terminal-bg border border-terminal-border/50 rounded px-1 py-0.5 text-terminal-text-primary outline-none"
      >
        <option value="">All Status</option>
        {statuses.map((s) => <option key={s} value={s}>{s}</option>)}
      </select>
      <div className="flex-1" />
      <span className="text-[8px] text-terminal-text-faint">Source: {source}</span>
      <span className="text-[8px] text-terminal-text-faint">Auto-refresh: 5s</span>
      <div className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
    </div>
  );
}

// ═══════════ MAIN COMPONENT ═══════════

export default function TransactionLog() {
  // ─── Engine API: primary = /portfolio/orders (Alpaca), secondary = /execution/l7/blotter ───
  const { data: ordersData } = useEngineQuery<{ orders: OrderRecord[]; source: string }>(
    "/portfolio/orders?limit=500",
    { refetchInterval: 5000 },
  );
  const { data: blotterData } = useEngineQuery<{ trades: BlotterRecord[]; broker_type: string }>(
    "/execution/l7/blotter?limit=500",
    { refetchInterval: 5000 },
  );

  // Merge: prefer blotter records (richer TCA data), augment with orders
  const allTxns: Transaction[] = useMemo(() => {
    const txns: Transaction[] = [];

    // Build a set of blotter tickers+times to avoid duplicates
    const blotterKeys = new Set<string>();

    // Primary: L7 blotter (has slippage, latency, TCA data)
    if (blotterData?.trades?.length) {
      for (let i = 0; i < blotterData.trades.length; i++) {
        const b = blotterData.trades[i];
        const key = `${b.ticker}|${b.timestamp}`;
        blotterKeys.add(key);
        const ts = b.timestamp ? new Date(b.timestamp) : null;
        txns.push({
          id: `BLT-${i}`,
          time: ts ? ts.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + ts.getMilliseconds().toString().padStart(3, "0") : "--:--:--.---",
          ticker: b.ticker || "—",
          side: b.side?.toUpperCase() || "—",
          qty: b.quantity || 0,
          price: b.fill_price || 0,
          notional: +(b.fill_price * b.quantity).toFixed(2),
          fillType: b.fill_price > 0 ? "FULL" : "PENDING",
          venue: b.routing_strategy || "ENGINE",
          signalType: b.signal_type || "—",
          latencyMs: 0,
          slippageBps: b.slippage_bps || 0,
          orderType: "MARKET",
          routingStrategy: b.routing_strategy || "SMART",
          urgency: b.urgency || "MEDIUM",
          productType: b.product_type || "EQUITY",
        });
      }
    }

    // Secondary: Alpaca orders (enriches with order status, type)
    if (ordersData?.orders?.length) {
      for (const o of ordersData.orders) {
        // Skip if already in blotter by ticker+time match
        const key = `${o.ticker}|${o.filled_at}`;
        if (blotterKeys.has(key)) continue;

        const ts = o.filled_at && o.filled_at !== "None" ? new Date(o.filled_at) : (
          o.submitted_at && o.submitted_at !== "None" ? new Date(o.submitted_at) : null
        );
        txns.push({
          id: o.id || `ORD-${txns.length}`,
          time: ts && !isNaN(ts.getTime()) ? ts.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }) + "." + ts.getMilliseconds().toString().padStart(3, "0") : "--:--:--.---",
          ticker: o.ticker || "—",
          side: o.side?.toUpperCase() || "—",
          qty: o.qty || 0,
          price: o.price || 0,
          notional: o.notional || 0,
          fillType: o.fill_type || "FULL",
          venue: "ALPACA",
          signalType: o.signal_type || "BROKER",
          latencyMs: 0,
          slippageBps: 0,
          orderType: o.order_type || "MARKET",
          routingStrategy: "SMART",
          urgency: "MEDIUM",
          productType: "EQUITY",
        });
      }
    }

    // Sort by time descending
    txns.sort((a, b) => (b.time > a.time ? 1 : b.time < a.time ? -1 : 0));
    return txns;
  }, [ordersData, blotterData]);

  const [filter, setFilter] = useState({ ticker: "", side: "", fill: "", signal: "" });
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Derive filter options from actual data
  const sides = useMemo(() => [...new Set(allTxns.map((t) => t.side).filter(Boolean))].sort(), [allTxns]);
  const statuses = useMemo(() => [...new Set(allTxns.map((t) => t.fillType).filter(Boolean))].sort(), [allTxns]);
  const dataSource = ordersData?.source || (blotterData?.broker_type ? `blotter:${blotterData.broker_type}` : "connecting...");

  const filteredTxns = useMemo(() => {
    return allTxns.filter((t) => {
      if (filter.ticker && !t.ticker.includes(filter.ticker)) return false;
      if (filter.side && t.side !== filter.side) return false;
      if (filter.fill && t.fillType !== filter.fill) return false;
      return true;
    });
  }, [allTxns, filter]);

  const noData = allTxns.length === 0;

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
          <FilterBar filter={filter} setFilter={setFilter} sides={sides} statuses={statuses} source={dataSource} />

          {/* Table header */}
          <div className="flex items-center px-2 py-1.5 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50 font-mono flex-shrink-0">
            <span className="w-[100px]">Time</span>
            <span className="w-[55px]">Ticker</span>
            <span className="w-[50px]">Side</span>
            <span className="w-[55px] text-right">Qty</span>
            <span className="w-[75px] text-right">Price</span>
            <span className="w-[90px] text-right">Notional</span>
            <span className="w-[55px] text-center">Status</span>
            <span className="w-[55px]">Venue</span>
            <span className="w-[85px]">Signal</span>
            <span className="w-[55px] text-right">Slip bps</span>
            <span className="w-px bg-terminal-border/30 h-3 mx-1" />
            <span className="w-[60px]">Type</span>
            <span className="w-[60px]">Route</span>
            <span className="flex-1 text-right">Product</span>
          </div>

          {/* Scrollable rows */}
          <div className="flex-1 overflow-auto">
            {noData && (
              <div className="flex items-center justify-center h-32 text-[10px] text-terminal-text-faint font-mono">
                Awaiting trade data from broker...
              </div>
            )}

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
                  <span className="w-[75px] text-right">{t.price > 0 ? `$${t.price.toFixed(2)}` : "—"}</span>
                  <span className="w-[90px] text-right">{t.notional >= 1e6 ? (t.notional / 1e6).toFixed(2) + "M" : t.notional > 0 ? "$" + t.notional.toLocaleString(undefined, { maximumFractionDigits: 0 }) : "—"}</span>
                  <span className={`w-[55px] text-center text-[8px] font-semibold ${
                    t.fillType === "FULL" ? "text-terminal-positive" : t.fillType === "PARTIAL" ? "text-terminal-warning" : t.fillType === "PENDING" ? "text-terminal-text-faint" : "text-terminal-negative"
                  }`}>
                    {t.fillType}
                  </span>
                  <span className="w-[55px] text-terminal-text-faint">{t.venue}</span>
                  <span className="w-[85px] text-terminal-text-muted truncate">{t.signalType}</span>
                  <span className={`w-[55px] text-right ${t.slippageBps > 1 ? "text-terminal-negative" : t.slippageBps < -0.5 ? "text-terminal-positive" : "text-terminal-text-muted"}`}>
                    {t.slippageBps !== 0 ? (t.slippageBps > 0 ? "+" : "") + t.slippageBps.toFixed(2) : "—"}
                  </span>
                  <span className="w-px bg-terminal-border/30 h-3 mx-1" />
                  <span className="w-[60px] text-terminal-text-faint">{t.orderType}</span>
                  <span className="w-[60px] text-terminal-text-faint">{t.routingStrategy}</span>
                  <span className="flex-1 text-right text-terminal-text-faint truncate">{t.productType}</span>
                </div>

                {/* Expanded detail row */}
                {expandedId === t.id && (
                  <div className="px-4 py-2 bg-terminal-surface-2/30 border-b border-terminal-border/20 text-[9px] font-mono">
                    <div className="grid grid-cols-6 gap-4">
                      <div>
                        <div className="text-terminal-text-faint">Order ID</div>
                        <div className="text-terminal-text-primary break-all">{t.id}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Signal Source</div>
                        <div className="text-terminal-accent">{t.signalType}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Order Type</div>
                        <div className="text-terminal-text-primary">{t.orderType}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Routing</div>
                        <div className="text-terminal-text-primary">{t.routingStrategy}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Cost Basis</div>
                        <div className="text-terminal-text-primary">${(t.qty * t.price).toLocaleString(undefined, { maximumFractionDigits: 0 })}</div>
                      </div>
                      <div>
                        <div className="text-terminal-text-faint">Execution Quality</div>
                        <div className={t.slippageBps < 0 ? "text-terminal-positive" : t.slippageBps > 2 ? "text-terminal-negative" : "text-terminal-text-primary"}>
                          {t.slippageBps < 0 ? "GOOD" : t.slippageBps > 2 ? "POOR" : t.slippageBps === 0 ? "—" : "FAIR"}
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
