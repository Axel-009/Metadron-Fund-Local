import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from "recharts";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ API TYPES ═══════════

interface ContractData {
  symbol: string;
  root: string;
  yahoo_ticker: string;
  name: string;
  exchange: string;
  expiry: string;
  expiry_date: string;
  days_to_expiry: number;
  tick_size: number;
  tick_value: number;
  multiplier: number;
  margin_init: number;
  margin_maint: number;
  last_price: number;
  change: number;
  change_pct: number;
  volume: number;
  high: number;
  low: number;
  settle: number;
  open: number;
  source: string;
}

interface PositionData {
  id: string;
  symbol: string;
  root: string;
  name: string;
  side: string;
  qty: number;
  avg_entry: number;
  last_price: number;
  unrealized_pnl: number;
  margin_used: number;
  notional: number;
  expiry: string;
  days_to_expiry: number;
  is_proxy: boolean;
}

interface OrderData {
  id: string;
  time: string;
  symbol: string;
  side: string;
  type: string;
  qty: number;
  price: number;
  stop_price: number;
  status: string;
  filled: number;
}

interface MarginData {
  account_equity: number;
  cash: number;
  total_margin_used: number;
  available_margin: number;
  maintenance_margin: number;
  margin_utilization: number;
  excess_liquidity: number;
  futures_positions: number;
}

interface CurvePoint {
  month: string;
  price: number;
  date: string;
  projected?: boolean;
}

interface RollData {
  contract: string;
  root: string;
  name: string;
  current_expiry: string;
  roll_start: string;
  roll_end: string;
  days_to_roll: number;
  days_to_expiry: number;
  status: string;
}

interface SummaryData {
  total_pnl: number;
  total_notional: number;
  total_margin: number;
  position_count: number;
  long_count: number;
  short_count: number;
  margin_utilization: number;
  nav: number;
}

// ═══════════ COMPONENTS ═══════════

function SummaryCards({ summary, beta }: { summary: SummaryData; beta: Record<string, unknown> }) {
  const cards = [
    { label: "POSITIONS", value: `${summary.position_count}`, sub: `${summary.long_count}L / ${summary.short_count}S` },
    { label: "UNREALIZED P&L", value: `$${summary.total_pnl >= 0 ? "+" : ""}${summary.total_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: "all contracts", color: summary.total_pnl >= 0 },
    { label: "NOTIONAL", value: `$${(summary.total_notional / 1e6).toFixed(2)}M`, sub: "total exposure" },
    { label: "MARGIN USED", value: `$${summary.total_margin.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, sub: `${summary.margin_utilization.toFixed(1)}% utilized` },
    { label: "PORTFOLIO BETA", value: `${Number(beta?.current_beta || 0).toFixed(3)}`, sub: `target: ${Number(beta?.target_beta || 0).toFixed(3)}` },
    { label: "NAV", value: `$${(summary.nav / 1e6).toFixed(2)}M`, sub: "account equity" },
  ];

  return (
    <div className="grid grid-cols-6 gap-[2px]">
      {cards.map((c) => (
        <div key={c.label} className="terminal-panel p-2">
          <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">{c.label}</div>
          <div className={`text-sm font-mono tabular-nums mt-0.5 ${
            c.color !== undefined ? (c.color ? "text-terminal-positive" : "text-terminal-negative") : "text-terminal-text-primary"
          }`}>{c.value}</div>
          <div className="text-[8px] text-terminal-text-faint mt-0.5">{c.sub}</div>
        </div>
      ))}
    </div>
  );
}

function ContractSpecs({ contracts, selected, onSelect }: { contracts: ContractData[]; selected: string; onSelect: (s: string) => void }) {
  return (
    <div className="space-y-0.5 text-[10px]">
      {contracts.map((c) => (
        <div
          key={c.symbol}
          onClick={() => onSelect(c.root)}
          className={`flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors ${
            selected === c.root
              ? "bg-terminal-accent/10 border border-terminal-accent/30"
              : "hover:bg-white/[0.03] border border-transparent"
          }`}
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono font-semibold text-terminal-text-primary w-12">{c.symbol}</span>
            <span className="text-terminal-text-muted truncate">{c.name}</span>
            <span className="text-[8px] text-terminal-text-faint">{c.exchange}</span>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <span className="font-mono text-terminal-text-primary">{c.last_price > 0 ? c.last_price.toLocaleString(undefined, { maximumFractionDigits: 4 }) : "—"}</span>
            <span className={`font-mono w-16 text-right ${c.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {c.last_price > 0 ? `${c.change >= 0 ? "+" : ""}${c.change_pct.toFixed(2)}%` : "—"}
            </span>
            <span className="text-terminal-text-faint font-mono w-16 text-right">{c.volume > 0 ? `${(c.volume / 1000).toFixed(0)}K` : "—"}</span>
            <span className="text-terminal-text-faint font-mono w-10 text-right text-[9px]">{c.days_to_expiry}d</span>
          </div>
        </div>
      ))}
      {contracts.length === 0 && (
        <div className="text-center py-4 text-terminal-text-faint">Loading contracts from OpenBB...</div>
      )}
    </div>
  );
}

function ContractDetail({ contract }: { contract: ContractData }) {
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px] px-1">
      {[
        ["Exchange", contract.exchange],
        ["Expiry", contract.expiry],
        ["Tick Size", contract.tick_size.toString()],
        ["Tick Value", `$${contract.tick_value.toFixed(2)}`],
        ["Multiplier", `$${contract.multiplier.toLocaleString()}`],
        ["Init Margin", `$${contract.margin_init.toLocaleString()}`],
        ["Maint Margin", `$${contract.margin_maint.toLocaleString()}`],
        ["Settle", contract.settle > 0 ? contract.settle.toLocaleString() : "—"],
        ["Day High", contract.high > 0 ? contract.high.toLocaleString() : "—"],
        ["Day Low", contract.low > 0 ? contract.low.toLocaleString() : "—"],
        ["Volume", contract.volume > 0 ? contract.volume.toLocaleString() : "—"],
        ["DTE", `${contract.days_to_expiry} days`],
        ["Source", contract.source || "—"],
      ].map(([label, val]) => (
        <div key={label} className="flex justify-between">
          <span className="text-terminal-text-faint">{label}</span>
          <span className="font-mono text-terminal-text-primary">{val}</span>
        </div>
      ))}
    </div>
  );
}

function PositionsTable({ positions, totals }: { positions: PositionData[]; totals: { pnl: number; margin: number; count: number } }) {
  return (
    <div className="text-[10px]">
      <div className="flex items-center gap-4 px-2 py-1 mb-1 text-terminal-text-faint">
        <span>POSITIONS: {totals.count}</span>
        <span>TOTAL P&L: <span className={totals.pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>${totals.pnl.toLocaleString()}</span></span>
        <span>MARGIN: <span className="text-terminal-text-primary">${totals.margin.toLocaleString()}</span></span>
      </div>
      {positions.length === 0 ? (
        <div className="text-center py-6 text-terminal-text-faint font-mono">No futures positions — awaiting broker data</div>
      ) : (
        <table className="w-full">
          <thead>
            <tr className="text-terminal-text-faint border-b border-terminal-border">
              {["Symbol","Side","Qty","Avg Entry","Last","Unrl P&L","Margin","Expiry","DTE"].map(h => (
                <th key={h} className="py-1 px-1.5 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {positions.map(p => (
              <tr key={p.id} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
                <td className="py-1.5 px-1.5 font-mono font-semibold text-terminal-text-primary">
                  {p.symbol}{p.is_proxy && <span className="text-[8px] text-terminal-text-faint ml-1">PROXY</span>}
                </td>
                <td className={`py-1.5 px-1.5 font-mono font-semibold ${p.side === "LONG" ? "text-terminal-positive" : "text-terminal-negative"}`}>{p.side}</td>
                <td className="py-1.5 px-1.5 font-mono">{p.qty}</td>
                <td className="py-1.5 px-1.5 font-mono">{p.avg_entry.toLocaleString()}</td>
                <td className="py-1.5 px-1.5 font-mono text-terminal-text-primary">{p.last_price.toLocaleString()}</td>
                <td className={`py-1.5 px-1.5 font-mono font-semibold ${p.unrealized_pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                  {p.unrealized_pnl >= 0 ? "+" : ""}${p.unrealized_pnl.toLocaleString()}
                </td>
                <td className="py-1.5 px-1.5 font-mono">${p.margin_used.toLocaleString()}</td>
                <td className="py-1.5 px-1.5 text-terminal-text-muted">{p.expiry}</td>
                <td className="py-1.5 px-1.5 font-mono">{p.days_to_expiry}d</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function OrderBook({ orders }: { orders: OrderData[] }) {
  return (
    <div className="text-[10px]">
      {orders.length === 0 ? (
        <div className="text-center py-4 text-terminal-text-faint font-mono">No futures orders</div>
      ) : (
        <table className="w-full">
          <thead>
            <tr className="text-terminal-text-faint border-b border-terminal-border">
              {["Time","Symbol","Side","Type","Qty","Price","Stop","Status","Filled"].map(h => (
                <th key={h} className="py-1 px-1.5 text-left font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {orders.map(o => (
              <tr key={o.id} className="border-b border-terminal-border/50 hover:bg-white/[0.02]">
                <td className="py-1.5 px-1.5 font-mono text-terminal-text-muted">{o.time?.slice(11, 19) || o.time}</td>
                <td className="py-1.5 px-1.5 font-mono font-semibold text-terminal-text-primary">{o.symbol}</td>
                <td className={`py-1.5 px-1.5 font-mono font-semibold ${o.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{o.side}</td>
                <td className="py-1.5 px-1.5 font-mono">{o.type}</td>
                <td className="py-1.5 px-1.5 font-mono">{o.qty}</td>
                <td className="py-1.5 px-1.5 font-mono">{o.price > 0 ? o.price.toLocaleString() : "—"}</td>
                <td className="py-1.5 px-1.5 font-mono">{o.stop_price > 0 ? o.stop_price.toLocaleString() : "—"}</td>
                <td className="py-1.5 px-1.5">
                  <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold ${
                    o.status === "FILLED" ? "bg-terminal-positive/20 text-terminal-positive" :
                    o.status === "WORKING" ? "bg-terminal-accent/20 text-terminal-accent" :
                    o.status === "PARTIAL" ? "bg-terminal-warning/20 text-terminal-warning" :
                    "bg-terminal-text-faint/20 text-terminal-text-faint"
                  }`}>{o.status}</span>
                </td>
                <td className="py-1.5 px-1.5 font-mono">{o.filled}/{o.qty}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

function OrderEntry({ contracts, selected }: { contracts: ContractData[]; selected: string }) {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState<"MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT">("LIMIT");
  const [qty, setQty] = useState("1");
  const [price, setPrice] = useState("");
  const [stopPrice, setStopPrice] = useState("");

  const contract = contracts.find(c => c.root === selected);

  return (
    <div className="space-y-2 text-[10px] px-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono font-semibold text-terminal-text-primary text-xs">{contract?.symbol || selected}</span>
        {contract && <span className="text-terminal-text-muted">{contract.name}</span>}
      </div>
      <div className="grid grid-cols-2 gap-1.5">
        <button
          onClick={() => setSide("BUY")}
          className={`py-1.5 rounded text-[10px] font-bold tracking-wider transition-colors ${
            side === "BUY"
              ? "bg-terminal-positive text-terminal-bg"
              : "bg-terminal-positive/10 text-terminal-positive hover:bg-terminal-positive/20"
          }`}
        >BUY</button>
        <button
          onClick={() => setSide("SELL")}
          className={`py-1.5 rounded text-[10px] font-bold tracking-wider transition-colors ${
            side === "SELL"
              ? "bg-terminal-negative text-terminal-bg"
              : "bg-terminal-negative/10 text-terminal-negative hover:bg-terminal-negative/20"
          }`}
        >SELL</button>
      </div>
      <div className="grid grid-cols-4 gap-1">
        {(["MARKET","LIMIT","STOP","STOP_LIMIT"] as const).map(t => (
          <button
            key={t}
            onClick={() => setOrderType(t)}
            className={`py-1 rounded text-[9px] font-medium transition-colors ${
              orderType === t
                ? "bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/30"
                : "bg-white/[0.03] text-terminal-text-muted hover:bg-white/[0.06] border border-transparent"
            }`}
          >{t.replace("_", " ")}</button>
        ))}
      </div>
      <div className="space-y-1.5">
        <div className="flex items-center gap-2">
          <label className="text-terminal-text-faint w-10">Qty</label>
          <input
            value={qty}
            onChange={e => setQty(e.target.value)}
            className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1 font-mono text-terminal-text-primary focus:border-terminal-accent outline-none"
            type="number"
            min="1"
          />
        </div>
        {(orderType === "LIMIT" || orderType === "STOP_LIMIT") && (
          <div className="flex items-center gap-2">
            <label className="text-terminal-text-faint w-10">Price</label>
            <input
              value={price}
              onChange={e => setPrice(e.target.value)}
              placeholder={contract ? contract.last_price.toString() : ""}
              className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1 font-mono text-terminal-text-primary focus:border-terminal-accent outline-none"
              type="number"
              step={contract ? contract.tick_size : 0.01}
            />
          </div>
        )}
        {(orderType === "STOP" || orderType === "STOP_LIMIT") && (
          <div className="flex items-center gap-2">
            <label className="text-terminal-text-faint w-10">Stop</label>
            <input
              value={stopPrice}
              onChange={e => setStopPrice(e.target.value)}
              className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1 font-mono text-terminal-text-primary focus:border-terminal-accent outline-none"
              type="number"
              step={contract ? contract.tick_size : 0.01}
            />
          </div>
        )}
      </div>
      {contract && (
        <div className="flex justify-between text-terminal-text-faint pt-1 border-t border-terminal-border/50">
          <span>Est. Margin: ${contract.margin_init.toLocaleString()}</span>
          <span>Notional: ${(contract.last_price * contract.multiplier * parseInt(qty || "1")).toLocaleString()}</span>
        </div>
      )}
      <button className={`w-full py-2 rounded text-[11px] font-bold tracking-wider transition-colors ${
        side === "BUY"
          ? "bg-terminal-positive hover:bg-terminal-positive/90 text-terminal-bg"
          : "bg-terminal-negative hover:bg-terminal-negative/90 text-terminal-bg"
      }`}>
        {side} {qty} {contract?.symbol || selected} @ {orderType}
      </button>
    </div>
  );
}

function MarginGauge({ margin }: { margin: MarginData }) {
  const pct = margin.margin_utilization || 0;
  const color = pct > 80 ? "#f85149" : pct > 60 ? "#d29922" : "#00d4aa";

  return (
    <div className="space-y-2 text-[10px] px-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-terminal-text-faint">MARGIN UTILIZATION</span>
        <span className="font-mono font-semibold" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${Math.min(pct, 100)}%`, backgroundColor: color }} />
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2">
        {[
          ["Account Equity", `$${margin.account_equity.toLocaleString()}`],
          ["Margin Used", `$${margin.total_margin_used.toLocaleString()}`],
          ["Available", `$${margin.available_margin.toLocaleString()}`],
          ["Maintenance", `$${margin.maintenance_margin.toLocaleString()}`],
          ["Excess Liquidity", `$${margin.excess_liquidity.toLocaleString()}`],
          ["Cash", `$${margin.cash.toLocaleString()}`],
        ].map(([l, v]) => (
          <div key={l} className="flex justify-between">
            <span className="text-terminal-text-faint">{l}</span>
            <span className="font-mono text-terminal-text-primary">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TermStructureChart({ curve, structure }: { curve: CurvePoint[]; structure: string }) {
  if (!curve.length) {
    return <div className="flex items-center justify-center h-full text-[10px] text-terminal-text-faint font-mono">Loading term structure...</div>;
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-2 px-2 pb-1 text-[9px]">
        <span className={`px-1.5 py-0.5 rounded font-semibold ${
          structure === "CONTANGO" ? "bg-terminal-positive/20 text-terminal-positive" :
          structure === "BACKWARDATION" ? "bg-terminal-negative/20 text-terminal-negative" :
          "bg-terminal-text-faint/20 text-terminal-text-faint"
        }`}>{structure}</span>
      </div>
      <div className="flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={curve} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
            <XAxis dataKey="month" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
            <YAxis tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} domain={["dataMin - 5", "dataMax + 5"]} />
            <Tooltip
              contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
              labelStyle={{ color: "#8b949e" }}
            />
            <Line type="monotone" dataKey="price" stroke="#00d4aa" strokeWidth={1.5} dot={({ cx, cy, payload }: { cx: number; cy: number; payload: CurvePoint }) => (
              <circle cx={cx} cy={cy} r={2.5} fill={payload.projected ? "#f0883e" : "#00d4aa"} stroke="none" />
            )} name="Price" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function RollCalendar({ rolls }: { rolls: RollData[] }) {
  if (!rolls.length) {
    return <div className="text-center py-4 text-[10px] text-terminal-text-faint font-mono">Loading roll calendar...</div>;
  }

  return (
    <div className="text-[10px] space-y-0.5">
      {rolls.map((r, i) => (
        <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <span className="font-mono font-semibold text-terminal-text-primary w-16">{r.contract}</span>
            <span className="text-terminal-text-muted truncate">{r.name}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="text-terminal-text-faint text-[9px]">{r.roll_start} → {r.roll_end}</span>
            <span className="font-mono text-terminal-text-primary w-10 text-right">{r.days_to_roll}d</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold ${
              r.status === "ACTIVE" ? "bg-terminal-negative/20 text-terminal-negative" :
              r.status === "UPCOMING" ? "bg-terminal-warning/20 text-terminal-warning" :
              "bg-terminal-accent/10 text-terminal-accent"
            }`}>{r.status}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function FuturesPage() {
  // ─── Engine API: all live data ───────────────────────────────
  const { data: contractsData } = useEngineQuery<{ contracts: ContractData[] }>(
    "/futures/contracts", { refetchInterval: 10000 },
  );
  const { data: positionsData } = useEngineQuery<{ positions: PositionData[]; totals: { pnl: number; margin: number; count: number } }>(
    "/futures/positions", { refetchInterval: 5000 },
  );
  const { data: ordersData } = useEngineQuery<{ orders: OrderData[] }>(
    "/futures/orders?limit=50", { refetchInterval: 5000 },
  );
  const { data: marginData } = useEngineQuery<{ margin: MarginData }>(
    "/futures/margin", { refetchInterval: 10000 },
  );
  const { data: rollsData } = useEngineQuery<{ rolls: RollData[] }>(
    "/futures/rolls", { refetchInterval: 60000 },
  );
  const { data: summaryData } = useEngineQuery<{ summary: SummaryData; beta: Record<string, unknown> }>(
    "/futures/summary", { refetchInterval: 5000 },
  );

  const [selectedRoot, setSelectedRoot] = useState("ES");

  // Term structure for selected contract
  const { data: curveData } = useEngineQuery<{ curve: CurvePoint[]; structure: string }>(
    `/futures/curve?root=${selectedRoot}`, { refetchInterval: 30000 },
  );

  const contracts = contractsData?.contracts || [];
  const positions = positionsData?.positions || [];
  const orders = ordersData?.orders || [];
  const margin = marginData?.margin || { account_equity: 0, cash: 0, total_margin_used: 0, available_margin: 0, maintenance_margin: 0, margin_utilization: 0, excess_liquidity: 0, futures_positions: 0 };
  const rolls = rollsData?.rolls || [];
  const summary = summaryData?.summary || { total_pnl: 0, total_notional: 0, total_margin: 0, position_count: 0, long_count: 0, short_count: 0, margin_utilization: 0, nav: 0 };
  const beta = summaryData?.beta || {};
  const curve = curveData?.curve || [];
  const structure = curveData?.structure || "FLAT";
  const totals = positionsData?.totals || { pnl: 0, margin: 0, count: 0 };

  const activeContract = useMemo(() =>
    contracts.find(c => c.root === selectedRoot) || contracts[0],
    [contracts, selectedRoot],
  );

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden" data-testid="futures-page">
      {/* Top: Summary cards */}
      <div className="flex-shrink-0">
        <SummaryCards summary={summary} beta={beta} />
      </div>

      {/* Contract list bar */}
      <div className="flex-shrink-0">
        <DashboardPanel title="FUTURES CONTRACTS" noPadding headerRight={
          <span className="text-[8px] text-terminal-text-faint font-mono">{contracts.length} contracts</span>
        }>
          <ContractSpecs contracts={contracts} selected={selectedRoot} onSelect={setSelectedRoot} />
        </DashboardPanel>
      </div>

      {/* Main resizable area */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[72, 28]} minSizes={[40, 18]}>
          {/* Left: Positions + Orders + Term Structure */}
          <div className="h-full flex flex-col gap-1">
            <DashboardPanel title="POSITIONS" className="flex-1" noPadding>
              <PositionsTable positions={positions} totals={totals} />
            </DashboardPanel>
            <DashboardPanel title="ORDERS" className="flex-shrink-0 max-h-[180px]" noPadding>
              <OrderBook orders={orders} />
            </DashboardPanel>
            <DashboardPanel title={`TERM STRUCTURE — ${selectedRoot}`} className="flex-shrink-0 h-[170px]">
              <TermStructureChart curve={curve} structure={structure} />
            </DashboardPanel>
          </div>

          {/* Right sidebar: Specs + Order Entry + Margin + Roll Calendar */}
          <div className="h-full flex flex-col gap-1">
            {activeContract && (
              <DashboardPanel title={`${activeContract.symbol} SPECS`} className="flex-shrink-0">
                <ContractDetail contract={activeContract} />
              </DashboardPanel>
            )}
            <DashboardPanel title="ORDER ENTRY" className="flex-1">
              <OrderEntry contracts={contracts} selected={selectedRoot} />
            </DashboardPanel>
            <DashboardPanel title="MARGIN SUMMARY" className="flex-shrink-0">
              <MarginGauge margin={margin} />
            </DashboardPanel>
            <DashboardPanel title="ROLL CALENDAR" className="flex-shrink-0">
              <RollCalendar rolls={rolls} />
            </DashboardPanel>
          </div>
        </ResizableDashboard>
      </div>
    </div>
  );
}
