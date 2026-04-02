import { useState, useEffect, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { ResizableDashboard } from "@/components/resizable-panel";
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  LineChart, Line, BarChart, Bar, ComposedChart,
} from "recharts";

// ═══════════ MOCK DATA ═══════════

interface FuturesContract {
  symbol: string;
  name: string;
  exchange: string;
  expiry: string;
  tickSize: number;
  tickValue: number;
  multiplier: number;
  marginInit: number;
  marginMaint: number;
  lastPrice: number;
  change: number;
  changePct: number;
  volume: number;
  openInterest: number;
  high: number;
  low: number;
  settle: number;
}

interface FuturesPosition {
  id: string;
  symbol: string;
  side: "LONG" | "SHORT";
  qty: number;
  avgEntry: number;
  lastPrice: number;
  unrealizedPnl: number;
  marginUsed: number;
  expiry: string;
  daysToExpiry: number;
}

interface FuturesOrder {
  id: string;
  time: string;
  symbol: string;
  side: "BUY" | "SELL";
  type: "LIMIT" | "MARKET" | "STOP" | "STOP_LIMIT";
  qty: number;
  price: number | null;
  stopPrice: number | null;
  status: "WORKING" | "FILLED" | "CANCELLED" | "PARTIAL";
  filled: number;
}

const CONTRACTS: FuturesContract[] = [
  { symbol: "ESM6", name: "E-mini S&P 500", exchange: "CME", expiry: "Jun 2026", tickSize: 0.25, tickValue: 12.50, multiplier: 50, marginInit: 12980, marginMaint: 11800, lastPrice: 5342.75, change: 18.50, changePct: 0.35, volume: 1847293, openInterest: 2841738, high: 5358.00, low: 5318.25, settle: 5324.25 },
  { symbol: "NQM6", name: "E-mini NASDAQ 100", exchange: "CME", expiry: "Jun 2026", tickSize: 0.25, tickValue: 5.00, multiplier: 20, marginInit: 18700, marginMaint: 17000, lastPrice: 18742.50, change: 62.25, changePct: 0.33, volume: 892461, openInterest: 483921, high: 18810.00, low: 18648.75, settle: 18680.25 },
  { symbol: "YMM6", name: "E-mini Dow", exchange: "CBOT", expiry: "Jun 2026", tickSize: 1.0, tickValue: 5.00, multiplier: 5, marginInit: 9680, marginMaint: 8800, lastPrice: 39847, change: 142, changePct: 0.36, volume: 214738, openInterest: 189432, high: 39920, low: 39648, settle: 39705 },
  { symbol: "CLN6", name: "Crude Oil WTI", exchange: "NYMEX", expiry: "Jul 2026", tickSize: 0.01, tickValue: 10.00, multiplier: 1000, marginInit: 7150, marginMaint: 6500, lastPrice: 78.42, change: -0.87, changePct: -1.10, volume: 642189, openInterest: 412847, high: 79.84, low: 77.91, settle: 79.29 },
  { symbol: "GCQ6", name: "Gold", exchange: "COMEX", expiry: "Aug 2026", tickSize: 0.10, tickValue: 10.00, multiplier: 100, marginInit: 11000, marginMaint: 10000, lastPrice: 2348.60, change: 14.30, changePct: 0.61, volume: 287432, openInterest: 524891, high: 2362.40, low: 2330.10, settle: 2334.30 },
  { symbol: "ZBM6", name: "US T-Bond 30Y", exchange: "CBOT", expiry: "Jun 2026", tickSize: 0.03125, tickValue: 31.25, multiplier: 1000, marginInit: 4620, marginMaint: 4200, lastPrice: 118.28125, change: -0.34375, changePct: -0.29, volume: 384219, openInterest: 1248732, high: 118.84375, low: 117.90625, settle: 118.625 },
  { symbol: "ZNM6", name: "US 10Y T-Note", exchange: "CBOT", expiry: "Jun 2026", tickSize: 0.015625, tickValue: 15.625, multiplier: 1000, marginInit: 2310, marginMaint: 2100, lastPrice: 110.640625, change: -0.15625, changePct: -0.14, volume: 1847329, openInterest: 4892134, high: 110.890625, low: 110.453125, settle: 110.796875 },
  { symbol: "6EM6", name: "Euro FX", exchange: "CME", expiry: "Jun 2026", tickSize: 0.00005, tickValue: 6.25, multiplier: 125000, marginInit: 2860, marginMaint: 2600, lastPrice: 1.08425, change: 0.00175, changePct: 0.16, volume: 189432, openInterest: 587432, high: 1.08640, low: 1.08180, settle: 1.08250 },
];

function generatePositions(): FuturesPosition[] {
  return [
    { id: "P001", symbol: "ESM6", side: "LONG", qty: 12, avgEntry: 5318.50, lastPrice: 5342.75, unrealizedPnl: 14550, marginUsed: 155760, expiry: "Jun 2026", daysToExpiry: 82 },
    { id: "P002", symbol: "NQM6", side: "LONG", qty: 6, avgEntry: 18680.00, lastPrice: 18742.50, unrealizedPnl: 7500, marginUsed: 112200, expiry: "Jun 2026", daysToExpiry: 82 },
    { id: "P003", symbol: "CLN6", side: "SHORT", qty: 8, avgEntry: 79.85, lastPrice: 78.42, unrealizedPnl: 11440, marginUsed: 57200, expiry: "Jul 2026", daysToExpiry: 112 },
    { id: "P004", symbol: "GCQ6", side: "LONG", qty: 4, avgEntry: 2328.40, lastPrice: 2348.60, unrealizedPnl: 8080, marginUsed: 44000, expiry: "Aug 2026", daysToExpiry: 143 },
    { id: "P005", symbol: "ZBM6", side: "SHORT", qty: 10, avgEntry: 119.125, lastPrice: 118.28125, unrealizedPnl: 8437.50, marginUsed: 46200, expiry: "Jun 2026", daysToExpiry: 82 },
    { id: "P006", symbol: "6EM6", side: "LONG", qty: 5, avgEntry: 1.08250, lastPrice: 1.08425, unrealizedPnl: 1093.75, marginUsed: 14300, expiry: "Jun 2026", daysToExpiry: 82 },
  ];
}

function generateOrders(): FuturesOrder[] {
  return [
    { id: "O001", time: "09:31:14", symbol: "ESM6", side: "BUY", type: "LIMIT", qty: 4, price: 5335.00, stopPrice: null, status: "WORKING", filled: 0 },
    { id: "O002", time: "09:32:08", symbol: "CLN6", side: "SELL", type: "STOP", qty: 3, price: null, stopPrice: 77.50, status: "WORKING", filled: 0 },
    { id: "O003", time: "09:28:41", symbol: "NQM6", side: "BUY", type: "MARKET", qty: 2, price: null, stopPrice: null, status: "FILLED", filled: 2 },
    { id: "O004", time: "09:35:22", symbol: "GCQ6", side: "SELL", type: "LIMIT", qty: 2, price: 2365.00, stopPrice: null, status: "WORKING", filled: 0 },
    { id: "O005", time: "09:30:55", symbol: "ZBM6", side: "BUY", type: "STOP_LIMIT", qty: 5, price: 117.50, stopPrice: 117.75, status: "WORKING", filled: 0 },
  ];
}

function generateCurveData() {
  const months = ["Apr'26","May'26","Jun'26","Jul'26","Aug'26","Sep'26","Oct'26","Nov'26","Dec'26","Jan'27","Feb'27","Mar'27"];
  const esBase = 5342.75;
  const clBase = 78.42;
  return months.map((m, i) => ({
    month: m,
    es: +(esBase + (i * 3.2) - Math.random() * 2).toFixed(2),
    cl: +(clBase - (i * 0.45) + Math.random() * 0.3).toFixed(2),
  }));
}

function generateRollCalendar() {
  return [
    { contract: "ESM6 → ESU6", currentExp: "Jun 20, 2026", rollStart: "Jun 11, 2026", rollEnd: "Jun 18, 2026", daysToRoll: 71, status: "UPCOMING" },
    { contract: "NQM6 → NQU6", currentExp: "Jun 20, 2026", rollStart: "Jun 11, 2026", rollEnd: "Jun 18, 2026", daysToRoll: 71, status: "UPCOMING" },
    { contract: "CLN6 → CLQ6", currentExp: "Jul 21, 2026", rollStart: "Jul 14, 2026", rollEnd: "Jul 20, 2026", daysToRoll: 104, status: "SCHEDULED" },
    { contract: "GCQ6 → GCV6", currentExp: "Aug 27, 2026", rollStart: "Aug 20, 2026", rollEnd: "Aug 26, 2026", daysToRoll: 147, status: "SCHEDULED" },
    { contract: "ZBM6 → ZBU6", currentExp: "Jun 19, 2026", rollStart: "Jun 01, 2026", rollEnd: "Jun 17, 2026", daysToRoll: 61, status: "UPCOMING" },
  ];
}

function generateMarginSummary() {
  return {
    totalMarginUsed: 429660,
    availableMargin: 570340,
    accountEquity: 1000000,
    maintenanceMargin: 384000,
    marginUtilization: 42.97,
    excessLiquidity: 616000,
  };
}

// ═══════════ COMPONENTS ═══════════

function ContractSpecs({ contracts, selected, onSelect }: { contracts: FuturesContract[]; selected: string; onSelect: (s: string) => void }) {
  return (
    <div className="space-y-0.5 text-[10px]">
      {contracts.map((c) => (
        <div
          key={c.symbol}
          onClick={() => onSelect(c.symbol)}
          className={`flex items-center justify-between px-2 py-1.5 rounded cursor-pointer transition-colors ${
            selected === c.symbol
              ? "bg-terminal-accent/10 border border-terminal-accent/30"
              : "hover:bg-white/[0.03] border border-transparent"
          }`}
        >
          <div className="flex items-center gap-2 min-w-0">
            <span className="font-mono font-semibold text-terminal-text-primary w-12">{c.symbol}</span>
            <span className="text-terminal-text-muted truncate">{c.name}</span>
          </div>
          <div className="flex items-center gap-3 flex-shrink-0">
            <span className="font-mono text-terminal-text-primary">{c.lastPrice.toLocaleString()}</span>
            <span className={`font-mono w-16 text-right ${c.change >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {c.change >= 0 ? "+" : ""}{c.changePct.toFixed(2)}%
            </span>
            <span className="text-terminal-text-faint font-mono w-16 text-right">{(c.volume / 1000).toFixed(0)}K</span>
          </div>
        </div>
      ))}
    </div>
  );
}

function ContractDetail({ contract }: { contract: FuturesContract }) {
  return (
    <div className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-[10px] px-1">
      {[
        ["Exchange", contract.exchange],
        ["Expiry", contract.expiry],
        ["Tick Size", contract.tickSize.toString()],
        ["Tick Value", `$${contract.tickValue.toFixed(2)}`],
        ["Multiplier", `$${contract.multiplier.toLocaleString()}`],
        ["Init Margin", `$${contract.marginInit.toLocaleString()}`],
        ["Maint Margin", `$${contract.marginMaint.toLocaleString()}`],
        ["Settle", contract.settle.toLocaleString()],
        ["Day High", contract.high.toLocaleString()],
        ["Day Low", contract.low.toLocaleString()],
        ["Volume", contract.volume.toLocaleString()],
        ["Open Interest", contract.openInterest.toLocaleString()],
      ].map(([label, val]) => (
        <div key={label} className="flex justify-between">
          <span className="text-terminal-text-faint">{label}</span>
          <span className="font-mono text-terminal-text-primary">{val}</span>
        </div>
      ))}
    </div>
  );
}

function PositionsTable({ positions }: { positions: FuturesPosition[] }) {
  const totalPnl = positions.reduce((s, p) => s + p.unrealizedPnl, 0);
  const totalMargin = positions.reduce((s, p) => s + p.marginUsed, 0);

  return (
    <div className="text-[10px]">
      <div className="flex items-center gap-4 px-2 py-1 mb-1 text-terminal-text-faint">
        <span>POSITIONS: {positions.length}</span>
        <span>TOTAL P&L: <span className={totalPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>${totalPnl.toLocaleString()}</span></span>
        <span>MARGIN: <span className="text-terminal-text-primary">${totalMargin.toLocaleString()}</span></span>
      </div>
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
              <td className="py-1.5 px-1.5 font-mono font-semibold text-terminal-text-primary">{p.symbol}</td>
              <td className={`py-1.5 px-1.5 font-mono font-semibold ${p.side === "LONG" ? "text-terminal-positive" : "text-terminal-negative"}`}>{p.side}</td>
              <td className="py-1.5 px-1.5 font-mono">{p.qty}</td>
              <td className="py-1.5 px-1.5 font-mono">{p.avgEntry.toLocaleString()}</td>
              <td className="py-1.5 px-1.5 font-mono text-terminal-text-primary">{p.lastPrice.toLocaleString()}</td>
              <td className={`py-1.5 px-1.5 font-mono font-semibold ${p.unrealizedPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                {p.unrealizedPnl >= 0 ? "+" : ""}${p.unrealizedPnl.toLocaleString()}
              </td>
              <td className="py-1.5 px-1.5 font-mono">${p.marginUsed.toLocaleString()}</td>
              <td className="py-1.5 px-1.5 text-terminal-text-muted">{p.expiry}</td>
              <td className="py-1.5 px-1.5 font-mono">{p.daysToExpiry}d</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function OrderBook({ orders }: { orders: FuturesOrder[] }) {
  return (
    <div className="text-[10px]">
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
              <td className="py-1.5 px-1.5 font-mono text-terminal-text-muted">{o.time}</td>
              <td className="py-1.5 px-1.5 font-mono font-semibold text-terminal-text-primary">{o.symbol}</td>
              <td className={`py-1.5 px-1.5 font-mono font-semibold ${o.side === "BUY" ? "text-terminal-positive" : "text-terminal-negative"}`}>{o.side}</td>
              <td className="py-1.5 px-1.5 font-mono">{o.type}</td>
              <td className="py-1.5 px-1.5 font-mono">{o.qty}</td>
              <td className="py-1.5 px-1.5 font-mono">{o.price ? o.price.toLocaleString() : "—"}</td>
              <td className="py-1.5 px-1.5 font-mono">{o.stopPrice ? o.stopPrice.toLocaleString() : "—"}</td>
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
    </div>
  );
}

function OrderEntry({ contracts, selected }: { contracts: FuturesContract[]; selected: string }) {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState<"MARKET" | "LIMIT" | "STOP" | "STOP_LIMIT">("LIMIT");
  const [qty, setQty] = useState("1");
  const [price, setPrice] = useState("");
  const [stopPrice, setStopPrice] = useState("");

  const contract = contracts.find(c => c.symbol === selected);

  return (
    <div className="space-y-2 text-[10px] px-1">
      <div className="flex items-center gap-2 mb-2">
        <span className="font-mono font-semibold text-terminal-text-primary text-xs">{selected}</span>
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
              placeholder={contract ? contract.lastPrice.toString() : ""}
              className="flex-1 bg-terminal-bg border border-terminal-border rounded px-2 py-1 font-mono text-terminal-text-primary focus:border-terminal-accent outline-none"
              type="number"
              step={contract ? contract.tickSize : 0.01}
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
              step={contract ? contract.tickSize : 0.01}
            />
          </div>
        )}
      </div>
      {contract && (
        <div className="flex justify-between text-terminal-text-faint pt-1 border-t border-terminal-border/50">
          <span>Est. Margin: ${contract.marginInit.toLocaleString()}</span>
          <span>Notional: ${(contract.lastPrice * contract.multiplier * parseInt(qty || "1")).toLocaleString()}</span>
        </div>
      )}
      <button className={`w-full py-2 rounded text-[11px] font-bold tracking-wider transition-colors ${
        side === "BUY"
          ? "bg-terminal-positive hover:bg-terminal-positive/90 text-terminal-bg"
          : "bg-terminal-negative hover:bg-terminal-negative/90 text-terminal-bg"
      }`}>
        {side} {qty} {selected} @ {orderType}
      </button>
    </div>
  );
}

function MarginGauge({ margin }: { margin: ReturnType<typeof generateMarginSummary> }) {
  const pct = margin.marginUtilization;
  const color = pct > 80 ? "#f85149" : pct > 60 ? "#d29922" : "#00d4aa";

  return (
    <div className="space-y-2 text-[10px] px-1">
      <div className="flex items-center justify-between mb-1">
        <span className="text-terminal-text-faint">MARGIN UTILIZATION</span>
        <span className="font-mono font-semibold" style={{ color }}>{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, backgroundColor: color }} />
      </div>
      <div className="grid grid-cols-2 gap-x-3 gap-y-1 mt-2">
        {[
          ["Account Equity", `$${margin.accountEquity.toLocaleString()}`],
          ["Margin Used", `$${margin.totalMarginUsed.toLocaleString()}`],
          ["Available", `$${margin.availableMargin.toLocaleString()}`],
          ["Maintenance", `$${margin.maintenanceMargin.toLocaleString()}`],
          ["Excess Liquidity", `$${margin.excessLiquidity.toLocaleString()}`],
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

function TermStructureChart({ data }: { data: ReturnType<typeof generateCurveData> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
        <XAxis dataKey="month" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} />
        <YAxis yAxisId="es" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} domain={["dataMin - 5", "dataMax + 5"]} />
        <YAxis yAxisId="cl" orientation="right" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} domain={["dataMin - 1", "dataMax + 1"]} />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          labelStyle={{ color: "#8b949e" }}
        />
        <Line yAxisId="es" type="monotone" dataKey="es" stroke="#00d4aa" strokeWidth={1.5} dot={{ fill: "#00d4aa", r: 2 }} name="ES (S&P 500)" />
        <Line yAxisId="cl" type="monotone" dataKey="cl" stroke="#f0883e" strokeWidth={1.5} dot={{ fill: "#f0883e", r: 2 }} name="CL (Crude)" />
      </LineChart>
    </ResponsiveContainer>
  );
}

function RollCalendar({ rolls }: { rolls: ReturnType<typeof generateRollCalendar> }) {
  return (
    <div className="text-[10px] space-y-0.5">
      {rolls.map((r, i) => (
        <div key={i} className="flex items-center justify-between px-2 py-1.5 rounded hover:bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <span className="font-mono font-semibold text-terminal-text-primary w-28">{r.contract}</span>
            <span className="text-terminal-text-muted">{r.rollStart} → {r.rollEnd}</span>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-terminal-text-primary">{r.daysToRoll}d</span>
            <span className={`px-1.5 py-0.5 rounded text-[9px] font-semibold ${
              r.status === "UPCOMING" ? "bg-terminal-warning/20 text-terminal-warning" : "bg-terminal-accent/10 text-terminal-accent"
            }`}>{r.status}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

export default function FuturesPage() {
  const [selectedContract, setSelectedContract] = useState("ESM6");
  const [positions] = useState(generatePositions);
  const [orders] = useState(generateOrders);
  const [curveData] = useState(generateCurveData);
  const [rolls] = useState(generateRollCalendar);
  const [margin] = useState(generateMarginSummary);

  // Live price ticking
  const [contracts, setContracts] = useState(CONTRACTS);
  useEffect(() => {
    const interval = setInterval(() => {
      setContracts(prev => prev.map(c => {
        const tick = (Math.random() - 0.5) * c.tickSize * 4;
        const newPrice = +(c.lastPrice + tick).toFixed(c.tickSize < 0.01 ? 5 : c.tickSize < 1 ? 2 : 0);
        return { ...c, lastPrice: newPrice, change: +(c.change + tick).toFixed(2) };
      }));
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  const activeContract = contracts.find(c => c.symbol === selectedContract) || contracts[0];

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden" data-testid="futures-page">
      {/* Top: Contracts bar */}
      <div className="flex-shrink-0">
        <DashboardPanel title="FUTURES CONTRACTS" noPadding>
          <ContractSpecs contracts={contracts} selected={selectedContract} onSelect={setSelectedContract} />
        </DashboardPanel>
      </div>

      {/* Main resizable area */}
      <div className="flex-1 min-h-0">
        <ResizableDashboard defaultSizes={[72, 28]} minSizes={[40, 18]}>
          {/* Left: Positions + Orders + Term Structure */}
          <div className="h-full flex flex-col gap-1">
            <DashboardPanel title="POSITIONS" className="flex-1" noPadding>
              <PositionsTable positions={positions} />
            </DashboardPanel>
            <DashboardPanel title="ORDERS" className="flex-shrink-0 max-h-[180px]" noPadding>
              <OrderBook orders={orders} />
            </DashboardPanel>
            <DashboardPanel title="TERM STRUCTURE" className="flex-shrink-0 h-[160px]">
              <TermStructureChart data={curveData} />
            </DashboardPanel>
          </div>

          {/* Right sidebar: Specs + Order Entry + Margin + Roll Calendar */}
          <div className="h-full flex flex-col gap-1">
            <DashboardPanel title={`${activeContract.symbol} SPECS`} className="flex-shrink-0">
              <ContractDetail contract={activeContract} />
            </DashboardPanel>
            <DashboardPanel title="ORDER ENTRY" className="flex-1">
              <OrderEntry contracts={contracts} selected={selectedContract} />
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
