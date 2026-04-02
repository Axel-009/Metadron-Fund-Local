import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  ReferenceLine,
} from "recharts";

// ═══════════ TYPES ═══════════

type ReconStatus = "MATCHED" | "MISMATCH" | "PAPER_ONLY" | "ALPACA_ONLY";

interface ReconPosition {
  ticker: string;
  sector: string;
  paperQty: number | null;
  alpacaQty: number | null;
  qtyMatch: boolean | null;
  paperAvgPrice: number | null;
  alpacaAvgPrice: number | null;
  priceDiff: number | null;
  paperPnl: number | null;
  alpacaPnl: number | null;
  pnlDiff: number | null;
  status: ReconStatus;
}

// ═══════════ MOCK DATA ═══════════

function generatePositions(): ReconPosition[] {
  const tickers: { t: string; s: string; base: number }[] = [
    { t: "AAPL", s: "Technology", base: 189.0 },
    { t: "MSFT", s: "Technology", base: 415.0 },
    { t: "NVDA", s: "Technology", base: 878.0 },
    { t: "GOOGL", s: "Communication", base: 174.0 },
    { t: "AMZN", s: "Consumer Disc.", base: 189.0 },
    { t: "JPM", s: "Financials", base: 198.0 },
    { t: "XOM", s: "Energy", base: 113.0 },
    { t: "JNJ", s: "Healthcare", base: 147.0 },
    { t: "V", s: "Financials", base: 274.0 },
    { t: "META", s: "Communication", base: 498.0 },
    { t: "UNH", s: "Healthcare", base: 521.0 },
    { t: "HD", s: "Consumer Disc.", base: 362.0 },
    { t: "BAC", s: "Financials", base: 38.0 },
    { t: "TSLA", s: "Consumer Disc.", base: 248.0 },
    { t: "LLY", s: "Healthcare", base: 786.0 },
    { t: "AVGO", s: "Technology", base: 1340.0 },
    { t: "MRK", s: "Healthcare", base: 128.0 },
    { t: "KO", s: "Consumer Staples", base: 62.0 },
    { t: "CVX", s: "Energy", base: 157.0 },
    { t: "PG", s: "Consumer Staples", base: 152.0 },
  ];

  return tickers.map(({ t, s, base }, i) => {
    const qty = Math.floor(100 + Math.random() * 900) * 10;
    const avgEntry = base * (0.88 + Math.random() * 0.10);
    const lastPrice = base;
    const pnl = (lastPrice - avgEntry) * qty;

    // Introduce specific anomalies
    const isMismatch = i === 3 || i === 8 || i === 14 || i === 17; // Qty mismatch
    const isPaperOnly = i === 19; // PG — paper only
    const isAlpacaOnly = i === 16; // MRK — alpaca only (paper position closed)

    if (isPaperOnly) {
      const pp = avgEntry * (1 + (Math.random() - 0.5) * 0.02);
      return {
        ticker: t,
        sector: s,
        paperQty: qty,
        alpacaQty: null,
        qtyMatch: null,
        paperAvgPrice: +avgEntry.toFixed(2),
        alpacaAvgPrice: null,
        priceDiff: null,
        paperPnl: +(pnl).toFixed(0),
        alpacaPnl: null,
        pnlDiff: null,
        status: "PAPER_ONLY" as ReconStatus,
      };
    }

    if (isAlpacaOnly) {
      const ap = avgEntry * (1 + (Math.random() - 0.5) * 0.02);
      return {
        ticker: t,
        sector: s,
        paperQty: null,
        alpacaQty: qty,
        qtyMatch: null,
        paperAvgPrice: null,
        alpacaAvgPrice: +avgEntry.toFixed(2),
        priceDiff: null,
        paperPnl: null,
        alpacaPnl: +(pnl).toFixed(0),
        pnlDiff: null,
        status: "ALPACA_ONLY" as ReconStatus,
      };
    }

    if (isMismatch) {
      // Qty is slightly off due to partial fill / rounding
      const alpacaQty = qty + (Math.random() > 0.5 ? 10 : -10);
      const alpacaAvg = avgEntry * (1 + (Math.random() - 0.5) * 0.008);
      const alpacaPnl = (lastPrice - alpacaAvg) * alpacaQty;
      const paperAvgFmt = +avgEntry.toFixed(2);
      const alpacaAvgFmt = +alpacaAvg.toFixed(2);
      return {
        ticker: t,
        sector: s,
        paperQty: qty,
        alpacaQty,
        qtyMatch: false,
        paperAvgPrice: paperAvgFmt,
        alpacaAvgPrice: alpacaAvgFmt,
        priceDiff: +(alpacaAvgFmt - paperAvgFmt).toFixed(2),
        paperPnl: +pnl.toFixed(0),
        alpacaPnl: +alpacaPnl.toFixed(0),
        pnlDiff: +(alpacaPnl - pnl).toFixed(0),
        status: "MISMATCH" as ReconStatus,
      };
    }

    // Perfectly matched
    const alpacaAvg = avgEntry * (1 + (Math.random() - 0.5) * 0.002); // tiny rounding
    const alpacaAvgFmt = +alpacaAvg.toFixed(2);
    const paperAvgFmt = +avgEntry.toFixed(2);
    const alpacaPnl = (lastPrice - alpacaAvg) * qty;
    return {
      ticker: t,
      sector: s,
      paperQty: qty,
      alpacaQty: qty,
      qtyMatch: true,
      paperAvgPrice: paperAvgFmt,
      alpacaAvgPrice: alpacaAvgFmt,
      priceDiff: +(alpacaAvgFmt - paperAvgFmt).toFixed(2),
      paperPnl: +pnl.toFixed(0),
      alpacaPnl: +alpacaPnl.toFixed(0),
      pnlDiff: +(alpacaPnl - pnl).toFixed(0),
      status: "MATCHED" as ReconStatus,
    };
  });
}

function generateNavHistory(baseNav: number) {
  const data: { date: string; paper: number; alpaca: number }[] = [];
  let paper = baseNav * 0.88;
  let alpaca = paper * (1 - 0.0003);
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000);
    const dailyReturn = 0.0004 + (Math.random() - 0.48) * 0.012;
    paper *= 1 + dailyReturn;
    alpaca *= 1 + dailyReturn + (Math.random() - 0.5) * 0.0008;
    data.push({
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      paper: +paper.toFixed(0),
      alpaca: +alpaca.toFixed(0),
    });
  }
  return data;
}

// ═══════════ STATUS HELPERS ═══════════

const STATUS_STYLES: Record<ReconStatus, { color: string; bg: string; border: string }> = {
  MATCHED: { color: "#3fb950", bg: "#3fb95018", border: "#3fb95040" },
  MISMATCH: { color: "#f85149", bg: "#f8514918", border: "#f8514940" },
  PAPER_ONLY: { color: "#d29922", bg: "#d2992218", border: "#d2992240" },
  ALPACA_ONLY: { color: "#58a6ff", bg: "#58a6ff18", border: "#58a6ff40" },
};

function StatusBadge({ status }: { status: ReconStatus }) {
  const s = STATUS_STYLES[status];
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[8px] font-bold tracking-wider whitespace-nowrap"
      style={{ color: s.color, backgroundColor: s.bg, border: `1px solid ${s.border}` }}
    >
      {status.replace("_", " ")}
    </span>
  );
}

// ═══════════ SUMMARY CARDS ═══════════

function SummaryCards({ positions }: { positions: ReconPosition[] }) {
  const paperPositions = positions.filter(p => p.paperQty !== null);
  const alpacaPositions = positions.filter(p => p.alpacaQty !== null);
  const matched = positions.filter(p => p.status === "MATCHED").length;
  const mismatched = positions.filter(p => p.status !== "MATCHED").length;

  const paperNAV = positions.reduce((s, p) => {
    if (p.paperQty === null || p.paperAvgPrice === null) return s;
    return s + p.paperQty * p.paperAvgPrice + (p.paperPnl ?? 0);
  }, 0);

  const alpacaNAV = positions.reduce((s, p) => {
    if (p.alpacaQty === null || p.alpacaAvgPrice === null) return s;
    return s + p.alpacaQty * p.alpacaAvgPrice + (p.alpacaPnl ?? 0);
  }, 0);

  const navDiff = alpacaNAV - paperNAV;
  const lastRecon = new Date(Date.now() - 4 * 60000).toLocaleString("en-US", {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
  });

  const cards = [
    { label: "PAPER POSITIONS", value: `${paperPositions.length}`, color: "text-terminal-text-primary" },
    { label: "ALPACA POSITIONS", value: `${alpacaPositions.length}`, color: "text-[#58a6ff]" },
    { label: "MATCHED", value: `${matched}`, color: "text-terminal-positive" },
    { label: "MISMATCHED", value: `${mismatched}`, color: mismatched > 0 ? "text-terminal-negative" : "text-terminal-positive" },
    { label: "PAPER NAV", value: `$${(paperNAV / 1e6).toFixed(2)}M`, color: "text-terminal-accent" },
    { label: "ALPACA NAV", value: `$${(alpacaNAV / 1e6).toFixed(2)}M`, color: "text-[#58a6ff]" },
    { label: "NAV DIFF", value: `${navDiff >= 0 ? "+" : ""}$${Math.abs(navDiff).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, color: Math.abs(navDiff) < 10000 ? "text-terminal-positive" : "text-terminal-negative" },
    { label: "LAST RECON", value: lastRecon, color: "text-terminal-text-muted" },
  ];

  return (
    <div className="grid grid-cols-8 gap-1">
      {cards.map(c => (
        <div key={c.label} className="bg-terminal-bg rounded border border-terminal-border/50 px-2 py-1.5">
          <div className="text-[9px] text-terminal-text-faint tracking-wider mb-0.5">{c.label}</div>
          <div className={`text-[11px] font-mono font-semibold truncate ${c.color}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ RECONCILIATION TABLE ═══════════

function ReconciliationTable({ positions }: { positions: ReconPosition[] }) {
  const [filter, setFilter] = useState<ReconStatus | "all">("all");

  const filtered = useMemo(() =>
    filter === "all" ? positions : positions.filter(p => p.status === filter),
    [positions, filter]
  );

  const statusCounts = useMemo(() => ({
    all: positions.length,
    MATCHED: positions.filter(p => p.status === "MATCHED").length,
    MISMATCH: positions.filter(p => p.status === "MISMATCH").length,
    PAPER_ONLY: positions.filter(p => p.status === "PAPER_ONLY").length,
    ALPACA_ONLY: positions.filter(p => p.status === "ALPACA_ONLY").length,
  }), [positions]);

  return (
    <div className="text-[10px] h-full flex flex-col">
      {/* Filter tabs */}
      <div className="flex items-center gap-1 mb-1.5 flex-shrink-0">
        {(["all", "MATCHED", "MISMATCH", "PAPER_ONLY", "ALPACA_ONLY"] as const).map(f => {
          const count = statusCounts[f];
          const s = f !== "all" ? STATUS_STYLES[f] : null;
          return (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-0.5 rounded text-[9px] transition-colors ${
                filter === f
                  ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30"
                  : "text-terminal-text-muted hover:text-terminal-text-primary border border-transparent"
              }`}
            >
              {f === "all" ? "ALL" : f.replace("_", " ")}{" "}
              <span className={`font-mono ${filter === f ? "" : ""}`} style={{ color: s?.color }}>({count})</span>
            </button>
          );
        })}
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-terminal-surface z-10">
            <tr className="border-b border-terminal-border">
              {["TICKER", "SECTOR", "PAPER QTY", "ALPACA QTY", "QTY ✓", "PAPER AVG", "ALPACA AVG", "PRICE DIFF", "PAPER P&L", "ALPACA P&L", "P&L DIFF", "STATUS"].map(h => (
                <th key={h} className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filtered.map(pos => {
              const rowBg = pos.status === "MISMATCH"
                ? "hover:bg-red-950/20"
                : pos.status === "PAPER_ONLY" || pos.status === "ALPACA_ONLY"
                ? "hover:bg-yellow-950/20"
                : "hover:bg-white/[0.02]";
              const leftBorder = pos.status === "MISMATCH"
                ? "border-l-2 border-l-red-500/50"
                : pos.status === "PAPER_ONLY"
                ? "border-l-2 border-l-yellow-500/50"
                : pos.status === "ALPACA_ONLY"
                ? "border-l-2 border-l-blue-500/50"
                : "";
              return (
                <tr key={pos.ticker} className={`border-b border-terminal-border/40 transition-colors ${rowBg} ${leftBorder}`}>
                  <td className="py-1.5 px-1.5 font-mono font-bold text-terminal-text-primary">{pos.ticker}</td>
                  <td className="py-1.5 px-1.5 text-terminal-text-faint text-[9px]">{pos.sector}</td>
                  <td className="py-1.5 px-1.5 font-mono">{pos.paperQty !== null ? pos.paperQty.toLocaleString() : <span className="text-terminal-text-faint">—</span>}</td>
                  <td className="py-1.5 px-1.5 font-mono">{pos.alpacaQty !== null ? pos.alpacaQty.toLocaleString() : <span className="text-terminal-text-faint">—</span>}</td>
                  <td className="py-1.5 px-1.5 font-mono text-center">
                    {pos.qtyMatch === null ? (
                      <span className="text-terminal-text-faint">—</span>
                    ) : pos.qtyMatch ? (
                      <span className="text-terminal-positive text-sm">✓</span>
                    ) : (
                      <span className="text-terminal-negative text-sm">✗</span>
                    )}
                  </td>
                  <td className="py-1.5 px-1.5 font-mono">{pos.paperAvgPrice !== null ? `$${pos.paperAvgPrice.toFixed(2)}` : <span className="text-terminal-text-faint">—</span>}</td>
                  <td className="py-1.5 px-1.5 font-mono">{pos.alpacaAvgPrice !== null ? `$${pos.alpacaAvgPrice.toFixed(2)}` : <span className="text-terminal-text-faint">—</span>}</td>
                  <td className={`py-1.5 px-1.5 font-mono ${pos.priceDiff !== null ? (Math.abs(pos.priceDiff) > 0.5 ? "text-terminal-negative" : "text-terminal-text-muted") : ""}`}>
                    {pos.priceDiff !== null ? `${pos.priceDiff >= 0 ? "+" : ""}$${pos.priceDiff.toFixed(2)}` : <span className="text-terminal-text-faint">—</span>}
                  </td>
                  <td className={`py-1.5 px-1.5 font-mono ${pos.paperPnl !== null ? (pos.paperPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative") : ""}`}>
                    {pos.paperPnl !== null ? `${pos.paperPnl >= 0 ? "+" : ""}$${Math.abs(pos.paperPnl).toLocaleString()}` : <span className="text-terminal-text-faint">—</span>}
                  </td>
                  <td className={`py-1.5 px-1.5 font-mono ${pos.alpacaPnl !== null ? (pos.alpacaPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative") : ""}`}>
                    {pos.alpacaPnl !== null ? `${pos.alpacaPnl >= 0 ? "+" : ""}$${Math.abs(pos.alpacaPnl).toLocaleString()}` : <span className="text-terminal-text-faint">—</span>}
                  </td>
                  <td className={`py-1.5 px-1.5 font-mono ${pos.pnlDiff !== null ? (Math.abs(pos.pnlDiff ?? 0) > 500 ? "text-terminal-negative" : "text-terminal-text-muted") : ""}`}>
                    {pos.pnlDiff !== null ? `${pos.pnlDiff >= 0 ? "+" : ""}$${Math.abs(pos.pnlDiff).toLocaleString()}` : <span className="text-terminal-text-faint">—</span>}
                  </td>
                  <td className="py-1.5 px-1.5">
                    <StatusBadge status={pos.status} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ═══════════ DISCREPANCY DETAIL ═══════════

function DiscrepancyDetail({ positions }: { positions: ReconPosition[] }) {
  const discrepant = positions.filter(p => p.status !== "MATCHED");

  return (
    <div className="text-[10px] space-y-2">
      {discrepant.length === 0 ? (
        <div className="text-terminal-positive text-center py-4">✓ No discrepancies detected</div>
      ) : (
        discrepant.map(pos => (
          <div
            key={pos.ticker}
            className="bg-terminal-bg border rounded p-2"
            style={{ borderColor: STATUS_STYLES[pos.status].border }}
          >
            <div className="flex items-center justify-between mb-1.5">
              <span className="font-mono font-bold text-terminal-text-primary">{pos.ticker}</span>
              <StatusBadge status={pos.status} />
            </div>
            <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
              {pos.status === "MISMATCH" && (
                <>
                  <div className="flex justify-between">
                    <span className="text-terminal-text-faint">Paper Qty</span>
                    <span className="font-mono">{pos.paperQty?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-text-faint">Alpaca Qty</span>
                    <span className="font-mono text-terminal-negative">{pos.alpacaQty?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-text-faint">Qty Δ</span>
                    <span className="font-mono text-terminal-negative">
                      {((pos.alpacaQty ?? 0) - (pos.paperQty ?? 0) >= 0 ? "+" : "")}{(pos.alpacaQty ?? 0) - (pos.paperQty ?? 0)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-terminal-text-faint">P&L Δ</span>
                    <span className="font-mono text-terminal-negative">${pos.pnlDiff?.toLocaleString()}</span>
                  </div>
                </>
              )}
              {pos.status === "PAPER_ONLY" && (
                <>
                  <div className="flex justify-between col-span-2">
                    <span className="text-terminal-text-faint">Paper Qty</span>
                    <span className="font-mono text-terminal-warning">{pos.paperQty?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between col-span-2">
                    <span className="text-terminal-text-faint">Paper Avg</span>
                    <span className="font-mono">${pos.paperAvgPrice?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between col-span-2 text-terminal-warning">
                    <span>⚠ Missing in Alpaca — confirm fill</span>
                  </div>
                </>
              )}
              {pos.status === "ALPACA_ONLY" && (
                <>
                  <div className="flex justify-between col-span-2">
                    <span className="text-terminal-text-faint">Alpaca Qty</span>
                    <span className="font-mono text-[#58a6ff]">{pos.alpacaQty?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between col-span-2">
                    <span className="text-terminal-text-faint">Alpaca Avg</span>
                    <span className="font-mono">${pos.alpacaAvgPrice?.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between col-span-2 text-[#58a6ff]">
                    <span>ℹ Alpaca position not in Paper — check order sync</span>
                  </div>
                </>
              )}
            </div>
          </div>
        ))
      )}
    </div>
  );
}

// ═══════════ NAV COMPARISON CHART ═══════════

function NAVComparisonChart({ data }: { data: ReturnType<typeof generateNavHistory> }) {
  const lastPaper = data[data.length - 1]?.paper ?? 0;
  const lastAlpaca = data[data.length - 1]?.alpaca ?? 0;
  const navDiff = lastAlpaca - lastPaper;
  const navDiffPct = ((navDiff / lastPaper) * 100).toFixed(3);

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center gap-4 mb-1.5 flex-shrink-0 text-[10px]">
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-terminal-accent inline-block" />
          <span className="text-terminal-text-faint">Paper</span>
          <span className="font-mono text-terminal-accent">${(lastPaper / 1e6).toFixed(2)}M</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 bg-[#58a6ff] inline-block" />
          <span className="text-terminal-text-faint">Alpaca</span>
          <span className="font-mono text-[#58a6ff]">${(lastAlpaca / 1e6).toFixed(2)}M</span>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <span className="text-terminal-text-faint">NAV Δ:</span>
          <span className={`font-mono font-semibold ${Math.abs(navDiff) < 50000 ? "text-terminal-positive" : "text-terminal-negative"}`}>
            {navDiff >= 0 ? "+" : ""}${Math.abs(navDiff).toLocaleString(undefined, { maximumFractionDigits: 0 })}
            {" "}({navDiff >= 0 ? "+" : ""}{navDiffPct}%)
          </span>
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 5, right: 10, left: 5, bottom: 0 }}>
            <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} interval={4} />
            <YAxis
              tick={{ fill: "#484f58", fontSize: 9 }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v) => `$${(v / 1e6).toFixed(1)}M`}
              domain={["auto", "auto"]}
            />
            <Tooltip
              contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
              formatter={(v: number) => [`$${(v / 1e6).toFixed(3)}M`]}
            />
            <Line type="monotone" dataKey="paper" stroke="#00d4aa" strokeWidth={1.5} dot={false} name="Paper NAV" />
            <Line type="monotone" dataKey="alpaca" stroke="#58a6ff" strokeWidth={1.5} dot={false} name="Alpaca NAV" strokeDasharray="4 2" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

// ═══════════ STATUS LEGEND ═══════════

function StatusLegend() {
  return (
    <div className="flex items-center gap-3 text-[9px] flex-wrap">
      {(Object.entries(STATUS_STYLES) as [ReconStatus, typeof STATUS_STYLES[ReconStatus]][]).map(([status, style]) => (
        <div key={status} className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: style.color }} />
          <span className="text-terminal-text-muted">{status.replace("_", " ")}</span>
        </div>
      ))}
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

const positions = generatePositions();
const navHistory = generateNavHistory(
  positions.reduce((s, p) => s + (p.paperQty ?? 0) * (p.paperAvgPrice ?? 0), 0)
);

export default function ReconciliationPage() {
  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">
      {/* Summary Cards */}
      <div className="flex-shrink-0">
        <SummaryCards positions={positions} />
      </div>

      {/* Main reconciliation table */}
      <div className="flex-1 overflow-hidden">
        <DashboardPanel
          title="BROKER RECONCILIATION — PAPERBROKER vs ALPACABROKER"
          className="h-full"
          noPadding
          headerRight={<StatusLegend />}
        >
          <div className="p-2 h-full overflow-hidden flex flex-col">
            <ReconciliationTable positions={positions} />
          </div>
        </DashboardPanel>
      </div>

      {/* Bottom row: discrepancies + NAV chart */}
      <div className="flex-shrink-0 grid grid-cols-[320px_1fr] gap-1 h-52">
        <DashboardPanel title="DISCREPANCY DETAIL" className="h-full overflow-auto">
          <DiscrepancyDetail positions={positions} />
        </DashboardPanel>
        <DashboardPanel title="30-DAY NAV COMPARISON" className="h-full">
          <NAVComparisonChart data={navHistory} />
        </DashboardPanel>
      </div>
    </div>
  );
}
