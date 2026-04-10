import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ═══════════ TYPES ═══════════

interface ArchiveDate {
  date: string;
  files_count: number;
  files: string[];
}

interface DailySummary {
  date: string;
  performance: {
    daily_pnl: number;
    realized_pnl: number;
    unrealized_pnl: number;
    sharpe_30d: number;
    win_rate: number;
    profit_factor: number;
    total_trades: number;
  };
  nav: {
    portfolio_nav: number;
    paper_nav: number;
    alpaca_nav: number;
    nav_delta: number;
    cash: number;
  };
  risk: {
    var_95: number;
    var_99: number;
    cvar_95: number;
    max_drawdown: number;
    beta_to_spy: number;
  };
  outlook: {
    regime: string;
    ml_consensus: string;
    vix: number;
    yield_2s10s: number;
  };
  pricing: {
    benchmarks: Record<string, number>;
  };
  generated_at: string;
}

interface MonthlySummary {
  year: number;
  month: number;
  days_archived: number;
  total_files: number;
  total_trades: number;
  total_errors: number;
  total_patterns: number;
}

// ═══════════ KPI CARD ═══════════

function ArchiveKpiCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="flex-1 border border-terminal-border/40 rounded bg-terminal-surface p-2">
      <div className="text-[8px] text-terminal-text-faint uppercase tracking-wider">{label}</div>
      <div className="text-sm font-bold text-terminal-text-primary font-mono tabular-nums mt-0.5">{value}</div>
      {sub && <div className="text-[8px] text-terminal-text-faint mt-0.5">{sub}</div>}
    </div>
  );
}

// ═══════════ DATE SELECTOR ═══════════

function DateSelector({
  dates,
  selectedDate,
  onSelect,
  selectedYear,
  selectedMonth,
  onYearChange,
  onMonthChange,
}: {
  dates: ArchiveDate[];
  selectedDate: string;
  onSelect: (d: string) => void;
  selectedYear: number;
  selectedMonth: number;
  onYearChange: (y: number) => void;
  onMonthChange: (m: number) => void;
}) {
  const now = new Date();
  const years = [now.getFullYear() - 1, now.getFullYear()];
  const months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
  const monthNames = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"];

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center gap-2">
        <div className="flex items-center gap-1">
          {years.map((y) => (
            <button
              key={y}
              onClick={() => onYearChange(y)}
              className={`px-2 py-0.5 text-[9px] font-mono rounded ${
                selectedYear === y
                  ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40"
                  : "text-terminal-text-muted border border-terminal-border/30 hover:text-terminal-text-primary"
              }`}
            >
              {y}
            </button>
          ))}
        </div>
        <div className="w-px h-4 bg-terminal-border/50" />
        <div className="flex items-center gap-0.5 flex-wrap">
          {months.map((m) => (
            <button
              key={m}
              onClick={() => onMonthChange(m)}
              className={`px-1.5 py-0.5 text-[8px] font-mono rounded ${
                selectedMonth === m
                  ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40"
                  : "text-terminal-text-muted border border-terminal-border/20 hover:text-terminal-text-primary"
              }`}
            >
              {monthNames[m - 1]}
            </button>
          ))}
        </div>
      </div>
      <div className="flex items-center gap-1 flex-wrap max-h-[40px] overflow-auto">
        {dates.length === 0 && (
          <span className="text-[8px] text-terminal-text-faint font-mono">No archive data for {selectedYear}-{String(selectedMonth).padStart(2, "0")}</span>
        )}
        {dates.map((d) => (
          <button
            key={d.date}
            onClick={() => onSelect(d.date)}
            className={`px-1.5 py-0.5 text-[8px] font-mono rounded ${
              selectedDate === d.date
                ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40"
                : "text-terminal-text-muted border border-terminal-border/20 hover:text-terminal-text-primary"
            }`}
            title={`${d.files_count} files`}
          >
            {d.date.slice(8)}
          </button>
        ))}
      </div>
    </div>
  );
}

// ═══════════ DAILY SUMMARY CARD ═══════════

function DailySummaryCard({ summary }: { summary: DailySummary | null }) {
  if (!summary) {
    return (
      <div className="text-[9px] text-terminal-text-faint font-mono p-3">
        Select a date to view the daily summary report.
      </div>
    );
  }

  const { performance: p, nav, risk, outlook, pricing } = summary;

  return (
    <div className="grid grid-cols-5 gap-2 p-2 text-[9px] font-mono">
      {/* Performance */}
      <div className="flex flex-col gap-1">
        <div className="text-[8px] text-terminal-accent tracking-wider font-semibold">PERFORMANCE</div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Daily P&L</span>
          <span className={p.daily_pnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>
            ${p.daily_pnl.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Sharpe 30d</span>
          <span className="text-terminal-text-primary">{p.sharpe_30d.toFixed(2)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Win Rate</span>
          <span className="text-terminal-text-primary">{(p.win_rate * 100).toFixed(1)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Trades</span>
          <span className="text-terminal-text-primary">{p.total_trades}</span>
        </div>
      </div>

      {/* NAV */}
      <div className="flex flex-col gap-1">
        <div className="text-[8px] text-terminal-accent tracking-wider font-semibold">NAV</div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Portfolio</span>
          <span className="text-terminal-text-primary">${nav.portfolio_nav.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Paper</span>
          <span className="text-terminal-text-muted">${nav.paper_nav.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Delta</span>
          <span className={nav.nav_delta >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>
            ${nav.nav_delta.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>

      {/* Pricing */}
      <div className="flex flex-col gap-1">
        <div className="text-[8px] text-terminal-accent tracking-wider font-semibold">BENCHMARKS</div>
        {Object.entries(pricing?.benchmarks || {}).map(([ticker, price]) => (
          <div key={ticker} className="flex justify-between">
            <span className="text-terminal-text-faint">{ticker}</span>
            <span className="text-terminal-text-primary">${(price as number).toFixed(2)}</span>
          </div>
        ))}
        {Object.keys(pricing?.benchmarks || {}).length === 0 && (
          <span className="text-terminal-text-faint">No data</span>
        )}
      </div>

      {/* Risk */}
      <div className="flex flex-col gap-1">
        <div className="text-[8px] text-terminal-accent tracking-wider font-semibold">RISK</div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">VaR 95%</span>
          <span className="text-terminal-text-primary">{(risk.var_95 * 100).toFixed(2)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">CVaR 95%</span>
          <span className="text-terminal-text-primary">{(risk.cvar_95 * 100).toFixed(2)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Max DD</span>
          <span className="text-terminal-negative">{(risk.max_drawdown * 100).toFixed(2)}%</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Beta SPY</span>
          <span className="text-terminal-text-primary">{risk.beta_to_spy.toFixed(2)}</span>
        </div>
      </div>

      {/* Outlook */}
      <div className="flex flex-col gap-1">
        <div className="text-[8px] text-terminal-accent tracking-wider font-semibold">OUTLOOK</div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">Regime</span>
          <span className={
            outlook.regime === "BULL" ? "text-terminal-positive" :
            outlook.regime === "BEAR" ? "text-terminal-negative" :
            "text-terminal-warning"
          }>{outlook.regime}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">ML</span>
          <span className="text-terminal-text-primary">{outlook.ml_consensus}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">VIX</span>
          <span className="text-terminal-text-primary">{outlook.vix.toFixed(1)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-terminal-text-faint">2s10s</span>
          <span className="text-terminal-text-primary">{outlook.yield_2s10s.toFixed(2)}</span>
        </div>
      </div>
    </div>
  );
}

// ═══════════ FILE BROWSER ═══════════

function FileBrowser({ files, selectedDate }: { files: Record<string, unknown>; selectedDate: string }) {
  const [expandedFile, setExpandedFile] = useState<string | null>(null);

  const fileLabels: Record<string, string> = {
    "broker_trades.json": "Broker Trades",
    "tech_system.json": "Tech/System Logs",
    "errors.json": "Error Logs",
    "transactions.json": "Transaction Log",
    "patterns.json": "Pattern Recognition",
    "daily_summary.json": "Daily Summary",
  };

  const fileEntries = Object.entries(files);

  if (fileEntries.length === 0) {
    return (
      <div className="text-[9px] text-terminal-text-faint font-mono p-3">
        No archive data for {selectedDate}
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      {fileEntries.map(([name, content]) => (
        <div key={name} className="border-b border-terminal-border/10">
          <button
            onClick={() => setExpandedFile(expandedFile === name ? null : name)}
            className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-white/[0.02] text-[9px] font-mono"
          >
            <span className="text-terminal-accent">{expandedFile === name ? "[-]" : "[+]"}</span>
            <span className="text-terminal-text-primary font-medium">{fileLabels[name] || name}</span>
            <span className="text-terminal-text-faint ml-auto">{name}</span>
          </button>
          {expandedFile === name && (
            <div className="px-3 py-2 bg-terminal-bg/50 text-[8px] font-mono text-terminal-text-muted max-h-[200px] overflow-auto">
              <pre className="whitespace-pre-wrap">{JSON.stringify(content, null, 2)}</pre>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

// ═══════════ MONTHLY SUMMARY BAR ═══════════

function MonthlySummaryBar({ summary }: { summary: MonthlySummary | null }) {
  if (!summary) return null;

  return (
    <div className="flex items-center gap-4 text-[9px] font-mono">
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">Days Archived:</span>
        <span className="text-terminal-text-primary font-semibold">{summary.days_archived}</span>
      </div>
      <div className="w-px h-3 bg-terminal-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">Total Files:</span>
        <span className="text-terminal-text-primary font-semibold">{summary.total_files}</span>
      </div>
      <div className="w-px h-3 bg-terminal-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">Total Trades:</span>
        <span className="text-terminal-text-primary font-semibold">{summary.total_trades.toLocaleString()}</span>
      </div>
      <div className="w-px h-3 bg-terminal-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">Errors:</span>
        <span className={summary.total_errors > 0 ? "text-terminal-negative font-semibold" : "text-terminal-text-primary font-semibold"}>
          {summary.total_errors}
        </span>
      </div>
      <div className="w-px h-3 bg-terminal-border/50" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">Patterns Found:</span>
        <span className="text-terminal-text-primary font-semibold">{summary.total_patterns}</span>
      </div>
    </div>
  );
}

// ═══════════ MAIN ARCHIVE PAGE ═══════════

export default function Archive() {
  const now = new Date();
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth() + 1);
  const [selectedDate, setSelectedDate] = useState("");
  const [archiving, setArchiving] = useState(false);
  const [archiveStatus, setArchiveStatus] = useState("");

  // ─── API Queries ───────────────────────────────────
  const { data: datesData } = useEngineQuery<{ dates: ArchiveDate[]; total: number }>(
    `/archive/dates?year=${selectedYear}&month=${selectedMonth}`,
    { refetchInterval: 30000 }
  );

  const { data: dailyData } = useEngineQuery<{ date: string; files: Record<string, unknown> }>(
    `/archive/daily/${selectedDate}`,
    { enabled: !!selectedDate, refetchInterval: false }
  );

  const { data: summaryData } = useEngineQuery<{ date: string; summary: DailySummary }>(
    `/archive/daily-summary/${selectedDate}`,
    { enabled: !!selectedDate, refetchInterval: false }
  );

  const { data: monthlyData } = useEngineQuery<MonthlySummary>(
    `/archive/monthly-summary?year=${selectedYear}&month=${selectedMonth}`,
    { refetchInterval: 60000 }
  );

  const dates = useMemo(() => datesData?.dates || [], [datesData]);
  const dailySummary = summaryData?.summary || null;
  const files = dailyData?.files || {};
  const monthly = monthlyData || null;

  const handleArchiveNow = async () => {
    setArchiving(true);
    setArchiveStatus("");
    try {
      const res = await fetch("/api/engine/archive/trigger", { method: "POST" });
      const data = await res.json();
      if (data.error) {
        setArchiveStatus(`Error: ${data.error}`);
      } else {
        setArchiveStatus(`Archived ${data.files_count || 0} files for ${data.date || "today"}`);
      }
    } catch {
      setArchiveStatus("Archive request failed");
    }
    setArchiving(false);
  };

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-hidden" data-testid="archive">
      {/* Date/Month Selector */}
      <DashboardPanel
        title="ARCHIVE NAVIGATOR"
        className="flex-shrink-0"
        headerRight={
          <span className="text-[8px] text-terminal-text-faint font-mono">
            {dates.length} dates available
          </span>
        }
      >
        <DateSelector
          dates={dates}
          selectedDate={selectedDate}
          onSelect={setSelectedDate}
          selectedYear={selectedYear}
          selectedMonth={selectedMonth}
          onYearChange={setSelectedYear}
          onMonthChange={setSelectedMonth}
        />
      </DashboardPanel>

      {/* Daily Summary Card */}
      <DashboardPanel
        title={`DAILY SUMMARY${selectedDate ? ` — ${selectedDate}` : ""}`}
        className="flex-shrink-0"
        headerRight={
          dailySummary?.generated_at ? (
            <span className="text-[8px] text-terminal-text-faint font-mono">
              Generated: {new Date(dailySummary.generated_at).toLocaleString()}
            </span>
          ) : undefined
        }
        noPadding
      >
        <DailySummaryCard summary={dailySummary} />
      </DashboardPanel>

      {/* File Browser */}
      <DashboardPanel
        title={`FILE BROWSER${selectedDate ? ` — ${selectedDate}` : ""}`}
        className="flex-1 min-h-0"
        headerRight={
          <span className="text-[8px] text-terminal-text-faint font-mono">
            {Object.keys(files).length} files
          </span>
        }
        noPadding
      >
        <FileBrowser files={files} selectedDate={selectedDate} />
      </DashboardPanel>

      {/* Monthly Summary Bar + Controls */}
      <div className="flex items-center gap-4 px-3 py-2 border border-terminal-border/40 rounded bg-terminal-surface flex-shrink-0 text-[10px] font-mono">
        <button
          onClick={handleArchiveNow}
          disabled={archiving}
          data-testid="button-archive-now"
          className="px-3 py-1.5 rounded text-[10px] font-semibold tracking-wider bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40 hover:bg-terminal-accent/25 transition-colors disabled:opacity-50"
        >
          {archiving ? "ARCHIVING..." : "ARCHIVE NOW"}
        </button>

        {archiveStatus && (
          <>
            <div className="w-px h-4 bg-terminal-border/50" />
            <span className="text-[8px] text-terminal-accent">{archiveStatus}</span>
          </>
        )}

        <div className="w-px h-4 bg-terminal-border/50" />
        <MonthlySummaryBar summary={monthly} />

        <div className="ml-auto flex items-center gap-3">
          <div className="flex items-center gap-1.5">
            <span className="text-terminal-text-faint">Retention:</span>
            <span className="text-terminal-text-primary font-semibold">90 days</span>
          </div>
          <div className="w-px h-4 bg-terminal-border/50" />
          <div className="flex items-center gap-1.5">
            <span className="text-terminal-text-faint">Schedule:</span>
            <span className="text-terminal-accent">Daily 00:00 UTC</span>
          </div>
          <div className="w-px h-4 bg-terminal-border/50" />
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
            <span className="text-terminal-positive text-[8px]">SCHEDULER ACTIVE</span>
          </div>
        </div>
      </div>
    </div>
  );
}
