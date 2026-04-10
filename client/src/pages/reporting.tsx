import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import { useState, useEffect } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// Allocation Rules Types
// ═══════════════════════════════════════════════════════════════════════════

interface AllocationRulesData {
  max_drawdown_kill_switch: number;
  single_name_ig_pct: number;
  single_name_hy_distressed_pct: number;
  div_cashflow_etf_pct: number;
  fi_macro_pct: number;
  event_driven_cvr_pct: number;
  options_notional_pct: number;
  options_ig_pct: number;
  options_hy_pct: number;
  options_distressed_pct: number;
  margin_real_capital_range_low: number;
  margin_real_capital_range_high: number;
  money_market_pct: number;
  drip_rule: boolean;
  alpha_primary_goal: boolean;
  timestamp: string;
}

interface AllocationStatus {
  rules: AllocationRulesData;
  bucket_utilization: Record<string, number>;
  kill_switch: { triggered: boolean; current_drawdown: number };
  beta_corridor: { corridor: string; leverage_multiplier: number; beta: number };
  drip_events: number;
  rule_changes: number;
}

interface ScanStatusData {
  cycle_number: number;
  phase: string;
  total_signals: number;
  runs: Array<{ universe: string; signals_discovered: number; completed: boolean }>;
  completed: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Allocation Rules Section
// ═══════════════════════════════════════════════════════════════════════════

const BUCKET_ROWS: Array<{ key: string; label: string; ruleKey: keyof AllocationRulesData; color: string }> = [
  { key: "IG_EQUITY", label: "IG Equities (30%)", ruleKey: "single_name_ig_pct", color: "text-terminal-blue" },
  { key: "HY_DISTRESSED", label: "HY / Distressed (20%)", ruleKey: "single_name_hy_distressed_pct", color: "text-amber-400" },
  { key: "DIV_CASHFLOW_ETF", label: "DIV/Cashflow ETFs (15%)", ruleKey: "div_cashflow_etf_pct", color: "text-terminal-positive" },
  { key: "FI_MACRO", label: "FI + Macro RV (5%)", ruleKey: "fi_macro_pct", color: "text-teal-400" },
  { key: "EVENT_DRIVEN_CVR", label: "Event-Driven / CVR (5%)", ruleKey: "event_driven_cvr_pct", color: "text-teal-300" },
  { key: "MONEY_MARKET", label: "Money Market (5%)", ruleKey: "money_market_pct", color: "text-terminal-text-muted" },
];

function AllocationRulesSection() {
  const [status, setStatus] = useState<AllocationStatus | null>(null);
  const [scanStatus, setScanStatus] = useState<ScanStatusData | null>(null);
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    const fetchData = () => {
      fetch("/api/allocation/status")
        .then((r) => r.json())
        .then((d) => setStatus(d))
        .catch(() => {});
      fetch("/api/allocation/scan/status")
        .then((r) => r.json())
        .then((d) => setScanStatus(d))
        .catch(() => {});
    };
    fetchData();
    const iv = setInterval(fetchData, 30000);
    return () => clearInterval(iv);
  }, []);

  const rules = status?.rules;
  const utilization = status?.bucket_utilization || {};
  const killSwitch = status?.kill_switch;
  const beta = status?.beta_corridor;

  return (
    <div className="border border-terminal-border rounded mt-3">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between p-3 hover:bg-white/[0.02] transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] font-mono font-bold text-terminal-accent tracking-wider">
            ALLOCATION RULES IN EFFECT
          </span>
          <span className="text-[8px] font-mono text-terminal-text-faint">
            {rules?.timestamp ? new Date(rules.timestamp).toLocaleDateString() : "—"}
          </span>
        </div>
        <svg width="10" height="10" viewBox="0 0 10 10" fill="none" className={`transition-transform ${expanded ? "rotate-180" : ""}`}>
          <path d="M2 4L5 7L8 4" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" className="text-terminal-text-muted" />
        </svg>
      </button>

      {expanded && (
        <div className="px-3 pb-3 space-y-3">
          {/* Bucket allocation table */}
          <div>
            <div className="grid grid-cols-[1fr_80px_80px_60px] gap-2 text-[8px] font-mono text-terminal-text-faint border-b border-terminal-border/30 pb-1 mb-1">
              <span>BUCKET</span>
              <span className="text-right">TARGET</span>
              <span className="text-right">CURRENT</span>
              <span className="text-right">DELTA</span>
            </div>
            {BUCKET_ROWS.map((row) => {
              const target = rules ? (rules[row.ruleKey] as number) : 0;
              const current = utilization[row.key] || 0;
              const delta = current - target;
              return (
                <div key={row.key} className="grid grid-cols-[1fr_80px_80px_60px] gap-2 text-[9px] font-mono py-0.5">
                  <span className={row.color}>{row.label}</span>
                  <span className="text-right text-terminal-text-primary">{(target * 100).toFixed(1)}%</span>
                  <span className="text-right text-terminal-text-primary">{(current * 100).toFixed(2)}%</span>
                  <span className={`text-right ${delta > 0 ? "text-terminal-warning" : "text-terminal-text-faint"}`}>
                    {delta !== 0 ? `${delta > 0 ? "+" : ""}${(delta * 100).toFixed(2)}%` : "—"}
                  </span>
                </div>
              );
            })}
          </div>

          {/* Options notional */}
          <div className="border-t border-terminal-border/30 pt-2">
            <div className="text-[8px] font-mono text-terminal-text-faint mb-1">OPTIONS NOTIONAL (25% TARGET)</div>
            <div className="grid grid-cols-3 gap-2 text-[9px] font-mono">
              <div>
                <span className="text-terminal-purple">IG: </span>
                <span className="text-terminal-text-primary">{rules ? (rules.options_ig_pct * 100).toFixed(0) : 10}%</span>
              </div>
              <div>
                <span className="text-terminal-purple">HY: </span>
                <span className="text-terminal-text-primary">{rules ? (rules.options_hy_pct * 100).toFixed(0) : 10}%</span>
              </div>
              <div>
                <span className="text-terminal-purple">Distressed: </span>
                <span className="text-terminal-text-primary">{rules ? (rules.options_distressed_pct * 100).toFixed(0) : 5}%</span>
              </div>
            </div>
          </div>

          {/* Status row */}
          <div className="border-t border-terminal-border/30 pt-2 grid grid-cols-2 md:grid-cols-4 gap-3">
            {/* Kill switch */}
            <div>
              <div className="text-[8px] font-mono text-terminal-text-faint">KILL SWITCH</div>
              <span className={`text-xs font-mono font-bold ${killSwitch?.triggered ? "text-terminal-negative" : "text-terminal-positive"}`}>
                {killSwitch?.triggered ? "TRIGGERED" : "CLEAR"}
              </span>
            </div>
            {/* Beta corridor */}
            <div>
              <div className="text-[8px] font-mono text-terminal-text-faint">BETA CORRIDOR</div>
              <span className="text-xs font-mono text-terminal-text-primary">
                {beta?.corridor || "NEUTRAL"} ({beta?.leverage_multiplier?.toFixed(1) || "1.0"}x)
              </span>
            </div>
            {/* DRIP */}
            <div>
              <div className="text-[8px] font-mono text-terminal-text-faint">DRIP EVENTS</div>
              <span className="text-xs font-mono text-terminal-text-primary">
                {status?.drip_events || 0} reinvestments
              </span>
            </div>
            {/* Rule changes */}
            <div>
              <div className="text-[8px] font-mono text-terminal-text-faint">RULE CHANGES</div>
              <span className="text-xs font-mono text-terminal-text-primary">
                {status?.rule_changes || 0} today
              </span>
            </div>
          </div>

          {/* Scan cycle summary */}
          {scanStatus && (
            <div className="border-t border-terminal-border/30 pt-2">
              <div className="text-[8px] font-mono text-terminal-text-faint mb-1">UNIVERSE SCAN CYCLES</div>
              <div className="grid grid-cols-3 gap-2 text-[9px] font-mono">
                <div>
                  <span className="text-terminal-text-faint">Cycles: </span>
                  <span className="text-terminal-text-primary">{scanStatus.cycle_number}</span>
                </div>
                <div>
                  <span className="text-terminal-text-faint">Signals: </span>
                  <span className="text-terminal-text-primary">{scanStatus.total_signals}</span>
                </div>
                <div>
                  <span className="text-terminal-text-faint">Phase: </span>
                  <span className="text-terminal-accent">{scanStatus.phase}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

const REPORTS = [
  { name: "Platinum Report", type: "Executive", desc: "Comprehensive portfolio overview with attribution analysis and risk decomposition for C-suite stakeholders.", lastGen: "Apr 01, 2026 — 18:00", status: "ready" },
  { name: "Daily P&L Report", type: "Operations", desc: "Detailed daily profit & loss breakdown by strategy, sector, and individual position.", lastGen: "Apr 01, 2026 — 16:30", status: "ready" },
  { name: "Portfolio Analytics", type: "Research", desc: "Factor exposure analysis, correlation matrices, and regime classification report.", lastGen: "Apr 01, 2026 — 16:00", status: "ready" },
  { name: "Risk Dashboard", type: "Risk", desc: "VaR analysis, stress test results, Greeks exposure, and liquidity risk assessment.", lastGen: "Apr 01, 2026 — 15:00", status: "ready" },
  { name: "Execution Quality", type: "Trading", desc: "Trade execution analysis: slippage, fill rates, market impact, and broker comparison.", lastGen: "Apr 01, 2026 — 14:00", status: "ready" },
  { name: "Monthly Investor", type: "Investor", desc: "Monthly performance letter with NAV history, benchmark comparison, and market outlook.", lastGen: "Mar 31, 2026 — 20:00", status: "ready" },
  { name: "Compliance Report", type: "Compliance", desc: "Regulatory compliance checks, position limits, and concentration risk analysis.", lastGen: "Apr 01, 2026 — 12:00", status: "generating" },
  { name: "ML Model Report", type: "Research", desc: "Model performance metrics, feature importance, and prediction accuracy analysis.", lastGen: "Apr 01, 2026 — 10:00", status: "ready" },
];

const TYPE_COLORS: Record<string, string> = {
  Executive: "text-terminal-accent bg-terminal-accent/10",
  Operations: "text-terminal-blue bg-terminal-blue/10",
  Research: "text-terminal-purple bg-terminal-purple/10",
  Risk: "text-terminal-negative bg-terminal-negative/10",
  Trading: "text-terminal-positive bg-terminal-positive/10",
  Investor: "text-terminal-warning bg-terminal-warning/10",
  Compliance: "text-terminal-cyan bg-terminal-cyan/10",
};

export default function Reporting() {
  const { data: reportsApi } = useEngineQuery<{ reports: Array<{ id: string; name: string; description: string; last_generated: string | null; status: string }> }>("/monitoring/reports/list", { refetchInterval: 30000 });

  const reports = reportsApi?.reports?.length
    ? reportsApi.reports.map((r) => ({
        name: r.name,
        type: r.id === "platinum" ? "Executive" : r.id === "portfolio" ? "Research" : r.id === "daily" ? "Operations" : "Risk",
        desc: r.description,
        lastGen: r.last_generated || "Not generated",
        status: r.status === "ready" ? "ready" : "generating",
      }))
    : REPORTS;

  return (
    <div className="h-full p-[2px] overflow-auto" data-testid="reporting">
      <DashboardPanel title="REPORT CENTER" className="h-full">
        <div className="grid grid-cols-2 gap-3">
          {reports.map((r, i) => (
            <div key={i} className="border border-terminal-border rounded p-3 hover:border-terminal-accent/30 transition-colors">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="text-sm font-semibold text-terminal-text-primary">{r.name}</h3>
                  <span className={`inline-block mt-1 px-1.5 py-0.5 rounded text-[8px] font-medium ${TYPE_COLORS[r.type] || "text-terminal-text-muted"}`}>
                    {r.type}
                  </span>
                </div>
                <div className="flex items-center gap-1">
                  {r.status === "generating" ? (
                    <span className="text-[8px] text-terminal-warning font-mono animate-pulse">GENERATING...</span>
                  ) : (
                    <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive" />
                  )}
                </div>
              </div>
              <p className="text-[9px] text-terminal-text-muted leading-relaxed mb-3">{r.desc}</p>
              <div className="flex items-center justify-between">
                <span className="text-[8px] text-terminal-text-faint font-mono">{r.lastGen}</span>
                <div className="flex gap-2">
                  <button className="px-2 py-1 text-[8px] font-mono font-medium text-terminal-accent border border-terminal-accent/30 rounded hover:bg-terminal-accent/10 transition-colors">
                    PDF
                  </button>
                  <button className="px-2 py-1 text-[8px] font-mono font-medium text-terminal-blue border border-terminal-blue/30 rounded hover:bg-terminal-blue/10 transition-colors">
                    CSV
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Allocation Rules Section — Daily Reports */}
        <AllocationRulesSection />
      </DashboardPanel>
    </div>
  );
}
