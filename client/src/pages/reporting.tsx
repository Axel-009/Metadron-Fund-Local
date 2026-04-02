import { DashboardPanel } from "@/components/dashboard-panel";

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
  return (
    <div className="h-full p-[2px] overflow-auto" data-testid="reporting">
      <DashboardPanel title="REPORT CENTER" className="h-full">
        <div className="grid grid-cols-2 gap-3">
          {REPORTS.map((r, i) => (
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
      </DashboardPanel>
    </div>
  );
}
