import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import { useMemo } from "react";

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
  const { data: reportsApi } = useEngineQuery<{ reports: Array<{ id: string; name: string; type?: string; description: string; last_generated: string | null; status: string }> }>("/monitoring/reports/list", { refetchInterval: 30000 });

  const reports = useMemo(() => {
    if (!reportsApi?.reports?.length) return [];
    return reportsApi.reports.map((r) => ({
      name: r.name,
      type: r.type || "Research",
      desc: r.description,
      lastGen: r.last_generated || "Not generated",
      status: r.status === "ready" ? "ready" as const : "generating" as const,
    }));
  }, [reportsApi]);

  return (
    <div className="h-full p-[2px] overflow-auto" data-testid="reporting">
      <DashboardPanel title="REPORT CENTER" className="h-full">
        <div className="grid grid-cols-2 gap-3">
          {reports.length === 0 && (
            <div className="col-span-2 py-12 text-center text-terminal-text-faint text-xs font-mono">Waiting for report data…</div>
          )}
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
      </DashboardPanel>
    </div>
  );
}
