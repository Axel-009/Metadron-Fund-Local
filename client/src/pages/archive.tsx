import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";

// ═══════════ DATA GENERATORS ═══════════

function getDates(count: number): string[] {
  const dates: string[] = [];
  const now = new Date();
  for (let i = count - 1; i >= 0; i--) {
    const d = new Date(now);
    d.setDate(d.getDate() - i);
    dates.push(d.toISOString().slice(0, 10));
  }
  return dates;
}

function generateTxLogs(dates: string[]) {
  return dates.map((date) => ({
    date,
    filename: `TX_LOG_${date}.csv`,
    records: Math.floor(1200 + Math.random() * 800),
    size: `${(0.8 + Math.random() * 1.2).toFixed(1)} MB`,
    status: Math.random() > 0.08 ? "Archived" : "Pending",
  }));
}

function generateReconLogs(dates: string[]) {
  return dates.map((date) => {
    const matches = Math.floor(900 + Math.random() * 200);
    const mismatches = Math.floor(Math.random() * 12);
    return {
      date,
      filename: `RECON_${date}.csv`,
      matches,
      mismatches,
      navDelta: `${(Math.random() - 0.5) < 0 ? "-" : "+"}$${(Math.random() * 850).toFixed(2)}`,
      navNeg: (Math.random() - 0.5) < 0,
      status: mismatches === 0 ? "Clean" : mismatches < 5 ? "Archived" : "Review",
    };
  });
}

function generateLearningLogs(dates: string[]) {
  return dates.map((date, i) => {
    const accDelta = (Math.random() - 0.42) * 0.8;
    return {
      date,
      filename: `LEARNING_${date}.json`,
      lessons: Math.floor(18 + Math.random() * 45),
      modelVersion: `v2.${Math.floor(14 + i * 0.1)}.${Math.floor(Math.random() * 9)}`,
      accDelta: +(accDelta).toFixed(3),
      status: Math.random() > 0.1 ? "Archived" : "Pending",
    };
  });
}

// ═══════════ STATUS BADGE ═══════════

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    Archived: "text-terminal-positive bg-terminal-positive/10 border-terminal-positive/30",
    Pending: "text-terminal-warning bg-terminal-warning/10 border-terminal-warning/30",
    Clean: "text-terminal-accent bg-terminal-accent/10 border-terminal-accent/30",
    Review: "text-terminal-negative bg-terminal-negative/10 border-terminal-negative/30",
  };
  const cls = colors[status] ?? "text-terminal-text-muted bg-terminal-bg border-terminal-border/30";
  return (
    <span className={`px-1.5 py-0.5 rounded text-[7px] font-semibold border ${cls}`}>
      {status.toUpperCase()}
    </span>
  );
}

// ═══════════ DOWNLOAD BUTTONS ═══════════

function DownloadButtons() {
  return (
    <div className="flex items-center gap-1">
      <button
        className="px-1.5 py-0.5 text-[7px] rounded border border-terminal-border/40 text-terminal-text-faint hover:text-terminal-accent hover:border-terminal-accent/40 transition-colors font-mono"
        title="Download CSV"
      >
        CSV
      </button>
      <button
        className="px-1.5 py-0.5 text-[7px] rounded border border-terminal-border/40 text-terminal-text-faint hover:text-terminal-accent hover:border-terminal-accent/40 transition-colors font-mono"
        title="Download PDF"
      >
        PDF
      </button>
    </div>
  );
}

// ═══════════ SECTION HEADERS ═══════════

function SectionHeader({ cols }: { cols: string[] }) {
  return (
    <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
      {cols.map((c, i) => (
        <span key={i} className={i === 0 ? "w-[90px]" : i === 1 ? "flex-1" : "w-[80px] text-right"}>{c}</span>
      ))}
    </div>
  );
}

// ═══════════ TRANSACTION LOG TABLE ═══════════

function TxLogTable({ logs }: { logs: ReturnType<typeof generateTxLogs> }) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[90px]">Date</span>
        <span className="flex-1">File Name</span>
        <span className="w-[80px] text-right">Records</span>
        <span className="w-[70px] text-right">Size</span>
        <span className="w-[70px] text-right">Status</span>
        <span className="w-[70px] text-right">Download</span>
      </div>
      <div className="flex-1 overflow-auto">
        {logs.map((log) => (
          <div
            key={log.date}
            className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono"
          >
            <span className="w-[90px] text-terminal-text-muted">{log.date}</span>
            <span className="flex-1 text-terminal-accent font-medium pr-2 truncate">{log.filename}</span>
            <span className="w-[80px] text-right tabular-nums text-terminal-text-primary">{log.records.toLocaleString()}</span>
            <span className="w-[70px] text-right tabular-nums text-terminal-text-muted">{log.size}</span>
            <span className="w-[70px] text-right"><StatusBadge status={log.status} /></span>
            <div className="w-[70px] flex justify-end"><DownloadButtons /></div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ RECONCILIATION TABLE ═══════════

function ReconTable({ logs }: { logs: ReturnType<typeof generateReconLogs> }) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[90px]">Date</span>
        <span className="flex-1">File Name</span>
        <span className="w-[70px] text-right">Matches</span>
        <span className="w-[70px] text-right">Mismatches</span>
        <span className="w-[90px] text-right">NAV Delta</span>
        <span className="w-[70px] text-right">Status</span>
      </div>
      <div className="flex-1 overflow-auto">
        {logs.map((log) => (
          <div
            key={log.date}
            className="flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono"
          >
            <span className="w-[90px] text-terminal-text-muted">{log.date}</span>
            <span className="flex-1 text-terminal-accent font-medium pr-2 truncate">{log.filename}</span>
            <span className="w-[70px] text-right tabular-nums text-terminal-positive">{log.matches.toLocaleString()}</span>
            <span className={`w-[70px] text-right tabular-nums ${log.mismatches > 0 ? "text-terminal-negative" : "text-terminal-text-faint"}`}>
              {log.mismatches}
            </span>
            <span className={`w-[90px] text-right tabular-nums ${log.navNeg ? "text-terminal-negative" : "text-terminal-positive"}`}>
              {log.navDelta}
            </span>
            <span className="w-[70px] text-right"><StatusBadge status={log.status} /></span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ LEARNING FILES TABLE ═══════════

function LearningTable({ logs }: { logs: ReturnType<typeof generateLearningLogs> }) {
  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center px-2 py-1 text-[8px] text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/40 flex-shrink-0">
        <span className="w-[90px]">Date</span>
        <span className="flex-1">File Name</span>
        <span className="w-[70px] text-right">Lessons</span>
        <span className="w-[80px] text-right">Model Ver</span>
        <span className="w-[90px] text-right">Acc Delta</span>
        <span className="w-[70px] text-right">Status</span>
      </div>
      <div className="flex-1 overflow-auto">
        {logs.map((log) => (
          <div
            key={log.date}
            className={`flex items-center px-2 py-1.5 border-b border-terminal-border/10 hover:bg-white/[0.02] text-[9px] font-mono ${
              log.accDelta > 0 ? "border-l-2 border-l-terminal-positive/30" : log.accDelta < -0.2 ? "border-l-2 border-l-terminal-negative/30" : ""
            }`}
          >
            <span className="w-[90px] text-terminal-text-muted">{log.date}</span>
            <span className="flex-1 text-terminal-accent font-medium pr-2 truncate">{log.filename}</span>
            <span className="w-[70px] text-right tabular-nums text-terminal-text-primary">{log.lessons}</span>
            <span className="w-[80px] text-right tabular-nums text-terminal-text-muted">{log.modelVersion}</span>
            <span className={`w-[90px] text-right tabular-nums font-semibold ${log.accDelta > 0 ? "text-terminal-positive" : log.accDelta < 0 ? "text-terminal-negative" : "text-terminal-text-muted"}`}>
              {log.accDelta >= 0 ? "+" : ""}{log.accDelta.toFixed(3)}%
            </span>
            <span className="w-[70px] text-right"><StatusBadge status={log.status} /></span>
          </div>
        ))}
      </div>
    </div>
  );
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

// ═══════════ MAIN ARCHIVE PAGE ═══════════

export default function Archive() {
  const [archiving, setArchiving] = useState(false);
  const [lastArchived, setLastArchived] = useState("Today");

  const dates = useMemo(() => getDates(30), []);
  const txLogs = useMemo(() => generateTxLogs(dates), [dates]);
  const reconLogs = useMemo(() => generateReconLogs(dates), [dates]);
  const learningLogs = useMemo(() => generateLearningLogs(dates), [dates]);

  const handleArchiveNow = () => {
    setArchiving(true);
    setTimeout(() => {
      setArchiving(false);
      setLastArchived(new Date().toLocaleTimeString());
    }, 1500);
  };

  const totalSize = (txLogs.reduce((s, l) => s + parseFloat(l.size), 0) * 3).toFixed(1);

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-hidden" data-testid="archive">
      {/* Summary KPIs */}
      <div className="flex gap-[2px] flex-shrink-0 h-[62px]">
        <ArchiveKpiCard label="Total Archives" value="847" sub="all-time records" />
        <ArchiveKpiCard label="Storage Used" value={`${totalSize} GB`} sub="across all types" />
        <ArchiveKpiCard label="Oldest Record" value="Jan 2, 2024" sub="TX_LOG_2024-01-02.csv" />
        <ArchiveKpiCard label="Last Archive" value={lastArchived} sub="auto-archived" />
      </div>

      {/* Three tables */}
      <div className="flex flex-col flex-1 gap-[2px] overflow-hidden min-h-0">
        {/* TX Logs */}
        <DashboardPanel
          title="TRANSACTION LOG ARCHIVES"
          className="flex-1"
          headerRight={
            <span className="text-[8px] text-terminal-text-faint font-mono">{txLogs.length} files · {txLogs.filter(l => l.status === "Archived").length} archived</span>
          }
          noPadding
        >
          <TxLogTable logs={txLogs} />
        </DashboardPanel>

        {/* Recon Logs */}
        <DashboardPanel
          title="RECONCILIATION ARCHIVES"
          className="flex-1"
          headerRight={
            <span className="text-[8px] text-terminal-text-faint font-mono">{reconLogs.length} files · {reconLogs.filter(l => l.status === "Clean").length} clean</span>
          }
          noPadding
        >
          <ReconTable logs={reconLogs} />
        </DashboardPanel>

        {/* Learning Files */}
        <DashboardPanel
          title="LEARNING FILES ARCHIVE"
          className="flex-1"
          headerRight={
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1 text-[7px] font-mono">
                <span className="w-2 h-2 rounded-sm bg-terminal-positive/50 inline-block" />
                <span className="text-terminal-text-faint">improved</span>
              </span>
              <span className="flex items-center gap-1 text-[7px] font-mono">
                <span className="w-2 h-2 rounded-sm bg-terminal-negative/30 border-l border-terminal-negative/60 inline-block" />
                <span className="text-terminal-text-faint">degraded</span>
              </span>
            </div>
          }
          noPadding
        >
          <LearningTable logs={learningLogs} />
        </DashboardPanel>
      </div>

      {/* Controls row */}
      <div className="flex items-center gap-4 px-3 py-2 border border-terminal-border/40 rounded bg-terminal-surface flex-shrink-0 text-[10px] font-mono">
        <button
          onClick={handleArchiveNow}
          disabled={archiving}
          data-testid="button-archive-now"
          className="px-3 py-1.5 rounded text-[10px] font-semibold tracking-wider bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/40 hover:bg-terminal-accent/25 transition-colors disabled:opacity-50"
        >
          {archiving ? "ARCHIVING..." : "ARCHIVE NOW"}
        </button>
        <div className="w-px h-4 bg-terminal-border/50" />
        <div className="flex items-center gap-1.5">
          <span className="text-terminal-text-faint">Retention Policy:</span>
          <span className="text-terminal-text-primary font-semibold">90 days</span>
        </div>
        <div className="w-px h-4 bg-terminal-border/50" />
        <div className="flex items-center gap-1.5">
          <span className="text-terminal-text-faint">Auto-Archive Schedule:</span>
          <span className="text-terminal-accent">Daily at 00:00 UTC</span>
        </div>
        <div className="w-px h-4 bg-terminal-border/50" />
        <div className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive animate-pulse" />
          <span className="text-terminal-positive text-[8px]">SCHEDULER ACTIVE</span>
        </div>
      </div>
    </div>
  );
}
