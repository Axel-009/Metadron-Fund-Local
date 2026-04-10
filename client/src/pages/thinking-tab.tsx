import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect, useRef, useCallback } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

interface ThinkingEvent {
  type: string;
  ticker?: string;
  signal_type?: string;
  instrument_type?: string;
  confidence?: number;
  alpha_score?: number;
  regime_context?: string;
  bucket?: string;
  universe?: string;
  run_number?: number;
  phase?: string;
  cycle_number?: number;
  positions_count?: number;
  total_signals?: number;
  kill_switch?: boolean;
  message?: string;
  scan_status?: ScanStatus;
  timestamp?: string;
  _emitted_at?: string;
}

interface ScanStatus {
  cycle_number: number;
  phase: string;
  current_run: number;
  current_universe: string;
  elapsed_seconds: number;
  total_signals: number;
  runs: Array<{
    universe: string;
    description: string;
    run_number: number;
    elapsed_seconds: number;
    heartbeat_total: number;
    signals_discovered: number;
    completed: boolean;
  }>;
  started_at: string;
  completed: boolean;
}

// ═══════════════════════════════════════════════════════════════════════════
// Constants
// ═══════════════════════════════════════════════════════════════════════════

const BUCKET_COLORS: Record<string, string> = {
  IG_EQUITY: "text-terminal-blue border-terminal-blue/30 bg-terminal-blue/10",
  HY_DISTRESSED: "text-amber-400 border-amber-400/30 bg-amber-400/10",
  DIV_CASHFLOW_ETF: "text-terminal-positive border-terminal-positive/30 bg-terminal-positive/10",
  FI_MACRO: "text-teal-400 border-teal-400/30 bg-teal-400/10",
  EVENT_DRIVEN_CVR: "text-teal-300 border-teal-300/30 bg-teal-300/10",
  OPTIONS_IG: "text-terminal-purple border-terminal-purple/30 bg-terminal-purple/10",
  OPTIONS_HY: "text-terminal-purple border-terminal-purple/30 bg-terminal-purple/10",
  OPTIONS_DISTRESSED: "text-terminal-negative border-terminal-negative/30 bg-terminal-negative/10",
  MARGIN: "text-terminal-warning border-terminal-warning/30 bg-terminal-warning/10",
  MONEY_MARKET: "text-terminal-text-muted border-terminal-border bg-terminal-surface",
};

const INSTRUMENT_BADGES: Record<string, { label: string; color: string }> = {
  EQUITY: { label: "EQ", color: "text-terminal-blue bg-terminal-blue/10" },
  OPTION: { label: "OPT", color: "text-terminal-purple bg-terminal-purple/10" },
  FUTURE: { label: "FUT", color: "text-terminal-warning bg-terminal-warning/10" },
  ETF: { label: "ETF", color: "text-terminal-positive bg-terminal-positive/10" },
  FIXED_INCOME: { label: "FI", color: "text-teal-400 bg-teal-400/10" },
  DERIVATIVE: { label: "DRV", color: "text-amber-400 bg-amber-400/10" },
};

const PHASE_LABELS: Record<string, { label: string; color: string }> = {
  SCANNING: { label: "SCANNING", color: "text-terminal-accent" },
  AGGREGATING: { label: "AGGREGATING", color: "text-terminal-blue" },
  EXECUTING: { label: "EXECUTING", color: "text-terminal-warning" },
  RISK_CHECK: { label: "RISK CHECK", color: "text-terminal-purple" },
  COOLDOWN: { label: "COOLDOWN", color: "text-terminal-text-muted" },
  IDLE: { label: "IDLE", color: "text-terminal-text-faint" },
  HALTED: { label: "HALTED", color: "text-terminal-negative" },
};

const UNIVERSES = ["SP500", "SP400_MIDCAP", "SP600_SMALLCAP", "ETF_FI"];
const UNIVERSE_LABELS: Record<string, string> = {
  SP500: "S&P 500",
  SP400_MIDCAP: "S&P 400 MIDCAP",
  SP600_SMALLCAP: "S&P 600 SMALLCAP",
  ETF_FI: "ETF + FI",
};

// ═══════════════════════════════════════════════════════════════════════════
// Live Clock
// ═══════════════════════════════════════════════════════════════════════════

function LiveClock() {
  const [time, setTime] = useState(new Date());
  useEffect(() => {
    const iv = setInterval(() => setTime(new Date()), 1000);
    return () => clearInterval(iv);
  }, []);
  return (
    <span className="text-[10px] font-mono text-terminal-text-muted tabular-nums">
      {time.toLocaleTimeString("en-US", { hour12: false, timeZone: "America/New_York" })} ET
    </span>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Signal Card
// ═══════════════════════════════════════════════════════════════════════════

function SignalCard({ event }: { event: ThinkingEvent }) {
  const bucketStyle = BUCKET_COLORS[event.bucket || ""] || BUCKET_COLORS.MONEY_MARKET;
  const badge = INSTRUMENT_BADGES[event.instrument_type || "EQUITY"] || INSTRUMENT_BADGES.EQUITY;

  return (
    <div className="border border-terminal-border/50 rounded p-2 hover:border-terminal-accent/30 transition-colors animate-in fade-in slide-in-from-left-1 duration-300">
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono font-bold text-terminal-text-primary">{event.ticker}</span>
          <span className={`px-1 py-0.5 rounded text-[7px] font-mono font-medium ${badge.color}`}>
            {badge.label}
          </span>
          <span className={`px-1.5 py-0.5 rounded border text-[7px] font-mono ${bucketStyle}`}>
            {(event.bucket || "").replace(/_/g, " ")}
          </span>
        </div>
        <span className="text-[8px] font-mono text-terminal-text-faint">
          {event.universe ? UNIVERSE_LABELS[event.universe] || event.universe : ""}
        </span>
      </div>
      <div className="flex items-center gap-3 text-[9px] font-mono">
        <span className={event.signal_type === "BUY" || event.signal_type === "RV_LONG" ? "text-terminal-positive" : event.signal_type === "SELL" ? "text-terminal-negative" : "text-terminal-text-muted"}>
          {event.signal_type}
        </span>
        <span className="text-terminal-text-faint">CONF</span>
        <span className="text-terminal-text-primary">{((event.confidence || 0) * 100).toFixed(0)}%</span>
        <span className="text-terminal-text-faint">ALPHA</span>
        <span className={`${(event.alpha_score || 0) > 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
          {((event.alpha_score || 0) * 100).toFixed(2)}%
        </span>
        <span className="text-terminal-text-faint">REGIME</span>
        <span className="text-terminal-text-muted">{event.regime_context}</span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

export default function ThinkingTab() {
  const [events, setEvents] = useState<ThinkingEvent[]>([]);
  const [scanStatus, setScanStatus] = useState<ScanStatus | null>(null);
  const [killSwitch, setKillSwitch] = useState(false);
  const [betaCorridor, setBetaCorridor] = useState<{ corridor: string; leverage_multiplier: number; beta: number }>({ corridor: "NEUTRAL", leverage_multiplier: 1.0, beta: 1.0 });
  const [connected, setConnected] = useState(false);
  const signalListRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Fetch initial status
  useEffect(() => {
    fetch("/api/allocation/status")
      .then((r) => r.json())
      .then((data) => {
        if (data.kill_switch) setKillSwitch(data.kill_switch.triggered);
        if (data.beta_corridor) setBetaCorridor(data.beta_corridor);
      })
      .catch(() => {});

    fetch("/api/allocation/scan/status")
      .then((r) => r.json())
      .then((data) => setScanStatus(data))
      .catch(() => {});
  }, []);

  // SSE connection for live thinking stream
  useEffect(() => {
    const es = new EventSource("/api/allocation/scan/thinking");
    eventSourceRef.current = es;

    es.addEventListener("thinking", (e) => {
      try {
        const data: ThinkingEvent = JSON.parse(e.data);

        if (data.type === "signal_discovered") {
          setEvents((prev) => [...prev.slice(-200), data]);
        } else if (data.type === "phase_update") {
          setScanStatus((prev) => ({
            ...(prev || { cycle_number: 0, phase: "IDLE", current_run: 0, current_universe: "", elapsed_seconds: 0, total_signals: 0, runs: [], started_at: "", completed: false }),
            phase: data.phase || prev?.phase || "IDLE",
            current_run: data.run_number || prev?.current_run || 0,
            current_universe: data.universe || prev?.current_universe || "",
            cycle_number: data.cycle_number || prev?.cycle_number || 0,
          }));
          if (data.kill_switch !== undefined) setKillSwitch(data.kill_switch);
        } else if (data.type === "cycle_complete") {
          if (data.kill_switch !== undefined) setKillSwitch(data.kill_switch);
        } else if (data.type === "heartbeat" && data.scan_status) {
          setScanStatus(data.scan_status);
        }

        setConnected(true);
      } catch {
        // skip malformed events
      }
    });

    es.onerror = () => setConnected(false);
    es.onopen = () => setConnected(true);

    return () => {
      es.close();
      eventSourceRef.current = null;
    };
  }, []);

  // Auto-scroll signal list
  useEffect(() => {
    if (signalListRef.current) {
      signalListRef.current.scrollTop = signalListRef.current.scrollHeight;
    }
  }, [events]);

  // Poll for status updates
  useEffect(() => {
    const iv = setInterval(() => {
      fetch("/api/allocation/scan/status")
        .then((r) => r.json())
        .then((data) => {
          if (data.cycle_number !== undefined) setScanStatus(data);
        })
        .catch(() => {});

      fetch("/api/allocation/status")
        .then((r) => r.json())
        .then((data) => {
          if (data.kill_switch) setKillSwitch(data.kill_switch.triggered);
          if (data.beta_corridor) setBetaCorridor(data.beta_corridor);
        })
        .catch(() => {});
    }, 10000);
    return () => clearInterval(iv);
  }, []);

  const phase = scanStatus?.phase || "IDLE";
  const phaseStyle = PHASE_LABELS[phase] || PHASE_LABELS.IDLE;
  const currentRun = scanStatus?.current_run || 0;
  const elapsed = scanStatus?.elapsed_seconds || 0;
  const cycleNum = scanStatus?.cycle_number || 0;
  const signalEvents = events.filter((e) => e.type === "signal_discovered");

  // Progress bar: 150s per run, 4 runs
  const runElapsed = currentRun > 0 ? Math.max(0, elapsed - (currentRun - 1) * 150) : 0;
  const runProgress = Math.min(100, (runElapsed / 150) * 100);

  return (
    <div className="h-full p-[2px] overflow-auto" data-testid="thinking-tab">
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-[2px] h-full">
        {/* Main signal stream */}
        <div className="flex flex-col gap-[2px] min-h-0">
          {/* Header */}
          <DashboardPanel
            title="SYSTEM THINKING"
            headerRight={
              <div className="flex items-center gap-3">
                <LiveClock />
                <div className={`flex items-center gap-1 ${connected ? "text-terminal-positive" : "text-terminal-text-faint"}`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${connected ? "bg-terminal-positive animate-pulse" : "bg-terminal-text-faint"}`} />
                  <span className="text-[8px] font-mono">{connected ? "LIVE" : "DISCONNECTED"}</span>
                </div>
              </div>
            }
          >
            {/* Scan Progress */}
            <div className="mb-3">
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-mono font-bold ${phaseStyle.color}`}>{phaseStyle.label}</span>
                  {currentRun > 0 && phase === "SCANNING" && (
                    <span className="text-[9px] font-mono text-terminal-text-muted">
                      RUN {currentRun}/4 — {UNIVERSE_LABELS[scanStatus?.current_universe || ""] || scanStatus?.current_universe}
                    </span>
                  )}
                </div>
                <span className="text-[9px] font-mono text-terminal-text-faint">
                  CYCLE {cycleNum} | {Math.floor(elapsed)}s elapsed
                </span>
              </div>

              {/* Universe run progress */}
              <div className="flex gap-1 mb-2">
                {UNIVERSES.map((u, i) => {
                  const runData = scanStatus?.runs?.find((r) => r.universe === u);
                  const isActive = i + 1 === currentRun && phase === "SCANNING";
                  const isDone = runData?.completed || false;
                  return (
                    <div key={u} className="flex-1">
                      <div className="text-[7px] font-mono text-terminal-text-faint mb-0.5 text-center">
                        {UNIVERSE_LABELS[u]}
                      </div>
                      <div className="h-1.5 bg-terminal-border/30 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-500 ${
                            isDone ? "bg-terminal-accent" : isActive ? "bg-terminal-accent/60" : "bg-terminal-border/50"
                          }`}
                          style={{ width: isDone ? "100%" : isActive ? `${runProgress}%` : "0%" }}
                        />
                      </div>
                      <div className="text-[7px] font-mono text-center mt-0.5">
                        {isDone ? (
                          <span className="text-terminal-accent">{runData?.signals_discovered || 0} signals</span>
                        ) : isActive ? (
                          <span className="text-terminal-text-muted">{Math.floor(runElapsed)}s / 150s</span>
                        ) : (
                          <span className="text-terminal-text-faint">pending</span>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Overall cycle phase indicator */}
              <div className="flex gap-0.5">
                {(["SCANNING", "AGGREGATING", "EXECUTING", "RISK_CHECK", "COOLDOWN"] as const).map((p) => {
                  const s = PHASE_LABELS[p];
                  const isActive = phase === p;
                  const isPast = (["SCANNING", "AGGREGATING", "EXECUTING", "RISK_CHECK", "COOLDOWN"].indexOf(phase) >
                    ["SCANNING", "AGGREGATING", "EXECUTING", "RISK_CHECK", "COOLDOWN"].indexOf(p));
                  return (
                    <div key={p} className={`flex-1 text-center py-0.5 text-[7px] font-mono rounded-sm transition-colors ${
                      isActive ? `${s.color} bg-white/[0.05]` : isPast ? "text-terminal-accent/50" : "text-terminal-text-faint"
                    }`}>
                      {s.label}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Live signal stream */}
            <div ref={signalListRef} className="space-y-1 max-h-[calc(100vh-340px)] overflow-y-auto pr-1">
              {signalEvents.length === 0 ? (
                <div className="text-center py-8 text-terminal-text-faint text-[10px] font-mono">
                  Awaiting scan signals...
                </div>
              ) : (
                signalEvents.map((e, i) => <SignalCard key={i} event={e} />)
              )}
            </div>
          </DashboardPanel>
        </div>

        {/* Side panel — monitors */}
        <div className="flex flex-col gap-[2px] min-h-0">
          {/* Kill Switch */}
          <DashboardPanel title="KILL SWITCH">
            <div className={`flex items-center justify-center py-3 rounded ${
              killSwitch
                ? "bg-terminal-negative/10 border border-terminal-negative/40"
                : "bg-terminal-positive/5 border border-terminal-positive/20"
            }`}>
              <div className="text-center">
                <div className={`text-lg font-mono font-bold ${
                  killSwitch ? "text-terminal-negative animate-pulse" : "text-terminal-positive"
                }`}>
                  {killSwitch ? "TRIGGERED" : "CLEAR"}
                </div>
                <div className="text-[8px] font-mono text-terminal-text-faint mt-0.5">
                  20% max drawdown threshold
                </div>
              </div>
            </div>
          </DashboardPanel>

          {/* Beta Corridor */}
          <DashboardPanel title="BETA CORRIDOR">
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">CORRIDOR</span>
                <span className={`text-sm font-mono font-bold ${
                  betaCorridor.corridor === "HIGH" ? "text-terminal-negative" :
                  betaCorridor.corridor === "LOW" ? "text-terminal-positive" : "text-terminal-accent"
                }`}>
                  {betaCorridor.corridor}
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">LEVERAGE</span>
                <span className="text-sm font-mono font-bold text-terminal-text-primary">
                  {betaCorridor.leverage_multiplier.toFixed(1)}x
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">BETA</span>
                <span className="text-sm font-mono text-terminal-text-primary">
                  {betaCorridor.beta.toFixed(2)}
                </span>
              </div>
              {/* Visual corridor bar */}
              <div className="mt-2">
                <div className="flex justify-between text-[7px] font-mono text-terminal-text-faint mb-0.5">
                  <span>LOW 0.7</span>
                  <span>NEUTRAL</span>
                  <span>HIGH 1.3</span>
                </div>
                <div className="h-2 bg-terminal-border/30 rounded-full relative">
                  <div className="absolute left-[23%] top-0 bottom-0 w-px bg-terminal-text-faint/30" />
                  <div className="absolute right-[23%] top-0 bottom-0 w-px bg-terminal-text-faint/30" />
                  <div
                    className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-terminal-accent border border-terminal-accent/50"
                    style={{ left: `${Math.min(100, Math.max(0, ((betaCorridor.beta - 0.3) / 1.4) * 100))}%` }}
                  />
                </div>
              </div>
            </div>
          </DashboardPanel>

          {/* Scan Summary */}
          <DashboardPanel title="SCAN SUMMARY">
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">TOTAL SIGNALS</span>
                <span className="text-xs font-mono text-terminal-text-primary">{scanStatus?.total_signals || signalEvents.length}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">CYCLES</span>
                <span className="text-xs font-mono text-terminal-text-primary">{cycleNum}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">RUNS DONE</span>
                <span className="text-xs font-mono text-terminal-text-primary">
                  {scanStatus?.runs?.filter((r) => r.completed).length || 0} / 4
                </span>
              </div>
              {/* Per-run signal counts */}
              {scanStatus?.runs && scanStatus.runs.length > 0 && (
                <div className="border-t border-terminal-border/30 pt-1.5 mt-1.5 space-y-1">
                  {scanStatus.runs.map((r) => (
                    <div key={r.universe} className="flex items-center justify-between text-[8px] font-mono">
                      <span className="text-terminal-text-faint">{UNIVERSE_LABELS[r.universe] || r.universe}</span>
                      <span className={r.completed ? "text-terminal-accent" : "text-terminal-text-muted"}>
                        {r.signals_discovered} signals {r.completed ? "" : "(scanning...)"}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
