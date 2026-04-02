import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect, useRef } from "react";

const ENGINES = [
  { id: "L1", name: "Data Ingestion", status: "online", latency: 1.2, cpu: 34, memory: 42, errors: 0 },
  { id: "L2", name: "Signal Gen", status: "online", latency: 3.4, cpu: 67, memory: 58, errors: 0 },
  { id: "L3", name: "Risk Engine", status: "online", latency: 2.1, cpu: 45, memory: 51, errors: 1 },
  { id: "L4", name: "Execution", status: "online", latency: 0.8, cpu: 28, memory: 35, errors: 0 },
  { id: "L5", name: "ML Pipeline", status: "degraded", latency: 12.4, cpu: 89, memory: 82, errors: 3 },
  { id: "L6", name: "Backtest", status: "online", latency: 45.2, cpu: 72, memory: 68, errors: 0 },
  { id: "L7", name: "Reporting", status: "online", latency: 8.6, cpu: 22, memory: 31, errors: 0 },
];

const VPS_METRICS = [
  { name: "US-EAST-1", cpu: 45, mem: 62, disk: 34, net: "2.4 Gbps", status: "healthy" },
  { name: "US-WEST-2", cpu: 38, mem: 55, disk: 28, net: "1.8 Gbps", status: "healthy" },
  { name: "EU-WEST-1", cpu: 72, mem: 78, disk: 45, net: "3.1 Gbps", status: "warning" },
  { name: "AP-EAST-1", cpu: 31, mem: 48, disk: 22, net: "1.2 Gbps", status: "healthy" },
];

const LOG_MESSAGES = [
  "[14:32:18.442] INFO  signal_gen: Generated 142 signals for AAPL basket",
  "[14:32:17.891] INFO  execution: Order 0x8f2a filled MSFT 200@420.12",
  "[14:32:17.234] WARN  ml_pipe: Model inference latency exceeded threshold (12.4ms > 10ms)",
  "[14:32:16.987] INFO  risk_eng: VaR recalculated: 1.22% (within limits)",
  "[14:32:16.512] INFO  data_ing: Tick data received: 1,247 symbols",
  "[14:32:15.890] ERROR ml_pipe: GPU memory allocation failed, retrying...",
  "[14:32:15.445] INFO  execution: Order 0x8f2b submitted NVDA BUY 100",
  "[14:32:14.901] INFO  backtest: Completed 252-day backtest in 45.2ms",
  "[14:32:14.234] WARN  risk_eng: Correlation matrix update detected regime change",
  "[14:32:13.567] INFO  reporting: Daily P&L snapshot exported",
  "[14:32:12.890] INFO  signal_gen: ML factor loading updated (α=0.034)",
  "[14:32:12.123] ERROR ml_pipe: TensorRT optimization failed for model v2.4.1",
  "[14:32:11.456] INFO  data_ing: WebSocket reconnected to exchange feed",
  "[14:32:10.789] INFO  execution: Slippage report: avg 0.02bps (target <0.05bps)",
];

export default function TechDashboard() {
  const [engines, setEngines] = useState(ENGINES);
  const [logs, setLogs] = useState(LOG_MESSAGES);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const iv = setInterval(() => {
      setEngines((prev) =>
        prev.map((e) => ({
          ...e,
          latency: +(e.latency + (Math.random() - 0.5) * 0.5).toFixed(1),
          cpu: Math.min(100, Math.max(10, e.cpu + Math.floor(Math.random() * 6 - 3))),
          memory: Math.min(100, Math.max(20, e.memory + Math.floor(Math.random() * 4 - 2))),
        }))
      );
    }, 3000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    const iv = setInterval(() => {
      const levels = ["INFO", "INFO", "INFO", "WARN", "ERROR"];
      const modules = ["signal_gen", "execution", "ml_pipe", "risk_eng", "data_ing"];
      const msgs = [
        "Tick processed in 0.8ms",
        "Order queue depth: 14",
        "Model checkpoint saved",
        "Feed heartbeat OK",
        "Cache hit rate: 94.2%",
      ];
      const lvl = levels[Math.floor(Math.random() * levels.length)];
      const mod = modules[Math.floor(Math.random() * modules.length)];
      const msg = msgs[Math.floor(Math.random() * msgs.length)];
      const now = new Date();
      const ts = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}.${now.getMilliseconds().toString().padStart(3, "0")}`;
      setLogs((prev) => [...prev.slice(-30), `[${ts}] ${lvl.padEnd(5)} ${mod}: ${msg}`]);
    }, 2000);
    return () => clearInterval(iv);
  }, []);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <div className="h-full grid grid-cols-[1fr_1fr] grid-rows-[auto_1fr_1fr] gap-[2px] p-[2px] overflow-auto" data-testid="tech-dashboard">
      {/* Engine Status Grid */}
      <DashboardPanel title="ACTIVE ENGINES" className="col-span-2">
        <div className="flex gap-2 overflow-x-auto pb-1">
          {engines.map((e) => (
            <div key={e.id} className="flex-shrink-0 border border-terminal-border rounded p-2.5 min-w-[130px]">
              <div className="flex items-center gap-2 mb-1.5">
                <span className={`w-2 h-2 rounded-full ${
                  e.status === "online" ? "bg-terminal-positive animate-pulse-live" :
                  e.status === "degraded" ? "bg-terminal-warning animate-pulse-live" :
                  "bg-terminal-negative"
                }`} />
                <span className="text-[10px] text-terminal-accent font-mono font-bold">{e.id}</span>
              </div>
              <div className="text-[9px] text-terminal-text-muted mb-2">{e.name}</div>
              <div className="space-y-1">
                <div className="flex justify-between text-[8px]">
                  <span className="text-terminal-text-faint">CPU</span>
                  <span className="text-terminal-text-primary font-mono tabular-nums">{e.cpu}%</span>
                </div>
                <div className="h-1 bg-terminal-surface-2 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full transition-all ${e.cpu > 80 ? "bg-terminal-negative" : e.cpu > 60 ? "bg-terminal-warning" : "bg-terminal-accent"}`}
                    style={{ width: `${e.cpu}%` }} />
                </div>
                <div className="flex justify-between text-[8px]">
                  <span className="text-terminal-text-faint">MEM</span>
                  <span className="text-terminal-text-primary font-mono tabular-nums">{e.memory}%</span>
                </div>
                <div className="h-1 bg-terminal-surface-2 rounded-full overflow-hidden">
                  <div className={`h-full rounded-full transition-all ${e.memory > 80 ? "bg-terminal-negative" : e.memory > 60 ? "bg-terminal-warning" : "bg-terminal-blue"}`}
                    style={{ width: `${e.memory}%` }} />
                </div>
                <div className="flex justify-between text-[8px] mt-1">
                  <span className="text-terminal-text-faint">Lat</span>
                  <span className="text-terminal-text-primary font-mono tabular-nums">{e.latency}ms</span>
                </div>
                {e.errors > 0 && (
                  <div className="text-[8px] text-terminal-negative mt-0.5">⚠ {e.errors} errors</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </DashboardPanel>

      {/* VPS Health */}
      <DashboardPanel title="VPS HEALTH" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Region</th>
                <th className="text-right px-2 py-1.5 font-medium">CPU</th>
                <th className="text-right px-2 py-1.5 font-medium">Mem</th>
                <th className="text-right px-2 py-1.5 font-medium">Disk</th>
                <th className="text-right px-2 py-1.5 font-medium">Net</th>
              </tr>
            </thead>
            <tbody>
              {VPS_METRICS.map((v, i) => (
                <tr key={i} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 flex items-center gap-1.5">
                    <span className={`w-1.5 h-1.5 rounded-full ${v.status === "healthy" ? "bg-terminal-positive" : "bg-terminal-warning"}`} />
                    <span className="text-terminal-text-primary">{v.name}</span>
                  </td>
                  <td className={`px-2 py-1.5 text-right tabular-nums ${v.cpu > 70 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>{v.cpu}%</td>
                  <td className={`px-2 py-1.5 text-right tabular-nums ${v.mem > 70 ? "text-terminal-warning" : "text-terminal-text-muted"}`}>{v.mem}%</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{v.disk}%</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted">{v.net}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* Latency Table */}
      <DashboardPanel title="ENGINE LATENCY" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[9px] font-mono">
            <thead>
              <tr className="text-terminal-text-faint uppercase tracking-wider border-b border-terminal-border/50">
                <th className="text-left px-2 py-1.5 font-medium">Engine</th>
                <th className="text-right px-2 py-1.5 font-medium">p50</th>
                <th className="text-right px-2 py-1.5 font-medium">p95</th>
                <th className="text-right px-2 py-1.5 font-medium">p99</th>
              </tr>
            </thead>
            <tbody>
              {engines.map((e) => (
                <tr key={e.id} className="border-b border-terminal-border/20">
                  <td className="px-2 py-1.5 text-terminal-accent">{e.id} {e.name}</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{e.latency.toFixed(1)}ms</td>
                  <td className="px-2 py-1.5 text-right text-terminal-text-muted tabular-nums">{(e.latency * 2.1).toFixed(1)}ms</td>
                  <td className="px-2 py-1.5 text-right text-terminal-warning tabular-nums">{(e.latency * 3.5).toFixed(1)}ms</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardPanel>

      {/* Live Error Log */}
      <DashboardPanel title="LIVE LOG" className="col-span-2" noPadding>
        <div ref={logRef} className="overflow-auto h-full p-2 bg-terminal-bg font-mono text-[9px] leading-relaxed">
          {logs.map((line, i) => {
            const isError = line.includes("ERROR");
            const isWarn = line.includes("WARN");
            return (
              <div key={i} className={`py-0.5 ${isError ? "text-terminal-negative" : isWarn ? "text-terminal-warning" : "text-terminal-text-muted"}`}>
                {line}
              </div>
            );
          })}
        </div>
      </DashboardPanel>
    </div>
  );
}
