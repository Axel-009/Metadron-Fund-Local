import { useState } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

interface SecurityStatus {
  healthy: boolean;
  phase_chain: { broken: boolean; break_phase: string; chain_length: number };
  broker_lock: { frozen: boolean; discrepancies: number; last_recon: string };
  ledger_entries: number;
  circuit_breaker: { locked: boolean; trips: number; current_rate: number; threshold: number };
  heartbeat: { all_healthy: boolean; missing: string[]; services: Record<string, number> };
  prompt_guard: { rejections: number; max_tokens: number };
  token_meter: {
    daily_used: number; daily_cap: number; daily_remaining: number; daily_pct: number;
    lockdown_active: boolean; hourly_baseline: number;
    models: Record<string, { current_hour_tokens: number; current_hour_requests: number; current_hour_anomaly: string }>;
  };
  timestamp: string;
}

interface LedgerData {
  entries: Array<{ seq: number; type: string; data: Record<string, unknown>; timestamp: string; signature: string }>;
  chain_integrity: { intact: boolean; total: number };
  total_entries: number;
}

function StatusBadge({ ok, label }: { ok: boolean; label: string }) {
  return (
    <span className={`text-[9px] px-2 py-0.5 rounded font-bold ${ok ? "bg-terminal-positive/20 text-terminal-positive" : "bg-terminal-negative/20 text-terminal-negative"}`}>
      {label}
    </span>
  );
}

function DefensePanel({ title, ok, children }: { title: string; ok: boolean; children: React.ReactNode }) {
  return (
    <div className={`border rounded p-3 ${ok ? "border-terminal-border" : "border-terminal-negative/50 bg-terminal-negative/5"}`}>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-terminal-text-primary text-[10px] font-bold tracking-wider">{title}</h3>
        <StatusBadge ok={ok} label={ok ? "SECURE" : "ALERT"} />
      </div>
      <div className="text-[10px] space-y-1">{children}</div>
    </div>
  );
}

function TokenMeterPanel({ meter }: { meter: SecurityStatus["token_meter"] | undefined }) {
  const [overriding, setOverriding] = useState(false);

  if (!meter) return null;

  const handleOverride = async () => {
    setOverriding(true);
    try {
      await fetch("/api/engine/security/tokens/override", { method: "POST" });
    } catch {}
    setOverriding(false);
  };

  return (
    <div className="border border-terminal-border rounded p-3 col-span-2">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-terminal-text-primary text-[10px] font-bold tracking-wider">TOKEN METER</h3>
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-terminal-text-muted">
            {(meter.daily_used / 1000).toFixed(0)}K / {(meter.daily_cap / 1000000).toFixed(0)}M
          </span>
          <div className="w-24 bg-terminal-border/30 rounded-full h-1.5">
            <div className={`h-1.5 rounded-full ${meter.daily_pct > 80 ? "bg-terminal-negative" : meter.daily_pct > 50 ? "bg-terminal-warning" : "bg-terminal-positive"}`}
              style={{ width: `${Math.min(meter.daily_pct, 100)}%` }} />
          </div>
          <span className="text-[10px] text-terminal-text-muted">{meter.daily_pct}%</span>
          {meter.lockdown_active && (
            <button onClick={handleOverride} disabled={overriding}
              className="px-2 py-0.5 text-[9px] bg-terminal-warning/20 text-terminal-warning border border-terminal-warning/30 rounded">
              {overriding ? "..." : "APPROVE OVERRIDE"}
            </button>
          )}
        </div>
      </div>
      <div className="grid grid-cols-4 gap-2">
        {meter.models && Object.entries(meter.models).map(([model, data]) => (
          <div key={model} className="border border-terminal-border/30 rounded p-2">
            <div className="text-[9px] text-terminal-accent font-mono mb-1">{model}</div>
            <div className="text-[10px] text-terminal-text-primary">{data.current_hour_tokens.toLocaleString()} tokens</div>
            <div className="text-[9px] text-terminal-text-muted">{data.current_hour_requests} requests</div>
            <span className={`text-[8px] ${data.current_hour_anomaly === "normal" ? "text-terminal-positive" : "text-terminal-negative"}`}>
              {data.current_hour_anomaly.toUpperCase()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function SecurityPage() {
  const { data: status } = useEngineQuery<SecurityStatus>("/security/status", { refetchInterval: 10000 });
  const { data: ledger } = useEngineQuery<LedgerData>("/security/ledger?limit=20", { refetchInterval: 30000 });

  const healthy = status?.healthy ?? true;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-terminal-text-primary text-sm font-bold tracking-wider">SECURITY</h1>
          <p className="text-[10px] text-terminal-text-muted">6-layer adversarial defense — phase chain, broker lock, circuit breaker, token meter, prompt guard, heartbeat</p>
        </div>
        <StatusBadge ok={healthy} label={healthy ? "ALL SYSTEMS SECURE" : "SECURITY ALERT"} />
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-3 gap-3">
        <DefensePanel title="1. PHASE CHAIN" ok={!status?.phase_chain?.broken}>
          <p className="text-terminal-text-muted">Chain length: {status?.phase_chain?.chain_length || 0}</p>
          {status?.phase_chain?.broken && <p className="text-terminal-negative">BROKEN at: {status.phase_chain.break_phase}</p>}
        </DefensePanel>

        <DefensePanel title="2. BROKER INTEGRITY" ok={!status?.broker_lock?.frozen}>
          <p className="text-terminal-text-muted">Discrepancies: {status?.broker_lock?.discrepancies || 0}</p>
          <p className="text-terminal-text-muted">Last recon: {status?.broker_lock?.last_recon || "never"}</p>
        </DefensePanel>

        <DefensePanel title="3. TRANSACTION LEDGER" ok={ledger?.chain_integrity?.intact ?? true}>
          <p className="text-terminal-text-muted">Entries: {ledger?.total_entries || 0}</p>
          <p className="text-terminal-text-muted">Chain: {ledger?.chain_integrity?.intact ? "INTACT" : "BROKEN"}</p>
        </DefensePanel>

        <DefensePanel title="4. CIRCUIT BREAKER" ok={!status?.circuit_breaker?.locked}>
          <p className="text-terminal-text-muted">Rate: {status?.circuit_breaker?.current_rate || 0} / {status?.circuit_breaker?.threshold || 200} per 10s</p>
          <p className="text-terminal-text-muted">Trips: {status?.circuit_breaker?.trips || 0}</p>
        </DefensePanel>

        <DefensePanel title="5. PROMPT GUARD" ok={(status?.prompt_guard?.rejections || 0) < 10}>
          <p className="text-terminal-text-muted">Rejections: {status?.prompt_guard?.rejections || 0}</p>
          <p className="text-terminal-text-muted">Max tokens: {status?.prompt_guard?.max_tokens || 8192}</p>
        </DefensePanel>

        <DefensePanel title="6. HEARTBEAT INTEGRITY" ok={status?.heartbeat?.all_healthy ?? true}>
          {status?.heartbeat?.missing?.length ? (
            <p className="text-terminal-negative">Missing: {status.heartbeat.missing.join(", ")}</p>
          ) : (
            <p className="text-terminal-text-muted">All services reporting</p>
          )}
        </DefensePanel>
      </div>

      <TokenMeterPanel meter={status?.token_meter} />

      <div className="border border-terminal-border rounded overflow-hidden">
        <div className="px-4 py-2 border-b border-terminal-border">
          <h3 className="text-terminal-text-primary text-[10px] font-bold tracking-wider">TRANSACTION LEDGER (recent)</h3>
        </div>
        <div className="max-h-[250px] overflow-y-auto">
          <table className="w-full text-[10px]">
            <thead className="sticky top-0 bg-terminal-bg">
              <tr className="text-terminal-text-muted border-b border-terminal-border">
                <th className="text-left px-3 py-1 w-10">#</th>
                <th className="text-left px-3 py-1">TYPE</th>
                <th className="text-left px-3 py-1">TIMESTAMP</th>
                <th className="text-left px-3 py-1">SIGNATURE</th>
              </tr>
            </thead>
            <tbody>
              {(ledger?.entries || []).slice(-20).reverse().map((e, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-terminal-accent/5">
                  <td className="px-3 py-1 text-terminal-text-muted">{e.seq}</td>
                  <td className="px-3 py-1 text-terminal-accent font-mono">{e.type}</td>
                  <td className="px-3 py-1 text-terminal-text-faint">{e.timestamp}</td>
                  <td className="px-3 py-1 text-terminal-text-muted font-mono">{e.signature?.slice(0, 12)}...</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
