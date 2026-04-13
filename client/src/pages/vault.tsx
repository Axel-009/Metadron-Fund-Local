import { useState } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

interface VaultSlot {
  label: string;
  target: string;
  required: boolean;
  category: string;
  configured: boolean;
  masked: string;
  is_boolean?: boolean;
  auto_generate?: boolean;
}

interface VaultStatus {
  total_slots: number;
  configured: number;
  slots: Record<string, VaultSlot>;
  timestamp: string;
}

const CATEGORY_COLORS: Record<string, string> = {
  execution: "text-terminal-warning",
  data: "text-terminal-accent",
  intelligence: "text-purple-400",
  system: "text-terminal-text-muted",
};

const CATEGORY_LABELS: Record<string, string> = {
  execution: "EXECUTION",
  data: "DATA",
  intelligence: "INTELLIGENCE",
  system: "SYSTEM",
};

function SlotRow({ slotKey, slot, onSave }: { slotKey: string; slot: VaultSlot; onSave: (key: string, value: string) => void }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<string | null>(null);

  const handleSave = () => {
    if (value.trim()) {
      onSave(slotKey, value.trim());
      setValue("");
      setEditing(false);
    }
  };

  const handleTest = async () => {
    setTesting(true);
    setTestResult(null);
    try {
      const res = await fetch("/api/engine/vault/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slot: slotKey, value: value || slot.masked }),
      });
      const data = await res.json();
      setTestResult(data.status);
    } catch {
      setTestResult("error");
    }
    setTesting(false);
  };

  return (
    <tr className="border-b border-terminal-border/30 hover:bg-terminal-accent/5">
      <td className="px-4 py-2">
        <span className={`text-[9px] font-bold ${CATEGORY_COLORS[slot.category] || "text-terminal-text-muted"}`}>
          {CATEGORY_LABELS[slot.category] || slot.category.toUpperCase()}
        </span>
      </td>
      <td className="px-4 py-2">
        <div className="text-[10px] text-terminal-text-primary font-mono">{slotKey}</div>
        <div className="text-[9px] text-terminal-text-muted">{slot.label}</div>
      </td>
      <td className="px-4 py-2 text-[10px] text-terminal-text-faint">{slot.target}</td>
      <td className="px-4 py-2">
        {slot.configured ? (
          <span className="text-[10px] text-terminal-positive">
            {slot.masked}
          </span>
        ) : (
          <span className="text-[10px] text-terminal-negative">NOT SET</span>
        )}
      </td>
      <td className="px-4 py-2">
        {slot.auto_generate ? (
          <span className="text-[9px] text-terminal-text-muted">auto</span>
        ) : editing ? (
          <div className="flex gap-1">
            <input
              type="password"
              value={value}
              onChange={(e) => setValue(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSave()}
              placeholder="Paste key..."
              className="w-40 bg-terminal-bg border border-terminal-border rounded px-2 py-0.5 text-[10px] text-terminal-text-primary"
              autoFocus
            />
            <button onClick={handleSave} className="px-2 py-0.5 text-[9px] bg-terminal-positive/20 text-terminal-positive rounded">SET</button>
            <button onClick={handleTest} disabled={testing} className="px-2 py-0.5 text-[9px] bg-terminal-accent/20 text-terminal-accent rounded">
              {testing ? "..." : "TEST"}
            </button>
            <button onClick={() => setEditing(false)} className="px-2 py-0.5 text-[9px] bg-terminal-negative/20 text-terminal-negative rounded">X</button>
          </div>
        ) : (
          <button
            onClick={() => setEditing(true)}
            className="px-2 py-0.5 text-[9px] bg-terminal-accent/20 text-terminal-accent border border-terminal-accent/30 rounded hover:bg-terminal-accent/30"
          >
            {slot.configured ? "UPDATE" : "SET KEY"}
          </button>
        )}
        {testResult && (
          <span className={`ml-2 text-[9px] ${testResult === "valid" ? "text-terminal-positive" : "text-terminal-warning"}`}>
            {testResult}
          </span>
        )}
      </td>
    </tr>
  );
}

export default function VaultPage() {
  const { data: status, refetch } = useEngineQuery<VaultStatus>("/vault/status", { refetchInterval: 30000 });

  const handleSave = async (slot: string, value: string) => {
    try {
      await fetch("/api/engine/vault/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ slot, value }),
      });
      refetch();
    } catch (e) {
      console.error("Vault save failed:", e);
    }
  };

  const configured = status?.configured || 0;
  const total = status?.total_slots || 0;

  return (
    <div className="h-full overflow-y-auto p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-terminal-text-primary text-sm font-bold tracking-wider">API VAULT</h1>
          <p className="text-[10px] text-terminal-text-muted">Centralized API key management — keys deploy to engines immediately on save</p>
        </div>
        <div className="flex items-center gap-3">
          <span className={`text-xs font-mono ${configured === total ? "text-terminal-positive" : "text-terminal-warning"}`}>
            {configured}/{total} CONFIGURED
          </span>
          <span className={`text-[10px] px-2 py-0.5 rounded ${configured === total ? "bg-terminal-positive/20 text-terminal-positive" : "bg-terminal-warning/20 text-terminal-warning"}`}>
            {configured === total ? "ALL KEYS SET" : "KEYS MISSING"}
          </span>
        </div>
      </div>

      <div className="border border-terminal-border rounded overflow-hidden">
        <table className="w-full text-[10px]">
          <thead className="bg-terminal-bg">
            <tr className="text-terminal-text-muted border-b border-terminal-border">
              <th className="text-left px-4 py-2 w-24">CATEGORY</th>
              <th className="text-left px-4 py-2 w-48">KEY SLOT</th>
              <th className="text-left px-4 py-2">DEPLOYS TO</th>
              <th className="text-left px-4 py-2 w-32">STATUS</th>
              <th className="text-left px-4 py-2 w-56">ACTION</th>
            </tr>
          </thead>
          <tbody>
            {status?.slots && Object.entries(status.slots)
              .sort(([, a], [, b]) => {
                const order = { execution: 0, data: 1, intelligence: 2, system: 3 };
                return (order[a.category as keyof typeof order] ?? 9) - (order[b.category as keyof typeof order] ?? 9);
              })
              .map(([key, slot]) => (
                <SlotRow key={key} slotKey={key} slot={slot} onSave={handleSave} />
              ))}
          </tbody>
        </table>
      </div>

      <div className="border border-terminal-border rounded p-3 text-[10px] text-terminal-text-muted">
        <p className="font-bold text-terminal-text-primary mb-1">How it works:</p>
        <ul className="space-y-0.5 list-disc list-inside">
          <li>Keys are stored in <code>data/vault/api_vault.json</code> and deployed to <code>os.environ</code> on startup</li>
          <li>The internal proxy token is auto-generated — the Express frontend uses it to authenticate with the backend</li>
          <li>API keys are never exposed in full — only masked previews shown</li>
          <li>Changes take effect immediately — no restart required</li>
        </ul>
      </div>
    </div>
  );
}
