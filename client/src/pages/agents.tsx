import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";

// ═══════════ TYPES ═══════════

type Tier = "GENERAL" | "CAPTAIN" | "LIEUTENANT" | "RECRUIT";
type Status = "ACTIVE" | "SUSPENDED" | "PROBATION" | "TERMINATED";
type AgentType = "investor_persona" | "sector_bot" | "research_bot" | "specialist" | "hybrid";

interface Agent {
  id: string;
  name: string;
  type: AgentType;
  tier: Tier;
  status: Status;
  accuracy: number;
  sharpe: number;
  hitRate: number;
  totalSignals: number;
  totalPnl: number;
  lastActive: string;
  weight: number;
}

// ═══════════ MOCK DATA ═══════════

function generateAgents(): Agent[] {
  const now = Date.now();
  const fmt = (offset: number) => {
    const d = new Date(now - offset * 60000);
    return d.toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
  };

  const agents: Agent[] = [
    // 12 investor personas — elites get 1.5x weight
    { id: "warren_buffett", name: "Warren Buffett", type: "investor_persona", tier: "GENERAL", status: "ACTIVE", accuracy: 71.4, sharpe: 2.84, hitRate: 68.2, totalSignals: 1847, totalPnl: 284720, lastActive: fmt(2), weight: 1.5 },
    { id: "charlie_munger", name: "Charlie Munger", type: "investor_persona", tier: "GENERAL", status: "ACTIVE", accuracy: 69.8, sharpe: 2.71, hitRate: 66.9, totalSignals: 1523, totalPnl: 241380, lastActive: fmt(3), weight: 1.5 },
    { id: "ben_graham", name: "Ben Graham", type: "investor_persona", tier: "GENERAL", status: "ACTIVE", accuracy: 67.3, sharpe: 2.52, hitRate: 64.1, totalSignals: 1319, totalPnl: 198450, lastActive: fmt(5), weight: 1.5 },
    { id: "stanley_druckenmiller", name: "S. Druckenmiller", type: "investor_persona", tier: "CAPTAIN", status: "ACTIVE", accuracy: 73.1, sharpe: 3.12, hitRate: 70.4, totalSignals: 2104, totalPnl: 318940, lastActive: fmt(1), weight: 1.5 },
    { id: "aswath_damodaran", name: "A. Damodaran", type: "investor_persona", tier: "CAPTAIN", status: "ACTIVE", accuracy: 68.9, sharpe: 2.67, hitRate: 65.8, totalSignals: 987, totalPnl: 176820, lastActive: fmt(8), weight: 1.5 },
    { id: "peter_lynch", name: "Peter Lynch", type: "investor_persona", tier: "CAPTAIN", status: "ACTIVE", accuracy: 64.2, sharpe: 2.14, hitRate: 61.7, totalSignals: 1654, totalPnl: 143210, lastActive: fmt(12), weight: 1.0 },
    { id: "phil_fisher", name: "Phil Fisher", type: "investor_persona", tier: "CAPTAIN", status: "ACTIVE", accuracy: 62.8, sharpe: 1.98, hitRate: 60.3, totalSignals: 1102, totalPnl: 122340, lastActive: fmt(18), weight: 1.0 },
    { id: "michael_burry", name: "Michael Burry", type: "investor_persona", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 58.4, sharpe: 1.74, hitRate: 57.1, totalSignals: 743, totalPnl: 87650, lastActive: fmt(34), weight: 1.0 },
    { id: "cathie_wood", name: "Cathie Wood", type: "investor_persona", tier: "LIEUTENANT", status: "PROBATION", accuracy: 41.7, sharpe: 0.52, hitRate: 39.8, totalSignals: 2341, totalPnl: -54820, lastActive: fmt(67), weight: 0.5 },
    { id: "bill_ackman", name: "Bill Ackman", type: "investor_persona", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 55.9, sharpe: 1.43, hitRate: 53.2, totalSignals: 892, totalPnl: 64180, lastActive: fmt(45), weight: 1.0 },
    { id: "rakesh_jhunjhunwala", name: "R. Jhunjhunwala", type: "investor_persona", tier: "RECRUIT", status: "ACTIVE", accuracy: 52.3, sharpe: 1.17, hitRate: 50.8, totalSignals: 587, totalPnl: 38940, lastActive: fmt(120), weight: 1.0 },
    { id: "mohnish_pabrai", name: "Mohnish Pabrai", type: "investor_persona", tier: "RECRUIT", status: "SUSPENDED", accuracy: 47.1, sharpe: 0.84, hitRate: 45.6, totalSignals: 423, totalPnl: 12340, lastActive: fmt(2880), weight: 0.0 },
    // 8 core analysis agents
    { id: "fundamentals", name: "Fundamentals Analyst", type: "research_bot", tier: "CAPTAIN", status: "ACTIVE", accuracy: 63.5, sharpe: 2.01, hitRate: 61.2, totalSignals: 3241, totalPnl: 164820, lastActive: fmt(1), weight: 1.0 },
    { id: "growth_agent", name: "Growth Agent", type: "research_bot", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 59.8, sharpe: 1.62, hitRate: 57.9, totalSignals: 2874, totalPnl: 98340, lastActive: fmt(2), weight: 1.0 },
    { id: "valuation", name: "Valuation Bot", type: "research_bot", tier: "CAPTAIN", status: "ACTIVE", accuracy: 65.1, sharpe: 2.23, hitRate: 63.4, totalSignals: 2109, totalPnl: 187640, lastActive: fmt(3), weight: 1.0 },
    { id: "sentiment", name: "Sentiment Engine", type: "hybrid", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 54.7, sharpe: 1.28, hitRate: 52.3, totalSignals: 4892, totalPnl: 54230, lastActive: fmt(1), weight: 1.0 },
    { id: "news_sentiment", name: "News Sentiment", type: "hybrid", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 53.2, sharpe: 1.11, hitRate: 51.0, totalSignals: 6743, totalPnl: 41870, lastActive: fmt(1), weight: 1.0 },
    { id: "technicals", name: "Technicals Bot", type: "specialist", tier: "CAPTAIN", status: "ACTIVE", accuracy: 61.4, sharpe: 1.87, hitRate: 59.7, totalSignals: 5218, totalPnl: 128940, lastActive: fmt(1), weight: 1.0 },
    { id: "portfolio_manager", name: "Portfolio Manager", type: "specialist", tier: "GENERAL", status: "ACTIVE", accuracy: 66.8, sharpe: 2.44, hitRate: 64.5, totalSignals: 892, totalPnl: 204180, lastActive: fmt(2), weight: 1.0 },
    { id: "risk_manager", name: "Risk Manager", type: "specialist", tier: "GENERAL", status: "ACTIVE", accuracy: 72.3, sharpe: 3.01, hitRate: 70.1, totalSignals: 743, totalPnl: 267340, lastActive: fmt(1), weight: 1.0 },
    // 11 GICS sector bots
    { id: "sector_energy", name: "Energy Sector Bot", type: "sector_bot", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 57.2, sharpe: 1.41, hitRate: 55.8, totalSignals: 1243, totalPnl: 72840, lastActive: fmt(15), weight: 1.0 },
    { id: "sector_materials", name: "Materials Sector Bot", type: "sector_bot", tier: "RECRUIT", status: "ACTIVE", accuracy: 50.8, sharpe: 0.98, hitRate: 49.3, totalSignals: 843, totalPnl: 18340, lastActive: fmt(28), weight: 1.0 },
    { id: "sector_industrials", name: "Industrials Bot", type: "sector_bot", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 55.4, sharpe: 1.32, hitRate: 53.9, totalSignals: 1087, totalPnl: 56420, lastActive: fmt(22), weight: 1.0 },
    { id: "sector_consumer_disc", name: "Consumer Disc. Bot", type: "sector_bot", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 58.1, sharpe: 1.54, hitRate: 56.7, totalSignals: 1432, totalPnl: 83210, lastActive: fmt(11), weight: 1.0 },
    { id: "sector_consumer_staples", name: "Consumer Staples Bot", type: "sector_bot", tier: "RECRUIT", status: "PROBATION", accuracy: 43.9, sharpe: 0.61, hitRate: 42.1, totalSignals: 687, totalPnl: -12840, lastActive: fmt(148), weight: 0.5 },
    { id: "sector_healthcare", name: "Healthcare Bot", type: "sector_bot", tier: "CAPTAIN", status: "ACTIVE", accuracy: 62.7, sharpe: 2.08, hitRate: 60.4, totalSignals: 1876, totalPnl: 143870, lastActive: fmt(9), weight: 1.0 },
    { id: "sector_financials", name: "Financials Bot", type: "sector_bot", tier: "CAPTAIN", status: "ACTIVE", accuracy: 60.3, sharpe: 1.79, hitRate: 58.1, totalSignals: 2143, totalPnl: 112480, lastActive: fmt(7), weight: 1.0 },
    { id: "sector_it", name: "IT Sector Bot", type: "sector_bot", tier: "CAPTAIN", status: "ACTIVE", accuracy: 64.8, sharpe: 2.17, hitRate: 62.5, totalSignals: 2841, totalPnl: 168940, lastActive: fmt(4), weight: 1.0 },
    { id: "sector_communication", name: "Communication Bot", type: "sector_bot", tier: "LIEUTENANT", status: "ACTIVE", accuracy: 56.9, sharpe: 1.38, hitRate: 54.7, totalSignals: 1543, totalPnl: 67820, lastActive: fmt(19), weight: 1.0 },
    { id: "sector_utilities", name: "Utilities Bot", type: "sector_bot", tier: "RECRUIT", status: "TERMINATED", accuracy: 38.4, sharpe: 0.21, hitRate: 36.9, totalSignals: 342, totalPnl: -38420, lastActive: fmt(14400), weight: 0.0 },
    { id: "sector_real_estate", name: "Real Estate Bot", type: "sector_bot", tier: "RECRUIT", status: "ACTIVE", accuracy: 49.6, sharpe: 0.87, hitRate: 48.1, totalSignals: 687, totalPnl: 11240, lastActive: fmt(63), weight: 1.0 },
  ];

  return agents;
}

function generateAccuracyTrend() {
  const trend: { date: string; accuracy: number; target: number }[] = [];
  let acc = 58.0;
  for (let i = 29; i >= 0; i--) {
    const d = new Date(Date.now() - i * 86400000);
    acc = Math.min(78, Math.max(48, acc + (Math.random() - 0.46) * 2.1));
    trend.push({
      date: d.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      accuracy: +acc.toFixed(2),
      target: 62.0,
    });
  }
  return trend;
}

// ═══════════ SMALL COMPONENTS ═══════════

const TIER_COLORS: Record<Tier, string> = {
  GENERAL: "#00d4aa",
  CAPTAIN: "#58a6ff",
  LIEUTENANT: "#d29922",
  RECRUIT: "#8b949e",
};

const STATUS_COLORS: Record<Status, string> = {
  ACTIVE: "#3fb950",
  SUSPENDED: "#d29922",
  PROBATION: "#f0883e",
  TERMINATED: "#f85149",
};

const TYPE_LABELS: Record<AgentType, string> = {
  investor_persona: "PERSONA",
  sector_bot: "SECTOR",
  research_bot: "RESEARCH",
  specialist: "SPECIALIST",
  hybrid: "HYBRID",
};

function TierBadge({ tier }: { tier: Tier }) {
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[8px] font-semibold tracking-wider"
      style={{ color: TIER_COLORS[tier], backgroundColor: `${TIER_COLORS[tier]}18`, border: `1px solid ${TIER_COLORS[tier]}40` }}
    >
      {tier}
    </span>
  );
}

function StatusBadge({ status }: { status: Status }) {
  return (
    <span
      className="px-1.5 py-0.5 rounded text-[8px] font-semibold tracking-wider"
      style={{ color: STATUS_COLORS[status], backgroundColor: `${STATUS_COLORS[status]}18`, border: `1px solid ${STATUS_COLORS[status]}40` }}
    >
      {status}
    </span>
  );
}

function MiniBar({ value, max, color }: { value: number; max: number; color: string }) {
  return (
    <div className="flex items-center gap-1.5">
      <span className="font-mono text-terminal-text-primary w-10 text-right">{value.toFixed(1)}</span>
      <div className="w-12 h-1 bg-terminal-bg rounded-full overflow-hidden">
        <div className="h-full rounded-full" style={{ width: `${(value / max) * 100}%`, backgroundColor: color }} />
      </div>
    </div>
  );
}

// ═══════════ SUMMARY CARDS ═══════════

function AgentSummaryRow({ agents }: { agents: Agent[] }) {
  const active = agents.filter(a => a.status === "ACTIVE").length;
  const suspended = agents.filter(a => a.status === "SUSPENDED").length;
  const probation = agents.filter(a => a.status === "PROBATION").length;
  const terminated = agents.filter(a => a.status === "TERMINATED").length;
  const avgAcc = agents.filter(a => a.status === "ACTIVE").reduce((s, a) => s + a.accuracy, 0) / active;
  const avgSharpe = agents.filter(a => a.status === "ACTIVE").reduce((s, a) => s + a.sharpe, 0) / active;
  const avgHit = agents.filter(a => a.status === "ACTIVE").reduce((s, a) => s + a.hitRate, 0) / active;

  const cards = [
    { label: "TOTAL AGENTS", value: `${agents.length}`, color: "text-terminal-text-primary" },
    { label: "ACTIVE", value: `${active}`, color: "text-terminal-positive" },
    { label: "SUSPENDED", value: `${suspended}`, color: "text-terminal-warning" },
    { label: "PROBATION", value: `${probation}`, color: "text-orange-400" },
    { label: "TERMINATED", value: `${terminated}`, color: "text-terminal-negative" },
    { label: "ENSEMBLE ACCURACY", value: `${avgAcc.toFixed(1)}%`, color: "text-terminal-accent" },
    { label: "AVG SHARPE", value: `${avgSharpe.toFixed(2)}`, color: "text-[#58a6ff]" },
    { label: "AVG HIT RATE", value: `${avgHit.toFixed(1)}%`, color: "text-terminal-positive" },
  ];

  return (
    <div className="grid grid-cols-8 gap-1">
      {cards.map(c => (
        <div key={c.label} className="bg-terminal-bg rounded border border-terminal-border/50 px-2 py-1.5">
          <div className="text-[9px] text-terminal-text-faint tracking-wider mb-0.5">{c.label}</div>
          <div className={`text-sm font-mono font-semibold ${c.color}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ AGENT REGISTRY TABLE ═══════════

type SortKey = "name" | "accuracy" | "sharpe" | "hitRate" | "totalSignals" | "totalPnl" | "weight";

function AgentRegistryTable({ agents }: { agents: Agent[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("accuracy");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("desc");
  const [filterType, setFilterType] = useState<AgentType | "all">("all");
  const [filterStatus, setFilterStatus] = useState<Status | "all">("all");

  const sorted = useMemo(() => {
    let filtered = agents;
    if (filterType !== "all") filtered = filtered.filter(a => a.type === filterType);
    if (filterStatus !== "all") filtered = filtered.filter(a => a.status === filterStatus);
    return [...filtered].sort((a, b) => {
      const av = a[sortKey] as number | string;
      const bv = b[sortKey] as number | string;
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "desc" ? bv - av : av - bv;
      }
      return sortDir === "desc"
        ? String(bv).localeCompare(String(av))
        : String(av).localeCompare(String(bv));
    });
  }, [agents, sortKey, sortDir, filterType, filterStatus]);

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortDir(d => d === "desc" ? "asc" : "desc");
    else { setSortKey(key); setSortDir("desc"); }
  }

  const ColH = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      className={`py-1 px-1.5 text-left font-medium cursor-pointer whitespace-nowrap select-none transition-colors ${sortKey === k ? "text-terminal-accent" : "text-terminal-text-faint hover:text-terminal-text-muted"}`}
      onClick={() => handleSort(k)}
    >
      {label}{sortKey === k ? (sortDir === "desc" ? " ↓" : " ↑") : ""}
    </th>
  );

  return (
    <div className="text-[10px] h-full flex flex-col">
      {/* Filters */}
      <div className="flex items-center gap-2 mb-1.5 flex-shrink-0 flex-wrap">
        <span className="text-terminal-text-faint">TYPE:</span>
        {(["all", "investor_persona", "sector_bot", "research_bot", "specialist", "hybrid"] as const).map(t => (
          <button
            key={t}
            onClick={() => setFilterType(t)}
            className={`px-2 py-0.5 rounded text-[9px] transition-colors ${filterType === t ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-muted hover:text-terminal-text-primary border border-transparent"}`}
          >
            {t === "all" ? "ALL" : TYPE_LABELS[t]}
          </button>
        ))}
        <span className="text-terminal-text-faint ml-2">STATUS:</span>
        {(["all", "ACTIVE", "SUSPENDED", "PROBATION", "TERMINATED"] as const).map(s => (
          <button
            key={s}
            onClick={() => setFilterStatus(s)}
            className={`px-2 py-0.5 rounded text-[9px] transition-colors ${filterStatus === s ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-muted hover:text-terminal-text-primary border border-transparent"}`}
          >
            {s}
          </button>
        ))}
        <span className="ml-auto text-terminal-text-faint">{sorted.length} / {agents.length} agents</span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-auto">
        <table className="w-full">
          <thead className="sticky top-0 bg-terminal-surface z-10">
            <tr className="border-b border-terminal-border">
              <th className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">#</th>
              <ColH k="name" label="AGENT" />
              <th className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">TYPE</th>
              <th className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">TIER</th>
              <th className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">STATUS</th>
              <ColH k="accuracy" label="ACCURACY" />
              <ColH k="sharpe" label="SHARPE" />
              <ColH k="hitRate" label="HIT RATE" />
              <ColH k="totalSignals" label="SIGNALS" />
              <ColH k="totalPnl" label="TOTAL P&L" />
              <th className="py-1 px-1.5 text-left font-medium text-terminal-text-faint whitespace-nowrap">LAST ACTIVE</th>
              <ColH k="weight" label="WEIGHT" />
            </tr>
          </thead>
          <tbody>
            {sorted.map((agent, idx) => {
              const isElite = agent.weight === 1.5;
              const isInactive = agent.status === "TERMINATED" || agent.status === "SUSPENDED";
              return (
                <tr
                  key={agent.id}
                  className={`border-b border-terminal-border/40 hover:bg-white/[0.02] transition-colors ${isInactive ? "opacity-50" : ""}`}
                >
                  <td className="py-1.5 px-1.5 font-mono text-terminal-text-faint">{idx + 1}</td>
                  <td className="py-1.5 px-1.5 font-mono font-semibold text-terminal-text-primary whitespace-nowrap">
                    {isElite && <span className="text-terminal-accent mr-1">★</span>}
                    {agent.name}
                  </td>
                  <td className="py-1.5 px-1.5">
                    <span className="px-1.5 py-0.5 rounded text-[8px] bg-white/5 text-terminal-text-muted">{TYPE_LABELS[agent.type]}</span>
                  </td>
                  <td className="py-1.5 px-1.5"><TierBadge tier={agent.tier} /></td>
                  <td className="py-1.5 px-1.5"><StatusBadge status={agent.status} /></td>
                  <td className="py-1.5 px-1.5">
                    <MiniBar value={agent.accuracy} max={80} color={agent.accuracy >= 65 ? "#3fb950" : agent.accuracy >= 55 ? "#d29922" : "#f85149"} />
                  </td>
                  <td className="py-1.5 px-1.5 font-mono" style={{ color: agent.sharpe >= 2 ? "#00d4aa" : agent.sharpe >= 1 ? "#d29922" : "#f85149" }}>
                    {agent.sharpe.toFixed(2)}
                  </td>
                  <td className="py-1.5 px-1.5">
                    <MiniBar value={agent.hitRate} max={75} color={agent.hitRate >= 60 ? "#3fb950" : agent.hitRate >= 50 ? "#d29922" : "#f85149"} />
                  </td>
                  <td className="py-1.5 px-1.5 font-mono text-terminal-text-muted">{agent.totalSignals.toLocaleString()}</td>
                  <td className={`py-1.5 px-1.5 font-mono font-semibold ${agent.totalPnl >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {agent.totalPnl >= 0 ? "+" : ""}${Math.abs(agent.totalPnl).toLocaleString()}
                  </td>
                  <td className="py-1.5 px-1.5 font-mono text-terminal-text-faint whitespace-nowrap">{agent.lastActive}</td>
                  <td className="py-1.5 px-1.5 font-mono" style={{ color: isElite ? "#00d4aa" : "#8b949e" }}>
                    {agent.weight.toFixed(1)}x
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

// ═══════════ TIER DISTRIBUTION PIE ═══════════

function TierDistribution({ agents }: { agents: Agent[] }) {
  const tierCounts = useMemo(() => {
    const map: Record<Tier, number> = { GENERAL: 0, CAPTAIN: 0, LIEUTENANT: 0, RECRUIT: 0 };
    agents.forEach(a => map[a.tier]++);
    return Object.entries(map).map(([tier, count]) => ({ tier, count }));
  }, [agents]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 min-h-0">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={tierCounts}
              dataKey="count"
              nameKey="tier"
              cx="50%"
              cy="50%"
              innerRadius="50%"
              outerRadius="80%"
              paddingAngle={3}
            >
              {tierCounts.map((entry) => (
                <Cell key={entry.tier} fill={TIER_COLORS[entry.tier as Tier]} fillOpacity={0.85} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
              formatter={(val: number, name: string) => [`${val} agents`, name]}
            />
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-1 flex-shrink-0">
        {tierCounts.map(({ tier, count }) => (
          <div key={tier} className="flex items-center gap-2 text-[10px]">
            <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: TIER_COLORS[tier as Tier] }} />
            <span className="text-terminal-text-muted flex-1">{tier}</span>
            <span className="font-mono text-terminal-text-primary font-semibold">{count}</span>
            <span className="font-mono text-terminal-text-faint text-[9px]">{((count / agents.length) * 100).toFixed(0)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ═══════════ SIGNAL CONSENSUS ═══════════

function SignalConsensus({ agents }: { agents: Agent[] }) {
  // Simulate current vote distribution
  const activeAgents = agents.filter(a => a.status === "ACTIVE");
  const bullish = Math.round(activeAgents.length * 0.52);
  const bearish = Math.round(activeAgents.length * 0.21);
  const neutral = activeAgents.length - bullish - bearish;
  const total = activeAgents.length;

  const bars = [
    { label: "BULLISH", count: bullish, pct: (bullish / total) * 100, color: "#3fb950" },
    { label: "NEUTRAL", count: neutral, pct: (neutral / total) * 100, color: "#8b949e" },
    { label: "BEARISH", count: bearish, pct: (bearish / total) * 100, color: "#f85149" },
  ];

  const consensus = bullish > bearish * 1.5 ? "BULLISH" : bearish > bullish * 1.5 ? "BEARISH" : "SPLIT";
  const consensusColor = consensus === "BULLISH" ? "#3fb950" : consensus === "BEARISH" ? "#f85149" : "#d29922";

  return (
    <div className="h-full flex flex-col gap-3">
      <div className="text-center">
        <div className="text-[9px] text-terminal-text-faint mb-1">CURRENT CONSENSUS</div>
        <div className="text-xl font-mono font-bold" style={{ color: consensusColor }}>{consensus}</div>
        <div className="text-[9px] text-terminal-text-faint">{activeAgents.length} active agents voting</div>
      </div>
      <div className="space-y-3 flex-1">
        {bars.map(b => (
          <div key={b.label}>
            <div className="flex justify-between text-[10px] mb-0.5">
              <span className="text-terminal-text-muted">{b.label}</span>
              <span className="font-mono" style={{ color: b.color }}>{b.count} ({b.pct.toFixed(0)}%)</span>
            </div>
            <div className="h-2 bg-terminal-bg rounded-full overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${b.pct}%`, backgroundColor: b.color }} />
            </div>
          </div>
        ))}
      </div>
      <div className="text-[9px] text-terminal-text-faint border-t border-terminal-border/50 pt-2 space-y-1">
        <div className="flex justify-between">
          <span>Weighted Bullish</span>
          <span className="font-mono text-terminal-positive">58.3%</span>
        </div>
        <div className="flex justify-between">
          <span>Signal Confidence</span>
          <span className="font-mono text-terminal-accent">0.72</span>
        </div>
        <div className="flex justify-between">
          <span>Ensemble Divergence</span>
          <span className="font-mono text-terminal-warning">0.31</span>
        </div>
      </div>
    </div>
  );
}

// ═══════════ ACCURACY TREND ═══════════

function AccuracyTrendChart({ data }: { data: ReturnType<typeof generateAccuracyTrend> }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={data} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
        <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 9 }} tickLine={false} axisLine={false} interval={4} />
        <YAxis
          tick={{ fill: "#484f58", fontSize: 9 }}
          tickLine={false}
          axisLine={false}
          domain={["auto", "auto"]}
          tickFormatter={(v) => `${v}%`}
        />
        <Tooltip
          contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 10 }}
          formatter={(v: number) => [`${v.toFixed(2)}%`]}
        />
        <Line type="monotone" dataKey="target" stroke="#484f58" strokeWidth={1} dot={false} strokeDasharray="4 4" name="Target" />
        <Line type="monotone" dataKey="accuracy" stroke="#00d4aa" strokeWidth={1.5} dot={false} name="Ensemble Accuracy" />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ═══════════ MAIN PAGE ═══════════

const agents = generateAgents();
const accuracyTrend = generateAccuracyTrend();

export default function AgentsPage() {
  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">
      {/* Summary Row */}
      <div className="flex-shrink-0">
        <AgentSummaryRow agents={agents} />
      </div>

      {/* Main content area */}
      <div className="flex-1 grid grid-cols-[1fr_200px] gap-1 overflow-hidden">
        {/* Left: registry + trend */}
        <div className="flex flex-col gap-1 overflow-hidden">
          {/* Agent Registry Table */}
          <DashboardPanel
            title="AGENT REGISTRY"
            className="flex-1 overflow-hidden"
            noPadding
            headerRight={
              <span className="text-[9px] text-terminal-text-faint font-mono">
                {agents.filter(a => a.status === "ACTIVE").length} ACTIVE / {agents.length} TOTAL
              </span>
            }
          >
            <div className="p-2 h-full overflow-hidden flex flex-col">
              <AgentRegistryTable agents={agents} />
            </div>
          </DashboardPanel>

          {/* Accuracy trend */}
          <DashboardPanel title="30-DAY ENSEMBLE ACCURACY TREND" className="h-28 flex-shrink-0">
            <AccuracyTrendChart data={accuracyTrend} />
          </DashboardPanel>
        </div>

        {/* Right sidebar */}
        <div className="flex flex-col gap-1 overflow-hidden">
          <DashboardPanel title="TIER DISTRIBUTION" className="flex-1">
            <TierDistribution agents={agents} />
          </DashboardPanel>
          <DashboardPanel title="SIGNAL CONSENSUS" className="flex-1">
            <SignalConsensus agents={agents} />
          </DashboardPanel>
        </div>
      </div>
    </div>
  );
}
