import { useState, useMemo } from "react";
import { DashboardPanel } from "@/components/dashboard-panel";
import { useEngineQuery } from "@/hooks/use-engine-api";
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, BarChart, Bar,
} from "recharts";

// ═══════════ TYPES ═══════════

interface AgentRecord {
  name: string;
  rank: number;
  category: string;
  description: string;
  strategy: string;
  style: string;
  sector_bias: string[];
  tier: string;
  is_active: boolean;
  weight: number;
  accuracy: number;
  sharpe: number;
  hit_rate: number;
  composite_score: number;
  total_signals: number;
  correct_signals: number;
  win_streak: number;
  loss_streak: number;
  max_win_streak: number;
  max_loss_streak: number;
  consecutive_top_weeks: number;
  consecutive_bottom_weeks: number;
  rolling_sharpe: number;
}

interface SystemStats {
  total_agents: number;
  active_agents: number;
  tier_distribution: Record<string, number>;
  category_distribution: Record<string, number>;
  avg_composite_score: number;
  avg_accuracy: number;
  avg_sharpe: number;
  total_signals_system: number;
  total_correct_system: number;
  consensus_history_size: number;
}

interface ConsensusSummary {
  bull_pct: number;
  bear_pct: number;
  neutral_pct: number;
  weighted_confidence: number;
  avg_agreement: number;
}

interface AccuracyDay {
  date: string;
  accuracy: number;
  agent_count: number;
}

interface SkillRecord {
  name: string;
  category: string;
  proficiency: number;
  times_used: number;
  source: string;
  agents_using: string[];
}

interface LearningEvent {
  id: string;
  timestamp: string;
  type: string;
  agent: string;
  what: string;
  confidence: number;
  severity: string;
}

interface LearningRateDay {
  day: string;
  rate: number;
}

interface ModelVersion {
  version: string;
  date: string;
  component: string;
  notes: string;
}

interface OrchestratorStatus {
  initialized: boolean;
  gsd_active: boolean;
  paul_active: boolean;
  wrapper_active: boolean;
  factory_active: boolean;
  enforcement_active: boolean;
  attached_agents: number;
  sector_bots: number;
  research_bots: number;
  personas: number;
  dynamic_agents: number;
}

interface EnforcementState {
  collective_state: {
    active_agents: number;
    consensus_score: number;
    herding_risk: number;
    concentration_risk: number;
    signal_diversity: number;
    gradient_alignment: number;
    total_pnl: number;
    avg_accuracy: number;
  };
  recent_events: Array<{
    timestamp: string;
    agent: string;
    rule: string;
    action: string;
    severity: string;
  }>;
}

// ═══════════ COLOR MAPS ═══════════

const TIER_COLORS: Record<string, string> = {
  TIER_1_General: "#ffd700",
  TIER_2_Captain: "#3fb950",
  TIER_3_Lieutenant: "#58a6ff",
  TIER_4_Recruit: "#8b949e",
};

const CATEGORY_COLORS: Record<string, string> = {
  investor_persona: "#bc8cff",
  analytical: "#58a6ff",
  engine: "#00d4aa",
};

const SKILL_CAT_COLORS: Record<string, string> = {
  Trading: "#f0883e",
  Analysis: "#58a6ff",
  Risk: "#f85149",
  Data: "#3fb950",
};

const LEARNING_TYPE_COLORS: Record<string, string> = {
  enforcement: "#f85149",
  promotion: "#3fb950",
  demotion: "#d29922",
  system: "#58a6ff",
};

const SEVERITY_COLORS: Record<string, string> = {
  INFO: "#8b949e",
  WARNING: "#d29922",
  CRITICAL: "#f85149",
};

// ═══════════ AGENT SUMMARY ROW ═══════════

function AgentSummaryRow({ stats, consensus }: {
  stats: SystemStats | null;
  consensus: ConsensusSummary | null;
}) {
  const cards = useMemo(() => {
    if (!stats) return [];
    const sysBullPct = consensus?.bull_pct ?? 0;
    const avgAcc = stats.avg_accuracy ? (stats.avg_accuracy * 100).toFixed(1) : "0.0";
    return [
      { label: "TOTAL AGENTS", value: `${stats.total_agents}`, color: "text-terminal-accent" },
      { label: "ACTIVE", value: `${stats.active_agents}`, color: "text-terminal-positive" },
      { label: "AVG ACCURACY", value: `${avgAcc}%`, color: "text-[#58a6ff]" },
      { label: "AVG SHARPE", value: `${stats.avg_sharpe.toFixed(2)}`, color: "text-terminal-warning" },
      { label: "TOTAL SIGNALS", value: stats.total_signals_system.toLocaleString(), color: "text-terminal-text-primary" },
      { label: "BULL CONSENSUS", value: `${sysBullPct.toFixed(1)}%`, color: sysBullPct > 50 ? "text-terminal-positive" : "text-terminal-negative" },
    ];
  }, [stats, consensus]);

  if (!stats) {
    return (
      <div className="grid grid-cols-6 gap-1">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="bg-terminal-bg rounded border border-terminal-border/50 px-2 py-1.5 animate-pulse">
            <div className="h-3 bg-terminal-border/30 rounded w-20 mb-1" />
            <div className="h-4 bg-terminal-border/30 rounded w-12" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-6 gap-1">
      {cards.map(c => (
        <div key={c.label} className="bg-terminal-bg rounded border border-terminal-border/50 px-2 py-1.5">
          <div className="text-[9px] text-terminal-text-faint tracking-wider mb-0.5">{c.label}</div>
          <div className={`text-sm font-mono font-semibold ${c.color}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}

// ═══════════ SIGNAL CONSENSUS PANEL ═══════════

function SignalConsensus({ consensus, enforcement }: {
  consensus: ConsensusSummary | null;
  enforcement: EnforcementState | null;
}) {
  const bull = consensus?.bull_pct ?? 0;
  const bear = consensus?.bear_pct ?? 0;
  const neutral = consensus?.neutral_pct ?? 0;
  const confidence = consensus?.weighted_confidence ?? 0;
  const agreement = consensus?.avg_agreement ?? 0;
  const herdRisk = enforcement?.collective_state?.herding_risk ?? 0;
  const gradAlign = enforcement?.collective_state?.gradient_alignment ?? 0;

  const pieData = [
    { name: "Bull", value: bull, color: "#3fb950" },
    { name: "Bear", value: bear, color: "#f85149" },
    { name: "Neutral", value: neutral, color: "#8b949e" },
  ];

  return (
    <DashboardPanel title="SIGNAL CONSENSUS" className="flex-shrink-0" headerRight={
      <div className="flex items-center gap-2 text-[9px] font-mono">
        <span className="text-terminal-positive">{bull.toFixed(1)}% BULL</span>
        <span className="text-terminal-negative">{bear.toFixed(1)}% BEAR</span>
        <span className="text-terminal-text-faint">{neutral.toFixed(1)}% HOLD</span>
      </div>
    }>
      <div className="flex items-center gap-4">
        {/* Pie chart */}
        <div className="w-20 h-20 flex-shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%" cy="50%"
                innerRadius={22} outerRadius={36}
                dataKey="value" stroke="none"
              >
                {pieData.map((d, i) => (
                  <Cell key={i} fill={d.color} fillOpacity={0.8} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Metrics */}
        <div className="grid grid-cols-4 gap-3 flex-1">
          <div>
            <div className="text-[8px] text-terminal-text-faint mb-0.5">WEIGHTED BULL</div>
            <div className="text-sm font-mono font-semibold text-terminal-positive">{bull.toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-[8px] text-terminal-text-faint mb-0.5">CONFIDENCE</div>
            <div className="text-sm font-mono font-semibold text-[#58a6ff]">{(confidence * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-[8px] text-terminal-text-faint mb-0.5">AGREEMENT</div>
            <div className="text-sm font-mono font-semibold text-terminal-warning">{agreement.toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-[8px] text-terminal-text-faint mb-0.5">HERDING RISK</div>
            <div className={`text-sm font-mono font-semibold ${herdRisk > 0.7 ? "text-terminal-negative" : herdRisk > 0.4 ? "text-terminal-warning" : "text-terminal-positive"}`}>
              {(herdRisk * 100).toFixed(1)}%
            </div>
          </div>
        </div>

        {/* Divergence gauge */}
        <div className="flex-shrink-0 text-center">
          <div className="text-[8px] text-terminal-text-faint mb-0.5">GRADIENT ALIGN</div>
          <div className="w-14 h-1.5 bg-terminal-bg rounded-full overflow-hidden mx-auto">
            <div
              className="h-full rounded-full bg-terminal-accent"
              style={{ width: `${gradAlign * 100}%` }}
            />
          </div>
          <div className="text-[9px] font-mono text-terminal-accent mt-0.5">{(gradAlign * 100).toFixed(0)}%</div>
        </div>
      </div>
    </DashboardPanel>
  );
}

// ═══════════ ACCURACY TREND CHART ═══════════

function AccuracyTrend({ data }: { data: AccuracyDay[] }) {
  return (
    <DashboardPanel title="ACCURACY TREND — 30 DAY" className="h-36">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
          <XAxis dataKey="date" tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false}
            tickFormatter={(v: string) => v.slice(5)} interval={6} />
          <YAxis tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false}
            domain={[0, 1]} tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip
            contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 9 }}
            formatter={(v: number) => [`${(v * 100).toFixed(1)}%`, "Accuracy"]}
            labelFormatter={(l: string) => l}
          />
          <Line type="monotone" dataKey="accuracy" stroke="#00d4aa" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </DashboardPanel>
  );
}

// ═══════════ TIER DISTRIBUTION PIE ═══════════

function TierDistribution({ stats }: { stats: SystemStats | null }) {
  const data = useMemo(() => {
    if (!stats?.tier_distribution) return [];
    return Object.entries(stats.tier_distribution).map(([tier, count]) => ({
      name: tier.replace("TIER_", "T").replace("_", " "),
      value: count as number,
      color: TIER_COLORS[tier] || "#8b949e",
    }));
  }, [stats]);

  if (!data.length) return null;

  return (
    <DashboardPanel title="TIER DISTRIBUTION" className="h-36">
      <div className="flex items-center gap-3 h-full">
        <ResponsiveContainer width="50%" height="100%">
          <PieChart>
            <Pie data={data} cx="50%" cy="50%" innerRadius={20} outerRadius={36} dataKey="value" stroke="none">
              {data.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.8} />)}
            </Pie>
            <Tooltip contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 9 }} />
          </PieChart>
        </ResponsiveContainer>
        <div className="flex flex-col gap-1">
          {data.map(d => (
            <div key={d.name} className="flex items-center gap-1.5 text-[9px] font-mono">
              <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
              <span className="text-terminal-text-muted">{d.name}</span>
              <span className="text-terminal-text-primary font-semibold">{d.value}</span>
            </div>
          ))}
        </div>
      </div>
    </DashboardPanel>
  );
}

// ═══════════ REGISTRY SUB-TAB ═══════════

function RegistryTab({ agents }: { agents: AgentRecord[] }) {
  const [sortBy, setSortBy] = useState<"rank" | "accuracy" | "sharpe" | "composite_score">("rank");
  const [filterCat, setFilterCat] = useState<string>("all");
  const [filterTier, setFilterTier] = useState<string>("all");

  const sorted = useMemo(() => {
    let list = [...agents];
    if (filterCat !== "all") list = list.filter(a => a.category === filterCat);
    if (filterTier !== "all") list = list.filter(a => a.tier === filterTier);
    list.sort((a, b) => {
      if (sortBy === "rank") return a.rank - b.rank;
      return (b as any)[sortBy] - (a as any)[sortBy];
    });
    return list;
  }, [agents, sortBy, filterCat, filterTier]);

  const categories = ["all", ...new Set(agents.map(a => a.category))];
  const tiers = ["all", ...new Set(agents.map(a => a.tier))];

  return (
    <DashboardPanel
      title="AGENT REGISTRY"
      className="flex-1 overflow-hidden"
      noPadding
      headerRight={
        <div className="flex items-center gap-1">
          {categories.map(c => (
            <button key={c} onClick={() => setFilterCat(c)}
              className={`px-1.5 py-0.5 rounded text-[8px] transition-colors ${filterCat === c ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"}`}>
              {c === "all" ? "ALL" : c.replace("_", " ").toUpperCase()}
            </button>
          ))}
          <span className="w-px h-3 bg-terminal-border mx-1" />
          {tiers.map(t => (
            <button key={t} onClick={() => setFilterTier(t)}
              className={`px-1.5 py-0.5 rounded text-[8px] transition-colors ${filterTier === t ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"}`}>
              {t === "all" ? "ALL" : t.replace("TIER_", "T").replace("_", " ")}
            </button>
          ))}
        </div>
      }
    >
      <div className="overflow-auto h-full">
        <table className="w-full text-[10px] font-mono">
          <thead className="sticky top-0 bg-terminal-surface z-10">
            <tr className="border-b border-terminal-border text-terminal-text-faint text-[9px] uppercase tracking-wider">
              <th className="py-1.5 px-2 text-left w-8 cursor-pointer" onClick={() => setSortBy("rank")}>#</th>
              <th className="py-1.5 px-2 text-left">Agent</th>
              <th className="py-1.5 px-2 text-left">Category</th>
              <th className="py-1.5 px-2 text-left">Tier</th>
              <th className="py-1.5 px-2 text-right cursor-pointer" onClick={() => setSortBy("accuracy")}>Accuracy</th>
              <th className="py-1.5 px-2 text-right cursor-pointer" onClick={() => setSortBy("sharpe")}>Sharpe</th>
              <th className="py-1.5 px-2 text-right cursor-pointer" onClick={() => setSortBy("composite_score")}>Score</th>
              <th className="py-1.5 px-2 text-right">Signals</th>
              <th className="py-1.5 px-2 text-right">Weight</th>
              <th className="py-1.5 px-2 text-center">Streaks</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map(agent => {
              const tierColor = TIER_COLORS[agent.tier] || "#8b949e";
              const catColor = CATEGORY_COLORS[agent.category] || "#8b949e";
              const accColor = agent.accuracy >= 0.8 ? "#3fb950" : agent.accuracy >= 0.55 ? "#d29922" : "#f85149";
              const isTop = agent.rank <= 3;
              return (
                <tr key={agent.name}
                  className={`border-b border-terminal-border/30 hover:bg-white/[0.02] transition-colors ${isTop ? "bg-terminal-accent/3" : ""}`}>
                  <td className="py-1.5 px-2 text-terminal-text-faint">{agent.rank}</td>
                  <td className="py-1.5 px-2">
                    <div className="flex items-center gap-1.5">
                      {isTop && <span className="text-terminal-accent text-[10px]">◆</span>}
                      <div>
                        <span className={`font-semibold ${isTop ? "text-terminal-accent" : "text-terminal-text-primary"}`}>
                          {agent.name}
                        </span>
                        {agent.description && (
                          <div className="text-[8px] text-terminal-text-faint truncate max-w-[200px]">{agent.description}</div>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="py-1.5 px-2">
                    <span className="px-1.5 py-0.5 rounded text-[8px] font-semibold"
                      style={{ color: catColor, backgroundColor: `${catColor}15`, border: `1px solid ${catColor}35` }}>
                      {agent.category.replace("_", " ").toUpperCase()}
                    </span>
                  </td>
                  <td className="py-1.5 px-2">
                    <span className="px-1.5 py-0.5 rounded text-[8px] font-semibold"
                      style={{ color: tierColor, backgroundColor: `${tierColor}15`, border: `1px solid ${tierColor}35` }}>
                      {agent.tier.replace("TIER_", "T").replace("_", " ")}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-right">
                    <span className="tabular-nums font-semibold" style={{ color: accColor }}>
                      {(agent.accuracy * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-right">
                    <span className={`tabular-nums font-semibold ${agent.sharpe >= 2 ? "text-terminal-positive" : agent.sharpe >= 1 ? "text-terminal-warning" : "text-terminal-negative"}`}>
                      {agent.sharpe.toFixed(2)}
                    </span>
                  </td>
                  <td className="py-1.5 px-2 text-right">
                    <div className="flex items-center justify-end gap-1.5">
                      <div className="w-10 h-1 bg-terminal-bg rounded-full overflow-hidden">
                        <div className="h-full rounded-full bg-terminal-accent"
                          style={{ width: `${Math.min(agent.composite_score * 100, 100)}%` }} />
                      </div>
                      <span className="tabular-nums text-terminal-text-primary text-[9px]">
                        {agent.composite_score.toFixed(3)}
                      </span>
                    </div>
                  </td>
                  <td className="py-1.5 px-2 text-right tabular-nums text-terminal-text-muted">
                    {agent.total_signals.toLocaleString()}
                  </td>
                  <td className="py-1.5 px-2 text-right tabular-nums text-terminal-text-muted">
                    {agent.weight.toFixed(2)}
                  </td>
                  <td className="py-1.5 px-2 text-center">
                    <span className="text-terminal-positive text-[9px]">W{agent.win_streak}</span>
                    <span className="text-terminal-text-faint mx-0.5">/</span>
                    <span className="text-terminal-negative text-[9px]">L{agent.loss_streak}</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </DashboardPanel>
  );
}

// ═══════════ ORCHESTRATOR STATUS PANEL ═══════════

function OrchestratorPanel({ status }: { status: OrchestratorStatus | null }) {
  if (!status) return null;

  const items = [
    { label: "GSD Plugin", active: status.gsd_active },
    { label: "Paul Plugin", active: status.paul_active },
    { label: "Learning Wrapper", active: status.wrapper_active },
    { label: "Agent Factory", active: status.factory_active },
    { label: "Enforcement", active: status.enforcement_active },
  ];

  return (
    <DashboardPanel title="PAUL ORCHESTRATOR" className="h-36">
      <div className="flex gap-4 h-full items-center">
        <div className="flex flex-col gap-1">
          {items.map(it => (
            <div key={it.label} className="flex items-center gap-1.5 text-[9px] font-mono">
              <span className={`w-1.5 h-1.5 rounded-full ${it.active ? "bg-terminal-positive" : "bg-terminal-text-faint"}`}
                style={it.active ? { boxShadow: "0 0 4px #3fb950" } : {}} />
              <span className={it.active ? "text-terminal-text-primary" : "text-terminal-text-faint"}>{it.label}</span>
            </div>
          ))}
        </div>
        <div className="border-l border-terminal-border/30 pl-4 flex flex-col gap-1">
          <div className="text-[8px] text-terminal-text-faint">FLEET</div>
          <div className="text-lg font-mono font-semibold text-terminal-accent">{status.attached_agents}</div>
          <div className="grid grid-cols-3 gap-2 text-[8px]">
            <div><span className="text-terminal-text-faint">Sector</span> <span className="text-terminal-text-primary font-semibold">{status.sector_bots}</span></div>
            <div><span className="text-terminal-text-faint">Research</span> <span className="text-terminal-text-primary font-semibold">{status.research_bots}</span></div>
            <div><span className="text-terminal-text-faint">Personas</span> <span className="text-terminal-text-primary font-semibold">{status.personas}</span></div>
          </div>
          {status.dynamic_agents > 0 && (
            <div className="text-[8px] text-terminal-warning">+{status.dynamic_agents} dynamic</div>
          )}
        </div>
      </div>
    </DashboardPanel>
  );
}

// ═══════════ ENFORCEMENT PANEL ═══════════

function EnforcementPanel({ data }: { data: EnforcementState | null }) {
  if (!data) return null;
  const cs = data.collective_state;

  const metrics = [
    { label: "HERDING RISK", value: (cs.herding_risk * 100).toFixed(1) + "%", color: cs.herding_risk > 0.7 ? "#f85149" : cs.herding_risk > 0.4 ? "#d29922" : "#3fb950" },
    { label: "CONCENTRATION", value: (cs.concentration_risk * 100).toFixed(1) + "%", color: cs.concentration_risk > 0.15 ? "#f85149" : "#3fb950" },
    { label: "SIGNAL DIVERSITY", value: `${cs.signal_diversity}`, color: cs.signal_diversity >= 3 ? "#3fb950" : "#d29922" },
    { label: "CONSENSUS", value: (cs.consensus_score * 100).toFixed(1) + "%", color: "#58a6ff" },
  ];

  return (
    <DashboardPanel title="ENFORCEMENT ENGINE" noPadding>
      <div className="px-2 py-1.5">
        <div className="grid grid-cols-4 gap-2 mb-2">
          {metrics.map(m => (
            <div key={m.label}>
              <div className="text-[8px] text-terminal-text-faint">{m.label}</div>
              <div className="text-xs font-mono font-semibold" style={{ color: m.color }}>{m.value}</div>
            </div>
          ))}
        </div>
      </div>
      {data.recent_events.length > 0 && (
        <div className="overflow-auto max-h-24 border-t border-terminal-border/30">
          <table className="w-full text-[9px] font-mono">
            <tbody>
              {data.recent_events.slice(0, 8).map((ev, i) => (
                <tr key={i} className="border-b border-terminal-border/20 hover:bg-white/[0.02]">
                  <td className="py-1 px-2 text-terminal-text-faint w-16">{ev.timestamp.slice(11, 19)}</td>
                  <td className="py-1 px-2">
                    <span className="px-1 py-0.5 rounded text-[7px] font-semibold"
                      style={{ color: SEVERITY_COLORS[ev.severity] || "#8b949e", backgroundColor: `${SEVERITY_COLORS[ev.severity] || "#8b949e"}15` }}>
                      {ev.severity}
                    </span>
                  </td>
                  <td className="py-1 px-2 text-terminal-text-muted truncate max-w-[150px]">{ev.agent}</td>
                  <td className="py-1 px-2 text-terminal-text-muted">{ev.rule}</td>
                  <td className="py-1 px-2 text-terminal-text-faint">{ev.action}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </DashboardPanel>
  );
}

// ═══════════ SKILLS SUB-TAB ═══════════

function SkillsTab({ skills }: { skills: SkillRecord[] }) {
  const [filterCat, setFilterCat] = useState<string>("all");
  const [filterSource, setFilterSource] = useState<string>("all");

  const cats = ["all", ...new Set(skills.map(s => s.category))];
  const sources = ["all", ...new Set(skills.map(s => s.source))];

  const filtered = useMemo(() => {
    return skills.filter(s => {
      if (filterCat !== "all" && s.category !== filterCat) return false;
      if (filterSource !== "all" && s.source !== filterSource) return false;
      return true;
    });
  }, [skills, filterCat, filterSource]);

  return (
    <div className="flex-1 overflow-hidden flex flex-col gap-1">
      <DashboardPanel
        title="AGENT SKILLS INVENTORY"
        className="flex-1 overflow-hidden"
        noPadding
        headerRight={
          <div className="flex items-center gap-1">
            {cats.map(c => (
              <button key={c} onClick={() => setFilterCat(c)}
                className={`px-1.5 py-0.5 rounded text-[8px] transition-colors ${filterCat === c ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"}`}>
                {c === "all" ? "ALL" : c.toUpperCase()}
              </button>
            ))}
            <span className="w-px h-3 bg-terminal-border mx-1" />
            {sources.map(s => (
              <button key={s} onClick={() => setFilterSource(s)}
                className={`px-1.5 py-0.5 rounded text-[8px] transition-colors ${filterSource === s ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30" : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"}`}>
                {s.toUpperCase()}
              </button>
            ))}
          </div>
        }
      >
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border text-terminal-text-faint text-[9px] uppercase tracking-wider">
                <th className="py-1.5 px-2 text-left">Skill Name</th>
                <th className="py-1.5 px-2 text-left">Category</th>
                <th className="py-1.5 px-2 text-left w-[180px]">Proficiency</th>
                <th className="py-1.5 px-2 text-right">Times Used</th>
                <th className="py-1.5 px-2 text-left">Source</th>
                <th className="py-1.5 px-2 text-left">Agents</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(skill => {
                const catColor = SKILL_CAT_COLORS[skill.category] || "#8b949e";
                const sourceColor = skill.source === "Built-in" ? "#00d4aa" : skill.source === "Learned" ? "#bc8cff" : "#58a6ff";
                return (
                  <tr key={skill.name} className="border-b border-terminal-border/30 hover:bg-white/[0.02] transition-colors">
                    <td className="py-1.5 px-2 font-semibold text-terminal-text-primary">{skill.name}</td>
                    <td className="py-1.5 px-2">
                      <span className="px-1.5 py-0.5 rounded text-[8px] font-semibold"
                        style={{ color: catColor, backgroundColor: `${catColor}15`, border: `1px solid ${catColor}35` }}>
                        {skill.category.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-1.5 px-2">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-terminal-bg rounded-full overflow-hidden">
                          <div className="h-full rounded-full transition-all"
                            style={{
                              width: `${skill.proficiency}%`,
                              backgroundColor: skill.proficiency >= 90 ? "#00d4aa" : skill.proficiency >= 75 ? "#3fb950" : "#d29922",
                            }} />
                        </div>
                        <span className="text-[9px] text-terminal-text-primary tabular-nums w-8 text-right">
                          {skill.proficiency.toFixed(0)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-1.5 px-2 text-right tabular-nums text-terminal-text-muted">
                      {skill.times_used.toLocaleString()}
                    </td>
                    <td className="py-1.5 px-2">
                      <span className="px-1.5 py-0.5 rounded text-[8px] font-semibold"
                        style={{ color: sourceColor, backgroundColor: `${sourceColor}15`, border: `1px solid ${sourceColor}35` }}>
                        {skill.source.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 text-terminal-text-faint text-[9px] truncate max-w-[120px]">
                      {skill.agents_using.join(", ")}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </DashboardPanel>
    </div>
  );
}

// ═══════════ LEARNING SUB-TAB ═══════════

function LearningTab({ learnings, learningRate, modelVersions, stats }: {
  learnings: LearningEvent[];
  learningRate: LearningRateDay[];
  modelVersions: ModelVersion[];
  stats: SystemStats | null;
}) {
  const totalLessons = stats?.total_signals_system ?? 0;
  const avgConf = learnings.length > 0
    ? (learnings.reduce((s, l) => s + l.confidence, 0) / learnings.length * 100).toFixed(1)
    : "0.0";
  const recentCount = learnings.length;
  const latestVersion = modelVersions.length > 0 ? modelVersions[0].version : "—";

  const statCards = [
    { label: "TOTAL SIGNALS PROCESSED", value: totalLessons.toLocaleString(), color: "text-terminal-accent" },
    { label: "RECENT EVENTS", value: `${recentCount}`, color: "text-terminal-positive" },
    { label: "AVG CONFIDENCE", value: `${avgConf}%`, color: "text-[#58a6ff]" },
    { label: "ORCHESTRATOR VER", value: latestVersion, color: "text-terminal-warning" },
    { label: "ACTIVE AGENTS", value: `${stats?.active_agents ?? 0}`, color: "text-terminal-text-primary" },
    { label: "AVG SHARPE", value: `${stats?.avg_sharpe?.toFixed(2) ?? "0.00"}`, color: "text-terminal-positive" },
  ];

  return (
    <div className="flex-1 overflow-hidden flex flex-col gap-1">
      {/* Stat cards */}
      <div className="grid grid-cols-6 gap-1 flex-shrink-0">
        {statCards.map(c => (
          <div key={c.label} className="bg-terminal-bg rounded border border-terminal-border/50 px-2 py-1.5">
            <div className="text-[9px] text-terminal-text-faint tracking-wider mb-0.5">{c.label}</div>
            <div className={`text-sm font-mono font-semibold ${c.color}`}>{c.value}</div>
          </div>
        ))}
      </div>

      {/* Learning rate chart + Model versions */}
      <div className="grid grid-cols-[1fr_260px] gap-1 flex-shrink-0 h-36">
        <DashboardPanel title="LEARNING RATE — 30 DAY TREND" className="overflow-hidden">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={learningRate} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
              <XAxis dataKey="day" tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false}
                interval={6} tickFormatter={(v: string) => v.slice(5)} />
              <YAxis tick={{ fill: "#484f58", fontSize: 8 }} tickLine={false} axisLine={false} />
              <Tooltip
                contentStyle={{ backgroundColor: "#0d1117", border: "1px solid #1e2530", borderRadius: "4px", fontSize: 9 }}
                formatter={(v: number) => [`${v} events/day`]}
              />
              <Bar dataKey="rate" fill="#00d4aa" fillOpacity={0.6} radius={[1, 1, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </DashboardPanel>

        <DashboardPanel title="COMPONENT VERSIONS" noPadding>
          <div className="overflow-auto h-full">
            {modelVersions.map(mv => (
              <div key={mv.component} className="flex items-start gap-2 px-2 py-1.5 border-b border-terminal-border/20 hover:bg-white/[0.02]">
                <span className="text-terminal-accent font-mono text-[9px] font-semibold flex-shrink-0 w-14">{mv.version}</span>
                <span className="text-terminal-text-faint text-[8px] font-mono flex-shrink-0 w-20">{mv.component}</span>
                <span className="text-terminal-text-muted text-[9px] leading-relaxed">{mv.notes}</span>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>

      {/* Recent learnings */}
      <DashboardPanel title="RECENT LEARNING EVENTS" className="flex-1 overflow-hidden" noPadding>
        <div className="overflow-auto h-full">
          <table className="w-full text-[10px] font-mono">
            <thead className="sticky top-0 bg-terminal-surface z-10">
              <tr className="border-b border-terminal-border text-terminal-text-faint text-[9px] uppercase tracking-wider">
                <th className="py-1.5 px-2 text-left w-20">Time</th>
                <th className="py-1.5 px-2 text-left w-24">Type</th>
                <th className="py-1.5 px-2 text-left w-28">Agent</th>
                <th className="py-1.5 px-2 text-left">What Was Learned</th>
                <th className="py-1.5 px-2 text-right w-24">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {learnings.map(l => {
                const typeColor = LEARNING_TYPE_COLORS[l.type] || "#8b949e";
                const confColor = l.confidence >= 0.85 ? "#3fb950" : l.confidence >= 0.70 ? "#d29922" : "#f85149";
                return (
                  <tr key={l.id} className="border-b border-terminal-border/20 hover:bg-white/[0.02] transition-colors">
                    <td className="py-1.5 px-2 text-terminal-text-faint">{l.timestamp.slice(11, 19) || "—"}</td>
                    <td className="py-1.5 px-2">
                      <span className="px-1.5 py-0.5 rounded text-[8px] font-semibold"
                        style={{ color: typeColor, backgroundColor: `${typeColor}15`, border: `1px solid ${typeColor}35` }}>
                        {l.type.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-1.5 px-2 text-terminal-text-muted truncate max-w-[120px]">{l.agent}</td>
                    <td className="py-1.5 px-2 text-terminal-text-muted leading-relaxed">{l.what}</td>
                    <td className="py-1.5 px-2 text-right">
                      <div className="flex items-center justify-end gap-1.5">
                        <div className="w-10 h-1 bg-terminal-bg rounded-full overflow-hidden">
                          <div className="h-full rounded-full" style={{ width: `${l.confidence * 100}%`, backgroundColor: confColor }} />
                        </div>
                        <span className="font-semibold tabular-nums" style={{ color: confColor }}>
                          {(l.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
              {learnings.length === 0 && (
                <tr><td colSpan={5} className="py-4 text-center text-terminal-text-faint">Awaiting learning events...</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </DashboardPanel>
    </div>
  );
}

// ═══════════ MAIN PAGE ═══════════

type AgentSubTab = "REGISTRY" | "SKILLS" | "LEARNING";

export default function AgentsPage() {
  // ─── Live API data ───────────────────────────────────
  const { data: leaderboardApi } = useEngineQuery<{ agents: AgentRecord[]; total: number }>(
    "/agents/leaderboard?top_n=25", { refetchInterval: 15000 }
  );
  const { data: statsApi } = useEngineQuery<{ stats: SystemStats }>(
    "/agents/system-stats", { refetchInterval: 15000 }
  );
  const { data: consensusApi } = useEngineQuery<{ summary: ConsensusSummary }>(
    "/agents/consensus", { refetchInterval: 10000 }
  );
  const { data: accuracyApi } = useEngineQuery<{ trend: AccuracyDay[] }>(
    "/agents/accuracy-trend?days=30", { refetchInterval: 60000 }
  );
  const { data: skillsApi } = useEngineQuery<{ skills: SkillRecord[] }>(
    "/agents/skills", { refetchInterval: 60000 }
  );
  const { data: learningsApi } = useEngineQuery<{ learnings: LearningEvent[] }>(
    "/agents/learnings?limit=50", { refetchInterval: 15000 }
  );
  const { data: learningRateApi } = useEngineQuery<{ trend: LearningRateDay[] }>(
    "/agents/learning-rate?days=30", { refetchInterval: 60000 }
  );
  const { data: versionsApi } = useEngineQuery<{ versions: ModelVersion[] }>(
    "/agents/model-versions", { refetchInterval: 120000 }
  );
  const { data: orchestratorApi } = useEngineQuery<{ status: OrchestratorStatus }>(
    "/agents/orchestrator-status", { refetchInterval: 15000 }
  );
  const { data: enforcementApi } = useEngineQuery<EnforcementState>(
    "/agents/enforcement", { refetchInterval: 15000 }
  );

  // Resolved data
  const agents = leaderboardApi?.agents ?? [];
  const stats = statsApi?.stats ?? null;
  const consensus = consensusApi?.summary ?? null;
  const accuracyTrend = accuracyApi?.trend ?? [];
  const skills = skillsApi?.skills ?? [];
  const learnings = learningsApi?.learnings ?? [];
  const learningRate = learningRateApi?.trend ?? [];
  const modelVersions = versionsApi?.versions ?? [];
  const orchStatus = orchestratorApi?.status ?? null;
  const enforcement = enforcementApi ?? null;

  const [subTab, setSubTab] = useState<AgentSubTab>("REGISTRY");

  return (
    <div className="h-full flex flex-col gap-1 p-1 overflow-hidden">
      {/* Summary Row */}
      <div className="flex-shrink-0">
        <AgentSummaryRow stats={stats} consensus={consensus} />
      </div>

      {/* Sub-tabs */}
      <div className="flex items-center gap-1 flex-shrink-0 border-b border-terminal-border/50 pb-1">
        {(["REGISTRY", "SKILLS", "LEARNING"] as AgentSubTab[]).map(tab => (
          <button
            key={tab}
            onClick={() => setSubTab(tab)}
            className={`px-3 py-1 text-[10px] font-mono font-semibold tracking-wider rounded-t transition-colors ${
              subTab === tab
                ? "bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30 border-b-0"
                : "text-terminal-text-faint hover:text-terminal-text-muted border border-transparent"
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {subTab === "REGISTRY" && (
        <div className="flex-1 flex flex-col gap-1 overflow-hidden">
          {/* Consensus + Accuracy + Tier + Orchestrator row */}
          <div className="flex-shrink-0">
            <SignalConsensus consensus={consensus} enforcement={enforcement} />
          </div>
          <div className="grid grid-cols-3 gap-1 flex-shrink-0">
            <AccuracyTrend data={accuracyTrend} />
            <TierDistribution stats={stats} />
            <OrchestratorPanel status={orchStatus} />
          </div>
          {/* Main registry table */}
          <RegistryTab agents={agents} />
          {/* Enforcement at bottom */}
          <div className="flex-shrink-0">
            <EnforcementPanel data={enforcement} />
          </div>
        </div>
      )}
      {subTab === "SKILLS" && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <SkillsTab skills={skills} />
        </div>
      )}
      {subTab === "LEARNING" && (
        <div className="flex-1 flex flex-col overflow-hidden">
          <LearningTab
            learnings={learnings}
            learningRate={learningRate}
            modelVersions={modelVersions}
            stats={stats}
          />
        </div>
      )}
    </div>
  );
}
