import { DashboardPanel } from "@/components/dashboard-panel";
import { useRef, useEffect, useCallback, useMemo } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";

// ─── Types ─────────────────────────────────────────────────
interface PipelineStage {
  id: string; label: string; type: "input" | "process" | "output" | "decision"; desc: string;
}

interface StrategyConfig {
  current_regime: string;
  regime_confidence: number;
  max_leverage: number;
  beta_cap: number;
  target_beta: number;
  liquidity: number;
  risk: number;
  flow: number;
  sleeve_allocation: Record<string, number>;
  cache_timestamp: string | null;
  pipeline_stages: PipelineStage[];
  regime_params: Record<string, Record<string, number>>;
  timestamp: string;
}

interface StrategyPerf {
  strategies: Array<{ name: string; pnl: number; trades: number; sharpe: number; status: string }>;
  perf_cards: Array<{ name: string; value: string }>;
}

interface VolSurface {
  atm_1m: number; atm_3m: number; atm_6m: number; atm_1y: number;
  skew_25d_1m: number; skew_25d_3m: number; term_spread: number;
  anomalies: string[];
  surface: Record<string, Record<string, number>>;
}

interface StatArbData {
  n_pairs: number; active_trades: number; portfolio_beta: number;
  pairs: Array<{ pair: string; ticker_a: string; ticker_b: string; zscore: number; half_life: number; status: string }>;
  signals: Array<Record<string, unknown>>;
}

interface MLEnsembleData {
  tier_weights: Record<string, number>;
  recent_votes: Record<string, { score: number; signal: string; timestamp: string }>;
  n_tickers_voted: number;
}

interface DecisionMatrixData {
  regime: string; max_leverage: number;
  gates: Array<{ gate: string; weight: number; threshold: number }>;
  approved: number; rejected: number;
}

interface StrategySignals {
  vol_surface: VolSurface;
  stat_arb: StatArbData;
  ml_ensemble: MLEnsembleData;
  decision_matrix: DecisionMatrixData;
  regime_context: Record<string, unknown>;
  timestamp: string;
}

// ─── Constants ─────────────────────────────────────────────
const REGIME_COLORS: Record<string, string> = {
  TRENDING: "#3fb950", RANGE: "#58a6ff", STRESS: "#d29922", CRASH: "#f85149",
};

const TIER_LABELS: Record<string, string> = {
  T1_neural: "Neural Net", T2_momentum: "Momentum/MR", T3_vol_regime: "Vol Regime",
  T4_monte_carlo: "Monte Carlo", T5_quality: "Quality", T6_social: "Social",
  T7_distress: "Distress", T8_event: "Event-Driven", T9_cvr: "CVR", T10_credit_quality: "Credit",
};

// ─── Flow Diagram (driven by pipeline_stages from API) ─────
function FlowDiagram({ stages }: { stages: PipelineStage[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container || !stages.length) return;

    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const cols = Math.min(stages.length, 4);
    const rows = Math.ceil(stages.length / cols);
    const cellW = rect.width / cols;
    const cellH = rect.height / rows;
    const nodeW = Math.min(100, cellW * 0.7);
    const nodeH = 26;

    const colors: Record<string, { bg: string; border: string }> = {
      input:    { bg: "rgba(0, 212, 170, 0.12)", border: "rgba(0, 212, 170, 0.4)" },
      process:  { bg: "rgba(88, 166, 255, 0.12)", border: "rgba(88, 166, 255, 0.4)" },
      output:   { bg: "rgba(63, 185, 80, 0.12)",  border: "rgba(63, 185, 80, 0.4)" },
      decision: { bg: "rgba(188, 140, 255, 0.12)", border: "rgba(188, 140, 255, 0.4)" },
    };

    const positions = stages.map((_, i) => ({
      x: cellW * (i % cols) + cellW / 2,
      y: cellH * Math.floor(i / cols) + cellH / 2,
    }));

    // Edges
    for (let i = 0; i < stages.length - 1; i++) {
      const from = positions[i];
      const to = positions[i + 1];
      ctx.beginPath();
      ctx.moveTo(from.x + nodeW / 2, from.y);
      ctx.lineTo(to.x - nodeW / 2, to.y);
      ctx.strokeStyle = "rgba(0, 212, 170, 0.2)";
      ctx.lineWidth = 1.5;
      ctx.stroke();
      const angle = Math.atan2(to.y - from.y, to.x - from.x);
      const ax = to.x - nodeW / 2;
      const ay = to.y;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(ax - 5 * Math.cos(angle - 0.4), ay - 5 * Math.sin(angle - 0.4));
      ctx.lineTo(ax - 5 * Math.cos(angle + 0.4), ay - 5 * Math.sin(angle + 0.4));
      ctx.closePath();
      ctx.fillStyle = "rgba(0, 212, 170, 0.3)";
      ctx.fill();
    }

    // Nodes
    positions.forEach((pos, i) => {
      const stage = stages[i];
      const c = colors[stage.type] || colors.process;
      const grad = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 40);
      grad.addColorStop(0, c.bg);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, 40, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
      ctx.beginPath();
      ctx.roundRect(pos.x - nodeW / 2, pos.y - nodeH / 2, nodeW, nodeH, 3);
      ctx.fillStyle = c.bg;
      ctx.fill();
      ctx.strokeStyle = c.border;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.font = "8px 'JetBrains Mono'";
      ctx.fillStyle = "rgba(230, 237, 243, 0.8)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(stage.label, pos.x, pos.y);
    });
  }, [stages]);

  useEffect(() => {
    draw();
    const h = () => draw();
    window.addEventListener("resize", h);
    return () => window.removeEventListener("resize", h);
  }, [draw]);

  return (
    <div ref={containerRef} className="w-full h-full min-h-[120px]">
      <canvas ref={canvasRef} />
    </div>
  );
}

// ─── Vol Surface Panel ─────────────────────────────────────
function VolSurfacePanel({ data }: { data?: VolSurface }) {
  if (!data || data.atm_1m === undefined) {
    return (
      <DashboardPanel title="VOL SURFACE" className="h-full">
        <div className="text-[9px] text-terminal-text-faint font-mono text-center py-3">Connecting to OptionsEngine…</div>
      </DashboardPanel>
    );
  }

  const termColor = data.term_spread < 0 ? "text-terminal-negative" : "text-terminal-positive";

  return (
    <DashboardPanel title="VOL SURFACE" noPadding className="h-full">
      <div className="px-2 py-1.5 space-y-1.5 text-[9px] font-mono">
        {/* ATM term structure */}
        <div className="space-y-0.5">
          <div className="text-terminal-text-faint uppercase tracking-wider text-[8px]">ATM Implied Vol</div>
          <div className="grid grid-cols-4 gap-1">
            {[
              { label: "1M", val: data.atm_1m },
              { label: "3M", val: data.atm_3m },
              { label: "6M", val: data.atm_6m },
              { label: "1Y", val: data.atm_1y },
            ].map((t) => (
              <div key={t.label} className="text-center p-1 rounded bg-terminal-surface-2/50 border border-terminal-border/20">
                <div className="text-terminal-text-faint">{t.label}</div>
                <div className="text-terminal-accent tabular-nums">{(t.val * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>
        {/* Skew + Term */}
        <div className="grid grid-cols-3 gap-1 pt-1 border-t border-terminal-border/20">
          <div className="text-center">
            <div className="text-terminal-text-faint">Skew 1M</div>
            <div className="text-terminal-purple tabular-nums">{(data.skew_25d_1m * 100).toFixed(1)}%</div>
          </div>
          <div className="text-center">
            <div className="text-terminal-text-faint">Skew 3M</div>
            <div className="text-terminal-purple tabular-nums">{(data.skew_25d_3m * 100).toFixed(1)}%</div>
          </div>
          <div className="text-center">
            <div className="text-terminal-text-faint">Term Spd</div>
            <div className={`tabular-nums ${termColor}`}>{(data.term_spread * 100).toFixed(1)}%</div>
          </div>
        </div>
        {/* Anomalies */}
        {data.anomalies?.length > 0 && (
          <div className="pt-1 border-t border-terminal-border/20">
            <div className="text-terminal-warning text-[8px]">ANOMALIES</div>
            {data.anomalies.map((a, i) => (
              <div key={i} className="text-terminal-warning">{a.replace(/_/g, " ")}</div>
            ))}
          </div>
        )}
      </div>
    </DashboardPanel>
  );
}

// ─── Stat Arb / Mean Reversion Panel ───────────────────────
function StatArbPanel({ data }: { data?: StatArbData }) {
  if (!data || data.n_pairs === undefined) {
    return (
      <DashboardPanel title="STAT ARB / MEAN REVERSION" className="h-full">
        <div className="text-[9px] text-terminal-text-faint font-mono text-center py-3">Connecting to StatArbEngine…</div>
      </DashboardPanel>
    );
  }

  return (
    <DashboardPanel title="STAT ARB / MEAN REVERSION" noPadding className="h-full">
      <div className="px-2 py-1.5 space-y-1.5 text-[9px] font-mono overflow-auto h-full">
        {/* Summary */}
        <div className="grid grid-cols-3 gap-1">
          <div className="text-center p-1 rounded bg-terminal-accent/5 border border-terminal-accent/10">
            <div className="text-terminal-text-faint">Pairs</div>
            <div className="text-terminal-accent tabular-nums">{data.n_pairs}</div>
          </div>
          <div className="text-center p-1 rounded bg-terminal-blue/5 border border-terminal-blue/10">
            <div className="text-terminal-text-faint">Active</div>
            <div className="text-terminal-blue tabular-nums">{data.active_trades}</div>
          </div>
          <div className="text-center p-1 rounded bg-terminal-purple/5 border border-terminal-purple/10">
            <div className="text-terminal-text-faint">Net Beta</div>
            <div className="text-terminal-purple tabular-nums">{data.portfolio_beta?.toFixed(4) ?? "0"}</div>
          </div>
        </div>
        {/* Pairs list */}
        {data.pairs?.length > 0 && (
          <div className="pt-1 border-t border-terminal-border/20 space-y-0.5">
            <div className="text-terminal-text-faint uppercase tracking-wider text-[8px]">Cointegrated Pairs</div>
            {data.pairs.slice(0, 8).map((p, i) => {
              const zColor = Math.abs(p.zscore) > 2 ? "text-terminal-warning" : Math.abs(p.zscore) > 1 ? "text-terminal-blue" : "text-terminal-text-muted";
              return (
                <div key={i} className="flex justify-between items-center">
                  <span className="text-terminal-text-primary truncate max-w-[80px]">{p.pair}</span>
                  <div className="flex gap-2 items-center">
                    <span className={`tabular-nums ${zColor}`}>z:{p.zscore.toFixed(2)}</span>
                    <span className="text-terminal-text-faint tabular-nums">HL:{p.half_life}d</span>
                    <span className={`w-1 h-1 rounded-full ${
                      p.status === "ACTIVE" ? "bg-terminal-positive" : "bg-terminal-text-faint"
                    }`} />
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </DashboardPanel>
  );
}

// ─── ML Ensemble Panel ─────────────────────────────────────
function MLEnsemblePanel({ data }: { data?: MLEnsembleData }) {
  if (!data || !data.tier_weights) {
    return (
      <DashboardPanel title="ML ENSEMBLE (10-TIER)" className="h-full">
        <div className="text-[9px] text-terminal-text-faint font-mono text-center py-3">Connecting to MLVoteEnsemble…</div>
      </DashboardPanel>
    );
  }

  const tiers = Object.entries(data.tier_weights || {});

  return (
    <DashboardPanel title="ML ENSEMBLE (10-TIER)" noPadding className="h-full">
      <div className="px-2 py-1.5 space-y-1.5 text-[9px] font-mono overflow-auto h-full">
        {/* Tier weights */}
        <div className="space-y-0.5">
          <div className="text-terminal-text-faint uppercase tracking-wider text-[8px]">Tier Weights</div>
          {tiers.map(([tier, weight]) => (
            <div key={tier} className="flex justify-between items-center">
              <span className="text-terminal-text-primary truncate max-w-[100px]">{TIER_LABELS[tier] || tier}</span>
              <div className="flex items-center gap-1.5">
                <div className="w-10 h-1 bg-terminal-border/20 rounded-full overflow-hidden">
                  <div className="h-full bg-terminal-accent/60 rounded-full" style={{ width: `${(weight / 1.2) * 100}%` }} />
                </div>
                <span className="text-terminal-accent tabular-nums w-6 text-right">{weight.toFixed(1)}</span>
              </div>
            </div>
          ))}
        </div>
        {/* Recent votes */}
        {data.recent_votes && Object.keys(data.recent_votes).length > 0 && (
          <div className="pt-1 border-t border-terminal-border/20 space-y-0.5">
            <div className="text-terminal-text-faint uppercase tracking-wider text-[8px]">Recent Votes ({data.n_tickers_voted} tickers)</div>
            {Object.entries(data.recent_votes).slice(0, 6).map(([ticker, v]) => {
              const sigColor = v.signal.includes("BUY") ? "text-terminal-positive" : v.signal.includes("SELL") ? "text-terminal-negative" : "text-terminal-text-muted";
              return (
                <div key={ticker} className="flex justify-between items-center">
                  <span className="text-terminal-text-primary">{ticker}</span>
                  <div className="flex gap-2 items-center">
                    <span className="text-terminal-blue tabular-nums">{v.score.toFixed(1)}</span>
                    <span className={`${sigColor} font-medium`}>{v.signal}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </DashboardPanel>
  );
}

// ─── Decision Matrix Panel ─────────────────────────────────
function DecisionMatrixPanel({ data }: { data?: DecisionMatrixData }) {
  if (!data || !data.gates) {
    return (
      <DashboardPanel title="DECISION MATRIX" className="h-full">
        <div className="text-[9px] text-terminal-text-faint font-mono text-center py-3">Connecting to DecisionMatrix…</div>
      </DashboardPanel>
    );
  }

  return (
    <DashboardPanel title="DECISION MATRIX (6-GATE)" noPadding className="h-full">
      <div className="px-2 py-1.5 space-y-1.5 text-[9px] font-mono overflow-auto h-full">
        {/* Gate configs */}
        <div className="space-y-0.5">
          <div className="text-terminal-text-faint uppercase tracking-wider text-[8px]">Gate Configuration — {data.regime}</div>
          {data.gates.map((g) => (
            <div key={g.gate} className="flex justify-between items-center">
              <span className="text-terminal-text-primary truncate max-w-[120px]">{g.gate.replace(/_/g, " ")}</span>
              <div className="flex gap-2 items-center">
                <span className="text-terminal-blue tabular-nums">W:{g.weight.toFixed(1)}</span>
                <span className="text-terminal-text-faint tabular-nums">T:{g.threshold.toFixed(2)}</span>
              </div>
            </div>
          ))}
        </div>
        {/* Execution stats */}
        <div className="pt-1 border-t border-terminal-border/20 grid grid-cols-2 gap-1">
          <div className="text-center p-1 rounded bg-terminal-positive/5 border border-terminal-positive/10">
            <div className="text-terminal-text-faint">Approved</div>
            <div className="text-terminal-positive tabular-nums">{data.approved}</div>
          </div>
          <div className="text-center p-1 rounded bg-terminal-negative/5 border border-terminal-negative/10">
            <div className="text-terminal-text-faint">Rejected</div>
            <div className="text-terminal-negative tabular-nums">{data.rejected}</div>
          </div>
        </div>
      </div>
    </DashboardPanel>
  );
}

// ─── Regime Indicator ──────────────────────────────────────
function RegimeIndicator({ config }: { config: StrategyConfig | undefined }) {
  if (!config?.current_regime) {
    return (
      <DashboardPanel title="REGIME" className="flex-1">
        <div className="text-xs text-terminal-text-faint font-mono text-center py-2">Waiting for cube state…</div>
      </DashboardPanel>
    );
  }

  const color = REGIME_COLORS[config.current_regime] || "#58a6ff";
  const conf = typeof config.regime_confidence === "number" ? (config.regime_confidence * 100).toFixed(0) : "—";

  return (
    <DashboardPanel title="REGIME" className="flex-1">
      <div className="flex flex-col items-center gap-0.5">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full animate-pulse" style={{ backgroundColor: color }} />
          <span className="text-lg font-mono font-bold" style={{ color }}>{config.current_regime}</span>
        </div>
        <div className="text-[9px] font-mono text-terminal-text-faint">Conf: {conf}% | Lev: {config.max_leverage?.toFixed(1)}x | β: {config.beta_cap?.toFixed(2)}</div>
      </div>
    </DashboardPanel>
  );
}

// ─── Cube Metrics Strip ────────────────────────────────────
function CubeMetrics({ config }: { config: StrategyConfig | undefined }) {
  if (!config) return null;
  return (
    <DashboardPanel title="CUBE STATE" className="flex-1">
      <div className="flex gap-3 justify-center text-[9px] font-mono">
        <div className="text-center">
          <div className="text-terminal-text-faint">L(t)</div>
          <div className="text-terminal-accent tabular-nums">{typeof config.liquidity === "number" ? config.liquidity.toFixed(3) : "—"}</div>
        </div>
        <div className="text-center">
          <div className="text-terminal-text-faint">R(t)</div>
          <div className="text-terminal-blue tabular-nums">{typeof config.risk === "number" ? config.risk.toFixed(3) : "—"}</div>
        </div>
        <div className="text-center">
          <div className="text-terminal-text-faint">F(t)</div>
          <div className="text-terminal-purple tabular-nums">{typeof config.flow === "number" ? config.flow.toFixed(3) : "—"}</div>
        </div>
        <div className="text-center">
          <div className="text-terminal-text-faint">β Target</div>
          <div className="text-terminal-text-primary tabular-nums">{config.target_beta?.toFixed(4) ?? "—"}</div>
        </div>
      </div>
    </DashboardPanel>
  );
}

// ─── Sleeve Allocation Panel ───────────────────────────────
function SleevePanel({ config }: { config: StrategyConfig | undefined }) {
  if (!config?.sleeve_allocation || Object.keys(config.sleeve_allocation).length === 0) return null;

  return (
    <DashboardPanel title="SLEEVE ALLOCATION" noPadding className="h-full">
      <div className="px-2 py-1.5 space-y-0.5 text-[9px] font-mono overflow-auto h-full">
        {Object.entries(config.sleeve_allocation).map(([k, v]) => (
          <div key={k} className="flex justify-between items-center">
            <span className="text-terminal-text-faint truncate max-w-[100px]">{k.replace(/_/g, " ")}</span>
            <div className="flex items-center gap-1.5">
              <div className="w-12 h-1 bg-terminal-border/20 rounded-full overflow-hidden">
                <div className="h-full bg-terminal-accent/60 rounded-full" style={{ width: `${(typeof v === "number" ? v : 0) * 100}%` }} />
              </div>
              <span className="text-terminal-text-primary tabular-nums w-8 text-right">{typeof v === "number" ? (v * 100).toFixed(0) : "—"}%</span>
            </div>
          </div>
        ))}
      </div>
    </DashboardPanel>
  );
}

// ─── Main Component ────────────────────────────────────────
export default function StrategyBuilder() {
  // ─── Live config from MetadronCube ─────────
  const { data: configApi } = useEngineQuery<StrategyConfig>("/ml/strategy/config", { refetchInterval: 10000 });

  // ─── Live strategy performance from ExecutionEngine ─────────
  const { data: stratApi } = useEngineQuery<StrategyPerf>("/ml/strategy/performance", { refetchInterval: 15000 });

  // ─── Aggregated signals: vol surface, stat arb, ML ensemble, decision matrix ─────────
  const { data: signalsApi } = useEngineQuery<StrategySignals>("/ml/strategy/signals", { refetchInterval: 20000 });

  const strategies = useMemo(() => {
    if (!stratApi?.strategies?.length) return [];
    return stratApi.strategies.map((s) => ({
      name: s.name,
      status: s.status,
      sharpe: s.sharpe,
      pnl: s.pnl > 0 ? `+${s.pnl.toLocaleString()}` : `${s.pnl.toLocaleString()}`,
      trades: s.trades,
    }));
  }, [stratApi]);

  const perfCards = useMemo(() => {
    if (!stratApi?.perf_cards?.length) return [];
    return stratApi.perf_cards.map((c, i) => ({
      label: c.name,
      value: c.value,
      color: ["#00d4aa", "#f85149", "#58a6ff", "#bc8cff"][i % 4],
    }));
  }, [stratApi]);

  const pipelineStages = useMemo(() => configApi?.pipeline_stages ?? [], [configApi]);

  return (
    <div className="h-full flex flex-col gap-[2px] p-[2px] overflow-auto" data-testid="strategy-builder">
      {/* Row 1: Regime + Cube Metrics + Performance Cards */}
      <div className="flex gap-[2px] flex-shrink-0">
        <RegimeIndicator config={configApi} />
        <CubeMetrics config={configApi} />
        {perfCards.map((p, i) => (
          <DashboardPanel key={i} title={p.label} className="flex-1">
            <div className="text-lg font-mono font-bold tabular-nums text-center" style={{ color: p.color }}>{p.value}</div>
          </DashboardPanel>
        ))}
        {perfCards.length === 0 && (
          <DashboardPanel title="PERFORMANCE" className="flex-1">
            <div className="text-xs text-terminal-text-faint font-mono text-center py-1">Waiting for data…</div>
          </DashboardPanel>
        )}
      </div>

      {/* Row 2: Pipeline Flow + Strategy List */}
      <div className="flex gap-[2px] flex-shrink-0" style={{ minHeight: 140 }}>
        <DashboardPanel title={`PIPELINE FLOW — ${configApi?.current_regime ?? "…"}`} noPadding className="flex-1">
          <FlowDiagram stages={pipelineStages} />
        </DashboardPanel>
        <DashboardPanel title="STRATEGIES" noPadding className="w-[220px]">
          <div className="overflow-auto h-full">
            {strategies.length === 0 && (
              <div className="px-2 py-4 text-center text-terminal-text-faint text-[9px] font-mono">Waiting…</div>
            )}
            {strategies.map((s, i) => (
              <div key={i} className="px-2 py-1.5 border-b border-terminal-border/20 hover:bg-white/[0.02] cursor-pointer">
                <div className="flex items-center gap-2">
                  <span className={`w-1.5 h-1.5 rounded-full ${
                    s.status === "active" ? "bg-terminal-positive" :
                    s.status === "testing" ? "bg-terminal-warning" : "bg-terminal-text-faint"
                  }`} />
                  <span className="text-[10px] text-terminal-text-primary font-medium">{s.name}</span>
                </div>
                <div className="flex items-center gap-2 mt-0.5 pl-3.5 text-[8px] font-mono tabular-nums">
                  <span className="text-terminal-text-faint">SR:{s.sharpe}</span>
                  <span className="text-terminal-positive">{s.pnl}</span>
                  <span className="text-terminal-text-faint">{s.trades}t</span>
                </div>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>

      {/* Row 3: Intelligence Grid — Vol Surface | Stat Arb | ML Ensemble | Decision Matrix */}
      <div className="grid grid-cols-4 gap-[2px] flex-1 min-h-[200px]">
        <VolSurfacePanel data={signalsApi?.vol_surface} />
        <StatArbPanel data={signalsApi?.stat_arb} />
        <MLEnsemblePanel data={signalsApi?.ml_ensemble} />
        <DecisionMatrixPanel data={signalsApi?.decision_matrix} />
      </div>

      {/* Row 4: Sleeve Allocation (compact) */}
      <div className="flex gap-[2px] flex-shrink-0" style={{ maxHeight: 100 }}>
        <SleevePanel config={configApi} />
        <DashboardPanel title="ACTIONS" className="w-[200px]">
          <div className="flex flex-col gap-1.5">
            <button className="w-full py-1.5 text-[10px] font-mono font-medium bg-terminal-accent/15 text-terminal-accent border border-terminal-accent/30 rounded hover:bg-terminal-accent/25 transition-colors">
              ▶ RUN BACKTEST
            </button>
            <button className="w-full py-1.5 text-[10px] font-mono font-medium bg-terminal-blue/15 text-terminal-blue border border-terminal-blue/30 rounded hover:bg-terminal-blue/25 transition-colors">
              + NEW STRATEGY
            </button>
          </div>
        </DashboardPanel>
      </div>
    </div>
  );
}
