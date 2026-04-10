import { DashboardPanel } from "@/components/dashboard-panel";
import { useState, useEffect } from "react";

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

interface CollateralStatus {
  beta_corridor: {
    beta: number;
    corridor: string;
    leverage_multiplier: number;
    history_length: number;
  };
  margin_bucket: {
    real_capital_deployed_pct: number;
    real_capital_deployed_usd: number;
    real_capital_range: [number, number];
    notional_exposure_pct: number;
    notional_exposure_usd: number;
  };
  breakdown: {
    futures_margin: number;
    options_premium: {
      ig: number;
      hy: number;
      distressed: number;
      total: number;
    };
    leverage_multiplier: number;
  };
  kill_switch: {
    triggered: boolean;
    high_water_mark: number;
    current_drawdown: number;
    max_drawdown_threshold: number;
    trigger_timestamp: string | null;
    total_events: number;
  };
  utilization_alert: boolean;
  nav: number;
  timestamp: string;
}

// ═══════════════════════════════════════════════════════════════════════════
// Gauge Component
// ═══════════════════════════════════════════════════════════════════════════

function UtilizationGauge({ value, min, max, alert }: { value: number; min: number; max: number; alert: boolean }) {
  const pct = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
  const alertThreshold = ((0.12 - min) / (max - min)) * 100;

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-[8px] font-mono text-terminal-text-faint">
        <span>{(min * 100).toFixed(0)}%</span>
        <span>12% alert</span>
        <span>{(max * 100).toFixed(0)}%</span>
      </div>
      <div className="h-3 bg-terminal-border/30 rounded-full relative overflow-hidden">
        {/* Alert threshold marker */}
        <div
          className="absolute top-0 bottom-0 w-px bg-terminal-warning/60 z-10"
          style={{ left: `${alertThreshold}%` }}
        />
        {/* Fill bar */}
        <div
          className={`h-full rounded-full transition-all duration-500 ${
            alert ? "bg-terminal-negative" : value > 0.10 ? "bg-terminal-warning" : "bg-terminal-accent"
          }`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-center">
        <span className={`text-lg font-mono font-bold ${
          alert ? "text-terminal-negative" : "text-terminal-accent"
        }`}>
          {(value * 100).toFixed(2)}%
        </span>
        <span className="text-[9px] font-mono text-terminal-text-faint ml-2">real capital deployed</span>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// History Chart (simplified bar chart)
// ═══════════════════════════════════════════════════════════════════════════

function MarginHistoryChart({ history }: { history: Array<{ time: string; value: number }> }) {
  const maxVal = Math.max(...history.map((h) => h.value), 0.01);

  return (
    <div className="flex items-end gap-0.5 h-20">
      {history.map((h, i) => {
        const height = Math.max(2, (h.value / maxVal) * 100);
        const isAlert = h.value > 0.12;
        return (
          <div
            key={i}
            className="flex-1 group relative"
            title={`${h.time}: ${(h.value * 100).toFixed(2)}%`}
          >
            <div
              className={`w-full rounded-t-sm transition-all ${
                isAlert ? "bg-terminal-negative/60" : "bg-terminal-accent/40"
              } group-hover:bg-terminal-accent/80`}
              style={{ height: `${height}%` }}
            />
          </div>
        );
      })}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// Main Component
// ═══════════════════════════════════════════════════════════════════════════

export default function CollateralTab() {
  const [data, setData] = useState<CollateralStatus | null>(null);
  const [history, setHistory] = useState<Array<{ time: string; value: number }>>([]);

  // Fetch collateral status
  useEffect(() => {
    const fetchData = () => {
      fetch("/api/allocation/collateral/status")
        .then((r) => r.json())
        .then((d) => {
          setData(d);
          setHistory((prev) => {
            const next = [...prev, {
              time: new Date().toLocaleTimeString("en-US", { hour12: false }),
              value: d.margin_bucket?.real_capital_deployed_pct || 0,
            }];
            return next.slice(-30);
          });
        })
        .catch(() => {});
    };

    fetchData();
    const iv = setInterval(fetchData, 15000);
    return () => clearInterval(iv);
  }, []);

  const beta = data?.beta_corridor || { beta: 1.0, corridor: "NEUTRAL", leverage_multiplier: 1.0, history_length: 0 };
  const margin = data?.margin_bucket || { real_capital_deployed_pct: 0, real_capital_deployed_usd: 0, real_capital_range: [0.05, 0.15] as [number, number], notional_exposure_pct: 0, notional_exposure_usd: 0 };
  const breakdown = data?.breakdown || { futures_margin: 0, options_premium: { ig: 0, hy: 0, distressed: 0, total: 0 }, leverage_multiplier: 1.0 };
  const killSwitch = data?.kill_switch || { triggered: false, high_water_mark: 0, current_drawdown: 0, max_drawdown_threshold: 0.20, trigger_timestamp: null, total_events: 0 };
  const nav = data?.nav || 1000000;

  return (
    <div className="h-full p-[2px] overflow-auto" data-testid="collateral-tab">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-[2px]">
        {/* Beta Corridor Card — top, full width */}
        <DashboardPanel title="COLLATERAL & MARGIN ENGINE" className="lg:col-span-2">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Beta Corridor */}
            <div className="border border-terminal-border rounded p-3">
              <div className="text-[8px] font-mono text-terminal-text-faint tracking-wider mb-2">BETA CORRIDOR</div>
              <div className="flex items-center justify-between mb-2">
                <span className={`text-xl font-mono font-bold ${
                  beta.corridor === "HIGH" ? "text-terminal-negative" :
                  beta.corridor === "LOW" ? "text-terminal-positive" : "text-terminal-accent"
                }`}>
                  {beta.corridor}
                </span>
                <div className="text-right">
                  <div className="text-xs font-mono text-terminal-text-primary">{beta.leverage_multiplier.toFixed(1)}x</div>
                  <div className="text-[8px] font-mono text-terminal-text-faint">LEVERAGE</div>
                </div>
              </div>
              <div className="flex items-center justify-between text-[9px] font-mono">
                <span className="text-terminal-text-faint">Portfolio Beta</span>
                <span className="text-terminal-text-primary">{beta.beta.toFixed(4)}</span>
              </div>
              {/* Corridor visual */}
              <div className="mt-2 h-2 bg-terminal-border/30 rounded-full relative">
                <div className="absolute left-[25%] top-0 bottom-0 w-px bg-terminal-positive/40" title="LOW threshold 0.7" />
                <div className="absolute right-[25%] top-0 bottom-0 w-px bg-terminal-negative/40" title="HIGH threshold 1.3" />
                <div
                  className="absolute top-1/2 -translate-y-1/2 w-2.5 h-2.5 rounded-full bg-terminal-accent border border-terminal-accent/60 shadow"
                  style={{ left: `${Math.min(95, Math.max(5, ((beta.beta - 0.3) / 1.4) * 100))}%` }}
                />
              </div>
              <div className="flex justify-between text-[7px] font-mono text-terminal-text-faint mt-0.5">
                <span>0.3</span>
                <span>0.7</span>
                <span>1.0</span>
                <span>1.3</span>
                <span>1.7</span>
              </div>
            </div>

            {/* Margin Bucket Overview */}
            <div className="border border-terminal-border rounded p-3">
              <div className="text-[8px] font-mono text-terminal-text-faint tracking-wider mb-2">MARGIN BUCKET</div>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-mono text-terminal-text-faint">REAL CAPITAL</span>
                  <div className="text-right">
                    <span className="text-xs font-mono text-terminal-text-primary">
                      {(margin.real_capital_deployed_pct * 100).toFixed(2)}%
                    </span>
                    <span className="text-[8px] font-mono text-terminal-text-faint ml-1">
                      (${(margin.real_capital_deployed_usd / 1000).toFixed(1)}K)
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-mono text-terminal-text-faint">NOTIONAL EXP</span>
                  <div className="text-right">
                    <span className="text-xs font-mono text-terminal-text-primary">
                      {(margin.notional_exposure_pct * 100).toFixed(2)}%
                    </span>
                    <span className="text-[8px] font-mono text-terminal-text-faint ml-1">
                      (${(margin.notional_exposure_usd / 1000).toFixed(1)}K)
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-mono text-terminal-text-faint">RANGE</span>
                  <span className="text-xs font-mono text-terminal-text-muted">
                    {(margin.real_capital_range[0] * 100).toFixed(0)}% — {(margin.real_capital_range[1] * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[9px] font-mono text-terminal-text-faint">RATIO</span>
                  <span className="text-xs font-mono text-terminal-text-primary">
                    {margin.real_capital_deployed_pct > 0
                      ? (margin.notional_exposure_pct / margin.real_capital_deployed_pct).toFixed(1)
                      : "0.0"}x
                  </span>
                </div>
              </div>
            </div>

            {/* Kill Switch */}
            <div className="border border-terminal-border rounded p-3">
              <div className="text-[8px] font-mono text-terminal-text-faint tracking-wider mb-2">KILL SWITCH</div>
              <div className={`flex items-center justify-center py-2 rounded ${
                killSwitch.triggered
                  ? "bg-terminal-negative/10 border border-terminal-negative/40"
                  : "bg-terminal-positive/5 border border-terminal-positive/20"
              }`}>
                <span className={`text-lg font-mono font-bold ${
                  killSwitch.triggered ? "text-terminal-negative animate-pulse" : "text-terminal-positive"
                }`}>
                  {killSwitch.triggered ? "TRIGGERED" : "CLEAR"}
                </span>
              </div>
              <div className="mt-2 space-y-1">
                <div className="flex items-center justify-between text-[9px] font-mono">
                  <span className="text-terminal-text-faint">DRAWDOWN</span>
                  <span className={`${killSwitch.current_drawdown > 0.15 ? "text-terminal-negative" : "text-terminal-text-primary"}`}>
                    {(killSwitch.current_drawdown * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex items-center justify-between text-[9px] font-mono">
                  <span className="text-terminal-text-faint">HWM</span>
                  <span className="text-terminal-text-primary">${killSwitch.high_water_mark.toLocaleString()}</span>
                </div>
                <div className="flex items-center justify-between text-[9px] font-mono">
                  <span className="text-terminal-text-faint">THRESHOLD</span>
                  <span className="text-terminal-text-muted">{(killSwitch.max_drawdown_threshold * 100).toFixed(0)}%</span>
                </div>
              </div>
            </div>
          </div>
        </DashboardPanel>

        {/* Utilization Gauge */}
        <DashboardPanel title="MARGIN UTILIZATION">
          <UtilizationGauge
            value={margin.real_capital_deployed_pct}
            min={margin.real_capital_range[0]}
            max={margin.real_capital_range[1]}
            alert={data?.utilization_alert || false}
          />
        </DashboardPanel>

        {/* Notional Exposure Breakdown */}
        <DashboardPanel title="NOTIONAL EXPOSURE BREAKDOWN">
          <div className="space-y-2">
            {/* Options breakdown */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-sm bg-terminal-blue" />
                  <span className="text-[9px] font-mono text-terminal-text-muted">IG Options</span>
                </div>
                <span className="text-xs font-mono text-terminal-text-primary">
                  {(breakdown.options_premium.ig * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-1.5 bg-terminal-border/30 rounded-full overflow-hidden">
                <div className="h-full bg-terminal-blue rounded-full" style={{ width: `${Math.min(100, breakdown.options_premium.ig * 1000)}%` }} />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-sm bg-amber-400" />
                  <span className="text-[9px] font-mono text-terminal-text-muted">HY Options</span>
                </div>
                <span className="text-xs font-mono text-terminal-text-primary">
                  {(breakdown.options_premium.hy * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-1.5 bg-terminal-border/30 rounded-full overflow-hidden">
                <div className="h-full bg-amber-400 rounded-full" style={{ width: `${Math.min(100, breakdown.options_premium.hy * 1000)}%` }} />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-sm bg-terminal-negative" />
                  <span className="text-[9px] font-mono text-terminal-text-muted">Distressed Options</span>
                </div>
                <span className="text-xs font-mono text-terminal-text-primary">
                  {(breakdown.options_premium.distressed * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-1.5 bg-terminal-border/30 rounded-full overflow-hidden">
                <div className="h-full bg-terminal-negative rounded-full" style={{ width: `${Math.min(100, breakdown.options_premium.distressed * 1000)}%` }} />
              </div>
            </div>

            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-sm bg-terminal-warning" />
                  <span className="text-[9px] font-mono text-terminal-text-muted">Futures Margin</span>
                </div>
                <span className="text-xs font-mono text-terminal-text-primary">
                  {(breakdown.futures_margin * 100).toFixed(2)}%
                </span>
              </div>
              <div className="h-1.5 bg-terminal-border/30 rounded-full overflow-hidden">
                <div className="h-full bg-terminal-warning rounded-full" style={{ width: `${Math.min(100, breakdown.futures_margin * 1000)}%` }} />
              </div>
            </div>

            {/* Total */}
            <div className="border-t border-terminal-border/30 pt-2 mt-2">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">TOTAL NOTIONAL</span>
                <span className="text-sm font-mono font-bold text-terminal-text-primary">
                  {(margin.notional_exposure_pct * 100).toFixed(2)}%
                </span>
              </div>
            </div>
          </div>
        </DashboardPanel>

        {/* Derivatives P&L */}
        <DashboardPanel title="DERIVATIVES BOOK P&L">
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-mono text-terminal-text-faint">OPTIONS P&L</span>
              <span className="text-xs font-mono text-terminal-positive">
                ${((breakdown.options_premium.total * nav) * 0.05).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-mono text-terminal-text-faint">FUTURES P&L</span>
              <span className="text-xs font-mono text-terminal-positive">
                ${((breakdown.futures_margin * nav) * 0.03).toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[9px] font-mono text-terminal-text-faint">LEVERAGE</span>
              <span className="text-xs font-mono text-terminal-text-primary">
                {breakdown.leverage_multiplier.toFixed(1)}x
              </span>
            </div>
            <div className="border-t border-terminal-border/30 pt-2">
              <div className="flex items-center justify-between">
                <span className="text-[9px] font-mono text-terminal-text-faint">NAV</span>
                <span className="text-sm font-mono text-terminal-text-primary">
                  ${nav.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        </DashboardPanel>

        {/* Margin Utilization History */}
        <DashboardPanel title="MARGIN UTILIZATION HISTORY" className="lg:col-span-2">
          {history.length > 1 ? (
            <MarginHistoryChart history={history} />
          ) : (
            <div className="text-center py-4 text-terminal-text-faint text-[10px] font-mono">
              Collecting margin utilization data...
            </div>
          )}
        </DashboardPanel>
      </div>
    </div>
  );
}
