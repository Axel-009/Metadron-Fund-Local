import { DashboardPanel } from "@/components/dashboard-panel";

const SECTORS = [
  { name: "Technology", daily: 1.8, weekly: 3.2, monthly: 5.4, color: "#00d4aa" },
  { name: "Healthcare", daily: 0.6, weekly: 1.1, monthly: 2.3, color: "#3fb950" },
  { name: "Financials", daily: -0.3, weekly: 0.8, monthly: 1.9, color: "#58a6ff" },
  { name: "Energy", daily: -1.2, weekly: -2.1, monthly: -3.8, color: "#f85149" },
  { name: "Consumer Disc.", daily: 0.9, weekly: 1.5, monthly: 3.1, color: "#4ecdc4" },
  { name: "Consumer Stap.", daily: 0.2, weekly: 0.4, monthly: 1.0, color: "#d29922" },
  { name: "Industrials", daily: 0.4, weekly: 0.9, monthly: 2.0, color: "#bc8cff" },
  { name: "Materials", daily: -0.5, weekly: -0.8, monthly: -1.2, color: "#f0883e" },
  { name: "Utilities", daily: -0.1, weekly: 0.3, monthly: 0.7, color: "#7d8590" },
  { name: "Real Estate", daily: -0.7, weekly: -1.3, monthly: -2.1, color: "#da3633" },
  { name: "Comm. Svcs.", daily: 1.2, weekly: 2.4, monthly: 4.1, color: "#58a6ff" },
];

const SCENARIOS = [
  { label: "BASE", prob: "60%", desc: "Soft landing, gradual disinflation. Fed cuts 2x in 2026. SPX +8-12% YoY.", color: "#00d4aa" },
  { label: "BULL", prob: "25%", desc: "AI-driven productivity boom. Strong earnings revisions upward. SPX +18-22%.", color: "#3fb950" },
  { label: "BEAR", prob: "15%", desc: "Recession triggered by credit event. Earnings -15%. SPX -20-25%.", color: "#f85149" },
];

const NEWS = [
  { time: "14:32", headline: "Fed minutes show split on rate path, dovish lean", source: "FOMC" },
  { time: "13:45", headline: "AAPL announces $110B buyback, largest in history", source: "Earnings" },
  { time: "12:20", headline: "China PMI below 50 for third consecutive month", source: "Macro" },
  { time: "11:05", headline: "10Y Treasury yield breaks above 4.35% resistance", source: "Rates" },
  { time: "10:15", headline: "NVDA guidance beats by 12%, AI capex acceleration", source: "Earnings" },
  { time: "09:30", headline: "Initial claims at 210K, labor market remains tight", source: "Labor" },
];

const FED_EVENTS = [
  { date: "Apr 15", event: "Fed Waller speech on inflation outlook" },
  { date: "Apr 22", event: "Beige Book release" },
  { date: "May 1", event: "FOMC Decision — rate hold expected" },
  { date: "May 3", event: "NFP April report" },
  { date: "May 14", event: "CPI April release" },
];

const EARNINGS = [
  { date: "Apr 8", ticker: "TSLA", type: "Earnings", est: "$0.45" },
  { date: "Apr 10", ticker: "JPM", type: "Earnings", est: "$4.12" },
  { date: "Apr 12", ticker: "UNH", type: "Earnings", est: "$6.75" },
  { date: "Apr 14", ticker: "MSFT", type: "Ex-Div", est: "$0.75" },
  { date: "Apr 15", ticker: "BAC", type: "Earnings", est: "$0.82" },
];

export default function MarketWrap() {
  return (
    <div className="h-full grid grid-cols-3 grid-rows-[auto_1fr_auto] gap-[2px] p-[2px] overflow-auto" data-testid="market-wrap">
      {/* Market Direction Narrative */}
      <DashboardPanel title="MARKET DIRECTION" className="col-span-2">
        <div className="space-y-2 text-[11px] text-terminal-text-muted leading-relaxed">
          <p>
            Markets are in a <span className="text-terminal-accent font-medium">risk-on</span> regime with strong breadth improvement.
            Tech continues to lead driven by AI infrastructure spending. The S&P 500 made new all-time highs with expanding participation
            beyond megacap tech. Bond yields remain elevated but stable. VIX at 14.2 suggests complacency — monitor for mean reversion.
          </p>
          <p>
            Key thesis: The <span className="text-terminal-positive">soft landing</span> narrative is gaining credibility as inflation
            decelerates without meaningful labor market deterioration. Fed messaging has shifted dovish at the margin.
            <span className="text-terminal-warning"> Energy sector</span> weakness driven by China demand concerns and inventory builds.
          </p>
        </div>
      </DashboardPanel>

      {/* Scenario Thesis */}
      <DashboardPanel title="SCENARIO THESIS">
        <div className="space-y-2">
          {SCENARIOS.map((s) => (
            <div key={s.label} className="border border-terminal-border rounded p-2">
              <div className="flex items-center gap-2 mb-1">
                <span className="text-[10px] font-mono font-bold" style={{ color: s.color }}>{s.label}</span>
                <span className="text-[9px] text-terminal-text-faint font-mono">{s.prob}</span>
              </div>
              <p className="text-[9px] text-terminal-text-muted leading-relaxed">{s.desc}</p>
            </div>
          ))}
        </div>
      </DashboardPanel>

      {/* GICS Sector Heatmap */}
      <DashboardPanel title="GICS SECTOR HEATMAP" className="col-span-2">
        <div className="grid grid-cols-4 gap-1.5">
          {SECTORS.map((s) => {
            const intensity = Math.min(Math.abs(s.daily) / 2, 1);
            const bg = s.daily >= 0
              ? `rgba(0, 212, 170, ${0.08 + intensity * 0.2})`
              : `rgba(248, 81, 73, ${0.08 + intensity * 0.2})`;
            return (
              <div key={s.name} className="rounded p-2 border border-terminal-border/50" style={{ background: bg }}>
                <div className="text-[9px] text-terminal-text-muted font-medium truncate">{s.name}</div>
                <div className="flex items-baseline gap-2 mt-0.5">
                  <span className={`text-sm font-mono font-bold tabular-nums ${s.daily >= 0 ? "text-terminal-positive" : "text-terminal-negative"}`}>
                    {s.daily >= 0 ? "+" : ""}{s.daily.toFixed(1)}%
                  </span>
                  <span className="text-[8px] text-terminal-text-faint font-mono">
                    W: {s.weekly >= 0 ? "+" : ""}{s.weekly.toFixed(1)}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </DashboardPanel>

      {/* Right column: Events */}
      <div className="flex flex-col gap-[2px]">
        {/* News */}
        <DashboardPanel title="CVR NEWS" className="flex-1">
          <div className="space-y-1.5">
            {NEWS.map((n, i) => (
              <div key={i} className="flex gap-2 text-[9px]">
                <span className="text-terminal-text-faint font-mono flex-shrink-0 w-10">{n.time}</span>
                <span className="text-terminal-text-muted flex-1">{n.headline}</span>
                <span className="text-terminal-accent text-[8px] flex-shrink-0">{n.source}</span>
              </div>
            ))}
          </div>
        </DashboardPanel>

        {/* Fed Calendar */}
        <DashboardPanel title="FED CALENDAR" className="flex-1">
          <div className="space-y-1.5">
            {FED_EVENTS.map((e, i) => (
              <div key={i} className="flex gap-2 text-[9px]">
                <span className="text-terminal-warning font-mono flex-shrink-0 w-12">{e.date}</span>
                <span className="text-terminal-text-muted">{e.event}</span>
              </div>
            ))}
          </div>
        </DashboardPanel>
      </div>

      {/* Bottom: Upcoming Earnings/Dividends */}
      <DashboardPanel title="INCOMING EVENTS" className="col-span-3">
        <div className="flex gap-3 overflow-x-auto pb-1">
          {EARNINGS.map((e, i) => (
            <div key={i} className="flex-shrink-0 border border-terminal-border rounded p-2 min-w-[140px]">
              <div className="flex items-center gap-2 text-[9px]">
                <span className="text-terminal-warning font-mono">{e.date}</span>
                <span className={`px-1 py-0.5 rounded text-[7px] font-medium ${
                  e.type === "Earnings" ? "bg-terminal-accent/10 text-terminal-accent" : "bg-terminal-blue/10 text-terminal-blue"
                }`}>{e.type}</span>
              </div>
              <div className="text-sm font-mono font-bold text-terminal-text-primary mt-1">{e.ticker}</div>
              <div className="text-[8px] text-terminal-text-faint font-mono">Est: {e.est}</div>
            </div>
          ))}
        </div>
      </DashboardPanel>
    </div>
  );
}
