import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useEffect, useState, useRef, useCallback } from "react";
import { useEngineQuery } from "@/hooks/use-engine-api";
import { StatusBadge } from "@/components/status-badge";
import NotFound from "@/pages/not-found";
import LiveDashboard from "@/pages/live-dashboard";
import MarketWrap from "@/pages/market-wrap";
import AssetAllocation from "@/pages/asset-allocation";
import RiskPortfolio from "@/pages/risk-portfolio";
import MachineLearning from "@/pages/machine-learning";
import TechDashboard from "@/pages/tech-dashboard";
import Reporting from "@/pages/reporting";
import StrategyBuilder from "@/pages/strategy-builder";
import OpenBBTerminal from "@/pages/openbb-terminal";
import TransactionLog from "@/pages/transaction-log";
import FuturesPage from "@/pages/futures";
import TCAPage from "@/pages/tca";
import AgentsPage from "@/pages/agents";
import QuantToolsPage from "@/pages/quant-tools";
import ReconciliationPage from "@/pages/reconciliation";
import ETFDashboard from "@/pages/etf";
import FixedIncomeDashboard from "@/pages/fixed-income";
import MacroDashboard from "@/pages/macro";
import ArbitrageDashboard from "@/pages/arbitrage";
import MLModelsPage from "@/pages/ml-models";
import MonteCarloPage from "@/pages/monte-carlo";
import SimulationsPage from "@/pages/simulations";
import ArchivePage from "@/pages/archive";
import BacktestingPage from "@/pages/backtesting";
import MetadronCubePage from "@/pages/metadron-cube";
import MoneyVelocityPage from "@/pages/money-velocity";
import ThinkingTab from "@/pages/thinking-tab";
import CollateralTab from "@/pages/collateral-tab";
import ChatTab from "@/pages/chat-tab";
import GraphifyPage from "@/pages/graphify";
import VaultPage from "@/pages/vault";
import SecurityPage from "@/pages/security";
import OpenJarvisPage from "@/pages/open-jarvis";

// ═══════════ TAB GROUPS (7 categories) ═══════════
export const TAB_GROUPS = [
  { group: "CORE", tabs: [
    { path: "/vault", label: "VAULT" },
    { path: "/security", label: "SECURITY" },
    { path: "/live", label: "LIVE" },
    { path: "/market-wrap", label: "WRAP" },
    { path: "/openbb", label: "OPENBB" },
    { path: "/velocity", label: "VELOCITY" },
    { path: "/cube", label: "CUBE" },
  ]},
  { group: "TRANSACTIONS", tabs: [
    { path: "/allocation", label: "ALLOC" },
    { path: "/thinking", label: "THINKING" },
    { path: "/risk", label: "RISK" },
    { path: "/collateral", label: "MARGIN" },
    { path: "/recon", label: "RECON" },
  ]},
  { group: "PRODUCTS", tabs: [
    { path: "/etf", label: "ETF" },
    { path: "/macro", label: "MACRO" },
    { path: "/fixed-income", label: "FIXED INC" },
    { path: "/futures", label: "FUTURES" },
  ]},
  { group: "AGENTS", tabs: [
    { path: "/agents", label: "AGENTS" },
    { path: "/chat", label: "CHAT" },
    { path: "/tech", label: "TECH" },
    { path: "/graphify", label: "GRAPHIFY" },
    { path: "/jarvis", label: "OPEN JARVIS" },
  ]},
  { group: "ANALYSIS", tabs: [
    { path: "/strategy", label: "STRAT" },
    { path: "/quant", label: "QUANT" },
    { path: "/arb", label: "ARB" },
    { path: "/backtesting", label: "BACKTEST" },
  ]},
  { group: "SIMULATION", tabs: [
    { path: "/monte-carlo", label: "MC SIM" },
    { path: "/simulations", label: "SIM" },
    { path: "/ml", label: "ML" },
    { path: "/ml-models", label: "ML MODELS" },
  ]},
  { group: "REPORTING", tabs: [
    { path: "/txlog", label: "TRANSACTION LOG" },
    { path: "/tca", label: "TCA" },
    { path: "/reports", label: "REPORTS" },
    { path: "/archive", label: "ARCHIVE" },
  ]},
];

// Derive flat list for backward compat
export const ALL_TABS = TAB_GROUPS.flatMap((g) => g.tabs);

const DEFAULT_PINNED = [
  "/live", "/cube", "/market-wrap", "/allocation", "/risk",
  "/openbb", "/txlog", "/futures", "/tca", "/agents",
];

const MAX_PINNED = 10;

function MetadronLogo() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-label="Metadron Fund Logo">
      <rect x="1" y="1" width="26" height="26" rx="4" stroke="#00d4aa" strokeWidth="1.5" />
      <path d="M7 20V8l7 8 7-8v12" stroke="#00d4aa" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      <circle cx="14" cy="12" r="2" fill="#00d4aa" fillOpacity="0.3" stroke="#00d4aa" strokeWidth="1" />
    </svg>
  );
}

function LiveMetrics() {
  // Real data: P&L from Alpaca broker, SPY from FMP/OpenBB, LAT from API round-trip
  const { data: portfolio } = useEngineQuery<{ nav: number; total_pnl: number }>("/portfolio/live", { refetchInterval: 15000 });
  const { data: macro } = useEngineQuery<{ spy_return_1d?: number; spy_price?: number }>("/macro/snapshot", { refetchInterval: 30000 });
  const [latency, setLatency] = useState(0);

  // Measure actual API latency
  useEffect(() => {
    const measure = async () => {
      const t0 = performance.now();
      try { await fetch("/api/engine/health"); } catch {}
      setLatency(performance.now() - t0);
    };
    measure();
    const interval = setInterval(measure, 10000);
    return () => clearInterval(interval);
  }, []);

  const pnl = portfolio?.total_pnl ?? 0;
  const spyReturn = macro?.spy_return_1d ?? 0;
  const pnlPositive = pnl >= 0;

  return (
    <div className="flex items-center gap-4 text-[10px] font-mono tabular-nums">
      <StatusBadge status="live" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">LAT</span>
        <span className="text-terminal-text-primary">{latency.toFixed(1)}ms</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">P&L</span>
        <span className={pnlPositive ? "text-terminal-positive" : "text-terminal-negative"}>
          {pnlPositive ? "+" : ""}${pnl.toFixed(2)}
        </span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">SPY</span>
        <span className={spyReturn >= 0 ? "text-terminal-positive" : "text-terminal-negative"}>
          {spyReturn >= 0 ? "+" : ""}{(spyReturn * 100).toFixed(2)}%
        </span>
        <span className={`w-1.5 h-1.5 rounded-full ${spyReturn >= 0 ? "bg-terminal-positive" : "bg-terminal-negative"}`} />
      </div>
    </div>
  );
}

// ═══════════ TAB SELECTOR DROPDOWN ═══════════
function TabSelectorDropdown({
  pinnedPaths,
  onToggle,
  onNavigate,
}: {
  pinnedPaths: string[];
  onToggle: (path: string) => void;
  onNavigate: (path: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  return (
    <div className="relative flex-shrink-0" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className={`flex items-center gap-1 px-2 py-1.5 text-[10px] font-medium tracking-[0.1em] rounded-sm transition-colors ${
          open
            ? "bg-terminal-accent/15 text-terminal-accent"
            : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03]"
        }`}
        data-testid="tab-selector-toggle"
      >
        <svg width="12" height="12" viewBox="0 0 12 12" fill="none">
          <rect x="1" y="2" width="10" height="1.5" rx="0.5" fill="currentColor" />
          <rect x="1" y="5.25" width="10" height="1.5" rx="0.5" fill="currentColor" />
          <rect x="1" y="8.5" width="10" height="1.5" rx="0.5" fill="currentColor" />
        </svg>
        <span>{ALL_TABS.length}</span>
        <svg width="8" height="8" viewBox="0 0 8 8" fill="none" className={`transition-transform ${open ? "rotate-180" : ""}`}>
          <path d="M2 3L4 5L6 3" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
        </svg>
      </button>

      {open && (
        <div className="absolute right-0 top-full mt-1 w-64 bg-terminal-surface border border-terminal-border rounded shadow-xl z-50 overflow-hidden">
          <div className="px-3 py-2 border-b border-terminal-border/50 flex items-center justify-between">
            <span className="text-[9px] text-terminal-text-faint tracking-wider">SELECT TABS ({pinnedPaths.length}/{MAX_PINNED} PINNED)</span>
          </div>
          <div className="max-h-[400px] overflow-y-auto">
            {TAB_GROUPS.map((group, gi) => (
              <div key={group.group}>
                {gi > 0 && <div className="border-t border-terminal-border/30 mx-3" />}
                <div className="px-3 pt-2 pb-1">
                  <span className="text-[8px] font-mono font-medium tracking-[0.15em] uppercase text-[#00d4aa]/50 select-none">
                    {group.group}
                  </span>
                </div>
                {group.tabs.map((tab) => {
                  const isPinned = pinnedPaths.includes(tab.path);
                  const atLimit = pinnedPaths.length >= MAX_PINNED;
                  return (
                    <div
                      key={tab.path}
                      className="flex items-center gap-2 px-3 py-1.5 hover:bg-white/[0.03] group"
                    >
                      <button
                        onClick={() => onToggle(tab.path)}
                        disabled={!isPinned && atLimit}
                        className={`w-4 h-4 rounded border flex items-center justify-center flex-shrink-0 transition-colors ${
                          isPinned
                            ? "bg-terminal-accent/20 border-terminal-accent text-terminal-accent"
                            : atLimit
                            ? "border-terminal-border/30 text-transparent cursor-not-allowed"
                            : "border-terminal-border hover:border-terminal-text-muted text-transparent"
                        }`}
                      >
                        {isPinned && (
                          <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
                            <path d="M2 5L4 7L8 3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
                          </svg>
                        )}
                      </button>
                      <button
                        onClick={() => { onNavigate(tab.path); setOpen(false); }}
                        className="flex-1 text-left text-[10px] font-medium tracking-[0.08em] text-terminal-text-muted hover:text-terminal-text-primary transition-colors"
                      >
                        {tab.label}
                      </button>
                      {isPinned && (
                        <span className="text-[8px] text-terminal-accent/60 tracking-wider">PINNED</span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ═══════════ SCROLLABLE TAB BAR ═══════════
function ScrollableTabBar({
  pinnedPaths,
  location,
}: {
  pinnedPaths: string[];
  location: string;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [canScrollLeft, setCanScrollLeft] = useState(false);
  const [canScrollRight, setCanScrollRight] = useState(false);

  const checkScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setCanScrollLeft(el.scrollLeft > 2);
    setCanScrollRight(el.scrollLeft + el.clientWidth < el.scrollWidth - 2);
  }, []);

  useEffect(() => {
    checkScroll();
    const el = scrollRef.current;
    if (el) {
      el.addEventListener("scroll", checkScroll, { passive: true });
      const obs = new ResizeObserver(checkScroll);
      obs.observe(el);
      return () => { el.removeEventListener("scroll", checkScroll); obs.disconnect(); };
    }
  }, [checkScroll, pinnedPaths]);

  const pinnedTabs = ALL_TABS.filter((t) => pinnedPaths.includes(t.path));

  const scroll = (dir: number) => {
    scrollRef.current?.scrollBy({ left: dir * 200, behavior: "smooth" });
  };

  return (
    <div className="flex items-center gap-0 mr-auto relative min-w-0">
      {canScrollLeft && (
        <button
          onClick={() => scroll(-1)}
          className="flex-shrink-0 w-5 h-6 flex items-center justify-center text-terminal-text-muted hover:text-terminal-accent transition-colors z-10 bg-gradient-to-r from-terminal-surface via-terminal-surface to-transparent"
        >
          <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
            <path d="M5 2L3 4L5 6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        </button>
      )}
      <div
        ref={scrollRef}
        className="flex items-center gap-0.5 overflow-x-auto scrollbar-hide min-w-0"
        style={{ scrollbarWidth: "none", msOverflowStyle: "none" }}
      >
        {pinnedTabs.map((tab) => {
          const isActive = location === tab.path || (location === "/" && tab.path === "/live");
          return (
            <Link
              key={tab.path}
              href={tab.path}
              className={`px-2.5 py-1.5 text-[10px] font-medium tracking-[0.1em] transition-colors rounded-sm whitespace-nowrap flex-shrink-0 ${
                isActive
                  ? "bg-terminal-accent/10 text-terminal-accent"
                  : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03]"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </div>
      {canScrollRight && (
        <button
          onClick={() => scroll(1)}
          className="flex-shrink-0 w-5 h-6 flex items-center justify-center text-terminal-text-muted hover:text-terminal-accent transition-colors z-10 bg-gradient-to-l from-terminal-surface via-terminal-surface to-transparent"
        >
          <svg width="8" height="8" viewBox="0 0 8 8" fill="none">
            <path d="M3 2L5 4L3 6" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
        </button>
      )}
    </div>
  );
}

// ═══════════ APP HEADER ═══════════
function AppHeader() {
  const [location, setLocation] = useLocation();
  const [pinnedPaths, setPinnedPaths] = useState<string[]>(DEFAULT_PINNED);

  const togglePin = useCallback((path: string) => {
    setPinnedPaths((prev) => {
      if (prev.includes(path)) {
        return prev.filter((p) => p !== path);
      }
      if (prev.length >= MAX_PINNED) return prev;
      return [...prev, path];
    });
  }, []);

  const handleNavigate = useCallback((path: string) => {
    // Auto-pin if navigating to an unpinned tab and there's room
    setPinnedPaths((prev) => {
      if (prev.includes(path)) return prev;
      if (prev.length >= MAX_PINNED) {
        // Replace last pinned tab
        return [...prev.slice(0, -1), path];
      }
      return [...prev, path];
    });
    setLocation(path);
  }, [setLocation]);

  return (
    <header className="h-10 flex items-center px-3 border-b border-terminal-border bg-terminal-surface flex-shrink-0 gap-1">
      <Link href="/live" className="flex items-center gap-2 mr-4 flex-shrink-0">
        <MetadronLogo />
        <span className="text-sm font-semibold tracking-[0.15em] text-terminal-text-primary uppercase">
          Metadron
        </span>
      </Link>

      <ScrollableTabBar pinnedPaths={pinnedPaths} location={location} />

      <div className="flex items-center gap-2 flex-shrink-0 ml-1">
        <TabSelectorDropdown
          pinnedPaths={pinnedPaths}
          onToggle={togglePin}
          onNavigate={handleNavigate}
        />
        <div className="w-px h-5 bg-terminal-border/50" />
        <LiveMetrics />
      </div>
    </header>
  );
}

// ═══════════ PROVIDER WARNING BANNER ═══════════
function ProviderWarningBanner() {
  const { data } = useEngineQuery<{
    status: string;
    providers: Record<string, { configured: boolean; live: boolean; error: string | null }>;
  }>("/health/providers", { refetchInterval: 60000 });

  if (!data || data.status === "ok") return null;

  const fmp = data.providers?.fmp;
  const openbb = data.providers?.openbb;
  const isCritical = data.status === "critical";

  let message = "";
  if (fmp && !fmp.configured) {
    message = "FMP_API_KEY not configured — market data, news, and fundamentals are unavailable.";
  } else if (fmp && !fmp.live) {
    message = `FMP API degraded — ${fmp.error || "quotes returning empty. Key may be expired or rate-limited."}`.replace(/\.$/, '') + '.';
  } else if (openbb && !openbb.live) {
    message = "OpenBB SDK unavailable — data pipeline offline.";
  } else {
    message = "One or more data providers are degraded. Check /health/providers for details.";
  }

  return (
    <div
      className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-medium tracking-wide flex-shrink-0"
      style={{
        background: isCritical
          ? "rgba(248, 113, 113, 0.12)"
          : "rgba(210, 153, 34, 0.10)",
        borderBottom: isCritical
          ? "1px solid rgba(248, 113, 113, 0.3)"
          : "1px solid rgba(210, 153, 34, 0.25)",
        color: isCritical ? "#fca5a5" : "#d29922",
      }}
    >
      <span style={{ fontSize: 12 }}>{isCritical ? "⚠" : "●"}</span>
      <span style={{ textTransform: "uppercase", letterSpacing: "0.08em" }}>
        {isCritical ? "CRITICAL" : "DEGRADED"}
      </span>
      <span style={{ color: "#7d8590", margin: "0 4px" }}>—</span>
      <span style={{ color: "#8b949e" }}>{message}</span>
    </div>
  );
}

// ═══════════ ROUTER ═══════════
function AppRouter() {
  return (
    <Switch>
      <Route path="/" component={LiveDashboard} />
      <Route path="/cube" component={MetadronCubePage} />
      <Route path="/market-wrap" component={MarketWrap} />
      <Route path="/live" component={LiveDashboard} />
      <Route path="/allocation" component={AssetAllocation} />
      <Route path="/risk" component={RiskPortfolio} />
      <Route path="/ml" component={MachineLearning} />
      <Route path="/tech" component={TechDashboard} />
      <Route path="/reports" component={Reporting} />
      <Route path="/strategy" component={StrategyBuilder} />
      <Route path="/openbb" component={OpenBBTerminal} />
      <Route path="/txlog" component={TransactionLog} />
      <Route path="/futures" component={FuturesPage} />
      <Route path="/tca" component={TCAPage} />
      <Route path="/agents" component={AgentsPage} />
      <Route path="/quant" component={QuantToolsPage} />
      <Route path="/recon" component={ReconciliationPage} />
      <Route path="/etf" component={ETFDashboard} />
      <Route path="/fixed-income" component={FixedIncomeDashboard} />
      <Route path="/macro" component={MacroDashboard} />
      <Route path="/arb" component={ArbitrageDashboard} />
      <Route path="/ml-models" component={MLModelsPage} />
      <Route path="/monte-carlo" component={MonteCarloPage} />
      <Route path="/simulations" component={SimulationsPage} />
      <Route path="/archive" component={ArchivePage} />
      <Route path="/backtesting" component={BacktestingPage} />
      <Route path="/velocity" component={MoneyVelocityPage} />
      <Route path="/thinking" component={ThinkingTab} />
      <Route path="/collateral" component={CollateralTab} />
      <Route path="/chat" component={ChatTab} />
      <Route path="/graphify" component={GraphifyPage} />
      <Route path="/vault" component={VaultPage} />
      <Route path="/security" component={SecurityPage} />
      <Route path="/jarvis" component={OpenJarvisPage} />
      <Route component={NotFound} />
    </Switch>
  );
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Router hook={useHashLocation}>
          <div className="h-screen flex flex-col bg-terminal-bg overflow-hidden">
            <AppHeader />
            <ProviderWarningBanner />
            <main className="flex-1 overflow-hidden">
              <AppRouter />
            </main>
          </div>
        </Router>
      </TooltipProvider>
    </QueryClientProvider>
  );
}

export default App;
