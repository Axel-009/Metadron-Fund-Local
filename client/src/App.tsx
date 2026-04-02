import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useEffect, useState, useRef, useCallback } from "react";
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
import MetadronCubePage from "@/pages/metadron-cube";
import MoneyVelocityPage from "@/pages/money-velocity";

// ═══════════ ALL AVAILABLE TABS ═══════════
const ALL_TABS = [
  { path: "/live", label: "LIVE" },
  { path: "/cube", label: "CUBE" },
  { path: "/market-wrap", label: "WRAP" },
  { path: "/allocation", label: "ALLOC" },
  { path: "/risk", label: "RISK" },
  { path: "/ml", label: "ML" },
  { path: "/tech", label: "TECH" },
  { path: "/reports", label: "REPORTS" },
  { path: "/strategy", label: "STRAT" },
  { path: "/openbb", label: "OPENBB" },
  { path: "/txlog", label: "TXLOG" },
  { path: "/futures", label: "FUTURES" },
  { path: "/tca", label: "TCA" },
  { path: "/agents", label: "AGENTS" },
  { path: "/quant", label: "QUANT" },
  { path: "/recon", label: "RECON" },
  { path: "/etf", label: "ETF" },
  { path: "/fixed-income", label: "FIXED INC" },
  { path: "/macro", label: "MACRO" },
  { path: "/arb", label: "ARB" },
  { path: "/ml-models", label: "ML MODELS" },
  { path: "/monte-carlo", label: "MC SIM" },
  { path: "/simulations", label: "SIM" },
  { path: "/archive", label: "ARCHIVE" },
  { path: "/velocity", label: "VELOCITY" },
];

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
  const [latency, setLatency] = useState(13.34);
  const [pnl, setPnl] = useState(422.18);
  const [nasdaq, setNasdaq] = useState(444);

  useEffect(() => {
    const interval = setInterval(() => {
      setLatency(12 + Math.random() * 4);
      setPnl(400 + Math.random() * 60 - 10);
      setNasdaq(430 + Math.floor(Math.random() * 30));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex items-center gap-4 text-[10px] font-mono tabular-nums">
      <StatusBadge status="live" />
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">LAT</span>
        <span className="text-terminal-text-primary">{latency.toFixed(2)}ms</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">+P&L</span>
        <span className="text-terminal-positive">+${pnl.toFixed(2)}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-terminal-text-faint">+{nasdaq}</span>
        <span className="w-1.5 h-1.5 rounded-full bg-terminal-positive" />
        <span className="text-terminal-text-muted">NASDAQ</span>
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
            {ALL_TABS.map((tab) => {
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
      <Route path="/velocity" component={MoneyVelocityPage} />
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
