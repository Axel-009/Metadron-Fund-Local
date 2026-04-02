import { Switch, Route, Router, Link, useLocation } from "wouter";
import { useHashLocation } from "wouter/use-hash-location";
import { queryClient } from "./lib/queryClient";
import { QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { useEffect, useState } from "react";
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

const TABS = [
  { path: "/market", label: "MARKET" },
  { path: "/market-wrap", label: "WRAP" },
  { path: "/live", label: "LIVE" },
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
];

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

function AppHeader() {
  const [location] = useLocation();

  return (
    <header className="h-10 flex items-center px-3 border-b border-terminal-border bg-terminal-surface flex-shrink-0">
      <Link href="/live" className="flex items-center gap-2 mr-6 flex-shrink-0">
        <MetadronLogo />
        <span className="text-sm font-semibold tracking-[0.15em] text-terminal-text-primary uppercase">
          Metadron Fund
        </span>
      </Link>
      <nav className="flex items-center gap-0.5 mr-auto">
        {TABS.map((tab) => {
          const isActive = location === tab.path || (location === "/" && tab.path === "/live");
          return (
            <Link
              key={tab.path}
              href={tab.path}
              className={`px-3 py-1.5 text-[10px] font-medium tracking-[0.1em] transition-colors rounded-sm ${
                isActive
                  ? "bg-terminal-accent/10 text-terminal-accent"
                  : "text-terminal-text-muted hover:text-terminal-text-primary hover:bg-white/[0.03]"
              }`}
            >
              {tab.label}
            </Link>
          );
        })}
      </nav>
      <LiveMetrics />
    </header>
  );
}

function AppRouter() {
  return (
    <Switch>
      <Route path="/" component={LiveDashboard} />
      <Route path="/market" component={LiveDashboard} />
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
