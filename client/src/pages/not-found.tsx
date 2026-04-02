import { Link } from "wouter";

export default function NotFound() {
  return (
    <div className="h-full flex items-center justify-center" data-testid="not-found">
      <div className="text-center">
        <div className="text-5xl font-mono font-bold text-terminal-accent mb-2">404</div>
        <div className="text-sm text-terminal-text-muted mb-4">Route not found</div>
        <Link
          href="/live"
          className="text-xs font-mono text-terminal-accent border border-terminal-accent/30 px-3 py-1.5 rounded hover:bg-terminal-accent/10 transition-colors"
        >
          ← BACK TO LIVE DASHBOARD
        </Link>
      </div>
    </div>
  );
}
