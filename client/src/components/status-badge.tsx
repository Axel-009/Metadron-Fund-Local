interface StatusBadgeProps {
  status: "live" | "eod" | "delayed" | "offline";
  className?: string;
}

const STATUS_CONFIG = {
  live: { color: "bg-terminal-positive", label: "LIVE", pulse: true },
  eod: { color: "bg-terminal-warning", label: "EOD", pulse: false },
  delayed: { color: "bg-terminal-warning", label: "DELAYED", pulse: true },
  offline: { color: "bg-terminal-text-faint", label: "OFFLINE", pulse: false },
};

export function StatusBadge({ status, className = "" }: StatusBadgeProps) {
  const config = STATUS_CONFIG[status];
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      <span
        className={`w-1.5 h-1.5 rounded-full ${config.color} ${config.pulse ? "animate-pulse-live" : ""}`}
      />
      <span className="text-[9px] font-mono font-medium tracking-wider text-terminal-text-muted">
        {config.label}
      </span>
    </span>
  );
}
