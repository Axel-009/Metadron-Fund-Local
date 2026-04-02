import { type ReactNode } from "react";

interface DashboardPanelProps {
  title: string;
  children: ReactNode;
  className?: string;
  headerRight?: ReactNode;
  noPadding?: boolean;
}

export function DashboardPanel({
  title,
  children,
  className = "",
  headerRight,
  noPadding = false,
}: DashboardPanelProps) {
  return (
    <div className={`terminal-panel flex flex-col ${className}`}>
      <div className="panel-header flex-shrink-0">
        <span className="tracking-widest">{title}</span>
        {headerRight && <div className="flex items-center gap-2">{headerRight}</div>}
      </div>
      <div className={`flex-1 overflow-auto ${noPadding ? "" : "p-2"}`}>
        {children}
      </div>
    </div>
  );
}
