/**
 * ResizableDashboard — wraps dashboard columns in drag-to-resize panels.
 *
 * Uses the existing shadcn/ui ResizablePanelGroup + ResizablePanel + ResizableHandle
 * from @/components/ui/resizable (react-resizable-panels).
 *
 * Usage:
 *   <ResizableDashboard defaultSizes={[25, 50, 25]} minSizes={[15, 30, 15]}>
 *     <LeftPane />
 *     <CenterPane />
 *     <RightPane />
 *   </ResizableDashboard>
 */

import { Children } from "react";
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from "@/components/ui/resizable";
import { cn } from "@/lib/utils";

// ═══════════ DARK THEMED HANDLE ═══════════

/**
 * DarkResizableHandle — a styled handle that fits the dark terminal aesthetic.
 * Shows 3 grip dots on hover. Invisible when idle to keep the UI clean.
 */
function DarkResizableHandle({ className }: { className?: string }) {
  return (
    <ResizableHandle
      className={cn(
        // Override the default light-mode border/bg with terminal colours
        "group relative flex w-[3px] items-center justify-center",
        "bg-transparent hover:bg-[#00d4aa]/20 active:bg-[#00d4aa]/30",
        "transition-colors duration-150",
        "cursor-col-resize",
        // Remove the default ::after hit-area sizing — we widen the element itself
        "after:hidden",
        className
      )}
      data-testid="resizable-handle"
    >
      {/* Grip dots — visible only on hover/drag */}
      <div
        className={cn(
          "z-20 flex flex-col items-center justify-center gap-[3px]",
          "opacity-0 group-hover:opacity-100 group-active:opacity-100",
          "transition-opacity duration-150",
          "pointer-events-none"
        )}
      >
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="w-[3px] h-[3px] rounded-full bg-[#00d4aa]/60"
          />
        ))}
      </div>

      {/* Vertical rule — always visible but very faint */}
      <div className="absolute inset-y-0 left-1/2 -translate-x-1/2 w-px bg-[#1e2633] group-hover:bg-[#00d4aa]/30 transition-colors duration-150" />
    </ResizableHandle>
  );
}

// ═══════════ RESIZABLE DASHBOARD ═══════════

interface ResizableDashboardProps {
  /** Child panels — each renders in its own ResizablePanel */
  children: React.ReactNode;
  /**
   * Initial size percentages for each panel.
   * Must sum to 100. Length must equal the number of children.
   * Defaults to equal distribution.
   */
  defaultSizes?: number[];
  /**
   * Minimum size percentages for each panel (default 15%).
   * Prevents panels from collapsing so far they break layout.
   */
  minSizes?: number[];
  /** Extra class names forwarded to the outer ResizablePanelGroup */
  className?: string;
}

export function ResizableDashboard({
  children,
  defaultSizes,
  minSizes,
  className,
}: ResizableDashboardProps) {
  const childArray = Children.toArray(children);
  const count = childArray.length;

  // Build default sizes if not provided
  const sizes =
    defaultSizes ??
    childArray.map(() => Math.round(100 / count));

  // Minimum panel widths as % — ensures no panel can squeeze below ~200px
  const mins =
    minSizes ??
    childArray.map(() => 10); // 10% minimum per panel

  return (
    <ResizablePanelGroup
      direction="horizontal"
      className={cn("h-full w-full", className)}
      data-testid="resizable-dashboard"
    >
      {childArray.map((child, index) => (
        <>
          <ResizablePanel
            key={index}
            defaultSize={sizes[index] ?? Math.round(100 / count)}
            minSize={mins[index] ?? 10}
            className="h-full min-w-0"
            data-testid={`resizable-panel-${index}`}
          >
            {child}
          </ResizablePanel>

          {/* Insert a handle between every pair of panels */}
          {index < childArray.length - 1 && (
            <DarkResizableHandle key={`handle-${index}`} />
          )}
        </>
      ))}
    </ResizablePanelGroup>
  );
}

export { DarkResizableHandle };
