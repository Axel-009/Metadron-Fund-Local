import { useEffect, useState } from "react";

const TICKER_MESSAGES = [
  "Tech sector is leading, Energy lagging, liquidity tightening",
  "AAPL +2.3% on earnings beat, MSFT neutral, GOOG -0.8%",
  "VIX at 14.2, below 20-day MA, low vol regime continues",
  "Fed speakers hawkish, 10Y yield +4bps to 4.32%",
  "BTC/USD testing 68k resistance, ETH/BTC ratio declining",
  "Crude oil -1.2% on inventory build, natgas +3.4%",
  "S&P 500 new ATH, breadth improving, NASDAQ outperforming",
  "China PMI miss, USDCNY at 7.24, emerging markets under pressure",
];

export function LiveTicker() {
  const [messages] = useState(TICKER_MESSAGES);

  return (
    <div className="overflow-hidden h-5 flex items-center border-t border-terminal-border bg-terminal-bg/50">
      <div className="flex items-center gap-1 px-2 flex-shrink-0">
        <span className="text-[9px] font-mono text-terminal-accent">▶</span>
        <span className="text-[9px] font-mono text-terminal-text-muted uppercase tracking-wider">MARKET UPDATE</span>
        <span className="text-[9px] font-mono text-terminal-text-faint mx-1">:</span>
      </div>
      <div className="overflow-hidden flex-1">
        <div className="animate-ticker whitespace-nowrap flex">
          {[...messages, ...messages].map((msg, i) => (
            <span key={i} className="text-[9px] font-mono text-terminal-text-muted mr-12">
              {msg}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
