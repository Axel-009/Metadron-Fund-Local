#!/usr/bin/env python3
"""Governance: Classify universe tickers as IG or HY using public credit data.

Uses OpenBB Yahoo provider for debt_to_equity, current_ratio, quick_ratio.
Outputs: fund/governance/credit_classification.json

Rules (IG vs HY):
  IG (A/B tiers): debt_equity < 1.0 AND current_ratio > 1.2
  HY (C/D/E tiers): everything else
  HY Distressed (F): debt_equity > 3.0 OR current_ratio < 0.8

NOT modifying engine code. This is a reference file the engine can read.
"""

import json
import sys
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.data.openbb_data import _obb, _openbb_available
from engine.data.universe_engine import get_engine

OUTPUT = Path(__file__).parent.parent / "governance" / "credit_classification.json"
OUTPUT.parent.mkdir(exist_ok=True)


def fetch_metrics(tickers: list[str], batch_size: int = 20) -> dict:
    """Fetch credit metrics from Yahoo via OpenBB."""
    results = {}
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        for ticker in batch:
            try:
                r = _obb.equity.fundamental.metrics(symbol=ticker, provider="yfinance")
                df = r.to_dataframe()
                if df.empty:
                    continue
                row = df.iloc[0]
                de = float(row.get("debt_to_equity", 0) or 0)
                cr = float(row.get("current_ratio", 0) or 0)
                qr = float(row.get("quick_ratio", 0) or 0)
                eg = float(row.get("earnings_growth", 0) or 0)
                mc = float(row.get("market_cap", 0) or 0)
                pe = float(row.get("pe_ratio", 0) or 0)
                results[ticker] = {
                    "debt_to_equity": de,
                    "current_ratio": cr,
                    "quick_ratio": qr,
                    "earnings_growth": eg,
                    "market_cap": mc,
                    "pe_ratio": pe,
                }
            except Exception as e:
                pass
        pct = min(100, (i + batch_size) / len(tickers) * 100)
        print(f"  {pct:.0f}% ({len(results)} tickers)", flush=True)
        time.sleep(0.3)
    return results


def classify(ticker: str, metrics: dict) -> dict:
    """Classify ticker as IG/HY based on credit metrics."""
    de = metrics.get("debt_to_equity", 0)
    cr = metrics.get("current_ratio", 0)
    eg = metrics.get("earnings_growth", 0)

    # IG: strong balance sheet
    if de < 1.0 and cr > 1.2:
        tier = "IG"
        sub = "A" if de < 0.5 and cr > 1.5 else "B"
    # HY: leveraged but not distressed
    elif de < 3.0 and cr > 0.8:
        tier = "HY"
        sub = "C" if de < 2.0 else "D"
    # Distressed
    else:
        tier = "HY_DISTRESSED"
        sub = "E"

    return {
        "ticker": ticker,
        "tier": tier,
        "sub": sub,
        "debt_to_equity": round(de, 2),
        "current_ratio": round(cr, 2),
        "earnings_growth": round(eg, 4),
        "market_cap": metrics.get("market_cap", 0),
    }


def main():
    print("=== GOVERNANCE: CREDIT CLASSIFICATION ===")

    # Load universe
    ue = get_engine()
    ue.load()
    tickers = [s.ticker for s in ue.get_all()]
    print(f"Universe: {len(tickers)} tickers")

    # Fetch metrics
    print("Fetching credit metrics via Yahoo...")
    metrics = fetch_metrics(tickers)
    print(f"Fetched: {len(metrics)} tickers")

    # Classify
    classifications = []
    for ticker, m in metrics.items():
        c = classify(ticker, m)
        classifications.append(c)

    # Stats
    ig = [c for c in classifications if c["tier"] == "IG"]
    hy = [c for c in classifications if c["tier"] == "HY"]
    hyd = [c for c in classifications if c["tier"] == "HY_DISTRESSED"]

    print(f"\nIG: {len(ig)} ({len(ig)/len(classifications)*100:.0f}%)")
    print(f"HY: {len(hy)} ({len(hy)/len(classifications)*100:.0f}%)")
    print(f"HY Distressed: {len(hyd)} ({len(hyd)/len(classifications)*100:.0f}%)")

    # Save
    output = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "total": len(classifications),
        "summary": {
            "IG": len(ig),
            "HY": len(hy),
            "HY_DISTRESSED": len(hyd),
        },
        "classifications": {c["ticker"]: c for c in classifications},
    }
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {OUTPUT}")

    # Top IG
    print("\nTop IG names:")
    for c in sorted(ig, key=lambda x: x["market_cap"], reverse=True)[:10]:
        print(f"  {c['ticker']:6s} D/E={c['debt_to_equity']:.2f} CR={c['current_ratio']:.2f} sub={c['sub']}")

    # Top HY
    print("\nTop HY names:")
    for c in sorted(hy, key=lambda x: x["market_cap"], reverse=True)[:10]:
        print(f"  {c['ticker']:6s} D/E={c['debt_to_equity']:.2f} CR={c['current_ratio']:.2f} sub={c['sub']}")


if __name__ == "__main__":
    main()
