"""Metadron Capital — S&P 1500 Universe for Full Universe Scan.

Provides universe composition for the 4-run scan architecture:
- Run 1: S&P 500 (large-cap)
- Run 2: S&P 400 MidCap
- Run 3: S&P 600 SmallCap
- Run 4: ETF + Fixed Income overlay

Loads from cross_asset_universe.py static fallback.
"""

import logging
from typing import List

logger = logging.getLogger("metadron.allocation.universe")

# Import from existing cross-asset universe
try:
    from engine.data.cross_asset_universe import (
        SP500_TICKERS, SP400_TICKERS, SP600_TICKERS,
    )
except ImportError:
    try:
        from ..data.cross_asset_universe import (
            SP500_TICKERS, SP400_TICKERS, SP600_TICKERS,
        )
    except ImportError:
        SP500_TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "JPM", "V", "UNH"]
        SP400_TICKERS = ["DECK", "SAIA", "TOST", "FIX", "COOP", "LNTH", "RBC", "WFRD"]
        SP600_TICKERS = ["SPSC", "CALM", "SIG", "BOOT", "ARCH", "CARG", "PTGX"]
        logger.warning("cross_asset_universe not available — using minimal fallback tickers.")


# ═══════════════════════════════════════════════════════════════════════════
# ETF Universe
# ═══════════════════════════════════════════════════════════════════════════

ETF_TICKERS = [
    # Specified in spec
    "TLTW", "QQQ", "SPY", "IWM", "HYG", "LQD", "TLT", "PDBC", "GLD", "USO",
    # GICS Sector ETFs
    "XLE", "XLB", "XLI", "XLY", "XLP", "XLV", "XLF", "XLK", "XLC", "XLU", "XLRE",
    # Additional broad / factor ETFs
    "DIA", "VTI", "VOO", "MDY", "IJR", "IEMG", "EFA", "VWO",
    # Dividend / Income ETFs
    "VIG", "SCHD", "DVY", "HDV", "JEPI", "JEPQ",
    # Commodity
    "SLV", "DBC", "COPX", "UNG",
]

# ═══════════════════════════════════════════════════════════════════════════
# Fixed Income Universe
# ═══════════════════════════════════════════════════════════════════════════

FI_TICKERS = [
    # Specified in spec
    "TLT", "IEF", "SHY", "HYG", "LQD", "EMB", "BKLN",
    # Additional FI
    "AGG", "BND", "VCIT", "VCSH", "MUB", "TIP", "GOVT", "MBB",
    "FLOT", "SCHO", "SCHR", "IGIB", "IGSB", "USIG",
    # TIPS / Inflation
    "STIP", "SCHP",
    # International FI
    "BNDX", "IAGG",
]

# ═══════════════════════════════════════════════════════════════════════════
# Universe Map
# ═══════════════════════════════════════════════════════════════════════════

UNIVERSES = {
    "SP500": "S&P 500 Large Cap",
    "SP400_MIDCAP": "S&P 400 MidCap",
    "SP600_SMALLCAP": "S&P 600 SmallCap",
    "ETF_FI": "ETF + Fixed Income",
}


def get_universe(run_name: str) -> List[str]:
    """Return ticker list for a given universe run.

    Args:
        run_name: One of SP500, SP400_MIDCAP, SP600_SMALLCAP, ETF_FI.

    Returns:
        List of ticker symbols.
    """
    if run_name == "SP500":
        return list(SP500_TICKERS)
    elif run_name == "SP400_MIDCAP":
        return list(SP400_TICKERS)
    elif run_name == "SP600_SMALLCAP":
        return list(SP600_TICKERS)
    elif run_name == "ETF_FI":
        # Combine ETFs and FI, de-duplicate
        combined = list(dict.fromkeys(ETF_TICKERS + FI_TICKERS))
        return combined
    else:
        logger.warning("Unknown universe: %s — returning empty list.", run_name)
        return []


def get_all_universes() -> dict:
    """Return all universe compositions with counts."""
    return {
        name: {
            "description": desc,
            "ticker_count": len(get_universe(name)),
            "tickers": get_universe(name),
        }
        for name, desc in UNIVERSES.items()
    }


def get_universe_summary() -> dict:
    """Return summary counts for all universes."""
    total = 0
    summary = {}
    for name, desc in UNIVERSES.items():
        count = len(get_universe(name))
        summary[name] = {"description": desc, "count": count}
        total += count
    summary["total_unique"] = total
    return summary
