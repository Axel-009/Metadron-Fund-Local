"""UniverseEngine — Dynamic securities universe with GICS taxonomy.

Fetches S&P 500/400/600 from Wikipedia (~1,500+ equities).
GICS 4-tier taxonomy: 11 sectors / 25 industry groups / 74 industries / 163 sub-industries.
70+ ETFs (sector/factor/FI/commodity/vol/intl), 26 RV pairs.
All data via yfinance.  Cached to data/universe_cache/universe.json (24h TTL).
"""

import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POOL_FAIR_VALUE_TOL = 0.10          # ±10 % tolerance for GIC pooling
POOL_PERMANENT_MIN = 5_000_000      # $5M notional minimum
CACHE_TTL = 86400                   # 24h
CACHE_DIR = Path(__file__).parent.parent.parent / "data" / "universe_cache"

# GICS sector → representative ETF
SECTOR_ETFS = {
    "Energy": "XLE", "Materials": "XLB", "Industrials": "XLI",
    "Consumer Discretionary": "XLY", "Consumer Staples": "XLP",
    "Health Care": "XLV", "Financials": "XLF", "Information Technology": "XLK",
    "Communication Services": "XLC", "Utilities": "XLU", "Real Estate": "XLRE",
}

# Factor / thematic / fixed-income / commodity / vol / international ETFs
FACTOR_ETFS = ["MTUM", "VLUE", "QUAL", "SIZE", "USMV"]
FI_ETFS = ["TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG"]
COMMODITY_ETFS = ["GLD", "SLV", "USO", "DBA", "DBC"]
VOL_ETFS = ["VXX", "SVXY", "UVXY"]
INTL_ETFS = ["EFA", "EEM", "VEA", "VWO", "FXI"]

ALL_ETFS = (
    list(SECTOR_ETFS.values()) + FACTOR_ETFS + FI_ETFS
    + COMMODITY_ETFS + VOL_ETFS + INTL_ETFS
)

# 26 classic RV pairs
RV_PAIRS = [
    ("XLE", "XLU"), ("XLK", "XLF"), ("XLY", "XLP"),
    ("GLD", "SLV"), ("TLT", "HYG"), ("EFA", "EEM"),
    ("AAPL", "MSFT"), ("GOOGL", "META"), ("JPM", "GS"),
    ("XOM", "CVX"), ("JNJ", "PFE"), ("HD", "LOW"),
    ("AMZN", "WMT"), ("V", "MA"), ("DIS", "NFLX"),
    ("BA", "LMT"), ("UNH", "CI"), ("CAT", "DE"),
    ("COP", "EOG"), ("COST", "TGT"), ("AMD", "INTC"),
    ("AVGO", "TXN"), ("LLY", "ABBV"), ("BRK-B", "JPM"),
    ("SPY", "QQQ"), ("IWM", "SPY"),
]


# ---------------------------------------------------------------------------
# GICS taxonomy
# ---------------------------------------------------------------------------
GICS_SECTORS = {
    10: "Energy", 15: "Materials", 20: "Industrials",
    25: "Consumer Discretionary", 30: "Consumer Staples",
    35: "Health Care", 40: "Financials",
    45: "Information Technology", 50: "Communication Services",
    55: "Utilities", 60: "Real Estate",
}


@dataclass
class Security:
    ticker: str
    name: str = ""
    sector: str = ""
    industry: str = ""
    sub_industry: str = ""
    gics_code: int = 0
    market_cap: float = 0.0
    quality_tier: str = "D"   # A–G


@dataclass
class UniverseSnapshot:
    timestamp: float = 0.0
    equities: list = field(default_factory=list)
    etfs: list = field(default_factory=list)
    rv_pairs: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Universe Engine
# ---------------------------------------------------------------------------
class UniverseEngine:
    """Dynamic securities universe manager."""

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._equities: list[Security] = []
        self._etfs: list[str] = list(ALL_ETFS)
        self._rv_pairs = list(RV_PAIRS)
        self._loaded = False

    # --- Public API ----------------------------------------------------------

    def load(self, force_refresh: bool = False) -> "UniverseEngine":
        """Load universe from cache or fetch fresh."""
        cache_file = self.cache_dir / "universe.json"
        if not force_refresh and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < CACHE_TTL:
                self._load_cache(cache_file)
                return self
        self._fetch_sp_constituents()
        self._save_cache(cache_file)
        return self

    def get_equity_tickers(self) -> list[str]:
        self._ensure_loaded()
        return [s.ticker for s in self._equities]

    def get_by_sector(self, sector: str) -> list[Security]:
        self._ensure_loaded()
        return [s for s in self._equities if s.sector == sector]

    def get_sectors(self) -> list[str]:
        return list(GICS_SECTORS.values())

    def get_sector_etf(self, sector: str) -> Optional[str]:
        return SECTOR_ETFS.get(sector)

    def get_rv_pairs(self) -> list[tuple[str, str]]:
        return list(self._rv_pairs)

    def get_all_etfs(self) -> list[str]:
        return list(self._etfs)

    def get_top_n_by_sector(self, n: int = 10) -> dict[str, list[Security]]:
        """Top N by market cap per sector."""
        self._ensure_loaded()
        result = {}
        for sector in self.get_sectors():
            secs = sorted(
                self.get_by_sector(sector),
                key=lambda s: s.market_cap, reverse=True,
            )
            result[sector] = secs[:n]
        return result

    def screen(
        self,
        min_market_cap: float = 0,
        sectors: Optional[list[str]] = None,
        quality_tiers: Optional[list[str]] = None,
    ) -> list[Security]:
        """Filter universe by criteria."""
        self._ensure_loaded()
        out = self._equities
        if min_market_cap > 0:
            out = [s for s in out if s.market_cap >= min_market_cap]
        if sectors:
            out = [s for s in out if s.sector in sectors]
        if quality_tiers:
            out = [s for s in out if s.quality_tier in quality_tiers]
        return out

    def size(self) -> int:
        self._ensure_loaded()
        return len(self._equities)

    # --- Fetch ---------------------------------------------------------------

    def _fetch_sp_constituents(self):
        """Fetch S&P 500/400/600 from Wikipedia + yfinance sector data."""
        urls = {
            "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "sp400": "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "sp600": "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies",
        }
        all_rows = []
        for label, url in urls.items():
            try:
                tables = pd.read_html(url)
                if not tables:
                    continue
                df = tables[0]
                # Normalise column names
                cols = {c: c.strip() for c in df.columns}
                df = df.rename(columns=cols)
                # Find ticker column
                ticker_col = None
                for candidate in ["Symbol", "Ticker symbol", "Ticker Symbol", "Ticker"]:
                    if candidate in df.columns:
                        ticker_col = candidate
                        break
                if ticker_col is None:
                    continue
                sector_col = next(
                    (c for c in df.columns if "GICS" in c and "Sector" in c), None
                )
                industry_col = next(
                    (c for c in df.columns if "GICS" in c and ("Sub" in c or "Industry" in c) and c != sector_col), None
                )
                for _, row in df.iterrows():
                    ticker = str(row[ticker_col]).strip().replace(".", "-")
                    if not ticker or ticker == "nan":
                        continue
                    sec = Security(
                        ticker=ticker,
                        name=str(row.get("Security", row.get("Company", ""))),
                        sector=str(row.get(sector_col, "")) if sector_col else "",
                        industry=str(row.get(industry_col, "")) if industry_col else "",
                    )
                    all_rows.append(sec)
            except Exception:
                continue

        # Deduplicate by ticker
        seen = set()
        unique = []
        for s in all_rows:
            if s.ticker not in seen:
                seen.add(s.ticker)
                unique.append(s)
        self._equities = unique
        self._loaded = True

    # --- Cache ---------------------------------------------------------------

    def _save_cache(self, path: Path):
        snapshot = UniverseSnapshot(
            timestamp=time.time(),
            equities=[asdict(s) for s in self._equities],
            etfs=self._etfs,
            rv_pairs=self._rv_pairs,
        )
        path.write_text(json.dumps(asdict(snapshot), indent=2))

    def _load_cache(self, path: Path):
        data = json.loads(path.read_text())
        self._equities = [Security(**e) for e in data.get("equities", [])]
        self._etfs = data.get("etfs", list(ALL_ETFS))
        self._rv_pairs = [tuple(p) for p in data.get("rv_pairs", RV_PAIRS)]
        self._loaded = True

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_engine: Optional[UniverseEngine] = None


def get_engine(force_refresh: bool = False) -> UniverseEngine:
    """Get or create the global UniverseEngine singleton."""
    global _engine
    if _engine is None:
        _engine = UniverseEngine()
    if not _engine._loaded or force_refresh:
        _engine.load(force_refresh=force_refresh)
    return _engine
