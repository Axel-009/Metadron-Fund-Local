"""Metadron Capital — Comprehensive Data Ingestion Orchestrator.

Manages continuous data ingestion across ALL asset classes with specific
constraints:
  - NO crypto
  - Equities: US securities (S&P 1500) + London FTSE 100 only
  - Commodities: Major ETFs only (price reference, cyclical patterns)
  - Indices: Major benchmarks for reference + monthly rebalancing
  - Fixed Income: G10 + India + Japan sovereign, US corporate only, major structured benchmarks
  - Currencies: G10 + India + Japan
  - Econometrics: FRED macro data
  - SEC Filings: Major updates only (10-K, 10-Q, 8-K material)
  - Options: Selected securities opportunistically
  - Futures: Beta management within corridor

All data sourced via OpenBB (34+ providers). Routes ingested data to the
appropriate engine layers (L1-L7) and feeds the UniversalDataPool.
"""

import logging
import time
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Asset Class Definitions & Universe Constraints
# ═══════════════════════════════════════════════════════════════════════════

class AssetClass(str, Enum):
    EQUITY = "equity"
    COMMODITY = "commodity"
    INDEX = "index"
    FIXED_INCOME = "fixed_income"
    CURRENCY = "currency"
    ECONOMETRIC = "econometric"
    SEC_FILING = "sec_filing"
    OPTIONS = "options"
    FUTURES = "futures"


class IngestionFrequency(str, Enum):
    REAL_TIME = "real_time"       # Tick-by-tick
    ONE_MIN = "1min"             # 1-minute bars
    FIVE_MIN = "5min"            # 5-minute bars
    FIFTEEN_MIN = "15min"        # 15-minute bars
    HOURLY = "hourly"            # Hourly bars
    DAILY = "daily"              # End-of-day
    WEEKLY = "weekly"            # Weekly aggregation
    MONTHLY = "monthly"          # Monthly (rebalancing, filings)


class FISubClass(str, Enum):
    SOVEREIGN = "sovereign"
    CORPORATE_IG = "corporate_ig"
    CORPORATE_HY = "corporate_hy"
    STRUCTURED_MBS = "structured_mbs"
    STRUCTURED_ABS = "structured_abs"
    STRUCTURED_CMBS = "structured_cmbs"


# ── FTSE 100 Universe ───────────────────────────────────────────────────
FTSE_100_TICKERS = [
    "III.L", "ADM.L", "AAF.L", "AAL.L", "ANTO.L", "AHT.L", "ABF.L",
    "AZN.L", "AUTO.L", "AVV.L", "AV.L", "BME.L", "BA.L", "BARC.L",
    "BDEV.L", "BEZ.L", "BKG.L", "BP.L", "BATS.L", "BLND.L", "BT.A.L",
    "BNZL.L", "BRBY.L", "CNA.L", "CCH.L", "CPG.L", "CTEC.L", "CRDA.L",
    "DCC.L", "DGE.L", "DPLM.L", "EDV.L", "ENT.L", "EXPN.L", "FCIT.L",
    "FRES.L", "GLEN.L", "GSK.L", "HLN.L", "HLMA.L", "HIK.L", "HSBA.L",
    "IMB.L", "INF.L", "IHG.L", "ICP.L", "IAG.L", "ITRK.L", "JD.L",
    "KGF.L", "LAND.L", "LGEN.L", "LLOY.L", "LSEG.L", "MNG.L", "MKS.L",
    "MRO.L", "MNDI.L", "NG.L", "NWG.L", "NXT.L", "OCDO.L", "PSON.L",
    "PSH.L", "PSN.L", "PHNX.L", "PRU.L", "RKT.L", "REL.L", "RTO.L",
    "RMV.L", "RIO.L", "RR.L", "RS1.L", "SGE.L", "SBRY.L", "SDR.L",
    "SMT.L", "SGRO.L", "SVT.L", "SHEL.L", "SN.L", "SMDS.L", "SMIN.L",
    "SPX.L", "SSE.L", "STJ.L", "STAN.L", "TW.L", "TSCO.L", "ULVR.L",
    "UTG.L", "UU.L", "VOD.L", "WEIR.L", "WTB.L", "WPP.L",
]

# ── Major Commodity ETFs (price reference + cyclical patterns) ──────────
COMMODITY_ETFS = {
    "GLD": "Gold",
    "SLV": "Silver",
    "USO": "Oil (WTI)",
    "UNG": "Natural Gas",
    "DBA": "Agriculture",
    "DBC": "Commodities Broad",
    "PDBC": "Commodities Optimized",
    "COPX": "Copper Miners",
    "WEAT": "Wheat",
    "CORN": "Corn",
    "PALL": "Palladium",
    "PPLT": "Platinum",
}

# ── Major Index ETFs (benchmarking + rebalancing) ──────────────────────
INDEX_BENCHMARKS = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "DIA": "Dow 30",
    "VT": "Total World",
    "EFA": "EAFE Developed",
    "EEM": "Emerging Markets",
    "MDY": "S&P MidCap 400",
    "IJR": "S&P SmallCap 600",
    "VTI": "Total US Market",
    "EWU": "FTSE UK (iShares)",
}

# ── Fixed Income ETFs ──────────────────────────────────────────────────
# G10 + India + Japan sovereign
SOVEREIGN_FI = {
    "TLT": "US Treasury 20+yr",
    "IEF": "US Treasury 7-10yr",
    "SHY": "US Treasury 1-3yr",
    "GOVT": "US Treasury All",
    "IGOV": "International Govt",
    "BWX": "Intl Treasury ex-US",
    "EMB": "EM Sovereign (USD)",
    "BNDX": "Intl Bond Aggregate",
    "VGIT": "US Intermediate Treasury",
    "SPTL": "US Long Treasury",
}

# G10 countries: US, UK, Germany, France, Japan, Canada, Australia, NZ, Switzerland, Norway, Sweden
# + India
G10_PLUS_COUNTRIES = [
    "US", "UK", "DE", "FR", "JP", "CA", "AU", "NZ", "CH", "NO", "SE", "IN",
]

# US Corporate credit bonds only (IG and HY)
CORPORATE_FI = {
    "LQD": "Investment Grade Corporate",
    "HYG": "High Yield Corporate",
    "JNK": "High Yield Corporate (SPDR)",
    "VCIT": "Intermediate Corporate",
    "VCSH": "Short-Term Corporate",
    "USIG": "US IG Corporate",
    "ANGL": "Fallen Angels HY",
    "BKLN": "Senior Loans",
}

# Structured Products — major benchmarks only
STRUCTURED_FI = {
    "MBB": "Mortgage-Backed (iShares)",
    "VMBS": "Mortgage-Backed (Vanguard)",
    "SPMB": "Mortgage-Backed (SPDR)",
    "CMBS": "Commercial MBS",
}

# ── Currency ETFs (G10 + India + Japan) ────────────────────────────────
CURRENCY_INSTRUMENTS = {
    "UUP": "US Dollar Index Bull",
    "FXE": "Euro",
    "FXB": "British Pound",
    "FXY": "Japanese Yen",
    "FXA": "Australian Dollar",
    "FXC": "Canadian Dollar",
    "FXF": "Swiss Franc",
    "FXS": "Swedish Krona",
    "BNZ": "New Zealand Dollar",  # WisdomTree
    "INR": "Indian Rupee (proxy via SMIN.L or direct)",
}

# ── FRED Econometric Series ────────────────────────────────────────────
FRED_SERIES = {
    # GDP & Output
    "GDP": "Gross Domestic Product",
    "GDPC1": "Real GDP",
    "A191RL1Q225SBEA": "Real GDP Growth Rate",
    # Inflation
    "CPIAUCSL": "CPI All Urban Consumers",
    "CPILFESL": "Core CPI (ex Food & Energy)",
    "PPIACO": "Producer Price Index",
    "T10YIE": "10-Year Breakeven Inflation",
    # Employment
    "PAYEMS": "Nonfarm Payrolls",
    "UNRATE": "Unemployment Rate",
    "ICSA": "Initial Jobless Claims",
    # Manufacturing & Services
    "MANEMP": "Manufacturing Employment",
    "INDPRO": "Industrial Production",
    # Consumer
    "UMCSENT": "Michigan Consumer Sentiment",
    "RSAFS": "Retail Sales",
    "HOUST": "Housing Starts",
    "PERMIT": "Building Permits",
    # Money & Velocity
    "M2SL": "M2 Money Supply",
    "M2V": "M2 Velocity",
    "M1SL": "M1 Money Supply",
    # Fed Plumbing
    "WALCL": "Fed Total Assets",
    "RRPONTSYD": "Reverse Repo (ON-RRP)",
    "WTREGEN": "Treasury General Account",
    "SOFR": "SOFR Rate",
    "DFF": "Effective Fed Funds Rate",
    "FEDFUNDS": "Fed Funds Target",
    "TOTRESNS": "Total Reserves",
    "EXCSRESNS": "Excess Reserves",
    # Credit & Spreads
    "BAMLH0A0HYM2": "HY OAS Spread",
    "BAMLC0A0CM": "IG OAS Spread",
    "T10Y2Y": "10Y-2Y Spread",
    "T10Y3M": "10Y-3M Spread",
    # Rates
    "DGS2": "2-Year Treasury",
    "DGS5": "5-Year Treasury",
    "DGS10": "10-Year Treasury",
    "DGS30": "30-Year Treasury",
    # Trade
    "BOPGSTB": "Trade Balance",
    # Lending
    "DPCREDIT": "Consumer Credit",
    "BUSLOANS": "Commercial & Industrial Loans",
    "REALLN": "Real Estate Loans",
}

# ── SEC Filing Types (major updates only) ──────────────────────────────
SEC_FILING_TYPES = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]

# ── Futures for Beta Management ────────────────────────────────────────
FUTURES_INSTRUMENTS = {
    "ES": "E-mini S&P 500",
    "NQ": "E-mini NASDAQ 100",
    "YM": "E-mini Dow 30",
    "RTY": "E-mini Russell 2000",
    "VX": "VIX Futures",
    "ZN": "10-Year T-Note",
    "ZB": "30-Year T-Bond",
    "ZF": "5-Year T-Note",
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Quality & Validation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DataQualityReport:
    """Data quality check results."""
    asset_class: str = ""
    symbol: str = ""
    timestamp: str = ""
    has_data: bool = False
    is_fresh: bool = False
    staleness_minutes: float = 0.0
    missing_fields: list = field(default_factory=list)
    outlier_detected: bool = False
    quality_score: float = 1.0  # 0-1


@dataclass
class IngestionResult:
    """Result of a single ingestion cycle."""
    asset_class: AssetClass = AssetClass.EQUITY
    symbols_requested: int = 0
    symbols_received: int = 0
    data_points: int = 0
    errors: list = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: str = ""
    quality_scores: dict = field(default_factory=dict)


@dataclass
class IngestionState:
    """Current state of the ingestion pipeline."""
    is_running: bool = False
    last_cycle_time: str = ""
    cycles_completed: int = 0
    total_data_points: int = 0
    errors_today: int = 0
    asset_class_status: dict = field(default_factory=dict)
    last_results: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# DataIngestionOrchestrator
# ═══════════════════════════════════════════════════════════════════════════

class DataIngestionOrchestrator:
    """Comprehensive data ingestion orchestrator for Metadron Capital.

    Manages continuous data ingestion across all asset classes with
    specific investment universe constraints. Routes data to the
    appropriate engine layers and the UniversalDataPool.

    NO CRYPTO is ingested. All data via OpenBB.

    Usage:
        orchestrator = DataIngestionOrchestrator()
        orchestrator.run_continuous_loop()  # blocking
        # or
        orchestrator.run_single_cycle()     # one-shot
    """

    # Frequency schedule per asset class
    FREQUENCY_MAP = {
        AssetClass.EQUITY: IngestionFrequency.ONE_MIN,
        AssetClass.COMMODITY: IngestionFrequency.FIVE_MIN,
        AssetClass.INDEX: IngestionFrequency.ONE_MIN,
        AssetClass.FIXED_INCOME: IngestionFrequency.FIFTEEN_MIN,
        AssetClass.CURRENCY: IngestionFrequency.FIVE_MIN,
        AssetClass.ECONOMETRIC: IngestionFrequency.DAILY,
        AssetClass.SEC_FILING: IngestionFrequency.DAILY,
        AssetClass.OPTIONS: IngestionFrequency.FIVE_MIN,
        AssetClass.FUTURES: IngestionFrequency.ONE_MIN,
    }

    def __init__(
        self,
        equity_universe: Optional[list] = None,
        selected_options_tickers: Optional[list] = None,
        log_dir: Optional[Path] = None,
    ):
        self.log_dir = log_dir or Path("logs/ingestion")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._state = IngestionState()
        self._data_cache: dict[str, dict] = defaultdict(dict)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

        # Equity universe from cross_asset_universe.py or provided
        self._equity_universe = equity_universe or []
        self._ftse_universe = FTSE_100_TICKERS
        self._selected_options = selected_options_tickers or []

        # Last ingestion timestamps per asset class
        self._last_ingestion: dict[str, datetime] = {}

        # OpenBB data handler
        self._obb = None
        self._init_openbb()

        # Quality tracker
        self._quality_reports: list[DataQualityReport] = []

        logger.info(
            "DataIngestionOrchestrator initialized: %d US equities, %d FTSE, "
            "%d commodity ETFs, %d indices, %d FRED series",
            len(self._equity_universe), len(self._ftse_universe),
            len(COMMODITY_ETFS), len(INDEX_BENCHMARKS), len(FRED_SERIES),
        )

    def _init_openbb(self):
        """Initialize OpenBB data source."""
        try:
            from ..data.openbb_data import get_adj_close, get_returns, get_prices
            self._get_adj_close = get_adj_close
            self._get_returns = get_returns
            self._get_prices = get_prices
            logger.info("OpenBB data source initialized")
        except ImportError:
            self._get_adj_close = None
            self._get_returns = None
            self._get_prices = None
            logger.warning("OpenBB data source not available — using fallback")

    # ── Equity Ingestion ────────────────────────────────────────────────

    def ingest_equities(self) -> IngestionResult:
        """Ingest US equities (S&P 1500) + London FTSE 100.

        NO crypto. Only US + FTSE 100 equities.
        """
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.EQUITY,
            timestamp=datetime.now().isoformat(),
        )

        # Combine US + FTSE universes
        all_tickers = list(self._equity_universe) + list(self._ftse_universe)
        result.symbols_requested = len(all_tickers)

        try:
            # US equities via OpenBB
            us_data = self._fetch_price_data(self._equity_universe, "US Equities")
            if us_data is not None:
                with self._lock:
                    self._data_cache["equity_us"] = us_data
                result.symbols_received += len(us_data)
                result.data_points += sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in us_data.values()
                )
        except Exception as e:
            result.errors.append(f"US equity ingestion error: {e}")
            logger.error("US equity ingestion failed: %s", e)

        try:
            # FTSE 100 via OpenBB
            ftse_data = self._fetch_price_data(self._ftse_universe, "FTSE 100")
            if ftse_data is not None:
                with self._lock:
                    self._data_cache["equity_ftse"] = ftse_data
                result.symbols_received += len(ftse_data)
                result.data_points += sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in ftse_data.values()
                )
        except Exception as e:
            result.errors.append(f"FTSE ingestion error: {e}")
            logger.error("FTSE 100 ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.EQUITY] = datetime.now()
        self._state.last_results[AssetClass.EQUITY] = result

        logger.info(
            "Equities ingested: %d/%d symbols, %d data points (%.0fms)",
            result.symbols_received, result.symbols_requested,
            result.data_points, result.duration_ms,
        )
        return result

    # ── Commodity Ingestion ─────────────────────────────────────────────

    def ingest_commodities(self) -> IngestionResult:
        """Ingest major commodity ETFs for price reference and cyclical patterns."""
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.COMMODITY,
            timestamp=datetime.now().isoformat(),
        )

        tickers = list(COMMODITY_ETFS.keys())
        result.symbols_requested = len(tickers)

        try:
            data = self._fetch_price_data(tickers, "Commodity ETFs")
            if data is not None:
                with self._lock:
                    self._data_cache["commodities"] = data
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )

                # Compute cyclical pattern signals
                self._compute_commodity_cyclical_signals(data)
        except Exception as e:
            result.errors.append(f"Commodity ingestion error: {e}")
            logger.error("Commodity ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.COMMODITY] = datetime.now()
        self._state.last_results[AssetClass.COMMODITY] = result

        logger.info(
            "Commodities ingested: %d/%d ETFs (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── Index Ingestion ─────────────────────────────────────────────────

    def ingest_indices(self) -> IngestionResult:
        """Ingest major index ETFs for benchmarking and monthly rebalancing."""
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.INDEX,
            timestamp=datetime.now().isoformat(),
        )

        tickers = list(INDEX_BENCHMARKS.keys())
        result.symbols_requested = len(tickers)

        try:
            data = self._fetch_price_data(tickers, "Index Benchmarks")
            if data is not None:
                with self._lock:
                    self._data_cache["indices"] = data
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )

                # Compute benchmark returns for constituent comparison
                self._compute_benchmark_returns(data)
        except Exception as e:
            result.errors.append(f"Index ingestion error: {e}")
            logger.error("Index ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.INDEX] = datetime.now()
        self._state.last_results[AssetClass.INDEX] = result

        logger.info(
            "Indices ingested: %d/%d benchmarks (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── Fixed Income Ingestion ──────────────────────────────────────────

    def ingest_fixed_income(self) -> IngestionResult:
        """Ingest fixed income data.

        Constraints:
        - Sovereign: G10 + India + Japan countries
        - Corporate: US credit bonds only (IG + HY)
        - Structured: Major benchmarks only (MBS, ABS, CMBS)
        """
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.FIXED_INCOME,
            timestamp=datetime.now().isoformat(),
        )

        # Sovereign ETFs
        sovereign_tickers = list(SOVEREIGN_FI.keys())
        # Corporate ETFs (US only)
        corporate_tickers = list(CORPORATE_FI.keys())
        # Structured ETFs
        structured_tickers = list(STRUCTURED_FI.keys())

        all_fi = sovereign_tickers + corporate_tickers + structured_tickers
        result.symbols_requested = len(all_fi)

        try:
            data = self._fetch_price_data(all_fi, "Fixed Income")
            if data is not None:
                with self._lock:
                    self._data_cache["fi_sovereign"] = {
                        k: v for k, v in data.items() if k in SOVEREIGN_FI
                    }
                    self._data_cache["fi_corporate"] = {
                        k: v for k, v in data.items() if k in CORPORATE_FI
                    }
                    self._data_cache["fi_structured"] = {
                        k: v for k, v in data.items() if k in STRUCTURED_FI
                    }
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )
        except Exception as e:
            result.errors.append(f"FI ingestion error: {e}")
            logger.error("Fixed income ingestion failed: %s", e)

        # Also ingest FRED yield curves for G10+India+Japan
        try:
            yield_data = self._fetch_fred_series([
                "DGS2", "DGS5", "DGS10", "DGS30", "T10Y2Y", "T10Y3M",
                "BAMLH0A0HYM2", "BAMLC0A0CM",
            ])
            if yield_data:
                with self._lock:
                    self._data_cache["fi_yields"] = yield_data
                result.data_points += sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in yield_data.values()
                )
        except Exception as e:
            result.errors.append(f"FRED yield ingestion error: {e}")

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.FIXED_INCOME] = datetime.now()
        self._state.last_results[AssetClass.FIXED_INCOME] = result

        logger.info(
            "Fixed income ingested: %d/%d instruments, yields: %s (%.0fms)",
            result.symbols_received, result.symbols_requested,
            "OK" if "fi_yields" in self._data_cache else "MISSING",
            result.duration_ms,
        )
        return result

    # ── Currency Ingestion ──────────────────────────────────────────────

    def ingest_currencies(self) -> IngestionResult:
        """Ingest G10 + India + Japan currencies."""
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.CURRENCY,
            timestamp=datetime.now().isoformat(),
        )

        tickers = list(CURRENCY_INSTRUMENTS.keys())
        result.symbols_requested = len(tickers)

        try:
            data = self._fetch_price_data(tickers, "Currencies")
            if data is not None:
                with self._lock:
                    self._data_cache["currencies"] = data
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )

                # Also fetch DXY from FRED
                dxy = self._fetch_fred_series(["DTWEXBGS"])
                if dxy:
                    self._data_cache["dxy"] = dxy
        except Exception as e:
            result.errors.append(f"Currency ingestion error: {e}")
            logger.error("Currency ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.CURRENCY] = datetime.now()
        self._state.last_results[AssetClass.CURRENCY] = result

        logger.info(
            "Currencies ingested: %d/%d pairs (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── Econometric Ingestion ───────────────────────────────────────────

    def ingest_econometrics(self) -> IngestionResult:
        """Ingest FRED macro data for research and cube flow."""
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.ECONOMETRIC,
            timestamp=datetime.now().isoformat(),
        )

        series_ids = list(FRED_SERIES.keys())
        result.symbols_requested = len(series_ids)

        try:
            data = self._fetch_fred_series(series_ids)
            if data:
                with self._lock:
                    self._data_cache["econometrics"] = data
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )

                # Compute velocity and key derived metrics
                self._compute_derived_econometrics(data)
        except Exception as e:
            result.errors.append(f"Econometric ingestion error: {e}")
            logger.error("Econometric ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.ECONOMETRIC] = datetime.now()
        self._state.last_results[AssetClass.ECONOMETRIC] = result

        logger.info(
            "Econometrics ingested: %d/%d FRED series (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── SEC Filing Ingestion ────────────────────────────────────────────

    def ingest_sec_filings(self) -> IngestionResult:
        """Ingest major SEC filings — NOT all daily filings.

        Only: 10-K, 10-Q, 8-K (material events), S-1, DEF 14A.
        Track monthly updates for our universe of securities.
        """
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.SEC_FILING,
            timestamp=datetime.now().isoformat(),
        )

        try:
            filings = self._fetch_sec_filings()
            if filings:
                with self._lock:
                    self._data_cache["sec_filings"] = filings
                result.symbols_received = len(filings)
                result.data_points = sum(
                    len(v) if isinstance(v, list) else 1 for v in filings.values()
                )
        except Exception as e:
            result.errors.append(f"SEC filing ingestion error: {e}")
            logger.error("SEC filing ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.SEC_FILING] = datetime.now()
        self._state.last_results[AssetClass.SEC_FILING] = result

        logger.info(
            "SEC filings ingested: %d symbols with filings (%.0fms)",
            result.symbols_received, result.duration_ms,
        )
        return result

    # ── Options Chain Ingestion ─────────────────────────────────────────

    def ingest_options_chain(self) -> IngestionResult:
        """Ingest options chains for selected securities.

        Options selected opportunistically to maximize profit and alpha.
        """
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.OPTIONS,
            timestamp=datetime.now().isoformat(),
        )

        tickers = self._selected_options
        if not tickers:
            # Default high-liquidity options universe
            tickers = [
                "SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMZN",
                "TSLA", "META", "GOOGL", "AMD", "NFLX",
            ]

        result.symbols_requested = len(tickers)

        try:
            chains = self._fetch_options_chains(tickers)
            if chains:
                with self._lock:
                    self._data_cache["options"] = chains
                result.symbols_received = len(chains)
                result.data_points = sum(
                    len(v) if isinstance(v, (list, dict)) else 1
                    for v in chains.values()
                )
        except Exception as e:
            result.errors.append(f"Options ingestion error: {e}")
            logger.error("Options ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.OPTIONS] = datetime.now()
        self._state.last_results[AssetClass.OPTIONS] = result

        logger.info(
            "Options chains ingested: %d/%d underlyings (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── Futures Ingestion ───────────────────────────────────────────────

    def ingest_futures(self) -> IngestionResult:
        """Ingest futures data for beta management within the corridor."""
        start = time.monotonic()
        result = IngestionResult(
            asset_class=AssetClass.FUTURES,
            timestamp=datetime.now().isoformat(),
        )

        instruments = list(FUTURES_INSTRUMENTS.keys())
        result.symbols_requested = len(instruments)

        try:
            data = self._fetch_futures_data(instruments)
            if data:
                with self._lock:
                    self._data_cache["futures"] = data
                result.symbols_received = len(data)
                result.data_points = sum(
                    len(v) if hasattr(v, '__len__') else 1 for v in data.values()
                )
        except Exception as e:
            result.errors.append(f"Futures ingestion error: {e}")
            logger.error("Futures ingestion failed: %s", e)

        result.duration_ms = (time.monotonic() - start) * 1000
        self._last_ingestion[AssetClass.FUTURES] = datetime.now()
        self._state.last_results[AssetClass.FUTURES] = result

        logger.info(
            "Futures ingested: %d/%d instruments (%.0fms)",
            result.symbols_received, result.symbols_requested, result.duration_ms,
        )
        return result

    # ── Full Cycle ──────────────────────────────────────────────────────

    def run_single_cycle(self) -> dict[str, IngestionResult]:
        """Run a single full ingestion cycle across all asset classes."""
        logger.info("=== INGESTION CYCLE START ===")
        cycle_start = time.monotonic()

        results = {}
        results["equities"] = self.ingest_equities()
        results["commodities"] = self.ingest_commodities()
        results["indices"] = self.ingest_indices()
        results["fixed_income"] = self.ingest_fixed_income()
        results["currencies"] = self.ingest_currencies()
        results["econometrics"] = self.ingest_econometrics()
        results["sec_filings"] = self.ingest_sec_filings()
        results["options"] = self.ingest_options_chain()
        results["futures"] = self.ingest_futures()

        total_ms = (time.monotonic() - cycle_start) * 1000
        self._state.cycles_completed += 1
        self._state.last_cycle_time = datetime.now().isoformat()

        total_points = sum(r.data_points for r in results.values())
        total_errors = sum(len(r.errors) for r in results.values())
        self._state.total_data_points += total_points
        self._state.errors_today += total_errors

        logger.info(
            "=== INGESTION CYCLE COMPLETE: %d data points, %d errors (%.0fms) ===",
            total_points, total_errors, total_ms,
        )

        # Route to engines after ingestion
        self.route_to_engines()

        return results

    def run_continuous_loop(self, heartbeat_sec: float = 60.0):
        """Run continuous ingestion loop with frequency-based scheduling.

        Each asset class is ingested at its configured frequency.
        Default heartbeat is 60 seconds (1-minute cadence).
        """
        self._state.is_running = True
        self._stop_event.clear()

        logger.info("Starting continuous ingestion loop (heartbeat=%.0fs)", heartbeat_sec)

        while not self._stop_event.is_set():
            cycle_start = time.monotonic()
            now = datetime.now()

            try:
                # Check each asset class against its frequency
                for asset_class in AssetClass:
                    if self._should_ingest(asset_class, now):
                        self._ingest_asset_class(asset_class)

                # Route accumulated data to engines
                self.route_to_engines()

                self._state.cycles_completed += 1
                self._state.last_cycle_time = now.isoformat()

            except Exception as e:
                self._state.errors_today += 1
                logger.error("Ingestion loop error: %s", e, exc_info=True)

            # Wait for next heartbeat
            elapsed = time.monotonic() - cycle_start
            sleep_time = max(0, heartbeat_sec - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

        self._state.is_running = False
        logger.info("Ingestion loop stopped after %d cycles", self._state.cycles_completed)

    def stop(self):
        """Signal the continuous loop to stop."""
        self._stop_event.set()
        logger.info("Ingestion stop signal sent")

    # ── Routing ─────────────────────────────────────────────────────────

    def route_to_engines(self):
        """Route ingested data to appropriate engine layers.

        L1 Data:        Equities, indices, commodities, FI, currencies
        L2 Signals:     Macro (econometrics, FI yields), MetadronCube (liquidity)
        L3 Intelligence: SEC filings, cross-asset signals
        L4 Portfolio:   Index benchmarks (rebalancing), beta corridor (futures)
        L7 Execution:   Options chains, futures
        """
        with self._lock:
            routing_map = {
                "L1_data": {
                    "equity_us": self._data_cache.get("equity_us", {}),
                    "equity_ftse": self._data_cache.get("equity_ftse", {}),
                    "indices": self._data_cache.get("indices", {}),
                    "commodities": self._data_cache.get("commodities", {}),
                    "fi_sovereign": self._data_cache.get("fi_sovereign", {}),
                    "fi_corporate": self._data_cache.get("fi_corporate", {}),
                    "currencies": self._data_cache.get("currencies", {}),
                },
                "L2_signals": {
                    "econometrics": self._data_cache.get("econometrics", {}),
                    "fi_yields": self._data_cache.get("fi_yields", {}),
                    "derived_metrics": self._data_cache.get("derived_metrics", {}),
                    "commodity_signals": self._data_cache.get("commodity_signals", {}),
                },
                "L3_intelligence": {
                    "sec_filings": self._data_cache.get("sec_filings", {}),
                },
                "L4_portfolio": {
                    "benchmarks": self._data_cache.get("indices", {}),
                    "futures": self._data_cache.get("futures", {}),
                },
                "L7_execution": {
                    "options": self._data_cache.get("options", {}),
                    "futures": self._data_cache.get("futures", {}),
                },
            }

        # Log routing summary
        for layer, data in routing_map.items():
            total = sum(len(v) for v in data.values() if isinstance(v, dict))
            logger.debug("Routed to %s: %d data items", layer, total)

        return routing_map

    # ── Data Access ─────────────────────────────────────────────────────

    def get_data(self, asset_class: str) -> dict:
        """Get cached data for an asset class."""
        with self._lock:
            return dict(self._data_cache.get(asset_class, {}))

    def get_all_data(self) -> dict:
        """Get all cached data."""
        with self._lock:
            return {k: dict(v) for k, v in self._data_cache.items()}

    def get_state(self) -> IngestionState:
        """Get current ingestion state."""
        return self._state

    def get_equity_universe(self) -> list:
        """Get combined US + FTSE equity universe."""
        return list(self._equity_universe) + list(self._ftse_universe)

    # ── Private Helpers ─────────────────────────────────────────────────

    def _should_ingest(self, asset_class: AssetClass, now: datetime) -> bool:
        """Check if an asset class is due for ingestion based on frequency."""
        freq = self.FREQUENCY_MAP.get(asset_class, IngestionFrequency.DAILY)
        last = self._last_ingestion.get(asset_class)
        if last is None:
            return True

        elapsed = (now - last).total_seconds()

        interval_map = {
            IngestionFrequency.REAL_TIME: 1,
            IngestionFrequency.ONE_MIN: 60,
            IngestionFrequency.FIVE_MIN: 300,
            IngestionFrequency.FIFTEEN_MIN: 900,
            IngestionFrequency.HOURLY: 3600,
            IngestionFrequency.DAILY: 86400,
            IngestionFrequency.WEEKLY: 604800,
            IngestionFrequency.MONTHLY: 2592000,
        }

        return elapsed >= interval_map.get(freq, 60)

    def _ingest_asset_class(self, asset_class: AssetClass):
        """Dispatch ingestion for a specific asset class."""
        dispatch = {
            AssetClass.EQUITY: self.ingest_equities,
            AssetClass.COMMODITY: self.ingest_commodities,
            AssetClass.INDEX: self.ingest_indices,
            AssetClass.FIXED_INCOME: self.ingest_fixed_income,
            AssetClass.CURRENCY: self.ingest_currencies,
            AssetClass.ECONOMETRIC: self.ingest_econometrics,
            AssetClass.SEC_FILING: self.ingest_sec_filings,
            AssetClass.OPTIONS: self.ingest_options_chain,
            AssetClass.FUTURES: self.ingest_futures,
        }
        handler = dispatch.get(asset_class)
        if handler:
            try:
                handler()
            except Exception as e:
                logger.error("Failed to ingest %s: %s", asset_class.value, e)
                self._state.errors_today += 1

    def _fetch_price_data(self, tickers: list, label: str) -> Optional[dict]:
        """Fetch price data via OpenBB or fallback."""
        if not tickers:
            return {}

        data = {}
        if self._get_adj_close is not None:
            try:
                for ticker in tickers:
                    try:
                        prices = self._get_adj_close(ticker)
                        if prices is not None:
                            data[ticker] = prices
                    except Exception:
                        pass  # Skip individual failures
            except Exception as e:
                logger.warning("Bulk price fetch for %s failed: %s", label, e)
        else:
            # Fallback: return empty but log
            logger.debug("No data source for %s — returning empty", label)

        return data if data else {}

    def _fetch_fred_series(self, series_ids: list) -> dict:
        """Fetch FRED economic data series."""
        data = {}
        try:
            from ..data.openbb_data import get_fred_series
            for sid in series_ids:
                try:
                    series = get_fred_series(sid)
                    if series is not None:
                        data[sid] = series
                except Exception:
                    pass
        except ImportError:
            logger.debug("FRED data source not available")

        return data

    def _fetch_sec_filings(self) -> dict:
        """Fetch major SEC filings for universe securities."""
        filings = {}
        try:
            from ..data.openbb_data import get_sec_filings
            # Only major filing types
            for filing_type in SEC_FILING_TYPES:
                try:
                    recent = get_sec_filings(filing_type=filing_type, limit=50)
                    if recent:
                        # Filter to our universe
                        for filing in recent:
                            ticker = filing.get("ticker", "")
                            if ticker in set(self._equity_universe):
                                if ticker not in filings:
                                    filings[ticker] = []
                                filings[ticker].append(filing)
                except Exception:
                    pass
        except ImportError:
            logger.debug("SEC filings data source not available")

        return filings

    def _fetch_options_chains(self, tickers: list) -> dict:
        """Fetch options chains for selected securities."""
        chains = {}
        try:
            from ..data.openbb_data import get_options_chain
            for ticker in tickers:
                try:
                    chain = get_options_chain(ticker)
                    if chain is not None:
                        chains[ticker] = chain
                except Exception:
                    pass
        except ImportError:
            logger.debug("Options data source not available")

        return chains

    def _fetch_futures_data(self, instruments: list) -> dict:
        """Fetch futures data for beta management."""
        data = {}
        try:
            from ..data.openbb_data import get_futures_data
            for inst in instruments:
                try:
                    futures = get_futures_data(inst)
                    if futures is not None:
                        data[inst] = futures
                except Exception:
                    pass
        except ImportError:
            logger.debug("Futures data source not available")

        return data

    def _compute_commodity_cyclical_signals(self, data: dict):
        """Compute cyclical pattern signals from commodity ETF data."""
        if np is None or pd is None:
            return

        signals = {}
        for ticker, prices in data.items():
            try:
                if hasattr(prices, 'values'):
                    arr = np.array(prices.values[-60:] if len(prices) > 60 else prices.values)
                elif isinstance(prices, (list, np.ndarray)):
                    arr = np.array(prices[-60:])
                else:
                    continue

                if len(arr) < 20:
                    continue

                # 20-day return
                ret_20d = (arr[-1] / arr[-20] - 1) if arr[-20] != 0 else 0
                # 60-day return (if available)
                ret_60d = (arr[-1] / arr[0] - 1) if arr[0] != 0 and len(arr) >= 60 else 0
                # Volatility
                daily_returns = np.diff(arr) / arr[:-1]
                vol = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0

                signals[ticker] = {
                    "name": COMMODITY_ETFS.get(ticker, ticker),
                    "return_20d": float(ret_20d),
                    "return_60d": float(ret_60d),
                    "volatility": vol,
                    "trend": "up" if ret_20d > 0.02 else ("down" if ret_20d < -0.02 else "flat"),
                }
            except Exception:
                pass

        with self._lock:
            self._data_cache["commodity_signals"] = signals

    def _compute_benchmark_returns(self, data: dict):
        """Compute benchmark returns for constituent comparison."""
        if np is None:
            return

        returns = {}
        for ticker, prices in data.items():
            try:
                if hasattr(prices, 'values'):
                    arr = np.array(prices.values[-252:] if len(prices) > 252 else prices.values)
                elif isinstance(prices, (list, np.ndarray)):
                    arr = np.array(prices[-252:])
                else:
                    continue

                if len(arr) < 2:
                    continue

                returns[ticker] = {
                    "name": INDEX_BENCHMARKS.get(ticker, ticker),
                    "return_1d": float((arr[-1] / arr[-2] - 1)) if len(arr) >= 2 else 0,
                    "return_5d": float((arr[-1] / arr[-5] - 1)) if len(arr) >= 5 else 0,
                    "return_20d": float((arr[-1] / arr[-20] - 1)) if len(arr) >= 20 else 0,
                    "return_ytd": float((arr[-1] / arr[0] - 1)) if len(arr) >= 20 else 0,
                    "last_price": float(arr[-1]),
                }
            except Exception:
                pass

        with self._lock:
            self._data_cache["benchmark_returns"] = returns

    def _compute_derived_econometrics(self, data: dict):
        """Compute derived econometric signals."""
        derived = {}

        # Money Velocity: V = GDP / M2
        gdp = data.get("GDP")
        m2 = data.get("M2SL")
        if gdp is not None and m2 is not None:
            try:
                if hasattr(gdp, 'iloc') and hasattr(m2, 'iloc'):
                    v = float(gdp.iloc[-1]) / float(m2.iloc[-1]) if float(m2.iloc[-1]) != 0 else 0
                else:
                    v = float(gdp) / float(m2) if float(m2) != 0 else 0
                derived["money_velocity"] = v
            except Exception:
                pass

        # Yield curve steepness
        t10 = data.get("DGS10")
        t2 = data.get("DGS2")
        if t10 is not None and t2 is not None:
            try:
                spread = float(t10.iloc[-1] if hasattr(t10, 'iloc') else t10) - \
                         float(t2.iloc[-1] if hasattr(t2, 'iloc') else t2)
                derived["yield_curve_spread"] = spread
                derived["curve_inverted"] = spread < 0
            except Exception:
                pass

        # Fed balance sheet change
        walcl = data.get("WALCL")
        if walcl is not None and hasattr(walcl, 'iloc') and len(walcl) >= 2:
            try:
                delta = float(walcl.iloc[-1]) - float(walcl.iloc[-2])
                derived["fed_bs_delta"] = delta
                derived["fed_expanding"] = delta > 0
            except Exception:
                pass

        with self._lock:
            self._data_cache["derived_metrics"] = derived

    def validate_data_quality(self) -> list[DataQualityReport]:
        """Run data quality checks across all cached data."""
        reports = []
        now = datetime.now()

        for asset_class in AssetClass:
            last = self._last_ingestion.get(asset_class)
            staleness = (now - last).total_seconds() / 60 if last else float('inf')

            report = DataQualityReport(
                asset_class=asset_class.value,
                timestamp=now.isoformat(),
                has_data=asset_class.value in self._data_cache or \
                         any(asset_class.value in k for k in self._data_cache),
                is_fresh=staleness < 10,
                staleness_minutes=staleness,
                quality_score=max(0, 1.0 - staleness / 60),
            )
            reports.append(report)

        self._quality_reports = reports
        return reports

    def format_status_report(self) -> str:
        """Format a human-readable ingestion status report."""
        lines = [
            "=" * 70,
            "DATA INGESTION ORCHESTRATOR — STATUS REPORT",
            f"Timestamp: {datetime.now().isoformat()}",
            f"Cycles completed: {self._state.cycles_completed}",
            f"Total data points: {self._state.total_data_points:,}",
            f"Errors today: {self._state.errors_today}",
            "=" * 70,
            "",
            f"{'Asset Class':<20} {'Symbols':>8} {'Fresh':>6} {'Last Update':<25}",
            "-" * 65,
        ]

        for ac in AssetClass:
            result = self._state.last_results.get(ac)
            last = self._last_ingestion.get(ac)
            symbols = result.symbols_received if result else 0
            fresh = "YES" if last and (datetime.now() - last).total_seconds() < 600 else "NO"
            last_str = last.strftime("%H:%M:%S") if last else "Never"
            lines.append(f"{ac.value:<20} {symbols:>8} {fresh:>6} {last_str:<25}")

        lines.extend([
            "",
            "CONSTRAINTS ENFORCED:",
            "  - NO crypto ingestion",
            "  - Equities: US S&P 1500 + FTSE 100 only",
            "  - Commodities: Major ETFs only (cyclical patterns + trade signals)",
            "  - Fixed Income: G10+India+Japan sovereign, US corporate only",
            "  - SEC Filings: Major updates only (10-K, 10-Q, 8-K material)",
            "=" * 70,
        ])

        return "\n".join(lines)
