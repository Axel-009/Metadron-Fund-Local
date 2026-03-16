# ============================================================
# SOURCE: https://github.com/Axel-009/CTA-code
# LAYER:  layer2_signals
# ROLE:   OpenBB data bridge for CTA/managed futures
# ============================================================
"""
Unified OpenBB Universe Module for CTA Trend-Following.

Provides access to the FULL investment universe using OpenBB as the SOLE
data source. Tailored for CTA/trend-following strategies that operate
across futures, FX, commodities, and equity indices.

Asset Classes Covered:
    - EQUITY: Stocks and equity futures proxies
    - FIXED_INCOME: Bond futures proxies and Treasury ETFs
    - COMMODITY: Energy, metals, agriculture futures
    - CRYPTO: Digital assets
    - FX: Major and minor currency pairs
    - ETF: Sector and asset-class ETFs
    - INDEX: Global equity and volatility indices

CTA-Specific Mathematical Foundation:
    Trend Signal (exponential crossover):
        signal_t = EMA(P, fast) - EMA(P, slow)
        normalized_signal = signal_t / ATR(P, n)

    Volatility Targeting:
        position_size = target_vol / (sigma_asset * contract_value)
        where sigma_asset = sqrt(252) * std(r_t, window)

    Carry:
        carry_return = (F_near - F_far) / F_near * (365 / days_between)

    Risk Budgeting:
        w_i = (target_risk / sigma_i) / sum_j(target_risk / sigma_j)

Usage:
    from openbb_data import get_full_universe, get_historical, AssetClass
    from openbb_data import get_multiple, detect_asset_class

    universe = get_full_universe()
    df = get_historical("ES=F", start="2020-01-01")
    dfs = get_multiple(["GC=F", "CL=F", "ZN=F"])
"""

import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Asset Class Enum
# ---------------------------------------------------------------------------

class AssetClass(Enum):
    """Enumeration of all supported asset classes."""
    EQUITY = "EQUITY"
    FIXED_INCOME = "FIXED_INCOME"
    COMMODITY = "COMMODITY"
    CRYPTO = "CRYPTO"
    FX = "FX"
    ETF = "ETF"
    INDEX = "INDEX"


# ---------------------------------------------------------------------------
# GICS Classification (Complete Hierarchy - All 11 Sectors)
# ---------------------------------------------------------------------------

GICS_CLASSIFICATION: dict[str, dict[str, dict[str, list[str]]]] = {
    "10 - Energy": {
        "1010 - Energy": {
            "101010 - Energy Equipment & Services": [
                "10101010 - Oil & Gas Drilling",
                "10101020 - Oil & Gas Equipment & Services",
            ],
            "101020 - Oil, Gas & Consumable Fuels": [
                "10102010 - Integrated Oil & Gas",
                "10102020 - Oil & Gas Exploration & Production",
                "10102030 - Oil & Gas Refining & Marketing",
                "10102040 - Oil & Gas Storage & Transportation",
                "10102050 - Coal & Consumable Fuels",
            ],
        },
    },
    "15 - Materials": {
        "1510 - Materials": {
            "151010 - Chemicals": [
                "15101010 - Commodity Chemicals",
                "15101020 - Diversified Chemicals",
                "15101030 - Fertilizers & Agricultural Chemicals",
                "15101040 - Industrial Gases",
                "15101050 - Specialty Chemicals",
            ],
            "151040 - Metals & Mining": [
                "15104010 - Aluminum",
                "15104020 - Diversified Metals & Mining",
                "15104030 - Gold",
                "15104050 - Steel",
            ],
        },
    },
    "20 - Industrials": {
        "2010 - Capital Goods": {
            "201010 - Aerospace & Defense": ["20101010 - Aerospace & Defense"],
            "201060 - Machinery": [
                "20106010 - Construction Machinery & Heavy Transportation Equipment",
                "20106020 - Industrial Machinery & Supplies & Components",
            ],
        },
        "2030 - Transportation": {
            "203040 - Ground Transportation": [
                "20304010 - Rail Transportation",
                "20304020 - Cargo Ground Transportation",
            ],
        },
    },
    "25 - Consumer Discretionary": {
        "2510 - Automobiles & Components": {
            "251020 - Automobiles": ["25102010 - Automobile Manufacturers"],
        },
        "2550 - Retailing": {
            "255030 - Broadline Retail": ["25503010 - Broadline Retail"],
        },
    },
    "30 - Consumer Staples": {
        "3020 - Food, Beverage & Tobacco": {
            "302010 - Beverages": ["30201030 - Soft Drinks & Non-alcoholic Beverages"],
            "302020 - Food Products": ["30202030 - Packaged Foods & Meats"],
        },
    },
    "35 - Health Care": {
        "3520 - Pharmaceuticals, Biotechnology & Life Sciences": {
            "352010 - Biotechnology": ["35201010 - Biotechnology"],
            "352020 - Pharmaceuticals": ["35202010 - Pharmaceuticals"],
        },
    },
    "40 - Financials": {
        "4010 - Banks": {
            "401010 - Banks": [
                "40101010 - Diversified Banks",
                "40101015 - Regional Banks",
            ],
        },
        "4020 - Financial Services": {
            "402030 - Capital Markets": [
                "40203010 - Asset Management & Custody Banks",
                "40203020 - Investment Banking & Brokerage",
            ],
        },
    },
    "45 - Information Technology": {
        "4510 - Software & Services": {
            "451030 - Software": [
                "45103010 - Application Software",
                "45103020 - Systems Software",
            ],
        },
        "4530 - Semiconductors & Semiconductor Equipment": {
            "453010 - Semiconductors & Semiconductor Equipment": [
                "45301020 - Semiconductors",
            ],
        },
    },
    "50 - Communication Services": {
        "5020 - Media & Entertainment": {
            "502030 - Interactive Media & Services": [
                "50203010 - Interactive Media & Services",
            ],
        },
    },
    "55 - Utilities": {
        "5510 - Utilities": {
            "551010 - Electric Utilities": ["55101010 - Electric Utilities"],
            "551030 - Multi-Utilities": ["55103010 - Multi-Utilities"],
        },
    },
    "60 - Real Estate": {
        "6010 - Equity Real Estate Investment Trusts": {
            "601080 - Specialized REITs": [
                "60108010 - Data Center REITs",
                "60108040 - Telecom Tower REITs",
            ],
        },
    },
}


# ---------------------------------------------------------------------------
# Hardcoded Constituent Lists
# ---------------------------------------------------------------------------

SP500_TOP_50 = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "BRK-B", "TSLA",
    "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO", "CVX",
    "MRK", "ABBV", "LLY", "COST", "PEP", "KO", "ADBE", "WMT", "MCD",
    "CSCO", "CRM", "ACN", "ABT", "TMO", "DHR", "LIN", "CMCSA", "NKE",
    "VZ", "NEE", "TXN", "PM", "INTC", "RTX", "HON", "UNP", "LOW",
    "QCOM", "ORCL", "AMGN", "IBM", "BA",
]

EQUITY_GICS_MAP: dict[str, str] = {
    "AAPL": "45 - Information Technology", "MSFT": "45 - Information Technology",
    "AMZN": "25 - Consumer Discretionary", "NVDA": "45 - Information Technology",
    "GOOGL": "50 - Communication Services", "META": "50 - Communication Services",
    "BRK-B": "40 - Financials", "TSLA": "25 - Consumer Discretionary",
    "UNH": "35 - Health Care", "XOM": "10 - Energy",
    "JNJ": "35 - Health Care", "JPM": "40 - Financials",
    "V": "40 - Financials", "PG": "30 - Consumer Staples",
    "MA": "40 - Financials", "HD": "25 - Consumer Discretionary",
    "AVGO": "45 - Information Technology", "CVX": "10 - Energy",
    "MRK": "35 - Health Care", "ABBV": "35 - Health Care",
    "LLY": "35 - Health Care", "COST": "30 - Consumer Staples",
    "PEP": "30 - Consumer Staples", "KO": "30 - Consumer Staples",
    "ADBE": "45 - Information Technology", "WMT": "30 - Consumer Staples",
    "MCD": "25 - Consumer Discretionary", "CSCO": "45 - Information Technology",
    "CRM": "45 - Information Technology", "ACN": "45 - Information Technology",
    "ABT": "35 - Health Care", "TMO": "35 - Health Care",
    "DHR": "35 - Health Care", "LIN": "15 - Materials",
    "CMCSA": "50 - Communication Services", "NKE": "25 - Consumer Discretionary",
    "VZ": "50 - Communication Services", "NEE": "55 - Utilities",
    "TXN": "45 - Information Technology", "PM": "30 - Consumer Staples",
    "INTC": "45 - Information Technology", "RTX": "20 - Industrials",
    "HON": "20 - Industrials", "UNP": "20 - Industrials",
    "LOW": "25 - Consumer Discretionary", "QCOM": "45 - Information Technology",
    "ORCL": "45 - Information Technology", "AMGN": "35 - Health Care",
    "IBM": "45 - Information Technology", "BA": "20 - Industrials",
}

MAJOR_BONDS = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "TIP", "AGG", "BND",
    "MUB", "EMB", "BNDX", "GOVT",
]

# CTA-focused commodity futures universe
MAJOR_COMMODITIES = [
    "GC=F",   # Gold
    "SI=F",   # Silver
    "CL=F",   # Crude Oil WTI
    "BZ=F",   # Brent Crude
    "NG=F",   # Natural Gas
    "HG=F",   # Copper
    "PL=F",   # Platinum
    "PA=F",   # Palladium
    "ZC=F",   # Corn
    "ZW=F",   # Wheat
    "ZS=F",   # Soybean
    "KC=F",   # Coffee
    "SB=F",   # Sugar
    "CT=F",   # Cotton
    "LE=F",   # Live Cattle
]

CRYPTO_TOP_20 = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "AVAX-USD", "DOT-USD", "LINK-USD",
    "MATIC-USD", "UNI-USD", "ATOM-USD", "LTC-USD", "FIL-USD",
    "NEAR-USD", "APT-USD", "ARB-USD", "OP-USD", "MKR-USD",
]

FX_MAJORS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "NZDUSD=X", "USDCAD=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURCHF=X", "USDMXN=X", "USDZAR=X", "USDTRY=X",
    "USDCNY=X", "USDINR=X", "USDBRL=X",
]

MAJOR_ETFS = [
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "EFA", "EEM",
    "VWO", "GLD", "SLV", "USO", "XLF", "XLK", "XLE", "XLV",
    "XLI", "XLP", "XLU", "XLB", "XLRE", "XLC", "XLY",
    "ARKK", "VIG", "SCHD", "VYM", "DVY",
]

GLOBAL_INDICES = [
    "^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX", "^FTSE", "^GDAXI",
    "^FCHI", "^N225", "^HSI", "000001.SS", "^STOXX50E", "^BVSP",
    "^AXJO", "^KS11", "^NSEI", "^GSPTSE",
]

# CTA-specific: futures proxies grouped by asset class for trend following
CTA_FUTURES_UNIVERSE: dict[str, list[str]] = {
    "equity_index": ["SPY", "QQQ", "IWM", "EFA", "EEM"],
    "fixed_income": ["TLT", "IEF", "SHY", "AGG"],
    "energy": ["CL=F", "BZ=F", "NG=F"],
    "metals": ["GC=F", "SI=F", "HG=F", "PL=F"],
    "agriculture": ["ZC=F", "ZW=F", "ZS=F", "KC=F", "SB=F", "CT=F"],
    "fx": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD"],
}


# ---------------------------------------------------------------------------
# Symbol -> Asset Class Detection
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = {"-USD", "-USDT", "-BTC", "-ETH"}
_FX_SUFFIX = "=X"
_FUTURES_SUFFIX = "=F"

_BOND_SYMBOLS = set(MAJOR_BONDS)
_COMMODITY_SYMBOLS = set(MAJOR_COMMODITIES)
_ETF_SYMBOLS = set(MAJOR_ETFS)
_INDEX_SYMBOLS = set(GLOBAL_INDICES)
_CRYPTO_SYMBOLS = set(CRYPTO_TOP_20)
_EQUITY_SYMBOLS = set(SP500_TOP_50)


def detect_asset_class(symbol: str) -> AssetClass:
    """
    Detect the asset class for a given symbol.

    Detection priority:
        1. Known index symbols or '^' prefix -> INDEX
        2. Crypto suffixes (-USD, -USDT) -> CRYPTO
        3. FX suffix (=X) excluding commodities -> FX
        4. Futures suffix (=F) or commodity set -> COMMODITY
        5. Bond ETF set -> FIXED_INCOME
        6. ETF set -> ETF
        7. Default -> EQUITY

    Parameters
    ----------
    symbol : str
        Ticker symbol to classify.

    Returns
    -------
    AssetClass
    """
    if symbol in _INDEX_SYMBOLS or symbol.startswith("^"):
        return AssetClass.INDEX
    if any(symbol.upper().endswith(s) for s in _CRYPTO_SUFFIXES) or symbol in _CRYPTO_SYMBOLS:
        return AssetClass.CRYPTO
    if symbol.endswith(_FX_SUFFIX) and symbol not in _COMMODITY_SYMBOLS:
        return AssetClass.FX
    if symbol.endswith(_FUTURES_SUFFIX) or symbol in _COMMODITY_SYMBOLS:
        return AssetClass.COMMODITY
    if symbol in _BOND_SYMBOLS:
        return AssetClass.FIXED_INCOME
    if symbol in _ETF_SYMBOLS:
        return AssetClass.ETF
    return AssetClass.EQUITY


# ---------------------------------------------------------------------------
# Universe Retrieval Functions
# ---------------------------------------------------------------------------

def get_full_universe() -> dict[AssetClass, list[str]]:
    """
    Return the complete investment universe organized by asset class.

    Returns
    -------
    dict[AssetClass, list[str]]
    """
    return {
        AssetClass.EQUITY: SP500_TOP_50.copy(),
        AssetClass.FIXED_INCOME: MAJOR_BONDS.copy(),
        AssetClass.COMMODITY: MAJOR_COMMODITIES.copy(),
        AssetClass.CRYPTO: CRYPTO_TOP_20.copy(),
        AssetClass.FX: FX_MAJORS.copy(),
        AssetClass.ETF: MAJOR_ETFS.copy(),
        AssetClass.INDEX: GLOBAL_INDICES.copy(),
    }


def get_equity_universe() -> list[str]:
    """Return equity constituent list."""
    return SP500_TOP_50.copy()


def get_bond_universe() -> list[str]:
    """Return fixed-income constituent list."""
    return MAJOR_BONDS.copy()


def get_commodity_universe() -> list[str]:
    """Return commodity futures constituent list."""
    return MAJOR_COMMODITIES.copy()


def get_crypto_universe() -> list[str]:
    """Return top 20 crypto constituent list."""
    return CRYPTO_TOP_20.copy()


def get_fx_universe() -> list[str]:
    """Return FX major pairs list."""
    return FX_MAJORS.copy()


def get_cta_universe() -> dict[str, list[str]]:
    """Return the CTA-specific futures universe grouped by sub-class."""
    return {k: v.copy() for k, v in CTA_FUTURES_UNIVERSE.items()}


def classify_by_gics(symbols: Optional[list[str]] = None) -> dict[str, dict[str, str]]:
    """
    Classify equity symbols by GICS sector.

    Parameters
    ----------
    symbols : list[str], optional
        Symbols to classify. Defaults to SP500_TOP_50.

    Returns
    -------
    dict[str, dict[str, str]]
    """
    if symbols is None:
        symbols = SP500_TOP_50
    return {
        sym: {"sector": EQUITY_GICS_MAP.get(sym, "Unknown"), "in_map": sym in EQUITY_GICS_MAP}
        for sym in symbols
    }


# ---------------------------------------------------------------------------
# OpenBB Data Retrieval (SOLE data source)
# ---------------------------------------------------------------------------

def _get_obb():
    """Lazy import of OpenBB SDK."""
    from openbb import obb
    return obb


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names to title-case: Date, Open, High, Low, Close, Volume."""
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower == "date":
            rename_map[col] = "Date"
        elif lower == "open":
            rename_map[col] = "Open"
        elif lower == "high":
            rename_map[col] = "High"
        elif lower == "low":
            rename_map[col] = "Low"
        elif lower in ("close", "adj close", "adj_close"):
            rename_map[col] = "Close"
        elif lower == "volume":
            rename_map[col] = "Volume"
    df = df.rename(columns=rename_map)

    if "Date" not in df.columns and df.index.name and df.index.name.lower() == "date":
        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "Date"})
    return df


def _resolve_dates(
    start: Optional[str], end: Optional[str], default_lookback_days: int = 1825
) -> tuple[str, str]:
    """Resolve start/end dates. CTA default: 5 years lookback."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    if start is None:
        start = (datetime.now() - timedelta(days=default_lookback_days)).strftime("%Y-%m-%d")
    return start, end


def get_historical(
    symbol: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data using OpenBB as the sole data source.

    CTA Mathematical Notes:
        Trend signal (EMA crossover):
            EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
            where alpha = 2 / (span + 1)
            signal = EMA(P, fast) - EMA(P, slow)

        Average True Range:
            TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
            ATR_n = EMA(TR, n)

        Volatility targeting:
            sigma_ann = sqrt(252) * std(log_returns, window)
            position_size = target_vol / sigma_ann

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g., "ES=F", "GC=F", "AAPL", "BTC-USD").
    start : str, optional
        Start date YYYY-MM-DD. Defaults to 5 years ago.
    end : str, optional
        End date YYYY-MM-DD. Defaults to today.
    provider : str
        OpenBB provider hint (default "yfinance").

    Returns
    -------
    pd.DataFrame
        Standardized OHLCV DataFrame.

    Raises
    ------
    RuntimeError
        If OpenBB cannot fetch the data.
    """
    start, end = _resolve_dates(start, end, default_lookback_days=1825)
    asset_class = detect_asset_class(symbol)
    obb = _get_obb()

    logger.info("Fetching %s [%s] via OpenBB (%s)", symbol, asset_class.value, provider)

    try:
        if asset_class == AssetClass.CRYPTO:
            result = obb.crypto.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        elif asset_class == AssetClass.FX:
            result = obb.currency.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )
        else:
            result = obb.equity.price.historical(
                symbol=symbol, start_date=start, end_date=end, provider=provider
            )

        df = result.to_dataframe()
        if df.empty:
            raise RuntimeError(f"OpenBB returned empty DataFrame for {symbol}")
        logger.info("OpenBB returned %d rows for %s", len(df), symbol)
        return _normalise(df)

    except ImportError:
        raise RuntimeError("OpenBB SDK is not installed. Install with: pip install openbb")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch {symbol} via OpenBB: {exc}") from exc


def get_multiple(
    symbols: list[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    provider: str = "yfinance",
) -> dict[str, pd.DataFrame]:
    """
    Fetch historical data for multiple symbols.

    Parameters
    ----------
    symbols : list[str]
        List of ticker symbols.
    start, end, provider
        Passed through to get_historical.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol -> DataFrame. Failures are logged and omitted.
    """
    result = {}
    for sym in symbols:
        try:
            result[sym] = get_historical(sym, start=start, end=end, provider=provider)
        except Exception as exc:
            logger.error("Skipping %s: %s", sym, exc)
    return result


def get_fundamentals(
    symbol: str,
    provider: str = "yfinance",
) -> dict[str, Any]:
    """
    Fetch fundamental data for an equity symbol via OpenBB.

    Valuation metrics:
        P/E = Price / EPS
        EV/EBITDA = (Market Cap + Debt - Cash) / EBITDA
        P/B = Price / Book Value per Share

    Parameters
    ----------
    symbol : str
        Equity ticker symbol.
    provider : str
        OpenBB provider.

    Returns
    -------
    dict[str, Any]
    """
    obb = _get_obb()
    result = {}

    try:
        overview = obb.equity.fundamental.overview(symbol=symbol, provider=provider)
        overview_df = overview.to_dataframe()
        if not overview_df.empty:
            row = overview_df.iloc[0]
            for key in [
                "market_cap", "pe_ratio", "forward_pe", "price_to_book",
                "dividend_yield", "beta", "eps", "sector", "industry",
            ]:
                result[key] = row.get(key, None)
    except Exception as exc:
        logger.warning("Could not fetch fundamentals for %s: %s", symbol, exc)

    return result


def get_macro_data(
    indicator: str = "GDP",
    country: str = "united_states",
    provider: str = "fred",
) -> pd.DataFrame:
    """
    Fetch macroeconomic data via OpenBB.

    Key CTA macro indicators:
        FEDFUNDS:  Federal Funds Rate (monetary policy direction -> bond trends)
        T10Y2Y:    10Y-2Y spread (recession predictor -> risk-off signal)
        DCOILWTICO: WTI crude oil price (energy trend confirmation)
        GOLDAMGBD228NLBM: Gold price (safe haven demand)

    Parameters
    ----------
    indicator : str
        FRED series ID.
    country : str
        Country.
    provider : str
        OpenBB provider.

    Returns
    -------
    pd.DataFrame
    """
    obb = _get_obb()
    try:
        result = obb.economy.fred_series(symbol=indicator, provider=provider)
        df = result.to_dataframe()
        if not df.empty:
            return df
    except Exception as exc:
        logger.warning("Failed to fetch macro data for %s: %s", indicator, exc)
    raise RuntimeError(f"Could not fetch macro indicator: {indicator}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Full Universe ===")
    universe = get_full_universe()
    for ac, symbols in universe.items():
        print(f"  {ac.value}: {len(symbols)} symbols")

    print("\n=== CTA Futures Universe ===")
    cta = get_cta_universe()
    for group, syms in cta.items():
        print(f"  {group}: {syms}")

    print("\n=== Asset Class Detection ===")
    test_symbols = ["ES=F", "GC=F", "BTC-USD", "EURUSD=X", "^GSPC", "SPY", "TLT", "AAPL"]
    for sym in test_symbols:
        print(f"  {sym}: {detect_asset_class(sym).value}")

    print("\n=== Historical Data (CL=F) ===")
    try:
        df = get_historical("CL=F")
        print(f"  Shape: {df.shape}")
        print(df.head())
    except RuntimeError as e:
        print(f"  Skipped: {e}")
