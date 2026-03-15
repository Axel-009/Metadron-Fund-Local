"""Unified Yahoo Finance data provider for the entire platform.

Single source of truth for all price, fundamental, and macro data.
Replaces all broker-specific APIs with yfinance.
"""

import time
from datetime import datetime, timedelta
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None


# ---------------------------------------------------------------------------
# Price Data
# ---------------------------------------------------------------------------

def get_prices(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV price data from Yahoo Finance.

    Returns DataFrame with MultiIndex columns (field, ticker) or single-ticker flat.
    """
    if yf is None:
        raise ImportError("yfinance is required: pip install yfinance")
    if isinstance(tickers, str):
        tickers = [tickers]
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    raw = yf.download(
        tickers, start=start, end=end, interval=interval,
        auto_adjust=False, progress=False, threads=True,
    )
    if raw.empty:
        return pd.DataFrame()
    return raw


def get_adj_close(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Get adjusted close prices, always returns DataFrame."""
    raw = get_prices(tickers, start, end)
    if raw.empty:
        return pd.DataFrame()
    if isinstance(raw.columns, pd.MultiIndex):
        top = raw.columns.get_level_values(0)
        key = "Adj Close" if "Adj Close" in top else "Close"
        data = raw[key].copy()
    else:
        data = raw[["Adj Close" if "Adj Close" in raw.columns else "Close"]].copy()
    if isinstance(data, pd.Series):
        data = data.to_frame(tickers[0] if isinstance(tickers, list) else tickers)
    return data


def get_returns(
    tickers: list[str] | str,
    start: str = "2020-01-01",
    end: Optional[str] = None,
    log_returns: bool = True,
) -> pd.DataFrame:
    """Compute returns from adjusted close prices."""
    prices = get_adj_close(tickers, start, end)
    if prices.empty:
        return pd.DataFrame()
    if log_returns:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# Market stats (for beta corridor engine)
# ---------------------------------------------------------------------------

def get_market_stats(
    benchmark: str = "^GSPC",
    lookback_years: int = 1,
) -> dict:
    """Compute annualised drift (Rm), realised vol (sigma), and last price."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback_years * 365)).strftime("%Y-%m-%d")
    prices = get_adj_close(benchmark, start, end)
    if prices.empty:
        return {"Rm": 0.05, "sigma_m": 0.15, "last_price": 5000.0}
    close = prices.iloc[:, 0] if isinstance(prices, pd.DataFrame) else prices
    log_ret = np.log(close / close.shift(1)).dropna()
    sigma_m = float(log_ret.std() * np.sqrt(252))
    Rm = float(log_ret.sum())
    return {"Rm": Rm, "sigma_m": sigma_m, "last_price": float(close.iloc[-1])}


# ---------------------------------------------------------------------------
# Fundamental data
# ---------------------------------------------------------------------------

def get_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental data for a single ticker."""
    if yf is None:
        return {}
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
        return {
            "ticker": ticker,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "pb_ratio": info.get("priceToBook"),
            "dividend_yield": info.get("dividendYield"),
            "roe": info.get("returnOnEquity"),
            "roa": info.get("returnOnAssets"),
            "debt_to_equity": info.get("debtToEquity"),
            "free_cash_flow": info.get("freeCashflow"),
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "profit_margin": info.get("profitMargins"),
            "sector": info.get("sector", ""),
            "industry": info.get("industry", ""),
            "beta": info.get("beta"),
            "52w_high": info.get("fiftyTwoWeekHigh"),
            "52w_low": info.get("fiftyTwoWeekLow"),
            "avg_volume": info.get("averageVolume"),
        }
    except Exception:
        return {"ticker": ticker}


def get_bulk_fundamentals(tickers: list[str]) -> pd.DataFrame:
    """Fetch fundamentals for multiple tickers. Returns DataFrame."""
    records = []
    for t in tickers:
        records.append(get_fundamentals(t))
    return pd.DataFrame(records).set_index("ticker") if records else pd.DataFrame()


# ---------------------------------------------------------------------------
# Macro / benchmark data
# ---------------------------------------------------------------------------

# Key macro ETF proxies (when FRED not available)
MACRO_PROXIES = {
    "SPY": "S&P 500",
    "QQQ": "NASDAQ 100",
    "IWM": "Russell 2000",
    "TLT": "20Y Treasury",
    "IEF": "7-10Y Treasury",
    "SHY": "1-3Y Treasury",
    "LQD": "IG Corporate",
    "HYG": "HY Corporate",
    "GLD": "Gold",
    "USO": "Crude Oil",
    "DXY": "US Dollar Index",
    "^VIX": "VIX",
    "^TNX": "10Y Yield",
    "^FVX": "5Y Yield",
    "^IRX": "3M T-Bill",
}


def get_macro_data(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch macro proxy data via ETFs/indices."""
    tickers = list(MACRO_PROXIES.keys())
    prices = get_adj_close(tickers, start, end)
    if not prices.empty:
        prices.columns = [MACRO_PROXIES.get(c, c) for c in prices.columns]
    return prices


def get_sector_performance(
    start: str = "2020-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Sector ETF performance matrix."""
    from .universe_engine import SECTOR_ETFS
    etfs = list(SECTOR_ETFS.values())
    prices = get_adj_close(etfs, start, end)
    if not prices.empty:
        inv = {v: k for k, v in SECTOR_ETFS.items()}
        prices.columns = [inv.get(c, c) for c in prices.columns]
    return prices
