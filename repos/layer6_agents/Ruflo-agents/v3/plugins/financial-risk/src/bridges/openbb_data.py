"""
OpenBB Data Bridge for Ruflo-agents Financial Risk Plugin
==========================================================

Provides market data access through the OpenBB Platform for the
financial-risk plugin's portfolio risk calculations (VaR, CVaR,
Monte Carlo simulations, Sharpe/Sortino ratios).

The economy-bridge.ts WASM module requires historical returns data
to compute risk metrics. This Python bridge fetches that data via
OpenBB and exposes it in formats consumable by the TypeScript layer
(JSON over stdout or via a thin HTTP endpoint).

Part of the Metadron Capital data platform.

Usage
-----
    from openbb_data import get_historical, get_portfolio_returns

    # Single symbol OHLCV
    df = get_historical("AAPL", start_date="2023-01-01")

    # Portfolio returns matrix for risk calculations
    returns = get_portfolio_returns(
        symbols=["AAPL", "MSFT", "GOOGL"],
        start_date="2023-01-01",
    )
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _get_obb():
    """Lazy-import OpenBB to avoid import-time failures when not installed."""
    try:
        from openbb import obb
        return obb
    except ImportError as e:
        raise ImportError(
            "OpenBB Platform is required. Install with: pip install openbb"
        ) from e


def get_historical(
    symbol: str,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    provider: str = "yfinance",
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch historical OHLCV data for a single symbol via OpenBB.

    Parameters
    ----------
    symbol : str
        Ticker symbol (e.g. "AAPL").
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : str, optional
        End date. Defaults to today.
    provider : str
        OpenBB data provider backend.
    interval : str
        Bar interval.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.
    """
    obb = _get_obb()
    result = obb.equity.price.historical(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        provider=provider,
        interval=interval,
    )
    df = result.to_dataframe()
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "datetime"
    return df


def get_portfolio_returns(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch daily returns for a portfolio of symbols.

    This is the primary entry point for the financial-risk plugin's
    economy-bridge, which needs a returns matrix for VaR/CVaR and
    Monte Carlo simulations.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols in the portfolio.
    start_date, end_date : str
        Date range.
    provider : str
        OpenBB data provider backend.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily log returns, one column per symbol.
    """
    close_prices = {}
    for sym in symbols:
        try:
            df = get_historical(sym, start_date, end_date, provider)
            close_prices[sym] = df["close"]
        except Exception:
            logger.warning("Failed to fetch %s, skipping", sym)

    if not close_prices:
        return pd.DataFrame()

    prices = pd.DataFrame(close_prices).dropna()
    returns = np.log(prices / prices.shift(1)).dropna()
    return returns


def get_covariance_matrix(
    symbols: List[str],
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Compute the covariance matrix of returns for the given symbols.

    Used by the economy-bridge Monte Carlo simulation which requires
    a covariance matrix input.

    Parameters
    ----------
    symbols : list of str
        Ticker symbols.
    start_date, end_date, provider
        Same as :func:`get_portfolio_returns`.

    Returns
    -------
    pd.DataFrame
        Covariance matrix (symbols x symbols).
    """
    returns = get_portfolio_returns(symbols, start_date, end_date, provider)
    if returns.empty:
        return pd.DataFrame()
    return returns.cov()


def get_risk_free_rate(provider: str = "yfinance") -> float:
    """Fetch a proxy for the current risk-free rate (13-week T-bill).

    Parameters
    ----------
    provider : str
        OpenBB data provider backend.

    Returns
    -------
    float
        Annualized risk-free rate.
    """
    obb = _get_obb()
    try:
        result = obb.equity.price.historical(
            symbol="^IRX",
            provider=provider,
            interval="1d",
        )
        df = result.to_dataframe()
        rate = df["close"].iloc[-1] / 100.0
        return float(rate)
    except Exception:
        logger.warning("Could not fetch risk-free rate, defaulting to 0.05")
        return 0.05


def to_json(data: pd.DataFrame) -> str:
    """Serialize a DataFrame to JSON for consumption by TypeScript bridge."""
    return data.to_json(orient="split", date_format="iso")


# ---------------------------------------------------------------------------
# CLI entry point: can be called from TypeScript via child_process
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Usage from TypeScript / Node.js:
        python openbb_data.py historical AAPL 2023-01-01 2024-01-01
        python openbb_data.py returns AAPL,MSFT,GOOGL 2023-01-01 2024-01-01
        python openbb_data.py covariance AAPL,MSFT,GOOGL 2023-01-01 2024-01-01
    """
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: openbb_data.py <command> <args...>"}))
        sys.exit(1)

    command = sys.argv[1]
    try:
        if command == "historical":
            symbol = sys.argv[2]
            start = sys.argv[3] if len(sys.argv) > 3 else "2020-01-01"
            end = sys.argv[4] if len(sys.argv) > 4 else None
            df = get_historical(symbol, start, end)
            print(to_json(df))

        elif command == "returns":
            symbols = sys.argv[2].split(",")
            start = sys.argv[3] if len(sys.argv) > 3 else "2020-01-01"
            end = sys.argv[4] if len(sys.argv) > 4 else None
            df = get_portfolio_returns(symbols, start, end)
            print(to_json(df))

        elif command == "covariance":
            symbols = sys.argv[2].split(",")
            start = sys.argv[3] if len(sys.argv) > 3 else "2020-01-01"
            end = sys.argv[4] if len(sys.argv) > 4 else None
            df = get_covariance_matrix(symbols, start, end)
            print(to_json(df))

        elif command == "risk_free_rate":
            rate = get_risk_free_rate()
            print(json.dumps({"risk_free_rate": rate}))

        else:
            print(json.dumps({"error": f"Unknown command: {command}"}))
            sys.exit(1)

    except ImportError:
        print(json.dumps({"error": "OpenBB not installed. pip install openbb"}))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)
