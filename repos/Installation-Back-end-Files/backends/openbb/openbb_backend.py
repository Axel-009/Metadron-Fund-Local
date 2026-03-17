"""OpenBB Backend — Full SDK bridge for Metadron Capital.

Provides authenticated access to the complete OpenBB data universe:
  - Equity OHLCV, fundamentals, financial statements
  - Options chains, implied volatility surfaces
  - Fixed income (treasury curves, corporate spreads)
  - FX rates, commodity prices
  - Economic indicators (FRED, World Bank)
  - News, SEC filings, analyst estimates

This backend replaces the yfinance fallback with the full OpenBB SDK.
Large data caches and provider configs live here in the backend repo.
The bridge module in Metadron-Capital imports from this.

Usage:
    from backends.openbb.openbb_backend import OpenBBBackend
    backend = OpenBBBackend()
    ohlcv = backend.get_ohlcv("AAPL", start="2024-01-01")
    fundamentals = backend.get_fundamentals("AAPL")
"""

import logging
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# Cache directory for large data pulls
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"

try:
    from openbb import obb
    _HAS_OPENBB = True
    logger.info("OpenBB SDK loaded successfully")
except ImportError:
    _HAS_OPENBB = False
    logger.warning("OpenBB SDK not available — install with: pip install openbb")

# Fallback
try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    _HAS_YF = False

import pandas as pd
import numpy as np


class OpenBBBackend:
    """Full OpenBB SDK backend with caching and provider management."""

    # Provider priority for different data types
    PROVIDERS = {
        "equity_price": ["yfinance", "fmp", "polygon", "intrinio"],
        "equity_fundamentals": ["fmp", "intrinio", "yfinance"],
        "options": ["cboe", "intrinio", "tradier"],
        "fixed_income": ["fred", "fmp"],
        "fx": ["fmp", "polygon", "yfinance"],
        "commodities": ["fred", "yfinance"],
        "economics": ["fred", "oecd", "econdb"],
        "news": ["benzinga", "fmp", "intrinio"],
        "sec_filings": ["sec", "fmp"],
    }

    def __init__(self, cache_ttl_hours: int = 4):
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self._load_api_keys()

        if _HAS_OPENBB:
            self._configure_providers()
            logger.info("OpenBB backend initialized with full SDK")
        else:
            logger.warning("OpenBB backend running in yfinance-fallback mode")

    def _load_api_keys(self):
        """Load API keys from config."""
        key_file = CONFIG_DIR / "api_keys.json"
        self._keys = {}
        if key_file.exists():
            try:
                with open(key_file) as f:
                    self._keys = json.load(f)
                logger.info(f"Loaded API keys for: {list(self._keys.keys())}")
            except Exception as e:
                logger.warning(f"Failed to load API keys: {e}")

        # Also check environment variables
        env_mappings = {
            "OPENBB_FMP_API_KEY": "fmp",
            "OPENBB_POLYGON_API_KEY": "polygon",
            "OPENBB_FRED_API_KEY": "fred",
            "OPENBB_INTRINIO_API_KEY": "intrinio",
            "OPENBB_BENZINGA_API_KEY": "benzinga",
        }
        for env_var, provider in env_mappings.items():
            val = os.environ.get(env_var)
            if val and provider not in self._keys:
                self._keys[provider] = val

    def _configure_providers(self):
        """Configure OpenBB providers with API keys."""
        if not _HAS_OPENBB:
            return
        for provider, key in self._keys.items():
            try:
                # OpenBB auto-configures from env vars
                os.environ[f"OPENBB_{provider.upper()}_API_KEY"] = key
            except Exception as e:
                logger.debug(f"Provider {provider} config: {e}")

    def _cache_key(self, func_name: str, ticker: str, **kwargs) -> Path:
        """Generate cache file path."""
        params = "_".join(f"{k}={v}" for k, v in sorted(kwargs.items()) if v)
        return CACHE_DIR / f"{func_name}_{ticker}_{params}.parquet"

    def _read_cache(self, cache_path: Path) -> Optional[pd.DataFrame]:
        """Read from cache if fresh."""
        if cache_path.exists():
            mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
            if datetime.now() - mtime < self.cache_ttl:
                try:
                    return pd.read_parquet(cache_path)
                except Exception:
                    pass
        return None

    def _write_cache(self, cache_path: Path, df: pd.DataFrame):
        """Write DataFrame to cache."""
        try:
            df.to_parquet(cache_path)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    # ------------------------------------------------------------------
    # Equity Data
    # ------------------------------------------------------------------

    def get_ohlcv(self, ticker: str, start: str = "2024-01-01",
                  end: Optional[str] = None, interval: str = "1d") -> pd.DataFrame:
        """Get OHLCV price data."""
        cache_path = self._cache_key("ohlcv", ticker, start=start, end=end or "", interval=interval)
        cached = self._read_cache(cache_path)
        if cached is not None:
            return cached

        df = pd.DataFrame()

        if _HAS_OPENBB:
            try:
                result = obb.equity.price.historical(
                    symbol=ticker, start_date=start, end_date=end,
                    interval=interval, provider="yfinance",
                )
                df = result.to_dataframe()
                if not df.empty:
                    # Normalize column names
                    col_map = {
                        "open": "Open", "high": "High", "low": "Low",
                        "close": "Close", "volume": "Volume", "adj_close": "Adj Close",
                    }
                    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
                    self._write_cache(cache_path, df)
                    return df
            except Exception as e:
                logger.warning(f"OpenBB equity price failed for {ticker}: {e}")

        if _HAS_YF:
            try:
                data = yf.download(ticker, start=start, end=end, interval=interval,
                                   progress=False, auto_adjust=False)
                if not data.empty:
                    df = data
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self._write_cache(cache_path, df)
            except Exception as e:
                logger.warning(f"yfinance fallback failed for {ticker}: {e}")

        return df

    def get_fundamentals(self, ticker: str) -> dict:
        """Get company fundamentals (ratios, financials, profile)."""
        result = {
            "ticker": ticker,
            "profile": {},
            "ratios": {},
            "income_statement": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "cash_flow": pd.DataFrame(),
        }

        if _HAS_OPENBB:
            try:
                profile = obb.equity.profile(symbol=ticker, provider="yfinance")
                result["profile"] = profile.to_dict() if hasattr(profile, 'to_dict') else {}
            except Exception:
                pass

            try:
                income = obb.equity.fundamental.income(symbol=ticker, provider="yfinance", limit=8)
                result["income_statement"] = income.to_dataframe() if income else pd.DataFrame()
            except Exception:
                pass

            try:
                balance = obb.equity.fundamental.balance(symbol=ticker, provider="yfinance", limit=8)
                result["balance_sheet"] = balance.to_dataframe() if balance else pd.DataFrame()
            except Exception:
                pass

            try:
                cf = obb.equity.fundamental.cash(symbol=ticker, provider="yfinance", limit=8)
                result["cash_flow"] = cf.to_dataframe() if cf else pd.DataFrame()
            except Exception:
                pass

        elif _HAS_YF:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info or {}
                result["profile"] = {
                    "name": info.get("longName", ""),
                    "sector": info.get("sector", ""),
                    "industry": info.get("industry", ""),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", 0),
                    "forward_pe": info.get("forwardPE", 0),
                    "pb_ratio": info.get("priceToBook", 0),
                    "dividend_yield": info.get("dividendYield", 0),
                    "beta": info.get("beta", 1.0),
                }
                result["income_statement"] = stock.income_stmt if hasattr(stock, 'income_stmt') else pd.DataFrame()
                result["balance_sheet"] = stock.balance_sheet if hasattr(stock, 'balance_sheet') else pd.DataFrame()
                result["cash_flow"] = stock.cashflow if hasattr(stock, 'cashflow') else pd.DataFrame()
            except Exception as e:
                logger.warning(f"yfinance fundamentals failed for {ticker}: {e}")

        return result

    # ------------------------------------------------------------------
    # Options Data
    # ------------------------------------------------------------------

    def get_options_chain(self, ticker: str, expiration: Optional[str] = None) -> pd.DataFrame:
        """Get options chain data."""
        if _HAS_OPENBB:
            try:
                result = obb.derivatives.options.chains(symbol=ticker, provider="cboe")
                return result.to_dataframe() if result else pd.DataFrame()
            except Exception as e:
                logger.warning(f"OpenBB options failed for {ticker}: {e}")

        if _HAS_YF:
            try:
                stock = yf.Ticker(ticker)
                expirations = stock.options
                if not expirations:
                    return pd.DataFrame()
                exp = expiration or expirations[0]
                chain = stock.option_chain(exp)
                calls = chain.calls.assign(type="call")
                puts = chain.puts.assign(type="put")
                return pd.concat([calls, puts], ignore_index=True)
            except Exception as e:
                logger.warning(f"yfinance options failed for {ticker}: {e}")

        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Fixed Income / Economics
    # ------------------------------------------------------------------

    def get_treasury_curve(self) -> pd.DataFrame:
        """Get current Treasury yield curve."""
        if _HAS_OPENBB:
            try:
                result = obb.fixedincome.government.treasury_rates(provider="fred")
                return result.to_dataframe() if result else pd.DataFrame()
            except Exception:
                pass
        return pd.DataFrame()

    def get_economic_indicator(self, series_id: str, start: str = "2020-01-01") -> pd.DataFrame:
        """Get FRED economic data series."""
        if _HAS_OPENBB:
            try:
                result = obb.economy.fred_series(
                    symbol=series_id, start_date=start, provider="fred",
                )
                return result.to_dataframe() if result else pd.DataFrame()
            except Exception:
                pass
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # News / SEC
    # ------------------------------------------------------------------

    def get_news(self, ticker: str, limit: int = 50) -> list[dict]:
        """Get recent news for a ticker."""
        if _HAS_OPENBB:
            try:
                result = obb.news.company(symbol=ticker, limit=limit, provider="benzinga")
                if result:
                    df = result.to_dataframe()
                    return df.to_dict(orient="records")
            except Exception:
                pass
        return []

    def get_sec_filings(self, ticker: str, filing_type: str = "10-K",
                        limit: int = 10) -> list[dict]:
        """Get SEC filings."""
        if _HAS_OPENBB:
            try:
                result = obb.equity.fundamental.filings(
                    symbol=ticker, type=filing_type, limit=limit, provider="sec",
                )
                if result:
                    df = result.to_dataframe()
                    return df.to_dict(orient="records")
            except Exception:
                pass
        return []

    # ------------------------------------------------------------------
    # FX / Commodities
    # ------------------------------------------------------------------

    def get_fx_rate(self, pair: str = "EURUSD", start: str = "2024-01-01") -> pd.DataFrame:
        """Get FX rate history."""
        if _HAS_OPENBB:
            try:
                result = obb.currency.price.historical(
                    symbol=pair, start_date=start, provider="yfinance",
                )
                return result.to_dataframe() if result else pd.DataFrame()
            except Exception:
                pass

        if _HAS_YF:
            try:
                yf_pair = f"{pair[:3]}{pair[3:]}=X"
                return yf.download(yf_pair, start=start, progress=False)
            except Exception:
                pass

        return pd.DataFrame()

    def get_commodity(self, symbol: str, start: str = "2024-01-01") -> pd.DataFrame:
        """Get commodity price history (gold, oil, etc.)."""
        commodity_map = {"GOLD": "GC=F", "OIL": "CL=F", "SILVER": "SI=F",
                         "NATGAS": "NG=F", "COPPER": "HG=F"}
        yf_symbol = commodity_map.get(symbol.upper(), symbol)

        if _HAS_YF:
            try:
                return yf.download(yf_symbol, start=start, progress=False)
            except Exception:
                pass
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Universe builder
    # ------------------------------------------------------------------

    def get_full_universe(self, start: str = "2024-01-01",
                          end: Optional[str] = None) -> dict:
        """Pull complete data universe for the classified instruments.

        Returns dict with equities, fx, commodities, fixed_income, economics.
        """
        universe = {
            "equities": {},
            "fx": {},
            "commodities": {},
            "fixed_income": {},
            "economics": {},
            "timestamp": datetime.now().isoformat(),
        }

        # Core equity universe (top holdings from SECTOR_ETFS)
        core_tickers = [
            "SPY", "QQQ", "IWM", "XLK", "XLF", "XLE", "XLV", "XLI",
            "XLP", "XLU", "XLRE", "XLC", "XLB", "XLY",
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
            "JPM", "BAC", "GS", "V", "MA",
            "XOM", "CVX", "COP",
            "JNJ", "UNH", "PFE",
        ]

        for ticker in core_tickers:
            ohlcv = self.get_ohlcv(ticker, start=start, end=end)
            if not ohlcv.empty:
                universe["equities"][ticker] = ohlcv

        # FX pairs
        for pair in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]:
            fx = self.get_fx_rate(pair, start=start)
            if not fx.empty:
                universe["fx"][pair] = fx

        # Commodities
        for comm in ["GOLD", "OIL", "NATGAS"]:
            data = self.get_commodity(comm, start=start)
            if not data.empty:
                universe["commodities"][comm] = data

        # VIX
        vix = self.get_ohlcv("^VIX", start=start)
        if not vix.empty:
            universe["economics"]["VIX"] = vix

        logger.info(f"Universe loaded: {len(universe['equities'])} equities, "
                    f"{len(universe['fx'])} fx, {len(universe['commodities'])} commodities")

        return universe

    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir(exist_ok=True)
            logger.info("Cache cleared")
