"""
OpenBB Bridge Module
====================

Provides the same interface as yfinance.Ticker but routes data requests
through OpenBB Platform when available, falling back to yfinance otherwise.

Usage:
    from openbb_bridge import OpenBBTicker, get_data

    # Drop-in replacement for yfinance.Ticker
    ticker = OpenBBTicker("AAPL")
    hist = ticker.history(start="2024-01-01", end="2024-06-01")
    info = ticker.info

    # Convenience function
    df = get_data("AAPL", start="2024-01-01", end="2024-06-01", interval="1d")
"""

import pandas as pd

# Try to import OpenBB; track availability for runtime fallback.
_OPENBB_AVAILABLE = False
try:
    from openbb import obb  # noqa: F401
    _OPENBB_AVAILABLE = True
except ImportError:
    obb = None

# yfinance is always expected to be present (this repo *is* a yfinance fork).
from yfinance import Ticker as _YFinanceTicker


# ---------------------------------------------------------------------------
# Interval mapping: yfinance intervals -> OpenBB provider-friendly intervals
# ---------------------------------------------------------------------------
_INTERVAL_MAP = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "1h",
    "1h": "1h",
    "90m": "90m",
    "1d": "1d",
    "5d": "5d",
    "1wk": "1W",
    "1mo": "1M",
    "3mo": "3M",
}


class OpenBBTicker:
    """Drop-in replacement for ``yfinance.Ticker`` that prefers OpenBB.

    When OpenBB Platform is installed, ``history()`` and ``info`` are served
    via ``openbb.obb``.  If OpenBB is unavailable or a call fails, the class
    transparently falls back to the underlying yfinance implementation.

    Parameters
    ----------
    ticker : str
        The ticker symbol (e.g. ``"AAPL"``).
    """

    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self._yf = _YFinanceTicker(self.ticker)
        self._cached_info: dict | None = None

    # ------------------------------------------------------------------
    # history()
    # ------------------------------------------------------------------
    def history(
        self,
        period: str = "1mo",
        interval: str = "1d",
        start: str | None = None,
        end: str | None = None,
        prepost: bool = False,
        actions: bool = True,
        auto_adjust: bool = True,
        back_adjust: bool = False,
        repair: bool = False,
        keepna: bool = False,
        rounding: bool = False,
        timeout: int = 10,
        raise_errors: bool = False,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Mirrors the ``yfinance.Ticker.history`` signature.  When OpenBB is
        available the request is routed through
        ``obb.equity.price.historical``; on failure it falls back to yfinance.
        """
        if _OPENBB_AVAILABLE:
            try:
                return self._history_openbb(
                    start=start,
                    end=end,
                    interval=interval,
                    period=period,
                )
            except Exception:
                pass  # fall through to yfinance

        return self._yf.history(
            period=period,
            interval=interval,
            start=start,
            end=end,
            prepost=prepost,
            actions=actions,
            auto_adjust=auto_adjust,
            back_adjust=back_adjust,
            repair=repair,
            keepna=keepna,
            rounding=rounding,
            timeout=timeout,
            raise_errors=raise_errors,
        )

    def _history_openbb(
        self,
        start: str | None,
        end: str | None,
        interval: str,
        period: str,
    ) -> pd.DataFrame:
        """Call OpenBB equity price historical endpoint and return a DataFrame
        whose columns match yfinance conventions (Open, High, Low, Close,
        Volume)."""
        obb_interval = _INTERVAL_MAP.get(interval, interval)

        kwargs: dict = {
            "symbol": self.ticker,
            "interval": obb_interval,
            "provider": "yfinance",  # OpenBB can use yfinance as a provider
        }
        if start is not None:
            kwargs["start_date"] = str(start)
        if end is not None:
            kwargs["end_date"] = str(end)

        result = obb.equity.price.historical(**kwargs)  # type: ignore[union-attr]
        df = result.to_dataframe()

        # Normalise column names to match yfinance output.
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "adj_close": "Adj Close",
            "date": "Date",
        }
        df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

        if "Date" in df.columns:
            df.set_index("Date", inplace=True)

        return df

    # ------------------------------------------------------------------
    # info property
    # ------------------------------------------------------------------
    @property
    def info(self) -> dict:
        """Company / security metadata.

        Attempts ``obb.equity.profile`` first, then falls back to
        ``yfinance.Ticker.info``.
        """
        if self._cached_info is not None:
            return self._cached_info

        if _OPENBB_AVAILABLE:
            try:
                self._cached_info = self._info_openbb()
                return self._cached_info
            except Exception:
                pass

        self._cached_info = self._yf.info
        return self._cached_info

    def _info_openbb(self) -> dict:
        """Retrieve company profile via OpenBB and return as a dict."""
        result = obb.equity.profile(  # type: ignore[union-attr]
            symbol=self.ticker,
            provider="yfinance",
        )
        df = result.to_dataframe()
        if df.empty:
            raise ValueError("OpenBB returned empty profile")
        return df.iloc[0].to_dict()

    # ------------------------------------------------------------------
    # Convenience pass-through so callers can still reach yfinance attrs
    # ------------------------------------------------------------------
    def __getattr__(self, name: str):
        """Proxy any attribute not defined here to the underlying yfinance
        Ticker so that ``OpenBBTicker`` remains a superset of
        ``yfinance.Ticker``."""
        return getattr(self._yf, name)

    def __repr__(self) -> str:
        backend = "openbb" if _OPENBB_AVAILABLE else "yfinance"
        return f"OpenBBTicker('{self.ticker}', backend={backend})"


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV data for *ticker* via OpenBB (preferred) or yfinance.

    Parameters
    ----------
    ticker : str
        Symbol, e.g. ``"AAPL"``.
    start : str or None
        Start date in ``YYYY-MM-DD`` format.
    end : str or None
        End date in ``YYYY-MM-DD`` format.
    interval : str
        Bar interval (default ``"1d"``).  Accepts yfinance-style strings.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``Open``, ``High``, ``Low``, ``Close``,
        ``Volume`` (and possibly ``Adj Close``).
    """
    t = OpenBBTicker(ticker)
    return t.history(start=start, end=end, interval=interval)
