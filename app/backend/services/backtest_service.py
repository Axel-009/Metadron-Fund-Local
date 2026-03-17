"""
Metadron Capital — Backtest service.

Wraps the engine's backtesting infrastructure (engine/ml/backtester.py)
behind a clean service interface consumed by API routes.
"""

import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("metadron.services.backtest")


class BacktestService:
    """High-level interface for running and querying backtests.

    Delegates heavy lifting to the engine's backtester module when
    available; otherwise returns stub / placeholder results.
    """

    def __init__(self):
        self._backtester = None
        try:
            from engine.ml.backtester import Backtester  # type: ignore[import]

            self._backtester = Backtester()
            logger.info("Engine backtester loaded successfully.")
        except ImportError:
            logger.warning(
                "engine.ml.backtester not available — backtest results will be stubs."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        tickers: List[str],
        start: date,
        end: date,
        strategy_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a backtest over the given ticker universe and date range.

        Parameters
        ----------
        tickers : list[str]
            Ticker symbols to include.
        start, end : date
            Backtest window.
        strategy_params : dict, optional
            Arbitrary strategy configuration forwarded to the engine.

        Returns
        -------
        dict
            Backtest result summary including PnL, Sharpe, drawdown, etc.
        """
        if self._backtester is not None:
            try:
                return self._backtester.run(
                    tickers=tickers,
                    start=str(start),
                    end=str(end),
                    params=strategy_params or {},
                )
            except Exception as exc:
                logger.error("Engine backtest failed: %s", exc)
                return self._stub_result(tickers, start, end, error=str(exc))

        return self._stub_result(tickers, start, end)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stub_result(
        tickers: List[str],
        start: date,
        end: date,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a placeholder backtest result."""
        result: Dict[str, Any] = {
            "tickers": tickers,
            "start": str(start),
            "end": str(end),
            "total_return_pct": 0.0,
            "sharpe_ratio": None,
            "max_drawdown_pct": None,
            "trade_count": 0,
            "status": "stub",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if error:
            result["status"] = "error"
            result["error"] = error
        return result
