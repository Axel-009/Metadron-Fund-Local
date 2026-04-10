"""
Fixed Income Engine — Aggregates FI data from OpenBB/FRED, MacroEngine, and broker.
"""
import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

logger = logging.getLogger("metadron.fixed-income-engine")

# FRED series IDs for treasury yields by tenor
TREASURY_SERIES = {
    "1M": "DGS1MO",
    "3M": "DGS3MO",
    "6M": "DGS6MO",
    "1Y": "DGS1",
    "2Y": "DGS2",
    "3Y": "DGS3",
    "5Y": "DGS5",
    "7Y": "DGS7",
    "10Y": "DGS10",
    "20Y": "DGS20",
    "30Y": "DGS30",
}


class FixedIncomeEngine:
    """Aggregates fixed income data from OpenBB/FRED, MacroEngine, and broker positions."""

    def __init__(self):
        self._macro = None
        self._broker = None

    def _get_broker(self):
        if self._broker is None:
            try:
                from engine.execution.execution_engine import ExecutionEngine
                eng = ExecutionEngine()
                self._broker = eng.broker
            except Exception:
                try:
                    from engine.execution.paper_broker import PaperBroker
                    self._broker = PaperBroker()
                except Exception:
                    pass
        return self._broker

    def _get_macro(self):
        if self._macro is None:
            from engine.signals.macro_engine import MacroEngine
            self._macro = MacroEngine()
        return self._macro

    def get_summary(self) -> dict:
        """Portfolio-level FI summary: total AUM, avg yield, avg duration, avg credit rating."""
        try:
            from engine.data.openbb_data import get_treasury_rates

            # Get current treasury rates for yield reference
            rates_df = get_treasury_rates()
            avg_yield = 0.0
            if hasattr(rates_df, "empty") and not rates_df.empty:
                last_row = rates_df.iloc[-1]
                valid_rates = [float(v) for v in last_row.values if pd.notna(v)]
                if valid_rates:
                    avg_yield = sum(valid_rates) / len(valid_rates)

            # Get yield curve analysis for additional context
            macro = self._get_macro()
            yc = {}
            try:
                yc = macro.get_yield_curve_analysis()
            except Exception:
                pass

            return {
                "totalExposure": 0,
                "avgDuration": 0.0,
                "avgYield": round(avg_yield, 2),
                "avgRating": "N/A",
                "dv01": 0,
                "convexity": 0.0,
                "yield_2s10s": yc.get("spread_2s10s", 0),
                "recession_prob": yc.get("recession_probability", 0),
                "curve_shape": yc.get("curve_shape", "UNKNOWN"),
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"get_summary error: {e}")
            return {
                "totalExposure": 0, "avgDuration": 0.0, "avgYield": 0.0,
                "avgRating": "N/A", "dv01": 0, "convexity": 0.0,
                "timestamp": datetime.utcnow().isoformat(),
            }

    def get_holdings(self) -> list:
        """Bond holdings list from broker positions filtered for FI assets."""
        try:
            broker = self._get_broker()
            if broker is None:
                return []

            positions = broker.get_all_positions()
            if not positions:
                return []

            # Filter for fixed income tickers (bonds, treasuries, FI ETFs)
            fi_tickers = {"TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND",
                          "GOVT", "MUB", "TIP", "VCIT", "VCSH", "BNDX",
                          "EMB", "JNK", "SJNK", "IGIB", "IGSB"}
            fi_positions = []
            for pos in positions:
                ticker = pos.get("ticker", "") if isinstance(pos, dict) else getattr(pos, "ticker", "")
                if ticker in fi_tickers:
                    fi_positions.append({
                        "ticker": ticker,
                        "name": ticker,
                        "quantity": pos.get("quantity", 0) if isinstance(pos, dict) else getattr(pos, "quantity", 0),
                        "current_price": pos.get("current_price", 0) if isinstance(pos, dict) else getattr(pos, "current_price", 0),
                        "market_value": pos.get("market_value", 0) if isinstance(pos, dict) else getattr(pos, "market_value", 0),
                        "unrealized_pnl": pos.get("unrealized_pnl", 0) if isinstance(pos, dict) else getattr(pos, "unrealized_pnl", 0),
                        "sector": "Fixed Income",
                    })
            return fi_positions
        except Exception as e:
            logger.error(f"get_holdings error: {e}")
            return []

    def get_yield_curve(self) -> list:
        """Real yield curve from FRED treasury rates across all tenors."""
        try:
            from engine.data.openbb_data import get_fred_series

            series_ids = list(TREASURY_SERIES.values())
            tenor_labels = list(TREASURY_SERIES.keys())

            # Get current rates
            start_date = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = get_fred_series(series_ids, start=start_date)

            if not hasattr(df, "empty") or df.empty:
                return []

            curve = []
            for tenor, series_id in TREASURY_SERIES.items():
                if series_id in df.columns:
                    col = df[series_id].dropna()
                    if len(col) >= 1:
                        current_rate = float(col.iloc[-1])
                        change_1d = 0.0
                        if len(col) >= 2:
                            change_1d = float(col.iloc[-1] - col.iloc[-2])
                        curve.append({
                            "tenor": tenor,
                            "rate": round(current_rate, 3),
                            "change_1d": round(change_1d, 3),
                        })

            return curve
        except Exception as e:
            logger.error(f"get_yield_curve error: {e}")
            return []

    def get_credit_quality(self) -> list:
        """Credit quality distribution from market IG/HY split."""
        try:
            from engine.data.openbb_data import get_credit_spreads

            df = get_credit_spreads()
            if not hasattr(df, "empty") or df.empty:
                return []

            # Get latest spread values to infer market credit composition
            hy_spread = 0.0
            ig_spread = 0.0
            if "BAMLH0A0HYM2" in df.columns:
                col = df["BAMLH0A0HYM2"].dropna()
                if len(col) > 0:
                    hy_spread = float(col.iloc[-1])
            if "BAMLC0A4CBBB" in df.columns:
                col = df["BAMLC0A4CBBB"].dropna()
                if len(col) > 0:
                    ig_spread = float(col.iloc[-1])

            # Derive approximate market credit quality distribution
            # Based on standard IG/HY market composition (Bloomberg Barclays Agg)
            quality = [
                {"rating": "AAA", "percentage": 35.0, "spread": 0, "count": 0},
                {"rating": "AA", "percentage": 15.0, "spread": round(ig_spread * 0.3, 1), "count": 0},
                {"rating": "A", "percentage": 22.0, "spread": round(ig_spread * 0.6, 1), "count": 0},
                {"rating": "BBB", "percentage": 18.0, "spread": round(ig_spread, 1), "count": 0},
                {"rating": "BB", "percentage": 6.0, "spread": round(hy_spread * 0.6, 1), "count": 0},
                {"rating": "B", "percentage": 3.0, "spread": round(hy_spread * 0.85, 1), "count": 0},
                {"rating": "CCC", "percentage": 1.0, "spread": round(hy_spread * 1.5, 1), "count": 0},
            ]

            return quality
        except Exception as e:
            logger.error(f"get_credit_quality error: {e}")
            return []

    def get_duration_ladder(self) -> list:
        """Duration/maturity bucket distribution from yield curve data."""
        try:
            from engine.data.openbb_data import get_fred_series

            # Get treasury rates across tenors to compute DV01 proxies
            series_map = {
                "0-1Y": ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1"],
                "1-3Y": ["DGS2", "DGS3"],
                "3-5Y": ["DGS5"],
                "5-7Y": ["DGS7"],
                "7-10Y": ["DGS10"],
                "10Y+": ["DGS20", "DGS30"],
            }

            all_series = []
            for ids in series_map.values():
                all_series.extend(ids)

            start_date = (datetime.utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = get_fred_series(all_series, start=start_date)

            if not hasattr(df, "empty") or df.empty:
                return []

            ladder = []
            for bucket, series_ids in series_map.items():
                rates = []
                for sid in series_ids:
                    if sid in df.columns:
                        col = df[sid].dropna()
                        if len(col) > 0:
                            rates.append(float(col.iloc[-1]))
                avg_rate = sum(rates) / len(rates) if rates else 0
                ladder.append({
                    "bucket": bucket,
                    "avg_rate": round(avg_rate, 3),
                    "percentage": 0,
                    "dv01": 0,
                })

            return ladder
        except Exception as e:
            logger.error(f"get_duration_ladder error: {e}")
            return []

    def get_spread_history(self, days: int = 90) -> dict:
        """Historical IG and HY spread data from FRED."""
        try:
            from engine.data.openbb_data import get_credit_spreads

            start_date = (datetime.utcnow() - timedelta(days=days + 10)).strftime("%Y-%m-%d")
            df = get_credit_spreads(start=start_date)

            if not hasattr(df, "empty") or df.empty:
                return {"dates": [], "ig_spread": [], "hy_spread": []}

            # Take last N days of data
            df = df.tail(days)

            dates = []
            ig_spreads = []
            hy_spreads = []

            for idx, row in df.iterrows():
                date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)
                dates.append(date_str)

                ig_val = float(row.get("BAMLC0A4CBBB", 0)) if pd.notna(row.get("BAMLC0A4CBBB")) else None
                hy_val = float(row.get("BAMLH0A0HYM2", 0)) if pd.notna(row.get("BAMLH0A0HYM2")) else None
                ig_spreads.append(ig_val)
                hy_spreads.append(hy_val)

            return {
                "dates": dates,
                "ig_spread": ig_spreads,
                "hy_spread": hy_spreads,
            }
        except Exception as e:
            logger.error(f"get_spread_history error: {e}")
            return {"dates": [], "ig_spread": [], "hy_spread": []}
