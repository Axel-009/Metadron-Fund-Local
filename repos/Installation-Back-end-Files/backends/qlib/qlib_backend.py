"""QLIB Backend — Microsoft's Quantitative Investment Library.

Full factor mining, alpha pipeline, and model training infrastructure.

QLIB provides:
    - Data Handler: efficient storage and retrieval of financial data
    - Factor Mining: automated feature engineering from OHLCV + fundamentals
    - Alpha Pipeline: factor → model → signal → portfolio → execution
    - Model Zoo: LightGBM, XGBoost, LSTM, Transformer, etc.
    - Backtest Framework: realistic simulation with transaction costs

This backend wraps QLIB for use in the Metadron Capital AlphaOptimizer.

Dependencies:
    pip install qlib
    qlib data download (run: python -m qlib.contrib.data.handler)

Usage:
    from backends.qlib.qlib_backend import QLIBBackend
    backend = QLIBBackend()
    backend.initialize()
    factors = backend.mine_factors(["AAPL", "MSFT"])
    predictions = backend.predict(["AAPL", "MSFT"])
"""

import logging
import os
from pathlib import Path
from typing import Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

QLIB_DATA_DIR = Path(__file__).parent / "data"
QLIB_DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "qlib"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

try:
    import qlib
    from qlib.config import REG_CN, REG_US
    from qlib.data import D
    _HAS_QLIB = True
    logger.info("QLIB loaded successfully")
except (ImportError, ModuleNotFoundError, Exception) as e:
    _HAS_QLIB = False
    logger.warning(f"QLIB not available: {e}")


class QLIBBackend:
    """Full QLIB backend for factor mining and alpha generation."""

    # Alpha158 factor set — 158 technical/fundamental factors
    ALPHA158_FEATURES = [
        # Price-based
        "KMID", "KLEN", "KMID2", "KUP", "KUP2", "KLOW", "KLOW2",
        "KSFT", "KSFT2",
        # Volume-based
        "VWAP", "VMA5", "VMA10", "VMA20", "VMA30", "VMA60",
        # Momentum
        "ROC5", "ROC10", "ROC20", "ROC60", "MA5", "MA10", "MA20", "MA30", "MA60",
        # Volatility
        "STD5", "STD10", "STD20", "STD60",
        # Technical indicators
        "RSI", "MACD", "BOLL_UB", "BOLL_LB", "PSY", "BIAS5", "BIAS10", "BIAS20",
    ]

    def __init__(self, region: str = "us", provider_uri: Optional[str] = None):
        self.region = region
        self.provider_uri = provider_uri or str(QLIB_DATA_DIR / f"qlib_data_{region}")
        self._initialized = False
        self._handler = None
        self._model = None

    def initialize(self) -> bool:
        """Initialize QLIB runtime and data."""
        if not _HAS_QLIB:
            logger.warning("QLIB not installed — using numpy factor fallback")
            self._initialized = False
            return False

        try:
            qlib.init(
                provider_uri=self.provider_uri,
                region_name=self.region,
            )
            self._initialized = True
            logger.info(f"QLIB initialized: region={self.region} data={self.provider_uri}")
            return True
        except Exception as e:
            logger.warning(f"QLIB init failed (data may need download): {e}")
            self._initialized = False
            return False

    def download_data(self, start: str = "2020-01-01") -> bool:
        """Download QLIB data for the US market."""
        if not _HAS_QLIB:
            return False
        try:
            from qlib.contrib.data.handler import check_qlib_data
            # This triggers data download if not present
            check_qlib_data(self.provider_uri)
            logger.info("QLIB data downloaded successfully")
            return True
        except Exception as e:
            logger.warning(f"QLIB data download failed: {e}")
            return False

    def mine_factors(self, tickers: list[str],
                      start: str = "2023-01-01",
                      end: Optional[str] = None) -> pd.DataFrame:
        """Mine Alpha158 factors for given tickers.

        Returns DataFrame with (datetime, ticker) index and factor columns.
        """
        if self._initialized and _HAS_QLIB:
            return self._mine_factors_qlib(tickers, start, end)
        else:
            return self._mine_factors_fallback(tickers, start, end)

    def _mine_factors_qlib(self, tickers: list[str],
                            start: str, end: Optional[str]) -> pd.DataFrame:
        """Mine factors using QLIB engine."""
        try:
            from qlib.contrib.data.handler import Alpha158
            handler = Alpha158(
                instruments=tickers,
                start_time=start,
                end_time=end or datetime.now().strftime("%Y-%m-%d"),
            )
            df = handler.fetch()
            self._handler = handler
            return df
        except Exception as e:
            logger.warning(f"QLIB factor mining failed: {e}")
            return self._mine_factors_fallback(tickers, start, end)

    def _mine_factors_fallback(self, tickers: list[str],
                                start: str, end: Optional[str]) -> pd.DataFrame:
        """Numpy-based factor mining fallback when QLIB unavailable.

        Computes a core subset of Alpha158 factors from yfinance data.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("Neither QLIB nor yfinance available for factor mining")
            return pd.DataFrame()

        all_factors = []

        for ticker in tickers:
            try:
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
                if data.empty or len(data) < 30:
                    continue

                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)

                o = data["Open"].values
                h = data["High"].values
                l = data["Low"].values
                c = data["Close"].values
                v = data["Volume"].values.astype(float)

                factors = pd.DataFrame(index=data.index)
                factors["ticker"] = ticker

                # Candlestick features
                body = c - o
                hl_range = h - l + 1e-10
                factors["KMID"] = body / hl_range
                factors["KLEN"] = hl_range / (c + 1e-10)
                factors["KUP"] = (h - np.maximum(o, c)) / hl_range
                factors["KLOW"] = (np.minimum(o, c) - l) / hl_range

                # Moving averages
                for w in [5, 10, 20, 30, 60]:
                    ma = pd.Series(c).rolling(w).mean().values
                    factors[f"MA{w}"] = (c - ma) / (ma + 1e-10)

                # Volume MAs
                for w in [5, 10, 20, 30, 60]:
                    vma = pd.Series(v).rolling(w).mean().values
                    factors[f"VMA{w}"] = (v - vma) / (vma + 1e-10)

                # ROC (rate of change)
                for w in [5, 10, 20, 60]:
                    factors[f"ROC{w}"] = pd.Series(c).pct_change(w).values

                # Standard deviations (realized vol)
                for w in [5, 10, 20, 60]:
                    factors[f"STD{w}"] = pd.Series(c).pct_change().rolling(w).std().values

                # RSI
                delta = pd.Series(c).diff()
                gain = delta.where(delta > 0, 0.0).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
                rs = gain / (loss + 1e-10)
                factors["RSI"] = (100.0 - (100.0 / (1.0 + rs))).values / 100.0

                # MACD
                ema12 = pd.Series(c).ewm(span=12).mean()
                ema26 = pd.Series(c).ewm(span=26).mean()
                factors["MACD"] = ((ema12 - ema26) / (c + 1e-10)).values

                # Bollinger Bands
                ma20 = pd.Series(c).rolling(20).mean()
                std20 = pd.Series(c).rolling(20).std()
                factors["BOLL_UB"] = ((ma20 + 2 * std20 - c) / (c + 1e-10)).values
                factors["BOLL_LB"] = ((ma20 - 2 * std20 - c) / (c + 1e-10)).values

                # VWAP
                cum_vol = pd.Series(v).cumsum()
                cum_vp = (pd.Series(c) * pd.Series(v)).cumsum()
                vwap = cum_vp / (cum_vol + 1e-10)
                factors["VWAP"] = ((vwap - c) / (c + 1e-10)).values

                # Bias
                for w in [5, 10, 20]:
                    ma = pd.Series(c).rolling(w).mean()
                    factors[f"BIAS{w}"] = ((c - ma) / (ma + 1e-10)).values

                all_factors.append(factors)

            except Exception as e:
                logger.warning(f"Factor mining failed for {ticker}: {e}")
                continue

        if not all_factors:
            return pd.DataFrame()

        result = pd.concat(all_factors, ignore_index=False)
        result.dropna(inplace=True)
        logger.info(f"Mined {len(result.columns)-1} factors for {len(tickers)} tickers "
                    f"({len(result)} rows) [numpy fallback]")
        return result

    def train_model(self, factors_df: pd.DataFrame,
                     target_col: str = "ROC5",
                     model_type: str = "lightgbm") -> dict:
        """Train a prediction model on mined factors.

        Args:
            factors_df: Output from mine_factors().
            target_col: Column to predict (future returns).
            model_type: "lightgbm", "xgboost", or "linear".

        Returns:
            Training metrics dict.
        """
        if factors_df.empty or target_col not in factors_df.columns:
            return {"error": "insufficient data"}

        feature_cols = [c for c in factors_df.columns
                       if c not in ["ticker", target_col] and factors_df[c].dtype in [np.float64, np.float32]]

        # Shift target forward (predict future returns)
        df = factors_df.copy()
        df["target"] = df.groupby("ticker")[target_col].shift(-5)
        df.dropna(inplace=True)

        if len(df) < 100:
            return {"error": "not enough data after processing"}

        X = df[feature_cols].values
        y = df["target"].values

        # Train/test split (time-based)
        split = int(len(df) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        if model_type == "lightgbm":
            try:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, verbose=-1,
                )
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                self._model = model
                return {"model": "lightgbm", "train_r2": train_score, "test_r2": test_score,
                        "n_features": len(feature_cols), "n_samples": len(df)}
            except ImportError:
                logger.info("LightGBM not available, falling back to linear")
                model_type = "linear"

        if model_type == "xgboost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, verbosity=0,
                )
                model.fit(X_train, y_train)
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                self._model = model
                return {"model": "xgboost", "train_r2": train_score, "test_r2": test_score,
                        "n_features": len(feature_cols), "n_samples": len(df)}
            except ImportError:
                logger.info("XGBoost not available, falling back to linear")
                model_type = "linear"

        # Linear fallback
        from numpy.linalg import lstsq
        X_aug = np.column_stack([X_train, np.ones(len(X_train))])
        coeffs, _, _, _ = lstsq(X_aug, y_train, rcond=None)
        y_pred_train = X_aug @ coeffs
        X_test_aug = np.column_stack([X_test, np.ones(len(X_test))])
        y_pred_test = X_test_aug @ coeffs

        ss_res_train = np.sum((y_train - y_pred_train) ** 2)
        ss_tot_train = np.sum((y_train - y_train.mean()) ** 2)
        train_r2 = 1 - ss_res_train / ss_tot_train if ss_tot_train > 0 else 0

        ss_res_test = np.sum((y_test - y_pred_test) ** 2)
        ss_tot_test = np.sum((y_test - y_test.mean()) ** 2)
        test_r2 = 1 - ss_res_test / ss_tot_test if ss_tot_test > 0 else 0

        return {"model": "linear", "train_r2": train_r2, "test_r2": test_r2,
                "n_features": len(feature_cols), "n_samples": len(df)}

    def predict(self, tickers: list[str],
                 start: str = "2024-01-01") -> dict[str, float]:
        """Generate alpha predictions for tickers.

        Returns dict of ticker -> predicted forward return.
        """
        factors = self.mine_factors(tickers, start=start)
        if factors.empty:
            return {}

        if self._model is None:
            # Train on available data first
            metrics = self.train_model(factors)
            logger.info(f"Auto-trained model: {metrics}")

        predictions = {}
        for ticker in tickers:
            ticker_data = factors[factors["ticker"] == ticker]
            if ticker_data.empty:
                continue

            feature_cols = [c for c in ticker_data.columns
                          if c not in ["ticker"] and ticker_data[c].dtype in [np.float64, np.float32]]

            latest = ticker_data[feature_cols].iloc[-1:].values

            if self._model is not None:
                try:
                    pred = float(self._model.predict(latest)[0])
                    predictions[ticker] = pred
                except Exception:
                    # Fallback: use momentum as prediction
                    predictions[ticker] = float(ticker_data.get("ROC5", pd.Series([0])).iloc[-1])
            else:
                predictions[ticker] = float(ticker_data.get("ROC5", pd.Series([0])).iloc[-1])

        return predictions
