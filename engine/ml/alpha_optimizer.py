"""AlphaOptimizer — Walk-forward ML alpha + mean-variance portfolio construction.

Incorporates Dataset 2: ML-based portfolio optimizer with walk-forward alpha
and turnover constraints.

Pipeline:
    1. Feature engineering (market, momentum, vol)
    2. Walk-forward linear regression for alpha prediction
    3. EWMA covariance estimation
    4. Mean-variance optimisation with turnover constraints
    5. Quality tier classification (A–G)

CAPM alpha ranking + factor integration.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from ..data.yahoo_data import get_adj_close, get_returns


# ---------------------------------------------------------------------------
# Quality tiers
# ---------------------------------------------------------------------------
QUALITY_TIERS = {
    "A": {"min_sharpe": 2.0, "min_momentum": 0.15},
    "B": {"min_sharpe": 1.5, "min_momentum": 0.10},
    "C": {"min_sharpe": 1.0, "min_momentum": 0.05},
    "D": {"min_sharpe": 0.5, "min_momentum": 0.00},
    "E": {"min_sharpe": 0.0, "min_momentum": -0.05},
    "F": {"min_sharpe": -0.5, "min_momentum": -0.10},
    "G": {"min_sharpe": -999, "min_momentum": -999},
}

# Alpha target — seeking alpha >= 2%
ALPHA_TARGET = 0.02
TRANSACTION_COST = 0.001
MAX_TURNOVER = 0.50


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class AlphaSignal:
    ticker: str
    alpha_pred: float = 0.0
    quality_tier: str = "D"
    sharpe_estimate: float = 0.0
    momentum_3m: float = 0.0
    momentum_1m: float = 0.0
    vol: float = 0.0
    weight: float = 0.0


@dataclass
class AlphaOutput:
    signals: list = field(default_factory=list)
    optimal_weights: dict = field(default_factory=dict)
    expected_annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    rebalance_cost: float = 0.0
    alpha_predictions: Optional[pd.Series] = None


# ---------------------------------------------------------------------------
# Feature engineering (Dataset 2)
# ---------------------------------------------------------------------------
def build_features(returns: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Build ML features from returns and prices."""
    features = pd.DataFrame(index=returns.index)
    features["mkt_mean"] = returns.mean(axis=1)
    features["mkt_vol"] = returns.std(axis=1)
    features["momentum_3m"] = prices.reindex(returns.index).ffill().pct_change(60).mean(axis=1)
    features["momentum_1m"] = prices.reindex(returns.index).ffill().pct_change(20).mean(axis=1)
    features["momentum_1w"] = prices.reindex(returns.index).ffill().pct_change(5).mean(axis=1)
    features["vol_20d"] = returns.rolling(20).std().mean(axis=1)
    features["vol_ratio"] = (
        returns.rolling(5).std().mean(axis=1)
        / returns.rolling(20).std().mean(axis=1).replace(0, np.nan)
    )
    features["return_dispersion"] = returns.std(axis=1) / returns.mean(axis=1).abs().replace(0, np.nan)
    features = features.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return features


# ---------------------------------------------------------------------------
# EWMA covariance (Dataset 2)
# ---------------------------------------------------------------------------
def ewma_cov(returns_df: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    """Exponentially weighted moving average covariance matrix."""
    cov = returns_df.cov().values.copy()
    for i in range(len(returns_df)):
        r = returns_df.iloc[i].values.reshape(-1, 1)
        cov = lam * cov + (1 - lam) * (r @ r.T)
    cov = cov + 1e-10 * np.eye(cov.shape[0])
    return cov


# ---------------------------------------------------------------------------
# Quality classification
# ---------------------------------------------------------------------------
def classify_quality(sharpe: float, momentum: float) -> str:
    """Assign quality tier A–G."""
    for tier, thresholds in QUALITY_TIERS.items():
        if sharpe >= thresholds["min_sharpe"] and momentum >= thresholds["min_momentum"]:
            return tier
    return "G"


# ---------------------------------------------------------------------------
# AlphaOptimizer
# ---------------------------------------------------------------------------
class AlphaOptimizer:
    """Walk-forward ML alpha with mean-variance portfolio construction.

    Hunting for 100% alpha with no decay in selection.
    """

    def __init__(
        self,
        train_ratio: float = 0.80,
        max_turnover: float = MAX_TURNOVER,
        transaction_cost: float = TRANSACTION_COST,
        alpha_headstart: float = ALPHA_TARGET,
    ):
        self.train_ratio = train_ratio
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.alpha_headstart = alpha_headstart
        self._last_output: Optional[AlphaOutput] = None

    def optimize(
        self,
        tickers: list[str],
        start: str = "2015-01-01",
        current_weights: Optional[np.ndarray] = None,
        sector_constraints: Optional[dict] = None,
    ) -> AlphaOutput:
        """Run full alpha optimization pipeline."""
        output = AlphaOutput()

        # 1. Get data
        prices = get_adj_close(tickers, start=start)
        if prices.empty or len(prices.columns) < 2:
            return output
        available = [c for c in tickers if c in prices.columns]
        prices = prices[available]
        tickers = available
        returns = np.log(prices / prices.shift(1)).dropna()
        n_assets = len(tickers)

        if current_weights is None:
            current_weights = np.ones(n_assets) / n_assets

        # 2. Feature engineering
        features = build_features(returns, prices)

        # 3. Walk-forward ML
        target = returns.mean(axis=1).shift(-1)
        dataset = pd.concat([features, target.rename("target")], axis=1).dropna()
        X = dataset.drop(columns=["target"])
        y = dataset["target"]

        split = int(len(X) * self.train_ratio)
        if split < 50:
            return output

        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        alpha_preds = pd.Series(model.predict(X_test_scaled), index=X_test.index)
        output.alpha_predictions = alpha_preds

        # 4. EWMA covariance
        test_returns = returns.loc[X_test.index[0]:][tickers]
        if len(test_returns) < 20:
            return output
        latest_cov = ewma_cov(test_returns)

        # 5. Expected returns with alpha headstart
        expected_returns = test_returns[tickers].mean().values + alpha_preds.iloc[-1] + self.alpha_headstart
        expected_returns = np.asarray(expected_returns, dtype=float)

        # 6. Optimize
        optimal_weights = self._optimize_weights(
            expected_returns, latest_cov, current_weights, n_assets,
        )

        # 7. Performance
        port_returns = test_returns[tickers] @ optimal_weights
        annual_ret = float(port_returns.mean() * 252)
        annual_vol = float(port_returns.std() * np.sqrt(252))
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0.0
        cum = (1 + port_returns).cumprod()
        dd = cum / cum.cummax() - 1
        max_dd = float(dd.min())
        rebal_cost = float(np.sum(np.abs(optimal_weights - current_weights)) * self.transaction_cost)

        # 8. Build signals
        signals = []
        for i, t in enumerate(tickers):
            t_ret = test_returns[t]
            mom_3m = float(t_ret.iloc[-63:].sum()) if len(t_ret) >= 63 else 0.0
            mom_1m = float(t_ret.iloc[-21:].sum()) if len(t_ret) >= 21 else 0.0
            t_vol = float(t_ret.std() * np.sqrt(252))
            t_sharpe = (float(t_ret.mean() * 252) / t_vol) if t_vol > 0 else 0.0

            sig = AlphaSignal(
                ticker=t,
                alpha_pred=float(expected_returns[i]),
                quality_tier=classify_quality(t_sharpe, mom_3m),
                sharpe_estimate=t_sharpe,
                momentum_3m=mom_3m,
                momentum_1m=mom_1m,
                vol=t_vol,
                weight=float(optimal_weights[i]),
            )
            signals.append(sig)

        output.signals = sorted(signals, key=lambda s: s.alpha_pred, reverse=True)
        output.optimal_weights = {t: float(optimal_weights[i]) for i, t in enumerate(tickers)}
        output.expected_annual_return = annual_ret
        output.annual_volatility = annual_vol
        output.sharpe_ratio = sharpe
        output.max_drawdown = max_dd
        output.rebalance_cost = rebal_cost

        self._last_output = output
        return output

    def _optimize_weights(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        current_weights: np.ndarray,
        n_assets: int,
    ) -> np.ndarray:
        """Mean-variance optimization with turnover constraints."""

        def portfolio_vol(w):
            vol_sq = max(1e-12, w.T @ cov_matrix @ w)
            return np.sqrt(vol_sq)

        def turnover(w, prev):
            return float(np.sum(np.abs(w - prev)))

        def objective(w):
            w = np.asarray(w)
            port_ret = float(w @ expected_returns)
            vol = portfolio_vol(w)
            cost = turnover(w, current_weights) * self.transaction_cost
            sharpe = (port_ret - cost) / vol if vol > 0 else 0.0
            return -sharpe

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w: self.max_turnover - turnover(np.asarray(w), current_weights)},
        ]
        bounds = [(0.0, 0.40)] * n_assets  # Max 40% per name

        try:
            result = minimize(
                objective, current_weights.copy(),
                method="SLSQP", bounds=bounds, constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-9},
            )
            if result.success:
                return np.asarray(result.x)
        except Exception:
            pass

        return current_weights

    def get_last(self) -> Optional[AlphaOutput]:
        return self._last_output
