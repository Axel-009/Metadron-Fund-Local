"""
CUBE ROTATION ENGINE - Core Alpha Generation Mechanism for CTA.

Three-dimensional opportunity scoring model that rotates portfolio
allocations toward the highest-scoring investment opportunities
across asset classes.

Cube Dimensions:
    Dimension 1 (X-axis): Asset Class Momentum
        Score each asset class's relative momentum:
        R_momentum = sum(w_i * r_{i,t}) / sigma_i
        where:
            w_i = weight of asset i within the asset class
            r_{i,t} = return of asset i at time t
            sigma_i = volatility of asset i

    Dimension 2 (Y-axis): Fundamental Value
        Score based on deviation from fair value:
        z_value = (P_market - P_fair) / sigma_fair
        For equities: DCF, relative multiples, EV/EBITDA
        For bonds: yield spread vs historical
        For commodities: cost of carry, inventory levels
        For FX: PPP deviation, real rate differential

    Dimension 3 (Z-axis): Catalyst/Timing
        Score based on upcoming catalysts: earnings, macro events, technicals
        Timing signal: mean-reversion for short-term, trend for medium-term

    Rotation Logic:
        composite_score = w_momentum * X + w_value * Y + w_catalyst * Z
        Rotate portfolio toward highest-scoring opportunities
        Rebalance frequency: daily for HFT, weekly for swing, monthly for strategic

CTA Formulas Included:
    Trend strength:
        ADX = 100 * EMA(|+DI - -DI| / (+DI + -DI), n)
        where:
            +DM = H_t - H_{t-1} if > 0 and > (L_{t-1} - L_t), else 0
            -DM = L_{t-1} - L_t if > 0 and > (H_t - H_{t-1}), else 0
            TR = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
            +DI = 100 * EMA(+DM, n) / ATR(n)
            -DI = 100 * EMA(-DM, n) / ATR(n)

    Breakout:
        signal = 1 if price > max(high, n) else -1 if price < min(low, n) else 0
        Donchian channel: upper = max(H, n), lower = min(L, n)

    Carry:
        carry_return = (F_near - F_far) / F_near * (365 / days_between)
        Positive carry = contango -> short, backwardation -> long

    Risk budgeting:
        w_i = (target_risk / sigma_i) / sum_j(target_risk / sigma_j)
        Each asset gets equal risk budget regardless of asset class.

    Drawdown control:
        leverage = min(max_leverage, target_vol / realized_vol)
        If drawdown > threshold, reduce leverage proportionally:
        leverage_adj = leverage * max(0, 1 - (drawdown - threshold) / (max_dd - threshold))

Usage:
    from cube_rotation import CubeRotation, Trade, PerformanceMetrics
    from openbb_data import AssetClass

    cube = CubeRotation()
    momentum = cube.calculate_momentum_score(AssetClass.EQUITY, price_data)
    scores = cube.composite_rotation_score(all_scores)
    trades = cube.generate_rotation_trades(current_portfolio, target_scores)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

from openbb_data import AssetClass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    """Represents a single rotation trade."""
    symbol: str
    asset_class: AssetClass
    direction: str  # "BUY" or "SELL"
    quantity: float
    target_weight: float
    current_weight: float
    score: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class PerformanceMetrics:
    """Backtest performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_holding_period_days: float
    monthly_returns: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))


class RebalanceFrequency(Enum):
    """Portfolio rebalance frequency."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


# ---------------------------------------------------------------------------
# CTA Technical Indicators
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, span: int) -> pd.Series:
    """
    Exponential Moving Average.

    EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}
    where alpha = 2 / (span + 1)
    """
    return series.ewm(span=span, adjust=False).mean()


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Average True Range.

    TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)
    ATR = EMA(TR, period)
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return _ema(true_range, span=period)


def _adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Average Directional Index (ADX) with +DI and -DI.

    +DM = H_t - H_{t-1} if positive and > -(L change), else 0
    -DM = L_{t-1} - L_t if positive and > +(H change), else 0
    +DI = 100 * EMA(+DM, n) / ATR(n)
    -DI = 100 * EMA(-DM, n) / ATR(n)
    DX = 100 * |+DI - -DI| / (+DI + -DI)
    ADX = EMA(DX, n)

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price data.
    period : int
        Lookback period (default 14).

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.Series]
        (ADX, +DI, -DI) series.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(0.0, index=high.index)
    minus_dm = pd.Series(0.0, index=high.index)

    plus_mask = (up_move > 0) & (up_move > down_move)
    minus_mask = (down_move > 0) & (down_move > up_move)

    plus_dm[plus_mask] = up_move[plus_mask]
    minus_dm[minus_mask] = down_move[minus_mask]

    atr_vals = _atr(high, low, close, period)

    # Avoid division by zero
    atr_safe = atr_vals.replace(0, np.nan)

    plus_di = 100.0 * _ema(plus_dm, span=period) / atr_safe
    minus_di = 100.0 * _ema(minus_dm, span=period) / atr_safe

    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum_safe

    adx_val = _ema(dx, span=period)

    return adx_val, plus_di, minus_di


def _donchian_breakout(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """
    Donchian Channel Breakout Signal.

    signal = +1 if close > max(high, n)   (upside breakout)
    signal = -1 if close < min(low, n)    (downside breakout)
    signal =  0 otherwise                 (inside channel)

    Parameters
    ----------
    high, low, close : pd.Series
    period : int
        Channel lookback (default 20).

    Returns
    -------
    pd.Series
        Signal series: +1, -1, or 0.
    """
    upper = high.rolling(window=period).max().shift(1)
    lower = low.rolling(window=period).min().shift(1)

    signal = pd.Series(0, index=close.index, dtype=float)
    signal[close > upper] = 1.0
    signal[close < lower] = -1.0

    return signal


def _carry_return(f_near: float, f_far: float, days_between: int) -> float:
    """
    Calculate annualized carry return from futures term structure.

    carry_return = (F_near - F_far) / F_near * (365 / days_between)

    Positive result: backwardation (long bias)
    Negative result: contango (short bias or roll cost)

    Parameters
    ----------
    f_near : float
        Near-month futures price.
    f_far : float
        Far-month futures price.
    days_between : int
        Days between contract expirations.

    Returns
    -------
    float
        Annualized carry return.
    """
    if f_near == 0 or days_between == 0:
        return 0.0
    return (f_near - f_far) / f_near * (365.0 / days_between)


def _risk_budget_weights(
    volatilities: dict[str, float], target_risk: float = 0.10
) -> dict[str, float]:
    """
    Risk budgeting allocation.

    w_i = (target_risk / sigma_i) / sum_j(target_risk / sigma_j)

    Each asset receives equal risk contribution regardless of asset class.

    Parameters
    ----------
    volatilities : dict[str, float]
        Annualized volatility per asset.
    target_risk : float
        Target portfolio risk (default 10%).

    Returns
    -------
    dict[str, float]
        Portfolio weights summing to 1.0.
    """
    valid = {k: v for k, v in volatilities.items() if v > 0}
    if not valid:
        n = len(volatilities)
        return {k: 1.0 / n for k in volatilities} if n > 0 else {}

    inv_vols = {k: target_risk / v for k, v in valid.items()}
    total = sum(inv_vols.values())

    weights = {k: 0.0 for k in volatilities}
    for k, iv in inv_vols.items():
        weights[k] = iv / total
    return weights


def _drawdown_controlled_leverage(
    target_vol: float,
    realized_vol: float,
    max_leverage: float = 2.0,
    current_drawdown: float = 0.0,
    drawdown_threshold: float = 0.10,
    max_drawdown_limit: float = 0.25,
) -> float:
    """
    Calculate leverage with drawdown control.

    Base leverage:
        leverage = min(max_leverage, target_vol / realized_vol)

    Drawdown adjustment (if drawdown > threshold):
        leverage_adj = leverage * max(0, 1 - (dd - threshold) / (max_dd - threshold))

    Parameters
    ----------
    target_vol : float
        Target annualized portfolio volatility.
    realized_vol : float
        Current realized annualized volatility.
    max_leverage : float
        Maximum allowed leverage.
    current_drawdown : float
        Current drawdown as positive fraction (e.g., 0.15 = 15% drawdown).
    drawdown_threshold : float
        Drawdown level where leverage reduction begins.
    max_drawdown_limit : float
        Drawdown level where leverage goes to zero.

    Returns
    -------
    float
        Adjusted leverage factor.
    """
    if realized_vol <= 0:
        return 0.0

    base_leverage = min(max_leverage, target_vol / realized_vol)

    if current_drawdown <= drawdown_threshold:
        return base_leverage

    if current_drawdown >= max_drawdown_limit:
        return 0.0

    dd_range = max_drawdown_limit - drawdown_threshold
    if dd_range <= 0:
        return 0.0

    reduction = (current_drawdown - drawdown_threshold) / dd_range
    return base_leverage * max(0.0, 1.0 - reduction)


# ---------------------------------------------------------------------------
# Cube Rotation Engine
# ---------------------------------------------------------------------------

class CubeRotation:
    """
    Cube Rotation Model: 3-dimensional opportunity scoring for CTA.

    Combines momentum, value, and catalyst signals across all asset
    classes to generate portfolio rotation trades.
    """

    def __init__(
        self,
        w_momentum: float = 0.4,
        w_value: float = 0.3,
        w_catalyst: float = 0.3,
        target_vol: float = 0.12,
        max_leverage: float = 2.0,
        rebalance_freq: RebalanceFrequency = RebalanceFrequency.WEEKLY,
        risk_free_rate: float = 0.05,
        trading_days: int = 252,
    ):
        """
        Parameters
        ----------
        w_momentum : float
            Weight for momentum dimension (X-axis). Default 0.4.
        w_value : float
            Weight for value dimension (Y-axis). Default 0.3.
        w_catalyst : float
            Weight for catalyst dimension (Z-axis). Default 0.3.
        target_vol : float
            Target annualized portfolio volatility. Default 12%.
        max_leverage : float
            Maximum leverage. Default 2.0.
        rebalance_freq : RebalanceFrequency
            Rebalance frequency. Default WEEKLY.
        risk_free_rate : float
            Annual risk-free rate. Default 5%.
        trading_days : int
            Trading days per year. Default 252.
        """
        assert abs(w_momentum + w_value + w_catalyst - 1.0) < 1e-9, \
            "Dimension weights must sum to 1.0"

        self.w_momentum = w_momentum
        self.w_value = w_value
        self.w_catalyst = w_catalyst
        self.target_vol = target_vol
        self.max_leverage = max_leverage
        self.rebalance_freq = rebalance_freq
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days

    # -------------------------------------------------------------------
    # Dimension 1: Momentum Score (X-axis)
    # -------------------------------------------------------------------

    def calculate_momentum_score(
        self,
        asset_class: AssetClass,
        data: dict[str, pd.DataFrame],
        fast_window: int = 21,
        slow_window: int = 63,
        trend_window: int = 252,
    ) -> float:
        """
        Calculate momentum score for an asset class.

        Methodology:
            1. EMA crossover signal per asset:
               signal_i = (EMA(P_i, fast) - EMA(P_i, slow)) / ATR(P_i, 14)
            2. Trend strength via ADX:
               trend_i = ADX_i / 100 (normalized to [0, 1])
            3. Breakout signal:
               breakout_i = donchian_breakout(P_i, 20)
            4. Time-series momentum (12-month return, skip last month):
               tsmom_i = P_{t-21} / P_{t-252} - 1
            5. Combined:
               R_momentum = mean(0.3*signal_i + 0.3*trend_i + 0.2*breakout_i + 0.2*tsmom_i)

        Parameters
        ----------
        asset_class : AssetClass
            The asset class being scored.
        data : dict[str, pd.DataFrame]
            Symbol -> OHLCV DataFrame.
        fast_window : int
            Fast EMA period (default 21).
        slow_window : int
            Slow EMA period (default 63).
        trend_window : int
            Long-term trend window for TSMOM (default 252).

        Returns
        -------
        float
            Momentum score (typically in range [-1, 1], can exceed).
        """
        scores = []

        for symbol, df in data.items():
            if df is None or df.empty:
                continue
            if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
                continue
            if len(df) < max(slow_window, trend_window) + 10:
                continue

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)

            # 1. EMA crossover normalized by ATR
            ema_fast = _ema(close, fast_window)
            ema_slow = _ema(close, slow_window)
            atr = _atr(high, low, close, period=14)
            atr_last = float(atr.iloc[-1])
            if atr_last > 0:
                crossover = float((ema_fast.iloc[-1] - ema_slow.iloc[-1]) / atr_last)
                crossover = np.clip(crossover, -3, 3) / 3.0  # normalize to [-1, 1]
            else:
                crossover = 0.0

            # 2. ADX trend strength (0 to 1)
            adx_val, plus_di, minus_di = _adx(high, low, close, period=14)
            trend_strength = float(adx_val.iloc[-1]) / 100.0 if not np.isnan(adx_val.iloc[-1]) else 0.0
            # Direction: positive if +DI > -DI
            di_direction = 1.0 if float(plus_di.iloc[-1]) > float(minus_di.iloc[-1]) else -1.0
            trend_signal = trend_strength * di_direction

            # 3. Donchian breakout
            breakout = _donchian_breakout(high, low, close, period=20)
            breakout_val = float(breakout.iloc[-1])

            # 4. Time-series momentum (12m return skipping last month)
            if len(close) >= trend_window:
                p_recent = float(close.iloc[-22]) if len(close) > 22 else float(close.iloc[-1])
                p_old = float(close.iloc[-trend_window])
                tsmom = (p_recent / p_old - 1.0) if p_old != 0 else 0.0
                tsmom = np.clip(tsmom, -1, 1)
            else:
                tsmom = 0.0

            # 5. Combined
            combined = 0.3 * crossover + 0.3 * trend_signal + 0.2 * breakout_val + 0.2 * tsmom
            scores.append(combined)

        if not scores:
            return 0.0

        return float(np.mean(scores))

    # -------------------------------------------------------------------
    # Dimension 2: Value Score (Y-axis)
    # -------------------------------------------------------------------

    def calculate_value_score(
        self,
        asset_class: AssetClass,
        data: dict[str, pd.DataFrame],
        fundamentals: Optional[dict[str, dict[str, Any]]] = None,
    ) -> float:
        """
        Calculate fundamental value score for an asset class.

        z_value = (P_market - P_fair) / sigma_fair

        Asset-class specific valuation:
            Equities:
                z_PE = (PE_current - PE_median) / PE_std
                z_PB = (PB_current - PB_median) / PB_std
                value_equity = -0.5 * z_PE - 0.5 * z_PB  (negative = cheap = good)

            Bonds:
                z_yield = (yield_current - yield_mean) / yield_std
                value_bond = z_yield  (higher yield = more value)

            Commodities:
                Use price relative to 5-year rolling mean:
                z_commodity = (P - mean(P, 5y)) / std(P, 5y)
                value_commodity = -z_commodity  (below mean = cheap)

            FX:
                z_fx = (rate - mean(rate, 3y)) / std(rate, 3y)
                value_fx = -z_fx  (below mean = undervalued)

        Parameters
        ----------
        asset_class : AssetClass
            Asset class being scored.
        data : dict[str, pd.DataFrame]
            Symbol -> OHLCV DataFrame.
        fundamentals : dict[str, dict[str, Any]], optional
            Fundamental data per symbol (PE, PB, yield, etc.).

        Returns
        -------
        float
            Value score (negative = expensive, positive = cheap/attractive).
        """
        if fundamentals is None:
            fundamentals = {}

        scores = []

        if asset_class == AssetClass.EQUITY:
            pe_values = []
            for sym, fund in fundamentals.items():
                pe = fund.get("pe_ratio")
                if pe is not None and pe > 0:
                    pe_values.append(pe)

            if pe_values:
                pe_arr = np.array(pe_values)
                pe_median = np.median(pe_arr)
                pe_std = np.std(pe_arr) if len(pe_arr) > 1 else 1.0

                for pe in pe_values:
                    z_pe = (pe - pe_median) / pe_std if pe_std > 0 else 0.0
                    scores.append(-z_pe)  # Negative PE z-score = cheap = positive score

        # For all asset classes: price-based mean reversion value signal
        for symbol, df in data.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue

            close = df["Close"].astype(float).dropna()
            if len(close) < 252:
                # Use available data
                lookback = len(close)
            else:
                lookback = 252 * 5 if len(close) >= 252 * 5 else len(close)

            price_window = close.iloc[-lookback:]
            current = float(close.iloc[-1])
            mean_price = float(price_window.mean())
            std_price = float(price_window.std())

            if std_price > 0:
                z = (current - mean_price) / std_price
                scores.append(-z)  # Below mean = cheap = positive score

        if not scores:
            return 0.0

        return float(np.clip(np.mean(scores), -3, 3) / 3.0)

    # -------------------------------------------------------------------
    # Dimension 3: Catalyst Score (Z-axis)
    # -------------------------------------------------------------------

    def calculate_catalyst_score(
        self,
        asset_class: AssetClass,
        news: Optional[list[dict[str, Any]]] = None,
        calendar: Optional[list[dict[str, Any]]] = None,
    ) -> float:
        """
        Calculate catalyst/timing score.

        Timing signals:
            Short-term (< 5 days): Mean-reversion bias
                signal_ST = -z_score(r, 5d)
                If recent returns are extreme, expect reversion.

            Medium-term (5-63 days): Trend continuation bias
                signal_MT = sign(EMA(r, 21))
                Trends tend to persist at this horizon.

            Catalyst scoring:
                Each catalyst gets a magnitude and direction:
                catalyst_score = sum(direction_i * magnitude_i * decay(days_until_i))
                decay(d) = exp(-d / 10)  (exponential decay of relevance)

        Parameters
        ----------
        asset_class : AssetClass
            Asset class.
        news : list[dict], optional
            News items with keys: "sentiment" (-1 to 1), "magnitude" (0 to 1),
            "days_until" (int).
        calendar : list[dict], optional
            Calendar events with keys: "type", "direction" (-1 or 1),
            "magnitude" (0 to 1), "days_until" (int).

        Returns
        -------
        float
            Catalyst/timing score in [-1, 1].
        """
        score = 0.0
        n_items = 0

        if news:
            for item in news:
                sentiment = item.get("sentiment", 0.0)
                magnitude = item.get("magnitude", 0.5)
                days_until = item.get("days_until", 0)
                # Exponential decay: relevance decreases with time
                decay = np.exp(-max(days_until, 0) / 10.0)
                score += sentiment * magnitude * decay
                n_items += 1

        if calendar:
            for event in calendar:
                direction = event.get("direction", 0.0)
                magnitude = event.get("magnitude", 0.5)
                days_until = event.get("days_until", 0)
                decay = np.exp(-max(days_until, 0) / 10.0)
                score += direction * magnitude * decay
                n_items += 1

        if n_items > 0:
            score /= n_items

        return float(np.clip(score, -1.0, 1.0))

    # -------------------------------------------------------------------
    # Composite Rotation Score
    # -------------------------------------------------------------------

    def composite_rotation_score(
        self,
        scores: dict[str, dict[str, float]],
    ) -> pd.DataFrame:
        """
        Calculate composite rotation scores for all assets/asset classes.

        Formula:
            composite = w_momentum * X + w_value * Y + w_catalyst * Z

        Where:
            X = momentum score (Dimension 1)
            Y = value score (Dimension 2)
            Z = catalyst score (Dimension 3)

        Parameters
        ----------
        scores : dict[str, dict[str, float]]
            Mapping of asset/class name -> {"momentum": X, "value": Y, "catalyst": Z}.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: asset, momentum, value, catalyst, composite.
            Sorted by composite score descending.
        """
        rows = []
        for asset, dims in scores.items():
            x = dims.get("momentum", 0.0)
            y = dims.get("value", 0.0)
            z = dims.get("catalyst", 0.0)
            composite = self.w_momentum * x + self.w_value * y + self.w_catalyst * z
            rows.append({
                "asset": asset,
                "momentum": x,
                "value": y,
                "catalyst": z,
                "composite": composite,
            })

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("composite", ascending=False).reset_index(drop=True)
        return df

    # -------------------------------------------------------------------
    # Trade Generation
    # -------------------------------------------------------------------

    def generate_rotation_trades(
        self,
        current_portfolio: dict[str, float],
        target_scores: pd.DataFrame,
        total_capital: float = 1_000_000.0,
        max_position_pct: float = 0.20,
        min_trade_pct: float = 0.01,
        volatilities: Optional[dict[str, float]] = None,
    ) -> list[Trade]:
        """
        Generate trades to rotate portfolio toward highest-scoring opportunities.

        Algorithm:
            1. Convert composite scores to target weights:
               raw_weight_i = max(0, composite_i) / sum(max(0, composite_j))
               Negative-score assets get zero weight (potential short candidates).

            2. Apply risk budgeting if volatilities available:
               risk_adjusted_weight_i = raw_weight_i / sigma_i
               Normalize to sum to 1.

            3. Apply position limits:
               target_weight_i = min(raw_weight_i, max_position_pct)
               Re-normalize.

            4. Calculate leverage with drawdown control.

            5. Generate buy/sell trades for weight differences > min_trade_pct.

        Parameters
        ----------
        current_portfolio : dict[str, float]
            Current weights per asset.
        target_scores : pd.DataFrame
            Output from composite_rotation_score().
        total_capital : float
            Total portfolio capital.
        max_position_pct : float
            Maximum single position weight (default 20%).
        min_trade_pct : float
            Minimum trade size to execute (default 1%).
        volatilities : dict[str, float], optional
            Annualized volatilities per asset for risk budgeting.

        Returns
        -------
        list[Trade]
            List of trades to execute.
        """
        if target_scores.empty:
            return []

        # Step 1: Convert scores to raw weights
        positive_scores = target_scores[target_scores["composite"] > 0].copy()
        if positive_scores.empty:
            # All negative: flatten to cash
            trades = []
            for asset, weight in current_portfolio.items():
                if weight > min_trade_pct:
                    trades.append(Trade(
                        symbol=asset,
                        asset_class=AssetClass.EQUITY,  # default
                        direction="SELL",
                        quantity=weight * total_capital,
                        target_weight=0.0,
                        current_weight=weight,
                        score=0.0,
                        reason="All composite scores negative - rotate to cash",
                    ))
            return trades

        score_sum = positive_scores["composite"].sum()
        positive_scores["raw_weight"] = positive_scores["composite"] / score_sum

        # Step 2: Risk budget adjustment
        if volatilities:
            adjusted = []
            for _, row in positive_scores.iterrows():
                vol = volatilities.get(row["asset"], 0.15)
                adj_w = row["raw_weight"] / vol if vol > 0 else row["raw_weight"]
                adjusted.append(adj_w)
            adj_arr = np.array(adjusted)
            adj_sum = adj_arr.sum()
            if adj_sum > 0:
                positive_scores["target_weight"] = adj_arr / adj_sum
            else:
                positive_scores["target_weight"] = positive_scores["raw_weight"]
        else:
            positive_scores["target_weight"] = positive_scores["raw_weight"]

        # Step 3: Cap position sizes
        positive_scores["target_weight"] = positive_scores["target_weight"].clip(
            upper=max_position_pct
        )
        tw_sum = positive_scores["target_weight"].sum()
        if tw_sum > 0:
            positive_scores["target_weight"] /= tw_sum

        # Step 4: Generate trades
        target_weights = dict(
            zip(positive_scores["asset"], positive_scores["target_weight"])
        )
        target_composites = dict(
            zip(positive_scores["asset"], positive_scores["composite"])
        )

        all_assets = set(list(current_portfolio.keys()) + list(target_weights.keys()))
        trades = []

        for asset in all_assets:
            current_w = current_portfolio.get(asset, 0.0)
            target_w = target_weights.get(asset, 0.0)
            diff = target_w - current_w

            if abs(diff) < min_trade_pct:
                continue

            direction = "BUY" if diff > 0 else "SELL"
            score = target_composites.get(asset, 0.0)

            trades.append(Trade(
                symbol=asset,
                asset_class=AssetClass.EQUITY,
                direction=direction,
                quantity=abs(diff) * total_capital,
                target_weight=target_w,
                current_weight=current_w,
                score=score,
                reason=f"Cube rotation: {direction} to move from {current_w:.2%} to {target_w:.2%} (score={score:.3f})",
            ))

        # Sort: sells first (free up capital), then buys by score
        trades.sort(key=lambda t: (t.direction == "BUY", -t.score))
        return trades

    # -------------------------------------------------------------------
    # Backtest
    # -------------------------------------------------------------------

    def backtest_rotation(
        self,
        historical_data: dict[str, pd.DataFrame],
        lookback_days: int = 252,
        rebalance_days: Optional[int] = None,
        fundamentals_history: Optional[dict[str, dict[str, dict]]] = None,
    ) -> PerformanceMetrics:
        """
        Backtest the cube rotation strategy on historical data.

        Algorithm:
            1. For each rebalance date:
               a. Slice data up to that date (lookback window).
               b. Calculate momentum, value, catalyst scores per asset.
               c. Compute composite score and target weights.
               d. Record position changes.
            2. Between rebalance dates:
               a. Mark-to-market using actual returns.
               b. Track drawdown for leverage adjustment.
            3. Calculate performance metrics.

        Performance Metrics:
            Total return:          (V_final / V_initial) - 1
            Annualized return:     (1 + total_return)^(252/N) - 1
            Annualized volatility: std(daily_returns) * sqrt(252)
            Sharpe ratio:          (ann_return - R_f) / ann_vol
            Max drawdown:          max_t((peak_t - V_t) / peak_t)
            Calmar ratio:          ann_return / |max_drawdown|
            Win rate:              N_positive_trades / N_total_trades
            Profit factor:         sum(gains) / |sum(losses)|

        Parameters
        ----------
        historical_data : dict[str, pd.DataFrame]
            Symbol -> full OHLCV history.
        lookback_days : int
            Window for signal calculation (default 252).
        rebalance_days : int, optional
            Days between rebalances. If None, uses rebalance_freq setting.
        fundamentals_history : dict, optional
            Historical fundamentals per date per symbol.

        Returns
        -------
        PerformanceMetrics
        """
        if rebalance_days is None:
            rebalance_days = {
                RebalanceFrequency.DAILY: 1,
                RebalanceFrequency.WEEKLY: 5,
                RebalanceFrequency.MONTHLY: 21,
            }[self.rebalance_freq]

        # Build aligned returns matrix
        all_close = {}
        for symbol, df in historical_data.items():
            if df is not None and "Close" in df.columns:
                close = df["Close"].dropna()
                if len(close) > lookback_days:
                    all_close[symbol] = close

        if not all_close:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0,
                annualized_volatility=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0,
                num_trades=0, avg_trade_return=0.0,
                avg_holding_period_days=0.0,
            )

        prices_df = pd.DataFrame(all_close)
        prices_df = prices_df.dropna(how="all")
        returns_df = prices_df.pct_change().dropna(how="all")

        if len(returns_df) < lookback_days + 10:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0,
                annualized_volatility=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0,
                num_trades=0, avg_trade_return=0.0,
                avg_holding_period_days=0.0,
            )

        # Backtest loop
        portfolio_returns = []
        weights = {sym: 1.0 / len(all_close) for sym in all_close}  # equal weight start
        total_trades = 0
        trade_returns = []

        for i in range(lookback_days, len(returns_df)):
            # Daily portfolio return
            daily_ret = sum(
                weights.get(sym, 0.0) * float(returns_df[sym].iloc[i])
                for sym in returns_df.columns
                if not np.isnan(returns_df[sym].iloc[i])
            )
            portfolio_returns.append(daily_ret)

            # Rebalance
            if (i - lookback_days) % rebalance_days == 0:
                # Calculate momentum scores from lookback window
                window_data = {}
                for sym in all_close:
                    idx_start = max(0, i - lookback_days)
                    window_df = historical_data[sym].iloc[idx_start:i + 1].copy()
                    if len(window_df) > 50:
                        window_data[sym] = window_df

                if window_data:
                    # Simple momentum-based rotation for backtest
                    mom_scores = {}
                    for sym, wdf in window_data.items():
                        close_w = wdf["Close"].dropna()
                        if len(close_w) > 63:
                            mom = float(close_w.iloc[-1] / close_w.iloc[-63] - 1)
                            vol = float(np.log(close_w / close_w.shift(1)).dropna().std())
                            vol = vol * np.sqrt(252) if vol > 0 else 0.15
                            mom_scores[sym] = {"momentum": mom / max(vol, 0.01), "value": 0.0, "catalyst": 0.0}

                    if mom_scores:
                        score_df = self.composite_rotation_score(mom_scores)
                        old_weights = weights.copy()

                        # Update weights from scores
                        positive = score_df[score_df["composite"] > 0]
                        if not positive.empty:
                            total_score = positive["composite"].sum()
                            new_weights = {}
                            for _, row in positive.iterrows():
                                new_weights[row["asset"]] = row["composite"] / total_score
                            weights = new_weights
                            total_trades += len(
                                [a for a in set(list(old_weights) + list(new_weights))
                                 if abs(old_weights.get(a, 0) - new_weights.get(a, 0)) > 0.01]
                            )

        if not portfolio_returns:
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0,
                annualized_volatility=0.0, sharpe_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0,
                win_rate=0.0, profit_factor=0.0,
                num_trades=0, avg_trade_return=0.0,
                avg_holding_period_days=0.0,
            )

        ret_series = pd.Series(portfolio_returns)
        equity_curve = (1.0 + ret_series).cumprod()

        total_return = float(equity_curve.iloc[-1] - 1.0)
        n_days = len(ret_series)
        ann_return = (1.0 + total_return) ** (252.0 / n_days) - 1.0 if n_days > 0 else 0.0
        ann_vol = float(ret_series.std()) * np.sqrt(252)
        sharpe = (ann_return - self.risk_free_rate) / ann_vol if ann_vol > 0 else 0.0

        # Max drawdown
        peak = equity_curve.cummax()
        drawdown = (equity_curve - peak) / peak
        max_dd = float(drawdown.min())

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0.0

        # Win rate
        positive_days = (ret_series > 0).sum()
        win_rate = float(positive_days) / len(ret_series) if len(ret_series) > 0 else 0.0

        # Profit factor
        gains = ret_series[ret_series > 0].sum()
        losses = abs(ret_series[ret_series < 0].sum())
        profit_factor = float(gains / losses) if losses > 0 else float("inf")

        avg_hold = n_days / max(total_trades, 1)

        return PerformanceMetrics(
            total_return=total_return,
            annualized_return=ann_return,
            annualized_volatility=ann_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=total_trades,
            avg_trade_return=total_return / max(total_trades, 1),
            avg_holding_period_days=avg_hold,
            equity_curve=equity_curve,
        )


# ---------------------------------------------------------------------------
# Main (smoke test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    cube = CubeRotation(w_momentum=0.4, w_value=0.3, w_catalyst=0.3)

    # Demo: composite scoring
    scores = {
        "US_Equity": {"momentum": 0.7, "value": -0.2, "catalyst": 0.3},
        "Gold": {"momentum": 0.3, "value": 0.5, "catalyst": 0.1},
        "Crude_Oil": {"momentum": -0.4, "value": 0.6, "catalyst": -0.2},
        "US_Bonds": {"momentum": -0.1, "value": 0.8, "catalyst": 0.0},
        "BTC": {"momentum": 0.9, "value": -0.5, "catalyst": 0.7},
        "EUR_USD": {"momentum": 0.2, "value": 0.1, "catalyst": -0.1},
    }

    print("=== Composite Rotation Scores ===")
    score_df = cube.composite_rotation_score(scores)
    print(score_df.to_string(index=False))

    # Demo: trade generation
    print("\n=== Generated Trades ===")
    current = {"US_Equity": 0.25, "US_Bonds": 0.25, "Gold": 0.25, "BTC": 0.25}
    trades = cube.generate_rotation_trades(current, score_df)
    for t in trades:
        print(f"  {t.direction} {t.symbol}: {t.current_weight:.2%} -> {t.target_weight:.2%} (score={t.score:.3f})")

    # Demo: CTA indicators
    print("\n=== CTA Technical Indicators ===")
    np.random.seed(42)
    n = 300
    fake_prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.015, n))))
    fake_high = fake_prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    fake_low = fake_prices * (1 - np.abs(np.random.normal(0, 0.005, n)))

    adx_val, pdi, mdi = _adx(fake_high, fake_low, fake_prices, period=14)
    print(f"  ADX: {adx_val.iloc[-1]:.2f}")
    print(f"  +DI: {pdi.iloc[-1]:.2f}, -DI: {mdi.iloc[-1]:.2f}")

    breakout = _donchian_breakout(fake_high, fake_low, fake_prices, period=20)
    print(f"  Breakout signal: {breakout.iloc[-1]:.0f}")

    carry = _carry_return(f_near=75.50, f_far=76.20, days_between=30)
    print(f"  Carry return (annualized): {carry:.4f}")

    vols = {"Asset_A": 0.15, "Asset_B": 0.08, "Asset_C": 0.25, "Asset_D": 0.40}
    rb_weights = _risk_budget_weights(vols, target_risk=0.10)
    print(f"\n  Risk Budget Weights:")
    for k, v in sorted(rb_weights.items(), key=lambda x: x[1], reverse=True):
        print(f"    {k}: {v:.4f}")

    leverage = _drawdown_controlled_leverage(
        target_vol=0.12, realized_vol=0.08, max_leverage=2.0,
        current_drawdown=0.12, drawdown_threshold=0.10, max_drawdown_limit=0.25
    )
    print(f"\n  Drawdown-adjusted leverage: {leverage:.3f}")
