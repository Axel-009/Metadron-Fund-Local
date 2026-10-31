# region imports
from AlgorithmImports import *
# endregion

from corridor import (
    target_leverage, should_rebalance, annualised_drift, realised_vol,
    R_LOW, R_HIGH, HURDLE,
)


class MetadronCorridor(QCAlgorithm):
    """Build step 1 — smart-beta corridor overlay on SPY.

    Each day 30 minutes after the open:
        Rm      = realised SPY drift (EWMA-smoothed trailing annualised return)
        σ_m     = realised SPY volatility (63 trading days)
        state   = corridor β(Rm), γ(Rm), leverage = β × 4.7 × vol_scale, capped
        trade   = set SPY holdings to the target leverage when it moves > throttle

    Everything is plotted on the QuantConnect dashboard: Rm vs the 7–12% corridor and the
    7.32% hurdle, β, γ (the flattened bell), target vs actual leverage, volatility.

    Parameters (editable in the QuantConnect UI / config.json):
        vol_mode      linear | squared       (open design decision — CLAUDE.md)
        leverage_cap  hard gross cap on SPY exposure (default 2.0)
        rm_lookback   trading days for the drift estimate (default 252)
        rm_smoothing  EWMA weight on the newest drift estimate (default 0.10)
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.vol_mode = self.get_parameter("vol_mode", "linear")
        self.leverage_cap = float(self.get_parameter("leverage_cap", 2.0))
        self.rm_lookback = int(self.get_parameter("rm_lookback", 252))
        self.rm_smoothing = float(self.get_parameter("rm_smoothing", 0.10))

        equity = self.add_equity("SPY", Resolution.DAILY)
        equity.set_leverage(max(self.leverage_cap, 1.0) + 0.5)
        self.spy = equity.symbol
        self.set_benchmark(self.spy)

        self.rm_ewma = None
        self.current_leverage = 0.0
        self.set_warm_up(self.rm_lookback + 5, Resolution.DAILY)

        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.after_market_open(self.spy, 30),
                         self.rebalance)

    def rebalance(self):
        if self.is_warming_up:
            return
        hist = self.history(self.spy, self.rm_lookback + 1, Resolution.DAILY)
        if hist.empty or "close" not in hist.columns:
            self.debug("No SPY history — skipping (data-health)")
            return
        closes = [float(x) for x in hist["close"].values]
        if len(closes) < 64:
            return

        drift = annualised_drift(closes)
        self.rm_ewma = drift if self.rm_ewma is None else (
            self.rm_smoothing * drift + (1 - self.rm_smoothing) * self.rm_ewma)
        sigma = realised_vol(closes[-64:])

        state = target_leverage(self.rm_ewma, sigma, cap=self.leverage_cap, vol_mode=self.vol_mode)

        if should_rebalance(self.current_leverage, state.leverage):
            self.set_holdings(self.spy, state.leverage,
                              tag=f"corridor {state.position} Rm={state.rm:.2%} β={state.beta:.3f}")
            self.current_leverage = state.leverage

        actual = self.portfolio[self.spy].holdings_value / max(self.portfolio.total_portfolio_value, 1.0)
        self.plot("Rm vs corridor", "Rm (smoothed)", state.rm * 100)
        self.plot("Rm vs corridor", "Rm (raw)", drift * 100)
        self.plot("Rm vs corridor", "R_LOW 7%", R_LOW * 100)
        self.plot("Rm vs corridor", "R_HIGH 12%", R_HIGH * 100)
        self.plot("Rm vs corridor", "hurdle 7.32%", HURDLE * 100)
        self.plot("Corridor beta", "beta", state.beta)
        self.plot("Corridor gamma", "gamma per 1% Rm", state.gamma * 0.01)
        self.plot("Leverage", "target", state.leverage)
        self.plot("Leverage", "actual", actual)
        self.plot("Volatility", "realised 63d %", sigma * 100)
        self.plot("Volatility", "vol scale", state.vol_scale)
