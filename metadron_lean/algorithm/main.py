# region imports
from AlgorithmImports import *
# endregion

from corridor import (
    target_leverage, should_rebalance, annualised_drift, realised_vol,
    R_LOW, R_HIGH, RISK_FREE, MAX_GROSS_LEVERAGE,
)
from risk_budget import beta_cap, hard_stops, daily_var


class MetadronCorridor(QCAlgorithm):
    """Build step 2a — smart-beta corridor v2 on the S&P 500.

    Daily, 30 minutes after the open:
        Rm      = annualised S&P 500 return estimate (EWMA-smoothed trailing drift;
                  replaced by the HMM/GMTF/Kronos forward estimate in step 2b)
        σm      = 63-day realised volatility
        regime  = placeholder from σm (TRENDING/RANGE/STRESS/CRASH) until the HMM (step 2b)
        cap     = min(4.5x, regime VaR ceiling)
        L       = k · s(Rm) · (Rm − 4%)/σm²   → 0 below 7%, ramp 7–12%, full Kelly above 12%
        hard stops: drawdown −4% trim (halve), −8% flat
    Market beta is delivered with Micro E-mini S&P 500 futures (default) or SPY (≤ 2x Reg-T).

    Parameters: kelly_fraction (1.0), beta_instrument (MES | SPY), rm_lookback (252),
                rm_smoothing (0.10)
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.k = float(self.get_parameter("kelly_fraction", 1.0))
        self.instrument = self.get_parameter("beta_instrument", "MES").upper()
        self.rm_lookback = int(self.get_parameter("rm_lookback", 252))
        self.rm_smoothing = float(self.get_parameter("rm_smoothing", 0.10))

        # Signal source: SPY (index proxy) for Rm and σm
        spy = self.add_equity("SPY", Resolution.DAILY)
        spy.set_leverage(2.0)
        self.spy = spy.symbol
        self.set_benchmark(self.spy)

        if self.instrument == "MES":
            self.future = self.add_future(
                Futures.Indices.MICRO_SP_500_E_MINI, Resolution.DAILY,
                data_normalization_mode=DataNormalizationMode.BACKWARDS_RATIO,
                data_mapping_mode=DataMappingMode.OPEN_INTEREST,
                contract_depth_offset=0)
            self.future.set_leverage(MAX_GROSS_LEVERAGE + 0.5)
        else:
            self.future = None

        self.rm_ewma = None
        self.current_leverage = 0.0
        self.peak_nav = self.portfolio.total_portfolio_value
        self.set_warm_up(self.rm_lookback + 5, Resolution.DAILY)
        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.after_market_open(self.spy, 30),
                         self.rebalance)

    # ------------------------------------------------------------------
    @staticmethod
    def placeholder_regime(sigma: float) -> str:
        if sigma < 0.15:
            return "TRENDING"
        if sigma < 0.22:
            return "RANGE"
        if sigma < 0.35:
            return "STRESS"
        return "CRASH"

    def _set_beta_leverage(self, leverage: float, tag: str):
        nav = self.portfolio.total_portfolio_value
        if self.future is None:
            self.set_holdings(self.spy, min(leverage, 2.0), tag=tag)
            return
        contract = self.future.mapped
        if contract is None:
            return
        sec = self.securities[contract]
        price = sec.price
        if price <= 0:
            return
        mult = sec.symbol_properties.contract_multiplier
        target_qty = int(round(leverage * nav / (price * mult)))
        delta = target_qty - int(self.portfolio[contract].quantity)
        if delta != 0:
            self.market_order(contract, delta, tag=tag)

    def on_symbol_changed_events(self, symbol_changed_events):
        """Roll the futures position into the newly mapped contract."""
        for _, changed in symbol_changed_events.items():
            old, new = changed.old_symbol, changed.new_symbol
            qty = self.portfolio[old].quantity
            if qty != 0:
                self.liquidate(old, tag="roll")
                self.market_order(new, qty, tag="roll")

    # ------------------------------------------------------------------
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
        regime = self.placeholder_regime(sigma)
        cap = beta_cap(regime, sigma)

        state = target_leverage(self.rm_ewma, sigma, k=self.k, cap=cap)
        target = state.leverage

        nav = self.portfolio.total_portfolio_value
        self.peak_nav = max(self.peak_nav, nav)
        stop = hard_stops(nav, self.peak_nav, self.current_leverage)
        if stop.action == "CLOSE":
            target = 0.0
        elif stop.action == "TRIM":
            target = min(target, self.current_leverage * 0.5)
        if stop.action == "CLOSE" and nav >= self.peak_nav * 0.97:
            self.peak_nav = nav       # re-arm once recovered close to the old peak

        if should_rebalance(self.current_leverage, target):
            self._set_beta_leverage(
                target, tag=f"{state.zone} Rm={state.rm:.2%} σ={sigma:.1%} {regime} stop={stop.action}")
            self.current_leverage = target

        held = (self.portfolio[self.future.mapped].holdings_value
                if self.future is not None and self.future.mapped is not None
                else self.portfolio[self.spy].holdings_value)
        actual = held / max(nav, 1.0)

        self.plot("Rm vs corridor", "Rm (smoothed)", state.rm * 100)
        self.plot("Rm vs corridor", "Rm (raw)", drift * 100)
        self.plot("Rm vs corridor", "7% floor", R_LOW * 100)
        self.plot("Rm vs corridor", "12% ceiling", R_HIGH * 100)
        self.plot("Rm vs corridor", "risk-free 4%", RISK_FREE * 100)
        self.plot("Corridor", "participation s(Rm)", state.participation)
        self.plot("Corridor", "gamma ds/dRm per 1%", state.gamma * 0.01)
        self.plot("Leverage", "Kelly (Rm−rf)/σ²", max(min(state.kelly, 6.0), -2.0))
        self.plot("Leverage", "cap", cap)
        self.plot("Leverage", "target", target)
        self.plot("Leverage", "actual", actual)
        self.plot("Risk", "realised vol %", sigma * 100)
        self.plot("Risk", "beta VaR95 1d %", daily_var(actual, sigma) * 100)
        self.plot("Risk", "drawdown %", (1 - nav / self.peak_nav) * 100)
