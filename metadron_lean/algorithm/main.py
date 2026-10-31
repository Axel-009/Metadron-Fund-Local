# region imports
from AlgorithmImports import *
# endregion

from corridor import (
    target_leverage, should_rebalance, annualised_drift, realised_vol,
    R_LOW, R_HIGH, RISK_FREE, MAX_GROSS_LEVERAGE,
)
from risk_budget import beta_cap, hard_stops, daily_var, check_alpha_trade, AlphaPosition
from rotation import (
    THEMES, DEFENSIVES, ESCALATION_SLEEVE, THEME_SUBSECTOR, build_plan, escalation_score, theme_index,
    realised_vol as name_vol,
)
import json
from macro_regime import factor_states, diagnose, market_bias, stock_bond_regime
from sector_rotation import evaluate, scenarios_from_json, summary
from market_map import CYCLICAL, SECULAR, DEFENSIVE


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

    Theme-crash rotation (rotation.py): daily theme states (e.g. AI credit hubs / over-spenders),
    geopolitical-escalation score and a rotation plan — cut crashed-theme longs, rotate into
    defensives, escalation posture (oil calls, refiners), harvest the drawdown (LEAPS on solvent
    names, straddles). Market beta is NOT cut twice: above Rm 12% a STRESS/CRASH regime does not
    apply its VaR ceiling on top of Kelly.

    Parameters: kelly_fraction (1.0), beta_instrument (MES | SPY), rm_lookback (252),
                rm_smoothing (0.10), rotation_trading (false — plan is plotted/logged only)
    """

    def initialize(self):
        self.set_start_date(2010, 1, 1)
        self.set_cash(1_000_000)
        self.set_brokerage_model(BrokerageName.INTERACTIVE_BROKERS_BROKERAGE, AccountType.MARGIN)

        self.k = float(self.get_parameter("kelly_fraction", 1.0))
        self.instrument = self.get_parameter("beta_instrument", "MES").upper()
        self.rm_lookback = int(self.get_parameter("rm_lookback", 252))
        self.rm_smoothing = float(self.get_parameter("rm_smoothing", 0.10))
        self.rotation_trading = str(self.get_parameter("rotation_trading", "false")).lower() == "true"

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

        # Theme / defensive / escalation names + escalation inputs
        tickers = sorted({t for v in THEMES.values() for t in v} | {t for v in DEFENSIVES.values() for t in v}
                         | set(ESCALATION_SLEEVE["refiners"]) | set(ESCALATION_SLEEVE["oil_option_underlyings"])
                         | {"GLD", "TLT", "HYG", "LQD"})
        self.rot_symbols = {}
        for t in tickers:
            try:
                self.rot_symbols[t] = self.add_equity(t, Resolution.DAILY).symbol
            except Exception as exc:          # unknown / not-yet-listed ticker → skip
                self.debug(f"rotation: cannot add {t}: {exc}")
        try:
            self.vix = self.add_index("VIX", Resolution.DAILY).symbol
        except Exception:
            self.vix = None
        self.last_plan_sig = None

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
        cap = beta_cap(regime, sigma, rm=self.rm_ewma)     # no double sizing above Rm 12%

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

        self.run_rotation(state.zone, closes)

    # ------------------------------------------------------------------
    def _closes(self, symbol, n=260):
        try:
            h = self.history(symbol, n, Resolution.DAILY)
            if h is None or h.empty or "close" not in h.columns:
                return []
            return [float(x) for x in h["close"].values]
        except Exception:
            return []

    def _store_read(self, key):
        try:
            if self.object_store.contains_key(key):
                return self.object_store.read(key)
        except Exception:
            pass
        return None

    def run_sector_view(self, zone, prices, spy_closes, esc, vix, plan):
        """Trader framework: macro question tree + scenarios → sub-sector edge / downside / tilts."""
        readings = {}
        raw = self._store_read("metadron/macro_inputs.json")       # written by the macro agent (FRED)
        if raw:
            try:
                readings.update({k: float(v) for k, v in json.loads(raw).items() if isinstance(v, (int, float))})
            except Exception as exc:
                self.debug(f"macro_inputs.json unreadable: {exc}")
        readings.setdefault("escalation", esc)
        if vix is not None:
            readings.setdefault("vix", vix)
        hyg, lqd = prices.get("HYG"), prices.get("LQD")
        if hyg and lqd and len(hyg) > 22 and len(lqd) > 22:
            rel = (hyg[-1] / hyg[-22]) - (lqd[-1] / lqd[-22])
            readings.setdefault("hy_oas_change_1m", -rel / 4.0)      # proxy: HY-vs-IG return / duration
        uso = prices.get("USO")
        if uso and len(uso) > 22:
            readings.setdefault("commodities_21d", uso[-1] / uso[-22] - 1)
        scen_raw = self._store_read("metadron/scenarios.json")      # written by the macro agent / owner
        scenarios = None
        if scen_raw:
            try:
                scenarios = scenarios_from_json(scen_raw)
            except Exception as exc:
                self.debug(f"scenarios.json unreadable: {exc}")

        factors = factor_states(readings)
        findings = diagnose(readings)
        bias = market_bias(factors, zone)
        sb = stock_bond_regime(spy_closes, prices.get("TLT", []))
        themes = {THEME_SUBSECTOR[t.theme]: t.state for t in plan.themes if t.theme in THEME_SUBSECTOR}
        views = evaluate(factors, scenarios=scenarios, findings=findings, theme_states=themes, bias=bias["bias"])

        mix = {CYCLICAL: 0.0, SECULAR: 0.0, DEFENSIVE: 0.0}
        for v in views:
            mix[v.nature] += v.weight
        self.plot("Market view", "bias score", bias["score"])
        self.plot("Market view", "stock-bond corr", sb["corr"] if sb["corr"] is not None else 0.0)
        for n, w in mix.items():
            self.plot("Equity mix", n.lower(), w)

        report = {"time": str(self.time), "zone": zone, "bias": bias, "stock_bond": sb,
                  "findings": [f.__dict__ for f in findings], "factors": factors, **summary(views, 8)}
        try:
            self.object_store.save("metadron/sector_views.json", json.dumps(report, default=str))
        except Exception:
            pass
        top = ", ".join(f"{v.subsector} {v.weight:.0%}" for v in views if v.weight > 0.05)
        sig = (bias["bias"], tuple(sorted(v.subsector for v in views if v.weight > 0.05)))
        last_t = getattr(self, "last_view_time", None)
        if sig != getattr(self, "last_view_sig", None) and (
                last_t is None or not hasattr(self.time, "__sub__") or (self.time - last_t).days >= 5):
            self.last_view_sig, self.last_view_time = sig, self.time
            self.debug(f"VIEW {bias['bias']} ({bias['score']:+.2f}) hedge={sb['hedge']} | {top}")

    def run_rotation(self, zone: str, spy_closes):
        prices = {t: c for t, sym in self.rot_symbols.items() if len(c := self._closes(sym)) > 21}
        nav = max(self.portfolio.total_portfolio_value, 1.0)
        holdings = {t: self.portfolio[sym].holdings_value / nav for t, sym in self.rot_symbols.items()}
        vix = None
        if self.vix is not None:
            v = self._closes(self.vix, 5)
            vix = v[-1] if v else None
        defense = theme_index(prices, DEFENSIVES["DEFENSE"])
        esc = escalation_score(oil=prices.get("USO"), gold=prices.get("GLD"),
                               defense_idx=defense or None, spy=spy_closes, vix=vix)
        plan = build_plan(zone, prices, spy_closes, holdings, escalation=esc)
        self.run_sector_view(zone, prices, spy_closes, esc, vix, plan)

        code = {"NORMAL": 0, "STRESS": 1, "CRASH": 2}
        for ts in plan.themes:
            self.plot("Themes", f"{ts.theme} state", code[ts.state])
            self.plot("Themes", f"{ts.theme} drawdown %", ts.drawdown * 100)
        self.plot("Themes", "escalation score", esc)

        sig = tuple((a.kind, a.ticker, round(a.target_weight, 3)) for a in plan.actions)
        if sig != self.last_plan_sig:
            self.last_plan_sig = sig
            for a in plan.actions:
                self.debug(f"ROTATION {a.kind} {a.ticker} w={a.target_weight:.3f} "
                           f"premium={a.premium:.3%} dte={a.dte} — {a.reason}")

        if not self.rotation_trading:
            return
        book = [AlphaPosition(t, w, name_vol(prices.get(t, []), 20) or 0.4)
                for t, w in holdings.items() if abs(w) > 1e-6]
        for a in plan.actions:
            if a.kind not in ("CUT", "ROTATE", "ESCALATION_EQUITY") or a.ticker not in self.rot_symbols:
                continue                      # option legs: options engine (build step 6)
            target = a.target_weight
            if target > holdings.get(a.ticker, 0.0):
                vol = name_vol(prices.get(a.ticker, []), 20) or 0.4
                chk = check_alpha_trade(book, AlphaPosition(a.ticker, target - holdings.get(a.ticker, 0.0), vol))
                target = holdings.get(a.ticker, 0.0) + chk.max_weight_allowed
            if abs(target - holdings.get(a.ticker, 0.0)) < 0.005:      # skip < 0.5% NAV changes (no churn)
                continue
            self.set_holdings(self.rot_symbols[a.ticker], target, tag=f"ROTATION {a.kind}: {a.reason}")
