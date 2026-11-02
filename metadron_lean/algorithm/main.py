# region imports
from AlgorithmImports import *
# endregion

from corridor import (
    target_leverage, should_rebalance, annualised_drift, realised_vol,
    R_LOW, R_HIGH, RISK_FREE, MAX_GROSS_LEVERAGE,
)
from risk_budget import beta_cap, hard_stops, daily_var, check_alpha_trade, AlphaPosition, momentum_confirms
from regime import RegimeModel, REGIMES
from forward_rm import estimate as forward_rm
from fundamentals import top_down, analyze, cap_structure_map
from universe import (SP1500_ETFS, EXPECTED_MEMBERS, SECTOR_TYPE, extract, subsector_for, passes_liquidity,
                      passes_fallback, shortlist, EpsHistory, PriceBuffer, SHORTLIST_N)
from rotation import (
    THEMES, DEFENSIVES, ESCALATION_SLEEVE, THEME_SUBSECTOR, build_plan, escalation_score, theme_index,
    realised_vol as name_vol,
)
import json
from macro_regime import factor_states, diagnose, market_bias, stock_bond_regime
from sector_rotation import evaluate, scenarios_from_json, summary
from market_map import CYCLICAL, SECULAR, DEFENSIVE


class MetadronCorridor(QCAlgorithm):
    """Build step 2b — smart-beta corridor v2 on the S&P 500.

    Daily, 30 minutes after the open:
        regime  = Gaussian HMM on SPY return / vol / VIX / HY−IG credit (regime.py; rules fallback)
        Rm      = forward estimate (forward_rm.py): valuation anchor + shrunk EWMA trend
                  + Σ P(regime)·drift + GMTF tension + Kronos (when the macro agent supplies them)
        σm      = 63-day realised volatility
        cap     = min(4.5x, regime VaR ceiling × beta burst when momentum confirms)

    Universe (step 3): point-in-time S&P 1500 (IVV + IJH + IJR constituents) → Morningstar fundamentals →
    Graham-Dodd security analysis + capital structure → Fundamentals gate → shortlist by sleeve
    (IG / HY / DISTRESSED) tilted by the sub-sector view. Rescored every `fundamental_refresh_days`;
    the aggregate earnings yield anchors forward Rm; Merton PD / coverage feed the LEAPS solvency check.
    Shortlisted names are subscribed for the decision matrix (step 5) — not traded yet.
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
        self.universe_mode = str(self.get_parameter("universe_mode", "sp1500")).lower()
        self.refresh_days = int(self.get_parameter("fundamental_refresh_days", 5))
        self.shortlist_n = int(self.get_parameter("shortlist_size", SHORTLIST_N))

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
        self.regime_model = RegimeModel()
        self.last_views = {}
        self._init_universe()
        self.current_leverage = 0.0
        self.peak_nav = self.portfolio.total_portfolio_value
        self.set_warm_up(self.rm_lookback + 5, Resolution.DAILY)
        self.schedule.on(self.date_rules.every_day(self.spy),
                         self.time_rules.after_market_open(self.spy, 30),
                         self.rebalance)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------ universe (step 3)
    def _init_universe(self):
        self.members = {}                     # etf → set(ticker)
        self.eps_hist, self.px_buf = EpsHistory(), PriceBuffer()
        self.scores, self.td, self.health = {}, None, {}
        self.selected, self.last_score_time = [], None
        self.subsector_overrides = {}
        self.universe_settings.resolution = Resolution.DAILY
        if self.universe_mode == "sp1500":
            for etf in SP1500_ETFS:
                try:
                    self.add_universe(self.universe.etf(etf, self.universe_settings,
                                                        lambda c, e=etf: self._etf_members(e, c)))
                except Exception as exc:
                    self.debug(f"universe: ETF constituents {etf} unavailable ({exc}) — fallback filter")
                    self.universe_mode = "fundamental"
        self.add_universe(self._fundamental_selection)

    def _etf_members(self, etf, constituents):
        self.members[etf] = {c.symbol.value for c in constituents}      # tickers (same-day feeds)
        return []                              # membership only; subscriptions come from the shortlist

    def _fundamental_selection(self, fundamental):
        members = set().union(*self.members.values()) if self.members else set()
        extra = {str(t) for t in getattr(self, "rot_symbols", {})}
        days = (self.time - self.last_score_time).days if self.last_score_time is not None else 999
        rescore = days >= self.refresh_days
        rows, missing_n, expected = {}, 0, 0
        for f in fundamental:
            sym = f.symbol
            t = sym.value
            in_universe = t in members if self.universe_mode == "sp1500" else bool(getattr(f, "has_fundamental_data", False))
            if not in_universe and t not in extra:
                continue
            self.px_buf.update(t, getattr(f, "price", None))
            if not rescore:
                continue
            d, missing = extract(f)
            if self.universe_mode != "sp1500" and t not in extra and not passes_fallback(d):
                continue
            expected += 1
            if missing:
                missing_n += 1
                continue
            self.eps_hist.update(t, d.get("eps_ttm"))
            rows[t] = (sym, d)
        if not rescore:
            return self.selected
        if self.subsector_overrides == {}:
            raw = self._store_read("metadron/subsector_map.json")
            try:
                self.subsector_overrides = json.loads(raw) if raw else {"_": "_"}
            except Exception:
                self.subsector_overrides = {"_": "_"}
        macro = self.macro_inputs()
        vix = None
        if getattr(self, "vix", None) is not None:
            v = self._closes(self.vix, 3)
            vix = v[-1] if v else None
        self.td = top_down([d for _, d in rows.values()], treasury_10y=macro.get("treasury_10y", 0.042), vix=vix)
        scores = {}
        for t, (sym, d) in rows.items():
            code = d.get("sector_code")
            try:
                stype = SECTOR_TYPE.get(int(code), "") if code is not None else ""
            except (TypeError, ValueError):
                stype = ""
            sc = analyze(t, d, self.td, equity_vol=self.px_buf.vol(t), eps_history=self.eps_hist.annual(t),
                         sector_type=stype)
            sc.subsector = subsector_for(t, code, d.get("group_code"), d.get("industry_code"), self.subsector_overrides)
            if t in extra or (passes_liquidity(d) or self.universe_mode != "sp1500"):
                scores[t] = sc
        self.scores = scores
        held = [str(k.value) for k, h in self.portfolio.items() if h.invested] if hasattr(self.portfolio, "items") else []
        sl = shortlist({t: s for t, s in scores.items() if t not in extra}, self.last_views, self.shortlist_n, held)
        sym_of = {t: sym for t, (sym, _) in rows.items()}
        self.selected = [sym_of[t] for t in sl.names if t in sym_of]
        self.last_score_time = self.time
        self.health = {"members": len(members) if self.universe_mode == "sp1500" else expected,
                       "expected": EXPECTED_MEMBERS if self.universe_mode == "sp1500" else expected,
                       "scored": len(scores), "missing_fields": missing_n,
                       "gate_pass": sum(1 for s in scores.values() if s.gate_pass),
                       "shortlist": len(sl.names), "vetoed": len(sl.vetoed)}
        self._publish_universe(sl)
        return self.selected

    def _publish_universe(self, sl):
        h, td = self.health, self.td
        for k in ("members", "scored", "missing_fields", "gate_pass", "shortlist"):
            self.plot("Data health", k, h.get(k, 0))
        self.plot("Valuation", "earnings yield %", td.earnings_yield * 100)
        self.plot("Valuation", "ERP %", td.erp * 100)
        self.plot("Valuation", "market P/E", min(td.market_pe, 60))
        self.plot("Valuation", "max investment P/E", td.max_pe)
        if h.get("members", 0) < 0.9 * h.get("expected", 1) or h.get("missing_fields", 0) > 0.2 * max(h.get("members", 1), 1):
            self.debug(f"DATA-HEALTH alert {h}")
        top = sorted(sl.rank.items(), key=lambda x: -x[1])[:100]
        report = {"time": str(self.time), "mode": self.universe_mode, "health": h, "top_down": td.__dict__,
                  "shortlist": sl.by_sleeve, "vetoed": sl.vetoed[:50],
                  "scores": {t: self.scores[t].to_dict() for t, _ in top},
                  "watch": {t: self.scores[t].to_dict() for t in self.scores if t in getattr(self, "rot_symbols", {})}}
        try:
            self.object_store.save("metadron/universe_scores.json", json.dumps(report, default=str))
        except Exception:
            pass
        self.debug(f"UNIVERSE {h} | EY {td.earnings_yield:.2%} ERP {td.erp:+.2%} {td.regime} | "
                   + " ".join(f"{k}:{','.join(v[:5])}" for k, v in sl.by_sleeve.items()))

    def subsector_fundamentals(self) -> dict:
        """{subsector: −1..1} = 2 × (mean gate score − 0.5) over scored names → sector_rotation input."""
        agg = {}
        for s in self.scores.values():
            agg.setdefault(s.subsector, []).append(s.gate_score)
        return {k: max(-1.0, min(1.0, 2 * (sum(v) / len(v) - 0.5))) for k, v in agg.items() if len(v) >= 3}

    def macro_inputs(self) -> dict:
        """Numeric readings written by the macro agent (FRED, GMTF, Kronos) — {} if absent/stale."""
        raw = self._store_read("metadron/macro_inputs.json")
        if not raw:
            return {}
        try:
            return {k: float(v) for k, v in json.loads(raw).items() if isinstance(v, (int, float))}
        except Exception as exc:
            self.debug(f"macro_inputs.json unreadable: {exc}")
            return {}

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

        long = self._closes(self.spy, 800) or closes
        vix_hist = self._closes(self.vix, 800) if self.vix is not None else []
        hyg_sym, lqd_sym = self.rot_symbols.get("HYG"), self.rot_symbols.get("LQD")
        hyg = self._closes(hyg_sym, 800) if hyg_sym else []
        lqd = self._closes(lqd_sym, 800) if lqd_sym else []
        reg = self.regime_model.classify(long, vix_hist or None, hyg or None, lqd or None)
        regime = reg.regime

        macro = self.macro_inputs()
        ey = macro.get("earnings_yield")
        if ey is None and self.td is not None and self.td.n_names >= 100:
            ey = self.td.earnings_yield            # universe aggregate (step 3)
        fwd = forward_rm(self.rm_ewma, regime_probs=reg.probs, gmtf_score=macro.get("gmtf_score"),
                         earnings_yield=ey, inflation=macro.get("inflation_yoy"),
                         kronos=macro.get("kronos_spx_annual"))
        rm = fwd.rm
        burst = regime != "CRASH" and momentum_confirms(long)
        cap = beta_cap(regime, sigma, rm=rm, burst=burst)   # no double sizing above Rm 12%

        state = target_leverage(rm, sigma, k=self.k, cap=cap)
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
                target, tag=f"{state.zone} Rm={state.rm:.2%} σ={sigma:.1%} {regime}"
                            f"{' burst' if burst else ''} stop={stop.action}")
            self.current_leverage = target

        held = (self.portfolio[self.future.mapped].holdings_value
                if self.future is not None and self.future.mapped is not None
                else self.portfolio[self.spy].holdings_value)
        actual = held / max(nav, 1.0)

        self.plot("Rm vs corridor", "Rm (forward)", state.rm * 100)
        self.plot("Rm vs corridor", "trailing EWMA", self.rm_ewma * 100)
        self.plot("Rm vs corridor", "valuation anchor", fwd.components["anchor"] * 100)
        for name in REGIMES:
            self.plot("Regime", f"P({name})", reg.probs.get(name, 0.0))
        self.plot("Regime", "burst", 1.0 if burst else 0.0)
        if "gmtf_score" in macro:
            self.plot("Regime", "GMTF tension pct", macro["gmtf_score"])
        if regime != getattr(self, "last_regime", None):
            self.last_regime = regime
            self.debug(f"REGIME {regime} ({reg.method}, p={reg.confidence:.2f}) Rm={rm:.2%} "
                       f"components={ {k: round(v, 4) for k, v in fwd.components.items()} }")
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
        readings = self.macro_inputs()                               # written by the macro agent (FRED)
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
        views = evaluate(factors, scenarios=scenarios, findings=findings, theme_states=themes, bias=bias["bias"],
                         fundamentals=self.subsector_fundamentals())
        self.last_views = {v.subsector: {"edge": v.edge, "weight": v.weight} for v in views}

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
        plan = build_plan(zone, prices, spy_closes, holdings, escalation=esc,
                          cap_structure=cap_structure_map(self.scores) if self.scores else None)
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
