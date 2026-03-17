"""AI-Newton Backend — PySR Symbolic Regression Engine.

Unsupervised discovery of mathematical relationships from raw market data.
Like the original AI-Newton paper (discovering Newton's laws from physics data),
this engine discovers financial "laws" from price/volume/fundamental data.

Examples of discoverable relationships:
    - Fair value formulas: P ≈ k₁·FCF^0.7 · growth^1.2 · (1/VIX)^0.3
    - Conservation laws: Gold·DXY^1.1 ≈ constant
    - Lead-lag formulas: Energy leads Financials by f(yield_curve_slope) days
    - Event sizing: |post-earnings move| = f(surprise, short_interest, vol_regime)

Dependencies:
    pip install pysr
    Julia runtime (auto-installed by PySR on first run)

Usage:
    from backends.pysr_newton.newton_backend import AINewtonEngine
    engine = AINewtonEngine()
    discoveries = engine.discover(universe_data)
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pysr import PySRRegressor
    _HAS_PYSR = True
    logger.info("PySR symbolic regression loaded")
except (ImportError, RuntimeError, Exception) as e:
    _HAS_PYSR = False
    PySRRegressor = None
    if "julia" in str(e).lower():
        logger.warning(f"PySR loaded but Julia runtime unavailable: {e}")
    else:
        logger.warning(f"PySR not available — install with: pip install pysr ({e})")

MODELS_DIR = Path(__file__).parent.parent.parent / "models" / "newton"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DiscoveredLaw:
    """A mathematical relationship discovered by AI-Newton."""
    source: str = "ai_newton"
    law_type: str = ""           # "fair_value" | "conservation" | "lead_lag" | "event_sizing"
    formula: str = ""            # symbolic formula string
    tickers: list[str] = field(default_factory=list)
    r_squared: float = 0.0
    complexity: int = 0          # formula complexity (lower = more fundamental)
    direction: int = 0           # trading implication: -1, 0, +1
    strength: float = 0.0
    confidence: float = 0.0
    description: str = ""
    metadata: dict = field(default_factory=dict)


class AINewtonEngine:
    """Symbolic regression engine for financial law discovery.

    Runs PySR on different data slices to find mathematical
    relationships that hold across the classified universe.
    """

    def __init__(
        self,
        n_iterations: int = 40,
        max_complexity: int = 20,
        populations: int = 15,
        population_size: int = 33,
    ):
        self.n_iterations = n_iterations
        self.max_complexity = max_complexity
        self.populations = populations
        self.population_size = population_size
        self._discoveries: list[DiscoveredLaw] = []
        self._models: dict = {}

        logger.info(f"AINewton initialized (PySR available: {_HAS_PYSR})")

    def _create_regressor(self, binary_operators: Optional[list] = None,
                           unary_operators: Optional[list] = None) -> "PySRRegressor":
        """Create a PySR regressor with financial-appropriate operators."""
        if not _HAS_PYSR:
            raise RuntimeError("PySR not installed")

        return PySRRegressor(
            niterations=self.n_iterations,
            maxsize=self.max_complexity,
            populations=self.populations,
            population_size=self.population_size,
            binary_operators=binary_operators or ["+", "-", "*", "/", "^"],
            unary_operators=unary_operators or ["log", "exp", "sqrt", "abs", "inv(x) = 1/x"],
            loss="loss(prediction, target) = (prediction - target)^2",
            model_selection="best",
            progress=False,
            verbosity=0,
            temp_equation_file=True,
        )

    def discover_fair_value(self, prices: dict[str, pd.Series],
                             fundamentals: dict[str, dict]) -> list[DiscoveredLaw]:
        """Discover fair value formulas from price + fundamental data.

        Looks for: P = f(revenue, earnings, growth, book_value, ...)
        """
        discoveries = []

        # Build feature matrix from fundamentals
        rows = []
        targets = []
        tickers_used = []

        for ticker, fund in fundamentals.items():
            if ticker not in prices or not fund:
                continue
            profile = fund.get("profile", {})
            if not profile:
                continue

            pe = profile.get("pe_ratio", 0)
            pb = profile.get("pb_ratio", 0)
            mcap = profile.get("market_cap", 0)
            div_yield = profile.get("dividend_yield", 0)
            beta = profile.get("beta", 1.0)

            if mcap <= 0 or pe <= 0:
                continue

            price = float(prices[ticker].iloc[-1])
            rows.append([pe, pb, div_yield or 0, beta or 1.0, np.log(mcap)])
            targets.append(np.log(price))
            tickers_used.append(ticker)

        if len(rows) < 10:
            logger.warning("Not enough fundamental data for fair value discovery")
            return discoveries

        X = np.array(rows)
        y = np.array(targets)

        if _HAS_PYSR:
            try:
                model = self._create_regressor()
                variable_names = ["PE", "PB", "DivYield", "Beta", "LogMcap"]
                model.fit(X, y, variable_names=variable_names)

                # Extract best equation
                best = model.get_best()
                formula = str(best["equation"]) if isinstance(best, dict) else str(best)
                r2 = float(best.get("r2", 0) if isinstance(best, dict) else model.score(X, y))

                discoveries.append(DiscoveredLaw(
                    law_type="fair_value",
                    formula=f"log(Price) = {formula}",
                    tickers=tickers_used,
                    r_squared=r2,
                    complexity=int(best.get("complexity", 0) if isinstance(best, dict) else 0),
                    strength=min(r2, 1.0),
                    confidence=min(r2 * 0.9, 1.0),
                    description=f"Fair value formula discovered across {len(tickers_used)} equities",
                    metadata={"variable_names": variable_names, "n_samples": len(rows)},
                ))

                self._models["fair_value"] = model
                logger.info(f"Fair value law discovered: R²={r2:.4f} formula={formula}")

            except Exception as e:
                logger.warning(f"PySR fair value discovery failed: {e}")
        else:
            # Numpy fallback: simple linear regression
            from numpy.linalg import lstsq
            X_aug = np.column_stack([X, np.ones(len(X))])
            coeffs, residuals, _, _ = lstsq(X_aug, y, rcond=None)
            y_pred = X_aug @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            names = ["PE", "PB", "DivYield", "Beta", "LogMcap"]
            terms = [f"{coeffs[i]:.4f}*{names[i]}" for i in range(len(names)) if abs(coeffs[i]) > 0.001]
            formula = " + ".join(terms) + f" + {coeffs[-1]:.4f}"

            discoveries.append(DiscoveredLaw(
                law_type="fair_value",
                formula=f"log(Price) = {formula}",
                tickers=tickers_used,
                r_squared=r2,
                complexity=len(terms),
                strength=min(r2, 1.0),
                confidence=min(r2 * 0.8, 1.0),
                description=f"Linear fair value (numpy fallback) across {len(tickers_used)} equities",
                metadata={"coefficients": coeffs.tolist(), "method": "OLS_fallback"},
            ))

        return discoveries

    def discover_conservation_laws(self, prices: dict[str, pd.Series]) -> list[DiscoveredLaw]:
        """Discover conservation-like relationships (near-constant ratios).

        Looks for: A * B^k ≈ constant (analogous to physics conservation laws).
        """
        discoveries = []
        tickers = list(prices.keys())

        for i, t1 in enumerate(tickers):
            for t2 in tickers[i+1:]:
                p1 = prices[t1].dropna()
                p2 = prices[t2].dropna()
                if len(p1) < 60 or len(p2) < 60:
                    continue

                # Align
                combined = pd.DataFrame({"a": p1, "b": p2}).dropna()
                if len(combined) < 40:
                    continue

                a, b = combined["a"].values, combined["b"].values

                # Check if ratio is approximately constant
                ratio = a / np.where(b != 0, b, 1e-10)
                cv = np.std(ratio) / np.mean(ratio) if np.mean(ratio) != 0 else 999

                if cv < 0.1:  # coefficient of variation < 10%
                    discoveries.append(DiscoveredLaw(
                        law_type="conservation",
                        formula=f"{t1} / {t2} ≈ {np.mean(ratio):.4f} (CV={cv:.4f})",
                        tickers=[t1, t2],
                        r_squared=1 - cv,
                        complexity=3,
                        strength=1 - cv,
                        confidence=min((0.1 - cv) / 0.1, 1.0),
                        description=f"Conservation law: {t1}/{t2} ratio is near-constant",
                        metadata={"mean_ratio": float(np.mean(ratio)),
                                  "cv": float(cv), "std": float(np.std(ratio))},
                    ))

                # Also check log-log relationship: log(A) = k*log(B) + c
                if np.all(a > 0) and np.all(b > 0):
                    log_a, log_b = np.log(a), np.log(b)
                    coeffs = np.polyfit(log_b, log_a, 1)
                    residuals = log_a - np.polyval(coeffs, log_b)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((log_a - log_a.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

                    if r2 > 0.85 and abs(coeffs[0]) > 0.1:
                        discoveries.append(DiscoveredLaw(
                            law_type="conservation",
                            formula=f"{t1} ≈ e^{coeffs[1]:.3f} * {t2}^{coeffs[0]:.3f}",
                            tickers=[t1, t2],
                            r_squared=r2,
                            complexity=5,
                            strength=r2,
                            confidence=min(r2 * 0.9, 1.0),
                            description=f"Power law: {t1} = k*{t2}^{coeffs[0]:.2f}",
                            metadata={"exponent": float(coeffs[0]),
                                      "constant": float(np.exp(coeffs[1]))},
                        ))

        return discoveries

    def discover_lead_lag(self, prices: dict[str, pd.Series],
                           max_lag: int = 10) -> list[DiscoveredLaw]:
        """Discover lead-lag relationships between instruments.

        Finds pairs where one instrument's returns predict another's.
        """
        discoveries = []
        tickers = list(prices.keys())
        returns = {t: prices[t].pct_change().dropna() for t in tickers if len(prices[t]) > 60}

        for t1 in returns:
            for t2 in returns:
                if t1 >= t2:
                    continue

                r1, r2 = returns[t1], returns[t2]
                combined = pd.DataFrame({"r1": r1, "r2": r2}).dropna()
                if len(combined) < 60:
                    continue

                best_lag = 0
                best_corr = 0.0

                for lag in range(1, max_lag + 1):
                    # Does t1 lead t2?
                    corr_fwd = combined["r1"].iloc[:-lag].corr(combined["r2"].iloc[lag:].reset_index(drop=True))
                    # Does t2 lead t1?
                    corr_bwd = combined["r2"].iloc[:-lag].corr(combined["r1"].iloc[lag:].reset_index(drop=True))

                    if abs(corr_fwd) > abs(best_corr):
                        best_corr = corr_fwd
                        best_lag = lag
                    if abs(corr_bwd) > abs(best_corr):
                        best_corr = corr_bwd
                        best_lag = -lag

                if abs(best_corr) > 0.2:
                    leader = t1 if best_lag > 0 else t2
                    follower = t2 if best_lag > 0 else t1
                    discoveries.append(DiscoveredLaw(
                        law_type="lead_lag",
                        formula=f"R({follower}, t) ≈ {best_corr:.3f} * R({leader}, t-{abs(best_lag)})",
                        tickers=[leader, follower],
                        r_squared=best_corr ** 2,
                        complexity=4,
                        direction=int(np.sign(best_corr)),
                        strength=abs(best_corr),
                        confidence=min(abs(best_corr) / 0.3, 1.0),
                        half_life_days=abs(best_lag),
                        description=f"{leader} leads {follower} by {abs(best_lag)} days",
                        metadata={"leader": leader, "follower": follower,
                                  "lag_days": abs(best_lag), "correlation": float(best_corr)},
                    ))

        return discoveries

    def discover(self, prices: dict[str, pd.Series],
                  fundamentals: Optional[dict[str, dict]] = None) -> list[DiscoveredLaw]:
        """Run all discovery modes on the universe.

        This is the main entry point for the Pattern Discovery Layer.

        Args:
            prices: Dict of ticker -> close price series.
            fundamentals: Optional dict of ticker -> fundamental data.

        Returns:
            List of DiscoveredLaw for the PatternDiscoveryBus.
        """
        all_discoveries = []

        # Conservation laws (ratios, power laws)
        logger.info("AI-Newton: Searching for conservation laws...")
        conservation = self.discover_conservation_laws(prices)
        all_discoveries.extend(conservation)
        logger.info(f"  Found {len(conservation)} conservation laws")

        # Lead-lag relationships
        logger.info("AI-Newton: Searching for lead-lag relationships...")
        lead_lag = self.discover_lead_lag(prices)
        all_discoveries.extend(lead_lag)
        logger.info(f"  Found {len(lead_lag)} lead-lag relationships")

        # Fair value formulas (if fundamentals available)
        if fundamentals:
            logger.info("AI-Newton: Searching for fair value formulas...")
            fair_value = self.discover_fair_value(prices, fundamentals)
            all_discoveries.extend(fair_value)
            logger.info(f"  Found {len(fair_value)} fair value formulas")

        # Sort by R² * confidence
        all_discoveries.sort(key=lambda d: d.r_squared * d.confidence, reverse=True)

        self._discoveries = all_discoveries
        logger.info(f"AI-Newton: {len(all_discoveries)} total laws discovered")

        return all_discoveries

    def get_discoveries(self) -> list[DiscoveredLaw]:
        return list(self._discoveries)

    def get_actionable(self, min_confidence: float = 0.5) -> list[DiscoveredLaw]:
        """Get only discoveries with trading implications."""
        return [d for d in self._discoveries
                if d.confidence >= min_confidence and d.direction != 0]
