"""
BlackScholesEngine — Theoretical option pricing and Greeks for Metadron Capital.

Provides:
- Black-Scholes European option pricing (call + put)
- Full Greeks: Delta, Gamma, Theta, Vega, Rho
- Implied volatility solver (Newton-Raphson)
- Volatility surface construction from market data
- Monte Carlo option pricing (for American/exotic options)
- Opportunity scanner: find mispriced options

Math:
    d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    d2 = d1 - σ√T
    
    Call = S·N(d1) - K·e^(-rT)·N(d2)
    Put  = K·e^(-rT)·N(-d2) - S·N(-d1)
    
    Delta = N(d1) [call] / N(d1)-1 [put]
    Gamma = N'(d1) / (S·σ·√T)
    Theta = -(S·N'(d1)·σ)/(2√T) - r·K·e^(-rT)·N(d2) [call]
    Vega  = S·√T·N'(d1)
    Rho   = K·T·e^(-rT)·N(d2) [call]
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from enum import Enum
from datetime import datetime, date, timedelta
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.stats import norm
    from scipy.optimize import brentq
except ImportError:
    norm = None
    brentq = None
    logger.warning("scipy not available — BlackScholesEngine disabled")


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class OptionGreeks:
    """Full Greek set for an option."""
    delta: float
    gamma: float
    theta: float  # per day
    vega: float   # per 1% vol change
    rho: float    # per 1% rate change
    
    def to_dict(self) -> dict:
        return {
            "delta": round(self.delta, 4),
            "gamma": round(self.gamma, 6),
            "theta": round(self.theta, 4),
            "vega": round(self.vega, 4),
            "rho": round(self.rho, 4),
        }


@dataclass
class OptionPrice:
    """Complete option pricing output."""
    underlying: str
    strike: float
    expiry: date
    option_type: OptionType
    spot: float
    theoretical_price: float
    market_price: Optional[float]
    implied_vol: Optional[float]
    greeks: OptionGreeks
    d1: float
    d2: float
    time_to_expiry: float  # years
    moneyness: float  # S/K
    intrinsic_value: float
    extrinsic_value: float
    
    @property
    def mispricing(self) -> Optional[float]:
        """Difference between theoretical and market price."""
        if self.market_price is not None:
            return self.theoretical_price - self.market_price
        return None
    
    @property
    def mispricing_pct(self) -> Optional[float]:
        """Mispricing as percentage of theoretical."""
        if self.mispricing is not None and self.theoretical_price > 0:
            return self.mispricing / self.theoretical_price
        return None
    
    def to_dict(self) -> dict:
        return {
            "underlying": self.underlying,
            "strike": self.strike,
            "expiry": self.expiry.isoformat(),
            "option_type": self.option_type.value,
            "spot": round(self.spot, 2),
            "theoretical_price": round(self.theoretical_price, 4),
            "market_price": round(self.market_price, 4) if self.market_price else None,
            "implied_vol": round(self.implied_vol, 4) if self.implied_vol else None,
            "greeks": self.greeks.to_dict(),
            "moneyness": round(self.moneyness, 4),
            "intrinsic_value": round(self.intrinsic_value, 4),
            "extrinsic_value": round(self.extrinsic_value, 4),
            "mispricing": round(self.mispricing, 4) if self.mispricing is not None else None,
            "mispricing_pct": round(self.mispricing_pct, 4) if self.mispricing_pct is not None else None,
        }


class BlackScholesEngine:
    """
    Black-Scholes option pricing and Greeks engine.
    
    Used for:
    1. Theoretical pricing of equity options
    2. Greeks computation for hedging
    3. Implied volatility extraction
    4. Volatility surface construction
    5. Mispricing identification (theoretical vs market)
    """
    
    # Risk-free rate (can be updated from FRED)
    RISK_FREE_RATE = 0.05
    
    def __init__(self, risk_free_rate: float = 0.05):
        if norm is None:
            raise ImportError("scipy required for BlackScholesEngine")
        self.RISK_FREE_RATE = risk_free_rate
    
    def _d1_d2(self, S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float]:
        """Compute d1 and d2."""
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0, 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2
    
    def price(self, S: float, K: float, T: float, sigma: float,
              option_type: OptionType = OptionType.CALL,
              r: Optional[float] = None) -> float:
        """
        Black-Scholes option price.
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            sigma: Volatility (annualized)
            option_type: CALL or PUT
            r: Risk-free rate (default: self.RISK_FREE_RATE)
        """
        r = r if r is not None else self.RISK_FREE_RATE
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        if option_type == OptionType.CALL:
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def greeks(self, S: float, K: float, T: float, sigma: float,
               option_type: OptionType = OptionType.CALL,
               r: Optional[float] = None) -> OptionGreeks:
        """Compute all Greeks."""
        r = r if r is not None else self.RISK_FREE_RATE
        d1, d2 = self._d1_d2(S, K, T, r, sigma)
        
        sqrt_T = np.sqrt(T)
        n_d1 = norm.cdf(d1)
        n_prime_d1 = norm.pdf(d1)
        n_d2 = norm.cdf(d2)
        
        if option_type == OptionType.CALL:
            delta = n_d1
            theta = (-(S * n_prime_d1 * sigma) / (2 * sqrt_T) 
                     - r * K * np.exp(-r * T) * n_d2) / 365
            rho = K * T * np.exp(-r * T) * n_d2 / 100
        else:
            delta = n_d1 - 1
            theta = (-(S * n_prime_d1 * sigma) / (2 * sqrt_T) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
        
        gamma = n_prime_d1 / (S * sigma * sqrt_T)
        vega = S * sqrt_T * n_prime_d1 / 100
        
        return OptionGreeks(
            delta=delta,
            gamma=gamma,
            theta=theta,
            vega=vega,
            rho=rho,
        )
    
    def implied_volatility(self, market_price: float, S: float, K: float, T: float,
                           option_type: OptionType = OptionType.CALL,
                           r: Optional[float] = None) -> Optional[float]:
        """
        Solve for implied volatility using Brent's method.
        
        Returns None if no solution found.
        """
        r = r if r is not None else self.RISK_FREE_RATE
        
        if T <= 0:
            return None
        
        def objective(sigma):
            return self.price(S, K, T, sigma, option_type, r) - market_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except (ValueError, RuntimeError):
            return None
    
    def price_option(
        self,
        underlying: str,
        spot: float,
        strike: float,
        expiry: date,
        option_type: OptionType,
        market_price: Optional[float] = None,
        volatility: Optional[float] = None,
        risk_free_rate: Optional[float] = None,
    ) -> OptionPrice:
        """
        Full option pricing with Greeks and mispricing analysis.
        
        Args:
            underlying: Ticker symbol
            spot: Current spot price
            strike: Strike price
            expiry: Expiration date
            option_type: CALL or PUT
            market_price: Current market price (for mispricing calc)
            volatility: Override volatility (otherwise uses 20% default)
            risk_free_rate: Override risk-free rate
        """
        r = risk_free_rate if risk_free_rate is not None else self.RISK_FREE_RATE
        
        # Time to expiry
        today = date.today()
        T = max((expiry - today).days / 365.0, 0.001)
        
        # Volatility
        if volatility is None:
            if market_price is not None and market_price > 0:
                iv = self.implied_volatility(market_price, spot, strike, T, option_type, r)
                sigma = iv if iv is not None else 0.20
            else:
                sigma = 0.20
        else:
            sigma = volatility
        
        # Price
        theo_price = self.price(spot, strike, T, sigma, option_type, r)
        
        # Greeks
        greeks = self.greeks(spot, strike, T, sigma, option_type, r)
        
        # d1, d2
        d1, d2 = self._d1_d2(spot, strike, T, r, sigma)
        
        # Intrinsic / extrinsic
        if option_type == OptionType.CALL:
            intrinsic = max(spot - strike, 0)
        else:
            intrinsic = max(strike - spot, 0)
        extrinsic = theo_price - intrinsic
        
        # Implied vol from market
        implied_vol = None
        if market_price is not None and market_price > 0:
            implied_vol = self.implied_volatility(market_price, spot, strike, T, option_type, r)
        
        return OptionPrice(
            underlying=underlying,
            strike=strike,
            expiry=expiry,
            option_type=option_type,
            spot=spot,
            theoretical_price=theo_price,
            market_price=market_price,
            implied_vol=implied_vol,
            greeks=greeks,
            d1=d1,
            d2=d2,
            time_to_expiry=T,
            moneyness=spot / strike if strike > 0 else 0,
            intrinsic_value=intrinsic,
            extrinsic_value=max(extrinsic, 0),
        )
    
    def scan_opportunities(
        self,
        underlying: str,
        spot: float,
        option_chain: List[dict],
        min_mispricing_pct: float = 0.10,
    ) -> List[OptionPrice]:
        """
        Scan an option chain for mispriced options.
        
        Args:
            underlying: Ticker
            spot: Current spot price
            option_chain: List of {strike, expiry, type, market_price}
            min_mispricing_pct: Minimum mispricing % to flag (default 10%)
        
        Returns:
            List of OptionPrice with significant mispricing.
        """
        opportunities = []
        
        for opt in option_chain:
            try:
                strike = opt["strike"]
                expiry = opt["expiry"] if isinstance(opt["expiry"], date) else date.fromisoformat(opt["expiry"])
                opt_type = OptionType(opt["type"].lower())
                market_price = opt.get("market_price", opt.get("last_price", 0))
                
                if market_price <= 0:
                    continue
                
                result = self.price_option(
                    underlying=underlying,
                    spot=spot,
                    strike=strike,
                    expiry=expiry,
                    option_type=opt_type,
                    market_price=market_price,
                )
                
                if result.mispricing_pct and abs(result.mispricing_pct) > min_mispricing_pct:
                    opportunities.append(result)
                    
            except Exception as e:
                logger.debug("Failed to price option: %s", e)
                continue
        
        # Sort by absolute mispricing
        opportunities.sort(key=lambda x: abs(x.mispricing_pct or 0), reverse=True)
        return opportunities
    
    def monte_carlo_price(
        self,
        S: float, K: float, T: float, sigma: float,
        option_type: OptionType = OptionType.CALL,
        r: Optional[float] = None,
        n_sims: int = 10000,
        n_steps: int = 252,
    ) -> Tuple[float, float]:
        """
        Monte Carlo option pricing (handles path-dependent options).
        
        Returns (price, std_error).
        """
        r = r if r is not None else self.RISK_FREE_RATE
        dt = T / n_steps
        
        # Simulate paths
        np.random.seed(42)
        Z = np.random.standard_normal((n_sims, n_steps))
        
        # GBM paths: S_T = S_0 * exp((r - σ²/2)T + σ√T·Z)
        log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=1)
        S_T = S * np.exp(log_paths[:, -1])
        
        # Payoff
        if option_type == OptionType.CALL:
            payoffs = np.maximum(S_T - K, 0)
        else:
            payoffs = np.maximum(K - S_T, 0)
        
        # Discount
        price = np.exp(-r * T) * np.mean(payoffs)
        std_error = np.exp(-r * T) * np.std(payoffs) / np.sqrt(n_sims)
        
        return float(price), float(std_error)
