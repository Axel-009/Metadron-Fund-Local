# MiroFish Agent Simulation — Integration Specification

## Overview

MiroFish has been repurposed from social media simulation to **market microstructure simulation**. Two new engines extend the Metadron Capital platform:

1. **AgentSimEngine** (`engine/signals/agent_sim_engine.py`) — Signal generation
2. **MonteCarloRiskEngine** (`engine/risk/monte_carlo_risk.py`) — Risk analytics

Both use MiroFish's `MarketSimulator` from `intelligence_platform/MiroFish/investment_platform_integration.py`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    OpenBB Data Layer                     │
│              (price, volume, volatility)                 │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
    ┌──────────▼──────────┐ ┌────────▼────────────────┐
    │  AgentSimEngine     │ │ MonteCarloRiskEngine    │
    │                     │ │                         │
    │  Per-ticker:        │ │  Per-ticker:            │
    │  • 100 MC paths     │ │  • 1000 MC paths        │
    │  • Calibrate agents │ │  • VaR 95/99            │
    │  • Simulate N steps │ │  • CVaR 95/99           │
    │  • Extract signal   │ │  • Stress VaR           │
    │                     │ │  • Tail risk score      │
    └──────┬──────────────┘ └──────┬──────────────────┘
           │                       │
           │                       │
    ┌──────▼──────────────┐ ┌─────▼───────────────────┐
    │  L2 Signal Pipeline │ │  Risk Gates (G1-G8)     │
    │                     │ │                         │
    │  AlphaOptimizer ←───┤ │  • VaR position limits  │
    │  DecisionMatrix ←───┤ │  • Stress circuit break │
    │  MLVoteEnsemble ←───┤ │  • Concentration check  │
    └─────────────────────┘ │  • Tail risk throttle   │
                            └─────┬───────────────────┘
                                  │
                           ┌──────▼──────────────┐
                           │  BetaCorridor       │
                           │                     │
                           │  • Vol-adjusted beta│
                           │  • Regime targeting  │
                           │  • Position sizing   │
                           └─────────────────────┘
```

---

## AgentSimEngine — Signal Generation

### Location
`engine/signals/agent_sim_engine.py`

### How It Works

For each ticker in the universe:

1. **Fetch data** from OpenBB (60-day lookback)
2. **Calibrate agents** to real market conditions:
   - High volatility → more noise traders
   - Strong trend → more momentum traders
   - Mean-reverting → more value/fundamental traders
   - Volume → total agent count and market maker participation
3. **Run 100 Monte Carlo simulations** (20 steps ahead each)
4. **Aggregate** into a signal:
   - `direction`: BUY/SELL/HOLD (sign of mean predicted return)
   - `confidence`: fraction of sims agreeing (0-1)
   - `regime`: trending/mean_reverting/random_walk (Hurst exponent)
   - `agent_consensus`: normalized mean return

### Output: `AgentSimSignal`

```python
@dataclass
class AgentSimSignal:
    ticker: str
    direction: str           # "BUY", "SELL", "HOLD"
    confidence: float        # [0, 1]
    predicted_return: float  # Expected return
    regime: str              # "trending", "mean_reverting", "random_walk"
    hurst_exponent: float    # H > 0.5 trending, H < 0.5 mean-reverting
    agent_consensus: float   # [-1, +1]
    price_target: float      # Predicted price
    volatility: float        # Annualized vol from simulations
```

### Pipeline Integration

```
AgentSimEngine.simulate_ticker("AAPL")
    │
    ▼
AgentSimSignal {
    direction: "BUY",
    confidence: 0.72,
    predicted_return: 0.015,
    regime: "trending",
    hurst_exponent: 0.61
}
    │
    ▼
AlphaOptimizer (uses confidence + predicted_return as features)
    │
    ▼
DecisionMatrix (uses regime for strategy selection)
    │
    ▼
MLVoteEnsemble (agent_consensus feeds tier voting)
```

### Usage

```python
from engine.signals.agent_sim_engine import AgentSimEngine

engine = AgentSimEngine(
    n_simulations=100,
    simulation_horizon=20,
    lookback_days=60,
)

# Single ticker
signal = engine.simulate_ticker("AAPL")
print(signal.direction, signal.confidence, signal.regime)

# Full universe
signals = engine.simulate_universe(["AAPL", "MSFT", "GOOGL", "AMZN"])
for ticker, sig in signals.items():
    print(f"{ticker}: {sig.direction} ({sig.confidence:.0%})")
```

---

## MonteCarloRiskEngine — Risk Analytics

### Location
`engine/risk/monte_carlo_risk.py`

### How It Works

For each ticker:

1. **Fetch 1 year** of historical data from OpenBB
2. **Run 1,000 agent-based simulations** (21-day horizon)
3. **Extract risk metrics**:
   - VaR 95% and 99% (dollar amounts)
   - CVaR 95% and 99% (Expected Shortfall)
   - Max drawdown across simulations
   - Tail risk score (excess kurtosis)
   - Stress VaR (under 2x volatility)
   - Sharpe ratio from simulations
   - Hurst exponent for regime

For the portfolio:

1. **Aggregate** per-ticker risks with position weights
2. **Compute diversification benefit** (correlation-adjusted)
3. **Run stress scenarios**:
   - Vol spike (2x)
   - Liquidity drain
   - Correlation spike (all → 1)
   - Flash crash (10%)
   - Black swan (20%)
4. **Risk budget utilization** (current VaR / budget)

### Output: `TickerRisk` and `PortfolioRisk`

```python
@dataclass
class TickerRisk:
    ticker: str
    var_95: float           # 95% VaR ($)
    var_99: float           # 99% VaR ($)
    cvar_95: float          # 95% CVaR ($)
    cvar_99: float          # 99% CVaR ($)
    annualized_vol: float
    max_drawdown_sim: float
    tail_risk_score: float  # [0, 1]
    stress_var_99: float
    sharpe_ratio: float
    hurst_exponent: float
    regime: str

@dataclass
class PortfolioRisk:
    portfolio_value: float
    total_var_95: float
    total_var_99: float
    diversification_benefit: float
    concentration_risk: float
    stress_scenarios: Dict[str, float]
    risk_budget_utilization: float
```

### Pipeline Integration

```
MonteCarloRiskEngine.compute_ticker_risk("AAPL", position_value=50000)
    │
    ▼
TickerRisk {
    var_99: 2340.50,
    cvar_99: 3120.75,
    tail_risk_score: 0.35,
    regime: "trending"
}
    │
    ├──► RiskGateManager
    │    • G1 (position size): if VaR_99 > limit → reject
    │    • G7 (drawdown): if max_drawdown_sim > threshold → circuit break
    │
    ├──► BetaCorridor
    │    • Uses annualized_vol for vol-adjusted beta targeting
    │    • Uses regime for strategy selection
    │
    └──► DecisionMatrix
         • tail_risk_score reduces allocation to high-risk tickers
```

### Usage

```python
from engine.risk.monte_carlo_risk import MonteCarloRiskEngine

risk_engine = MonteCarloRiskEngine(
    n_simulations=1000,
    simulation_horizon=21,
    lookback_days=252,
)

# Single ticker risk
ticker_risk = risk_engine.compute_ticker_risk("AAPL", position_value=50000)
print(f"AAPL VaR 99%: ${ticker_risk.var_99:,.2f}")
print(f"Regime: {ticker_risk.regime}")

# Portfolio risk
positions = {"AAPL": 50000, "MSFT": 30000, "GOOGL": 20000}
portfolio_risk = risk_engine.compute_portfolio_risk(
    positions=positions,
    portfolio_value=100000,
    risk_budget=0.02,
)
print(f"Portfolio VaR 99%: ${portfolio_risk.total_var_99:,.2f}")
print(f"Risk budget utilization: {portfolio_risk.risk_budget_utilization:.0%}")
print(f"Stress scenarios: {portfolio_risk.stress_scenarios}")
```

---

## Integration with ExecutionEngine

### Adding AgentSimEngine to the Pipeline

In `engine/execution/execution_engine.py`, add after the existing signal engines:

```python
try:
    from engine.signals.agent_sim_engine import AgentSimEngine
except ImportError:
    AgentSimEngine = None

# In ExecutionEngine.__init__:
if AgentSimEngine is not None:
    self.agent_sim = AgentSimEngine(
        n_simulations=100,
        simulation_horizon=20,
    )
else:
    self.agent_sim = None
    logger.warning("AgentSimEngine unavailable")

# In run_pipeline(), after existing signal generation:
if self.agent_sim is not None:
    agent_signals = self.agent_sim.simulate_universe(tickers)
    # Feed into alpha optimization as additional feature
    for ticker, signal in agent_signals.items():
        if ticker in alpha_features:
            alpha_features[ticker]["agent_sim_direction"] = 1 if signal.direction == "BUY" else -1 if signal.direction == "SELL" else 0
            alpha_features[ticker]["agent_sim_confidence"] = signal.confidence
            alpha_features[ticker]["agent_sim_regime"] = signal.hurst_exponent
```

### Adding MonteCarloRiskEngine to Risk Gates

In `engine/execution/execution_engine.py` RiskGateManager:

```python
try:
    from engine.risk.monte_carlo_risk import MonteCarloRiskEngine
except ImportError:
    MonteCarloRiskEngine = None

# In RiskGateManager.__init__:
if MonteCarloRiskEngine is not None:
    self.mc_risk = MonteCarloRiskEngine(n_simulations=1000)
else:
    self.mc_risk = None

# In gate checks (G1 - Position Size):
if self.mc_risk is not None:
    ticker_risk = self.mc_risk.compute_ticker_risk(ticker, proposed_position_value)
    if ticker_risk.var_99 > max_position_var:
        return False, f"G1: VaR ${ticker_risk.var_99:.2f} exceeds limit ${max_position_var}"
```

---

## Configuration

### Defaults

| Parameter | AgentSimEngine | MonteCarloRiskEngine |
|---|---|---|
| n_simulations | 100 | 1,000 |
| simulation_horizon | 20 steps | 21 days |
| lookback_days | 60 | 252 |
| confidence_levels | — | [0.95, 0.99] |

### Tuning

- **More simulations** = more accurate but slower
- **Longer horizon** = more uncertainty, wider VaR
- **Longer lookback** = more stable calibration, slower adaptation
- For HFT: reduce n_simulations to 50, horizon to 5
- For portfolio risk: keep n_simulations at 1000+

---

## Performance

Expected runtime per ticker:
- AgentSimEngine: ~2-5 seconds (100 sims × 20 steps)
- MonteCarloRiskEngine: ~10-30 seconds (1000 sims × 21 steps)

For a 50-ticker universe:
- AgentSimEngine: ~2-4 minutes
- MonteCarloRiskEngine: ~8-25 minutes

**Optimization opportunities:**
- Parallelize simulations (multiprocessing)
- Cache calibrated parameters between runs
- Reduce simulation count during off-hours
- Use vectorized numpy operations (already partially done)

---

## What This Does NOT Cover

- ❌ Real-time intraday simulation (batch only for now)
- ❌ Cross-asset contagion (single-ticker only)
- ❌ Agent learning/adaptation (static populations)
- ❌ Order book reconstruction (synthetic only)

These are future enhancements once the basic pipeline is running.

---

*Spec by Bobby. Build it, test it, deploy it.*
