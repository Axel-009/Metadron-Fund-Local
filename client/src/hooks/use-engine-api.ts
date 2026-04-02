import { useQuery, UseQueryResult } from "@tanstack/react-query";

const ENGINE_BASE = "/api/engine";

/**
 * Fetch from the engine API bridge.
 * Returns null on error (no mock fallback — errors surface in TECH tab).
 */
async function engineFetch<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${ENGINE_BASE}${path}`);
    if (!res.ok) return null;
    const data = await res.json();
    if (data.error) {
      console.warn(`[Engine API] ${path}: ${data.error}`);
      return null;
    }
    return data as T;
  } catch {
    return null;
  }
}

/**
 * Hook to query the engine API with automatic polling.
 * Returns { data, isLoading, error, refetch }.
 * On failure, data is null — no mock fallback.
 */
export function useEngineQuery<T>(
  path: string,
  options?: {
    refetchInterval?: number | false;
    enabled?: boolean;
  }
): UseQueryResult<T | null> {
  return useQuery<T | null>({
    queryKey: ["engine", path],
    queryFn: () => engineFetch<T>(path),
    refetchInterval: options?.refetchInterval ?? false,
    refetchOnWindowFocus: false,
    staleTime: options?.refetchInterval ? 0 : 30000,
    retry: false,
    enabled: options?.enabled ?? true,
  });
}

// ─── Response Types ────────────────────────────────────────

export interface PortfolioLive {
  nav: number;
  cash: number;
  total_pnl: number;
  gross_exposure: number;
  net_exposure: number;
  positions_count: number;
  win_count: number;
  loss_count: number;
  timestamp: string;
}

export interface PortfolioPosition {
  ticker: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  unrealized_pnl: number;
  realized_pnl: number;
  sector: string;
}

export interface TradeRecord {
  id: string;
  ticker: string;
  side: string;
  quantity: number;
  fill_price: number;
  signal_type: string;
  timestamp: string;
  reason: string;
}

export interface CubeState {
  regime: string;
  liquidity: Record<string, number>;
  risk: Record<string, number>;
  sleeves: Record<string, number>;
  target_beta: number;
  max_leverage: number;
  timestamp: string;
}

export interface MacroSnapshot {
  regime: string;
  vix: number;
  spy_return_1m: number;
  spy_return_3m: number;
  yield_10y: number;
  yield_2y: number;
  yield_spread: number;
  credit_spread: number;
  gold_momentum: number;
  sector_rankings: Record<string, number>;
  gmtf_score: number;
  money_velocity_signal: number;
  cube_regime: string;
  timestamp: string;
}

export interface RiskPortfolio {
  current_beta: number;
  target_beta: number;
  corridor_position: string;
  analytics: Record<string, unknown>;
  timestamp: string;
}

export interface RiskGreeks {
  delta: number;
  gamma: number;
  theta: number;
  vega: number;
  rho: number;
  timestamp: string;
}

export interface AllocationData {
  signals: Array<{
    ticker: string;
    weight: number;
    quality_tier: string;
    alpha_pred: number;
    sharpe_estimate: number;
    momentum_3m: number;
  }>;
  optimal_weights: Record<string, number>;
  expected_return: number;
  annual_volatility: number;
  sharpe_ratio: number;
  timestamp: string;
}

export interface BetaState {
  state: {
    current_beta: number;
    target_beta: number;
    corridor_position: string;
    vol_adjustment: number;
  };
  analytics: Record<string, unknown>;
  timestamp: string;
}

export interface MarketWrapData {
  indices: Array<{
    ticker: string;
    name: string;
    last_price: number;
    change_1d: number;
    change_1w: number;
    change_1m: number;
    change_ytd: number;
  }>;
  sectors: Array<{
    sector: string;
    etf: string;
    return_1d: number;
    return_1w: number;
    return_1m: number;
    relative_strength: number;
  }>;
  top_gainers: Array<{ ticker: string; change: number }>;
  top_losers: Array<{ ticker: string; change: number }>;
  breadth: {
    advancing: number;
    declining: number;
    advance_decline_ratio: number;
    breadth_thrust: number;
  };
  macro: {
    yield_10y: number;
    yield_2y: number;
    yield_spread: number;
    vix: number;
    dxy: number;
    gold: number;
    oil: number;
  };
  market_tone: string;
  timestamp: string;
}

export interface TCAData {
  trades: Array<{
    ticker: string;
    side: string;
    quantity: number;
    fill_price: number;
    slippage: number;
    signal_type: string;
    timestamp: string;
  }>;
  summary: {
    total_trades: number;
    fill_rate: number;
    avg_slippage: number;
    total_cost: number;
  };
  timestamp: string;
}

export interface EngineHealth {
  engines: Array<{
    id: string;
    name: string;
    level: string;
    status: string;
    latency: number;
    errors: number;
    error_msg?: string;
  }>;
  timestamp: string;
}
