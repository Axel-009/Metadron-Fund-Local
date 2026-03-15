"""Sector Micro-Bots — 11 specialized GICS sector agents.

Each bot runs within its sector, learning and improving to maximise alpha.
Bots are scored weekly: 40% accuracy + 30% Sharpe + 30% hit rate.

Tiers:
    TIER_1 Generals    — Sharpe >2.0, accuracy >80%
    TIER_2 Captains    — Sharpe >1.5, accuracy >55%
    TIER_3 Lieutenants — Sharpe >1.0, accuracy >50%
    TIER_4 Recruits    — below thresholds
"""

import json
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd

from ..data.universe_engine import GICS_SECTORS, SECTOR_ETFS, Security
from ..data.yahoo_data import get_returns, get_adj_close, get_fundamentals
from ..ml.alpha_optimizer import classify_quality


# ---------------------------------------------------------------------------
# Agent tiers
# ---------------------------------------------------------------------------
class AgentTier:
    GENERAL = "TIER_1_General"
    CAPTAIN = "TIER_2_Captain"
    LIEUTENANT = "TIER_3_Lieutenant"
    RECRUIT = "TIER_4_Recruit"


TIER_THRESHOLDS = {
    AgentTier.GENERAL: {"min_sharpe": 2.0, "min_accuracy": 0.80},
    AgentTier.CAPTAIN: {"min_sharpe": 1.5, "min_accuracy": 0.55},
    AgentTier.LIEUTENANT: {"min_sharpe": 1.0, "min_accuracy": 0.50},
    AgentTier.RECRUIT: {"min_sharpe": -999, "min_accuracy": 0.0},
}


@dataclass
class BotScore:
    accuracy: float = 0.0
    sharpe: float = 0.0
    hit_rate: float = 0.0
    composite: float = 0.0    # 40% acc + 30% sharpe + 30% hit
    tier: str = AgentTier.RECRUIT
    total_signals: int = 0
    correct_signals: int = 0
    consecutive_top_weeks: int = 0
    consecutive_bottom_weeks: int = 0


@dataclass
class SectorSignal:
    ticker: str
    sector: str
    direction: str = "HOLD"   # BUY / SELL / HOLD
    confidence: float = 0.0
    quality_tier: str = "D"
    alpha_estimate: float = 0.0
    momentum: float = 0.0
    reasoning: str = ""


@dataclass
class SectorBot:
    """Micro-bot specialised for a single GICS sector."""
    sector: str
    gics_code: int = 0
    etf: str = ""
    score: BotScore = field(default_factory=BotScore)
    signals: list = field(default_factory=list)
    is_active: bool = True

    def analyze(self, tickers: list[str], lookback_days: int = 252) -> list[SectorSignal]:
        """Analyse all tickers in this sector and generate signals."""
        self.signals = []
        if not tickers:
            return self.signals

        start = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days + 30)).strftime("%Y-%m-%d")

        for ticker in tickers:
            signal = self._analyze_single(ticker, start)
            if signal:
                self.signals.append(signal)

        # Sort by alpha estimate descending
        self.signals.sort(key=lambda s: s.alpha_estimate, reverse=True)
        return self.signals

    def _analyze_single(self, ticker: str, start: str) -> Optional[SectorSignal]:
        """Single-stock analysis within this sector."""
        try:
            rets = get_returns(ticker, start=start)
            if isinstance(rets, pd.DataFrame) and not rets.empty:
                r = rets.iloc[:, 0].dropna()
            elif isinstance(rets, pd.Series):
                r = rets.dropna()
            else:
                return None

            if len(r) < 21:
                return None

            # Momentum signals
            mom_1m = float(r.iloc[-21:].sum())
            mom_3m = float(r.iloc[-63:].sum()) if len(r) >= 63 else mom_1m
            vol = float(r.std() * np.sqrt(252))
            sharpe = (float(r.mean() * 252) / vol) if vol > 0 else 0.0

            # Quality tier
            tier = classify_quality(sharpe, mom_3m)

            # Alpha estimate: momentum + risk-adjusted
            alpha = mom_3m + sharpe * 0.01

            # Direction
            if alpha > 0.03 and tier in ("A", "B", "C"):
                direction = "BUY"
                confidence = min(1.0, alpha * 10)
            elif alpha < -0.03 and tier in ("F", "G"):
                direction = "SELL"
                confidence = min(1.0, abs(alpha) * 10)
            else:
                direction = "HOLD"
                confidence = 0.3

            return SectorSignal(
                ticker=ticker,
                sector=self.sector,
                direction=direction,
                confidence=confidence,
                quality_tier=tier,
                alpha_estimate=alpha,
                momentum=mom_3m,
                reasoning=f"Mom3m={mom_3m:.3f} Sharpe={sharpe:.2f} Vol={vol:.2%} Tier={tier}",
            )

        except Exception:
            return None

    def get_buy_signals(self) -> list[SectorSignal]:
        return [s for s in self.signals if s.direction == "BUY"]

    def get_sell_signals(self) -> list[SectorSignal]:
        return [s for s in self.signals if s.direction == "SELL"]


# ---------------------------------------------------------------------------
# Agent Scorecard
# ---------------------------------------------------------------------------
class AgentScorecard:
    """Track and rank all 11 sector bots + any additional agents.

    Weekly score: 40% accuracy + 30% Sharpe + 30% hit rate.
    Promotion after 4 consecutive top weeks; demotion after 2 bottom weeks.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or Path("logs/agent_scorecard")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.scores: dict[str, BotScore] = {}

    def update_score(
        self,
        agent_name: str,
        accuracy: float,
        sharpe: float,
        hit_rate: float,
        is_top_this_week: bool = False,
        is_bottom_this_week: bool = False,
    ):
        """Update an agent's weekly score."""
        score = self.scores.get(agent_name, BotScore())
        score.accuracy = accuracy
        score.sharpe = sharpe
        score.hit_rate = hit_rate
        score.composite = 0.40 * accuracy + 0.30 * min(sharpe / 3.0, 1.0) + 0.30 * hit_rate

        # Promotion/demotion
        if is_top_this_week:
            score.consecutive_top_weeks += 1
            score.consecutive_bottom_weeks = 0
        elif is_bottom_this_week:
            score.consecutive_bottom_weeks += 1
            score.consecutive_top_weeks = 0
        else:
            score.consecutive_top_weeks = max(0, score.consecutive_top_weeks)
            score.consecutive_bottom_weeks = max(0, score.consecutive_bottom_weeks)

        # Assign tier
        if score.consecutive_top_weeks >= 4 and sharpe > 2.0 and accuracy > 0.80:
            score.tier = AgentTier.GENERAL
        elif sharpe > 1.5 and accuracy > 0.55:
            score.tier = AgentTier.CAPTAIN
        elif sharpe > 1.0 and accuracy > 0.50:
            score.tier = AgentTier.LIEUTENANT
        else:
            score.tier = AgentTier.RECRUIT

        # Demotion override
        if score.consecutive_bottom_weeks >= 2 and score.tier != AgentTier.RECRUIT:
            tier_order = [AgentTier.GENERAL, AgentTier.CAPTAIN, AgentTier.LIEUTENANT, AgentTier.RECRUIT]
            idx = tier_order.index(score.tier)
            if idx < len(tier_order) - 1:
                score.tier = tier_order[idx + 1]

        self.scores[agent_name] = score

    def get_leaderboard(self) -> list[tuple[str, BotScore]]:
        """Sorted leaderboard by composite score."""
        return sorted(
            self.scores.items(),
            key=lambda x: x[1].composite,
            reverse=True,
        )

    def save(self):
        """Persist scorecard to JSON."""
        week = datetime.now().strftime("%Y%W")
        data = {name: asdict(score) for name, score in self.scores.items()}
        (self.log_dir / f"{week}.json").write_text(json.dumps(data, indent=2))
        # Also save leaderboard
        lb = [{"rank": i+1, "agent": name, **asdict(score)}
              for i, (name, score) in enumerate(self.get_leaderboard())]
        (self.log_dir / "leaderboard.json").write_text(json.dumps(lb, indent=2))

    def print_leaderboard(self) -> str:
        """ASCII box-drawing leaderboard."""
        lines = []
        lines.append("┌─────┬────────────────────────────┬──────────┬─────────┬──────────┬───────────┐")
        lines.append("│Rank │ Agent                      │ Score    │ Sharpe  │ Accuracy │ Tier      │")
        lines.append("├─────┼────────────────────────────┼──────────┼─────────┼──────────┼───────────┤")
        for i, (name, score) in enumerate(self.get_leaderboard()):
            lines.append(
                f"│ {i+1:<3} │ {name:<26} │ {score.composite:.4f}  │ {score.sharpe:>6.2f}  │ {score.accuracy:>7.1%}  │ {score.tier:<9} │"
            )
        lines.append("└─────┴────────────────────────────┴──────────┴─────────┴──────────┴───────────┘")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sector Bot Manager
# ---------------------------------------------------------------------------
class SectorBotManager:
    """Manages all 11 GICS sector micro-bots."""

    def __init__(self):
        self.bots: dict[str, SectorBot] = {}
        self.scorecard = AgentScorecard()
        self._init_bots()

    def _init_bots(self):
        """Create one bot per GICS sector."""
        for code, sector in GICS_SECTORS.items():
            etf = SECTOR_ETFS.get(sector, "")
            self.bots[sector] = SectorBot(
                sector=sector,
                gics_code=code,
                etf=etf,
            )

    def run_all(self, universe: dict[str, list] = None) -> dict[str, list[SectorSignal]]:
        """Run all 11 sector bots. Returns sector → signals."""
        results = {}
        for sector, bot in self.bots.items():
            if universe and sector in universe:
                tickers = [s.ticker if hasattr(s, 'ticker') else s for s in universe[sector]]
            else:
                tickers = []
            signals = bot.analyze(tickers)
            results[sector] = signals
        return results

    def get_all_buy_signals(self) -> list[SectorSignal]:
        """Aggregate buy signals across all sectors."""
        buys = []
        for bot in self.bots.values():
            buys.extend(bot.get_buy_signals())
        return sorted(buys, key=lambda s: s.alpha_estimate, reverse=True)

    def get_all_sell_signals(self) -> list[SectorSignal]:
        sells = []
        for bot in self.bots.values():
            sells.extend(bot.get_sell_signals())
        return sells
