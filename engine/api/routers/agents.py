"""
Agents router — AGENTS tab
Wraps: ResearchBotManager, AgentScorecard, SectorBots, InvestorPersonas
"""
from fastapi import APIRouter, Query
from datetime import datetime
import logging

logger = logging.getLogger("metadron-api.agents")
router = APIRouter()

_research = None
_scorecard = None
_sector_bots = None
_personas = None


def _get_research():
    global _research
    if _research is None:
        from engine.agents.research_bots import ResearchBotManager
        _research = ResearchBotManager()
    return _research


def _get_scorecard():
    global _scorecard
    if _scorecard is None:
        from engine.agents.agent_scorecard import AgentScorecard
        _scorecard = AgentScorecard()
    return _scorecard


def _get_sector_bots():
    global _sector_bots
    if _sector_bots is None:
        from engine.agents.sector_bots import SectorBotManager
        _sector_bots = SectorBotManager()
    return _sector_bots


def _get_personas():
    global _personas
    if _personas is None:
        from engine.agents.investor_personas import InvestorPersonaManager
        _personas = InvestorPersonaManager()
    return _personas


@router.get("/research/rankings")
async def agent_rankings():
    """All research bots with rankings, accuracy, Sharpe."""
    try:
        mgr = _get_research()
        bots = []

        if hasattr(mgr, "bots"):
            for name, bot in mgr.bots.items():
                perf = bot.get_performance() if hasattr(bot, "get_performance") else None
                bots.append({
                    "name": name,
                    "sector": bot.sector if hasattr(bot, "sector") else "",
                    "accuracy": perf.accuracy if perf and hasattr(perf, "accuracy") else 0,
                    "sharpe": perf.sharpe if perf and hasattr(perf, "sharpe") else 0,
                    "hit_rate": perf.hit_rate if perf and hasattr(perf, "hit_rate") else 0,
                    "composite_score": perf.composite_score if perf and hasattr(perf, "composite_score") else 0,
                    "rank": perf.rank if perf and hasattr(perf, "rank") else "RECRUIT",
                    "total_signals": perf.total_signals if perf and hasattr(perf, "total_signals") else 0,
                    "total_pnl": perf.total_pnl if perf and hasattr(perf, "total_pnl") else 0,
                })

        return {"agents": bots, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"agents/rankings error: {e}")
        return {"agents": [], "error": str(e)}


@router.get("/scorecard")
async def agent_scorecard():
    """Agent hierarchy scorecard: DIRECTOR, GENERAL, CAPTAIN, etc."""
    try:
        sc = _get_scorecard()
        if hasattr(sc, "get_rankings"):
            rankings = sc.get_rankings()
        elif hasattr(sc, "rankings"):
            rankings = sc.rankings
        else:
            rankings = {}

        return {
            "rankings": rankings if isinstance(rankings, (dict, list)) else str(rankings),
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"agents/scorecard error: {e}")
        return {"rankings": {}, "error": str(e)}


@router.get("/sector-bots")
async def sector_bots():
    """11 GICS sector micro-bots status and signals."""
    try:
        sb = _get_sector_bots()
        bots = []
        if hasattr(sb, "bots"):
            for name, bot in sb.bots.items():
                bots.append({
                    "name": name,
                    "sector": getattr(bot, "sector", ""),
                    "status": getattr(bot, "status", "active"),
                    "last_signal": getattr(bot, "last_signal", None),
                })
        return {"bots": bots, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"agents/sector-bots error: {e}")
        return {"bots": [], "error": str(e)}


@router.get("/personas")
async def investor_personas():
    """12 investor persona agents (Buffett, Munger, etc.)."""
    try:
        pm = _get_personas()
        personas = []
        if hasattr(pm, "personas"):
            for name, p in pm.personas.items():
                personas.append({
                    "name": name,
                    "style": getattr(p, "style", ""),
                    "bias": getattr(p, "bias", ""),
                    "last_opinion": getattr(p, "last_opinion", None),
                })
        elif hasattr(pm, "get_all"):
            all_p = pm.get_all()
            for p in (all_p if isinstance(all_p, list) else []):
                personas.append({
                    "name": getattr(p, "name", ""),
                    "style": getattr(p, "style", ""),
                })
        return {"personas": personas, "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error(f"agents/personas error: {e}")
        return {"personas": [], "error": str(e)}
