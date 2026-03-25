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

# --- agent_skills integration -------------------------------------------------
try:
    from intelligence_platform.agent_skills import (
        create_skill, list_custom_skills, test_skill,
        extract_file_ids, download_file, download_all_files,
    )
    AGENT_SKILLS_AVAILABLE = True
except ImportError:
    AGENT_SKILLS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Agent tiers
# ---------------------------------------------------------------------------
class AgentTier:
    GENERAL = "TIER_1_General"
    CAPTAIN = "TIER_2_Captain"
    LIEUTENANT = "TIER_3_Lieutenant"
    RECRUIT = "TIER_4_Recruit"
