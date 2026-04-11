"""API Vault — Metadron Platform

Only 3 authorized external APIs.  All keys loaded from environment.
No other API keys should exist anywhere in the platform.

Authorized APIs:
    1. OpenBB / FMP   — Market data
    2. Alpaca         — Trade execution
    3. Xiaomi Mimo V2 Pro — Brain Power (LLM / intelligence)
"""

import os
import logging
from dataclasses import dataclass

logger = logging.getLogger("metadron.vault")


@dataclass
class APIVault:
    """Immutable container for the 3 authorized API credentials."""

    # OpenBB / FMP — Market Data
    fmp_api_key: str

    # Alpaca — Trade Execution
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str

    # Xiaomi Mimo V2 Pro — Brain Power
    xiaomi_mimo_api_key: str


def load_vault() -> APIVault:
    """Load and validate all authorized API credentials from environment.

    Warns (but does NOT crash) if a key is missing so the platform can
    still start in degraded mode.
    """
    fmp_key = os.environ.get("FMP_API_KEY", "")
    alpaca_key = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    alpaca_url = os.environ.get(
        "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
    )
    xiaomi_key = os.environ.get("XIAOMI_MIMO_API_KEY", "")

    # --- Validation (warn, never crash) ---
    if not fmp_key:
        logger.warning(
            "FMP_API_KEY not set — OpenBB market-data calls will be limited"
        )
    else:
        logger.info("FMP_API_KEY loaded — OpenBB / FMP data source ready")

    if not alpaca_key or not alpaca_secret:
        logger.warning(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — "
            "trade execution unavailable (paper broker will be used)"
        )
    else:
        logger.info(
            "Alpaca credentials loaded — execution via %s", alpaca_url
        )

    if not xiaomi_key:
        logger.warning(
            "XIAOMI_MIMO_API_KEY not set — Brain Power in stub mode "
            "(platform functional, LLM responses are placeholders)"
        )
    else:
        logger.info("XIAOMI_MIMO_API_KEY loaded — Brain Power ready")

    return APIVault(
        fmp_api_key=fmp_key,
        alpaca_api_key=alpaca_key,
        alpaca_secret_key=alpaca_secret,
        alpaca_base_url=alpaca_url,
        xiaomi_mimo_api_key=xiaomi_key,
    )


# ---------------------------------------------------------------------------
# Singleton — import once, use everywhere
# ---------------------------------------------------------------------------
_vault: APIVault | None = None


def get_vault() -> APIVault:
    """Return the singleton vault, creating it on first call."""
    global _vault
    if _vault is None:
        _vault = load_vault()
    return _vault
