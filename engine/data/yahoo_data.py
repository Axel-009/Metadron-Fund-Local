"""DEPRECATED: This module exists only for backward compatibility.

All imports are forwarded directly to openbb_data.py — the sole data source.
New code should import from engine.data.openbb_data directly.
"""
import logging as _logging

_logger = _logging.getLogger(__name__)
_logger.debug("yahoo_data imported — forwarding to openbb_data (sole data source)")

# Direct re-exports — no transformation, no fallback, no interception.
from .openbb_data import (  # noqa: F401
    get_prices,
    get_adj_close,
    get_returns,
    get_market_stats,
    get_fundamentals,
    get_bulk_fundamentals,
    get_macro_data,
    get_sector_performance,
    MACRO_PROXIES,
    get_fred_series,
    get_treasury_rates,
    get_fed_balance_sheet,
    get_credit_spreads,
    get_monetary_data,
    get_sofr_rate,
    get_effr_rate,
    get_cpi,
    get_unemployment,
    get_macro_data_enriched,
    get_company_filings,
    get_insider_trading,
    get_company_news,
    get_world_news,
    get_options_chains,
    get_etf_holdings,
    get_economic_calendar,
    search_equities,
    get_data_source_status,
    FRED_SERIES,
)
