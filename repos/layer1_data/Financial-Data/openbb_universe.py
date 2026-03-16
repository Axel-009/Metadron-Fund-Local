# ============================================================
# SOURCE: https://github.com/Axel-009/Financial-Data
# LAYER:  layer1_data
# ROLE:   OpenBB universe data bridge for financial data collection
# ============================================================
"""
OpenBB Universe Data Module
============================

The SOLE canonical data source for the entire Metadron Capital investment
platform.  Every asset class — equities, fixed income, commodities, crypto,
FX, ETFs, indices, macro — is accessed exclusively through the OpenBB
Platform SDK.  No yfinance fallback.  No alternative providers.

This module is the single source of truth and is consumed by:
  - Financial-Data (this repo)
  - QLIB (qlib/data/openbb_universe.py — adapted copy)
  - All downstream strategy, risk, and execution systems

Usage
-----
    from openbb_universe import (
        AssetClass,
        get_full_universe,
        get_equity_universe,
        get_historical,
        get_macro_data,
        classify_by_gics,
    )

    universe = get_full_universe()
    equities = get_equity_universe()
    hist = get_historical("AAPL", "2020-01-01", "2024-12-31")

Mathematical Formulas (for ML model docstring training)
-------------------------------------------------------
Sharpe Ratio:
    S = (R_p - R_f) / sigma_p
    where R_p = portfolio return, R_f = risk-free rate, sigma_p = std dev of
    portfolio excess returns.

Sortino Ratio:
    S = (R_p - R_f) / sigma_d
    where sigma_d = sqrt(E[min(R - R_f, 0)^2]) is the downside deviation.

Kelly Criterion:
    f* = (b * p - q) / b
    where b = decimal odds (net profit per $1 wagered), p = probability of
    winning, q = 1 - p.

Black-Scholes European Call:
    C = S * N(d1) - K * e^(-rT) * N(d2)
    d1 = (ln(S/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    where S = spot price, K = strike, r = risk-free rate, T = time to
    expiry (years), sigma = implied volatility, N() = standard normal CDF.

Value at Risk (parametric):
    VaR_alpha = mu - z_alpha * sigma
    where mu = expected return, z_alpha = normal quantile at confidence
    level alpha, sigma = portfolio std dev.

Covariance:
    Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)]
    = E[XY] - mu_X * mu_Y

Capital Asset Pricing Model (CAPM):
    E[R_i] = R_f + beta_i * (E[R_m] - R_f)
    beta_i = Cov(R_i, R_m) / Var(R_m)

Information Ratio:
    IR = (R_p - R_b) / sigma_(p-b)
    where R_b = benchmark return, sigma_(p-b) = tracking error.

Maximum Drawdown:
    MDD = max_{t in [0,T]} ( max_{s in [0,t]} V_s - V_t ) / max_{s in [0,t]} V_s

Herfindahl-Hirschman Index (concentration):
    HHI = sum_{i=1}^{N} w_i^2
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OpenBB SDK accessor (lazy import)
# ---------------------------------------------------------------------------

_obb = None


def _get_obb():
    """Lazy-import the OpenBB SDK so the module can be imported without it."""
    global _obb
    if _obb is None:
        try:
            from openbb import obb
            _obb = obb
        except ImportError as exc:
            raise ImportError(
                "OpenBB Platform is required.  Install with:  pip install openbb"
            ) from exc
    return _obb


# ═══════════════════════════════════════════════════════════════════════════
# Asset-class taxonomy
# ═══════════════════════════════════════════════════════════════════════════

class AssetClass(Enum):
    """Enumeration of all supported asset classes."""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITY = "commodity"
    CRYPTO = "crypto"
    FX = "fx"
    ETF = "etf"
    INDEX = "index"


# ═══════════════════════════════════════════════════════════════════════════
# GICS Classification — complete hierarchy
# ═══════════════════════════════════════════════════════════════════════════

GICS_SECTORS: Dict[int, dict] = {
    10: {
        "name": "Energy",
        "groups": {
            1010: {
                "name": "Energy",
                "industries": {
                    101010: {
                        "name": "Energy Equipment & Services",
                        "sub_industries": {
                            10101010: "Oil & Gas Drilling",
                            10101020: "Oil & Gas Equipment & Services",
                        },
                    },
                    101020: {
                        "name": "Oil, Gas & Consumable Fuels",
                        "sub_industries": {
                            10102010: "Integrated Oil & Gas",
                            10102020: "Oil & Gas Exploration & Production",
                            10102030: "Oil & Gas Refining & Marketing",
                            10102040: "Oil & Gas Storage & Transportation",
                            10102050: "Coal & Consumable Fuels",
                        },
                    },
                },
            },
        },
    },
    15: {
        "name": "Materials",
        "groups": {
            1510: {
                "name": "Materials",
                "industries": {
                    151010: {
                        "name": "Chemicals",
                        "sub_industries": {
                            15101010: "Commodity Chemicals",
                            15101020: "Diversified Chemicals",
                            15101030: "Fertilizers & Agricultural Chemicals",
                            15101040: "Industrial Gases",
                            15101050: "Specialty Chemicals",
                        },
                    },
                    151020: {
                        "name": "Construction Materials",
                        "sub_industries": {
                            15102010: "Construction Materials",
                        },
                    },
                    151030: {
                        "name": "Containers & Packaging",
                        "sub_industries": {
                            15103010: "Metal & Glass Containers",
                            15103020: "Paper Packaging",
                        },
                    },
                    151040: {
                        "name": "Metals & Mining",
                        "sub_industries": {
                            15104010: "Aluminum",
                            15104020: "Diversified Metals & Mining",
                            15104025: "Copper",
                            15104030: "Gold",
                            15104040: "Precious Metals & Minerals",
                            15104045: "Silver",
                            15104050: "Steel",
                        },
                    },
                    151050: {
                        "name": "Paper & Forest Products",
                        "sub_industries": {
                            15105010: "Forest Products",
                            15105020: "Paper Products",
                        },
                    },
                },
            },
        },
    },
    20: {
        "name": "Industrials",
        "groups": {
            2010: {
                "name": "Capital Goods",
                "industries": {
                    201010: {
                        "name": "Aerospace & Defense",
                        "sub_industries": {
                            20101010: "Aerospace & Defense",
                        },
                    },
                    201020: {
                        "name": "Building Products",
                        "sub_industries": {
                            20102010: "Building Products",
                        },
                    },
                    201030: {
                        "name": "Construction & Engineering",
                        "sub_industries": {
                            20103010: "Construction & Engineering",
                        },
                    },
                    201040: {
                        "name": "Electrical Equipment",
                        "sub_industries": {
                            20104010: "Electrical Components & Equipment",
                            20104020: "Heavy Electrical Equipment",
                        },
                    },
                    201050: {
                        "name": "Industrial Conglomerates",
                        "sub_industries": {
                            20105010: "Industrial Conglomerates",
                        },
                    },
                    201060: {
                        "name": "Machinery",
                        "sub_industries": {
                            20106010: "Construction Machinery & Heavy Trucks",
                            20106015: "Agricultural & Farm Machinery",
                            20106020: "Industrial Machinery",
                        },
                    },
                    201070: {
                        "name": "Trading Companies & Distributors",
                        "sub_industries": {
                            20107010: "Trading Companies & Distributors",
                        },
                    },
                },
            },
            2020: {
                "name": "Commercial & Professional Services",
                "industries": {
                    202010: {
                        "name": "Commercial Services & Supplies",
                        "sub_industries": {
                            20201010: "Commercial Printing",
                            20201050: "Environmental & Facilities Services",
                            20201060: "Office Services & Supplies",
                            20201070: "Diversified Support Services",
                            20201080: "Security & Alarm Services",
                        },
                    },
                    202020: {
                        "name": "Professional Services",
                        "sub_industries": {
                            20202010: "Human Resource & Employment Services",
                            20202020: "Research & Consulting Services",
                        },
                    },
                },
            },
            2030: {
                "name": "Transportation",
                "industries": {
                    203010: {
                        "name": "Air Freight & Logistics",
                        "sub_industries": {
                            20301010: "Air Freight & Logistics",
                        },
                    },
                    203020: {
                        "name": "Airlines",
                        "sub_industries": {
                            20302010: "Airlines",
                        },
                    },
                    203030: {
                        "name": "Marine",
                        "sub_industries": {
                            20303010: "Marine",
                        },
                    },
                    203040: {
                        "name": "Road & Rail",
                        "sub_industries": {
                            20304010: "Railroads",
                            20304020: "Trucking",
                        },
                    },
                    203050: {
                        "name": "Transportation Infrastructure",
                        "sub_industries": {
                            20305010: "Airport Services",
                            20305020: "Highways & Railtracks",
                            20305030: "Marine Ports & Services",
                        },
                    },
                },
            },
        },
    },
    25: {
        "name": "Consumer Discretionary",
        "groups": {
            2510: {
                "name": "Automobiles & Components",
                "industries": {
                    251010: {
                        "name": "Auto Components",
                        "sub_industries": {
                            25101010: "Auto Parts & Equipment",
                            25101020: "Tires & Rubber",
                        },
                    },
                    251020: {
                        "name": "Automobiles",
                        "sub_industries": {
                            25102010: "Automobile Manufacturers",
                            25102020: "Motorcycle Manufacturers",
                        },
                    },
                },
            },
            2520: {
                "name": "Consumer Durables & Apparel",
                "industries": {
                    252010: {
                        "name": "Household Durables",
                        "sub_industries": {
                            25201010: "Consumer Electronics",
                            25201020: "Home Furnishings",
                            25201030: "Homebuilding",
                            25201040: "Household Appliances",
                            25201050: "Housewares & Specialties",
                        },
                    },
                    252020: {
                        "name": "Leisure Products",
                        "sub_industries": {
                            25202010: "Leisure Products",
                        },
                    },
                    252030: {
                        "name": "Textiles, Apparel & Luxury Goods",
                        "sub_industries": {
                            25203010: "Apparel, Accessories & Luxury Goods",
                            25203020: "Footwear",
                            25203030: "Textiles",
                        },
                    },
                },
            },
            2530: {
                "name": "Consumer Services",
                "industries": {
                    253010: {
                        "name": "Hotels, Restaurants & Leisure",
                        "sub_industries": {
                            25301010: "Casinos & Gaming",
                            25301020: "Hotels, Resorts & Cruise Lines",
                            25301030: "Leisure Facilities",
                            25301040: "Restaurants",
                        },
                    },
                    253020: {
                        "name": "Diversified Consumer Services",
                        "sub_industries": {
                            25302010: "Education Services",
                            25302020: "Specialized Consumer Services",
                        },
                    },
                },
            },
            2550: {
                "name": "Retailing",
                "industries": {
                    255010: {
                        "name": "Distributors",
                        "sub_industries": {
                            25501010: "Distributors",
                        },
                    },
                    255020: {
                        "name": "Internet & Direct Marketing Retail",
                        "sub_industries": {
                            25502020: "Internet & Direct Marketing Retail",
                        },
                    },
                    255030: {
                        "name": "Multiline Retail",
                        "sub_industries": {
                            25503010: "Department Stores",
                            25503020: "General Merchandise Stores",
                        },
                    },
                    255040: {
                        "name": "Specialty Retail",
                        "sub_industries": {
                            25504010: "Apparel Retail",
                            25504020: "Computer & Electronics Retail",
                            25504030: "Home Improvement Retail",
                            25504040: "Specialty Stores",
                            25504050: "Automotive Retail",
                            25504060: "Homefurnishing Retail",
                        },
                    },
                },
            },
        },
    },
    30: {
        "name": "Consumer Staples",
        "groups": {
            3010: {
                "name": "Food & Staples Retailing",
                "industries": {
                    301010: {
                        "name": "Food & Staples Retailing",
                        "sub_industries": {
                            30101010: "Drug Retail",
                            30101020: "Food Distributors",
                            30101030: "Food Retail",
                            30101040: "Hypermarkets & Super Centers",
                        },
                    },
                },
            },
            3020: {
                "name": "Food, Beverage & Tobacco",
                "industries": {
                    302010: {
                        "name": "Beverages",
                        "sub_industries": {
                            30201010: "Brewers",
                            30201020: "Distillers & Vintners",
                            30201030: "Soft Drinks",
                        },
                    },
                    302020: {
                        "name": "Food Products",
                        "sub_industries": {
                            30202010: "Agricultural Products",
                            30202030: "Packaged Foods & Meats",
                        },
                    },
                    302030: {
                        "name": "Tobacco",
                        "sub_industries": {
                            30203010: "Tobacco",
                        },
                    },
                },
            },
            3030: {
                "name": "Household & Personal Products",
                "industries": {
                    303010: {
                        "name": "Household Products",
                        "sub_industries": {
                            30301010: "Household Products",
                        },
                    },
                    303020: {
                        "name": "Personal Products",
                        "sub_industries": {
                            30302010: "Personal Products",
                        },
                    },
                },
            },
        },
    },
    35: {
        "name": "Health Care",
        "groups": {
            3510: {
                "name": "Health Care Equipment & Services",
                "industries": {
                    351010: {
                        "name": "Health Care Equipment & Supplies",
                        "sub_industries": {
                            35101010: "Health Care Equipment",
                            35101020: "Health Care Supplies",
                        },
                    },
                    351020: {
                        "name": "Health Care Providers & Services",
                        "sub_industries": {
                            35102010: "Health Care Distributors",
                            35102015: "Health Care Services",
                            35102020: "Health Care Facilities",
                            35102030: "Managed Health Care",
                        },
                    },
                    351030: {
                        "name": "Health Care Technology",
                        "sub_industries": {
                            35103010: "Health Care Technology",
                        },
                    },
                },
            },
            3520: {
                "name": "Pharmaceuticals, Biotechnology & Life Sciences",
                "industries": {
                    352010: {
                        "name": "Biotechnology",
                        "sub_industries": {
                            35201010: "Biotechnology",
                        },
                    },
                    352020: {
                        "name": "Pharmaceuticals",
                        "sub_industries": {
                            35202010: "Pharmaceuticals",
                        },
                    },
                    352030: {
                        "name": "Life Sciences Tools & Services",
                        "sub_industries": {
                            35203010: "Life Sciences Tools & Services",
                        },
                    },
                },
            },
        },
    },
    40: {
        "name": "Financials",
        "groups": {
            4010: {
                "name": "Banks",
                "industries": {
                    401010: {
                        "name": "Banks",
                        "sub_industries": {
                            40101010: "Diversified Banks",
                            40101015: "Regional Banks",
                        },
                    },
                    401020: {
                        "name": "Thrifts & Mortgage Finance",
                        "sub_industries": {
                            40102010: "Thrifts & Mortgage Finance",
                        },
                    },
                },
            },
            4020: {
                "name": "Diversified Financials",
                "industries": {
                    402010: {
                        "name": "Diversified Financial Services",
                        "sub_industries": {
                            40201020: "Other Diversified Financial Services",
                            40201030: "Multi-Sector Holdings",
                            40201040: "Specialized Finance",
                        },
                    },
                    402020: {
                        "name": "Consumer Finance",
                        "sub_industries": {
                            40202010: "Consumer Finance",
                        },
                    },
                    402030: {
                        "name": "Capital Markets",
                        "sub_industries": {
                            40203010: "Asset Management & Custody Banks",
                            40203020: "Investment Banking & Brokerage",
                            40203030: "Diversified Capital Markets",
                            40203040: "Financial Exchanges & Data",
                        },
                    },
                    402040: {
                        "name": "Mortgage Real Estate Investment Trusts (REITs)",
                        "sub_industries": {
                            40204010: "Mortgage REITs",
                        },
                    },
                },
            },
            4030: {
                "name": "Insurance",
                "industries": {
                    403010: {
                        "name": "Insurance",
                        "sub_industries": {
                            40301010: "Insurance Brokers",
                            40301020: "Life & Health Insurance",
                            40301030: "Multi-line Insurance",
                            40301040: "Property & Casualty Insurance",
                            40301050: "Reinsurance",
                        },
                    },
                },
            },
        },
    },
    45: {
        "name": "Information Technology",
        "groups": {
            4510: {
                "name": "Software & Services",
                "industries": {
                    451010: {
                        "name": "Internet Software & Services",
                        "sub_industries": {
                            45101010: "Internet Software & Services",
                        },
                    },
                    451020: {
                        "name": "IT Services",
                        "sub_industries": {
                            45102010: "IT Consulting & Other Services",
                            45102020: "Data Processing & Outsourced Services",
                        },
                    },
                    451030: {
                        "name": "Software",
                        "sub_industries": {
                            45103010: "Application Software",
                            45103020: "Systems Software",
                        },
                    },
                },
            },
            4520: {
                "name": "Technology Hardware & Equipment",
                "industries": {
                    452010: {
                        "name": "Communications Equipment",
                        "sub_industries": {
                            45201020: "Communications Equipment",
                        },
                    },
                    452020: {
                        "name": "Technology Hardware, Storage & Peripherals",
                        "sub_industries": {
                            45202030: "Technology Hardware, Storage & Peripherals",
                        },
                    },
                    452030: {
                        "name": "Electronic Equipment, Instruments & Components",
                        "sub_industries": {
                            45203010: "Electronic Equipment & Instruments",
                            45203015: "Electronic Components",
                            45203020: "Electronic Manufacturing Services",
                            45203030: "Technology Distributors",
                        },
                    },
                },
            },
            4530: {
                "name": "Semiconductors & Semiconductor Equipment",
                "industries": {
                    453010: {
                        "name": "Semiconductors & Semiconductor Equipment",
                        "sub_industries": {
                            45301010: "Semiconductor Equipment",
                            45301020: "Semiconductors",
                        },
                    },
                },
            },
        },
    },
    50: {
        "name": "Communication Services",
        "groups": {
            5010: {
                "name": "Telecommunication Services",
                "industries": {
                    501010: {
                        "name": "Diversified Telecommunication Services",
                        "sub_industries": {
                            50101010: "Alternative Carriers",
                            50101020: "Integrated Telecommunication Services",
                        },
                    },
                    501020: {
                        "name": "Wireless Telecommunication Services",
                        "sub_industries": {
                            50102010: "Wireless Telecommunication Services",
                        },
                    },
                },
            },
            5020: {
                "name": "Media & Entertainment",
                "industries": {
                    502010: {
                        "name": "Media",
                        "sub_industries": {
                            50201010: "Advertising",
                            50201020: "Broadcasting",
                            50201030: "Cable & Satellite",
                            50201040: "Publishing",
                        },
                    },
                    502020: {
                        "name": "Entertainment",
                        "sub_industries": {
                            50202010: "Movies & Entertainment",
                            50202020: "Interactive Home Entertainment",
                        },
                    },
                    502030: {
                        "name": "Interactive Media & Services",
                        "sub_industries": {
                            50203010: "Interactive Media & Services",
                        },
                    },
                },
            },
        },
    },
    55: {
        "name": "Utilities",
        "groups": {
            5510: {
                "name": "Utilities",
                "industries": {
                    551010: {
                        "name": "Electric Utilities",
                        "sub_industries": {
                            55101010: "Electric Utilities",
                        },
                    },
                    551020: {
                        "name": "Gas Utilities",
                        "sub_industries": {
                            55102010: "Gas Utilities",
                        },
                    },
                    551030: {
                        "name": "Multi-Utilities",
                        "sub_industries": {
                            55103010: "Multi-Utilities",
                        },
                    },
                    551040: {
                        "name": "Water Utilities",
                        "sub_industries": {
                            55104010: "Water Utilities",
                        },
                    },
                    551050: {
                        "name": "Independent Power and Renewable Electricity Producers",
                        "sub_industries": {
                            55105010: "Independent Power Producers & Energy Traders",
                            55105020: "Renewable Electricity",
                        },
                    },
                },
            },
        },
    },
    60: {
        "name": "Real Estate",
        "groups": {
            6010: {
                "name": "Real Estate",
                "industries": {
                    601010: {
                        "name": "Equity Real Estate Investment Trusts (REITs)",
                        "sub_industries": {
                            60101010: "Diversified REITs",
                            60101020: "Industrial REITs",
                            60101030: "Hotel & Resort REITs",
                            60101040: "Office REITs",
                            60101050: "Health Care REITs",
                            60101060: "Residential REITs",
                            60101070: "Retail REITs",
                            60101080: "Specialized REITs",
                        },
                    },
                    601020: {
                        "name": "Real Estate Management & Development",
                        "sub_industries": {
                            60102010: "Diversified Real Estate Activities",
                            60102020: "Real Estate Operating Companies",
                            60102030: "Real Estate Development",
                            60102040: "Real Estate Services",
                        },
                    },
                },
            },
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# Universe constituent lists
# ═══════════════════════════════════════════════════════════════════════════

# S&P 500 representative tickers (top ~60 by market cap + sector coverage)
SP500_REPRESENTATIVE: List[str] = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "BRK.B",
    "TSLA", "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "AVGO",
    "CVX", "MRK", "LLY", "ABBV", "COST", "PEP", "KO", "ADBE", "WMT",
    "MCD", "CRM", "CSCO", "TMO", "ACN", "ABT", "LIN", "NKE", "DHR",
    "ORCL", "TXN", "CMCSA", "PM", "NEE", "UNP", "RTX", "LOW", "INTC",
    "HON", "AMD", "AMGN", "UPS", "IBM", "GE", "CAT", "BA", "GS",
    "SBUX", "BLK", "ISRG", "MDLZ", "PLD", "DE",
]

# Bond ETFs
BOND_ETFS: List[str] = [
    "TLT", "IEF", "SHY", "LQD", "HYG", "AGG", "BND", "BNDX", "TIP",
    "VCSH", "VCIT", "VCLT", "MUB", "EMB", "JNK", "GOVT", "SCHO",
    "SCHR", "SCHZ", "IGIB", "IGSB", "USIG", "FLOT", "MINT",
]

# Commodity futures and ETFs
COMMODITY_SYMBOLS: List[str] = [
    # Futures-style symbols
    "GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "PL=F", "PA=F",
    "ZC=F", "ZW=F", "ZS=F", "KC=F", "CC=F", "SB=F", "CT=F",
    "LE=F", "HE=F",
    # ETFs tracking commodities
    "GLD", "SLV", "USO", "UNG", "DBA", "DBC", "PDBC", "GSG", "CPER",
    "PPLT", "PALL", "WEAT", "CORN", "SOYB", "JO", "NIB", "BAL",
    "IAU", "SGOL", "AAAU",
]

# Top crypto assets
CRYPTO_SYMBOLS: List[str] = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "ADA-USD", "SOL-USD",
    "DOGE-USD", "DOT-USD", "AVAX-USD", "MATIC-USD", "LINK-USD",
    "UNI-USD", "ATOM-USD", "LTC-USD", "ETC-USD", "XLM-USD",
    "ALGO-USD", "NEAR-USD", "FTM-USD", "AAVE-USD",
]

# Major FX pairs
FX_MAJOR: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X",
]
FX_MINOR: List[str] = [
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDNZD=X", "EURCHF=X",
    "AUDCAD=X", "CADJPY=X", "NZDJPY=X", "GBPCHF=X", "EURAUD=X",
    "AUDCHF=X", "GBPAUD=X",
]
FX_EXOTIC: List[str] = [
    "USDMXN=X", "USDZAR=X", "USDTRY=X", "USDBRL=X", "USDPLN=X",
    "USDSEK=X", "USDNOK=X", "USDDKK=X", "USDSGD=X", "USDHKD=X",
    "USDCNY=X", "USDINR=X", "USDTHB=X", "USDKRW=X", "USDTWD=X",
]
FX_ALL: List[str] = FX_MAJOR + FX_MINOR + FX_EXOTIC

# Global indices
INDEX_SYMBOLS: List[str] = [
    "^GSPC", "^DJI", "^IXIC", "^RUT",          # US
    "^FTSE", "^GDAXI", "^FCHI", "^STOXX50E",    # Europe
    "^N225", "^HSI", "000001.SS", "^KS11",       # Asia
    "^BSESN", "^NSEI",                            # India
    "^AXJO", "^NZ50",                              # Oceania
    "^BVSP", "^MXX",                               # Latin America
    "^GSPTSE",                                     # Canada
    "^VIX",                                         # Volatility
]

# Major ETFs across asset classes
ETF_CORE: List[str] = [
    # Broad equity
    "SPY", "IVV", "VOO", "VTI", "QQQ", "DIA", "IWM", "VEA", "VWO", "EFA",
    "EEM", "IEMG",
    # Sector
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB",
    "XLRE", "XLC",
    # Thematic
    "ARKK", "ARKG", "ARKW", "ARKF", "ARKQ",
    # Factor
    "MTUM", "QUAL", "VLUE", "SIZE", "USMV",
    # International
    "FXI", "EWJ", "EWG", "EWU", "EWZ", "INDA",
]


# ═══════════════════════════════════════════════════════════════════════════
# GICS helper — flattened lookup tables
# ═══════════════════════════════════════════════════════════════════════════

def _build_gics_flat() -> dict:
    """Build flat lookup dicts from the nested GICS_SECTORS hierarchy."""
    sector_map: Dict[int, str] = {}
    group_map: Dict[int, str] = {}
    industry_map: Dict[int, str] = {}
    sub_industry_map: Dict[int, str] = {}

    for sector_code, sector_info in GICS_SECTORS.items():
        sector_map[sector_code] = sector_info["name"]
        for group_code, group_info in sector_info["groups"].items():
            group_map[group_code] = group_info["name"]
            for industry_code, industry_info in group_info["industries"].items():
                industry_map[industry_code] = industry_info["name"]
                for sub_code, sub_name in industry_info.get("sub_industries", {}).items():
                    sub_industry_map[sub_code] = sub_name

    return {
        "sector": sector_map,
        "group": group_map,
        "industry": industry_map,
        "sub_industry": sub_industry_map,
    }


_GICS_FLAT = _build_gics_flat()


def gics_sector_name(code: int) -> Optional[str]:
    """Return the GICS sector name for a sector code (10, 15, ..., 60)."""
    return _GICS_FLAT["sector"].get(code)


def gics_industry_group_name(code: int) -> Optional[str]:
    """Return the GICS industry group name for a group code."""
    return _GICS_FLAT["group"].get(code)


def gics_industry_name(code: int) -> Optional[str]:
    """Return the GICS industry name for an industry code."""
    return _GICS_FLAT["industry"].get(code)


def gics_sub_industry_name(code: int) -> Optional[str]:
    """Return the GICS sub-industry name for a sub-industry code."""
    return _GICS_FLAT["sub_industry"].get(code)


# ═══════════════════════════════════════════════════════════════════════════
# Asset-class detection
# ═══════════════════════════════════════════════════════════════════════════

def detect_asset_class(symbol: str) -> AssetClass:
    """Auto-detect the asset class of a symbol based on its format.

    Detection rules (applied in order):
      - Contains "=X" suffix           -> FX
      - Contains "-USD" or "-BTC"      -> CRYPTO
      - Starts with "^"               -> INDEX
      - Contains "=F" suffix           -> COMMODITY (futures)
      - Is in COMMODITY_SYMBOLS        -> COMMODITY
      - Is in BOND_ETFS               -> FIXED_INCOME
      - Is in ETF_CORE                -> ETF
      - Otherwise                     -> EQUITY

    Parameters
    ----------
    symbol : str
        Ticker or symbol string.

    Returns
    -------
    AssetClass
        Detected asset class enum member.
    """
    s = symbol.upper().strip()

    if s.endswith("=X"):
        return AssetClass.FX
    if re.search(r"-USD$|-BTC$|-ETH$|-USDT$", s):
        return AssetClass.CRYPTO
    if s.startswith("^"):
        return AssetClass.INDEX
    if s.endswith("=F"):
        return AssetClass.COMMODITY
    if s in {c.upper() for c in COMMODITY_SYMBOLS}:
        return AssetClass.COMMODITY
    if s in {b.upper() for b in BOND_ETFS}:
        return AssetClass.FIXED_INCOME
    if s in {e.upper() for e in ETF_CORE}:
        return AssetClass.ETF
    return AssetClass.EQUITY


# ═══════════════════════════════════════════════════════════════════════════
# Core data functions — OpenBB as sole data source
# ═══════════════════════════════════════════════════════════════════════════

def _obb_result_to_df(result) -> pd.DataFrame:
    """Convert an OpenBB OBBject result to a pandas DataFrame."""
    if hasattr(result, "to_df"):
        return result.to_df()
    if hasattr(result, "results") and result.results is not None:
        return pd.DataFrame([r.model_dump() for r in result.results])
    return pd.DataFrame()


def get_full_universe(
    asset_classes: Optional[Sequence[AssetClass]] = None,
) -> Dict[AssetClass, List[str]]:
    """Return all tradeable symbols grouped by asset class.

    If *asset_classes* is ``None`` every class is included.

    Parameters
    ----------
    asset_classes : sequence of AssetClass, optional
        Subset of classes to return.  ``None`` means all.

    Returns
    -------
    dict[AssetClass, list[str]]
        Mapping from asset class to constituent symbol lists.
    """
    full: Dict[AssetClass, List[str]] = {
        AssetClass.EQUITY: list(SP500_REPRESENTATIVE),
        AssetClass.FIXED_INCOME: list(BOND_ETFS),
        AssetClass.COMMODITY: list(COMMODITY_SYMBOLS),
        AssetClass.CRYPTO: list(CRYPTO_SYMBOLS),
        AssetClass.FX: list(FX_ALL),
        AssetClass.ETF: list(ETF_CORE),
        AssetClass.INDEX: list(INDEX_SYMBOLS),
    }

    # Attempt to augment the equity universe from OpenBB screener
    try:
        obb = _get_obb()
        screener_result = obb.equity.screener(provider="fmp")
        df = _obb_result_to_df(screener_result)
        if not df.empty and "symbol" in df.columns:
            existing = set(full[AssetClass.EQUITY])
            for sym in df["symbol"].dropna().unique():
                if sym not in existing:
                    full[AssetClass.EQUITY].append(sym)
                    existing.add(sym)
    except Exception as exc:
        logger.warning("Could not augment equity universe from screener: %s", exc)

    if asset_classes is not None:
        requested = set(asset_classes)
        return {k: v for k, v in full.items() if k in requested}
    return full


def get_equity_universe() -> pd.DataFrame:
    """Return all equities with GICS classification columns.

    Fetches profile data from ``obb.equity.profile()`` for each symbol in
    the representative universe and maps sector/industry fields to the GICS
    hierarchy.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, name, sector, industry_group, industry,
        sub_industry, market_cap, exchange, country.
    """
    obb = _get_obb()
    records: List[dict] = []

    # Batch in groups to avoid rate limits
    batch_size = 20
    symbols = list(SP500_REPRESENTATIVE)
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            result = obb.equity.profile(symbol=",".join(batch), provider="fmp")
            df = _obb_result_to_df(result)
            if not df.empty:
                for _, row in df.iterrows():
                    record = {
                        "symbol": row.get("symbol", ""),
                        "name": row.get("name", row.get("company_name", "")),
                        "sector": row.get("sector", ""),
                        "industry_group": "",
                        "industry": row.get("industry", ""),
                        "sub_industry": "",
                        "market_cap": row.get("market_cap", row.get("mktCap", np.nan)),
                        "exchange": row.get("exchange", row.get("exchange_short_name", "")),
                        "country": row.get("country", ""),
                    }
                    # Attempt to map to GICS from sector name
                    for code, info in GICS_SECTORS.items():
                        if info["name"].lower() == str(record["sector"]).lower():
                            record["gics_sector_code"] = code
                            break
                    records.append(record)
        except Exception as exc:
            logger.warning("Profile fetch failed for batch %s: %s", batch[:3], exc)
            # Still add symbols with minimal info
            for sym in batch:
                records.append({"symbol": sym})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["symbol", "name", "sector", "industry_group", "industry",
                 "sub_industry", "market_cap", "exchange", "country"]
    )


def get_bond_universe() -> pd.DataFrame:
    """Return fixed-income universe: treasuries and corporates.

    Fetches:
      - US Treasury rates via ``obb.fixedincome.rate.treasury()``
      - Corporate bond prices via ``obb.fixedincome.corporate.bond_prices()``

    Returns
    -------
    pd.DataFrame
        Combined treasury + corporate bond data.
    """
    obb = _get_obb()
    frames: List[pd.DataFrame] = []

    # Treasury rates
    try:
        result = obb.fixedincome.rate.treasury(provider="federal_reserve")
        df = _obb_result_to_df(result)
        if not df.empty:
            df["bond_type"] = "treasury"
            frames.append(df)
    except Exception as exc:
        logger.warning("Treasury rate fetch failed: %s", exc)

    # Corporate bonds
    try:
        result = obb.fixedincome.corporate.bond_prices(provider="tmx")
        df = _obb_result_to_df(result)
        if not df.empty:
            df["bond_type"] = "corporate"
            frames.append(df)
    except Exception as exc:
        logger.warning("Corporate bond fetch failed: %s", exc)

    # Bond ETF price data as supplementary
    try:
        for etf_sym in BOND_ETFS[:10]:  # first 10 to keep response fast
            result = obb.etf.price.historical(
                symbol=etf_sym,
                start_date=(date.today() - timedelta(days=5)).isoformat(),
                provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = etf_sym
                df["bond_type"] = "etf"
                frames.append(df.tail(1))  # latest row only
    except Exception as exc:
        logger.warning("Bond ETF fetch failed: %s", exc)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def get_commodity_universe() -> pd.DataFrame:
    """Return all commodity symbols with latest price data.

    Uses ``obb.commodity.price.historical()`` for each commodity symbol.

    Returns
    -------
    pd.DataFrame
        Columns include symbol, date, open, high, low, close, volume.
    """
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    for sym in COMMODITY_SYMBOLS:
        try:
            result = obb.commodity.price.historical(
                symbol=sym,
                start_date=start,
                provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                df["asset_class"] = "commodity"
                records.append(df.tail(1))
        except Exception as exc:
            logger.debug("Commodity fetch failed for %s: %s", sym, exc)
            # Try via equity endpoint (for ETFs like GLD)
            try:
                result = obb.equity.price.historical(
                    symbol=sym,
                    start_date=start,
                    provider="yfinance",
                )
                df = _obb_result_to_df(result)
                if not df.empty:
                    df["symbol"] = sym
                    df["asset_class"] = "commodity_etf"
                    records.append(df.tail(1))
            except Exception:
                pass

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def get_crypto_universe() -> pd.DataFrame:
    """Return top crypto assets with latest price data.

    Uses ``obb.crypto.price.historical()`` for each crypto symbol.

    Returns
    -------
    pd.DataFrame
        Columns include symbol, date, open, high, low, close, volume.
    """
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    for sym in CRYPTO_SYMBOLS:
        try:
            result = obb.crypto.price.historical(
                symbol=sym,
                start_date=start,
                provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                records.append(df.tail(1))
        except Exception as exc:
            logger.debug("Crypto fetch failed for %s: %s", sym, exc)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def get_fx_universe() -> pd.DataFrame:
    """Return FX universe: major, minor, and exotic pairs.

    Uses ``obb.currency.price.historical()`` for each FX pair.

    Returns
    -------
    pd.DataFrame
        Columns include symbol, pair_type, date, open, high, low, close.
    """
    obb = _get_obb()
    records: List[pd.DataFrame] = []
    start = (date.today() - timedelta(days=5)).isoformat()

    pair_type_map = {}
    for s in FX_MAJOR:
        pair_type_map[s] = "major"
    for s in FX_MINOR:
        pair_type_map[s] = "minor"
    for s in FX_EXOTIC:
        pair_type_map[s] = "exotic"

    for sym in FX_ALL:
        try:
            result = obb.currency.price.historical(
                symbol=sym,
                start_date=start,
                provider="yfinance",
            )
            df = _obb_result_to_df(result)
            if not df.empty:
                df["symbol"] = sym
                df["pair_type"] = pair_type_map.get(sym, "unknown")
                records.append(df.tail(1))
        except Exception as exc:
            logger.debug("FX fetch failed for %s: %s", sym, exc)

    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def classify_by_gics(symbols: List[str]) -> pd.DataFrame:
    """Map equity symbols to their GICS Sector / Industry Group / Industry / Sub-Industry.

    Fetches company profiles via ``obb.equity.profile()`` and matches the
    returned sector/industry strings against the GICS hierarchy.

    Parameters
    ----------
    symbols : list[str]
        Equity ticker symbols to classify.

    Returns
    -------
    pd.DataFrame
        Columns: symbol, sector, sector_code, industry_group,
        industry_group_code, industry, industry_code, sub_industry,
        sub_industry_code.
    """
    obb = _get_obb()
    records: List[dict] = []

    # Build reverse lookup: sector name -> code
    sector_name_to_code: Dict[str, int] = {}
    for code, info in GICS_SECTORS.items():
        sector_name_to_code[info["name"].lower()] = code

    # Build reverse lookup: industry name -> (sector_code, group_code, industry_code)
    industry_name_lookup: Dict[str, tuple] = {}
    for sector_code, sector_info in GICS_SECTORS.items():
        for group_code, group_info in sector_info["groups"].items():
            for ind_code, ind_info in group_info["industries"].items():
                industry_name_lookup[ind_info["name"].lower()] = (
                    sector_code, group_code, ind_code
                )

    batch_size = 20
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i : i + batch_size]
        try:
            result = obb.equity.profile(symbol=",".join(batch), provider="fmp")
            df = _obb_result_to_df(result)
            if not df.empty:
                for _, row in df.iterrows():
                    sym = row.get("symbol", "")
                    sector_str = str(row.get("sector", "")).lower()
                    industry_str = str(row.get("industry", "")).lower()

                    record: Dict[str, Any] = {
                        "symbol": sym,
                        "sector": row.get("sector", ""),
                        "sector_code": sector_name_to_code.get(sector_str),
                        "industry_group": "",
                        "industry_group_code": None,
                        "industry": row.get("industry", ""),
                        "industry_code": None,
                        "sub_industry": "",
                        "sub_industry_code": None,
                    }

                    # Try matching industry name to GICS
                    if industry_str in industry_name_lookup:
                        sc, gc, ic = industry_name_lookup[industry_str]
                        record["sector_code"] = sc
                        record["sector"] = GICS_SECTORS[sc]["name"]
                        record["industry_group_code"] = gc
                        record["industry_group"] = GICS_SECTORS[sc]["groups"][gc]["name"]
                        record["industry_code"] = ic
                        record["industry"] = GICS_SECTORS[sc]["groups"][gc]["industries"][ic]["name"]

                    records.append(record)
        except Exception as exc:
            logger.warning("GICS classification failed for batch %s: %s", batch[:3], exc)
            for sym in batch:
                records.append({"symbol": sym})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["symbol", "sector", "sector_code", "industry_group",
                 "industry_group_code", "industry", "industry_code",
                 "sub_industry", "sub_industry_code"]
    )


def get_historical(
    symbol: str,
    start: str,
    end: Optional[str] = None,
    asset_class: Optional[AssetClass] = None,
    interval: str = "1d",
    provider: str = "yfinance",
) -> pd.DataFrame:
    """Fetch historical OHLCV data with smart routing to the correct OBB endpoint.

    The function auto-detects the asset class (if not provided) and routes
    the request to the appropriate OpenBB endpoint:
      - EQUITY / ETF  -> obb.equity.price.historical()
      - CRYPTO        -> obb.crypto.price.historical()
      - FX            -> obb.currency.price.historical()
      - INDEX         -> obb.index.price.historical()
      - COMMODITY     -> obb.commodity.price.historical()
      - FIXED_INCOME  -> obb.etf.price.historical() (bond ETFs)

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    start : str
        Start date (YYYY-MM-DD).
    end : str, optional
        End date (YYYY-MM-DD).  Defaults to today.
    asset_class : AssetClass, optional
        Force a specific asset class.  Auto-detected if ``None``.
    interval : str
        Bar interval (e.g. "1d", "1h", "1w").
    provider : str
        OpenBB data provider backend.

    Returns
    -------
    pd.DataFrame
        OHLCV DataFrame with DatetimeIndex.

    Notes
    -----
    Sharpe Ratio: S = (R_p - R_f) / sigma_p
    Use the returned close prices to compute daily returns, then
    annualise: Sharpe = mean(r - r_f) / std(r - r_f) * sqrt(252).
    """
    obb = _get_obb()

    if asset_class is None:
        asset_class = detect_asset_class(symbol)

    if end is None:
        end = date.today().isoformat()

    kwargs: Dict[str, Any] = {
        "symbol": symbol,
        "start_date": start,
        "end_date": end,
        "provider": provider,
    }

    endpoint_map = {
        AssetClass.EQUITY: obb.equity.price.historical,
        AssetClass.ETF: obb.etf.price.historical,
        AssetClass.CRYPTO: obb.crypto.price.historical,
        AssetClass.FX: obb.currency.price.historical,
        AssetClass.INDEX: obb.index.price.historical,
        AssetClass.COMMODITY: obb.commodity.price.historical,
        AssetClass.FIXED_INCOME: obb.etf.price.historical,
    }

    endpoint = endpoint_map.get(asset_class, obb.equity.price.historical)

    result = endpoint(**kwargs)
    df = _obb_result_to_df(result)

    if not df.empty:
        # Normalise the index to DatetimeIndex
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

    return df


def get_multiple_historical(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    interval: str = "1d",
    provider: str = "yfinance",
) -> Dict[str, pd.DataFrame]:
    """Batch-fetch historical data for multiple symbols.

    Iterates over *symbols* and delegates each to :func:`get_historical`.

    Parameters
    ----------
    symbols : list[str]
        List of ticker symbols (can be mixed asset classes).
    start : str
        Start date (YYYY-MM-DD).
    end : str, optional
        End date.
    interval : str
        Bar interval.
    provider : str
        OpenBB data provider.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of symbol -> OHLCV DataFrame.

    Notes
    -----
    Value at Risk (parametric):
        VaR_alpha = mu - z_alpha * sigma
    Combine the returned DataFrames into a portfolio-level matrix for
    VaR computation.
    """
    results: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            df = get_historical(sym, start, end, interval=interval, provider=provider)
            if not df.empty:
                results[sym] = df
        except Exception as exc:
            logger.warning("Historical fetch failed for %s: %s", sym, exc)
    return results


def get_fundamentals(symbol: str, provider: str = "fmp") -> dict:
    """Fetch fundamental data: financials, ratios, and company profile.

    Calls:
      - ``obb.equity.profile()``
      - ``obb.equity.fundamental.ratios()``
      - ``obb.equity.fundamental.income()``
      - ``obb.equity.fundamental.balance()``
      - ``obb.equity.fundamental.cash()``

    Parameters
    ----------
    symbol : str
        Equity ticker symbol.
    provider : str
        OpenBB data provider.

    Returns
    -------
    dict
        Keys: "profile", "ratios", "income", "balance", "cash".
        Each value is a pd.DataFrame or dict.

    Notes
    -----
    Kelly Criterion: f* = (b*p - q) / b
    Use fundamental ratios to estimate win probability *p* and payoff *b*
    for position sizing.
    """
    obb = _get_obb()
    data: Dict[str, Any] = {}

    try:
        result = obb.equity.profile(symbol=symbol, provider=provider)
        data["profile"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Profile fetch failed for %s: %s", symbol, exc)
        data["profile"] = pd.DataFrame()

    try:
        result = obb.equity.fundamental.ratios(symbol=symbol, provider=provider)
        data["ratios"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Ratios fetch failed for %s: %s", symbol, exc)
        data["ratios"] = pd.DataFrame()

    try:
        result = obb.equity.fundamental.income(symbol=symbol, provider=provider)
        data["income"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Income statement fetch failed for %s: %s", symbol, exc)
        data["income"] = pd.DataFrame()

    try:
        result = obb.equity.fundamental.balance(symbol=symbol, provider=provider)
        data["balance"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Balance sheet fetch failed for %s: %s", symbol, exc)
        data["balance"] = pd.DataFrame()

    try:
        result = obb.equity.fundamental.cash(symbol=symbol, provider=provider)
        data["cash"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Cash flow fetch failed for %s: %s", symbol, exc)
        data["cash"] = pd.DataFrame()

    return data


def get_macro_data(provider: str = "federal_reserve") -> dict:
    """Fetch macroeconomic data: GDP, CPI, unemployment, treasury rates, yield curve.

    Calls:
      - ``obb.economy.gdp.nominal()``
      - ``obb.economy.cpi()``
      - ``obb.economy.unemployment()``
      - ``obb.fixedincome.rate.treasury()``

    Parameters
    ----------
    provider : str
        OpenBB data provider.

    Returns
    -------
    dict
        Keys: "gdp", "cpi", "unemployment", "treasury_rates".
        Each value is a pd.DataFrame.

    Notes
    -----
    CAPM: E[R_i] = R_f + beta_i * (E[R_m] - R_f)
    Use the treasury rate from this function as the risk-free rate R_f
    in CAPM calculations.
    """
    obb = _get_obb()
    data: Dict[str, pd.DataFrame] = {}

    # GDP
    try:
        result = obb.economy.gdp.nominal(provider="oecd")
        data["gdp"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("GDP fetch failed: %s", exc)
        data["gdp"] = pd.DataFrame()

    # CPI
    try:
        result = obb.economy.cpi(provider="fred")
        data["cpi"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("CPI fetch failed: %s", exc)
        data["cpi"] = pd.DataFrame()

    # Unemployment
    try:
        result = obb.economy.unemployment(provider="oecd")
        data["unemployment"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Unemployment fetch failed: %s", exc)
        data["unemployment"] = pd.DataFrame()

    # Treasury rates
    try:
        result = obb.fixedincome.rate.treasury(provider="federal_reserve")
        data["treasury_rates"] = _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Treasury rate fetch failed: %s", exc)
        data["treasury_rates"] = pd.DataFrame()

    return data


def get_news_sentiment(
    symbol: Optional[str] = None,
    limit: int = 50,
    provider: str = "benzinga",
) -> pd.DataFrame:
    """Fetch news with sentiment scores.

    Uses ``obb.news.company()`` for symbol-specific news or
    ``obb.news.world()`` for global headlines.

    Parameters
    ----------
    symbol : str, optional
        Ticker symbol.  If ``None``, fetches world news.
    limit : int
        Maximum number of articles.
    provider : str
        OpenBB news provider.

    Returns
    -------
    pd.DataFrame
        Columns include title, date, text, url, and (where available)
        sentiment fields.

    Notes
    -----
    Sentiment can be used as a feature in ML models alongside fundamental
    and technical factors.  Combine with Sortino Ratio for downside-aware
    risk assessment: S = (R_p - R_f) / sigma_d.
    """
    obb = _get_obb()

    try:
        if symbol is not None:
            result = obb.news.company(symbol=symbol, limit=limit, provider=provider)
        else:
            result = obb.news.world(limit=limit, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("News fetch failed: %s", exc)
        return pd.DataFrame()


def get_options_chain(
    symbol: str,
    provider: str = "cboe",
) -> pd.DataFrame:
    """Fetch the full options chain for an equity symbol.

    Uses ``obb.derivatives.options.chains()`` to retrieve all available
    strikes and expirations.

    Parameters
    ----------
    symbol : str
        Underlying equity ticker.
    provider : str
        OpenBB options data provider.

    Returns
    -------
    pd.DataFrame
        Options chain with columns: strike, expiration, option_type,
        bid, ask, last_price, volume, open_interest, implied_volatility, etc.

    Notes
    -----
    Black-Scholes European Call:
        C = S * N(d1) - K * e^(-rT) * N(d2)
        d1 = (ln(S/K) + (r + sigma^2 / 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

    Use the implied_volatility column to calibrate the Black-Scholes model
    or as an input to volatility surface construction.
    """
    obb = _get_obb()

    try:
        result = obb.derivatives.options.chains(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Options chain fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def get_earnings_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch the earnings calendar.

    Uses ``obb.equity.calendar.earnings()``.

    Parameters
    ----------
    start_date : str, optional
        Start date (YYYY-MM-DD).  Defaults to today.
    end_date : str, optional
        End date (YYYY-MM-DD).  Defaults to 7 days from today.
    provider : str
        OpenBB data provider.

    Returns
    -------
    pd.DataFrame
        Earnings calendar entries with symbol, date, EPS estimate/actual, etc.
    """
    obb = _get_obb()

    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() + timedelta(days=7)).isoformat()

    try:
        result = obb.equity.calendar.earnings(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Earnings calendar fetch failed: %s", exc)
        return pd.DataFrame()


def get_economic_calendar(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    provider: str = "fmp",
) -> pd.DataFrame:
    """Fetch the economic events calendar.

    Uses ``obb.economy.calendar()``.

    Parameters
    ----------
    start_date : str, optional
        Start date (YYYY-MM-DD).  Defaults to today.
    end_date : str, optional
        End date (YYYY-MM-DD).  Defaults to 7 days from today.
    provider : str
        OpenBB data provider.

    Returns
    -------
    pd.DataFrame
        Economic events with date, event, country, actual, forecast,
        previous values.
    """
    obb = _get_obb()

    if start_date is None:
        start_date = date.today().isoformat()
    if end_date is None:
        end_date = (date.today() + timedelta(days=7)).isoformat()

    try:
        result = obb.economy.calendar(
            start_date=start_date,
            end_date=end_date,
            provider=provider,
        )
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("Economic calendar fetch failed: %s", exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# ETF-specific helpers
# ═══════════════════════════════════════════════════════════════════════════

def get_etf_holdings(symbol: str, provider: str = "fmp") -> pd.DataFrame:
    """Fetch the holdings of an ETF.

    Uses ``obb.etf.holdings()``.

    Parameters
    ----------
    symbol : str
        ETF ticker symbol.
    provider : str
        OpenBB data provider.

    Returns
    -------
    pd.DataFrame
        Holdings with symbol, name, weight, shares, market_value columns.
    """
    obb = _get_obb()

    try:
        result = obb.etf.holdings(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("ETF holdings fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


def get_etf_info(symbol: str, provider: str = "fmp") -> pd.DataFrame:
    """Fetch ETF metadata (expense ratio, AUM, inception date, etc.).

    Uses ``obb.etf.info()``.

    Parameters
    ----------
    symbol : str
        ETF ticker symbol.
    provider : str
        OpenBB data provider.

    Returns
    -------
    pd.DataFrame
        ETF information.
    """
    obb = _get_obb()

    try:
        result = obb.etf.info(symbol=symbol, provider=provider)
        return _obb_result_to_df(result)
    except Exception as exc:
        logger.warning("ETF info fetch failed for %s: %s", symbol, exc)
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# Convenience / aggregation utilities
# ═══════════════════════════════════════════════════════════════════════════

def build_correlation_matrix(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    field: str = "close",
) -> pd.DataFrame:
    """Build a correlation matrix of daily returns for a set of symbols.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols (can be cross-asset).
    start : str
        Start date (YYYY-MM-DD).
    end : str, optional
        End date.
    field : str
        Price field to use for returns (default "close").

    Returns
    -------
    pd.DataFrame
        Correlation matrix (symbols x symbols).

    Notes
    -----
    Covariance:
        Cov(X, Y) = E[(X - mu_X)(Y - mu_Y)]
    The correlation matrix is the normalised covariance matrix:
        corr(X, Y) = Cov(X, Y) / (sigma_X * sigma_Y)
    """
    data = get_multiple_historical(symbols, start, end)
    if not data:
        return pd.DataFrame()

    closes: Dict[str, pd.Series] = {}
    for sym, df in data.items():
        if field in df.columns:
            closes[sym] = df[field]

    if not closes:
        return pd.DataFrame()

    price_df = pd.DataFrame(closes)
    returns_df = price_df.pct_change().dropna()
    return returns_df.corr()


def build_covariance_matrix(
    symbols: List[str],
    start: str,
    end: Optional[str] = None,
    field: str = "close",
    annualise: bool = True,
) -> pd.DataFrame:
    """Build a covariance matrix of daily returns.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols.
    start : str
        Start date (YYYY-MM-DD).
    end : str, optional
        End date.
    field : str
        Price field.
    annualise : bool
        If True, multiply by 252 (trading days per year).

    Returns
    -------
    pd.DataFrame
        Covariance matrix (symbols x symbols).

    Notes
    -----
    Cov(X,Y) = E[(X - mu_X)(Y - mu_Y)]
    Annualised covariance = daily covariance * 252
    """
    data = get_multiple_historical(symbols, start, end)
    if not data:
        return pd.DataFrame()

    closes: Dict[str, pd.Series] = {}
    for sym, df in data.items():
        if field in df.columns:
            closes[sym] = df[field]

    if not closes:
        return pd.DataFrame()

    price_df = pd.DataFrame(closes)
    returns_df = price_df.pct_change().dropna()
    cov = returns_df.cov()
    if annualise:
        cov = cov * 252
    return cov


def compute_var(
    returns: pd.Series,
    confidence: float = 0.95,
    method: str = "parametric",
) -> float:
    """Compute Value at Risk for a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    confidence : float
        Confidence level (e.g. 0.95 for 95% VaR).
    method : str
        "parametric" (Gaussian) or "historical".

    Returns
    -------
    float
        VaR estimate (as a positive loss number).

    Notes
    -----
    Parametric VaR:
        VaR_alpha = mu - z_alpha * sigma
    where mu = mean return, z_alpha = standard normal quantile,
    sigma = std dev of returns.

    Historical VaR:
        VaR_alpha = -percentile(returns, 1 - alpha)
    """
    from scipy import stats as scipy_stats

    if method == "parametric":
        mu = returns.mean()
        sigma = returns.std()
        z = scipy_stats.norm.ppf(1 - confidence)
        return -(mu + z * sigma)
    elif method == "historical":
        return -np.percentile(returns.dropna(), (1 - confidence) * 100)
    else:
        raise ValueError(f"Unknown VaR method: {method}")


def compute_sharpe(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualise: bool = True,
) -> float:
    """Compute the Sharpe Ratio of a return series.

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Daily risk-free rate (default 0).
    annualise : bool
        If True, annualise by * sqrt(252).

    Returns
    -------
    float
        Sharpe Ratio.

    Notes
    -----
    Sharpe Ratio: S = (R_p - R_f) / sigma_p
    Annualised: S_annual = S_daily * sqrt(252)
    """
    excess = returns - risk_free_rate
    if excess.std() == 0:
        return 0.0
    sharpe = excess.mean() / excess.std()
    if annualise:
        sharpe *= np.sqrt(252)
    return float(sharpe)


def compute_sortino(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    annualise: bool = True,
) -> float:
    """Compute the Sortino Ratio (downside-deviation only).

    Parameters
    ----------
    returns : pd.Series
        Daily return series.
    risk_free_rate : float
        Daily risk-free rate.
    annualise : bool
        If True, annualise by * sqrt(252).

    Returns
    -------
    float
        Sortino Ratio.

    Notes
    -----
    Sortino Ratio: S = (R_p - R_f) / sigma_d
    where sigma_d = sqrt(E[min(R - R_f, 0)^2]) — the downside deviation.
    """
    excess = returns - risk_free_rate
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf")
    downside_std = np.sqrt((downside ** 2).mean())
    if downside_std == 0:
        return float("inf")
    sortino = excess.mean() / downside_std
    if annualise:
        sortino *= np.sqrt(252)
    return float(sortino)


def compute_max_drawdown(prices: pd.Series) -> float:
    """Compute the maximum drawdown from a price series.

    Parameters
    ----------
    prices : pd.Series
        Price series (not returns).

    Returns
    -------
    float
        Maximum drawdown as a positive fraction (e.g. 0.25 = 25%).

    Notes
    -----
    MDD = max_{t in [0,T]} (max_{s in [0,t]} V_s - V_t) / max_{s in [0,t]} V_s
    """
    cummax = prices.cummax()
    drawdown = (cummax - prices) / cummax
    return float(drawdown.max())


def kelly_criterion(win_prob: float, win_loss_ratio: float) -> float:
    """Compute the Kelly Criterion optimal bet fraction.

    Parameters
    ----------
    win_prob : float
        Probability of winning (0 < p < 1).
    win_loss_ratio : float
        Ratio of average win to average loss (b = avg_win / avg_loss).

    Returns
    -------
    float
        Optimal fraction of capital to risk.

    Notes
    -----
    Kelly Criterion: f* = (b * p - q) / b
    where b = win/loss ratio, p = win probability, q = 1 - p.
    """
    q = 1.0 - win_prob
    return (win_loss_ratio * win_prob - q) / win_loss_ratio


# ═══════════════════════════════════════════════════════════════════════════
# Module-level convenience
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums / constants
    "AssetClass",
    "GICS_SECTORS",
    "SP500_REPRESENTATIVE",
    "BOND_ETFS",
    "COMMODITY_SYMBOLS",
    "CRYPTO_SYMBOLS",
    "FX_MAJOR",
    "FX_MINOR",
    "FX_EXOTIC",
    "FX_ALL",
    "INDEX_SYMBOLS",
    "ETF_CORE",
    # GICS helpers
    "gics_sector_name",
    "gics_industry_group_name",
    "gics_industry_name",
    "gics_sub_industry_name",
    # Detection
    "detect_asset_class",
    # Universe functions
    "get_full_universe",
    "get_equity_universe",
    "get_bond_universe",
    "get_commodity_universe",
    "get_crypto_universe",
    "get_fx_universe",
    "classify_by_gics",
    # Historical data
    "get_historical",
    "get_multiple_historical",
    # Fundamentals
    "get_fundamentals",
    # Macro
    "get_macro_data",
    # News
    "get_news_sentiment",
    # Derivatives
    "get_options_chain",
    # Calendars
    "get_earnings_calendar",
    "get_economic_calendar",
    # ETF
    "get_etf_holdings",
    "get_etf_info",
    # Analytics
    "build_correlation_matrix",
    "build_covariance_matrix",
    "compute_var",
    "compute_sharpe",
    "compute_sortino",
    "compute_max_drawdown",
    "kelly_criterion",
]
