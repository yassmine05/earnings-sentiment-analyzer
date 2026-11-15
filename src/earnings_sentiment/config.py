"""Paths, sample dates and environment access. No personal data lives in this file."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"
RAW = DATA / "raw"                # git-ignored: EDGAR cache, prices, factors
INTERIM = DATA / "interim"        # git-ignored: cleaned text, sentence files
EXTERNAL = DATA / "external"      # git-ignored: Loughran-McDonald dictionary (user-supplied)
REFERENCE = DATA / "reference"    # committed: small hand-reviewed inputs
RESULTS = ROOT / "results"        # committed: tables and figures produced by code
DOCS = ROOT / "docs"

EDGAR_CACHE = RAW / "edgar"
PRICES_DIR = RAW / "prices"
FACTORS_DIR = RAW / "factors"

# Sample: earnings releases accepted by EDGAR (Eastern time) inside this window.
SAMPLE_START = "2019-01-01"
SAMPLE_END = "2024-12-31"

# Prices: 250 trading days before the first event, and 5+ days after the last.
PRICE_START = "2017-06-01"
PRICE_END = "2025-03-31"

# Basket definition.
BASKET_ASOF = "2018-12-31"
FIRMS_PER_SECTOR = 5
CANDIDATES_PER_SECTOR = 15

# Select Sector SPDR Trust, form N-Q for the period ending 2018-12-31.
SPDR_CIK = 1064641
SPDR_NQ_ACCESSION = "0001193125-19-056828"
SPDR_NQ_DOCUMENT = "d659817dnq.htm"

MARKET_TZ = "America/New_York"


def sec_user_agent() -> str:
    """Read the SEC User-Agent from .env. It is never hard-coded or logged."""
    load_dotenv(ROOT / ".env")
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "example.com" in ua:
        raise RuntimeError(
            "SEC_USER_AGENT is not set. Copy .env.example to .env and put your own "
            "name and contact email in it (required by SEC fair-access rules)."
        )
    return ua
