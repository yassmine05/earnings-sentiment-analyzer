"""Build the firm basket from SEC filings only.

Membership + GICS sector: Select Sector SPDR Trust form N-Q, holdings at 2018-12-31.
Market cap (approximate): cover-page shares outstanding nearest 2018-12-31
(EDGAR XBRL, dei:EntityCommonStockSharesOutstanding) x the 2018-12-31 price
implied by the same N-Q (position value / shares held).
"""
from __future__ import annotations

import html as htmllib
import re

import pandas as pd
from bs4 import BeautifulSoup

from . import config
from .edgar import EdgarClient

FUND_TO_SECTOR = {
    "COMMUNICATION SERVICES": "Communication Services",
    "CONSUMER DISCRETIONARY": "Consumer Discretionary",
    "CONSUMER STAPLES": "Consumer Staples",
    "ENERGY": "Energy",
    "FINANCIAL": "Financials",
    "HEALTH CARE": "Health Care",
    "INDUSTRIAL": "Industrials",
    "MATERIALS": "Materials",
    "REAL ESTATE": "Real Estate",
    "TECHNOLOGY": "Information Technology",
    "UTILITIES": "Utilities",
}

_FUND_RE = re.compile(r"THE\s+([A-Z ]+?)\s+SELECT\s+SECTOR\s+SPDR\s+FUND")
_INT_RE = re.compile(r"^\$?\s*([\d,]+)$")


def _clean(s: str) -> str:
    s = htmllib.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def _clean_name(s: str) -> str:
    s = re.sub(r"\(\s*[a-z]\s*\)", " ", _clean(s))  # footnote markers such as (a)
    return re.sub(r"\s+", " ", s).strip(" .")


def parse_spdr_nq(html: str) -> pd.DataFrame:
    """Return one row per common-stock holding: sector, name, shares, value."""
    heads = [(m.start(), m.group(1)) for m in _FUND_RE.finditer(html)]
    rows = []
    for k, (pos, fund) in enumerate(heads):
        fund = re.sub(r"\s+", " ", fund).strip()
        if fund not in FUND_TO_SECTOR:
            continue
        end = heads[k + 1][0] if k + 1 < len(heads) else len(html)
        segment = html[pos:end]
        if "SCHEDULE OF INVESTMENTS" not in segment[:3000].upper():
            continue
        soup = BeautifulSoup(segment, "lxml")
        in_common = False
        for tr in soup.find_all("tr"):
            cells = [_clean(td.get_text(" ")) for td in tr.find_all("td")]
            cells = [c for c in cells if c and c != "$"]
            if not cells:
                continue
            first = cells[0].upper()
            if first.startswith("COMMON STOCKS"):
                in_common = True
                continue
            if first.startswith("TOTAL COMMON STOCKS") or first.startswith("SHORT-TERM"):
                in_common = False
                continue
            if not in_common or len(cells) != 3:
                continue
            m_sh, m_val = _INT_RE.match(cells[1]), _INT_RE.match(cells[2])
            if not (m_sh and m_val):
                continue
            rows.append(
                {
                    "sector": FUND_TO_SECTOR[fund],
                    "name": _clean_name(cells[0]),
                    "shares_held": int(m_sh.group(1).replace(",", "")),
                    "value_usd": int(m_val.group(1).replace(",", "")),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(["sector", "name"]).reset_index(drop=True)
    df["implied_price"] = df["value_usd"] / df["shares_held"]
    return df


def load_spdr_holdings(client: EdgarClient) -> pd.DataFrame:
    url = f"{client.filing_base(config.SPDR_CIK, config.SPDR_NQ_ACCESSION)}/{config.SPDR_NQ_DOCUMENT}"
    return parse_spdr_nq(client.get_text(url))


def nearest_shares(facts: pd.DataFrame, asof: str = config.BASKET_ASOF, max_days: int = 100):
    """Cover-page share count whose date is closest to `asof`; None if none within max_days."""
    if facts.empty:
        return None
    f = facts.copy()
    f["end"] = pd.to_datetime(f["end"])
    f["gap"] = (f["end"] - pd.Timestamp(asof)).abs().dt.days
    f = f[f["gap"] <= max_days].sort_values(["gap", "filed"])
    if f.empty:
        return None
    r = f.iloc[0]
    return {
        "shares_out": float(r["val"]),
        "shares_date": r["end"].date().isoformat(),
        "shares_form": r["form"],
        "shares_accession": r.get("accn", ""),
    }
