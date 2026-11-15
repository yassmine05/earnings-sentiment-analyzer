"""Stage 1, step 1a: parse SPDR sector holdings at 2018-12-31 and propose name -> CIK matches.

Outputs
  results/stage1/spdr_holdings_2018-12-31.csv   every common-stock holding of the 11 sector funds
  results/stage1/issuer_map_proposed.csv        candidate names with proposed CIK/ticker, for review

The reviewed mapping is saved by hand as data/reference/issuer_map.csv and used by step 1b.
"""
from __future__ import annotations

import difflib
import re

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.basket import load_spdr_holdings
from earnings_sentiment.edgar import EdgarClient

SUFFIX = re.compile(
    r"\b(inc|incorporated|corp|corporation|co|company|companies|plc|ltd|limited|holdings|holding|group|"
    r"class [a-c]|cl [a-c]|the|nv|sa|lp|llc|reit|trust)\b"
)


def norm(name: str) -> str:
    s = name.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def search_terms(name: str) -> list[str]:
    base = re.sub(r"\bClass [A-C]\b", "", name).strip(" ,.")
    core = re.sub(r",?\s*\b(Inc|Corp|Co|Ltd|PLC|plc|Corporation|Company|Incorporated)\b\.?$", "", base).strip(" ,.")
    words = core.split()
    terms = [core]
    if len(words) > 2:
        terms.append(" ".join(words[:2]))
    if len(words) > 1:
        terms.append(words[0])
    return list(dict.fromkeys(t for t in terms if t))


def main() -> None:
    out_dir = config.RESULTS / "stage1"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = EdgarClient()

    holdings = load_spdr_holdings(client)
    holdings.to_csv(out_dir / "spdr_holdings_2018-12-31.csv", index=False)
    print("holdings:", len(holdings))
    print(holdings.groupby("sector").agg(n=("name", "size"), value=("value_usd", "sum")).to_string())

    cand = (
        holdings.sort_values("value_usd", ascending=False)
        .groupby("sector")
        .head(config.CANDIDATES_PER_SECTOR)
        .sort_values(["sector", "value_usd"], ascending=[True, False])
    )

    tick = client.ticker_map()
    tick["norm"] = tick["title"].map(norm)
    # company_tickers.json lists the primary ticker first for each CIK
    primary = tick.drop_duplicates("cik").set_index("cik")

    rows = []
    for r in cand.itertuples():
        n = norm(r.name)
        hit = primary[primary["norm"] == n]
        method, cik, edgar_name, score = "", None, "", 0.0
        if len(hit) == 1:
            method, cik, edgar_name, score = "ticker_map_exact", int(hit.index[0]), hit["title"].iloc[0], 1.0
        else:
            best = None
            for term in search_terms(r.name):
                for c in client.company_search(term):
                    s = difflib.SequenceMatcher(None, n, norm(c["edgar_name"])).ratio()
                    if best is None or s > best[0]:
                        best = (s, c, term)
                if best and best[0] >= 0.85:
                    break
            if best:
                method, cik, edgar_name, score = f"edgar_search:{best[2]}", best[1]["cik"], best[1]["edgar_name"], best[0]
        ticker = primary["ticker"].get(cik, "") if cik is not None else ""
        rows.append(
            {
                "sector": r.sector, "name": r.name, "value_usd": r.value_usd, "cik": cik,
                "edgar_name": edgar_name, "ticker_today": ticker, "match_score": round(score, 3), "method": method,
            }
        )
    prop = pd.DataFrame(rows)
    prop.to_csv(out_dir / "issuer_map_proposed.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 45):
        print(prop.drop(columns=["value_usd"]).to_string())
    print("EDGAR requests made:", client.n_requests)


if __name__ == "__main__":
    main()
