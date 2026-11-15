"""Stage 1, step 1c (aid for the hand-check): print the cover-page share-count sentences of the
10-K/10-Q closest to 2018-12-31 for issuers that have no usable XBRL share count.

The numbers are then entered by hand in data/reference/multiclass_shares.csv with the accession number.
Usage: python scripts/01c_cover_page_shares.py [CIK ...]   (default: issuers marked MISSING)
"""
from __future__ import annotations

import re
import sys

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.edgar import EdgarClient
from earnings_sentiment.textclean import html_to_text


def main() -> None:
    client = EdgarClient()
    if len(sys.argv) > 1:
        ciks = [int(x) for x in sys.argv[1:]]
    else:
        cand = pd.read_csv(config.RESULTS / "stage1" / "basket_candidates.csv")
        ciks = cand.loc[cand["shares_source"] == "MISSING", "cik"].tolist()
    for cik in ciks:
        head, df = client.submissions(cik)
        per = df[df["form"].isin(["10-K", "10-Q"]) & (df["filingDate"] >= "2018-10-15") & (df["filingDate"] <= "2019-03-15")].copy()
        print(f"\n=== CIK {cik} {head['name']} | periodic filings near 2018-12-31: {len(per)}")
        if per.empty:
            continue
        per["gap"] = (pd.to_datetime(per["filingDate"]) - pd.Timestamp(config.BASKET_ASOF)).abs().dt.days
        r = per.sort_values("gap").iloc[0]
        url = f"{client.filing_base(cik, r['accessionNumber'])}/{r['primaryDocument']}"
        text = html_to_text(client.get_text(url), drop_tables=False)[:30000]
        flat = re.sub(r"\s+", " ", text)
        print(f"    {r['form']} filed {r['filingDate']} period {r['reportDate']} accession {r['accessionNumber']}")
        seen = 0
        for m in re.finditer(r"outstanding", flat, re.I):
            s = flat[max(0, m.start() - 330): m.end() + 330]
            if re.search(r"\d{1,3}(,\d{3}){1,}", s) and re.search(r"shares|stock", s, re.I):
                print("    ...", s, "...")
                seen += 1
                if seen >= 2:
                    break


if __name__ == "__main__":
    main()
