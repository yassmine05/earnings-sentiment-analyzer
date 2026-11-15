"""Stage 1, step 2: list every 8-K Item 2.02 filing of the basket firms and download the EX-99 exhibit.

Input   results/stage1/basket.csv
Output  data/interim/filings_all.csv   one row per Item 2.02 filing (metadata, text features and the first
        220 characters of the exhibit). Git-ignored, like the raw exhibits cached under data/raw/edgar/.
"""
from __future__ import annotations

import sys

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.edgar import EdgarClient
from earnings_sentiment.events import collect_firm_events


def main() -> None:
    out_dir = config.RESULTS / "stage1"
    basket = pd.read_csv(out_dir / "basket.csv")
    only = set(sys.argv[1:])
    client = EdgarClient()
    frames = []
    for r in basket.itertuples():
        if only and r.ticker not in only:
            continue
        df = collect_firm_events(client, [int(c) for c in str(r.cik_all).split(";")], r.ticker)
        df.insert(0, "sector", r.sector)
        frames.append(df)
        in_sample = df["accepted_et_date"].between(config.SAMPLE_START, config.SAMPLE_END).sum()
        print(f"{r.ticker:6s} filings {len(df):3d} | in sample window {in_sample:3d} | no exhibit {(df['exhibit_url'] == '').sum()}", flush=True)
    allf = pd.concat(frames, ignore_index=True)
    if not only:
        config.INTERIM.mkdir(parents=True, exist_ok=True)
        allf.to_csv(config.INTERIM / "filings_all.csv", index=False)
    print("total filings:", len(allf), "| EDGAR requests made this run:", client.n_requests)


if __name__ == "__main__":
    main()
