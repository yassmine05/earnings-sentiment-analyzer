"""Stage 4, step 2a: join scores to CARs and run the four headline tests H1-H4 ONCE.

This is the first script in the project that puts a sentiment score and a return in the same table.
It follows docs/stage4_analysis_plan.md exactly and refuses to run a second time: if
results/stage4/headline_results.csv already exists it stops without touching anything.

Inputs   results/stage1/events_selected.csv, results/stage1/basket.csv,
         results/stage2/event_cars.csv, results/stage3/document_scores.csv
Outputs  results/stage4/analysis_dataset.csv   one row per event: scores, CARs, controls, flags (no text)
         results/stage4/headline_results.csv   H1-H4
"""
from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pandas as pd

from earnings_sentiment import config, stage4

OUT = config.RESULTS / "stage4"
HEADLINE = [
    ("H1", "reaction", "car_0_1", "fb_tone", False),
    ("H2", "reaction", "car_0_1", "lm_tone", False),
    ("H3", "drift", "car_1_5", "fb_tone", True),
    ("H4", "drift", "car_1_5", "lm_tone", True),
]


def build_dataset() -> pd.DataFrame:
    ev = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")[
        ["accession", "ticker", "sector", "fiscal_period_end", "event_day0", "event_bucket", "event_time_changed",
         "had_other_202_before", "gap_fill"]]
    mc = pd.read_csv(config.RESULTS / "stage1" / "basket.csv")[["ticker", "market_cap_bn"]]
    cars = pd.read_csv(config.RESULTS / "stage2" / "event_cars.csv").drop(
        columns=["ticker", "sector", "fiscal_period_end", "event_time_et", "event_bucket", "event_time_changed",
                 "accepted_et", "timing_bucket", "day0_8k", "had_other_202_before", "gap_fill", "day0"], errors="ignore")
    sc = pd.read_csv(config.RESULTS / "stage3" / "document_scores.csv").drop(
        columns=["ticker", "sector", "fiscal_period_end", "event_day0"], errors="ignore")
    d = ev.merge(mc, on="ticker", validate="m:1").merge(cars, on="accession", validate="1:1").merge(sc, on="accession", validate="1:1")
    d = d.rename(columns={"ticker": "firm", "event_day0": "date"})
    d["quarter"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="Q").astype(str)
    d["size"] = np.log(d["market_cap_bn"])
    d["runup"], d["vol"], d["ar0"] = d["car_m20_m1"], d["sigma"], d["car_0_0"]
    d["market_hours"] = d["event_bucket"] == "market hours"
    d["no_tone_surprise"] = d["fb_tone_surprise"].isna() | d["lm_tone_surprise"].isna()
    return d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "headline_results.csv"
    if target.exists():
        print(f"{target} already exists. The headline tests are run once; nothing was done.")
        return 1
    d = build_dataset()
    assert len(d) == 1316 and d["accession"].is_unique
    d.to_csv(OUT / "analysis_dataset.csv", index=False)

    rows = []
    for hid, family, y, score, drift in HEADLINE:
        fit = stage4.estimate(d, y, [score], "A", drift=drift, clustering="twoway")
        r = fit.table().iloc[0].to_dict()
        rows.append({"test": hid, "family": family, "outcome": y, "specification": "A (firm and calendar-quarter FE)",
                     "clustering": "firm and event day", **r})
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for fam, idx in res.groupby("family").groups.items():
        res.loc[idx, "p_holm"] = stage4.holm(res.loc[idx, "p"].tolist())
    res["significant_5pct_holm"] = res["p_holm"] < 0.05
    res["run_at"] = dt.datetime.now().isoformat(timespec="seconds")
    res.to_csv(target, index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
        print(res.drop(columns=["specification", "clustering", "run_at"]).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
