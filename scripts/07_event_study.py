"""Stage 2, step 1: abnormal returns, CARs and abnormal volume for every event.

Inputs   results/stage1/events_selected.csv, results/stage1/price_excluded_days.csv, data/raw/prices, data/raw/factors
Outputs  results/stage2/event_cars.csv        one row per event: model parameters and every CAR window
                                              (market model = primary; ff4 and sector columns are suffixed)
         results/stage2/ar_by_relday.csv      market-model AR, standardised AR and abnormal volume, relative days -20..+10
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from earnings_sentiment import config
from earnings_sentiment import eventstudy as es

OUT = config.RESULTS / "stage2"
ID = ["ticker", "accession"]


def run(events: pd.DataFrame, md: es.MarketData, day0_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    car_rows, long_rows = [], []
    for e in events.itertuples():
        d0 = pd.Timestamp(getattr(e, day0_col))
        row = {"ticker": e.ticker, "accession": e.accession, "sector": e.sector, "day0": d0.date().isoformat()}
        for model in ("market", "ff4", "sector"):
            r = es.event_abnormal_returns(md, e.ticker, e.sector, d0, model)
            if r is None:
                continue
            sfx = "" if model == "market" else f"_{model}"
            cars = es.cumulate(r["rel"], r["ar"])
            row.update({f"{k}{sfx}": v for k, v in cars.items()})
            row.update({f"n_est{sfx}": r["n_est"], f"est_ok{sfx}": r["est_ok"], f"beta{sfx}": r["beta"],
                        f"sigma{sfx}": r["sigma"], f"r2{sfx}": r["r2"]})
            if model == "market":
                row["alpha"] = r["alpha"]
                row["bhar_0_1"] = es.buy_and_hold(r["rel"], r["ret"], r["expected"], 0, 1)
                row["bhar_1_5"] = es.buy_and_hold(r["rel"], r["ret"], r["expected"], 1, 5)
                row["missing_ar_m1_p5"] = bool(np.isnan(r["ar"][(r["rel"] >= -1) & (r["rel"] <= 5)]).any())
                av = es.abnormal_volume(md, e.ticker, d0)
                for rel, ar, ret, a in zip(r["rel"], r["ar"], r["ret"], av):
                    long_rows.append({"ticker": e.ticker, "accession": e.accession, "rel": int(rel), "ret": ret,
                                      "ar": ar, "sar": ar / r["sigma"], "abn_log_volume": a})
        car_rows.append(row)
    return pd.DataFrame(car_rows), pd.DataFrame(long_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    excluded = pd.read_csv(config.RESULTS / "stage1" / "price_excluded_days.csv")
    tickers = sorted(set(events["ticker"]) | {"SPY"} | set(es.SECTOR_ETF.values()))
    md = es.load_market_data(tickers, excluded)

    cars, long = run(events, md, "event_day0")
    meta = events[ID + ["fiscal_period_end", "event_time_et", "event_bucket", "event_time_changed", "accepted_et",
                        "timing_bucket", "day0_8k", "had_other_202_before", "gap_fill"]]
    cars = meta.merge(cars, on=ID, how="left")
    cars.to_csv(OUT / "event_cars.csv", index=False)
    long.to_csv(OUT / "ar_by_relday.csv", index=False)

    # The 15 clock-adjusted events again under the unadjusted 8-K clock, for the timing check.
    chg = events[events["event_time_changed"]]
    cars_8k, long_8k = run(chg, md, "day0_8k")
    long_8k.to_csv(OUT / "ar_by_relday_changed_events_8k_clock.csv", index=False)

    print("events:", len(events), "| with market-model CARs:", int(cars["car_0_1"].notna().sum()),
          "| estimation window too short:", int((~cars["est_ok"].fillna(False).astype(bool)).sum()),
          "| missing AR inside [-1,+5]:", int(cars["missing_ar_m1_p5"].sum()))
    print(cars[["n_est", "beta", "r2", "sigma", "car_0_1", "car_1_5"]].describe().round(4).to_string())
    print("rows in long table:", len(long), "| changed-clock rows:", len(long_8k))


if __name__ == "__main__":
    main()
