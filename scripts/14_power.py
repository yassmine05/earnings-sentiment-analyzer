"""Stage 4, step 1: sample counts and minimum detectable effects for the analysis plan.

IMPORTANT: this script never places a sentiment score and a return in the same table. It computes
  (a) from the Stage 2 file alone: the residual standard deviation of each CAR after the fixed effects
      and controls of each specification, and
  (b) from the Stage 3 file alone: the residual standard deviation of each standardised score after the
      fixed effects of each specification,
and combines the two NUMBERS in the textbook formula for the standard error of a regression slope.
No covariance or correlation between a score and a return is computed.

Outputs  results/stage4/plan_sample_counts.csv, results/stage4/plan_power.csv
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from earnings_sentiment import config

OUT = config.RESULTS / "stage4"
Z_ALPHA = {"5% two-sided": 1.960, "2.5% two-sided (strictest Holm step, 2 tests)": 2.241}
Z_POWER_80 = 0.842


def dummies(s: pd.Series) -> np.ndarray:
    return pd.get_dummies(s, drop_first=True).to_numpy(dtype=float)


def resid_sd(y: pd.Series, blocks: list[np.ndarray]) -> tuple[float, int, int]:
    """Residual standard deviation of y after an intercept and the given regressor blocks; also n and parameters."""
    X = np.column_stack([np.ones(len(y))] + blocks)
    ok = ~np.isnan(y.to_numpy(dtype=float)) & ~np.isnan(X).any(axis=1)
    yy, XX = y.to_numpy(dtype=float)[ok], X[ok]
    beta, *_ = np.linalg.lstsq(XX, yy, rcond=None)
    r = yy - XX @ beta
    k = np.linalg.matrix_rank(XX)
    return float(np.sqrt(r @ r / (len(yy) - k))), int(len(yy)), int(k)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    basket = pd.read_csv(config.RESULTS / "stage1" / "basket.csv")[["ticker", "market_cap_bn"]]
    design = events[["accession", "ticker", "sector", "event_day0", "event_bucket", "event_time_changed",
                     "had_other_202_before", "gap_fill"]].merge(basket, on="ticker")
    design["quarter"] = pd.PeriodIndex(pd.to_datetime(design["event_day0"]), freq="Q").astype(str)
    design["log_mcap"] = np.log(design["market_cap_bn"])

    # ---------------- (a) returns side: Stage 2 file only ----------------
    cars = pd.read_csv(config.RESULTS / "stage2" / "event_cars.csv")[
        ["accession", "car_0_1", "car_1_5", "car_0_0", "car_m20_m1", "sigma", "est_ok"]]
    R = design.merge(cars, on="accession", how="left")
    # ---------------- (b) scores side: Stage 3 file only ------------------
    sc = pd.read_csv(config.RESULTS / "stage3" / "document_scores.csv")[
        ["accession", "fb_tone", "lm_tone", "fb_tone_surprise", "lm_tone_surprise"]]
    S = design.merge(sc, on="accession", how="left")

    has_ctrl = set(R.loc[R["car_m20_m1"].notna() & R["sigma"].notna(), "accession"])
    has_surprise = set(S.loc[S["fb_tone_surprise"].notna() & S["lm_tone_surprise"].notna(), "accession"])
    counts = pd.DataFrame([
        {"group": "events kept in Stage 1", "events": len(design)},
        {"group": "with market-model CARs and a full estimation window", "events": int(R["est_ok"].fillna(False).astype(bool).sum())},
        {"group": "with both sentiment scores", "events": int(S[["fb_tone", "lm_tone"]].notna().all(axis=1).sum())},
        {"group": "missing the pre-event run-up control CAR[-20,-1]", "events": int(R["car_m20_m1"].isna().sum())},
        {"group": "HEADLINE SAMPLE (specifications A and B)", "events": len(has_ctrl)},
        {"group": "with tone surprise for both methods", "events": len(has_surprise)},
        {"group": "SPECIFICATION C SAMPLE (tone surprise and all controls)", "events": len(has_ctrl & has_surprise)},
        {"group": "flag: earlier Item 2.02 filing in the same quarter (pre-announcement)", "events": int(design["had_other_202_before"].sum())},
        {"group": "flag: market-hours event", "events": int((design["event_bucket"] == "market hours").sum())},
        {"group": "flag: event clock moved to the 10-Q/10-K acceptance time", "events": int(design["event_time_changed"].sum())},
        {"group": "flag: earnings release filed without Item 2.02 (hand-reviewed gap fill)", "events": int(design["gap_fill"].sum())},
        {"group": "flag: no tone surprise", "events": len(design) - len(has_surprise)},
        {"group": "firms", "events": int(design["ticker"].nunique())},
        {"group": "distinct event days (clusters by date), headline sample", "events": int(design[design["accession"].isin(has_ctrl)]["event_day0"].nunique())},
        {"group": "calendar quarters", "events": int(design["quarter"].nunique())},
    ])
    counts.to_csv(OUT / "plan_sample_counts.csv", index=False)
    print(counts.to_string(index=False))
    missing = R[R["car_m20_m1"].isna()][["ticker", "event_day0"]]
    print("event(s) missing the run-up control:", missing.values.tolist())

    rows = []
    specs = {
        "A: firm + quarter FE": (has_ctrl, lambda d: [dummies(d["ticker"]), dummies(d["quarter"])], False),
        "B: sector + quarter FE": (has_ctrl, lambda d: [dummies(d["sector"]), dummies(d["quarter"])], True),
        "C: tone surprise, quarter FE": (has_ctrl & has_surprise, lambda d: [dummies(d["quarter"])], True),
    }
    for spec, (keep, fe, with_mcap) in specs.items():
        r = R[R["accession"].isin(keep)].reset_index(drop=True)
        s = S[S["accession"].isin(keep)].reset_index(drop=True)
        for fam, ycol, extra in (("reaction CAR[0,+1]", "car_0_1", []), ("drift CAR[+1,+5]", "car_1_5", ["car_0_0"])):
            ctrl = ["car_m20_m1", "sigma"] + extra + (["log_mcap"] if with_mcap else [])
            sd_y, n, k_y = resid_sd(r[ycol], fe(r) + [r[ctrl].to_numpy(dtype=float)])
            for method, col in (("FinBERT", "fb_tone"), ("Loughran-McDonald", "lm_tone")):
                col = col + ("_surprise" if spec.startswith("C") else "")
                z = (s[col] - s[col].mean()) / s[col].std()
                sd_x, _, _ = resid_sd(z, fe(s) + ([s[["log_mcap"]].to_numpy(dtype=float)] if with_mcap else []))
                se = sd_y / (np.sqrt(n) * sd_x)
                rec = {"specification": spec, "family": fam, "method": method, "n": n,
                       "sd of CAR after FE and controls (%)": sd_y * 100,
                       "sd of standardised score after FE": sd_x,
                       "iid standard error (bp per 1 sd of score)": se * 1e4}
                for lbl, za in Z_ALPHA.items():
                    rec[f"MDE at {lbl} (bp)"] = (za + Z_POWER_80) * se * 1e4
                rec["MDE at 5% if clustering inflates the SE by 1.5 (bp)"] = (1.960 + Z_POWER_80) * se * 1.5 * 1e4
                rec["MDE as a share of the CAR's sd"] = (1.960 + Z_POWER_80) * se / sd_y
                rows.append(rec)
    power = pd.DataFrame(rows)
    power.to_csv(OUT / "plan_power.csv", index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.2f}".format):
        print(power.to_string(index=False))


if __name__ == "__main__":
    main()
