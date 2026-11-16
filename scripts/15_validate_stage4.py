"""Stage 4 code validation on SIMULATED data only. No real return and no real sentiment score is read.

(a) The real event structure (55 firms, 24 calendar quarters, the real event days and sectors) is filled with
    simulated tone, controls and CARs. A known tone effect is planted and specification A is run with the
    production function `stage4.estimate`:
      - planted effect of 50 bp: is it recovered, and does the 95% interval cover it 95% of the time?
      - planted effect of zero: is it rejected about 5% of the time at the 5% level?
(b) One regression is cross-checked, point estimate and two-way clustered standard error, against a second,
    independent implementation (statsmodels OLS with explicit dummies and its own two-way cluster covariance),
    and compared with a third (linearmodels AbsorbingLS).

Inputs   results/stage1/events_selected.csv, results/stage1/basket.csv   (structure only: who, when, which sector)
Outputs  results/stage4/validation_simulation.csv, validation_crosscheck.csv, docs/stage4_validation.md
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

from earnings_sentiment import config, stage4

OUT = config.RESULTS / "stage4"
SEED = 20261116
N_RECOVER, N_SIZE = 1000, 2000
PLANTED_BP = 50.0


def structure() -> pd.DataFrame:
    ev = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")[["ticker", "sector", "event_day0"]]
    mc = pd.read_csv(config.RESULTS / "stage1" / "basket.csv")[["ticker", "market_cap_bn"]]
    d = ev.merge(mc, on="ticker").rename(columns={"ticker": "firm", "event_day0": "date"})
    d["quarter"] = pd.PeriodIndex(pd.to_datetime(d["date"]), freq="Q").astype(str)
    d["size"] = np.log(d["market_cap_bn"])
    return d.drop(columns="market_cap_bn")


def simulate(base: pd.DataFrame, rng: np.random.Generator, beta_bp: float, drift: bool) -> pd.DataFrame:
    """Simulated tone, controls and CAR on the real structure.

    Errors have a firm-specific volatility and a shock shared by all firms reporting on the same day; tone also
    has a same-day component. Both are needed for date clustering to matter, as it may in the real data.
    """
    d = base.copy()
    n = len(d)
    firm, quarter, date = (pd.factorize(d[c])[0] for c in ("firm", "quarter", "date"))
    tone = (rng.normal(0, 0.7, firm.max() + 1)[firm] + rng.normal(0, 0.2, quarter.max() + 1)[quarter]
            + rng.normal(0, 0.3, date.max() + 1)[date] + rng.normal(0, 0.6, n))
    d["tone"] = tone
    d["runup"] = rng.normal(0, 0.06, n)
    d["vol"] = np.exp(rng.normal(np.log(0.013), 0.3, n))
    d["ar0"] = rng.normal(0, 0.043, n)
    firm_sd = rng.uniform(0.02, 0.06, firm.max() + 1)[firm]
    u = rng.normal(0, 0.015, date.max() + 1)[date] + rng.normal(0, 1, n) * firm_sd
    z = (tone - tone.mean()) / tone.std()
    d["y"] = (beta_bp / 1e4 * z + 0.05 * d["runup"] - (0.03 * d["ar0"] if drift else 0)
              + rng.normal(0, 0.003, firm.max() + 1)[firm] + rng.normal(0, 0.004, quarter.max() + 1)[quarter] + u)
    return d


def statsmodels_fit(d: pd.DataFrame, correction: bool = True) -> tuple[float, float]:
    """Independent implementation: explicit dummies, statsmodels two-way cluster covariance."""
    z = (d["tone"] - d["tone"].mean()) / d["tone"].std()
    X = pd.concat([z.rename("tone"), d[["runup", "vol"]],
                   pd.get_dummies(d["firm"], prefix="f", drop_first=True, dtype=float),
                   pd.get_dummies(d["quarter"], prefix="q", drop_first=True, dtype=float)], axis=1)
    g = np.column_stack([pd.factorize(d["firm"])[0], pd.factorize(d["date"])[0]])
    r = sm.OLS(d["y"], sm.add_constant(X)).fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": correction})
    return float(r.params["tone"]), float(r.bse["tone"])


def linearmodels_fit(d: pd.DataFrame) -> tuple[float, float]:
    """Third implementation with its own conventions: absorbed fixed effects (linearmodels AbsorbingLS)."""
    from linearmodels.iv.absorbing import AbsorbingLS

    z = ((d["tone"] - d["tone"].mean()) / d["tone"].std()).rename("tone")
    exog = pd.concat([z, d[["runup", "vol"]]], axis=1)
    absorb = pd.DataFrame({"firm": pd.Categorical(d["firm"]), "quarter": pd.Categorical(d["quarter"])})
    clusters = pd.DataFrame({"firm": pd.factorize(d["firm"])[0], "date": pd.factorize(d["date"])[0]})
    r = AbsorbingLS(d["y"], exog, absorb=absorb).fit(cov_type="clustered", clusters=clusters)
    return float(r.params["tone"]), float(r.std_errors["tone"])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = structure()
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    # ---------------- (a) simulation ----------------
    rows = []
    for label, beta, reps, drift in (("planted effect 50 bp, reaction equation", PLANTED_BP, N_RECOVER, False),
                                     ("planted effect 50 bp, drift equation", PLANTED_BP, N_RECOVER, True),
                                     ("planted effect 0, reaction equation", 0.0, N_SIZE, False),
                                     ("planted effect 0, drift equation", 0.0, N_SIZE, True)):
        est = {c: [] for c in stage4.CLUSTERINGS}
        for _ in range(reps):
            d = simulate(base, rng, beta, drift)
            for c in stage4.CLUSTERINGS:
                f = stage4.estimate(d, "y", ["tone"], "A", drift=drift, clustering=c)
                tab = f.table().iloc[0]
                est[c].append((tab["coef_bp"], tab["se_bp"], tab["p"], tab["ci_low_bp"] <= beta <= tab["ci_high_bp"]))
        for c, vals in est.items():
            a = np.array(vals, dtype=float)
            rows.append({"scenario": label, "clustering": c, "simulations": reps, "planted_bp": beta,
                         "mean_estimate_bp": a[:, 0].mean(), "sd_of_estimates_bp": a[:, 0].std(ddof=1),
                         "mean_reported_se_bp": a[:, 1].mean(), "rejection_rate_5pct": (a[:, 2] < 0.05).mean(),
                         "ci_coverage_95pct": a[:, 3].mean()})
        print(f"{label}: done ({time.time() - t0:.0f}s)", flush=True)
    sim = pd.DataFrame(rows)
    sim.to_csv(OUT / "validation_simulation.csv", index=False)

    # ---------------- (b) cross-check on one simulated data set ----------------
    d = simulate(base, np.random.default_rng(SEED + 1), PLANTED_BP, drift=False)
    f = stage4.estimate(d, "y", ["tone"], "A", drift=False, clustering="twoway")
    f0 = stage4.estimate(d, "y", ["tone"], "A", drift=False, clustering="twoway", correction=False)
    sb, sse = statsmodels_fit(d, True)
    sb0, sse0 = statsmodels_fit(d, False)
    lb, lse = linearmodels_fit(d)
    cc = pd.DataFrame([
        {"implementation": "this project (stage4.estimate), with small-sample correction", "coef_bp": f.coef[0] * 1e4, "se_bp": f.se[0] * 1e4},
        {"implementation": "statsmodels OLS + two-way cluster, with correction", "coef_bp": sb * 1e4, "se_bp": sse * 1e4},
        {"implementation": "this project, no small-sample correction", "coef_bp": f0.coef[0] * 1e4, "se_bp": f0.se[0] * 1e4},
        {"implementation": "statsmodels, no small-sample correction", "coef_bp": sb0 * 1e4, "se_bp": sse0 * 1e4},
        {"implementation": "linearmodels AbsorbingLS, two-way clustered (its own correction)", "coef_bp": lb * 1e4, "se_bp": lse * 1e4},
    ])
    cc.to_csv(OUT / "validation_crosscheck.csv", index=False)
    rel = {"coef, project vs statsmodels": abs(f.coef[0] - sb) / abs(sb), "se, project vs statsmodels (with correction)": abs(f.se[0] - sse) / sse,
           "se, project vs statsmodels (no correction)": abs(f0.se[0] - sse0) / sse0,
           "coef, project vs linearmodels": abs(f.coef[0] - lb) / abs(lb), "se, project vs linearmodels": abs(f.se[0] - lse) / lse}
    agree = rel["coef, project vs statsmodels"] < 1e-8 and rel["se, project vs statsmodels (with correction)"] < 1e-6 \
        and rel["se, project vs statsmodels (no correction)"] < 1e-6

    def md(df, fmt="{:.3f}"):
        out = ["| " + " | ".join(df.columns) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
        for r in df.itertuples(index=False):
            out.append("| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
        return out

    mc_lo, mc_hi = 0.05 - 1.96 * np.sqrt(0.05 * 0.95 / N_SIZE), 0.05 + 1.96 * np.sqrt(0.05 * 0.95 / N_SIZE)
    L = ["# Stage 4: code validation on simulated data", "",
         "Generated by `scripts/15_validate_stage4.py`. **Simulated data only**: the script reads which firm reported "
         "on which day and in which sector, and nothing else. No real return and no real sentiment score is read.", "",
         "## (a) Planted effects on the real event structure", "",
         f"{len(base)} events, {base['firm'].nunique()} firms, {base['quarter'].nunique()} calendar quarters, {base['date'].nunique()} event days. "
         "Simulated errors have firm-specific volatility and a shock shared by all firms reporting on the same day; simulated "
         "tone has firm, quarter and same-day components. Specification A (firm and calendar-quarter fixed effects) is run with "
         f"the production function. Random seed {SEED}.", ""]
    L += md(sim) + ["",
          f"With {N_SIZE} simulations, a correct 5% test should reject between {mc_lo * 100:.1f}% and {mc_hi * 100:.1f}% of the time "
          "(95% Monte Carlo band). The headline clustering is `twoway` (firm and event day).", "",
          "## (b) Cross-check of one regression", "",
          "One simulated data set, specification A, reaction equation, two-way clustering by firm and event day:", ""]
    L += md(cc, "{:.6f}") + ["", "Relative differences:", ""] + [f"- {k}: {v:.2e}" for k, v in rel.items()] + ["",
          f"**Agreement with the independent implementation (statsmodels): {'YES' if agree else 'NO'}.** "
          "The project code and statsmodels use the same textbook formulas and are written independently; they are compared with and "
          "without the small-sample correction. linearmodels absorbs the fixed effects and applies its own degrees-of-freedom "
          "convention, so its standard error is shown for reference and is not expected to match to the last digit.", ""]
    (config.DOCS / "stage4_validation.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    if not agree:
        raise SystemExit("CROSS-CHECK FAILED: implementations disagree")


if __name__ == "__main__":
    main()
