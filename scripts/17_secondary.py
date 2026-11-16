"""Stage 4, step 2b: pre-specified secondary analyses S1-S8, labelled exploratory analyses, figures and report.

Runs only after the headline tests are saved (results/stage4/headline_results.csv). The headline tests are
NOT re-run here: the report reads the saved file.

Inputs   results/stage4/analysis_dataset.csv, headline_results.csv, plan_power.csv
Outputs  results/stage4/s1_... s8_....csv, exploratory_*.csv, figures/*.png, docs/stage4_results.md
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from earnings_sentiment import config, stage4

OUT = config.RESULTS / "stage4"
FIG = OUT / "figures"
PLAN_HASH, HEADLINE_HASH = "24ff20c", "ca419f6"
BLUE, ORANGE, SURFACE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"fb": BLUE, "lm": ORANGE}
NAME = {"fb": "FinBERT", "lm": "Loughran-McDonald"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
})
FAMILIES = {"reaction": ("car_0_1", False), "drift": ("car_1_5", True)}
HEAD = [("H1", "reaction", "fb_tone"), ("H2", "reaction", "lm_tone"), ("H3", "drift", "fb_tone"), ("H4", "drift", "lm_tone")]
COLS = ["coef_bp", "se_bp", "ci_low_bp", "ci_high_bp", "t", "p", "n"]


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> list[str]:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for row in df.itertuples(index=False):
        cells = []
        for c, v in zip(df.columns, row):
            if isinstance(v, (float, np.floating)):
                cells.append("" if np.isnan(v) else ("{:.3f}".format(v) if c.startswith("p") or "r2" in c else fmt.format(v)))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return out


def fit_rows(d, y, scores, spec, drift, clustering="twoway", **labels) -> list[dict]:
    f = stage4.estimate(d, y, scores, spec, drift=drift, clustering=clustering)
    return [{**labels, "outcome": y, **r} for r in f.table().to_dict("records")]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    head = pd.read_csv(OUT / "headline_results.csv")
    d = pd.read_csv(OUT / "analysis_dataset.csv")
    power = pd.read_csv(OUT / "plan_power.csv")
    res: dict[str, pd.DataFrame] = {}

    # ---------------- S1: both scores together, specification A ----------------
    rows = []
    for fam, (y, drift) in FAMILIES.items():
        both = stage4.estimate(d, y, ["fb_tone", "lm_tone"], "A", drift)
        fb = stage4.estimate(d, y, ["fb_tone"], "A", drift)
        lmf = stage4.estimate(d, y, ["lm_tone"], "A", drift)
        t, p = both.equal_test()
        for r in both.table().to_dict("records"):
            rows.append({"family": fam, "outcome": y, **{k: r[k] for k in ["regressor"] + COLS},
                         "t_equal_coefs": t, "p_equal_coefs": p, "r2_within_both": both.r2_within,
                         "r2_gain_from_adding_this_score": both.r2_within - (lmf.r2_within if r["regressor"] == "fb_tone" else fb.r2_within)})
    res["s1_both_scores"] = pd.DataFrame(rows)

    # ---------------- S2 and S3: specifications B and C ----------------
    for key, spec, cols in (("s2_pooled_spec_B", "B", ("fb_tone", "lm_tone")), ("s3_tone_surprise_spec_C", "C", ("fb_tone_surprise", "lm_tone_surprise"))):
        rows = []
        for fam, (y, drift) in FAMILIES.items():
            for scores, lbl in (([cols[0]], "alone"), ([cols[1]], "alone"), (list(cols), "both together")):
                rows += [{k: r[k] for k in ["family", "model", "regressor"] + COLS}
                         for r in fit_rows(d, y, scores, spec, drift, family=fam, model=lbl)]
        res[key] = pd.DataFrame(rows)

    # ---------------- S4: horizon profile ----------------
    rows = []
    windows = {"reaction": [f"car_0_{k}" for k in range(1, 6)], "drift": ["ar_1"] + [f"car_1_{k}" for k in range(2, 6)]}
    for fam, ys in windows.items():
        for score in ("fb_tone", "lm_tone"):
            part = [{k: r[k] for k in ["family", "regressor", "outcome"] + COLS}
                    for y in ys for r in fit_rows(d, y, [score], "A", fam == "drift", family=fam)]
            for r, ph in zip(part, stage4.holm([r["p"] for r in part])):
                r["p_holm_across_5_windows"] = ph
            rows += part
    res["s4_horizon_profile"] = pd.DataFrame(rows)

    # ---------------- S5: quintile sorts ----------------
    qrows, srows = [], []
    for score in ("fb_tone", "lm_tone", "fb_tone_surprise", "lm_tone_surprise"):
        q = stage4.quintiles_from_previous_quarter(d, score)
        for fam, (y, _) in FAMILIES.items():
            s = d.assign(q=q).dropna(subset=["q", y])
            for k, g in s.groupby("q"):
                qrows.append({"score": score, "family": fam, "outcome": y, "quintile": int(k), "events": len(g),
                              "mean_car_bp": g[y].mean() * 1e4, "median_car_bp": g[y].median() * 1e4})
            X = np.column_stack([np.ones(len(s))] + [(s["q"] == k).to_numpy(float) for k in (2, 3, 4, 5)])
            yy = s[y].to_numpy(float)
            b, *_ = np.linalg.lstsq(X, yy, rcond=None)
            V, g = stage4.cluster_cov(X, yy - X @ b, [s["firm"].to_numpy(), s["date"].to_numpy()])
            se, dof = np.sqrt(V[4, 4]), min(g.values()) - 1
            srows.append({"score": score, "family": fam, "outcome": y, "events_sorted": len(s), "quarters": s["quarter"].nunique(),
                          "q5_minus_q1_bp": b[4] * 1e4, "se_bp": se * 1e4, "t": b[4] / se, "p": 2 * stats.t.sf(abs(b[4] / se), dof)})
    res["s5_quintile_means"], res["s5_quintile_spreads"] = pd.DataFrame(qrows), pd.DataFrame(srows)

    # ---------------- S6: inference robustness ----------------
    rows = []
    for hid, fam, score in HEAD:
        y, drift = FAMILIES[fam]
        for cl in ("twoway", "firm", "quarter"):
            f = stage4.estimate(d, y, [score], "A", drift, clustering=cl).table().iloc[0]
            rows.append({"test": hid, "clustering": cl, **{k: f[k] for k in COLS}, "clusters": f["clusters"]})
    res["s6_clustering"] = pd.DataFrame(rows)

    # ---------------- S7: sample robustness ----------------
    samples = {
        "headline sample": d,
        "without pre-announced quarters": d[~d["had_other_202_before"]],
        "without market-hours events": d[~d["market_hours"]],
        "without clock-adjusted events": d[~d["event_time_changed"]],
        "without hand-reviewed gap fills": d[~d["gap_fill"]],
        "only events with tone surprise (specification C sample)": d[~d["no_tone_surprise"]],
    }
    rows = []
    for lbl, sub in samples.items():
        for hid, fam, score in HEAD:
            y, drift = FAMILIES[fam]
            f = stage4.estimate(sub, y, [score], "A", drift).table().iloc[0]
            rows.append({"sample": lbl, "test": hid, **{k: f[k] for k in COLS}})
    res["s7_samples"] = pd.DataFrame(rows)

    # ---------------- S8: measurement robustness ----------------
    for y in ("car_0_1", "car_1_5"):
        lo, hi = d[y].quantile([0.01, 0.99])
        d[y + "_w"] = d[y].clip(lo, hi)
    variants = [
        ("safe-harbor text kept", {"fb_tone": "fb_tone_sh", "lm_tone": "lm_tone_sh"}, {}),
        ("FinBERT label shares (fb_tone_labels)", {"fb_tone": "fb_tone_labels"}, {}),
        ("dictionary tone per word (lm_tone_per_word)", {"lm_tone": "lm_tone_per_word"}, {}),
        ("Fama-French 4-factor CARs", {}, {"car_0_1": "car_0_1_ff4", "car_1_5": "car_1_5_ff4"}),
        ("sector-ETF CARs", {}, {"car_0_1": "car_0_1_sector", "car_1_5": "car_1_5_sector"}),
        ("CARs winsorised at 1% and 99%", {}, {"car_0_1": "car_0_1_w", "car_1_5": "car_1_5_w"}),
        ("buy-and-hold abnormal returns", {}, {"car_0_1": "bhar_0_1", "car_1_5": "bhar_1_5"}),
    ]
    rows = []
    for lbl, smap, ymap in variants:
        for hid, fam, score in HEAD:
            if smap and score not in smap:
                continue
            y, drift = FAMILIES[fam]
            f = stage4.estimate(d, ymap.get(y, y), [smap.get(score, score)], "A", drift).table().iloc[0]
            rows.append({"variant": lbl, "test": hid, "regressor": f["regressor"], "outcome": ymap.get(y, y), **{k: f[k] for k in COLS}})
    res["s8_measurement"] = pd.DataFrame(rows)

    # ---------------- exploratory (not tests of the hypothesis) ----------------
    rng = np.random.default_rng(20261116)
    firms = d["firm"].unique()
    groups = {f: g for f, g in d.groupby("firm")}
    rows = []
    for fam, (y, _) in FAMILIES.items():
        def rho(frame, within):
            out = {}
            for s in ("fb_tone", "lm_tone"):
                a, b = frame[s], frame[y]
                if within:
                    a, b = a - frame.groupby("firm")[s].transform("mean"), b - frame.groupby("firm")[y].transform("mean")
                out[s] = stats.spearmanr(a, b, nan_policy="omit")[0]
            out["diff"] = out["fb_tone"] - out["lm_tone"]
            return out
        for within in (False, True):
            point = rho(d, within)
            boot = []
            for _ in range(1000):
                frame = pd.concat([groups[f].assign(firm=f"{f}_{i}") for i, f in enumerate(rng.choice(firms, len(firms)))])
                boot.append(rho(frame, within))
            boot = pd.DataFrame(boot)
            for k, lbl in (("fb_tone", "FinBERT"), ("lm_tone", "Loughran-McDonald"), ("diff", "FinBERT minus Loughran-McDonald")):
                rows.append({"family": fam, "outcome": y, "version": "within firm" if within else "pooled", "quantity": lbl,
                             "spearman": point[k], "ci_low": boot[k].quantile(0.025), "ci_high": boot[k].quantile(0.975)})
    res["exploratory_rank_correlations"] = pd.DataFrame(rows)

    rows = []
    d["year2020"] = pd.to_datetime(d["date"]).dt.year == 2020
    d["large"] = d["size"] >= d.drop_duplicates("firm")["size"].median()
    splits = {"2020 only": d[d["year2020"]], "all years except 2020": d[~d["year2020"]],
              "pre-open events": d[d["event_bucket"] == "pre-open"], "after-close events": d[d["event_bucket"] == "after close"],
              "larger half of firms": d[d["large"]], "smaller half of firms": d[~d["large"]]}
    for lbl, sub in splits.items():
        for hid, fam, score in HEAD:
            y, drift = FAMILIES[fam]
            f = stage4.estimate(sub, y, [score], "A", drift).table().iloc[0]
            rows.append({"split": lbl, "family": fam, "regressor": score, **{k: f[k] for k in COLS}})
    res["exploratory_splits"] = pd.DataFrame(rows)

    rows = []
    for fam, (y, drift) in FAMILIES.items():
        for score in ("fb_share_pos", "fb_share_neg"):
            f = stage4.estimate(d, y, [score], "A", drift).table().iloc[0]
            rows.append({"analysis": "share of positive / negative FinBERT sentences", "family": fam, "regressor": score, **{k: f[k] for k in COLS}})
    for score in ("fb_tone", "lm_tone"):
        # drift=False leaves the day-0 abnormal return out of the controls
        f = stage4.estimate(d, "car_1_5", [score], "A", drift=False).table().iloc[0]
        rows.append({"analysis": "drift equation WITHOUT the day-0 return control (total association)", "family": "drift", "regressor": score, **{k: f[k] for k in COLS}})
        f = stage4.estimate(d, "car_0_5", [score], "A", drift=False).table().iloc[0]
        rows.append({"analysis": "CAR[0,+5] (reaction and drift together)", "family": "both", "regressor": score, **{k: f[k] for k in COLS}})
    res["exploratory_other"] = pd.DataFrame(rows)

    for k, v in res.items():
        v.to_csv(OUT / f"{k}.csv", index=False)

    # ---------------- figures ----------------
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    ypos = [3, 2, 1, 0]
    for yp, r in zip(ypos, head.itertuples()):
        c = COLOR["fb" if r.regressor == "fb_tone" else "lm"]
        ax.plot([r.ci_low_bp, r.ci_high_bp], [yp, yp], color=c)
        ax.plot(r.coef_bp, yp, "o", color=c, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{r.coef_bp:+.0f} bp", (r.coef_bp, yp), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(["H1  CAR[0,+1] on FinBERT", "H2  CAR[0,+1] on Loughran-McDonald", "H3  CAR[+1,+5] on FinBERT", "H4  CAR[+1,+5] on Loughran-McDonald"])
    ax.set_xlabel("bp of CAR per 1 sd of tone (95% interval)")
    ax.set_title("Headline tests H1 to H4 (specification A)")
    ax.set_ylim(-0.6, 3.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG / "headline_coefficients.png", dpi=150); plt.close(fig)

    hp = res["s4_horizon_profile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, fam, ttl in zip(axes, ("reaction", "drift"), ("Reaction family: CAR[0, +k]", "Drift family: CAR[+1, +k]")):
        for j, (score, key) in enumerate((("fb_tone", "fb"), ("lm_tone", "lm"))):
            g = hp[(hp["family"] == fam) & (hp["regressor"] == score)].reset_index(drop=True)
            x = np.arange(1, 6) + (j - 0.5) * 0.16
            ax.vlines(x, g["ci_low_bp"], g["ci_high_bp"], color=COLOR[key], linewidth=2)
            ax.plot(x, g["coef_bp"], "o" if key == "fb" else "s", color=COLOR[key], label=NAME[key], markersize=6)
        ax.axhline(0, color=INK2, linewidth=1)
        ax.set_xticks(range(1, 6)); ax.set_xlabel("k (trading days after day 0)"); ax.set_title(ttl)
    axes[0].set_ylabel("bp per 1 sd of tone (95% interval)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(FIG / "horizon_profile.png", dpi=150); plt.close(fig)

    qm = res["s5_quintile_means"]
    for tag, pair in (("raw_tone", ("fb_tone", "lm_tone")), ("tone_surprise", ("fb_tone_surprise", "lm_tone_surprise"))):
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
        for ax, fam, ttl in zip(axes, ("reaction", "drift"), ("Mean CAR[0, +1]", "Mean CAR[+1, +5]")):
            for j, (score, key) in enumerate(zip(pair, ("fb", "lm"))):
                g = qm[(qm["family"] == fam) & (qm["score"] == score)]
                ax.bar(g["quintile"] + (j - 0.5) * 0.38, g["mean_car_bp"], width=0.34, color=COLOR[key], label=NAME[key],
                       edgecolor=SURFACE, linewidth=2)
            ax.axhline(0, color=INK2, linewidth=1)
            ax.set_xticks(range(1, 6)); ax.set_xlabel("quintile (1 = most negative tone, 5 = most positive)"); ax.set_title(ttl)
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("basis points"); axes[0].legend(frameon=False)
        fig.suptitle("Quintiles of " + ("raw tone" if tag == "raw_tone" else "tone surprise") + ", breakpoints from the previous calendar quarter",
                     x=0.01, ha="left", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(FIG / f"quintiles_{tag}.png", dpi=150); plt.close(fig)

    # ---------------- report ----------------
    pa = power[power["specification"].str.startswith("A")].reset_index(drop=True)
    L = ["# Stage 4: results", "",
         f"Analysis plan: `docs/stage4_analysis_plan.md`, committed as `{PLAN_HASH}` before any score was joined to a return. "
         f"Headline tests: run once and committed as `{HEADLINE_HASH}` before any secondary or exploratory analysis. "
         "This report is generated by `scripts/17_secondary.py`; the headline table is read from the saved file, not re-run.", "",
         "All coefficients are in basis points of CAR per one standard deviation of tone. p-values are two-sided.", "",
         "## 1. Headline results (H1 to H4)", "",
         "Specification A: firm and calendar-quarter fixed effects, controls for pre-event run-up and volatility "
         "(plus the day-0 abnormal return in the drift tests), standard errors clustered by firm and by event day.", ""]
    h = head[["test", "family", "outcome", "regressor", "coef_bp", "se_bp", "ci_low_bp", "ci_high_bp", "t", "p", "p_holm", "n", "clusters"]]
    L += md(h) + ["", "![Headline coefficients](../results/stage4/figures/headline_coefficients.png)", ""]
    nsig = int(head["significant_5pct_holm"].sum())
    L += [f"**{nsig} of the 4 headline tests are significant at the 5% level after the Holm correction.** "
          + ("" if nsig else "None is significant before the correction either "
             f"(smallest unadjusted p-value: {head['p'].min():.3f})."), ""]
    for r in head.itertuples():
        kind = "association between tone and the reaction to the same release" if r.family == "reaction" else \
            "prediction of the return over days +1 to +5, beyond what the day-0 reaction already reflects"
        L.append(f"- **{r.test}** ({NAME['fb' if r.regressor == 'fb_tone' else 'lm']}, {kind}): {r.coef_bp:+.1f} bp, "
                 f"95% interval {r.ci_low_bp:+.1f} to {r.ci_high_bp:+.1f} bp, p = {r.p:.3f}, Holm p = {r.p_holm:.3f}. No effect detected.")
    L += ["", "### Reading the nulls against the power of the test", "",
          "The plan gave the smallest effect detectable with 80% power assuming independent errors. The realised standard errors:", ""]
    cmp_ = pd.DataFrame({"test": head["test"], "planned iid SE (bp)": pa["iid standard error (bp per 1 sd of score)"],
                         "realised two-way clustered SE (bp)": head["se_bp"],
                         "ratio": head["se_bp"] / pa["iid standard error (bp per 1 sd of score)"],
                         "MDE at 5% with the realised SE (bp)": head["se_bp"] * (1.960 + 0.842),
                         "MDE at 2.5%, strictest Holm step (bp)": head["se_bp"] * (2.241 + 0.842)})
    L += md(cmp_) + ["",
          "A null here means that no effect was detected, not that there is none: effects smaller than the MDE cannot be ruled out. "
          "The 95% intervals above give the range of effects that remain compatible with the data.", ""]

    def section(title, key, note="", fmt="{:.2f}"):
        return [f"## {title}", ""] + ([note, ""] if note else []) + md(res[key], fmt) + [""]

    L += ["# Pre-specified secondary analyses", "",
          "These were listed in the plan. They carry no multiple-testing claim across the whole set and cannot replace the headline results.", ""]
    L += section("2. S1: FinBERT and Loughran-McDonald together (specification A)", "s1_both_scores",
                 "`p_equal_coefs` tests whether the two coefficients are equal. `r2_gain_from_adding_this_score` is the increase in within R2 when the score is added to a model that already holds the other one.")
    L += section("3. S2: pooled specification B (sector and calendar-quarter fixed effects)", "s2_pooled_spec_B")
    L += section("4. S3: tone surprise, specification C (calendar-quarter fixed effects)", "s3_tone_surprise_spec_C")
    L += section("5. S4: horizon profile (specification A, each score alone)", "s4_horizon_profile",
                 "Holm correction across the five windows of a family, separately for each score.")
    L += ["![Horizon profile](../results/stage4/figures/horizon_profile.png)", ""]
    L += ["## 6. S5: quintile sorts", "",
          "Breakpoints are the 20th to 80th percentiles of the score among the events of the previous calendar quarter. "
          "The first quarter with scores is left out. Spread = mean CAR of quintile 5 minus quintile 1, standard error clustered by firm and event day.", "",
          "### Spreads", ""] + md(res["s5_quintile_spreads"]) + ["", "### Raw tone: mean CAR by quintile", ""]
    for score_set, lbl in ((("fb_tone", "lm_tone"), None), (("fb_tone_surprise", "lm_tone_surprise"), "### Tone surprise: mean CAR by quintile")):
        if lbl:
            L += [lbl, ""]
        t = qm[qm["score"].isin(score_set)].pivot_table(index=["score", "outcome"], columns="quintile", values="mean_car_bp").reset_index()
        t.columns = [str(c) if not isinstance(c, (int, np.integer)) else f"Q{c} mean (bp)" for c in t.columns]
        n = qm[qm["score"].isin(score_set)].pivot_table(index=["score", "outcome"], columns="quintile", values="events").reset_index()
        t["events per quintile"] = [" / ".join(str(int(v)) for v in row[2:]) for row in n.itertuples(index=False)]
        L += md(t) + [""]
        L += [f"![Quintiles](../results/stage4/figures/quintiles_{'raw_tone' if lbl is None else 'tone_surprise'}.png)", ""]
    L += section("7. S6: other clusterings for H1 to H4", "s6_clustering",
                 "Same point estimates; only the standard errors change. Calendar-quarter clustering has 24 clusters, which is few.")
    L += section("8. S7: sample robustness for H1 to H4", "s7_samples")
    L += section("9. S8: measurement robustness for H1 to H4", "s8_measurement",
                 "One element changes per row; controls stay as in the headline equations.")
    allsec = pd.concat([res[k].assign(src=k) for k in ("s1_both_scores", "s2_pooled_spec_B", "s3_tone_surprise_spec_C", "s4_horizon_profile", "s7_samples", "s8_measurement")])
    L += ["## 10. Count of nominally significant secondary coefficients", "",
          f"Across S1, S2, S3, S4, S7 and S8 there are {len(allsec)} coefficient estimates. "
          f"**{int((allsec['p'] < 0.05).sum())}** have an unadjusted p-value below 0.05 "
          f"(about {len(allsec) * 0.05:.0f} would be expected by chance alone if every true effect were zero, and the estimates are far from independent). "
          f"Quintile spreads below 0.05: {int((res['s5_quintile_spreads']['p'] < 0.05).sum())} of {len(res['s5_quintile_spreads'])}.", ""]
    sig = allsec[allsec["p"] < 0.05]
    if len(sig):
        keep = [c for c in ["src", "family", "model", "sample", "variant", "test", "regressor", "outcome", "coef_bp", "se_bp", "t", "p"] if c in sig.columns]
        L += ["They are listed here in full so that none is hidden:", ""] + md(sig[keep].fillna("")) + [""]

    L += ["# Exploratory analyses (not tests of the hypothesis)", "",
          "Nothing in this part was a pre-specified test. It is descriptive, is not corrected for multiple comparisons, and must not be read as confirming or rejecting the hypothesis.", ""]
    L += section("11. Exploratory: rank correlations between tone and CAR", "exploratory_rank_correlations",
                 "Spearman correlations with 95% intervals from 1,000 bootstrap resamples of firms. No controls and no fixed effects in the pooled version; firm means removed in the within-firm version.",
                 fmt="{:.3f}")
    L += section("12. Exploratory: specification A in sub-samples", "exploratory_splits")
    L += section("13. Exploratory: other variants", "exploratory_other",
                 "The row without the day-0 control re-estimates the drift equation with that control removed, to show the total association rather than the part beyond the day-0 reaction.")
    L += ["The earnings-surprise control named in the plan as exploratory was **not** run: the EPS history was not assembled.", "",
          "# Deviations from the analysis plan", "", "**None.** Equations, samples, tests, clustering and the Holm correction were applied as written.", "",
          "Clarifications of points the plan left open (decided before looking at any result of the item concerned):", "",
          "- Standard errors use the small-sample correction G/(G-1) x (N-1)/(N-K) for each clustering dimension and a t distribution with "
          "(smallest number of clusters - 1) degrees of freedom. This is the convention validated in `docs/stage4_validation.md`.",
          "- In S8, when the outcome is computed with another expected-return model, the controls (run-up, volatility, day-0 return) stay the market-model ones.",
          "- In S5, the quintile spread comes from a regression of the CAR on quintile indicators with no fixed effects and no controls.",
          "- S7 uses the plan's four flags and the specification C sample; no other sample was tried.", ""]
    (config.DOCS / "stage4_results.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
