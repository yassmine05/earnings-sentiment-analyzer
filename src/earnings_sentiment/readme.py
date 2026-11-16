"""Build README.md and its figures from the saved result files.

Every number in the README is read from a file in `results/` (or from a constant in this package that the
pipeline itself used) at the time the README is built. Nothing is typed by hand. `tests/test_readme.py`
rebuilds the text and fails if the committed README differs from what the saved results give.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from earnings_sentiment import config, eventstudy, finbert_score, lm, scoring_text, tone_surprise

R = config.RESULTS
FIG_DIR = R / "readme" / "figures"
README = config.ROOT / "README.md"

# Same z-values as the analysis plan (section 8) and the Stage 4 results report.
Z_975, Z_9875, Z_80 = 1.960, 2.241, 0.842
SCORE_NAME = {"fb_tone": "FinBERT", "lm_tone": "Loughran-McDonald",
              "fb_tone_surprise": "FinBERT surprise", "lm_tone_surprise": "Loughran-McDonald surprise"}
OUTCOME_NAME = {"car_0_1": "CAR[0,+1]", "car_1_5": "CAR[+1,+5]"}
FAMILY_NAME = {"reaction": "reaction (association)", "drift": "drift (prediction)"}
SECONDARY_FILES = ("s1_both_scores", "s2_pooled_spec_B", "s3_tone_surprise_spec_C", "s4_horizon_profile",
                   "s7_samples", "s8_measurement")


# ---------------------------------------------------------------- loading
def load() -> dict:
    s4 = R / "stage4"
    d = {name: pd.read_csv(s4 / f"{name}.csv") for name in (
        "headline_results", "plan_sample_counts", "s1_both_scores", "s2_pooled_spec_B", "s3_tone_surprise_spec_C",
        "s4_horizon_profile", "s5_quintile_spreads", "s5_quintile_means", "s6_clustering", "s7_samples",
        "s8_measurement", "exploratory_rank_correlations", "exploratory_splits", "exploratory_other",
        "validation_simulation", "validation_crosscheck")}
    d["basket"] = pd.read_csv(R / "stage1" / "basket.csv")
    d["events"] = pd.read_csv(R / "stage1" / "events_selected.csv", usecols=["ticker", "event_day0", "event_time_changed"])
    d["peaks"] = pd.read_csv(R / "stage2" / "timing_check_peaks.csv")
    d["cleaning"] = pd.read_csv(R / "stage3" / "cleaning_stats.csv")
    d["method_corr"] = pd.read_csv(R / "stage3" / "method_correlations.csv")
    d["fb_complete"] = json.loads((R / "stage3" / "finbert_run_complete.json").read_text(encoding="utf-8"))
    d["fb_meta"] = json.loads((R / "stage3" / "finbert_scores.meta.json").read_text(encoding="utf-8"))
    d["pilot"] = json.loads((R / "stage3" / "pilot_timing.json").read_text(encoding="utf-8"))
    return d


# ---------------------------------------------------------------- formatting
def bp(x: float) -> str:
    return f"{x:+.1f}"


def pv(x: float) -> str:
    return f"{x:.3f}"


def n(x) -> str:
    return f"{int(x):,}"


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def count(d: dict, group: str) -> int:
    pc = d["plan_sample_counts"]
    return int(pc.loc[pc["group"] == group, "events"].iloc[0])


def mde(se_bp: float, z: float = Z_975) -> float:
    return se_bp * (z + Z_80)


# ---------------------------------------------------------------- figures
def make_figures(d: dict | None = None) -> None:
    """Draw the README figures from the saved CSV files only."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = d or load()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    blue, orange, surface, ink, ink2, grid = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    color = {"fb_tone": blue, "lm_tone": orange}
    plt.rcParams.update({
        "figure.facecolor": surface, "axes.facecolor": surface, "savefig.facecolor": surface, "axes.edgecolor": grid,
        "axes.labelcolor": ink2, "xtick.color": ink2, "ytick.color": ink2, "text.color": ink, "axes.grid": True,
        "grid.color": grid, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "lines.linewidth": 2})

    # 1. Headline coefficients with their intervals and the minimum detectable effect.
    h = d["headline_results"]
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ypos = [3, 2, 1, 0]
    for yp, r in zip(ypos, h.itertuples()):
        c = color[r.regressor]
        m = mde(r.se_bp)
        ax.plot([-m, m], [yp, yp], color=grid, linewidth=9, solid_capstyle="butt", zorder=1)
        ax.plot([r.ci_low_bp, r.ci_high_bp], [yp, yp], color=c, zorder=2)
        ax.plot(r.coef_bp, yp, "o", color=c, markersize=8, markeredgecolor=surface, markeredgewidth=2, zorder=3)
        ax.annotate(f"{r.coef_bp:+.0f} bp, Holm p = {r.p_holm:.2f}", (r.coef_bp, yp), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9)
    ax.axvline(0, color=ink2, linewidth=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels([f"{r.test}  {OUTCOME_NAME[r.outcome]} on {SCORE_NAME[r.regressor]}" for r in h.itertuples()])
    ax.set_xlabel("bp of CAR per 1 sd of tone.  Line: 95% interval.  Grey band: ± minimum detectable effect.")
    ax.set_title("Headline tests H1 to H4: no effect detected")
    ax.set_ylim(-0.6, 3.7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(); fig.savefig(FIG_DIR / "headline.png", dpi=150); plt.close(fig)

    # 2. Horizon profile (S4).
    hp = d["s4_horizon_profile"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, fam, ttl in zip(axes, ("reaction", "drift"), ("Reaction: CAR[0, +k]", "Drift: CAR[+1, +k]")):
        for j, score in enumerate(("fb_tone", "lm_tone")):
            g = hp[(hp["family"] == fam) & (hp["regressor"] == score)].reset_index(drop=True)
            x = np.arange(1, len(g) + 1) + (j - 0.5) * 0.16
            ax.vlines(x, g["ci_low_bp"], g["ci_high_bp"], color=color[score], linewidth=2)
            ax.plot(x, g["coef_bp"], "o" if j == 0 else "s", color=color[score], label=SCORE_NAME[score], markersize=6)
        ax.axhline(0, color=ink2, linewidth=1)
        ax.set_xticks(range(1, 6)); ax.set_xlabel("k (trading days after day 0)"); ax.set_title(ttl)
    axes[0].set_ylabel("bp per 1 sd of tone (95% interval)")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Secondary S4: every window, specification A", x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(FIG_DIR / "horizon_profile.png", dpi=150); plt.close(fig)

    # 3. Quintile means (S5), raw tone and tone surprise.
    qm = d["s5_quintile_means"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4), sharey="row")
    for row, (kind, pair) in enumerate((("raw tone", ("fb_tone", "lm_tone")),
                                         ("tone surprise", ("fb_tone_surprise", "lm_tone_surprise")))):
        for col, (fam, out) in enumerate((("reaction", "CAR[0, +1]"), ("drift", "CAR[+1, +5]"))):
            ax = axes[row, col]
            for j, (score, base) in enumerate(zip(pair, ("fb_tone", "lm_tone"))):
                g = qm[(qm["family"] == fam) & (qm["score"] == score)]
                ax.bar(g["quintile"] + (j - 0.5) * 0.38, g["mean_car_bp"], width=0.34, color=color[base],
                       label=SCORE_NAME[base], edgecolor=surface, linewidth=2)
            ax.axhline(0, color=ink2, linewidth=1)
            ax.set_xticks(range(1, 6)); ax.grid(axis="x", visible=False)
            ax.set_title(f"Mean {out} by quintile of {kind}")
            if row == 1:
                ax.set_xlabel("quintile (1 = most negative, 5 = most positive)")
        axes[row, 0].set_ylabel("basis points")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Secondary S5: quintile sorts, breakpoints from the previous calendar quarter", x=0.01, ha="left",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(FIG_DIR / "quintiles.png", dpi=150); plt.close(fig)


# ---------------------------------------------------------------- text
def build_readme(d: dict | None = None) -> str:
    d = d or load()
    h = d["headline_results"]
    basket, ev, cl = d["basket"], d["events"], d["cleaning"]
    years = pd.to_datetime(ev["event_day0"]).dt.year
    n_firms, n_sectors = len(basket), basket["sector"].nunique()
    per_sector = int(basket.groupby("sector").size().max())
    n_events = len(ev)
    n_head = int(h["n"].iloc[0])
    n_spec_c = count(d, "SPECIFICATION C SAMPLE (tone surprise and all controls)")
    n_days = count(d, "distinct event days (clusters by date), headline sample")
    n_quarters = count(d, "calendar quarters")
    n_changed = int(ev["event_time_changed"].sum())
    clock_firms = ev.loc[ev["event_time_changed"], "ticker"].value_counts()
    fbc, fbm, pilot = d["fb_complete"], d["fb_meta"], d["pilot"]
    pk = d["peaks"].set_index("bucket")
    h_min_holm = h["p_holm"].min()
    h_min_p = h["p"].min()
    n_sig = int(h["significant_5pct_holm"].sum())

    L: list[str] = []
    a = L.append

    # ---- title and question
    a("# Does the tone of earnings press releases move stock prices?")
    a("")
    a("An event study of the earnings press releases of large US companies. It asks whether the tone of the release "
      "is associated with the stock's abnormal return when the news comes out, and whether it predicts the abnormal "
      "return over the following days. Tone is measured two ways and compared: FinBERT, a finance-tuned language "
      "model, and the Loughran-McDonald finance dictionary.")
    a("")
    a(f"Sample: {n_firms} firms ({per_sector} of the largest in each of the {n_sectors} GICS sectors), "
      f"{n(n_events)} quarterly earnings releases from {years.min()} to {years.max()}, text and timestamps from SEC EDGAR. "
      "The analysis plan was written and committed before any sentiment score was put next to a return "
      "([docs/stage4_analysis_plan.md](docs/stage4_analysis_plan.md)), and the four headline tests were run once.")
    a("")
    a("*This README is generated by `scripts/18_readme.py` from the files in `results/`. Do not edit it by hand.*")
    a("")

    # ---- headline
    a("## Headline result: no effect detected")
    a("")
    a(f"Specification A: firm and calendar-quarter fixed effects, controls for the pre-event run-up and the stock's "
      f"volatility (plus the day-0 abnormal return in the drift tests). {n(n_head)} events; standard errors clustered by "
      f"firm ({n_firms}) and by event day ({n(n_days)}). Coefficients are basis points of cumulative abnormal return (CAR) "
      "per one standard deviation of tone. Holm correction within each family of two tests.")
    a("")
    rows = []
    for r in h.itertuples():
        rows.append([r.test, FAMILY_NAME[r.family], OUTCOME_NAME[r.outcome], SCORE_NAME[r.regressor], bp(r.coef_bp),
                     f"{bp(r.ci_low_bp)} to {bp(r.ci_high_bp)}", pv(r.p), pv(r.p_holm), f"{mde(r.se_bp):.0f}"])
    L += table(["Test", "Family", "Outcome", "Tone score", "Coefficient (bp)", "95% interval (bp)", "p",
                "Holm p", "MDE (bp)"], rows)
    a("")
    a("MDE: minimum detectable effect, the smallest effect this test would find with 80% power at the 5% level, "
      "computed from its realised standard error as SE x (1.960 + 0.842).")
    a("")
    a("![Headline coefficients](results/readme/figures/headline.png)")
    a("")
    a(("**None of the 4 headline tests is significant after the Holm correction** " if n_sig == 0 else
       f"**{n_sig} of the 4 headline tests are significant after the Holm correction** ") +
      f"(smallest Holm-adjusted p-value {pv(h_min_holm)}; smallest unadjusted p-value {pv(h_min_p)}). "
      "No effect of tone was detected, for either measure, on either the reaction or the drift. "
      "This is not evidence that tone has no effect: effects smaller than the MDE in the table, "
      f"{h['se_bp'].apply(mde).min():.0f} to {h['se_bp'].apply(mde).max():.0f} bp per standard deviation of tone, "
      "cannot be ruled out, and the 95% intervals show the range of effects that remain compatible with the data.")
    a("")
    a("How to read the two families:")
    a("")
    a("- **Reaction (H1, H2)** asks whether tone is associated with the price move on days 0 to +1. The release and "
      "the move happen together, so this is an association, not a trading result.")
    a("- **Drift (H3, H4)** asks whether tone predicts the return on days +1 to +5. Because the day-0 abnormal return "
      "is a control, the coefficient measures prediction beyond what the day-0 reaction already reflects. "
      f"Both drift estimates are negative ({bp(h.loc[2, 'coef_bp'])} and {bp(h.loc[3, 'coef_bp'])} bp), "
      "the opposite sign to a slow reaction to good news, and neither is distinguishable from zero.")
    s1 = d["s1_both_scores"]
    p_eq = s1.groupby("family")["p_equal_coefs"].first()
    a(f"- **FinBERT against the dictionary.** The data do not separate the two measures. With both scores in the same "
      f"regression (secondary analysis S1), the test that their coefficients are equal gives p = {pv(p_eq['reaction'])} "
      f"for the reaction and p = {pv(p_eq['drift'])} for the drift. This study does not show that either method is better.")
    a("")

    # ---- secondary
    a("## Pre-specified secondary analyses")
    a("")
    a("These were listed in the analysis plan before the join. They carry no multiple-testing claim across the set and "
      "do not replace or re-frame the headline result. Full tables: [docs/stage4_results.md](docs/stage4_results.md).")
    a("")
    sec = pd.concat([d[k].assign(src=k) for k in SECONDARY_FILES], ignore_index=True)
    sec_sig = sec[sec["p"] < 0.05]
    a(f"**Count.** Across S1, S2, S3, S4, S7 and S8 there are {len(sec)} coefficient estimates; {len(sec_sig)} have an "
      f"unadjusted p-value below 0.05, against about {0.05 * len(sec):.0f} expected by chance if every true effect were "
      "zero (the estimates overlap and are far from independent). "
      + ("Both are in S2." if len(sec_sig) and sec_sig["src"].nunique() == 1 and sec_sig["src"].iloc[0] == "s2_pooled_spec_B"
         else "They are listed in the results report."))
    a("")

    s2 = d["s2_pooled_spec_B"]
    a("### S2: pooled specification (sector instead of firm fixed effects): a cross-firm association")
    a("")
    s2a = s2[s2["model"] == "alone"]
    L += table(["Family", "Tone score", "Coefficient (bp)", "95% interval (bp)", "p", "Events"],
               [[FAMILY_NAME[r.family], SCORE_NAME[r.regressor], bp(r.coef_bp), f"{bp(r.ci_low_bp)} to {bp(r.ci_high_bp)}",
                 pv(r.p), n(r.n)] for r in s2a.itertuples()])
    a("")
    a("Without firm fixed effects, both tone measures are associated with the reaction. A large share of the variation in tone "
      "is between firms (each company has a house style), so this coefficient compares firms with each other: firms "
      "whose releases are habitually more positive had better reactions on average in this sample. It is a cross-firm "
      "association and **not an effect of tone on returns**: firm-level differences such as industry, size or business "
      "quality could produce it. The headline specification, which compares each firm with itself, detects no effect. "
      "The drift coefficients are not significant in this specification either.")
    a("")

    s3 = d["s3_tone_surprise_spec_C"]
    s3a = s3[s3["model"] == "alone"]
    a("### S3: tone surprise (tone minus the firm's average over its previous four releases)")
    a("")
    L += table(["Family", "Tone score", "Coefficient (bp)", "95% interval (bp)", "p", "Events"],
               [[FAMILY_NAME[r.family], SCORE_NAME[r.regressor], bp(r.coef_bp), f"{bp(r.ci_low_bp)} to {bp(r.ci_high_bp)}",
                 pv(r.p), n(r.n)] for r in s3a.itertuples()])
    a("")
    a(f"{int((s3['p'] < 0.05).sum())} of the {len(s3)} tone-surprise coefficients (each score alone and both together) "
      "has a p-value below 0.05.")
    a("")

    hp = d["s4_horizon_profile"]
    a("### S4: horizon profile")
    a("")
    a(f"Specification A for every window from CAR[0,+1] to CAR[0,+5] and from AR[+1] to CAR[+1,+5]: "
      f"{int((hp['p'] < 0.05).sum())} of {len(hp)} coefficients have an unadjusted p-value below 0.05, and "
      f"{int((hp['p_holm_across_5_windows'] < 0.05).sum())} after the Holm correction across the five windows.")
    a("")
    a("![Horizon profile](results/readme/figures/horizon_profile.png)")
    a("")

    qs = d["s5_quintile_spreads"]
    q_sig = qs[qs["p"] < 0.05]
    a("### S5: quintile sorts")
    a("")
    a("Events are sorted into quintiles of tone with breakpoints from the previous calendar quarter only. The spread is "
      "the mean CAR of the most positive quintile minus the most negative, with standard errors clustered by firm and "
      "event day.")
    a("")
    L += table(["Sorted on", "Outcome", "Events", "Q5 minus Q1 (bp)", "SE (bp)", "p"],
               [[SCORE_NAME[r.score], OUTCOME_NAME[r.outcome], n(r.events_sorted), bp(r.q5_minus_q1_bp), f"{r.se_bp:.1f}",
                 pv(r.p)] for r in qs.itertuples()])
    a("")
    sig_desc = "; ".join(f"{SCORE_NAME[r.score]} on {OUTCOME_NAME[r.outcome]}, {bp(r.q5_minus_q1_bp)} bp (p = {pv(r.p)})"
                         for r in q_sig.itertuples())
    surprise_twin = qs[(qs["score"] == "fb_tone_surprise") & (qs["outcome"] == "car_0_1")].iloc[0]
    a(f"**{len(q_sig)} of {len(qs)} spreads** has a p-value below 0.05: {sig_desc}. Raw-tone sorts compare firms with "
      "different house styles, like S2. The same sort on FinBERT tone surprise, which compares each firm with its own "
      f"recent past, gives {bp(surprise_twin.q5_minus_q1_bp)} bp (p = {pv(surprise_twin.p)}). No drift spread is "
      "significant.")
    a("")
    a("![Quintiles](results/readme/figures/quintiles.png)")
    a("")

    s6, s7, s8 = d["s6_clustering"], d["s7_samples"], d["s8_measurement"]
    s6o = s6[s6["clustering"] != "twoway"]
    s7o = s7[s7["sample"] != "headline sample"]
    a("### S6 to S8: robustness of H1 to H4")
    a("")
    L += table(["Check", "Re-estimates of H1 to H4", "With p below 0.05"], [
        [f"S6: other clusterings ({', '.join(sorted(s6o['clustering'].unique()))})", str(len(s6o)), str(int((s6o['p'] < 0.05).sum()))],
        [f"S7: other samples ({s7o['sample'].str.startswith('without').sum() // 4} flagged groups removed one at a time, and the tone-surprise sample)",
         str(len(s7o)), str(int((s7o['p'] < 0.05).sum()))],
        [f"S8: other measures ({s8['variant'].nunique()} variants: safe-harbor text kept, alternative scores, "
         "4-factor and sector-ETF CARs, winsorised CARs, buy-and-hold returns)", str(len(s8)), str(int((s8['p'] < 0.05).sum()))],
    ])
    a("")
    a("No robustness check turns a headline test significant, even before any multiple-testing correction.")
    a("")

    # ---- exploratory
    rc, sp, eo = d["exploratory_rank_correlations"], d["exploratory_splits"], d["exploratory_other"]
    a("## Exploratory analyses (not tests of the hypothesis)")
    a("")
    a("Nothing here was a pre-specified test. It is descriptive and not corrected for multiple comparisons.")
    a("")
    rcm = rc[rc["quantity"].isin(["FinBERT", "Loughran-McDonald"])]
    L += table(["Family", "Version", "Tone score", "Spearman correlation with CAR", "95% interval (firm bootstrap)"],
               [[FAMILY_NAME[r.family], r.version, r.quantity, f"{r.spearman:+.3f}", f"{r.ci_low:+.3f} to {r.ci_high:+.3f}"]
                for r in rcm.itertuples()])
    a("")
    sp_sig = sp[sp["p"] < 0.05]
    sp_desc = "; ".join(f"{SCORE_NAME[r.regressor]} {r.family} among {r.split}, {bp(r.coef_bp)} bp (p = {pv(r.p)}, "
                        f"{n(r.n)} events)" for r in sp_sig.itertuples())
    a(f"- **Sub-samples.** Specification A was re-run in {sp['split'].nunique()} sub-samples (2020 only, all years "
      f"except 2020, pre-open and after-close releases, larger and smaller half of firms), {len(sp)} coefficients in all. "
      f"**{len(sp_sig)} of {len(sp)}** has a p-value below 0.05: {sp_desc}. With this many splits, about "
      f"{0.05 * len(sp):.0f} would be expected by chance; it is not read as a finding.")
    no_ar0 = eo[eo["analysis"].str.contains("WITHOUT the day-0")]
    a("- **Drift without the day-0 control.** Removing the day-0 abnormal return from the drift equation barely changes "
      "the estimates: " + ", ".join(f"{SCORE_NAME[r.regressor]} {bp(r.coef_bp)} bp (p = {pv(r.p)})"
                                     for r in no_ar0.itertuples()) + ".")
    a("- **Earnings-surprise control.** Not run: the EPS history was not assembled. It is listed under next steps.")
    a("")

    # ---- data
    a("## Data and sources")
    a("")
    L += table(["Input", "Source", "Notes"], [
        ["Earnings press releases", "SEC EDGAR, form 8-K Item 2.02, Exhibit 99.1", "Public filings. Raw text is not redistributed here"],
        ["Event timestamps", "EDGAR acceptance time (UTC, converted to US Eastern)", "Mapped to trading sessions with the NYSE calendar"],
        ["Firm basket", "Select Sector SPDR Trust form N-Q, holdings at 2018-12-31",
         f"The {per_sector} largest firms in each of the {n_sectors} GICS sectors, fixed before the sample starts"],
        ["Prices", "Yahoo Finance via `yfinance`", "Daily returns and volume for stocks, SPY and sector ETFs"],
        ["Factors", "Ken French data library", "Daily Fama-French 3 factors and momentum (robustness only)"],
        ["Dictionary", "Loughran-McDonald Master Dictionary", "Download it yourself (see below). Not redistributed"],
    ])
    a("")
    a(f"One release per firm and fiscal quarter: {n(n_events)} events, {n(n_head)} in the headline sample (one event "
      f"lacks the run-up control) and {n(n_spec_c)} with a tone surprise. The selection rule and the dropped filings are "
      "documented in [docs/stage1_event_rule.md](docs/stage1_event_rule.md).")
    a("")

    # ---- method
    a("## Method")
    a("")
    a(f"- **Event clock.** Day 0 is the first NYSE session in which the market could trade on the release, from the "
      f"EDGAR acceptance time: {n(pk.loc['pre-open', 'events'])} releases arrived before the open, "
      f"{n(pk.loc['market hours', 'events'])} during market hours and {n(pk.loc['after close', 'events'])} after the close. "
      f"For {n_changed} events ({', '.join(f'{t} {c}' for t, c in clock_firms.items())}) the 10-Q or 10-K reached EDGAR "
      "first, and its earlier time is used. A timing check confirms the clock: in every bucket the mean absolute "
      "abnormal return and the abnormal volume peak on day 0 (mean |AR| on day -1 / 0 / +1: " +
      "; ".join(f"{b} {pk.loc[b, '|AR| day -1 (bp)']:.0f} / {pk.loc[b, '|AR| day 0 (bp)']:.0f} / "
                f"{pk.loc[b, '|AR| day +1 (bp)']:.0f} bp" for b in pk.index) +
      "). Details: [docs/stage2_event_study.md](docs/stage2_event_study.md).")
    a(f"- **Text cleaning.** Tables, the safe-harbor (forward-looking statements) boilerplate and non-prose lines are "
      f"removed, and the text is split into sentences of {scoring_text.MIN_WORDS} to {scoring_text.MAX_WORDS} words. "
      f"The median release keeps {n(cl['words_scored_main'].median())} words in {cl['n_sentences_main'].median():.0f} "
      f"sentences ({n(cl['n_sentences_main'].sum())} sentences in all, {n(fbc['distinct_sentences'])} distinct). The rules were frozen before any return was "
      "looked at: [docs/stage3_scoring_spec.md](docs/stage3_scoring_spec.md).")
    a(f"- **FinBERT.** `{finbert_score.MODEL_ID}` at revision `{fbm['revision'][:12]}`, maximum {fbm['max_length']} tokens "
      f"({n(fbc['truncated_at_max_length'])} of {n(fbc['distinct_sentences'])} distinct sentences were truncated). "
      "Release tone is the mean over sentences of P(positive) minus P(negative).")
    a(f"- **Loughran-McDonald.** Tone = (positive - negative) / (positive + negative) word counts summed over the release; a positive word is "
      f"not counted when one of {', '.join(sorted(lm.NEGATORS))} occurs within the {lm.NEGATION_WINDOW} words before it.")
    a(f"- **Tone surprise.** Tone minus the firm's mean over its previous {tone_surprise.REQUIRED_PRIOR} releases, "
      f"all within {tone_surprise.LOOKBACK_DAYS} days.")
    a(f"- **Abnormal returns.** Market model against SPY, estimated over relative days [{eventstudy.EST_START}, "
      f"{eventstudy.EST_END}] with at least {eventstudy.EST_MIN_OBS} observations, so the estimation window ends before "
      "the event.")
    a("- **Regressions.** Ordinary least squares with firm and calendar-quarter fixed effects; standard errors two-way "
      "clustered by firm and event day (Cameron-Gelbach-Miller) with a small-sample correction; Holm correction within "
      "each family.")
    sim = d["validation_simulation"]
    tw = sim[sim["clustering"] == "twoway"]
    planted = tw[tw["planted_bp"] > 0]
    null = tw[tw["planted_bp"] == 0]
    cc = d["validation_crosscheck"].set_index("implementation")["se_bp"]
    se_diff = abs(cc.iloc[0] - cc.iloc[1]) / cc.iloc[1]
    a(f"- **Validation on simulated data** before the real run: a planted {planted['planted_bp'].iloc[0]:.0f} bp effect "
      f"was recovered as {' and '.join(f'{x:.1f}' for x in planted['mean_estimate_bp'])} bp, a zero effect was rejected "
      f"{' and '.join(f'{x * 100:.1f}%' for x in null['rejection_rate_5pct'])} of the time at the 5% level, and the "
      f"two-way clustered standard error matched an independent statsmodels implementation (relative difference "
      f"{se_diff:.0e}). See [docs/stage4_validation.md](docs/stage4_validation.md).")
    a("")

    # ---- limits
    corr = d["method_corr"].set_index("pair")
    a("## Limits")
    a("")
    a(f"- **Very large firms only.** {n_firms} of the largest US companies. Their prices absorb news quickly, so the "
      "results may not carry over to smaller firms.")
    a("- **Press releases, not call transcripts.** Much of the information, and the tone of management's answers, is in "
      "the earnings call, which this study does not use.")
    a(f"- **{years.min()} to {years.max()} only**, a period that includes the 2020 pandemic shock.")
    a("- **Earnings news is not controlled for.** Neither family controls for the earnings surprise. In the reaction "
      "family a positive association could simply reflect good results being described in positive words.")
    a(f"- **Power.** With {n(n_head)} events and firm fixed effects, effects smaller than "
      f"{h['se_bp'].apply(mde).min():.0f} to {h['se_bp'].apply(mde).max():.0f} bp per standard deviation of tone "
      "cannot be detected. Drift effects of a few basis points, the size often reported for large firms, are well below that.")
    a(f"- **The two measures overlap.** FinBERT and dictionary tone correlate at "
      f"{corr.loc['primary scores', 'Pearson']:.2f} (Pearson) across releases.")
    a("- **Event time** is the EDGAR acceptance time, which can lag the newswire release. The timing check "
      "shows that the day-0 assignment works.")
    a("")

    # ---- how to run
    full_min = fbm["seconds"] / 60
    a("## How to run")
    a("")
    a("Windows PowerShell, Python 3.11, no GPU needed.")
    a("")
    a("```powershell")
    a("py -3.11 -m venv .venv")
    a(".\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt")
    a(".\\.venv\\Scripts\\python.exe -m pip install -e .")
    a("Copy-Item .env.example .env")
    a("```")
    a("")
    a("**SEC User-Agent.** SEC EDGAR requires every automated request to identify itself. Open `.env` and replace the "
      "placeholder in `SEC_USER_AGENT` with your own name and contact email. `.env` is git-ignored; never commit it.")
    a("")
    a("**Loughran-McDonald dictionary.** Download the Master Dictionary CSV yourself from the authors' page, "
      "<https://sraf.nd.edu/loughranmcdonald-master-dictionary/>, and save it in `data/external/` with its original name "
      "(`Loughran-McDonald_MasterDictionary_*.csv`). That folder is git-ignored. Check the licence terms on that page.")
    a("")
    a("Then run the scripts in order:")
    a("")
    a("```powershell")
    steps = [
        ("01a_holdings_and_issuers.py", "basket: SPDR holdings and issuers"),
        ("01b_market_caps.py", "basket: market caps"),
        ("01d_select_basket.py", "basket: top firms per sector"),
        ("02_fetch_filings.py", "EDGAR filings and exhibits (rate-limited, cached)"),
        ("03_select_events.py", "one release per firm-quarter"),
        ("04_fetch_prices.py", "prices and factors"),
        ("05_data_quality.py", ""),
        ("06_price_artifacts.py", ""),
        ("07_event_study.py", "abnormal returns"),
        ("08_stage2_report.py", "timing check"),
        ("09_build_sentences.py", "cleaning and sentences"),
        ("10_pilot.py", "optional: 5-firm FinBERT pilot"),
        ("11_safe_harbor_audit.py", ""),
        ("12_score_finbert.py", f"FinBERT on CPU: about {full_min:.0f} minutes, resumable"),
        ("13_stage3_report.py", "score checks"),
        ("14_power.py", "power, before the join"),
        ("15_validate_stage4.py", "validation on simulated data"),
        ("16_headline.py", "H1 to H4; refuses to run if results/stage4/headline_results.csv exists"),
        ("17_secondary.py", "S1 to S8, exploratory, docs/stage4_results.md"),
        ("18_readme.py", "this README and its figures"),
    ]
    cmds = [(f".\\.venv\\Scripts\\python.exe scripts\\{s}", note) for s, note in steps]
    width = max(len(c) for c, _ in cmds) + 3
    for cmd, note in cmds:
        a(f"{cmd.ljust(width)}# {note}" if note else cmd)
    a(".\\.venv\\Scripts\\python.exe -m pytest -q")
    a("```")
    a("")
    a(f"FinBERT scoring time, measured on this project's CPU run (`results/stage3/finbert_scores.meta.json`): "
      f"{n(fbm['sentences_scored_this_run'])} sentences in {full_min:.0f} minutes, {fbm['sentences_per_second']:.0f} "
      f"sentences per second (`torch {fbm['torch']}`), after the pilot had scored {n(pilot['sentences'])} sentences in "
      f"{pilot['seconds'] / 60:.0f} minutes with {pilot['threads']} CPU threads. Scoring saves its "
      "progress and resumes if interrupted. `scripts/01c_cover_page_shares.py` only prints the cover-page share counts "
      "that were checked by hand for multi-class firms (`data/reference/multiclass_shares.csv`).")
    a("")
    a("`scripts/18_readme.py --check` and `tests/test_readme.py` fail if this README differs from what the saved "
      "results produce.")
    a("")

    # ---- licences
    a("## Licences and intended use")
    a("")
    a("- This is **non-commercial research**, published as a portfolio project.")
    a("- FinBERT (`ProsusAI/finbert`) is licensed CC BY-NC-SA 4.0 (non-commercial, share-alike).")
    a("- The Loughran-McDonald dictionary is free for academic research; commercial use needs a licence from its "
      "authors. It is not included in this repository.")
    a("- Prices come from Yahoo Finance through `yfinance`, for personal research use; price files are not committed.")
    a("- Raw filing text, price files and the dictionary are never committed. The repository holds code, derived scores "
      "and results.")
    a("- **Not investment advice.** Nothing here is a recommendation to buy or sell any security.")
    a("")

    # ---- next
    a("## What I would do next")
    a("")
    a("- Add the earnings-surprise control (seasonal random-walk EPS surprise from EDGAR XBRL), so that tone can be "
      "separated from the news in the numbers.")
    a("- Score earnings-call transcripts, especially the question-and-answer section, from a source with clear "
      "redistribution rights.")
    a("- Extend to smaller firms and a longer period: more events would lower the MDE, and smaller firms are where "
      "slow reactions are more plausible.")
    a("- Use newswire timestamps to measure the reaction in intraday windows.")
    a("- Score the outlook or guidance sections on their own, where forward-looking tone is concentrated.")
    a("- Write and commit a new analysis plan before any of these, as was done here.")
    a("")

    # ---- docs
    a("## Stage reports")
    a("")
    for f, label in (("stage0_feasibility.md", "Stage 0: feasibility"), ("stage1_event_rule.md", "Stage 1: event selection rule"),
                     ("stage1_data_quality.md", "Stage 1: data quality"), ("stage1_price_artifacts.md", "Stage 1: price artifacts"),
                     ("stage2_event_study.md", "Stage 2: event study and timing check"),
                     ("stage3_scoring_spec.md", "Stage 3: frozen scoring specification"),
                     ("stage3_pilot_report.md", "Stage 3: pilot"), ("stage3_scores_report.md", "Stage 3: score checks"),
                     ("stage4_analysis_plan.md", "Stage 4: analysis plan (committed before the join)"),
                     ("stage4_validation.md", "Stage 4: validation on simulated data"),
                     ("stage4_results.md", "Stage 4: all results")):
        a(f"- [{label}](docs/{f})")
    a("")
    return "\n".join(L)
