"""Stage 3, step 4: document scores for both text versions, sanity checks, and docs/stage3_scores_report.md.

Inputs   results/stage3/finbert_scores.parquet, data/interim/sentences_{main,with_sh}.parquet,
         results/stage1/events_selected.csv, the LM dictionary in data/external/
Outputs  results/stage3/document_scores.csv         one row per event, both methods, both versions (no text)
         results/stage3/lm_sentence_counts.parquet   LM counts per distinct sentence (no text)
         results/stage3/*.csv tables, results/stage3/figures/*.png
         docs/stage3_scores_report.md
         data/interim/stage3_excerpts.txt            sentences behind the extreme documents (git-ignored: text)

No price, return or CAR is read anywhere in this script.
"""
from __future__ import annotations

import collections
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from earnings_sentiment import config
from earnings_sentiment import finbert_score as fb
from earnings_sentiment import lm
from earnings_sentiment.tone_surprise import tone_surprise

OUT = config.RESULTS / "stage3"
FIG = OUT / "figures"
BLUE, ORANGE, SURFACE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})
_TOKEN = re.compile(r"[A-Za-z]+")


def md(df: pd.DataFrame, fmt: str = "{:.3f}") -> list[str]:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(
            ("" if np.isnan(v) else fmt.format(v)) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return out


def describe(s: pd.Series) -> dict:
    q = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return {"n": int(s.notna().sum()), "mean": s.mean(), "std": s.std(), "min": s.min(), "5%": q[0.05], "25%": q[0.25],
            "median": q[0.5], "75%": q[0.75], "95%": q[0.95], "max": s.max()}


def firm_share(df: pd.DataFrame, col: str) -> float:
    """Share of the variance of a score that lies between firms (R2 of firm means)."""
    x = df[col].dropna()
    means = df.loc[x.index].groupby("ticker")[col].transform("mean")
    return float(1 - ((x - means) ** 2).sum() / ((x - x.mean()) ** 2).sum())


def top_words(texts: pd.Series, vocab: frozenset, k: int = 6) -> str:
    c = collections.Counter(t.upper() for s in texts for t in _TOKEN.findall(s) if t.upper() in vocab)
    return ", ".join(f"{w.lower()} {n}" for w, n in c.most_common(k))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    scores = pd.read_parquet(OUT / "finbert_scores.parquet")
    lmd = lm.load_dictionary()
    sents = {v: pd.read_parquet(config.INTERIM / f"sentences_{v}.parquet") for v in ("main", "with_sh")}

    # ---- verification ---------------------------------------------------------
    uniq_ids = set(pd.concat([s["text_id"] for s in sents.values()]))
    ver = [{"check": "distinct sentences across both versions", "value": len(uniq_ids)},
           {"check": "distinct sentences with a FinBERT score", "value": int(scores["text_id"].isin(uniq_ids).sum())},
           {"check": "distinct sentences without a score", "value": len(uniq_ids - set(scores["text_id"]))},
           {"check": "duplicate score rows", "value": int(scores["text_id"].duplicated().sum())}]
    for v, s in sents.items():
        ver += [{"check": f"{v}: sentences", "value": len(s)},
                {"check": f"{v}: sentences without a score", "value": int((~s["text_id"].isin(set(scores["text_id"]))).sum())},
                {"check": f"{v}: events with at least one sentence (of {len(events)})", "value": int(s["accession"].nunique())}]
    ver = pd.DataFrame(ver)
    if int(ver.loc[ver["check"].str.contains("without a score"), "value"].sum()) != 0:
        raise SystemExit("scores are incomplete; run scripts/12_score_finbert.py again")

    # ---- LM counts per distinct sentence, then document scores for both versions ----
    all_s = pd.concat(sents.values()).drop_duplicates("text_id")
    lm_counts = pd.DataFrame([lm.score_sentence(t, lmd) for t in all_s["text"]], index=all_s["text_id"]).reset_index()
    lm_counts.to_parquet(OUT / "lm_sentence_counts.parquet", index=False)
    meta = events[["sector", "ticker", "accession", "fiscal_period_end", "event_day0"]]
    doc, sent_scored = meta.copy(), {}
    for v, s in sents.items():
        j = s.merge(scores, on="text_id", how="left").merge(lm_counts, on="text_id", how="left")
        sent_scored[v] = j
        d = fb.aggregate(j).merge(lm.aggregate(j).drop(columns="ticker"), on="accession")
        sfx = "" if v == "main" else "_sh"
        d = d.rename(columns={c: c + sfx for c in d.columns if c != "accession"})
        doc = doc.merge(d, on="accession", how="left")
    for col in ("fb_tone", "lm_tone", "fb_tone_sh", "lm_tone_sh"):
        doc[col.replace("_tone", "_tone_surprise")] = tone_surprise(doc, col)
    doc.to_csv(OUT / "document_scores.csv", index=False)

    L = ["# Stage 3: sentiment scores and sanity checks", "",
         "Generated by `scripts/13_stage3_report.py` with the rules frozen in `docs/stage3_scoring_spec.md`. "
         "**No price, return or CAR is read in Stage 3.** The only relationships computed are between the two "
         "sentiment methods and between the two text versions.", "",
         "## 1. Verification of the scoring run", ""]
    L += md(ver) + ["",
          f"FinBERT: `{fb.MODEL_ID}` at revision `{fb.MODEL_REVISION}`, float32 on CPU, `max_length` {fb.MAX_LENGTH}, batch {fb.BATCH_SIZE}. "
          f"Loughran-McDonald file: `{lmd.file_name}`.", ""]

    # ---- truncation --------------------------------------------------------------
    m = sent_scored["main"]
    L += ["## 2. Sentences truncated at 128 tokens", "",
          f"- Distinct sentences: {int(scores['truncated'].sum()):,} of {len(scores):,} (**{scores['truncated'].mean() * 100:.2f}%**).",
          f"- Sentence occurrences in the main version: {int(m['truncated'].sum()):,} of {len(m):,} (**{m['truncated'].mean() * 100:.2f}%**).",
          f"- Releases with at least one truncated sentence: {int((doc['fb_n_truncated'] > 0).sum())} of {len(doc)}; "
          f"mean share of a release's sentences that are truncated: {(doc['fb_n_truncated'] / doc['fb_n_sentences']).mean() * 100:.2f}%, "
          f"maximum {(doc['fb_n_truncated'] / doc['fb_n_sentences']).max() * 100:.1f}%.",
          f"- Token length of scored sentences: median {int(scores['n_tokens'].median())}, 95% {int(scores['n_tokens'].quantile(0.95))}, "
          f"maximum {int(scores['n_tokens'].max())}.", ""]
    tr = (doc.assign(share=doc["fb_n_truncated"] / doc["fb_n_sentences"]).groupby("ticker")["share"].mean() * 100).sort_values(ascending=False)
    L += ["Firms with the highest mean share of truncated sentences (%): " + ", ".join(f"{t} {v:.1f}" for t, v in tr.head(6).items()) + ".", ""]

    # ---- distributions --------------------------------------------------------------
    cols = ["fb_tone", "fb_tone_labels", "lm_tone", "lm_tone_per_word", "fb_tone_surprise", "lm_tone_surprise", "fb_tone_sh", "lm_tone_sh"]
    dist = pd.DataFrame([{"score": c, **describe(doc[c])} for c in cols])
    dist.to_csv(OUT / "score_distributions.csv", index=False)
    lab = m[["p_pos", "p_neg", "p_neu"]].to_numpy().argmax(1)
    L += ["## 3. Score distributions (1,316 releases)", "",
          "`_sh` is the version that keeps safe-harbor text. Tone surprise needs four prior quarters.", ""] + md(dist) + ["",
          f"- FinBERT sentence labels, main version: {np.mean(lab == 0) * 100:.1f}% positive, {np.mean(lab == 1) * 100:.1f}% negative, "
          f"{np.mean(lab == 2) * 100:.1f}% neutral.",
          f"- Loughran-McDonald, main version: {int(doc['lm_pos'].sum()):,} positive and {int(doc['lm_neg'].sum()):,} negative word hits; "
          f"{int(doc['lm_negated_pos'].sum()):,} positive words cancelled by the negation rule "
          f"({doc['lm_negated_pos'].sum() / (doc['lm_pos'].sum() + doc['lm_negated_pos'].sum()) * 100:.2f}% of positive hits).",
          f"- Releases with no Loughran-McDonald sentiment word (score missing): {int(doc['lm_tone'].isna().sum())}.",
          f"- Sentiment-word hits per release (positive + negative): minimum {int((doc['lm_pos'] + doc['lm_neg']).min())}, "
          f"5% {int((doc['lm_pos'] + doc['lm_neg']).quantile(0.05))}, median {int((doc['lm_pos'] + doc['lm_neg']).median())}. "
          f"Releases with fewer than 20 hits: {int(((doc['lm_pos'] + doc['lm_neg']) < 20).sum())}. "
          f"Releases where `lm_tone` is exactly +1 or -1 because every hit has the same sign: **{int((doc['lm_tone'].abs() == 1).sum())}** "
          f"({doc[doc['lm_tone'].abs() == 1]['ticker'].value_counts().to_dict()}). With few hits the ratio (pos - neg)/(pos + neg) is coarse; "
          f"`lm_tone_per_word` does not saturate (correlation with `lm_tone`: {doc['lm_tone'].corr(doc['lm_tone_per_word']):.3f}).",
          f"- Share of the variance that lies between firms: `fb_tone` {firm_share(doc, 'fb_tone') * 100:.0f}%, `lm_tone` {firm_share(doc, 'lm_tone') * 100:.0f}%. "
          "The rest is variation within a firm over time, which is what the headline specification with firm fixed effects uses.", "",
          "![Distributions of the two primary scores](../results/stage3/figures/score_distributions.png)", ""]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for ax, c, colr, ttl in ((axes[0], "fb_tone", BLUE, "FinBERT tone (fb_tone)"), (axes[1], "lm_tone", ORANGE, "Loughran-McDonald tone (lm_tone)")):
        ax.hist(doc[c].dropna(), bins=40, color=colr, edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_title(f"{ttl}: mean {doc[c].mean():.3f}, sd {doc[c].std():.3f}")
        ax.set_xlabel("document score")
    axes[0].set_ylabel("releases")
    fig.tight_layout(); fig.savefig(FIG / "score_distributions.png", dpi=150); plt.close(fig)

    yr = doc.assign(year=pd.to_datetime(doc["event_day0"]).dt.year).groupby("year").agg(
        releases=("accession", "size"), fb_tone=("fb_tone", "mean"), lm_tone=("lm_tone", "mean"),
        share_pos=("fb_share_pos", "mean"), share_neg=("fb_share_neg", "mean")).reset_index()
    L += ["Mean scores by year of release (a plausibility check: 2020 should look worse than its neighbours):", ""] + md(yr) + [""]
    sec = doc.groupby("sector").agg(releases=("accession", "size"), fb_tone=("fb_tone", "mean"), lm_tone=("lm_tone", "mean")).reset_index()
    L += ["Mean scores by sector:", ""] + md(sec) + [""]

    # ---- correlations ------------------------------------------------------------------
    def within(a: str, b: str) -> float:
        d = doc[["ticker", a, b]].dropna()
        da, db = d[a] - d.groupby("ticker")[a].transform("mean"), d[b] - d.groupby("ticker")[b].transform("mean")
        return float(da.corr(db))
    pairs = [("fb_tone", "lm_tone", "primary scores"), ("fb_tone_labels", "lm_tone", "FinBERT label version vs LM"),
             ("fb_tone", "lm_tone_per_word", "FinBERT vs LM per word"), ("fb_tone_surprise", "lm_tone_surprise", "tone surprises"),
             ("fb_tone_sh", "lm_tone_sh", "primary scores, safe-harbor text kept")]
    cor = pd.DataFrame([{"pair": lbl, "n": int(doc[[a, b]].dropna().shape[0]), "Pearson": doc[a].corr(doc[b]),
                         "Spearman": doc[a].corr(doc[b], method="spearman"), "within firm (Pearson)": within(a, b)} for a, b, lbl in pairs])
    cor.to_csv(OUT / "method_correlations.csv", index=False)
    per_firm = doc.groupby("ticker").apply(lambda g: g["fb_tone"].corr(g["lm_tone"]), include_groups=False).rename("corr").sort_values()
    per_firm.reset_index().to_csv(OUT / "method_correlation_by_firm.csv", index=False)
    L += ["## 4. Agreement between FinBERT and Loughran-McDonald", ""] + md(cor) + ["",
          "\"Within firm\" removes each firm's own mean from both scores first, so it measures whether the two methods move "
          "together over time for the same firm.", "",
          f"Correlation computed separately for each of the 55 firms: minimum {per_firm.min():.2f}, 25% {per_firm.quantile(0.25):.2f}, "
          f"median **{per_firm.median():.2f}**, 75% {per_firm.quantile(0.75):.2f}, maximum {per_firm.max():.2f}. "
          f"Firms with a negative correlation: {int((per_firm < 0).sum())}.", "",
          "Lowest five: " + ", ".join(f"{t} {v:.2f}" for t, v in per_firm.head(5).items()) + ". Highest five: "
          + ", ".join(f"{t} {v:.2f}" for t, v in per_firm.tail(5).items()) + ". Full list: `results/stage3/method_correlation_by_firm.csv`.", "",
          "![FinBERT against Loughran-McDonald](../results/stage3/figures/fb_vs_lm.png)", ""]
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.scatter(doc["lm_tone"], doc["fb_tone"], s=9, color=BLUE, alpha=0.45, linewidths=0)
    ax.axhline(0, color=INK2, linewidth=1); ax.axvline(0, color=INK2, linewidth=1)
    ax.set_xlabel("Loughran-McDonald tone"); ax.set_ylabel("FinBERT tone")
    ax.set_title(f"One point per release (n = {int(doc[['fb_tone', 'lm_tone']].dropna().shape[0])}, r = {doc['fb_tone'].corr(doc['lm_tone']):.2f})")
    fig.tight_layout(); fig.savefig(FIG / "fb_vs_lm.png", dpi=150); plt.close(fig)
    ver_cor = pd.DataFrame([{"method": lbl, "Pearson main vs with safe-harbor text": doc[a].corr(doc[b]),
                             "mean change (with - main)": (doc[b] - doc[a]).mean(), "largest absolute change": (doc[b] - doc[a]).abs().max()}
                            for a, b, lbl in (("fb_tone", "fb_tone_sh", "FinBERT"), ("lm_tone", "lm_tone_sh", "Loughran-McDonald"))])
    L += ["Effect of the safe-harbor removal on each method:", ""] + md(ver_cor) + [""]

    # ---- extremes ---------------------------------------------------------------------------
    doc["fb_z"] = (doc["fb_tone"] - doc["fb_tone"].mean()) / doc["fb_tone"].std()
    doc["lm_z"] = (doc["lm_tone"] - doc["lm_tone"].mean()) / doc["lm_tone"].std()
    doc["fb_pct"] = doc["fb_tone"].rank(pct=True) * 100
    doc["lm_pct"] = doc["lm_tone"].rank(pct=True) * 100
    excerpts: list[str] = []

    def table(rows: pd.DataFrame, title: str) -> list[str]:
        out, recs = [], []
        for r in rows.itertuples():
            g = m[m["accession"] == r.accession]
            recs.append({"firm": r.ticker, "fiscal period": r.fiscal_period_end, "day 0": r.event_day0,
                         "fb_tone": r.fb_tone, "FinBERT percentile": r.fb_pct, "lm_tone": r.lm_tone, "LM percentile": r.lm_pct,
                         "sentences": int(r.fb_n_sentences), "% positive": r.fb_share_pos * 100, "% negative": r.fb_share_neg * 100,
                         "LM positive words": int(r.lm_pos), "LM negative words": int(r.lm_neg),
                         "most frequent LM negative words": top_words(g["text"], lmd.negative),
                         "most frequent LM positive words": top_words(g["text"], lmd.positive)})
            net = g["p_pos"] - g["p_neg"]
            excerpts.append(f"\n===== [{title}] {r.ticker} period {r.fiscal_period_end} day0 {r.event_day0} | fb_tone {r.fb_tone:.3f} "
                            f"(pct {r.fb_pct:.0f}) | lm_tone {r.lm_tone:.3f} (pct {r.lm_pct:.0f}) | accession {r.accession}")
            for lbl, idx in (("most positive FinBERT sentences", net.nlargest(4).index), ("most negative FinBERT sentences", net.nsmallest(4).index)):
                excerpts.append(f"  -- {lbl}")
                excerpts.extend([f"     ({net.loc[i]:+.2f}) {g.loc[i, 'text'][:330]}" for i in idx])
            hit = g.assign(k=g["lm_neg"] - g["lm_pos"])
            hit = hit[hit["lm_neg"] > 0].nlargest(3, "k")
            excerpts.append("  -- sentences with the most LM negative words")
            excerpts.extend([f"     (neg {int(x.lm_neg)}, pos {int(x.lm_pos)}; FinBERT {x.p_pos - x.p_neg:+.2f}) {x.text[:330]}" for x in hit.itertuples()])
        out += md(pd.DataFrame(recs), "{:.2f}") + [""]
        return out

    d = doc.dropna(subset=["fb_tone", "lm_tone"])
    L += ["## 5. Most positive and most negative releases, per method", "",
          "Percentiles are ranks within the 1,316 releases (100 = most positive). The sentences behind each row are written "
          "to `data/interim/stage3_excerpts.txt`, which is git-ignored because it contains release text.", ""]
    for name, col in (("FinBERT", "fb_tone"), ("Loughran-McDonald", "lm_tone")):
        L += [f"### {name}: 5 most positive", ""] + table(d.nlargest(5, col), f"{name} most positive")
        L += [f"### {name}: 5 most negative", ""] + table(d.nsmallest(5, col), f"{name} most negative")

    d = d.assign(gap=d["fb_z"] - d["lm_z"])
    big = d.reindex(d["gap"].abs().sort_values(ascending=False).index).head(10)
    L += ["## 6. The 10 largest disagreements", "",
          "Each score is standardised over the 1,316 releases; the disagreement is the FinBERT z-score minus the "
          "Loughran-McDonald z-score. Positive gap: FinBERT reads the release as more positive than the dictionary does.", "",
          "Gap of each row, in order: " + ", ".join(f"{r.ticker} {r.fiscal_period_end} ({r.gap:+.2f})" for r in big.itertuples()) + ".", ""]
    L += table(big, "disagreement")
    L += [f"Direction of the 10: {int((big['gap'] > 0).sum())} with FinBERT more positive, {int((big['gap'] < 0).sum())} with the dictionary more positive. "
          f"Firms: {big['ticker'].value_counts().to_dict()}.", "",
          f"Across all releases the gap has standard deviation {d['gap'].std():.2f}; {int((d['gap'].abs() > 2).sum())} releases have an absolute gap above 2.", ""]
    gf = d.groupby("ticker")["gap"].mean().sort_values()
    L += ["Mean gap by firm, the five most negative and five most positive (a persistent gap is a difference in house style, "
          "which firm fixed effects absorb): " + ", ".join(f"{t} {v:+.2f}" for t, v in gf.head(5).items()) + " ... "
          + ", ".join(f"{t} {v:+.2f}" for t, v in gf.tail(5).items()) + ".", ""]

    # ---- tone surprise -------------------------------------------------------------------------
    L += ["## 7. Tone surprise coverage", "",
          f"- `fb_tone_surprise` available for **{int(doc['fb_tone_surprise'].notna().sum())}** of {len(doc)} releases; "
          f"`lm_tone_surprise` for **{int(doc['lm_tone_surprise'].notna().sum())}**.",
          f"- Releases without it: {int(doc['fb_tone_surprise'].isna().sum())} (each firm's first four releases, and those that follow a missing quarter).", "",
          "## 8. Files", "",
          "- `results/stage3/document_scores.csv`: one row per release, both methods, both text versions, tone surprises. No text.",
          "- `results/stage3/finbert_scores.parquet`: FinBERT probabilities per distinct sentence, keyed by a hash of the sentence. No text.",
          "- `results/stage3/lm_sentence_counts.parquet`: dictionary counts per distinct sentence. No text.", ""]

    (config.INTERIM / "stage3_excerpts.txt").write_text("\n".join(excerpts), encoding="utf-8")
    (config.DOCS / "stage3_scores_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
