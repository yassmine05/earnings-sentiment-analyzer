"""Stage 3, step 2: 5-firm CPU pilot of the frozen scoring pipeline, with a time estimate for the full run.

Inputs   data/interim/sentences_main.parquet, sentences_with_sh.parquet, finbert_input.parquet,
         results/stage1/events_selected.csv, results/stage3/cleaning_stats.csv, the LM dictionary
Outputs  results/stage3/finbert_scores.parquet     sentence scores keyed by text hash (no text); the full run resumes from it
         results/stage3/pilot_*.csv                 pilot tables
         docs/stage3_pilot_report.md

No price or return is read. The only relationship computed is between the two sentiment scores.
"""
from __future__ import annotations

import json
import platform
import random
import time

import numpy as np
import pandas as pd

from earnings_sentiment import config
from earnings_sentiment import finbert_score as fb
from earnings_sentiment import lm
from earnings_sentiment.tone_surprise import REQUIRED_PRIOR, prior_quarter_counts

OUT = config.RESULTS / "stage3"
CACHE = str(config.ROOT / ".cache" / "hf")
PILOT_SEED, PILOT_FIRMS, THREADS, QUANT_SAMPLE = 42, 5, 4, 3000


def md(df: pd.DataFrame, fmt: str = "{:.2f}") -> list[str]:
    out = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join(["---"] * len(df.columns)) + "|"]
    for row in df.itertuples(index=False):
        out.append("| " + " | ".join(
            ("" if np.isnan(v) else fmt.format(v)) if isinstance(v, (float, np.floating)) else str(v) for v in row) + " |")
    return out


def padded_tokens(lengths: np.ndarray, batch: int = fb.BATCH_SIZE) -> int:
    """Tokens actually processed: each length-sorted batch is padded to its longest (truncated) member."""
    l = np.sort(np.minimum(lengths, fb.MAX_LENGTH))
    return int(sum(len(l[i:i + batch]) * l[i:i + batch].max() for i in range(0, len(l), batch)))


def timed(tok, model, texts, lengths):
    t = time.perf_counter()
    p = fb.score_texts(tok, model, texts, lengths)
    return p, time.perf_counter() - t


def main() -> None:
    import torch

    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    stats = pd.read_csv(OUT / "cleaning_stats.csv")
    main_s = pd.read_parquet(config.INTERIM / "sentences_main.parquet")
    with_sh = pd.read_parquet(config.INTERIM / "sentences_with_sh.parquet")
    uniq = pd.read_parquet(config.INTERIM / "finbert_input.parquet")

    pilot = sorted(random.Random(PILOT_SEED).sample(sorted(events["ticker"].unique()), PILOT_FIRMS))
    pm = main_s[main_s["ticker"].isin(pilot)]
    pilot_input = uniq[uniq["text_id"].isin(set(pm["text_id"]))].reset_index(drop=True)
    pilot_path = config.INTERIM / "pilot_input.parquet"
    pilot_input.to_parquet(pilot_path, index=False)

    # ---- token statistics for the whole sample (tokenizer only, no model) ----
    tok, model = fb.load("cpu", cache_dir=CACHE, threads=THREADS)
    all_len = fb.token_lengths(tok, uniq["text"].tolist())
    uniq["n_tokens"] = all_len

    # ---- pilot run, float32, resumable output ---------------------------------
    scores_path = OUT / "finbert_scores.parquet"
    already = pd.read_parquet(scores_path) if scores_path.exists() else pd.DataFrame(columns=["text_id"])
    todo = pilot_input[~pilot_input["text_id"].isin(set(already["text_id"]))]
    meta_path = OUT / "pilot_timing.json"
    if len(todo):
        lengths = fb.token_lengths(tok, todo["text"].tolist())
        probs, secs = timed(tok, model, todo["text"].tolist(), lengths)
        part = todo[["text_id"]].copy()
        part["n_tokens"], part["truncated"] = lengths, lengths > fb.MAX_LENGTH
        part[["p_pos", "p_neg", "p_neu"]] = probs
        pd.concat([already, part], ignore_index=True).to_parquet(scores_path, index=False)
        timing = {"sentences": len(todo), "seconds": secs, "padded_tokens": padded_tokens(lengths)}
        # ---- int8 dynamic quantization on a fixed subsample: speed and agreement ----
        sub = todo.sample(min(QUANT_SAMPLE, len(todo)), random_state=PILOT_SEED)
        sub_len = fb.token_lengths(tok, sub["text"].tolist())
        p32, s32 = timed(tok, model, sub["text"].tolist(), sub_len)
        _, qmodel = fb.load("cpu", quantize=True, cache_dir=CACHE, threads=THREADS)
        p8, s8 = timed(tok, qmodel, sub["text"].tolist(), sub_len)
        net32, net8 = p32[:, 0] - p32[:, 1], p8[:, 0] - p8[:, 1]
        timing.update({
            "quant_sample": len(sub), "float32_seconds_sample": s32, "int8_seconds_sample": s8,
            "label_agreement": float((p32.argmax(1) == p8.argmax(1)).mean()),
            "net_score_corr": float(np.corrcoef(net32, net8)[0, 1]),
            "net_score_mean_abs_diff": float(np.abs(net32 - net8).mean()),
            "net_score_max_abs_diff": float(np.abs(net32 - net8).max()),
            "torch": torch.__version__, "threads": THREADS, "processor": platform.processor(),
        })
        meta_path.write_text(json.dumps(timing, indent=2))
    timing = json.loads(meta_path.read_text())

    # ---- document scores for the pilot firms ------------------------------------
    scores = pd.read_parquet(scores_path)
    sent = pm.merge(scores, on="text_id", how="left")
    assert sent["p_pos"].notna().all(), "pilot sentences without a FinBERT score"
    fb_doc = fb.aggregate(sent)
    lmd = lm.load_dictionary()
    lm_doc = lm.aggregate(lm.score_sentences(pm, lmd))
    doc = lm_doc.merge(fb_doc, on="accession").merge(
        events[["accession", "fiscal_period_end", "accepted_et_date"]], on="accession")
    doc.to_csv(OUT / "pilot_document_scores.csv", index=False)

    # ---- estimate for the full run -----------------------------------------------
    rest = uniq[~uniq["text_id"].isin(set(scores["text_id"]))]
    rate_sent = timing["sentences"] / timing["seconds"]
    rate_tok = timing["padded_tokens"] / timing["seconds"]
    rest_tokens = padded_tokens(rest["n_tokens"].to_numpy())
    est_hours = rest_tokens / rate_tok / 3600
    int8_speedup = timing["float32_seconds_sample"] / timing["int8_seconds_sample"]

    # ---- tone surprise coverage (needs no score) ------------------------------------
    prior = prior_quarter_counts(events)
    lose = events[prior < REQUIRED_PRIOR]
    first_four = lose.groupby("ticker").cumcount() < 4

    L = ["# Stage 3: pilot report (5 firms, CPU)", "",
         "Generated by `scripts/10_pilot.py` with the scoring rules frozen in `docs/stage3_scoring_spec.md` "
         "(version 1.0). No price or return was read; the only relationship computed is between the two sentiment scores.", "",
         "## 1. What each cleaning step removes (all 1,316 events)", "",
         "Words are alphabetic words. Means and medians are per event.", ""]
    steps = [("In the exhibit, tables included", "words_with_tables"), ("Removed with tables", "words_removed_tables"),
             ("Removed as safe-harbor text", "words_removed_safe_harbor"),
             ("Removed by the sentence filter", "words_removed_sentence_filter"),
             ("Scored, main version", "words_scored_main"), ("Scored, version with safe-harbor text", "words_scored_with_sh")]
    base = stats["words_with_tables"].mean()
    L += md(pd.DataFrame([{"step": n, "mean words": stats[c].mean(), "median words": stats[c].median(),
                           "share of exhibit (%)": stats[c].mean() / base * 100} for n, c in steps]), "{:.0f}") + [""]
    L += [f"- Sentences per event, main version: mean {stats['n_sentences_main'].mean():.0f}, median "
          f"{stats['n_sentences_main'].median():.0f}, minimum {stats['n_sentences_main'].min()}, maximum {stats['n_sentences_main'].max()}.",
          f"- Events where the safe-harbor rule found nothing to remove: {int((stats['pieces_removed_safe_harbor'] == 0).sum())} "
          f"({stats[stats['pieces_removed_safe_harbor'] == 0]['ticker'].value_counts().to_dict()}).",
          f"- Sentences in total: {len(main_s):,} in the main version and {len(with_sh):,} with safe-harbor text. "
          f"Distinct sentences across both: **{len(uniq):,}**, because boilerplate repeats from quarter to quarter. "
          "Each distinct sentence is scored once.", ""]

    q = pd.Series(all_len).quantile([0.5, 0.9, 0.95, 0.99]).round(0).astype(int)
    L += ["## 2. Token lengths (all distinct sentences)", "",
          f"- Tokens per sentence including the two special tokens: median {q[0.5]}, 90% {q[0.9]}, 95% {q[0.95]}, 99% {q[0.99]}, maximum {int(all_len.max())}.",
          f"- Sentences longer than `max_length` = {fb.MAX_LENGTH} and therefore truncated: **{int((all_len > fb.MAX_LENGTH).sum()):,}** "
          f"({(all_len > fb.MAX_LENGTH).mean() * 100:.2f}%).", ""]

    L += ["## 3. Pilot run", "",
          f"- Firms (random draw with seed {PILOT_SEED}): **{', '.join(pilot)}**; {pm['accession'].nunique()} releases, "
          f"{len(pm):,} sentences, {len(pilot_input):,} distinct.",
          f"- Machine: {timing['processor']}, {timing['threads']} threads, torch {timing['torch']}, float32, batch {fb.BATCH_SIZE}, "
          f"model revision `{fb.MODEL_REVISION[:12]}`.",
          f"- Scored {timing['sentences']:,} sentences in {timing['seconds'] / 60:.1f} minutes: **{rate_sent:.1f} sentences per second** "
          f"({rate_tok:,.0f} padded tokens per second).", "",
          "### int8 quantization (optional CPU fallback)", "",
          f"Same {timing['quant_sample']:,} pilot sentences scored with float32 and with int8 dynamic quantization:", "",
          f"- Time: float32 {timing['float32_seconds_sample']:.0f} s, int8 {timing['int8_seconds_sample']:.0f} s, a speed-up of **{int8_speedup:.2f}x**.",
          f"- Same label for {timing['label_agreement'] * 100:.1f}% of sentences; correlation of sentence scores {timing['net_score_corr']:.4f}; "
          f"mean absolute difference {timing['net_score_mean_abs_diff']:.3f}, maximum {timing['net_score_max_abs_diff']:.2f} (scores run from -1 to 1).", ""]

    L += ["## 4. Estimated time for the full sample", "",
          f"- Distinct sentences still to score: **{len(rest):,}** of {len(uniq):,}.",
          f"- CPU, float32: about **{est_hours:.1f} hours** on this laptop, from padded tokens "
          f"({rest_tokens:,} to process at {rate_tok:,.0f} per second). A plain sentence count gives {len(rest) / rate_sent / 3600:.1f} hours.",
          f"- CPU, int8: about {est_hours / int8_speedup:.1f} hours, at the cost of the differences shown above.",
          "- The run is resumable: it saves every 100 batches and skips sentences already scored.", ""]

    L += ["## 5. Pilot scores (5 firms)", "", "Per firm, document scores of the main version:", ""]
    pf = doc.groupby("ticker").agg(releases=("accession", "size"), fb_tone_mean=("fb_tone", "mean"), fb_tone_std=("fb_tone", "std"),
                                   lm_tone_mean=("lm_tone", "mean"), lm_tone_std=("lm_tone", "std"),
                                   share_pos=("fb_share_pos", "mean"), share_neg=("fb_share_neg", "mean"),
                                   sentences=("fb_n_sentences", "mean")).reset_index()
    L += md(pf, "{:.3f}") + [""]
    within = doc.groupby("ticker").apply(lambda g: g["fb_tone"].corr(g["lm_tone"]), include_groups=False)
    L += [f"- Correlation between `fb_tone` and `lm_tone` across the {len(doc)} pilot releases: **{doc['fb_tone'].corr(doc['lm_tone']):.2f}** "
          f"(Spearman {doc['fb_tone'].corr(doc['lm_tone'], method='spearman'):.2f}).",
          "- Within each firm: " + ", ".join(f"{t} {v:.2f}" for t, v in within.items()) + ".",
          f"- Sentence labels in the pilot: {sent[['p_pos', 'p_neg', 'p_neu']].to_numpy().argmax(1).tolist().count(0) / len(sent) * 100:.1f}% positive, "
          f"{sent[['p_pos', 'p_neg', 'p_neu']].to_numpy().argmax(1).tolist().count(1) / len(sent) * 100:.1f}% negative, the rest neutral.",
          f"- Releases with no Loughran-McDonald sentiment word: {int(doc['lm_tone'].isna().sum())}.", "",
          "The full sanity checks (distributions, most positive and most negative documents, largest disagreements "
          "with excerpts) need the full run and come after it.", ""]

    L += ["## 6. Tone surprise: events that lose their score", "",
          f"Rule: the firm's previous four fiscal quarters must all be present. This depends only on the event list, so it is known now.", "",
          f"- Events without four prior quarters: **{len(lose)}** of {len(events)} ({len(lose) / len(events) * 100:.1f}%).",
          f"- Of these, {int(first_four.sum())} are each firm's first four releases in the sample (2019), and "
          f"{int((~first_four).sum())} follow a missing quarter.",
          f"- Events that keep a tone surprise: **{len(events) - len(lose)}**.", ""]
    by_year = events.assign(year=pd.to_datetime(events["accepted_et_date"]).dt.year, lost=prior < REQUIRED_PRIOR)
    L += md(by_year.groupby("year").agg(events=("lost", "size"), lose_tone_surprise=("lost", "sum")).reset_index()) + [""]
    gap = lose[~first_four.values]
    if len(gap):
        L += ["Events after the first four that still lose it (they follow a gap in the firm's history):", "",
              ", ".join(f"{r.ticker} {r.fiscal_period_end}" for r in gap.itertuples()) + ".", ""]

    (config.DOCS / "stage3_pilot_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
