"""Stage 3, step 1: clean every kept exhibit and write the sentence files that both scoring methods read.

Inputs   results/stage1/events_selected.csv, cached exhibits in data/raw/edgar
Outputs  data/interim/sentences_main.parquet       safe-harbor text removed (primary version)   [git-ignored: text]
         data/interim/sentences_with_sh.parquet    safe-harbor text kept (robustness version)   [git-ignored: text]
         data/interim/finbert_input.parquet        every distinct sentence once (text_id, text) [git-ignored: text]
         data/interim/text_tablefree/<accession>.txt  cache of the table-free text              [git-ignored: text]
         results/stage3/cleaning_stats.csv         word counts per event and per cleaning step (no text)

No prices or returns are read here.
"""
from __future__ import annotations

import hashlib

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.edgar import EdgarClient
from earnings_sentiment.scoring_text import SPEC_VERSION, count_words, prepare
from earnings_sentiment.textclean import html_to_text

OUT = config.RESULTS / "stage3"
TXT = config.INTERIM / "text_tablefree"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TXT.mkdir(parents=True, exist_ok=True)
    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    client = EdgarClient()
    stats, rows = [], {"main": [], "with_sh": []}
    for k, e in enumerate(events.itertuples(), 1):
        cache = TXT / f"{e.accession}.txt"
        full_cache = TXT / f"{e.accession}.full_words"
        if cache.exists() and full_cache.exists():
            text = cache.read_text(encoding="utf-8")
            words_full = int(full_cache.read_text())
        else:
            html = client.get_text(e.exhibit_url)          # served from the on-disk EDGAR cache
            words_full = count_words(html_to_text(html, drop_tables=False))
            text = html_to_text(html, drop_tables=True)
            cache.write_text(text, encoding="utf-8")
            full_cache.write_text(str(words_full))
        rec = {"ticker": e.ticker, "accession": e.accession, "words_with_tables": words_full}
        for version, flag in (("main", True), ("with_sh", False)):
            p = prepare(text, remove_safe_harbor_text=flag)
            for i, s in enumerate(p["sentences"]):
                rows[version].append({
                    "ticker": e.ticker, "accession": e.accession, "sent_idx": i, "text": s,
                    "text_id": hashlib.sha1(s.encode("utf-8")).hexdigest()[:20],
                })
            if version == "main":
                rec.update({
                    "words_table_free": p["words_paragraphs"],
                    "words_removed_tables": words_full - p["words_paragraphs"],
                    "words_removed_safe_harbor": p["words_safe_harbor_removed"],
                    "pieces_removed_safe_harbor": p["pieces_safe_harbor_removed"],
                    "words_removed_sentence_filter": p["words_sentence_filter_removed"],
                    "words_scored_main": p["words_scored"], "n_sentences_main": p["n_sentences"],
                })
            else:
                rec.update({"words_scored_with_sh": p["words_scored"], "n_sentences_with_sh": p["n_sentences"]})
        stats.append(rec)
        if k % 100 == 0:
            print(f"{k}/{len(events)}", flush=True)
    stats = pd.DataFrame(stats)
    stats["spec_version"] = SPEC_VERSION
    stats.to_csv(OUT / "cleaning_stats.csv", index=False)
    frames = {v: pd.DataFrame(r) for v, r in rows.items()}
    for version, df in frames.items():
        df.to_parquet(config.INTERIM / f"sentences_{version}.parquet", index=False)
    # Each distinct sentence is scored once, whichever version or document it appears in.
    uniq = pd.concat(frames.values())[["text_id", "text"]].drop_duplicates("text_id").reset_index(drop=True)
    uniq.to_parquet(config.INTERIM / "finbert_input.parquet", index=False)
    print("distinct sentences to score with FinBERT:", len(uniq))
    print("events:", len(stats), "| sentences main:", len(rows["main"]), "| with safe harbor:", len(rows["with_sh"]))
    print(stats.drop(columns=["ticker", "accession", "spec_version"]).describe().round(0).T[["mean", "50%", "min", "max"]].to_string())
    print("events with no safe-harbor paragraph found:", int((stats["pieces_removed_safe_harbor"] == 0).sum()))


if __name__ == "__main__":
    main()
