"""Stage 3, step 3: full FinBERT run on CPU (float32, as frozen in docs/stage3_scoring_spec.md).

Run it from the project folder:

    .\\.venv\\Scripts\\python.exe scripts\\12_score_finbert.py

- Resumable: scores are saved every 100 batches; running the same command again continues where it stopped.
- Progress is printed and appended to logs/finbert_full_run.log.
- At the end it checks that every distinct sentence has exactly one valid score and writes
  results/stage3/finbert_run_complete.json. Exit code 0 means complete, 1 means incomplete.

Inputs   data/interim/finbert_input.parquet      (text_id, text), git-ignored
Outputs  results/stage3/finbert_scores.parquet   (text_id, n_tokens, truncated, p_pos, p_neg, p_neu), no text
No price or return is read.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from earnings_sentiment import config
from earnings_sentiment import finbert_score as fb

DEFAULT_IN = config.INTERIM / "finbert_input.parquet"
DEFAULT_OUT = config.RESULTS / "stage3" / "finbert_scores.parquet"
DEFAULT_LOG = config.ROOT / "logs" / "finbert_full_run.log"
CACHE = config.ROOT / ".cache" / "hf"
THREADS = 4


class Tee:
    """Write everything printed to the terminal and to the log file, with a timestamp per line in the log."""

    def __init__(self, stream, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream, self.file, self.at_line_start = stream, open(path, "a", encoding="utf-8"), True

    def write(self, text: str) -> int:
        self.stream.write(text)
        for chunk in text.splitlines(keepends=True):
            if self.at_line_start:
                self.file.write(dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S  "))
            self.file.write(chunk)
            self.at_line_start = chunk.endswith("\n")
        self.file.flush()
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


def verify(input_path: Path, out_path: Path) -> dict:
    """Every distinct sentence must have exactly one score with valid probabilities."""
    want = pd.read_parquet(input_path, columns=["text_id"])["text_id"]
    got = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame(columns=["text_id", "p_pos", "p_neg", "p_neu", "truncated"])
    probs = got[["p_pos", "p_neg", "p_neu"]].to_numpy(dtype=float) if len(got) else np.zeros((0, 3))
    missing = int((~want.isin(set(got["text_id"]))).sum())
    report = {
        "distinct_sentences": int(len(want)),
        "scored": int(got["text_id"].isin(set(want)).sum()),
        "missing": missing,
        "duplicate_scores": int(got["text_id"].duplicated().sum()),
        "scores_not_in_input": int((~got["text_id"].isin(set(want))).sum()),
        "invalid_probabilities": int((np.isnan(probs).any(axis=1) | (np.abs(probs.sum(axis=1) - 1) > 1e-3)
                                      | (probs < 0).any(axis=1)).sum()) if len(got) else 0,
        "truncated_at_max_length": int(got["truncated"].sum()) if len(got) else 0,
        "model": fb.MODEL_ID, "revision": fb.MODEL_REVISION, "max_length": fb.MAX_LENGTH, "batch_size": fb.BATCH_SIZE,
        "checked_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    report["complete"] = bool(missing == 0 and report["duplicate_scores"] == 0 and report["invalid_probabilities"] == 0
                              and report["distinct_sentences"] > 0)
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Full FinBERT run (CPU, float32) with a completeness check.")
    ap.add_argument("--input", default=str(DEFAULT_IN))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--log", default=str(DEFAULT_LOG))
    ap.add_argument("--limit", type=int, default=None, help="score at most this many sentences (for a quick test)")
    ap.add_argument("--check-only", action="store_true", help="do not score; only check whether the run is complete")
    a = ap.parse_args()
    sys.stdout = Tee(sys.stdout, Path(a.log))
    sys.stderr = Tee(sys.stderr, Path(a.log))
    inp, out = Path(a.input), Path(a.out)
    print(f"=== FinBERT run started | input {inp.name} | output {out} | revision {fb.MODEL_REVISION[:12]} | float32 | {THREADS} threads")
    if not a.check_only:
        if (CACHE / "models--ProsusAI--finbert").exists():
            os.environ.setdefault("HF_HUB_OFFLINE", "1")   # the pinned model is already on disk: no network needed
        fb.run(str(inp), str(out), device="cpu", quantize=False, cache_dir=str(CACHE), threads=THREADS, limit=a.limit)
    report = verify(inp, out)
    if out == DEFAULT_OUT:
        (out.parent / "finbert_run_complete.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    if report["complete"]:
        print(f"RUN COMPLETE: all {report['distinct_sentences']:,} distinct sentences have a score.")
        return 0
    print(f"RUN INCOMPLETE: {report['missing']:,} sentences have no score yet. Run the same command again to continue.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
