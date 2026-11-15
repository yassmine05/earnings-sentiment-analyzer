"""Loughran-McDonald dictionary tone. Part of the frozen scoring specification (docs/stage3_scoring_spec.md).

The dictionary file is supplied by the user in data/external/ and is never committed or redistributed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

NEGATORS = frozenset({"NO", "NOT", "NONE", "NEITHER", "NEVER", "NOBODY"})
NEGATION_WINDOW = 3
_TOKEN = re.compile(r"[A-Za-z]+")


@dataclass(frozen=True)
class LMDictionary:
    positive: frozenset
    negative: frozenset
    vocabulary: frozenset
    file_name: str


def find_dictionary() -> Path:
    files = sorted(config.EXTERNAL.glob("Loughran-McDonald_MasterDictionary_*.csv"))
    if not files:
        raise FileNotFoundError(
            "Loughran-McDonald Master Dictionary CSV not found in data/external/. Download it yourself from "
            "https://sraf.nd.edu/loughranmcdonald-master-dictionary/ (it is not redistributed with this project)."
        )
    return files[-1]


def load_dictionary(path: Path | None = None) -> LMDictionary:
    path = path or find_dictionary()
    d = pd.read_csv(path, keep_default_na=False, usecols=["Word", "Negative", "Positive"])
    words = d["Word"].astype(str).str.upper()
    # A positive year means the word is in the list; 0 = never, a negative year = removed in that year.
    pos = frozenset(words[pd.to_numeric(d["Positive"]) > 0])
    neg = frozenset(words[pd.to_numeric(d["Negative"]) > 0])
    return LMDictionary(positive=pos, negative=neg, vocabulary=frozenset(words), file_name=path.name)


def score_sentence(sentence: str, lm: LMDictionary) -> dict:
    """Counts for one sentence. A positive word with a negator among the 3 preceding words is not counted
    as positive (and is not added to the negative count). Negation never crosses a sentence boundary."""
    toks = [t.upper() for t in _TOKEN.findall(sentence)]
    pos = neg = negated = in_vocab = 0
    for i, t in enumerate(toks):
        if t in lm.vocabulary:
            in_vocab += 1
        if t in lm.negative:
            neg += 1
        elif t in lm.positive:
            if any(p in NEGATORS for p in toks[max(0, i - NEGATION_WINDOW):i]):
                negated += 1
            else:
                pos += 1
    return {"lm_pos": pos, "lm_neg": neg, "lm_negated_pos": negated, "lm_words": in_vocab}


def score_sentences(sentences: pd.DataFrame, lm: LMDictionary) -> pd.DataFrame:
    """Per-sentence counts, aligned with the input rows (text_id, accession, sent_idx kept)."""
    counts = pd.DataFrame([score_sentence(s, lm) for s in sentences["text"]], index=sentences.index)
    return pd.concat([sentences[["text_id", "ticker", "accession", "sent_idx"]], counts], axis=1)


def aggregate(sent_scores: pd.DataFrame) -> pd.DataFrame:
    """Document tone. Primary: (pos - neg) / (pos + neg). Secondary: (pos - neg) / dictionary words."""
    g = sent_scores.groupby(["ticker", "accession"], sort=False)[["lm_pos", "lm_neg", "lm_negated_pos", "lm_words"]].sum()
    net = g["lm_pos"] - g["lm_neg"]
    hits = g["lm_pos"] + g["lm_neg"]
    g["lm_tone"] = np.where(hits > 0, net / hits.where(hits > 0, 1), np.nan)
    g["lm_tone_per_word"] = np.where(g["lm_words"] > 0, net / g["lm_words"].where(g["lm_words"] > 0, 1), np.nan)
    return g.reset_index()
