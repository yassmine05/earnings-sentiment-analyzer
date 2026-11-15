"""FinBERT sentence scoring. Part of the frozen scoring specification (docs/stage3_scoring_spec.md).

This file is self-contained on purpose (it imports nothing else from the project), so the same file
runs locally on CPU and on Kaggle with a GPU:

    python finbert_score.py --sentences finbert_input.parquet --out finbert_scores.parquet

Input   a parquet file with columns text_id, text (one row per distinct sentence)
Output  a parquet file with text_id, n_tokens, truncated, p_pos, p_neg, p_neu (no text).
        The run is resumable: sentences already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_ID = "ProsusAI/finbert"
MODEL_REVISION = "0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031"   # exact commit on the Hugging Face hub
LABELS = ("positive", "negative", "neutral")                   # must equal the model's id2label order
MAX_LENGTH = 128        # tokens, including [CLS] and [SEP]; longer sentences are truncated and counted
BATCH_SIZE = 32
SAVE_EVERY_BATCHES = 100


def load(device: str = "cpu", quantize: bool = False, cache_dir: str | None = None, threads: int | None = None):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if threads:
        torch.set_num_threads(threads)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_dir, use_safetensors=True
    )
    got = tuple(model.config.id2label[i].lower() for i in range(3))
    if got != LABELS:
        raise RuntimeError(f"unexpected label order {got}; expected {LABELS}")
    model.eval()
    if quantize:
        if device != "cpu":
            raise ValueError("int8 dynamic quantization is CPU-only")
        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    model.to(device)
    return tok, model


def token_lengths(tok, texts: list[str]) -> np.ndarray:
    """Untruncated token count of each sentence, including the two special tokens."""
    enc = tok(texts, add_special_tokens=True, truncation=False, padding=False,
              return_attention_mask=False, return_token_type_ids=False, verbose=False)
    return np.array([len(x) for x in enc["input_ids"]], dtype=np.int32)


def score_texts(tok, model, texts: list[str], lengths: np.ndarray | None = None, device: str = "cpu",
                batch_size: int = BATCH_SIZE, on_batch=None) -> np.ndarray:
    """Class probabilities (positive, negative, neutral) for each text, in the input order.

    Sentences are sorted by token length so each batch is padded only to its own longest member.
    Padding does not change a sentence's result beyond floating-point noise.
    """
    import torch

    if lengths is None:
        lengths = token_lengths(tok, texts)
    order = np.argsort(lengths, kind="stable")
    probs = np.zeros((len(texts), 3), dtype=np.float32)
    with torch.inference_mode():
        for b, start in enumerate(range(0, len(order), batch_size)):
            idx = order[start:start + batch_size]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
            if on_batch is not None:
                on_batch(b, idx, probs)
    return probs


def run(sentences_path: str, out_path: str, device: str = "auto", quantize: bool = False,
        cache_dir: str | None = None, threads: int | None = None, limit: int | None = None) -> dict:
    import torch

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    sents = pd.read_parquet(sentences_path)
    out = Path(out_path)
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["text_id"])
    todo = sents[~sents["text_id"].isin(set(done["text_id"]))].reset_index(drop=True)
    if limit:
        todo = todo.iloc[:limit]
    print(f"sentences: {len(sents)} | already scored: {len(done)} | to score now: {len(todo)} | device: {device}", flush=True)
    if todo.empty:
        return {"scored": 0}
    tok, model = load(device, quantize, cache_dir, threads)
    texts = todo["text"].tolist()
    lengths = token_lengths(tok, texts)
    t0 = time.perf_counter()
    scored_mask = np.zeros(len(todo), dtype=bool)

    def save(probs):
        part = todo.loc[scored_mask, ["text_id"]].copy()
        part["n_tokens"] = lengths[scored_mask]
        part["truncated"] = lengths[scored_mask] > MAX_LENGTH
        part[["p_pos", "p_neg", "p_neu"]] = probs[scored_mask]
        pd.concat([done, part], ignore_index=True).to_parquet(out, index=False)

    def on_batch(b, idx, probs):
        scored_mask[idx] = True
        if (b + 1) % SAVE_EVERY_BATCHES == 0:
            save(probs)
            n = int(scored_mask.sum())
            el = time.perf_counter() - t0
            print(f"  {n}/{len(todo)} sentences | {n / el:.1f} per second | about {(len(todo) - n) / (n / el) / 60:.0f} min left", flush=True)

    probs = score_texts(tok, model, texts, lengths, device, on_batch=on_batch)
    save(probs)
    elapsed = time.perf_counter() - t0
    meta = {"model": MODEL_ID, "revision": MODEL_REVISION, "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "device": device, "quantized_int8": quantize, "sentences_scored_this_run": len(todo),
            "seconds": round(elapsed, 1), "sentences_per_second": round(len(todo) / elapsed, 2),
            "torch": torch.__version__}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)
    return meta


def aggregate(sent_scores: pd.DataFrame) -> pd.DataFrame:
    """Document scores from one row per (accession, sentence). Primary: mean of P(positive) - P(negative)."""
    s = sent_scores.copy()
    s["net"] = s["p_pos"] - s["p_neg"]
    label = s[["p_pos", "p_neg", "p_neu"]].to_numpy().argmax(axis=1)
    s["is_pos"], s["is_neg"] = label == 0, label == 1
    g = s.groupby("accession", sort=False).agg(
        fb_tone=("net", "mean"), fb_share_pos=("is_pos", "mean"), fb_share_neg=("is_neg", "mean"),
        fb_n_sentences=("net", "size"), fb_n_truncated=("truncated", "sum"),
    )
    g["fb_tone_labels"] = g["fb_share_pos"] - g["fb_share_neg"]
    return g.reset_index()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Score sentences with the pinned FinBERT revision.")
    ap.add_argument("--sentences", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--quantize", action="store_true", help="int8 dynamic quantization (CPU fallback only)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    run(a.sentences, a.out, a.device, a.quantize, a.cache_dir, a.threads, a.limit)
