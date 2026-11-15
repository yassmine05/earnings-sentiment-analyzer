"""Stage 3 audit: does the frozen safe-harbor rule miss disclaimers because of how the phrase is written?

For every kept exhibit, the RAW HTML is searched with a deliberately loose pattern (entities decoded, tags
replaced by spaces, every Unicode hyphen/dash, soft hyphen, zero-width and non-breaking character normalised,
white space collapsed). The result is compared with what the frozen rule actually removed.

Inputs   results/stage1/events_selected.csv, results/stage3/cleaning_stats.csv, cached exhibits, table-free text
Outputs  results/stage3/safe_harbor_audit.csv            one row per event, counts only (no text)
         data/interim/safe_harbor_audit_excerpts.txt     passages for manual reading (git-ignored: text)

This script changes no rule and no score.
"""
from __future__ import annotations

import html as htmllib
import re
import sys

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.edgar import EdgarClient
from earnings_sentiment.scoring_text import TRIGGER, paragraphs, remove_safe_harbor

OUT = config.RESULTS / "stage3"
HYPHENS = "‐‑‒–—―−﹘﹣－­"
INVISIBLE = "​‌‍⁠﻿"
LOOSE = {
    "forward looking (any spelling)": re.compile(r"forward[\s\-]*looking", re.I),
    "safe harbo(u)r": re.compile(r"safe[\s\-]*harbou?r", re.I),
    "Private Securities Litigation Reform Act": re.compile(r"private\s+securities\s+litigation", re.I),
    "cautionary": re.compile(r"cautionary", re.I),
}
# spellings the frozen TRIGGER regex cannot see even though the phrase is there
VARIANTS = {
    "Unicode hyphen in forward-looking": re.compile(rf"forward\s*[{HYPHENS}]\s*looking", re.I),
    "no hyphen or space (forwardlooking)": re.compile(r"forwardlooking", re.I),
    "safe harbour (British spelling)": re.compile(r"safe[\s\-]*harbour", re.I),
}


def loose_text(raw_html: str, normalise_hyphens: bool = True) -> str:
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", raw_html)
    t = re.sub(r"<[^>]+>", " ", t)
    t = htmllib.unescape(htmllib.unescape(t))
    t = t.replace("\xa0", " ")
    for ch in INVISIBLE:
        t = t.replace(ch, "")
    if normalise_hyphens:
        for ch in HYPHENS:
            t = t.replace(ch, "-")
    return re.sub(r"\s+", " ", t)


def main() -> None:
    events = pd.read_csv(config.RESULTS / "stage1" / "events_selected.csv")
    stats = pd.read_csv(OUT / "cleaning_stats.csv").set_index("accession")
    client = EdgarClient()
    rows, excerpts = [], []
    for e in events.itertuples():
        raw = client.get_text(e.exhibit_url)
        loose = loose_text(raw)
        keep_hyph = loose_text(raw, normalise_hyphens=False)
        tf = (config.INTERIM / "text_tablefree" / f"{e.accession}.txt").read_text(encoding="utf-8")
        pars = paragraphs(tf)
        _, removed = remove_safe_harbor(pars)
        rec = {"ticker": e.ticker, "accession": e.accession, "accepted_et_date": e.accepted_et_date,
               "pieces_removed": len(removed), "words_removed": int(stats.loc[e.accession, "words_removed_safe_harbor"]),
               "strict_trigger_in_table_free_text": bool(TRIGGER.search(tf)),
               "strict_trigger_in_raw": bool(TRIGGER.search(loose))}
        for name, rx in LOOSE.items():
            rec[f"raw: {name}"] = len(rx.findall(loose))
        for name, rx in VARIANTS.items():
            rec[f"variant: {name}"] = len(rx.findall(keep_hyph))
        rec["any_loose_hit"] = any(rec[f"raw: {n}"] > 0 for n in LOOSE)
        # forward-looking written with a following word the strict rule does not list
        rec["forward_looking_other_noun"] = len(re.findall(r"forward-?\s*looking(?!\s+(statement|information))", loose, re.I))
        rows.append(rec)
        if rec["pieces_removed"] == 0:
            hits = [m.start() for rx in LOOSE.values() for m in rx.finditer(loose)]
            excerpts.append(f"\n===== {e.ticker} {e.accession} {e.accepted_et_date} | loose hits in raw HTML: {len(hits)}")
            for h in sorted(hits)[:4]:
                excerpts.append("   ..." + loose[max(0, h - 300): h + 500] + "...")
            if not hits:
                excerpts.append("   (no mention of forward-looking statements, safe harbor, the Reform Act or 'cautionary' anywhere in the raw HTML)")
                excerpts.append("   last 600 characters of the exhibit: ..." + loose[-600:])
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "safe_harbor_audit.csv", index=False)
    (config.INTERIM / "safe_harbor_audit_excerpts.txt").write_text("\n".join(excerpts), encoding="utf-8")

    n = len(a)
    none = a[a["pieces_removed"] == 0]
    print(f"exhibits: {n}")
    print(f"nothing removed by the rule: {len(none)}  by firm: {none['ticker'].value_counts().to_dict()}")
    print(f"  of these, raw HTML contains ANY loose variant: {int(none['any_loose_hit'].sum())}  by firm: {none[none['any_loose_hit']]['ticker'].value_counts().to_dict()}")
    print(f"  of these, raw HTML contains none: {int((~none['any_loose_hit']).sum())}  by firm: {none[~none['any_loose_hit']]['ticker'].value_counts().to_dict()}")
    print("\nvariant spellings in the raw HTML (exhibits containing at least one):")
    for name in VARIANTS:
        c = a[a[f"variant: {name}"] > 0]
        print(f"  {name}: {len(c)} exhibits; of which nothing removed: {int((c['pieces_removed'] == 0).sum())}; by firm: {c['ticker'].value_counts().to_dict()}")
    print("\nloose pattern counts (exhibits with at least one hit):")
    for name in LOOSE:
        print(f"  {name}: {int((a[f'raw: {name}'] > 0).sum())}")
    miss = a[a["any_loose_hit"] & ~a["strict_trigger_in_table_free_text"]]
    print(f"\nloose hit in raw HTML but strict trigger absent from the table-free text: {len(miss)}  by firm: {miss['ticker'].value_counts().to_dict()}")
    print(f"  of these, strict trigger present in raw HTML (so the phrase sat in a removed table): {int(miss['strict_trigger_in_raw'].sum())}")
    partial = a[(a["pieces_removed"] > 0) & (a["words_removed"] < 60)]
    print(f"\nsomething removed but under 60 words: {len(partial)}  by firm: {partial['ticker'].value_counts().to_dict()}")
    fl_other = a[(a["forward_looking_other_noun"] > 0)]
    print(f"exhibits where 'forward-looking' is followed by a word other than statement/information at least once: {len(fl_other)}")
    print(f"  of which nothing removed: {int((fl_other['pieces_removed'] == 0).sum())}")
    # ---- the other direction: disclaimer wording that is still in the scored sentences ----
    s = pd.read_parquet(config.INTERIM / "sentences_main.parquet")
    norm = s["text"].str.replace(f"[{HYPHENS}]", "-", regex=True)
    loose_left = norm.str.contains(r"forward[\s\-]*looking|safe[\s\-]*harbou?r|private\s+securities\s+litigation|\bcautionary",
                                   case=False, regex=True)
    nongaap = norm.str.contains(r"non-?\s*gaap|reconcil|guidance|measure|estimate|basis|outlook", case=False, regex=True)
    disclaimer_left = norm.str.contains(
        r"differ\s+materially|undertakes?\s+no\s+(?:obligation|duty)|no\s+obligation\s+to\s+update|risks?\s+and\s+uncertainties",
        case=False, regex=True) & ~loose_left
    res = pd.DataFrame({
        "check": ["scored sentences mentioning forward-looking / safe harbor / cautionary",
                  "  of which about non-GAAP measures, guidance or estimates (not a disclaimer)",
                  "scored sentences with disclaimer wording and no trigger phrase"],
        "sentences": [int(loose_left.sum()), int((loose_left & nongaap).sum()), int(disclaimer_left.sum())],
        "releases": [s[loose_left]["accession"].nunique(), s[loose_left & nongaap]["accession"].nunique(),
                     s[disclaimer_left]["accession"].nunique()],
    })
    res.to_csv(OUT / "safe_harbor_residual.csv", index=False)
    print("\nwhat is left in the scored sentences (main version, %d sentences):" % len(s))
    print(res.to_string(index=False))
    print("disclaimer wording left, by firm (sentences):", s[disclaimer_left]["ticker"].value_counts().to_dict())
    print("disclaimer wording left, by firm (releases):", s[disclaimer_left].groupby("ticker")["accession"].nunique().to_dict())
    lines = []
    for label, mask in (("TRIGGER-LIKE WORDING LEFT IN SCORED TEXT", loose_left), ("DISCLAIMER WORDING LEFT IN SCORED TEXT", disclaimer_left)):
        lines.append("\n######## " + label)
        for tk, g in s[mask].groupby("ticker"):
            lines.append(f"\n== {tk}: {len(g)} sentences in {g['accession'].nunique()} releases")
            lines += ["   - " + t[:400] for t in g["text"].drop_duplicates().head(4)]
    (config.INTERIM / "safe_harbor_residual_excerpts.txt").write_text("\n".join(lines), encoding="utf-8")
    if len(sys.argv) > 1:
        print("\n".join(excerpts)[: int(sys.argv[1])])


if __name__ == "__main__":
    main()
