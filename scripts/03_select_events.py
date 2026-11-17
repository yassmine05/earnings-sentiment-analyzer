"""Stage 1, step 3: one earnings release per firm and fiscal period.

Inputs   results/stage1/basket.csv, data/interim/filings_all.csv
Outputs  results/stage1/events_selected.csv    kept events (PROPOSED: not used downstream until approved)
         results/stage1/events_dropped.csv     every Item 2.02 filing not kept, with the reason
         data/interim/gap_fill_candidates.csv  non-2.02 8-Ks examined for periods with no kept filing (git-ignored)
         docs/stage1_event_rule.md             the exact rule and the dropped list per firm
"""
from __future__ import annotations

import re

import pandas as pd

from earnings_sentiment import config
from earnings_sentiment.edgar import EdgarClient
from earnings_sentiment.textclean import html_to_text
from earnings_sentiment.events import RULE_TEXT, add_flags, collect_firm_events, list_filings, select_events

OUT = config.RESULTS / "stage1"
# Fiscal periods whose release can fall inside the sample window.
SAMPLE_N, SAMPLE_SEED = 20, 42
# A snippet that contains an email address, a phone number or a "contact" heading is replaced in the report.
CONTACT_DETAILS = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|\(?\b\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b|\bcontacts?\s*:", re.I)


def redact(snippet: str) -> str:
    """Never publish individuals' names, phone numbers or email addresses from a filing's contact block."""
    return "[contact details omitted]" if CONTACT_DETAILS.search(snippet) else snippet
EXPECT_FROM, EXPECT_TO = "2018-09-01", "2024-11-30"


def in_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["accepted_et_date"].between(config.SAMPLE_START, config.SAMPLE_END)].copy()


def short(s, n=80) -> str:
    s = "" if pd.isna(s) else str(s)
    s = pd.Series([s]).str.replace(r"^EX-99\S*\s+\d+\s+\S+\s+", "", regex=True).iloc[0]
    return s[:n].replace("|", "/")


def main() -> None:
    basket = pd.read_csv(OUT / "basket.csv")
    filings = pd.read_csv(config.INTERIM / "filings_all.csv")
    kept, dropped = select_events(filings)
    kept["gap_fill"] = False

    # ---- step 8: gap fill ------------------------------------------------
    client = EdgarClient()
    reviewed = pd.read_csv(config.REFERENCE / "gap_fill_reviewed.csv").set_index("accession")
    gap_rows, unfilled = [], []
    for b in basket.itertuples():
        ciks = [int(c) for c in str(b.cik_all).split(";")]
        periodic = pd.concat([list_filings(client, c)[1] for c in ciks]).drop_duplicates("reportDate")
        expected = periodic[(periodic["reportDate"] >= EXPECT_FROM) & (periodic["reportDate"] <= EXPECT_TO)]["reportDate"]
        have = set(pd.to_datetime(kept.loc[kept["ticker"] == b.ticker, "fiscal_period_end"]))
        for p in sorted(set(expected) - have):
            def keep(ev, p=p):
                is_202 = ev["items"].fillna("").str.contains("2.02", regex=False)
                return (ev["fiscal_period_end"] == p) & ~is_202 & (ev["form"] == "8-K") & (ev["days_before_periodic"] >= 0)
            cand = collect_firm_events(client, ciks, b.ticker, item_202_only=False, keep=keep)
            if len(cand):
                cand.insert(0, "sector", b.sector)
                for c in ("head_results", "head_prelim"):
                    cand[c] = cand.get(c, False)
                    cand[c] = cand[c].fillna(False).astype(bool)
                cand["words_prose"] = cand.get("words_prose", 0)
                cand["words_prose"] = cand["words_prose"].fillna(0)
                ok = cand[(cand["exhibit_url"] != "") & cand["head_results"] & ~cand["head_prelim"]]
                cand["gap_fill_proposed"] = False
                cand["gap_fill_selected"] = False
                cand["review"] = cand["accession"].map(reviewed["decision"]).fillna("")
                cand["review_note"] = cand["accession"].map(reviewed["note"]).fillna("")
                if len(ok):
                    best = ok.sort_values(["words_prose", "accepted_utc"], ascending=[False, True]).index[0]
                    cand.loc[best, "gap_fill_proposed"] = True
                    cand.loc[best, "gap_fill_selected"] = cand.loc[best, "review"] == "accept"
                    if cand.loc[best, "review"] == "":
                        print(f"NOT YET REVIEWED: {b.ticker} {cand.loc[best, 'accession']} (add it to gap_fill_reviewed.csv)")
                gap_rows.append(cand)
            if not len(cand) or not cand["gap_fill_selected"].any():
                unfilled.append({"ticker": b.ticker, "fiscal_period_end": p.date().isoformat(),
                                 "non_2.02_8-Ks_examined": len(cand)})
    gaps = pd.concat(gap_rows, ignore_index=True) if gap_rows else pd.DataFrame()
    if len(gaps):
        gaps.to_csv(config.INTERIM / "gap_fill_candidates.csv", index=False)
        add = gaps[gaps["gap_fill_selected"]].drop(
            columns=["gap_fill_selected", "gap_fill_proposed", "review", "review_note"]
        ).copy()
        add["gap_fill"] = True
        add["had_other_202_before"] = False
        add["fiscal_period_end"] = add["fiscal_period_end"].dt.strftime("%Y-%m-%d")
        add["periodic_filed"] = add["periodic_filed"].dt.strftime("%Y-%m-%d")
        kept = pd.concat([kept, add], ignore_index=True)

    kept = add_flags(in_window(kept)).sort_values(["sector", "ticker", "accepted_utc"]).reset_index(drop=True)
    dropped = in_window(dropped)
    # step 9: image-only exhibits cannot be scored
    low = kept[kept["low_text"]].copy()
    low["drop_reason"] = "step 9: fewer than 200 prose words (image-only exhibit, cannot be scored)"
    n_low = len(low)
    kept = kept[~kept["low_text"]].drop(columns="low_text").reset_index(drop=True)
    dropped = pd.concat([dropped, low[dropped.columns.intersection(low.columns)]], ignore_index=True)
    dropped = dropped.sort_values(["ticker", "accepted_utc"]).reset_index(drop=True)
    unfilled = pd.DataFrame(unfilled)
    if len(unfilled):  # only gaps whose release would fall in the sample window matter
        unfilled = unfilled[(unfilled["fiscal_period_end"] >= "2018-12-01") & (unfilled["fiscal_period_end"] <= "2024-09-30")]
    # The committed tables carry no exhibit text; the short openings appear only in the review document.
    kept.drop(columns="head_text").to_csv(OUT / "events_selected.csv", index=False)
    dropped.drop(columns="head_text").to_csv(OUT / "events_dropped.csv", index=False)
    unfilled.to_csv(OUT / "events_unfilled_gaps.csv", index=False)

    # ---- the review document ----------------------------------------------
    n_in = len(in_window(filings))
    L = ["# Stage 1: one earnings release per quarter (approved rule)", "",
         "Generated by `scripts/03_select_events.py`. The rule below is implemented in "
         "`src/earnings_sentiment/events.py` (`select_events`).", "",
         "## The rule", "", "```text", RULE_TEXT.format(start=config.SAMPLE_START, end=config.SAMPLE_END).rstrip(), "```", "",
         "## Outcome", "",
         f"- Item 2.02 filings in the window: **{n_in}**",
         f"- Kept by steps 1-7: **{int((~kept['gap_fill']).sum()) + n_low}**",
         f"- Dropped by steps 1-7: **{len(dropped) - n_low}**",
         f"- Added by step 8 (gap fill): **{int(kept['gap_fill'].sum())}**",
         f"- Dropped by step 9 (image-only exhibit): **{n_low}**",
         f"- Kept events in total: **{len(kept)}** across {kept['ticker'].nunique()} firms",
         f"- Firm-quarters with no release found on EDGAR: **{len(unfilled)}**", "",
         "Dropped filings by reason:", ""]
    rs = dropped["drop_reason"].str.replace(r" \(.*\)$", "", regex=True).value_counts()
    L += ["| reason | filings |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rs.items()] + [""]

    L += ["Flags on kept events:", "",
          f"- `had_other_202_before` (a pre-announcement or update came first): {int(kept['had_other_202_before'].sum())}",
          f"- `periodic_before_8k` (10-Q/10-K accepted before the 8-K): {int(kept['periodic_before_8k'].sum())}, "
          f"by firm: {kept[kept['periodic_before_8k']]['ticker'].value_counts().to_dict()}",
          f"- `event_time_changed` (event clock moved to the 10-Q/10-K acceptance time): {int(kept['event_time_changed'].sum())}", "",
          "Event-clock buckets after the adjustment: "
          + ", ".join(f"{k} {v}" for k, v in kept["event_bucket"].value_counts().items()) + ".", ""]

    L += ["## Gap fill (step 8): earnings releases filed without Item 2.02", ""]
    if len(gaps):
        gsel = gaps[gaps["gap_fill_proposed"]]
        gsel = gsel[gsel["accepted_et_date"].between(config.SAMPLE_START, config.SAMPLE_END)]
        L += ["Proposed by the code, then read by hand:", "",
              "| firm | fiscal period | accepted (ET) | items reported | prose words | hand review | note |", "|---|---|---|---|---|---|---|"]
        for r in gsel.itertuples():
            L.append(f"| {r.ticker} | {r.fiscal_period_end.date()} | {r.accepted_et} | {r.items} | {int(r.words_prose)} | {r.review or 'pending'} | {r.review_note} |")
        L.append("")
    L += ["Firm-quarters still without a release (kept as gaps, not invented):", ""]
    L += (["| firm | fiscal period | non-2.02 8-Ks examined |", "|---|---|---|"]
          + [f"| {r.ticker} | {r.fiscal_period_end} | {r._3} |" for r in unfilled.itertuples()] if len(unfilled) else ["None."])
    # ---- timing question ---------------------------------------------------
    pb = kept[kept["periodic_before_8k"]].copy()
    pb["lag_hours"] = (pd.to_datetime(pb["accepted_et"]) - pd.to_datetime(pb["periodic_accepted_et"])).dt.total_seconds() / 3600
    L += ["", "## Timing adjustment (approved): 8-K accepted after the 10-Q/10-K", "",
          "For these events EDGAR accepted the 10-Q/10-K before the earnings 8-K, so the numbers were public "
          "before the 8-K timestamp. Where that moves day 0, the event clock is the 10-Q/10-K acceptance time "
          "(`event_time_changed = True`). The unadjusted 8-K clock stays in `accepted_et` and `day0_8k`. "
          "The Stage 2 timing check tests the adjustment.", "",
          "| firm | events | median lag (hours) | max lag (hours) | day 0 changes |", "|---|---|---|---|---|"]
    for t, g in pb.groupby("ticker"):
        L.append(f"| {t} | {len(g)} | {g['lag_hours'].median():.1f} | {g['lag_hours'].max():.1f} | {int(g['day0_differs'].sum())} |")
    L += ["", f"Events whose clock was changed: **{int(kept['event_time_changed'].sum())}** of {len(kept)}.", ""]
    dd = kept[kept["day0_differs"]]
    if len(dd):
        L += ["| firm | 8-K accepted (ET) | 10-Q/10-K accepted (ET) = event time used | day 0 from 8-K | day 0 used |", "|---|---|---|---|---|"]
        for r in dd.itertuples():
            L.append(f"| {r.ticker} | {r.accepted_et} | {r.periodic_accepted_et} | {r.day0_8k} | {r.day0_earliest} |")
    # ---- random sample of kept events ---------------------------------------
    smp = kept.sample(SAMPLE_N, random_state=SAMPLE_SEED).sort_values(["ticker", "accepted_utc"])
    L += ["", f"## Random sample of kept events ({SAMPLE_N} events, seed {SAMPLE_SEED})", "",
          "First 300 characters of each sampled exhibit, tables included, so the kept events can be confirmed as results releases.", "",
          "| firm | accepted (ET) | fiscal period | first 300 characters |", "|---|---|---|---|"]
    sample_lines = []
    for r in smp.itertuples():
        text = re.sub(r"\s+", " ", html_to_text(client.get_text(r.exhibit_url), drop_tables=False))
        text = re.sub(r"^EX-99\S*\s+\d+\s+\S+\s+", "", text)[:300].replace("|", "/")
        text = redact(text)
        L.append(f"| {r.ticker} | {r.accepted_et} | {r.fiscal_period_end} | {text} |")
        sample_lines.append(f"{r.ticker:6s} {r.accepted_et}  {text}")
    L += ["", "## Dropped filings, per firm", ""]
    kept_by = kept.groupby("ticker").size()
    for t in basket["ticker"]:
        d = dropped[dropped["ticker"] == t]
        L += [f"### {t}: {int(kept_by.get(t, 0))} kept, {len(d)} dropped", ""]
        if d.empty:
            continue
        L += ["| accepted (ET) | fiscal period | items | prose words | reason | exhibit opens with |", "|---|---|---|---|---|---|"]
        for r in d.itertuples():
            L.append(f"| {r.accepted_et} | {r.fiscal_period_end} | {r.items} | {int(r.words_prose)} | {r.drop_reason} | {redact(short(r.head_text, 70))} |")
        L.append("")
    (config.DOCS / "stage1_event_rule.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L[L.index("## Outcome"):L.index("## Dropped filings, per firm")]))
    per = kept.groupby("ticker").size()
    print("\n".join(sample_lines))
    print("events per firm:", per.value_counts().sort_index().to_dict())
    print("firms not at 24:", per[per != 24].to_dict())


if __name__ == "__main__":
    main()
