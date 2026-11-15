"""Earnings events from 8-K Item 2.02 filings: listing, exhibit download, fiscal-period assignment."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import config
from .edgar import EdgarClient, FilingDoc
from .textclean import html_to_text, word_count
from .timing import day0, nyse_sessions, timing_bucket

PERIODIC_FORMS = {"10-K", "10-Q", "10-KT", "10-QT", "10-K405"}
# Listing window is wider than the sample so period assignment near the edges is right.
LIST_START = "2018-10-01"
LIST_END = "2025-03-31"

_TEXT_EXT = (".htm", ".html", ".txt")


def list_filings(client: EdgarClient, cik: int, item_202_only: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (8-K filings, periodic reports) for one company. By default only 8-Ks with Item 2.02."""
    _, df = client.submissions(cik)
    df = df[(df["filingDate"] >= LIST_START) & (df["filingDate"] <= LIST_END)].copy()

    periodic = df[df["form"].isin(PERIODIC_FORMS)][
        ["form", "reportDate", "filingDate", "accessionNumber", "acceptanceDateTime"]
    ].copy()
    periodic = periodic[periodic["reportDate"] != ""]
    periodic["reportDate"] = pd.to_datetime(periodic["reportDate"])
    periodic["filingDate"] = pd.to_datetime(periodic["filingDate"])
    periodic["periodic_accepted_et"] = (
        pd.to_datetime(periodic.pop("acceptanceDateTime"), utc=True)
        .dt.tz_convert(config.MARKET_TZ).dt.strftime("%Y-%m-%d %H:%M:%S")
    )
    periodic = periodic.sort_values("reportDate").drop_duplicates("reportDate").reset_index(drop=True)

    is_8k = df["form"].isin(["8-K", "8-K/A"])
    has_202 = df["items"].fillna("").str.split(",").map(lambda xs: "2.02" in [x.strip() for x in xs])
    ev = df[is_8k & (has_202 | (not item_202_only))].copy()
    # EDGAR's acceptanceDateTime is UTC (verified in Stage 0); convert to Eastern.
    acc_utc = pd.to_datetime(ev["acceptanceDateTime"], utc=True)
    ev["accepted_utc"] = acc_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    acc_et = acc_utc.dt.tz_convert(config.MARKET_TZ)
    ev["accepted_et"] = acc_et.dt.strftime("%Y-%m-%d %H:%M:%S")
    ev["accepted_et_date"] = acc_et.dt.strftime("%Y-%m-%d")
    ev["cik"] = int(cik)
    ev = ev.rename(columns={"accessionNumber": "accession", "filingDate": "filing_date", "reportDate": "report_date"})
    cols = ["cik", "accession", "form", "items", "filing_date", "report_date", "accepted_utc", "accepted_et", "accepted_et_date"]
    return ev[cols].sort_values("accepted_utc").reset_index(drop=True), periodic


def assign_fiscal_period(ev: pd.DataFrame, periodic: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest fiscal period end strictly before the filing, and that period's 10-Q/10-K date."""
    ev = ev.copy()
    ev["_d"] = pd.to_datetime(ev["accepted_et_date"])
    left = ev.sort_values("_d")
    right = periodic.rename(
        columns={"reportDate": "fiscal_period_end", "filingDate": "periodic_filed", "form": "periodic_form"}
    )[["fiscal_period_end", "periodic_filed", "periodic_form", "periodic_accepted_et"]].sort_values("fiscal_period_end")
    out = pd.merge_asof(
        left, right, left_on="_d", right_on="fiscal_period_end", direction="backward", allow_exact_matches=False
    )
    out["days_after_period_end"] = (out["_d"] - out["fiscal_period_end"]).dt.days
    out["days_before_periodic"] = (out["periodic_filed"] - out["_d"]).dt.days
    return out.drop(columns="_d")


def pick_exhibit(docs: list[FilingDoc]) -> tuple[FilingDoc | None, int]:
    """EX-99.1 if present, otherwise the first EX-99.x; text/HTML documents only."""
    ex = [d for d in docs if d.doc_type.upper().startswith("EX-99") and d.href.lower().endswith(_TEXT_EXT)]
    if not ex:
        # Filer typed the exhibit type wrongly but described it as EX-99 (seen once: Linde, April 2022).
        ex = [
            d for d in docs
            if d.doc_type.upper().startswith("EX-") and d.description.upper().startswith("EX-99")
            and d.href.lower().endswith(_TEXT_EXT)
        ]
        return (ex[0], len(ex)) if ex else (None, 0)
    first = [d for d in ex if d.doc_type.upper() in ("EX-99.1", "EX-99.01", "EX-99")]
    return (first[0] if first else ex[0]), len(ex)


_RESULTS_RE = re.compile(
    r"\b(first|second|third|fourth|1st|2nd|3rd|4th)[\s\-]+quarter\b|\bq[1-4]\b|\b[1-4]q\s?\d\d|"
    r"\bfull[\s\-]+year\b|\bfiscal\s+(year\s+)?(20)?\d\d\b|\bquarterly\b|\byear[\s\-]+end\b|\bannual\b",
    re.I,
)
_RESULTS_WORD_RE = re.compile(r"\b(results|earnings|net income|revenues?|sales|eps)\b", re.I)
_PRELIM_RE = re.compile(
    r"\bpreliminary\b|\bpre-?announce|\b(updates?|updated|revises?|revised|lowers?|raises?|withdraws?|withdrawn|suspends?)\b"
    r".{0,60}\b(guidance|outlook|forecast|expectations)\b|\bletter\s+to\s+(apple\s+)?investors\b|\bbusiness\s+update\b",
    re.I | re.S,
)


def exhibit_features(html: str) -> dict:
    full = html_to_text(html, drop_tables=False)
    prose = html_to_text(html, drop_tables=True)
    head = full[:1500]
    return {
        "words_full": word_count(full),
        "words_prose": word_count(prose),
        "head_results": bool(_RESULTS_RE.search(head) and _RESULTS_WORD_RE.search(head)),
        "head_prelim": bool(_PRELIM_RE.search(head)),
        "head_text": re.sub(r"\s+", " ", head[:220]),
    }


def collect_firm_events(
    client: EdgarClient, ciks: list[int], ticker: str, item_202_only: bool = True, keep=None
) -> pd.DataFrame:
    """8-K filings of one firm with exhibit metadata and text features. Downloads and caches exhibits.

    `ciks` holds every SEC identifier the firm filed under during the sample (usually one).
    `keep` is an optional row filter applied after fiscal-period assignment and before any exhibit download.
    """
    parts = [list_filings(client, c, item_202_only) for c in ciks]
    ev = pd.concat([p[0] for p in parts], ignore_index=True).drop_duplicates("accession")
    periodic = pd.concat([p[1] for p in parts], ignore_index=True)
    periodic = periodic.sort_values(["reportDate", "filingDate"]).drop_duplicates("reportDate").reset_index(drop=True)
    ev = assign_fiscal_period(ev, periodic)
    if keep is not None:
        ev = ev[keep(ev)]
    rows = []
    for r in ev.to_dict("records"):
        cik = r["cik"]
        docs = client.filing_docs(cik, r["accession"])
        doc, n_ex99 = pick_exhibit(docs)
        r.update({"ticker": ticker, "n_ex99": n_ex99, "exhibit_type": "", "exhibit_url": "", "exhibit_bytes": 0})
        if doc is not None:
            html = client.get_text(doc.url)
            r.update({"exhibit_type": doc.doc_type, "exhibit_url": doc.url, "exhibit_bytes": len(html)})
            r.update(exhibit_features(html))
        rows.append(r)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# One earnings release per firm and fiscal period
# ---------------------------------------------------------------------------
RULE_TEXT = """\
Scope: 8-K filings of the 55 basket firms accepted by EDGAR between {start} and {end} (Eastern time).
A filing's fiscal period is the latest fiscal period end, taken from the firm's own 10-Q/10-K list,
that falls strictly before the filing's acceptance date.

Step 1. Candidates are 8-K filings that report Item 2.02 (Results of Operations and Financial Condition).
Step 2. Drop amendments (form 8-K/A). The original filing carries the news and its timestamp.
Step 3. Drop filings with no HTML or text EX-99 exhibit (there is no press release to score).
Step 4. Group the remaining filings by firm and fiscal period.
Step 5. Inside a group, drop filings dated after the day the 10-Q/10-K for that period was filed,
        provided at least one filing dated on or before it remains. (These are later supplements and recasts.)
Step 6. If more than one filing is still left, prefer those whose exhibit opens like a results release
        and not like a preliminary update. The test reads the first 1,500 characters of the exhibit:
          results release  = a period phrase (first/second/third/fourth quarter, Q1-Q4, full year,
                             fiscal year, quarterly, annual, year-end) AND a results word
                             (results, earnings, net income, revenue, sales, EPS);
          preliminary      = "preliminary", "pre-announce", "business update", "letter to investors",
                             or update/revise/lower/raise/withdraw/suspend near guidance/outlook/
                             forecast/expectations.
        If no filing passes, all remain.
Step 7. Keep the filing with the most prose words (financial tables removed); ties go to the earliest.
        Everything else in the group is dropped and listed.
Step 8. Gap fill. If a firm has no kept filing for a fiscal period, look at its 8-Ks WITHOUT Item 2.02
        accepted after the period end and no later than the day the 10-Q/10-K was filed. The code proposes
        the one whose exhibit passes the results-release test of step 6 (most prose words wins). A proposal
        is used only if it has been read by hand and marked `accept` in data/reference/gap_fill_reviewed.csv.
        These are earnings releases filed under the wrong item number and are flagged `gap_fill`.
Step 9. Drop kept events whose exhibit has fewer than 200 prose words (image-only exhibits cannot be scored).

Event clock. The event time is the EDGAR acceptance time of the 8-K, with one exception: when the 10-Q/10-K
for the same period was accepted before the 8-K and using it moves day 0, the event time is the 10-Q/10-K
acceptance time (the numbers were already public). Day 0 is the same session if the event time is before
16:00 Eastern on a trading day, otherwise the next session.

Columns on kept events:
  had_other_202_before  another Item 2.02 filing for the same period came earlier (pre-announcement or update)
  periodic_before_8k    the 10-Q/10-K was accepted by EDGAR before this 8-K
  event_time_changed    True where the event clock was moved to the 10-Q/10-K acceptance time
  event_time_et, event_time_source, event_bucket, event_day0   the clock actually used
  accepted_et, timing_bucket, day0_8k                          the unadjusted 8-K clock, kept for comparison
"""


def _reason(row: pd.Series, kept: pd.Series) -> str:
    when = "earlier than" if row["accepted_utc"] < kept["accepted_utc"] else "later than"
    kind = "preliminary/update wording" if bool(row.get("head_prelim")) else "fewer prose words or not a results release"
    return f"step 6-7: same fiscal period, {when} the kept release ({kind})"


def select_events(filings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply steps 1-7 to Item 2.02 filings. Returns (kept, dropped-with-reason)."""
    f = filings.copy()
    for c in ("head_results", "head_prelim"):
        f[c] = f[c].fillna(False).astype(bool)
    f["words_prose"] = f["words_prose"].fillna(0)
    f["drop_reason"] = ""
    f.loc[f["form"] != "8-K", "drop_reason"] = "step 2: amendment (8-K/A)"
    no_ex = f["exhibit_url"].isna() | (f["exhibit_url"] == "")
    f.loc[no_ex & (f["drop_reason"] == ""), "drop_reason"] = "step 3: no EX-99 text exhibit"

    kept_idx = []
    for _, g in f[f["drop_reason"] == ""].groupby(["ticker", "fiscal_period_end"], dropna=False):
        pool = g
        on_time = g[g["days_before_periodic"].isna() | (g["days_before_periodic"] >= 0)]
        if len(on_time) and len(on_time) < len(g):
            late = g.index.difference(on_time.index)
            f.loc[late, "drop_reason"] = "step 5: filed after the 10-Q/10-K for the period (supplement or recast)"
            pool = on_time
        pref = pool[pool["head_results"] & ~pool["head_prelim"]]
        cand = pref if len(pref) else pool
        best = cand.sort_values(["words_prose", "accepted_utc"], ascending=[False, True]).iloc[0]
        kept_idx.append(best.name)
        for i in pool.index.difference([best.name]):
            f.loc[i, "drop_reason"] = _reason(f.loc[i], best)
        same = f[(f["ticker"] == best["ticker"]) & (f["fiscal_period_end"] == best["fiscal_period_end"]) & (f["form"] == "8-K")]
        f.loc[best.name, "had_other_202_before"] = bool((same["accepted_utc"] < best["accepted_utc"]).any())

    kept = f.loc[kept_idx].drop(columns="drop_reason").copy()
    kept["had_other_202_before"] = kept["had_other_202_before"].astype(bool)
    dropped = f[f["drop_reason"] != ""].drop(columns="had_other_202_before", errors="ignore").copy()
    return kept, dropped


def add_flags(kept: pd.DataFrame) -> pd.DataFrame:
    k = kept.copy()
    t8k = pd.to_datetime(k["accepted_et"])
    tper = pd.to_datetime(k["periodic_accepted_et"])
    k["periodic_before_8k"] = tper < t8k
    k["low_text"] = k["words_prose"].fillna(0) < 200
    # Day 0 under the default clock (8-K acceptance) and under the earliest public filing for the period.
    sessions = nyse_sessions()
    earliest = t8k.where(~k["periodic_before_8k"], tper)
    k["earliest_filing_et"] = earliest.dt.strftime("%Y-%m-%d %H:%M:%S")
    k["timing_bucket"] = timing_bucket(k["accepted_et"], sessions)
    k["day0_8k"] = day0(k["accepted_et"], sessions).dt.strftime("%Y-%m-%d")
    k["day0_earliest"] = day0(k["earliest_filing_et"], sessions).dt.strftime("%Y-%m-%d")
    k["day0_differs"] = k["day0_8k"] != k["day0_earliest"]
    # Approved timing rule: where the 10-Q/10-K reached EDGAR first AND that moves day 0, the event
    # clock is the earlier acceptance time. Everywhere else it is the 8-K acceptance time.
    k["event_time_changed"] = k["day0_differs"]
    k["event_time_source"] = np.where(k["event_time_changed"], "10-Q/10-K acceptance", "8-K acceptance")
    k["event_time_et"] = np.where(k["event_time_changed"], k["earliest_filing_et"], k["accepted_et"])
    k["event_bucket"] = timing_bucket(k["event_time_et"], sessions)
    k["event_day0"] = day0(k["event_time_et"], sessions).dt.strftime("%Y-%m-%d")
    return k
