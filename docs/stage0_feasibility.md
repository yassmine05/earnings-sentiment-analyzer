# Stage 0: feasibility note

Date of probes: 2026-11-15. Every statement marked **Probed** comes from a request that was actually run on this machine with Python 3.11.9 (standard library only, nothing installed). Statements marked **Not probed** are from documentation or prior knowledge and still need confirming.

## 1. Verdict

Track A (SEC EDGAR 8-K Item 2.02 earnings press releases) is feasible end to end with free, permitted sources. Two planned sources failed and have replacements (section 6). Nothing blocks Stage 1.

## 2. Text and event timestamps: SEC EDGAR

| Check | Result | Status |
|---|---|---|
| Access terms | SEC fair-access page: free to access and download, max 10 requests/second, declared User-Agent required. The page says nothing explicit about redistribution, so raw text stays out of the repo regardless. | Probed |
| Ticker to CIK map | `sec.gov/files/company_tickers.json`, 10,404 tickers | Probed |
| Filing index | `data.sec.gov/submissions/CIK##########.json` gives form, items, acceptance time, accession number | Probed |
| AAPL, 2019-01 to 2025-03 | 26 filings with Item 2.02 | Probed |
| JPM, 2019-01 to 2025-03 | 25 filings with Item 2.02 | Probed |
| Exhibit 99.1 | Located from the filing index page; AAPL release 1,388 words, JPM narrative 4,098 words (both counts include table text, before cleaning) | Probed |

Findings that change the implementation:

1. **Acceptance time is true UTC.** The JSON field reads `2025-01-30T21:30:23.000Z` for the Apple filing whose index page and SGML header both say 16:30:23 Eastern. All times will be converted from UTC to America/New_York before bucketing. Treating the field as Eastern would have put every after-close release on the wrong day.
2. **Timing looks plausible on the two probes.** JPM files at about 06:45 ET (pre-open), AAPL at about 16:30 ET (after close), which matches how those firms are known to release. The Stage 2 abnormal-return and volume plot remains the real test.
3. **Not every Item 2.02 filing is a quarterly release.** AAPL's 2019-01-02 filing is the revenue-guidance warning, not earnings. Stage 1 needs a rule to keep one results release per fiscal quarter and to list everything it drops (question 3 below).
4. **Heavy filers need paging.** JPM's "recent" block covers only 12 months (26,439 rows) and the rest sits in 72 extra index files, so the downloader must walk those.
5. **Skip EX-99.2 and later.** JPM's EX-99.2 is a 3.3 MB financial supplement of tables. Only the narrative exhibit will be scored.

## 3. Prices, benchmark and factors

| Source | Result | Status |
|---|---|---|
| Yahoo chart endpoint (what yfinance wraps) | AAPL, JPM, SPY, XLK: 1,800 daily rows each for 2018-01-02 to 2025-02-28, close, adjusted close and volume present, no null closes | Probed |
| Yahoo, delisted ticker | CELG (acquired 2019) returns HTTP 404 | Probed |
| Stooq daily CSV | Returns a JavaScript verification page, not data | Probed, **failed** |
| Ken French daily factors | 3-factor and momentum files download; daily rows through 2026-09-30 | Probed |
| yfinance package itself | Version 1.8.0 available for Python 3.11; not installed or run yet | Not probed |

Consequences:

- Stooq is dropped as the cross-check. Replacement sanity check: SPY daily returns against the Ken French market return (Mkt-RF + RF) over the sample.
- Firms that left the market after January 2019 have no Yahoo history. Any basket member in that position is dropped, counted, and named in the README as a survivorship limitation.

## 4. Basket: January 2019 membership, sectors and market caps

**Membership and sector (exact, official, point in time).** The Select Sector SPDR Trust (CIK 1064641) filed form N-Q on 2019-02-28 with the full schedule of investments of all 11 sector funds as of 2018-12-31 (accession 0001193125-19-056828). The 11 funds together hold the S&P 500 split by GICS sector, so this one public-domain SEC filing gives both membership and sector on the last trading day before January 2019. **Probed:** the filing downloads (1.7 MB) and the per-fund schedules parse into company name, shares and value.

**Market cap (approximate).** Fund position values cannot be used to rank firms, because the sector funds cap large positions. The probe shows this directly: the Communication Services fund held about $128.0M of Verizon and $130.3M of Netflix, although Verizon was the far larger company. Market cap will instead be computed as:

- shares outstanding from the cover page of the 10-K or 10-Q nearest to 2018-12-31 (EDGAR XBRL concept `dei:EntityCommonStockSharesOutstanding`), times
- the unadjusted Yahoo close on 2018-12-31.

**Probed:** the share-count concept returns usable values for AAPL and JPM. It returns nothing usable for Alphabet (404) and Berkshire (no facts near the date), because multi-class firms report shares per class. Those cases will be read from the cover-page text of the filing and summed across classes.

Why this is approximate, to be stated in the README:

1. The share count is dated a few weeks either side of 2018-12-31, not on it.
2. It is total market cap, not the float-adjusted cap S&P uses for index weights.
3. Multi-class firms need a hand-checked share count.
4. Company names in the N-Q must be mapped to tickers and CIKs; the mapping will be code-assisted and saved as a reviewed CSV.

**Rule:** rank all members of each sector by this market cap and take the top 5, giving 55 firms. To keep the work small, caps are computed for the 15 largest fund positions per sector, which safely contains the true top 5 despite capping. The resulting list, with every input, is a Stage 1 output for review before any bulk download.

**Fallback considered:** the `fja05680/sp500` historical-components CSV on GitHub (MIT licence, **Probed** reachable). It is built from Wikipedia edits and has no sectors or caps, so it is kept only as a cross-check on membership. iShares historical holdings were also tried and returned a gated web page instead of data (**Probed, failed**).

## 5. Sentiment inputs

| Item | Result | Status |
|---|---|---|
| ProsusAI/finbert | Hugging Face API reports licence tag `cc-by-nc-sa-4.0` (non-commercial), `model.safetensors` 438 MB | Probed |
| yiyanghkust/finbert-tone (robustness) | No licence tag; only `pytorch_model.bin` (439 MB), no safetensors file | Probed |
| Loughran-McDonald Master Dictionary | Current file `Loughran-McDonald_MasterDictionary_1993-2025.csv`, hosted on Google Drive from the SRAF page. Terms quoted on the page: "free for use in academic research"; commercial use needs a licence. | Probed (page only, file not downloaded) |
| Kaggle GPU path | Needs your own Kaggle account; sentences file uploaded and scores file downloaded by hand | Not probed |

Notes:

- The licence tags make this a non-commercial research project. The README should say so.
- finbert-tone ships only pickle-format weights. That is a known loading restriction in recent `transformers`/`torch` and a mild safety consideration, so it stays a secondary robustness model and cannot block the main results.
- Press releases are much shorter than call transcripts (about 1,400 to 4,100 words on the two probes, tables included), so the CPU scoring load is far below the transcript estimate in the plan. The 5-firm pilot will measure actual sentences per document and sentences per second.

## 6. What failed and what replaces it

| Planned | Outcome | Replacement |
|---|---|---|
| Stooq price cross-check | Blocked by JavaScript verification | SPY versus Ken French market return |
| iShares holdings for the 2019 basket | Gated page, no data | Select Sector SPDR N-Q on EDGAR plus XBRL share counts |
| Yahoo history for delisted firms | 404 | Drop, count and report |

## 7. Environment

| Item | Result |
|---|---|
| Python | 3.11.9 at the per-user install; 3.14 is also installed and will not be used |
| pip | 24.0 |
| Packages available for 3.11 | torch 2.15.0, transformers 5.23.0, yfinance 1.8.0, pandas 3.1.1, statsmodels 0.15.0, linearmodels 7.1, pandas-market-calendars 5.6.1, onnxruntime 1.32.0 |
| CPU | Intel Core i5-1135G7, 4 cores, 8 threads |
| Free disk | 43 GB |
| git | 2.48.1; the project folder is not yet a repository |
| Python SSL | One host (api.github.com) failed certificate verification from Python while working from curl. All data hosts worked. Noted in case it recurs. |

Nothing has been installed and `.venv` has not been created yet. Both happen at the start of Stage 1, with versions pinned in a requirements file.

## 8. Files created in Stage 0

- `.gitignore`: ignores `.env`, `.venv/`, raw and intermediate data, the dictionary folder and model caches.
- `.env.example`: placeholder User-Agent.
- `.env`: the real User-Agent, git-ignored.
- `docs/stage0_feasibility.md`: this note.

The probe scripts were throwaway and are not part of the project.

## 9. Decisions needed before Stage 1

1. **Basket method.** Approve section 4: membership and sector from the SPDR N-Q, market cap from XBRL shares times the 2018-12-31 close, documented as approximate.
2. **Loughran-McDonald file.** Either you download the CSV from the SRAF page into `data/external/`, or you approve the download by script.
3. **One release per quarter.** Proposed rule: where a firm has more than one Item 2.02 filing for the same fiscal quarter, keep the one whose exhibit is the quarterly results release and log the others as dropped. The exact test is fixed in Stage 1 after seeing all 55 firms' filings, and shown for review.
4. **git.** Initialise a repository now, with no commits until you ask, or leave it for later.
