# Earnings press-release sentiment and post-earnings returns

**Status: work in progress.** Stage 1 (data collection) is complete; no results yet. Every number that
appears in this README at the end of the project will come from code in this repository that was actually run.

## Question

Does the tone of a company's quarterly earnings press release predict its stock's abnormal return over the
following 1 to 5 trading days? Tone is measured two ways and compared: FinBERT (a finance-tuned transformer)
and the Loughran-McDonald dictionary.

## Data

| Input | Source | Notes |
|---|---|---|
| Earnings press releases | SEC EDGAR, 8-K Item 2.02, Exhibit 99.1 | Public filings; raw text is not redistributed here |
| Event timestamps | EDGAR acceptance time (UTC, converted to Eastern) | |
| Firm basket | Select Sector SPDR Trust form N-Q, holdings at 2018-12-31 | 5 largest firms in each of the 11 GICS sectors |
| Prices | Yahoo Finance via `yfinance` | Personal, non-commercial use |
| Factors | Ken French data library | Daily 3 factors and momentum |
| Dictionary | Loughran-McDonald Master Dictionary | You must download it yourself, see below |

This study uses earnings **press releases**, not earnings-call transcripts. Free transcript sources are either
rate-limited or carry unclear redistribution rights, so the project starts from text that is unambiguously public.

## Licences and intended use

This is **non-commercial research**.

- `ProsusAI/finbert` is published under CC BY-NC-SA 4.0 (non-commercial).
- The Loughran-McDonald dictionary is free for academic research only; commercial use needs a licence from its authors.
- Raw filing text, price files and the dictionary are never committed. The repository holds code, derived scores and results.

## Setup

Windows PowerShell, Python 3.11:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
Copy-Item .env.example .env   # then edit .env: SEC requires your own name and contact email in the User-Agent
```

**Loughran-McDonald dictionary.** Download the Master Dictionary CSV yourself from the authors' page,
<https://sraf.nd.edu/loughranmcdonald-master-dictionary/>, and place it in `data/external/`. That folder is git-ignored.

## Reproducing Stage 1

```powershell
.\.venv\Scripts\python.exe scripts\01a_holdings_and_issuers.py
.\.venv\Scripts\python.exe scripts\01b_market_caps.py
.\.venv\Scripts\python.exe scripts\01d_select_basket.py
.\.venv\Scripts\python.exe scripts\02_fetch_filings.py
.\.venv\Scripts\python.exe scripts\03_select_events.py
.\.venv\Scripts\python.exe scripts\04_fetch_prices.py
.\.venv\Scripts\python.exe scripts\05_data_quality.py
```

`scripts/01c_cover_page_shares.py` prints the cover-page share counts used for the hand-checked,
multi-class firms in `data/reference/multiclass_shares.csv`.

Stage reports: `docs/stage0_feasibility.md`, `docs/stage1_event_rule.md`, `docs/stage1_data_quality.md`.

## Known approximations (so far)

- **Market cap used to pick the basket is approximate.** It is cover-page shares outstanding (dated within a few
  weeks of 2018-12-31) times the 2018-12-31 price implied by the SPDR filing. It is total, not float-adjusted,
  market cap, and multi-class firms use hand-checked share counts. Close calls are listed in the data-quality report.
- **Survivorship.** The basket is fixed from end-2018 membership, not today's index, and no selected firm had to be
  dropped for missing prices. It is still a basket of the largest firms, which all survived the sample period.
- **Event time.** Day 0 comes from the EDGAR acceptance time of the 8-K, which can lag the newswire release.
