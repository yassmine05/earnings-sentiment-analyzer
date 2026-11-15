# Project instructions for Claude

## Goal

Event study testing whether the tone of 8-K earnings press releases predicts abnormal stock returns,
comparing FinBERT with the Loughran-McDonald dictionary. 55 firms (5 per GICS sector), 2019-2024.
Track A only: text and event times come from SEC EDGAR. Stage reports are in `docs/`.

## No lookahead

- Day 0 comes from the approved event clock: EDGAR acceptance time converted from UTC to Eastern,
  mapped to a session with the NYSE calendar (`src/earnings_sentiment/timing.py`, `events.py`).
- Estimation windows end before the event.
- Scoring rules are frozen and committed before any score is joined to a return.

## Never commit, never expose

- Never commit `data/raw`, `data/external`, `data/interim`, `.env` or `.venv`.
- Never put the owner's name or email in source files, notebooks, the README or commit messages.
  The SEC User-Agent lives only in `.env`.

## Licences

- Non-commercial research. FinBERT (`ProsusAI/finbert`) is CC BY-NC-SA 4.0; the Loughran-McDonald
  dictionary is for academic use.
- Never redistribute the dictionary file.

## Results

- Every number in the README must come from code that was actually run.

## Workflow

- Work in stages, one branch per stage (`stage-N-...`).
- Stop for the owner's review at the end of each stage.

## Environment

- Windows, PowerShell, Python 3.11.9 in `.venv`, no GPU.

## Git

- Never add Co-Authored-By or other AI attribution lines to commit messages.
