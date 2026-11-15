"""Tone surprise: a document score minus the firm's own mean over its previous four fiscal quarters.

Part of the frozen scoring specification (docs/stage3_scoring_spec.md, section 6).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LOOKBACK_DAYS = 380
REQUIRED_PRIOR = 4


def prior_quarter_counts(events: pd.DataFrame) -> pd.Series:
    """Number of the same firm's kept events whose fiscal period end lies in the 380 days before each event's."""
    fpe = pd.to_datetime(events["fiscal_period_end"])
    out = pd.Series(0, index=events.index, dtype=int)
    for _, idx in events.groupby("ticker").groups.items():
        d = fpe.loc[idx]
        for i in idx:
            gap = (d.loc[i] - d).dt.days
            out.loc[i] = int(((gap > 0) & (gap <= LOOKBACK_DAYS)).sum())
    return out


def tone_surprise(events: pd.DataFrame, score_col: str) -> pd.Series:
    """Score minus the mean score of the previous four fiscal quarters; missing unless all four exist
    and all four have a score. Uses only earlier quarters of the same firm."""
    fpe = pd.to_datetime(events["fiscal_period_end"])
    out = pd.Series(np.nan, index=events.index, dtype=float)
    for _, idx in events.groupby("ticker").groups.items():
        d, s = fpe.loc[idx], events.loc[idx, score_col]
        for i in idx:
            gap = (d.loc[i] - d).dt.days
            prior = s[(gap > 0) & (gap <= LOOKBACK_DAYS)]
            if len(prior) == REQUIRED_PRIOR and prior.notna().all() and pd.notna(s.loc[i]):
                out.loc[i] = s.loc[i] - prior.mean()
    return out
