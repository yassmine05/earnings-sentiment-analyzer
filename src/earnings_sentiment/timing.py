"""Map an Eastern-time timestamp to event day 0 (the first trading session that can react to the news)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from . import config

CLOSE_MINUTE = 16 * 60
OPEN_MINUTE = 9 * 60 + 30


def nyse_sessions(start: str = config.PRICE_START, end: str = "2025-12-31") -> pd.DatetimeIndex:
    return mcal.get_calendar("NYSE").schedule(start, end).index.normalize()


def timing_bucket(ts_et: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """pre-open (<09:30), market hours (09:30-15:59), after close (>=16:00), or non-trading day."""
    t = pd.to_datetime(ts_et)
    minutes = t.dt.hour * 60 + t.dt.minute
    out = pd.Series("market hours", index=t.index)
    out[minutes < OPEN_MINUTE] = "pre-open"
    out[minutes >= CLOSE_MINUTE] = "after close"
    out[~t.dt.normalize().isin(sessions)] = "non-trading day"
    return out


def day0(ts_et: pd.Series, sessions: pd.DatetimeIndex) -> pd.Series:
    """Same session if the news arrives before 16:00 on a trading day, otherwise the next session.

    Early-close days (13:00) are treated like normal days; none of the sample's releases depend on it
    unless flagged in Stage 2.
    """
    t = pd.to_datetime(ts_et)
    d = t.dt.normalize()
    minutes = t.dt.hour * 60 + t.dt.minute
    same_day = d.isin(sessions) & (minutes < CLOSE_MINUTE)
    side = np.where(same_day, "left", "right")
    arr = sessions.values
    pos = np.array([np.searchsorted(arr, x, side=s) for x, s in zip(d.values, side)])
    return pd.Series(arr[pos], index=t.index)
