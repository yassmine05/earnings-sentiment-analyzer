"""Event-study engine: expected-return models, abnormal returns, cumulative abnormal returns, abnormal volume.

Conventions
  relative day 0 = event day 0 (first session that can react to the release)
  estimation window = relative days [-250, -11], at least 120 valid observations
  AR_t  = R_t - E[R_t]          (simple daily returns from dividend- and split-adjusted closes)
  CAR[a,b] = sum of AR_t for t in [a, b]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

EST_START, EST_END, EST_MIN_OBS = -250, -11, 120
REL_MIN, REL_MAX = -20, 10

SECTOR_ETF = {
    "Communication Services": "XLC", "Consumer Discretionary": "XLY", "Consumer Staples": "XLP", "Energy": "XLE",
    "Financials": "XLF", "Health Care": "XLV", "Industrials": "XLI", "Information Technology": "XLK",
    "Materials": "XLB", "Real Estate": "XLRE", "Utilities": "XLU",
}

# (name, first day, last day) of every cumulative window that is reported
CAR_WINDOWS = (
    [("car_0_0", 0, 0)]
    + [(f"car_0_{k}", 0, k) for k in range(1, 6)]          # reaction family
    + [("ar_1", 1, 1)] + [(f"car_1_{k}", 1, k) for k in range(2, 6)]  # drift family
    + [("car_m20_m1", -20, -1), ("car_m10_m2", -10, -2)]   # pre-event run-up (control / placebo)
)


@dataclass
class MarketData:
    ret: pd.DataFrame        # daily simple returns, one column per ticker (stocks, SPY, sector ETFs)
    ret_est: pd.DataFrame    # same, with excluded firm-days set to NaN
    volume: pd.DataFrame
    factors: pd.DataFrame    # mkt_rf, smb, hml, rf, mom (decimals), indexed by date

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.ret.index


def load_market_data(tickers: list[str], excluded: pd.DataFrame | None = None) -> MarketData:
    frames = [pd.read_parquet(config.PRICES_DIR / f"{t}.parquet") for t in tickers]
    px = pd.concat(frames, ignore_index=True)
    adj = px.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    vol = px.pivot(index="date", columns="ticker", values="volume").sort_index()
    ret = adj.pct_change()
    ret_est = ret.copy()
    if excluded is not None:
        for t, d in zip(excluded["ticker"], pd.to_datetime(excluded["date"])):
            if t in ret_est.columns and d in ret_est.index:
                ret_est.loc[d, t] = np.nan
    ff = pd.read_parquet(config.FACTORS_DIR / "ff_daily.parquet").set_index("date").reindex(ret.index)
    return MarketData(ret=ret, ret_est=ret_est, volume=vol, factors=ff)


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float, float, int]:
    """OLS with intercept. Returns (coefficients incl. intercept, residual std, R2, n). NaN rows are dropped."""
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    y, X = y[ok], X[ok]
    n = len(y)
    A = np.column_stack([np.ones(n), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = n - A.shape[1]
    sigma = float(np.sqrt(resid @ resid / dof)) if dof > 0 else np.nan
    tss = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / tss if tss > 0 else np.nan
    return coef, sigma, r2, n


def _regressors(md: MarketData, model: str, sector: str) -> tuple[pd.DataFrame, pd.Series | None]:
    """Regressor frame and the series subtracted from the stock return (rf for factor models)."""
    if model == "market":
        return md.ret[["SPY"]], None
    if model == "sector":
        return md.ret[[SECTOR_ETF[sector]]], None
    if model == "ff4":
        return md.factors[["mkt_rf", "smb", "hml", "mom"]], md.factors["rf"]
    raise ValueError(model)


def event_abnormal_returns(md: MarketData, ticker: str, sector: str, day0: pd.Timestamp, model: str = "market") -> dict | None:
    """Abnormal returns for relative days REL_MIN..REL_MAX of one event. None if day 0 is outside the price data."""
    sessions = md.sessions
    if day0 not in sessions:
        return None
    i = sessions.get_loc(day0)
    if i + EST_START < 1 or i + REL_MAX >= len(sessions):
        return None
    X, rf = _regressors(md, model, sector)
    y_all = md.ret_est[ticker] - (rf if rf is not None else 0.0)

    est = slice(i + EST_START, i + EST_END + 1)
    coef, sigma, r2, n = ols(y_all.iloc[est].to_numpy(), X.iloc[est].to_numpy())
    out = {"model": model, "n_est": n, "alpha": coef[0], "beta": coef[1], "sigma": sigma, "r2": r2,
           "est_ok": n >= EST_MIN_OBS}
    win = slice(i + REL_MIN, i + REL_MAX + 1)
    Xw = X.iloc[win].to_numpy()
    expected = coef[0] + Xw @ coef[1:]
    ar = y_all.iloc[win].to_numpy() - expected
    out["rel"] = np.arange(REL_MIN, REL_MAX + 1)
    out["ar"] = ar
    out["ret"] = md.ret_est[ticker].iloc[win].to_numpy()
    out["expected"] = expected + (rf.iloc[win].to_numpy() if rf is not None else 0.0)
    return out


def cumulate(rel: np.ndarray, ar: np.ndarray) -> dict:
    """All reported CAR windows. A window containing a missing AR is NaN (never silently shortened)."""
    res = {}
    for name, a, b in CAR_WINDOWS:
        seg = ar[(rel >= a) & (rel <= b)]
        res[name] = float(seg.sum()) if len(seg) == b - a + 1 and not np.isnan(seg).any() else np.nan
    return res


def buy_and_hold(rel: np.ndarray, ret: np.ndarray, expected: np.ndarray, a: int, b: int) -> float:
    """BHAR[a,b] = prod(1 + R) - prod(1 + E[R])."""
    m = (rel >= a) & (rel <= b)
    if np.isnan(ret[m]).any() or np.isnan(expected[m]).any():
        return np.nan
    return float(np.prod(1 + ret[m]) - np.prod(1 + expected[m]))


def abnormal_volume(md: MarketData, ticker: str, day0: pd.Timestamp) -> np.ndarray | None:
    """log(volume_t) minus the mean log volume of the estimation window, for relative days REL_MIN..REL_MAX."""
    sessions = md.sessions
    i = sessions.get_loc(day0)
    v = md.volume[ticker].replace(0, np.nan)
    lv = np.log(v.to_numpy(dtype=float))
    base = np.nanmean(lv[i + EST_START: i + EST_END + 1])
    return lv[i + REL_MIN: i + REL_MAX + 1] - base
