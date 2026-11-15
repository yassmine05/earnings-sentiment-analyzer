"""Unit tests for the event-study engine. The arrays below are constructed test fixtures, not study data."""
import numpy as np
import pandas as pd

from earnings_sentiment import eventstudy as es


def _market_data(n=320, alpha=0.0002, beta=1.3, jump_at=None, jump=0.05, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    mkt = rng.normal(0, 0.01, n)
    stock = alpha + beta * mkt            # exact linear relation: residuals are zero
    if jump_at is not None:
        stock[jump_at] += jump
    ret = pd.DataFrame({"AAA": stock, "SPY": mkt, "XLK": mkt}, index=idx)
    vol = pd.DataFrame({"AAA": np.full(n, 1_000.0), "SPY": 1.0, "XLK": 1.0}, index=idx)
    if jump_at is not None:
        vol.iloc[jump_at, 0] = 1_000.0 * np.e
    ff = pd.DataFrame({"mkt_rf": mkt, "smb": 0.0, "hml": 0.0, "rf": 0.0, "mom": 0.0}, index=idx)
    return es.MarketData(ret=ret, ret_est=ret.copy(), volume=vol, factors=ff), idx


def test_market_model_recovers_parameters_and_isolates_the_event_day():
    md, idx = _market_data(jump_at=280)
    r = es.event_abnormal_returns(md, "AAA", "Information Technology", idx[280], "market")
    assert r["n_est"] == 240 and r["est_ok"]
    assert abs(r["beta"] - 1.3) < 1e-9 and abs(r["alpha"] - 0.0002) < 1e-9
    ar = dict(zip(r["rel"], r["ar"]))
    assert abs(ar[0] - 0.05) < 1e-9
    assert all(abs(v) < 1e-9 for k, v in ar.items() if k != 0)
    cars = es.cumulate(r["rel"], r["ar"])
    assert abs(cars["car_0_1"] - 0.05) < 1e-9 and abs(cars["car_0_5"] - 0.05) < 1e-9
    assert abs(cars["car_1_5"]) < 1e-9 and abs(cars["ar_1"]) < 1e-9 and abs(cars["car_m20_m1"]) < 1e-9


def test_estimation_window_ends_before_the_event():
    # A jump on day -5 must not leak into the estimated parameters (window stops at -11).
    md, idx = _market_data(jump_at=275)
    r = es.event_abnormal_returns(md, "AAA", "Information Technology", idx[280], "market")
    assert abs(r["beta"] - 1.3) < 1e-9
    assert abs(dict(zip(r["rel"], r["ar"]))[-5] - 0.05) < 1e-9


def test_missing_return_makes_the_window_nan_instead_of_shortening_it():
    md, idx = _market_data()
    md.ret_est.iloc[282, 0] = np.nan      # relative day +2
    r = es.event_abnormal_returns(md, "AAA", "Information Technology", idx[280], "market")
    cars = es.cumulate(r["rel"], r["ar"])
    assert np.isnan(cars["car_0_2"]) and np.isnan(cars["car_1_5"])
    assert not np.isnan(cars["car_0_1"])


def test_excluded_day_is_dropped_from_estimation_only():
    md, idx = _market_data(jump_at=200)   # a bad print inside the estimation window
    r_bad = es.event_abnormal_returns(md, "AAA", "Information Technology", idx[280], "market")
    md.ret_est.iloc[200, 0] = np.nan
    r_ok = es.event_abnormal_returns(md, "AAA", "Information Technology", idx[280], "market")
    assert r_ok["n_est"] == 239 and abs(r_ok["beta"] - 1.3) < 1e-9
    assert abs(r_bad["alpha"] - 0.0002) > 1e-5      # the bad print would have biased alpha


def test_abnormal_volume_and_buy_and_hold():
    md, idx = _market_data(jump_at=280)
    av = es.abnormal_volume(md, "AAA", idx[280])
    rel = np.arange(es.REL_MIN, es.REL_MAX + 1)
    assert abs(av[rel == 0][0] - 1.0) < 1e-9 and abs(av[rel == 1][0]) < 1e-9
    ret = np.array([0.10, -0.05]); exp = np.array([0.01, 0.01])
    assert abs(es.buy_and_hold(np.array([0, 1]), ret, exp, 0, 1) - (1.10 * 0.95 - 1.01 * 1.01)) < 1e-12


def test_day_outside_price_data_returns_none():
    md, idx = _market_data()
    assert es.event_abnormal_returns(md, "AAA", "Information Technology", idx[100], "market") is None
    assert es.event_abnormal_returns(md, "AAA", "Information Technology", pd.Timestamp("2030-01-01"), "market") is None
