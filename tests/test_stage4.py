"""Tests of the Stage 4 estimator. All data here are constructed fixtures (random numbers), not study data."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from earnings_sentiment import stage4


def _frame(n_firms=12, n_quarters=10, seed=0, beta=0.01):
    rng = np.random.default_rng(seed)
    rows = []
    for f in range(n_firms):
        for q in range(n_quarters):
            rows.append({"firm": f"F{f}", "sector": f"S{f % 4}", "quarter": str(pd.Period("2019Q1", freq="Q") + q),
                         "date": f"d{q}_{rng.integers(0, 6)}"})
    d = pd.DataFrame(rows)
    n = len(d)
    day = d["date"].map({k: v for k, v in zip(d["date"].unique(), rng.normal(0, 0.01, d["date"].nunique()))})
    d["tone"] = rng.normal(size=n) + d["firm"].map({f"F{i}": i / 5 for i in range(n_firms)})
    d["other"] = 0.5 * d["tone"] + rng.normal(size=n)
    d["runup"], d["vol"], d["size"], d["ar0"] = rng.normal(0, 0.05, n), rng.uniform(0.005, 0.03, n), d["firm"].map(
        {f"F{i}": 3 + i / 3 for i in range(n_firms)}), rng.normal(0, 0.03, n)
    z = (d["tone"] - d["tone"].mean()) / d["tone"].std()
    d["y"] = beta * z + 0.1 * d["runup"] + day + rng.normal(0, 0.03, n)
    return d


def _statsmodels(d, y, scores, fe, controls, groups, correction=True):
    Z = pd.DataFrame({s: (d[s] - d[s].mean()) / d[s].std() for s in scores})
    X = pd.concat([Z, d[controls]] + [pd.get_dummies(d[f], prefix=f, drop_first=True, dtype=float) for f in fe], axis=1)
    X = sm.add_constant(X)
    g = np.column_stack([pd.factorize(d[c])[0] for c in groups])
    g = g[:, 0] if g.shape[1] == 1 else g
    r = sm.OLS(d[y], X).fit(cov_type="cluster", cov_kwds={"groups": g, "use_correction": correction})
    return r.params[scores].to_numpy(), r.bse[scores].to_numpy()


def test_point_estimates_and_cluster_errors_match_statsmodels():
    d = _frame()
    for spec, fe in (("A", ["firm", "quarter"]), ("B", ["sector", "quarter"]), ("C", ["quarter"])):
        for clustering, groups in (("twoway", ["firm", "date"]), ("firm", ["firm"]), ("quarter", ["quarter"])):
            fit = stage4.estimate(d, "y", ["tone"], spec, drift=True, clustering=clustering)
            b, se = _statsmodels(d, "y", ["tone"], fe, stage4.SPECS[spec]["controls"] + ["ar0"], groups)
            assert np.allclose(fit.coef, b, rtol=1e-9, atol=1e-12), (spec, clustering)
            assert np.allclose(fit.se, se, rtol=1e-7), (spec, clustering, fit.se, se)


def test_two_scores_together_and_equality_test():
    d = _frame(seed=3)
    fit = stage4.estimate(d, "y", ["tone", "other"], "A", drift=False)
    b, se = _statsmodels(d, "y", ["tone", "other"], ["firm", "quarter"], ["runup", "vol"], ["firm", "date"])
    assert np.allclose(fit.coef, b) and np.allclose(fit.se, se, rtol=1e-7)
    t, p = fit.equal_test()
    assert np.isfinite(t) and 0 <= p <= 1


def test_scores_are_standardised_inside_the_estimation_sample():
    d = _frame(seed=1)
    a = stage4.estimate(d, "y", ["tone"], "A", drift=False)
    d2 = d.copy(); d2["tone"] = 100 + 7 * d2["tone"]            # a change of units must not change anything
    b = stage4.estimate(d2, "y", ["tone"], "A", drift=False)
    assert np.allclose(a.coef, b.coef) and np.allclose(a.se, b.se)


def test_rows_with_a_missing_control_are_dropped_not_filled():
    d = _frame(seed=2); d.loc[0, "runup"] = np.nan
    assert stage4.estimate(d, "y", ["tone"], "A", drift=False).n == len(d) - 1


def test_holm_adjustment():
    assert np.allclose(stage4.holm([0.01, 0.04]), [0.02, 0.04])
    assert np.allclose(stage4.holm([0.04, 0.01]), [0.04, 0.02])
    assert np.allclose(stage4.holm([0.03, 0.02, 0.9]), [0.06, 0.06, 0.9])


def test_quintile_breakpoints_come_from_the_previous_quarter_only():
    d = pd.DataFrame({"quarter": ["2019Q1"] * 10 + ["2019Q2"] * 5, "s": list(range(10)) + [-5, 2.5, 4.5, 8.5, 100]})
    q = stage4.quintiles_from_previous_quarter(d, "s")
    assert q.iloc[:10].isna().all()                             # first quarter has no previous quarter
    assert q.iloc[10:].tolist() == [1, 2, 3, 5, 5]              # cuts from 2019Q1: 1.8, 3.6, 5.4, 7.2
    d2 = d.copy(); d2.loc[14, "s"] = 1e6                        # a later value cannot move the breakpoints
    assert stage4.quintiles_from_previous_quarter(d2, "s").iloc[10:14].tolist() == [1, 2, 3, 5]
