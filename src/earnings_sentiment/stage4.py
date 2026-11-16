"""Stage 4 estimation, exactly as fixed in docs/stage4_analysis_plan.md.

One function, `estimate`, runs every regression of the plan. The validation on simulated data and the real
run both call it, so the code that is validated is the code that produces the results.

Estimator: OLS with explicit fixed-effect dummies.
Standard errors: cluster-robust (Cameron, Gelbach and Miller 2011 for two-way clustering):
    V = V_a + V_b - V_ab,   V_g = c_g * (X'X)^-1 [ sum over clusters of (X_c'u_c)(X_c'u_c)' ] (X'X)^-1
    c_g = G/(G-1) * (N-1)/(N-K)      (G clusters, N observations, K estimated parameters incl. fixed effects)
p-values and confidence intervals use a t distribution with (smallest number of clusters - 1) degrees of freedom.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

# Fixed effects and controls of each specification (plan, section 2).
SPECS = {
    "A": {"fe": ["firm", "quarter"], "controls": ["runup", "vol"]},
    "B": {"fe": ["sector", "quarter"], "controls": ["runup", "vol", "size"]},
    "C": {"fe": ["quarter"], "controls": ["runup", "vol", "size"]},
}
DRIFT_EXTRA = ["ar0"]                     # day-0 abnormal return: drift family only
CLUSTERINGS = {"twoway": ["firm", "date"], "firm": ["firm"], "quarter": ["quarter"]}


@dataclass
class Fit:
    names: list[str]                      # regressors of interest (the standardised scores)
    coef: np.ndarray                      # in the units of y per 1 sd of the score
    cov: np.ndarray                       # covariance of those coefficients
    n: int
    k: int
    n_clusters: dict
    dof: int
    r2_within: float                      # share of the variance left after FE that the model explains
    ssr: float

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    def table(self, scale: float = 1e4) -> pd.DataFrame:
        t = self.coef / self.se
        crit = stats.t.ppf(0.975, self.dof)
        return pd.DataFrame({
            "regressor": self.names, "coef_bp": self.coef * scale, "se_bp": self.se * scale,
            "ci_low_bp": (self.coef - crit * self.se) * scale, "ci_high_bp": (self.coef + crit * self.se) * scale,
            "t": t, "p": 2 * stats.t.sf(np.abs(t), self.dof), "n": self.n,
            "clusters": "; ".join(f"{k} {v}" for k, v in self.n_clusters.items()), "r2_within": self.r2_within,
        })

    def equal_test(self) -> tuple[float, float]:
        """t-statistic and p-value of H0: first coefficient = second coefficient."""
        d = self.coef[0] - self.coef[1]
        se = np.sqrt(self.cov[0, 0] + self.cov[1, 1] - 2 * self.cov[0, 1])
        return float(d / se), float(2 * stats.t.sf(abs(d / se), self.dof))


def _dummies(s: pd.Series) -> np.ndarray:
    return pd.get_dummies(s.astype(str), drop_first=True).to_numpy(dtype=float)


def cluster_cov(X: np.ndarray, u: np.ndarray, groups: list[np.ndarray], correction: bool = True) -> tuple[np.ndarray, dict]:
    """Cluster-robust covariance of all OLS coefficients for one-way or two-way clustering."""
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    scores = X * u[:, None]

    def one(codes: np.ndarray) -> tuple[np.ndarray, int]:
        g = int(codes.max()) + 1
        S = np.zeros((g, k))
        np.add.at(S, codes, scores)
        c = g / (g - 1) * (n - 1) / (n - k) if correction else 1.0
        return c * bread @ (S.T @ S) @ bread, g

    codes = [pd.factorize(g)[0] for g in groups]
    if len(codes) == 1:
        V, g = one(codes[0])
        return V, {"g0": g}
    Va, ga = one(codes[0])
    Vb, gb = one(codes[1])
    inter = pd.factorize(pd.Series(list(zip(codes[0], codes[1]))))[0]
    Vab, _ = one(inter)
    return Va + Vb - Vab, {"g0": ga, "g1": gb}


def estimate(df: pd.DataFrame, y: str, scores: list[str], spec: str, drift: bool, clustering: str = "twoway",
             correction: bool = True) -> Fit:
    """Run one regression of the plan.

    df must hold: y, the score columns, firm, sector, quarter, date, and the controls runup, vol, size, ar0.
    Scores are standardised over the estimation sample. Returns the coefficients of the scores.
    """
    controls = SPECS[spec]["controls"] + (DRIFT_EXTRA if drift else [])
    need = [y] + scores + controls + SPECS[spec]["fe"] + CLUSTERINGS[clustering]
    d = df.dropna(subset=list(dict.fromkeys(need))).reset_index(drop=True)
    Z = np.column_stack([(d[s] - d[s].mean()) / d[s].std() for s in scores])
    C = d[controls].to_numpy(dtype=float)
    F = np.column_stack([np.ones(len(d))] + [_dummies(d[f]) for f in SPECS[spec]["fe"]])
    X = np.column_stack([Z, C, F])
    yy = d[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    if np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("collinear design")
    u = yy - X @ beta
    V, g = cluster_cov(X, u, [d[c].to_numpy() for c in CLUSTERINGS[clustering]], correction)
    m = len(scores)
    # within R2: against the model with fixed effects only
    bF, *_ = np.linalg.lstsq(F, yy, rcond=None)
    ssr_fe = float(((yy - F @ bF) ** 2).sum())
    ssr = float(u @ u)
    names = dict(zip(["g0", "g1"], CLUSTERINGS[clustering]))
    n_clusters = {names[k]: v for k, v in g.items()}
    cov = V[:m, :m]
    if (np.diag(cov) <= 0).any():
        raise ValueError("two-way covariance is not positive definite for a coefficient of interest")
    return Fit(names=scores, coef=beta[:m], cov=cov, n=len(d), k=X.shape[1], n_clusters=n_clusters,
               dof=min(n_clusters.values()) - 1, r2_within=1 - ssr / ssr_fe, ssr=ssr)


def holm(p: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def quintiles_from_previous_quarter(df: pd.DataFrame, score: str) -> pd.Series:
    """Quintile 1-5 of each event, using the 20/40/60/80th percentiles of the score among the events of the
    PREVIOUS calendar quarter. Missing for the first quarter that has scores and wherever the score is missing."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    quarters = sorted(df.loc[df[score].notna(), "quarter"].unique(), key=lambda q: pd.Period(q, freq="Q"))
    for q in quarters:
        prev = str(pd.Period(q, freq="Q") - 1)
        base = df.loc[(df["quarter"] == prev) & df[score].notna(), score]
        if len(base) < 5:
            continue
        cuts = np.quantile(base, [0.2, 0.4, 0.6, 0.8])
        idx = df.index[(df["quarter"] == q) & df[score].notna()]
        out.loc[idx] = 1 + np.searchsorted(cuts, df.loc[idx, score].to_numpy(), side="right")
    return out
