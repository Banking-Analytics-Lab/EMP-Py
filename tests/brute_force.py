"""Reference values for the tests, computed from the definition of EMP.

These functions do not use the convex hull. They take the maximum profit over all
ROC points for each value of the random parameter, and average over a grid.
Label 1 = default (credit) or churner (churn); higher score = more likely class 1.
"""
import numpy as np
from scipy.stats import beta as beta_distribution


def roc_points(scores, labels):
    """All ROC points (FPR, TPR) at distinct thresholds, from (0,0) to (1,1)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(y)[last]
    fp = np.cumsum(1 - y)[last]
    tpr = np.r_[0.0, tp / y.sum()]
    fpr = np.r_[0.0, fp / (len(y) - y.sum())]
    return fpr, tpr


def credit(scores, labels, p_0=0.55, p_1=0.1, ROI=0.2644, grid=20001):
    """EMPC and its fraction: E over lambda of the maximum profit over all thresholds."""
    y = np.asarray(labels, int)
    pi_0, pi_1 = y.mean(), 1 - y.mean()
    F_1, F_0 = roc_points(scores, labels)

    def best(lam):
        profit = lam * pi_0 * F_0 - ROI * pi_1 * F_1
        k = int(np.argmax(profit))
        return profit[k], pi_0 * F_0[k] + pi_1 * F_1[k]

    lambdas = (np.arange(grid) + 0.5) / grid  # midpoint rule on (0,1)
    values = np.array([best(lam) for lam in lambdas])
    b_0, f_0 = best(0.0)
    b_1, f_1 = best(1.0)
    alpha = 1 - p_0 - p_1
    return (p_0 * b_0 + p_1 * b_1 + alpha * values[:, 0].mean(),
            p_0 * f_0 + p_1 * f_1 + alpha * values[:, 1].mean())


def churn(scores, labels, alpha=6, beta=14, clv=200, d=10, f=1, grid=20001):
    """MP, MP fraction, EMP and EMP fraction, with gamma ~ Beta(alpha, beta).

    The expectation over gamma uses the midpoints of a uniform grid on the
    quantile scale, so each grid point has the same probability.
    """
    y = np.asarray(labels, int)
    pi_0, pi_1 = y.mean(), 1 - y.mean()
    F_1, F_0 = roc_points(scores, labels)
    delta, phi = d / clv, f / clv

    def best(gamma):
        profit = clv * ((gamma * (1 - delta) - phi) * pi_0 * F_0 - (delta + phi) * pi_1 * F_1)
        k = int(np.argmax(profit))
        return profit[k], pi_0 * F_0[k] + pi_1 * F_1[k]

    gammas = beta_distribution.ppf((np.arange(grid) + 0.5) / grid, alpha, beta)
    values = np.array([best(g) for g in gammas])
    mp, mp_fraction = best(alpha / (alpha + beta))
    return mp, mp_fraction, values[:, 0].mean(), values[:, 1].mean()


def credit_like(n, seed, base=-3.2, slope=1.3):
    """Scores and labels with a default rate near 6% when base = -3.2."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    y = (rng.random(n) < 1 / (1 + np.exp(-(base + slope * z)))).astype(int)
    s = 1 / (1 + np.exp(-(base + slope * z + rng.normal(scale=0.8, size=n))))
    return s, y
