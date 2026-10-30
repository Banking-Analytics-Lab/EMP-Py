"""Tests for EMP.metrics.

Run from the repository root with: python -m pytest
"""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

import EMP
from EMP.metrics import empChurn, empCreditScoring

import brute_force

README_SCORES = [0.34, 0.44, 0.67, 0.83]
README_CLASSES = [0, 0, 1, 0]

# Tolerances against brute force. The profit is a continuous function of the
# random parameter, so the midpoint rule is accurate. The fraction is a step
# function of it, so its grid error is of the order of 1/grid.
TOL_VALUE = 1e-6
TOL_FRACTION = 1e-3


def assert_credit_matches_brute(s, y, **kw):
    out = empCreditScoring(s, y, **kw)
    ref = brute_force.credit(s, y, **kw)
    assert out.EMPC == pytest.approx(ref[0], abs=TOL_VALUE)
    assert out.EMPC_fraction == pytest.approx(ref[1], abs=TOL_FRACTION)


def assert_churn_matches_brute(s, y, **kw):
    out = empChurn(s, y, **kw)
    ref = brute_force.churn(s, y, **kw)
    scale = kw.get("clv", 200)
    assert out.MP == pytest.approx(ref[0], abs=TOL_VALUE * scale)
    assert out.MP_fraction == pytest.approx(ref[1], abs=TOL_FRACTION)
    assert out.EMP == pytest.approx(ref[2], abs=TOL_VALUE * scale)
    assert out.EMP_fraction == pytest.approx(ref[3], abs=TOL_FRACTION)


# README and test.py examples. The expected values are the output of version 2.0.4.

def test_readme_default_returns_without_printing(capsys):
    out = empCreditScoring(README_SCORES, README_CLASSES)
    assert capsys.readouterr().out == ""
    assert out.EMPC == 0.042063447000000004
    assert out.EMPC_fraction == 0.17873


def test_readme_print_and_return(capsys):
    out = empCreditScoring(README_SCORES, README_CLASSES, print_output=True)
    printed = capsys.readouterr().out
    assert "EMPC:  0.042063447" in printed
    assert out.EMPC == 0.042063447000000004


def test_readme_print_only(capsys):
    out = empCreditScoring(README_SCORES, README_CLASSES, print_output=True, return_output=False)
    assert out is None
    assert "EMP Credit Scoring Output" in capsys.readouterr().out


def test_readme_rounding():
    out = empCreditScoring(README_SCORES, README_CLASSES, rounding=2)
    assert (out.EMPC, out.EMPC_fraction) == (0.04, 0.18)


def test_test_py_example():
    out = EMP.empCreditScoring([.4, .2], [1, 0])
    assert out.EMPC == 0.1375
    assert out.EMPC_fraction == 0.22499999999999998


def test_readme_matches_brute_force():
    assert_credit_matches_brute(README_SCORES, README_CLASSES)
    assert_churn_matches_brute(README_SCORES, README_CLASSES)


# Random and credit-like data against brute force

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_credit_uniform_scores(seed):
    rng = np.random.default_rng(seed)
    s = rng.random(1000)
    y = (rng.random(1000) < s).astype(int)
    assert_credit_matches_brute(s, y)
    assert_credit_matches_brute(s, y, p_0=0.3, p_1=0.2, ROI=0.1)


@pytest.mark.parametrize("seed", [1, 2])
def test_credit_like_data(seed):
    s, y = brute_force.credit_like(5000, seed)
    assert_credit_matches_brute(s, y)
    assert_credit_matches_brute(np.round(s, 2), y)  # many ties


def test_credit_like_data_large():
    s, y = brute_force.credit_like(20000, 3)
    assert_credit_matches_brute(s, y)


@pytest.mark.parametrize("seed", [1, 2])
def test_churn_like_data(seed):
    s, y = brute_force.credit_like(5000, seed, base=-1.5)
    assert_churn_matches_brute(s, y)
    assert_churn_matches_brute(s, y, alpha=2, beta=5, clv=100, d=5, f=2)


@pytest.mark.parametrize("seed", range(20))
def test_small_samples_with_ties(seed):
    # Small samples with ties give collinear ROC points. Version 2.0.4 failed on some of them.
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    y = rng.integers(0, 2, n)
    y[:2] = [0, 1]
    s = rng.integers(0, 4, n) / 3
    assert_credit_matches_brute(s, y)
    assert_churn_matches_brute(s, y)


def test_worse_than_random():
    s, y = brute_force.credit_like(5000, 4)
    assert_credit_matches_brute(1 - s, y)


def test_perfect_separation():
    s, y = brute_force.credit_like(5000, 5)
    assert_credit_matches_brute(y * 0.9 + 0.05, y)
    assert_churn_matches_brute(y * 0.9 + 0.05, y)


# Edge cases that failed in version 2.0.4

def test_diagonal_roc_with_ties():
    # scipy's ConvexHull raised QhullError here, because all ROC points lie on one line
    s, y = [0.3, 0.3, 0.7, 0.7], [0, 1, 0, 1]
    out = empCreditScoring(s, y)
    # Closed form for the diagonal: pi_0 = pi_1 = 1/2, so lambda = ROI on the only segment
    ROI, alpha, p_1 = 0.2644, 0.35, 0.1
    expected = alpha * (1 - ROI) * (0.5 * (1 + ROI) / 2 - ROI * 0.5) + p_1 * (0.5 - ROI * 0.5)
    assert out.EMPC == pytest.approx(expected, abs=1e-15)
    assert out.EMPC == pytest.approx(0.084127, abs=5e-7)
    assert_credit_matches_brute(s, y)
    assert_churn_matches_brute(s, y)


def test_constant_scores():
    y = (np.arange(1000) % 10 == 0).astype(int)
    s = np.full(1000, 0.5)
    assert_credit_matches_brute(s, y)
    assert_churn_matches_brute(s, y)


def test_two_observations():
    assert_credit_matches_brute([0.4, 0.2], [1, 0])
    assert_credit_matches_brute([0.2, 0.4], [1, 0])
    assert_credit_matches_brute([0.3, 0.3], [1, 0])


@pytest.mark.parametrize("labels", [[0, 0, 0], [1, 1, 1], []])
def test_one_class_raises(labels):
    with pytest.raises(ValueError, match="must contain both classes"):
        empCreditScoring([0.1, 0.2, 0.3][:len(labels)], labels)
    with pytest.raises(ValueError, match="must contain both classes"):
        empChurn([0.1, 0.2, 0.3][:len(labels)], labels)


def test_more_than_two_classes_raises():
    with pytest.raises(ValueError, match="More than 2 classes"):
        empCreditScoring([0.1, 0.2, 0.3], [0, 1, 2])


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="not equal"):
        empCreditScoring([0.1, 0.2, 0.3], [0, 1])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_non_finite_scores_raise(bad):
    with pytest.raises(ValueError, match="NaN or infinite"):
        empCreditScoring([0.1, bad, 0.3], [0, 1, 1])


def test_non_numeric_scores_raise():
    with pytest.raises(ValueError, match="must be numeric"):
        empCreditScoring(["a", "b"], [0, 1])


def test_two_dimensional_input_raises():
    with pytest.raises(ValueError, match="one-dimensional"):
        empCreditScoring([[0.1, 0.2], [0.3, 0.4]], [[0, 1], [1, 0]])


def test_invalid_input_does_not_exit():
    # Version 2.0.4 called sys.exit, which ends the Python process
    try:
        empCreditScoring([0.1, 0.2, 0.3], [0, 1, 2])
    except SystemExit:  # pragma: no cover
        pytest.fail("invalid input raised SystemExit")
    except ValueError:
        pass


# Scores are not restricted to [0,1]

def test_only_the_order_of_scores_matters():
    s, y = brute_force.credit_like(5000, 6)
    reference = empCreditScoring(s, y)
    assert empCreditScoring(rankdata(s), y) == reference
    assert empCreditScoring(1000 * rankdata(s) - 5e5, y) == reference
    assert empChurn(rankdata(s), y) == empChurn(s, y)


def test_logits_and_scorecard_points():
    s, y = brute_force.credit_like(5000, 7)
    logits = np.log(s / (1 - s))
    assert_credit_matches_brute(logits, y)
    points = np.round(600 - 50 * logits / np.log(2))  # higher points = lower risk
    assert_credit_matches_brute(-points, y)


# Input types

@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")  # from np.asmatrix
def test_input_types_give_the_same_result():
    s, y = brute_force.credit_like(3000, 8, base=-1.5)
    reference = empCreditScoring(s, y)
    assert empCreditScoring(list(s), list(y)) == reference
    assert empCreditScoring(pd.Series(s), pd.Series(y)) == reference
    assert empCreditScoring(pd.Series(s, index=np.arange(3000)[::-1]), pd.Series(y)) == reference
    assert empCreditScoring(np.asmatrix(s), np.asmatrix(y)) == reference
    assert empCreditScoring(s.reshape(-1, 1), y.reshape(-1, 1)) == reference
    assert empCreditScoring(pd.DataFrame({"s": s})[["s"]], y) == reference
    assert empCreditScoring(s, y.astype(bool)) == reference
    assert empCreditScoring(s, np.where(y == 1, 1, -1)) == reference
    assert empCreditScoring(s.astype(np.float32), y) == empCreditScoring(s.astype(np.float32).astype(float), y)


# Warnings

def test_no_warnings_on_ordinary_data():
    s, y = brute_force.credit_like(5000, 9)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        empCreditScoring(s, y)
        empChurn(s, y)
