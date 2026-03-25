"""Tests for core statistical functions in models.py."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pytest
from models import (
    diebold_mariano_test,
    clark_west_test,
    holm_bonferroni,
    conformal_prediction_intervals,
    r2_oos,
    ForecastResult,
)


# -----------------------------------------------------------------------
# Diebold-Mariano test
# -----------------------------------------------------------------------

class TestDieboldMariano:
    def test_identical_errors_gives_nan(self):
        """Identical errors → zero loss differential variance → NaN (correct)."""
        e = np.random.default_rng(42).normal(0, 1, 50)
        stat, pval = diebold_mariano_test(e, e.copy())
        assert np.isnan(stat) and np.isnan(pval)

    def test_clearly_better_model(self):
        rng = np.random.default_rng(42)
        e1 = rng.normal(0, 0.5, 100)  # small errors
        e2 = rng.normal(0, 3.0, 100)  # large errors
        stat, pval = diebold_mariano_test(e1, e2)
        assert stat < 0  # negative = model 1 has smaller loss
        assert pval < 0.05

    def test_two_sided_pvalue(self):
        """p-value should be symmetric: DM(e1,e2) test stat = -DM(e2,e1)."""
        rng = np.random.default_rng(7)
        e1 = rng.normal(0, 1, 60)
        e2 = rng.normal(0, 2, 60)
        stat1, pval1 = diebold_mariano_test(e1, e2)
        stat2, pval2 = diebold_mariano_test(e2, e1)
        assert stat1 == pytest.approx(-stat2, abs=1e-10)
        assert pval1 == pytest.approx(pval2, abs=1e-10)

    def test_too_few_observations(self):
        e = np.array([1.0, 2.0, 3.0])
        stat, pval = diebold_mariano_test(e, e)
        assert np.isnan(stat) and np.isnan(pval)


# -----------------------------------------------------------------------
# Clark-West test
# -----------------------------------------------------------------------

class TestClarkWest:
    def test_returns_nan_for_few_obs(self):
        e = np.ones(5)
        stat, pval = clark_west_test(e, e, e, e)
        assert np.isnan(stat) and np.isnan(pval)

    def test_one_sided_pvalue_in_unit_range(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 10, 50)
        e_unrest = rng.normal(0, 1, 50)
        e_rest = rng.normal(0, 5, 50)
        y_pred_rest = y - e_rest
        stat, pval = clark_west_test(e_unrest, e_rest, y, y_pred_rest)
        assert 0 <= pval <= 1

    def test_better_unrestricted_model_gives_low_pval(self):
        rng = np.random.default_rng(42)
        y = rng.normal(100, 10, 80)
        e_unrest = rng.normal(0, 0.5, 80)  # much better
        e_rest = rng.normal(0, 5.0, 80)
        y_pred_rest = y - e_rest
        stat, pval = clark_west_test(e_unrest, e_rest, y, y_pred_rest)
        assert stat > 0
        assert pval < 0.05


# -----------------------------------------------------------------------
# Holm-Bonferroni correction
# -----------------------------------------------------------------------

class TestHolmBonferroni:
    def test_single_pvalue_unchanged(self):
        result = holm_bonferroni(np.array([0.03]))
        assert result[0] == pytest.approx(0.03)

    def test_known_example(self):
        # 3 p-values: 0.01, 0.04, 0.03
        # sorted: 0.01, 0.03, 0.04
        # adjusted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04
        # monotonicity: 0.03, 0.06, 0.06
        pvals = np.array([0.01, 0.04, 0.03])
        adj = holm_bonferroni(pvals)
        assert adj[0] == pytest.approx(0.03)   # 0.01 * 3
        assert adj[2] == pytest.approx(0.06)   # 0.03 * 2
        assert adj[1] == pytest.approx(0.06)   # max(0.04*1, 0.06) monotonicity

    def test_never_exceeds_one(self):
        pvals = np.array([0.5, 0.6, 0.9])
        adj = holm_bonferroni(pvals)
        assert np.all(adj <= 1.0)

    def test_empty_input(self):
        adj = holm_bonferroni(np.array([]))
        assert len(adj) == 0


# -----------------------------------------------------------------------
# Conformal Prediction Intervals
# -----------------------------------------------------------------------

class TestConformalIntervals:
    def _make_models(self, n=100, noise=1.0, bias=0.0, seed=42):
        rng = np.random.default_rng(seed)
        actuals = rng.normal(100, 10, n)
        preds = actuals + rng.normal(bias, noise, n)
        res = ForecastResult("Test")
        res.predictions = preds.tolist()
        res.actuals = actuals.tolist()
        res.rmse = np.sqrt(np.mean((actuals - preds) ** 2))
        return {"Test": res}

    def test_coverage_near_nominal(self):
        models = self._make_models(n=200, noise=1.0)
        df = conformal_prediction_intervals(models, alpha=0.10)
        assert len(df) == 1
        assert df.iloc[0]["empirical_coverage"] >= 0.70  # should be ~0.90

    def test_too_few_obs_skipped(self):
        models = self._make_models(n=5)
        df = conformal_prediction_intervals(models, alpha=0.10)
        assert len(df) == 0

    def test_width_positive(self):
        models = self._make_models(n=100)
        df = conformal_prediction_intervals(models, alpha=0.10)
        assert df.iloc[0]["mean_width"] > 0


# -----------------------------------------------------------------------
# R² Out-of-Sample
# -----------------------------------------------------------------------

class TestR2OOS:
    def test_perfect_model(self):
        y = [1, 2, 3, 4, 5]
        assert r2_oos(y, y, [3, 3, 3, 3, 3]) == pytest.approx(1.0)

    def test_benchmark_equals_model(self):
        y = [1, 2, 3, 4, 5]
        b = [1.1, 2.1, 2.9, 4.1, 5.1]
        assert r2_oos(y, b, b) == pytest.approx(0.0)

    def test_worse_than_benchmark_is_negative(self):
        y = [1, 2, 3, 4, 5]
        bad = [10, 10, 10, 10, 10]  # terrible predictions
        bench = [1.5, 2.5, 3.5, 3.5, 4.5]  # decent benchmark
        assert r2_oos(y, bad, bench) < 0

    def test_nan_handling(self):
        result = r2_oos([1], [1], [1])
        assert np.isnan(result)
