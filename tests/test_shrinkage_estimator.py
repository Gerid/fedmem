"""Tests for the anisotropic shrinkage estimator."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.estimators.shrinkage import (
    ShrinkageEstimator,
    compute_effective_rank,
    compute_effective_rank_from_covariance,
    compute_shrinkage_lambda,
    estimate_sigma_B2,
)


class TestEffectiveRank:
    def test_isotropic_data_near_d(self):
        rng = np.random.RandomState(42)
        d = 20
        X = rng.randn(1000, d)
        r_eff = compute_effective_rank(X)
        assert 0.8 * d <= r_eff <= 1.1 * d

    def test_spiked_data_much_less_than_d(self):
        rng = np.random.RandomState(42)
        d = 100
        X = rng.randn(500, d)
        X[:, :3] *= 20
        r_eff = compute_effective_rank(X)
        assert r_eff < 0.3 * d

    def test_rank1_data(self):
        rng = np.random.RandomState(42)
        v = rng.randn(50, 1)
        X = v @ rng.randn(1, 10) + 1e-6 * rng.randn(50, 10)
        r_eff = compute_effective_rank(X)
        assert r_eff < 3.0

    def test_minimum_samples_raises(self):
        with pytest.raises(ValueError):
            compute_effective_rank(np.array([[1.0]]))

    def test_from_covariance_matches(self):
        rng = np.random.RandomState(42)
        X = rng.randn(500, 20)
        X[:, :5] *= 10
        cov = np.cov(X, rowvar=False)
        r_from_X = compute_effective_rank(X)
        r_from_cov = compute_effective_rank_from_covariance(cov)
        assert abs(r_from_X - r_from_cov) < 1.0


class TestShrinkageLambda:
    def test_high_separation_low_lambda(self):
        lam = compute_shrinkage_lambda(
            sigma2=1.0, sigma_B2=10.0, K=20, C=4, n=200, d_eff=50,
        )
        assert lam < 0.1

    def test_low_separation_high_lambda(self):
        lam = compute_shrinkage_lambda(
            sigma2=1.0, sigma_B2=0.001, K=20, C=4, n=200, d_eff=50,
        )
        assert lam > 0.9

    def test_lambda_in_unit_interval(self):
        for sigma_B2 in [0.0, 0.01, 1.0, 100.0]:
            lam = compute_shrinkage_lambda(
                sigma2=1.0, sigma_B2=sigma_B2, K=10, C=2, n=100, d_eff=20,
            )
            assert 0.0 <= lam <= 1.0

    def test_anisotropic_gives_different_lambda(self):
        lam_d = compute_shrinkage_lambda(
            sigma2=0.5, sigma_B2=0.1, K=12, C=8, n=400, d_eff=128,
        )
        lam_reff = compute_shrinkage_lambda(
            sigma2=0.5, sigma_B2=0.1, K=12, C=8, n=400, d_eff=22,
        )
        assert lam_reff < lam_d


class TestEstimateSigmaB2:
    def test_identical_concepts_zero(self):
        w = np.ones(10)
        sigma_B2 = estimate_sigma_B2([w, w, w], sigma2=0.5, K=12, C=3, n=100)
        assert sigma_B2 == 0.0

    def test_separated_concepts_positive(self):
        w1 = np.zeros(10)
        w2 = np.ones(10) * 5.0
        sigma_B2 = estimate_sigma_B2([w1, w2], sigma2=0.1, K=10, C=2, n=100)
        assert sigma_B2 > 0.0


class TestShrinkageEstimator:
    def test_fit_predict_returns_result(self):
        rng = np.random.RandomState(42)
        d = 20
        X = rng.randn(200, d)
        X[:, :3] *= 5
        concept_ests = [rng.randn(d) for _ in range(4)]
        global_est = np.mean(concept_ests, axis=0)

        est = ShrinkageEstimator(use_anisotropic=True)
        result = est.fit_predict(
            concept_ests, global_est,
            sigma2=1.0, K=12, C=4, n=50,
            feature_matrix=X,
        )

        assert len(result.shrunk_estimates) == 4
        assert 0.0 <= result.lambda_value <= 1.0
        assert result.r_eff < d

    def test_isotropic_vs_anisotropic_differ(self):
        rng = np.random.RandomState(42)
        d = 50
        X = rng.randn(300, d)
        X[:, :3] *= 15

        concept_ests = [rng.randn(d) * 2 for _ in range(3)]
        global_est = np.mean(concept_ests, axis=0)

        est_iso = ShrinkageEstimator(use_anisotropic=False)
        est_aniso = ShrinkageEstimator(use_anisotropic=True)

        r_iso = est_iso.fit_predict(
            concept_ests, global_est, sigma2=1.0, K=9, C=3, n=100,
        )
        r_aniso = est_aniso.fit_predict(
            concept_ests, global_est, sigma2=1.0, K=9, C=3, n=100,
            feature_matrix=X,
        )

        assert r_iso.lambda_value != r_aniso.lambda_value

    def test_grid_search_finds_reasonable_lambda(self):
        rng = np.random.RandomState(42)
        d = 10
        true_weights = [rng.randn(d) * 2 for _ in range(3)]
        concept_ests = [w + rng.randn(d) * 0.5 for w in true_weights]
        global_est = np.mean(concept_ests, axis=0)

        est = ShrinkageEstimator()
        lam_star, mse_star = est.grid_search_lambda(
            concept_ests, global_est, true_weights,
        )
        assert 0.0 <= lam_star <= 1.0
        assert mse_star >= 0.0
