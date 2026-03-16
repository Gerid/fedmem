"""Tests for the Gibbs posterior concept assignment module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.gibbs import (
    GibbsPosterior,
    PosteriorAssignment,
    TransitionPrior,
    _log_sum_exp,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fingerprint(
    mean: np.ndarray,
    n_samples: int = 50,
    n_classes: int = 2,
    label_dist: np.ndarray | None = None,
    seed: int = 0,
) -> ConceptFingerprint:
    """Create a ConceptFingerprint populated with synthetic data.

    Generates data clustered around ``mean`` so the fingerprint's
    internal statistics reflect that centre.
    """
    rng = np.random.RandomState(seed)
    n_features = len(mean)
    fp = ConceptFingerprint(n_features=n_features, n_classes=n_classes)

    if label_dist is None:
        label_dist = np.ones(n_classes) / n_classes

    X = rng.randn(n_samples, n_features) * 0.1 + mean
    y = rng.choice(n_classes, size=n_samples, p=label_dist)
    fp.update(X, y)
    return fp


# ---------------------------------------------------------------------------
# TransitionPrior
# ---------------------------------------------------------------------------

class TestTransitionPrior:
    """Tests for the sticky HMM transition prior."""

    def test_kappa_validation(self) -> None:
        with pytest.raises(ValueError, match="kappa must be in"):
            TransitionPrior(kappa=0.0)
        with pytest.raises(ValueError, match="kappa must be in"):
            TransitionPrior(kappa=1.0)
        with pytest.raises(ValueError, match="kappa must be in"):
            TransitionPrior(kappa=-0.1)

    def test_uniform_prior_at_t0(self) -> None:
        prior = TransitionPrior(kappa=0.8)
        ids = [0, 1, 2]
        for cid in ids:
            lp = prior.log_transition(None, cid, ids)
            assert lp == pytest.approx(-math.log(3))

    def test_sticky_self_transition(self) -> None:
        kappa = 0.9
        prior = TransitionPrior(kappa=kappa)
        lp = prior.log_transition(0, 0, [0, 1, 2])
        assert lp == pytest.approx(math.log(kappa))

    def test_non_self_transition(self) -> None:
        kappa = 0.9
        prior = TransitionPrior(kappa=kappa)
        lp = prior.log_transition(0, 1, [0, 1, 2])
        expected = math.log((1.0 - kappa) / 2)
        assert lp == pytest.approx(expected)

    def test_single_concept_always_stays(self) -> None:
        prior = TransitionPrior(kappa=0.8)
        lp = prior.log_transition(0, 0, [0])
        assert lp == pytest.approx(0.0)

    def test_empty_concepts_raises(self) -> None:
        prior = TransitionPrior(kappa=0.8)
        with pytest.raises(ValueError, match="non-empty"):
            prior.log_transition(0, 0, [])

    def test_probabilities_sum_to_one(self) -> None:
        prior = TransitionPrior(kappa=0.7)
        ids = [0, 1, 2, 3]
        probs = [math.exp(prior.log_transition(1, cid, ids)) for cid in ids]
        assert sum(probs) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# GibbsPosterior — construction
# ---------------------------------------------------------------------------

class TestGibbsPosteriorInit:
    """Tests for GibbsPosterior parameter validation."""

    def test_omega_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="omega must be > 0"):
            GibbsPosterior(omega=0.0)
        with pytest.raises(ValueError, match="omega must be > 0"):
            GibbsPosterior(omega=-1.0)

    def test_novelty_threshold_bounds(self) -> None:
        with pytest.raises(ValueError, match="novelty_threshold"):
            GibbsPosterior(novelty_threshold=0.0)
        with pytest.raises(ValueError, match="novelty_threshold"):
            GibbsPosterior(novelty_threshold=1.0)

    def test_default_construction(self) -> None:
        gp = GibbsPosterior()
        assert gp.omega == 1.0
        assert gp.novelty_threshold == 0.3
        assert isinstance(gp.transition_prior, TransitionPrior)


# ---------------------------------------------------------------------------
# GibbsPosterior — compute_loss
# ---------------------------------------------------------------------------

class TestComputeLoss:
    """Tests for the loss function ℓ(o, m_k)."""

    def test_identical_fingerprints_low_loss(self) -> None:
        gp = GibbsPosterior()
        fp = _make_fingerprint(np.array([1.0, 2.0]), seed=42)
        loss = gp.compute_loss(fp, fp)
        # Self-similarity should be high → loss should be low
        assert 0.0 <= loss <= 0.3

    def test_very_different_fingerprints_high_loss(self) -> None:
        gp = GibbsPosterior()
        fp_a = _make_fingerprint(
            np.array([0.0, 0.0]), label_dist=np.array([0.9, 0.1]), seed=10,
        )
        fp_b = _make_fingerprint(
            np.array([10.0, 10.0]), label_dist=np.array([0.1, 0.9]), seed=20,
        )
        loss = gp.compute_loss(fp_a, fp_b)
        assert loss > 0.5

    def test_loss_in_unit_interval(self) -> None:
        gp = GibbsPosterior()
        fp_a = _make_fingerprint(np.array([0.0, 0.0]), seed=1)
        fp_b = _make_fingerprint(np.array([5.0, 5.0]), seed=2)
        loss = gp.compute_loss(fp_a, fp_b)
        assert 0.0 <= loss <= 1.0


# ---------------------------------------------------------------------------
# GibbsPosterior — compute_posterior
# ---------------------------------------------------------------------------

class TestComputePosterior:
    """Tests for the full posterior computation."""

    def test_empty_library_raises(self) -> None:
        gp = GibbsPosterior()
        obs = _make_fingerprint(np.array([0.0, 0.0]))
        with pytest.raises(ValueError, match="non-empty"):
            gp.compute_posterior(obs, {})

    def test_single_concept_gets_probability_one(self) -> None:
        gp = GibbsPosterior()
        fp = _make_fingerprint(np.array([1.0, 2.0]), seed=42)
        result = gp.compute_posterior(fp, {0: fp}, prev_concept_id=None)

        assert result.map_concept_id == 0
        assert result.probabilities[0] == pytest.approx(1.0)
        assert result.entropy == pytest.approx(0.0, abs=1e-10)
        assert not result.is_novel

    def test_probabilities_sum_to_one(self) -> None:
        gp = GibbsPosterior(omega=2.0)
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([5.0, 5.0]), seed=20),
            2: _make_fingerprint(np.array([10.0, 10.0]), seed=30),
        }
        obs = _make_fingerprint(np.array([0.1, 0.1]), seed=40)
        result = gp.compute_posterior(obs, library)

        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0)

    def test_map_matches_closest_concept(self) -> None:
        gp = GibbsPosterior(omega=5.0)
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([10.0, 10.0]), seed=20),
        }
        obs = _make_fingerprint(np.array([0.1, 0.1]), seed=40)
        result = gp.compute_posterior(obs, library, prev_concept_id=None)

        assert result.map_concept_id == 0
        assert result.probabilities[0] > result.probabilities[1]

    def test_stickiness_biases_toward_previous(self) -> None:
        """With two equidistant concepts, stickiness should bias toward prev."""
        gp = GibbsPosterior(
            omega=0.01,  # very low omega → likelihood almost uniform
            transition_prior=TransitionPrior(kappa=0.95),
        )
        fp = _make_fingerprint(np.array([5.0, 5.0]), seed=42)
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([10.0, 10.0]), seed=20),
        }

        # With prev=0, should bias toward 0
        result_0 = gp.compute_posterior(fp, library, prev_concept_id=0)
        assert result_0.probabilities[0] > result_0.probabilities[1]

        # With prev=1, should bias toward 1
        result_1 = gp.compute_posterior(fp, library, prev_concept_id=1)
        assert result_1.probabilities[1] > result_1.probabilities[0]

    def test_high_omega_sharpens_posterior(self) -> None:
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([10.0, 10.0]), seed=20),
        }
        obs = _make_fingerprint(np.array([0.2, 0.2]), seed=40)

        gp_low = GibbsPosterior(omega=0.5)
        gp_high = GibbsPosterior(omega=10.0)

        result_low = gp_low.compute_posterior(obs, library)
        result_high = gp_high.compute_posterior(obs, library)

        # Higher omega → lower entropy (more peaked)
        assert result_high.entropy < result_low.entropy

    def test_novelty_detection_when_all_bad(self) -> None:
        """When observation is far from all concepts, is_novel should be True."""
        gp = GibbsPosterior(omega=5.0, novelty_threshold=0.8)
        library = {
            0: _make_fingerprint(
                np.array([0.0, 0.0]), label_dist=np.array([0.9, 0.1]), seed=10,
            ),
            1: _make_fingerprint(
                np.array([0.0, 0.0]), label_dist=np.array([0.9, 0.1]), seed=20,
            ),
        }
        obs = _make_fingerprint(
            np.array([50.0, 50.0]), label_dist=np.array([0.1, 0.9]), seed=40,
        )
        result = gp.compute_posterior(obs, library)

        # With high novelty_threshold and spread posterior, should detect novelty
        # (since neither concept dominates strongly when observation is far from both)
        # The MAP probability should be below the threshold
        assert result.probabilities[result.map_concept_id] < 0.8

    def test_novelty_not_triggered_for_good_match(self) -> None:
        gp = GibbsPosterior(omega=5.0, novelty_threshold=0.3)
        fp = _make_fingerprint(np.array([1.0, 2.0]), seed=42)
        result = gp.compute_posterior(fp, {0: fp})

        assert not result.is_novel

    def test_posterior_assignment_has_log_likelihood(self) -> None:
        gp = GibbsPosterior()
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([5.0, 5.0]), seed=20),
        }
        obs = _make_fingerprint(np.array([0.1, 0.1]), seed=40)
        result = gp.compute_posterior(obs, library)

        assert set(result.log_likelihood.keys()) == {0, 1}
        # Log-likelihoods should be non-positive (since loss >= 0 and omega > 0)
        for ll in result.log_likelihood.values():
            assert ll <= 0.0


# ---------------------------------------------------------------------------
# GibbsPosterior — soft_weights
# ---------------------------------------------------------------------------

class TestSoftWeights:
    """Tests for soft aggregation weight extraction."""

    def test_weights_sum_to_one(self) -> None:
        gp = GibbsPosterior()
        fp = _make_fingerprint(np.array([0.0, 0.0]), seed=42)
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([5.0, 5.0]), seed=20),
        }
        result = gp.compute_posterior(fp, library)
        weights = gp.soft_weights(result)

        assert sum(weights.values()) == pytest.approx(1.0)

    def test_weights_subset(self) -> None:
        gp = GibbsPosterior()
        fp = _make_fingerprint(np.array([0.0, 0.0]), seed=42)
        library = {
            0: _make_fingerprint(np.array([0.0, 0.0]), seed=10),
            1: _make_fingerprint(np.array([5.0, 5.0]), seed=20),
            2: _make_fingerprint(np.array([10.0, 10.0]), seed=30),
        }
        result = gp.compute_posterior(fp, library)
        weights = gp.soft_weights(result, concept_ids=[0, 1])

        assert set(weights.keys()) == {0, 1}
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_weights_missing_concept_gets_zero(self) -> None:
        gp = GibbsPosterior()
        assignment = PosteriorAssignment(
            probabilities={0: 0.7, 1: 0.3},
            map_concept_id=0,
            is_novel=False,
            entropy=0.5,
        )
        weights = gp.soft_weights(assignment, concept_ids=[0, 1, 99])
        assert weights[99] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _log_sum_exp
# ---------------------------------------------------------------------------

class TestLogSumExp:
    """Tests for the numerically stable log-sum-exp utility."""

    def test_basic(self) -> None:
        vals = np.array([1.0, 2.0, 3.0])
        expected = math.log(math.exp(1) + math.exp(2) + math.exp(3))
        assert _log_sum_exp(vals) == pytest.approx(expected)

    def test_large_values(self) -> None:
        """Should not overflow with large values."""
        vals = np.array([1000.0, 1001.0, 1002.0])
        result = _log_sum_exp(vals)
        assert np.isfinite(result)
        # Should be close to 1002 + log(exp(-2) + exp(-1) + 1)
        expected = 1002.0 + math.log(math.exp(-2) + math.exp(-1) + 1.0)
        assert result == pytest.approx(expected)

    def test_single_element(self) -> None:
        assert _log_sum_exp(np.array([5.0])) == pytest.approx(5.0)

    def test_negative_values(self) -> None:
        vals = np.array([-10.0, -20.0, -30.0])
        result = _log_sum_exp(vals)
        expected = -10.0 + math.log(1.0 + math.exp(-10) + math.exp(-20))
        assert result == pytest.approx(expected)
