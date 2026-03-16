"""Tests for TwoPhaseFedProTrack protocol."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior.two_phase_protocol import (
    PhaseAResult,
    PhaseBResult,
    TwoPhaseConfig,
    TwoPhaseFedProTrack,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fp(n_features: int = 2, n_classes: int = 2, n_samples: int = 30,
             mean_shift: float = 0.0, seed: int = 0) -> ConceptFingerprint:
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features) + mean_shift
    y = rng.randint(0, n_classes, size=n_samples)
    fp = ConceptFingerprint(n_features, n_classes)
    fp.update(X, y)
    return fp


def _make_model_params(n_features: int = 2, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    return {
        "coef": rng.randn(1, n_features).astype(np.float64),
        "intercept": rng.randn(1).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# TwoPhaseConfig
# ---------------------------------------------------------------------------

class TestTwoPhaseConfig:
    def test_defaults(self) -> None:
        cfg = TwoPhaseConfig()
        assert cfg.omega == 1.0
        assert cfg.kappa == 0.8
        assert cfg.n_features == 2

    def test_custom(self) -> None:
        cfg = TwoPhaseConfig(omega=2.0, kappa=0.9, n_features=3)
        assert cfg.omega == 2.0
        assert cfg.n_features == 3


# ---------------------------------------------------------------------------
# PhaseAResult / PhaseBResult
# ---------------------------------------------------------------------------

class TestResults:
    def test_phase_a_total_bytes(self) -> None:
        r = PhaseAResult(assignments={}, posteriors={}, bytes_up=100.0, bytes_down=20.0)
        assert r.total_bytes == 120.0

    def test_phase_b_total_bytes(self) -> None:
        r = PhaseBResult(aggregated_params={}, bytes_up=500.0, bytes_down=300.0)
        assert r.total_bytes == 800.0


# ---------------------------------------------------------------------------
# TwoPhaseFedProTrack - Phase A
# ---------------------------------------------------------------------------

class TestPhaseA:
    def test_first_round_bootstrap(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0), 1: _make_fp(seed=1, mean_shift=5.0)}
        result = proto.phase_a(fps)

        assert isinstance(result, PhaseAResult)
        assert len(result.assignments) == 2
        assert 0 in result.assignments
        assert 1 in result.assignments
        assert result.bytes_up > 0
        assert result.bytes_down > 0

    def test_assigns_concept_ids(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        # Round 1: bootstrap
        fps = {0: _make_fp(seed=0), 1: _make_fp(seed=1, mean_shift=10.0)}
        r1 = proto.phase_a(fps)

        # Round 2: use prev assignments
        fps2 = {0: _make_fp(seed=2), 1: _make_fp(seed=3, mean_shift=10.0)}
        r2 = proto.phase_a(fps2, prev_assignments=r1.assignments)

        assert len(r2.assignments) == 2
        assert all(isinstance(v, int) for v in r2.assignments.values())

    def test_novelty_spawns_concept(self) -> None:
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            novelty_threshold=0.99,  # very high → almost everything is novel
            loss_novelty_threshold=0.01,  # very low → sensitive to any difference
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap with 2 distinct concepts so posterior has multiple options
        fps = {
            0: _make_fp(seed=0, mean_shift=0.0),
            1: _make_fp(seed=1, mean_shift=10.0),
        }
        proto.phase_a(fps)
        n_before = proto.memory_bank.n_concepts
        assert n_before >= 2

        # A very different fingerprint should trigger novelty (MAP prob < 0.99)
        fps2 = {2: _make_fp(seed=10, mean_shift=100.0)}
        proto.phase_a(fps2)
        assert proto.memory_bank.n_concepts >= n_before + 1

    def test_posteriors_present(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0)}
        result = proto.phase_a(fps)
        assert 0 in result.posteriors

    def test_bytes_accounting(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0), 1: _make_fp(seed=1)}
        result = proto.phase_a(fps)

        # Upload: 2 fingerprints * fingerprint_bytes(2, 2)
        from fedprotrack.baselines.comm_tracker import fingerprint_bytes
        expected_up = 2 * fingerprint_bytes(2, 2)
        assert result.bytes_up == expected_up
        # Download: 2 clients * 4 bytes
        assert result.bytes_down == 8.0

    def test_empty_clients(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)
        result = proto.phase_a({})
        assert result.assignments == {}
        assert result.bytes_up == 0.0


# ---------------------------------------------------------------------------
# TwoPhaseFedProTrack - Phase B
# ---------------------------------------------------------------------------

class TestPhaseB:
    def test_aggregation_within_clusters(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        # Setup: 2 clients in same concept
        params = {
            0: _make_model_params(seed=0),
            1: _make_model_params(seed=1),
        }
        assignments = {0: 0, 1: 0}

        result = proto.phase_b(params, assignments)
        assert isinstance(result, PhaseBResult)
        assert 0 in result.aggregated_params
        assert result.bytes_up > 0
        assert result.bytes_down > 0

    def test_separate_clusters(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        params = {
            0: _make_model_params(seed=0),
            1: _make_model_params(seed=1),
        }
        assignments = {0: 0, 1: 1}

        result = proto.phase_b(params, assignments)
        assert 0 in result.aggregated_params
        assert 1 in result.aggregated_params
        # Each cluster has 1 client, so aggregated = original
        np.testing.assert_allclose(
            result.aggregated_params[0]["coef"], params[0]["coef"]
        )

    def test_empty_phase_b(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        result = proto.phase_b({}, {})
        assert result.aggregated_params == {}
        assert result.bytes_up == 0.0
        assert result.bytes_down == 0.0

    def test_phase_b_bytes_scale_with_clients(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        # 3 clients, all same concept
        params = {i: _make_model_params(seed=i) for i in range(3)}
        assignments = {0: 0, 1: 0, 2: 0}

        r = proto.phase_b(params, assignments)
        # Upload: 3 models, Download: 3 * aggregated model
        assert r.bytes_up > 0
        assert r.bytes_down > 0


# ---------------------------------------------------------------------------
# End-to-end protocol
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_full_round(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        # Phase A
        fps = {
            0: _make_fp(seed=0, mean_shift=0.0),
            1: _make_fp(seed=1, mean_shift=0.0),
            2: _make_fp(seed=2, mean_shift=10.0),
        }
        a_result = proto.phase_a(fps)

        # Phase B
        params = {i: _make_model_params(seed=i) for i in range(3)}
        b_result = proto.phase_b(params, a_result.assignments)

        total = a_result.total_bytes + b_result.total_bytes
        assert total > 0

    def test_multi_round(self) -> None:
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        prev = None
        for rnd in range(3):
            fps = {i: _make_fp(seed=rnd * 10 + i) for i in range(4)}
            a_result = proto.phase_a(fps, prev_assignments=prev)
            prev = a_result.assignments

        assert proto.memory_bank.n_concepts >= 1
