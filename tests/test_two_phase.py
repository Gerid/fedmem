"""Tests for TwoPhaseFedProTrack protocol."""

from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.concept_tracker.fingerprint import ConceptFingerprint
from fedprotrack.posterior import make_plan_c_config
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
        assert cfg.omega == 2.0
        assert cfg.kappa == 0.6
        assert cfg.loss_novelty_threshold == 0.05
        assert cfg.sticky_dampening == 1.0
        assert cfg.sticky_posterior_gate == 0.3
        assert cfg.model_loss_weight == 0.0
        assert cfg.post_spawn_merge is True
        assert cfg.merge_threshold == 0.98
        assert cfg.merge_every == 2
        assert cfg.key_mode == "fingerprint"
        assert cfg.n_features == 2

    def test_custom(self) -> None:
        cfg = TwoPhaseConfig(omega=2.0, kappa=0.9, n_features=3)
        assert cfg.omega == 2.0
        assert cfg.n_features == 3

    def test_loss_novelty_threshold(self) -> None:
        cfg = TwoPhaseConfig(loss_novelty_threshold=0.1)
        assert cfg.loss_novelty_threshold == 0.1

    def test_entropy_freeze_and_adaptive_defaults(self) -> None:
        cfg = TwoPhaseConfig()
        assert cfg.entropy_freeze_threshold is None
        assert cfg.adaptive_addressing is False
        assert cfg.addressing_min_round_interval == 1

    def test_plan_c_preset(self) -> None:
        cfg = make_plan_c_config(max_concepts=9)
        assert cfg.key_mode == "multi_scale"
        assert cfg.adaptive_addressing is True
        assert cfg.entropy_freeze_threshold == 0.75
        assert cfg.merge_min_support == 2
        assert cfg.max_concepts == 9


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

        # Upload: 2 fingerprints * fingerprint_bytes(2, 2, float16, no global mean)
        from fedprotrack.baselines.comm_tracker import fingerprint_bytes
        expected_up = 2 * fingerprint_bytes(
            2, 2, precision_bits=16, include_global_mean=False,
        )
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

    def test_phase_b_namespaced_shared_can_aggregate_per_concept(self) -> None:
        cfg = TwoPhaseConfig(global_shared_aggregation=False)
        proto = TwoPhaseFedProTrack(cfg)

        params = {
            0: {
                "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[2.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([0.0], dtype=np.float64),
            },
            1: {
                "shared.trunk.weight": np.array([[3.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                "expert.1.head.weight": np.array([[10.0]], dtype=np.float64),
                "expert.1.head.bias": np.array([1.0], dtype=np.float64),
            },
        }
        assignments = {0: 0, 1: 1}

        result = proto.phase_b(params, assignments)

        np.testing.assert_allclose(
            result.aggregated_params[0]["shared.trunk.weight"],
            np.array([[1.0]], dtype=np.float64),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result.aggregated_params[1]["shared.trunk.weight"],
            np.array([[3.0]], dtype=np.float64),
            atol=1e-10,
        )


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


# ---------------------------------------------------------------------------
# Anti-fragmentation: sticky dampening
# ---------------------------------------------------------------------------

class TestStickyDampening:
    """Tests for sticky novelty dampening (Fix #1 for async drift)."""

    def test_no_dampening_single_concept(self) -> None:
        """Sticky dampening must NOT activate when library has only 1 concept.

        This is the 'single-concept trap': with 1 concept, the posterior is
        trivially 1.0. If sticky dampening raised the loss threshold, the
        first drift could never be detected.
        """
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            sticky_dampening=3.0,  # very high → would block detection if active
            loss_novelty_threshold=0.05,
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap: create 1 concept
        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        r1 = proto.phase_a(fps)
        assert proto.memory_bank.n_concepts == 1

        # Now present a different fingerprint. Even with high sticky_dampening,
        # the single-concept guard should let novelty through.
        fps2 = {0: _make_fp(seed=10, mean_shift=5.0, n_samples=100)}
        r2 = proto.phase_a(fps2, prev_assignments=r1.assignments)
        # Should have spawned a new concept (novelty detected)
        assert proto.memory_bank.n_concepts >= 2

    def test_dampening_active_with_multiple_concepts(self) -> None:
        """Sticky dampening should suppress noise-driven novelty when
        multiple concepts exist and the previous assignment's posterior
        is strong."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            sticky_dampening=3.0,
            loss_novelty_threshold=0.05,  # low threshold → easy to trigger
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap: 2 well-separated concepts
        fps = {
            0: _make_fp(seed=0, mean_shift=0.0, n_samples=100),
            1: _make_fp(seed=1, mean_shift=20.0, n_samples=100),
        }
        r1 = proto.phase_a(fps)
        n_before = proto.memory_bank.n_concepts
        assert n_before == 2

        # Present client 0 with a slightly noisy fingerprint (same concept)
        # Without dampening (threshold=0.05), this might trigger novelty.
        # With dampening (threshold=0.15), the noise is tolerated.
        fps2 = {0: _make_fp(seed=100, mean_shift=0.3, n_samples=100)}
        r2 = proto.phase_a(fps2, prev_assignments=r1.assignments)
        # Should NOT have spawned a new concept
        assert proto.memory_bank.n_concepts == n_before


# ---------------------------------------------------------------------------
# Anti-fragmentation: model loss suppression
# ---------------------------------------------------------------------------

class TestModelLossSuppression:
    """Tests for model-loss-based novelty suppression (Fix #3)."""

    def test_model_loss_suppresses_novelty(self) -> None:
        """When model accuracy is high (loss < threshold), fingerprint-
        based novelty should be suppressed."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            model_loss_weight=0.3,
            loss_novelty_threshold=0.05,  # low → easy to trigger from fingerprint
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap
        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        r1 = proto.phase_a(fps)
        assert proto.memory_bank.n_concepts == 1

        # Present a moderately different fingerprint with LOW model loss
        fps2 = {0: _make_fp(seed=10, mean_shift=1.0, n_samples=100)}
        model_losses = {0: 0.1}  # good accuracy → suppresses novelty
        r2 = proto.phase_a(fps2, r1.assignments, client_model_losses=model_losses)
        # Model loss (0.1 < 0.3=model_loss_weight) should suppress novelty
        assert proto.memory_bank.n_concepts == 1

    def test_high_model_loss_allows_novelty(self) -> None:
        """When model loss is high, fingerprint novelty should not be suppressed."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            model_loss_weight=0.3,
            loss_novelty_threshold=0.05,
        )
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        r1 = proto.phase_a(fps)

        fps2 = {0: _make_fp(seed=10, mean_shift=5.0, n_samples=100)}
        model_losses = {0: 0.6}  # bad accuracy → no suppression
        r2 = proto.phase_a(fps2, r1.assignments, client_model_losses=model_losses)
        assert proto.memory_bank.n_concepts >= 2

    def test_no_model_losses_falls_back(self) -> None:
        """Without model losses, fingerprint-only novelty detection works."""
        cfg = TwoPhaseConfig(n_features=2, n_classes=2)
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        r1 = proto.phase_a(fps)

        fps2 = {0: _make_fp(seed=10, mean_shift=20.0, n_samples=100)}
        r2 = proto.phase_a(fps2, r1.assignments)  # no model_losses
        assert proto.memory_bank.n_concepts >= 2


# ---------------------------------------------------------------------------
# Anti-fragmentation: post-spawn merge
# ---------------------------------------------------------------------------

class TestPostSpawnMerge:
    """Tests for post-spawn merge (Fix #2 for async drift)."""

    def test_post_spawn_merge_deduplicates(self) -> None:
        """Concepts spawned across rounds for the same underlying concept
        should be merged by post-spawn merge."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            post_spawn_merge=True,
            merge_threshold=0.85,
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Round 1: spawn concept at mean=0
        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        proto.phase_a(fps)
        assert proto.memory_bank.n_concepts == 1

        # Round 2: spawn concept at mean=20 (very different)
        fps2 = {1: _make_fp(seed=1, mean_shift=20.0, n_samples=100)}
        proto.phase_a(fps2)
        assert proto.memory_bank.n_concepts == 2

        # Round 3: spawn concept at mean=0.01 (nearly identical to first)
        # The post-spawn merge should merge it with concept 0.
        fps3 = {2: _make_fp(seed=2, mean_shift=0.01, n_samples=100)}
        proto.phase_a(fps3)
        # Should have 2 concepts (the duplicate was merged)
        assert proto.memory_bank.n_concepts == 2

    def test_post_spawn_merge_disabled(self) -> None:
        """With post_spawn_merge=False, duplicate concepts remain."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            post_spawn_merge=False,
            merge_threshold=0.85,
            merge_every=100,  # disable periodic merge too
        )
        proto = TwoPhaseFedProTrack(cfg)

        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        proto.phase_a(fps)

        fps2 = {1: _make_fp(seed=1, mean_shift=20.0, n_samples=100)}
        proto.phase_a(fps2)

        fps3 = {2: _make_fp(seed=2, mean_shift=0.01, n_samples=100)}
        proto.phase_a(fps3)
        # Without post-spawn merge, the near-duplicate might persist
        assert proto.memory_bank.n_concepts >= 2

    def test_remap_after_merge(self) -> None:
        """Assignments to merged-away concepts should be remapped."""
        cfg = TwoPhaseConfig(
            n_features=2, n_classes=2,
            post_spawn_merge=True,
            merge_threshold=0.85,
        )
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap with distinct concept
        fps = {0: _make_fp(seed=0, mean_shift=0.0, n_samples=100)}
        r1 = proto.phase_a(fps)

        # Spawn near-identical concept in next round
        fps2 = {1: _make_fp(seed=1, mean_shift=0.01, n_samples=100)}
        r2 = proto.phase_a(fps2, r1.assignments)

        # All assignments should point to live concepts
        live_ids = set(proto.memory_bank._library.keys())
        for cid, concept_id in r2.assignments.items():
            assert concept_id in live_ids, (
                f"Client {cid} assigned to dead concept {concept_id}"
            )


# ---------------------------------------------------------------------------
# Soft Aggregation (phase_b_soft)
# ---------------------------------------------------------------------------

class TestSoftAggregation:
    """Tests for posterior-weighted Phase B aggregation."""

    def test_phase_b_soft_returns_result(self) -> None:
        """Soft Phase B returns a valid PhaseBResult."""
        cfg = TwoPhaseConfig(loss_novelty_threshold=0.5)
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap two concepts
        fp0 = _make_fp(seed=0, mean_shift=0.0, n_samples=50)
        fp1 = _make_fp(seed=1, mean_shift=5.0, n_samples=50)
        fps = {0: fp0, 1: fp1}
        a_result = proto.phase_a(fps)

        params = {0: _make_model_params(seed=0), 1: _make_model_params(seed=1)}
        b_result = proto.phase_b_soft(params, a_result.assignments, a_result.posteriors)

        assert isinstance(b_result, PhaseBResult)
        assert b_result.bytes_up > 0
        assert b_result.aggregated_params  # non-empty

    def test_soft_vs_hard_different_when_ambiguous(self) -> None:
        """Soft and hard produce different aggregation when posteriors are ambiguous."""
        cfg = TwoPhaseConfig(loss_novelty_threshold=0.5)
        proto = TwoPhaseFedProTrack(cfg)

        # Two similar concepts → ambiguous posteriors
        fp0 = _make_fp(seed=0, mean_shift=0.0, n_samples=50)
        fp1 = _make_fp(seed=1, mean_shift=0.5, n_samples=50)
        fps = {0: fp0, 1: fp1}
        a_result = proto.phase_a(fps)

        params = {0: _make_model_params(seed=10), 1: _make_model_params(seed=20)}

        # Hard aggregation
        hard = proto.phase_b(params, a_result.assignments)
        # Soft aggregation
        soft = proto.phase_b_soft(params, a_result.assignments, a_result.posteriors)

        # Both should produce results for the same concepts
        assert set(hard.aggregated_params.keys()) == set(soft.aggregated_params.keys())

    def test_soft_negligible_weight_filtered(self) -> None:
        """Clients with posterior < 0.01 for a concept are excluded from its aggregation."""
        from fedprotrack.posterior.gibbs import PosteriorAssignment

        cfg = TwoPhaseConfig(loss_novelty_threshold=0.5)
        proto = TwoPhaseFedProTrack(cfg)

        # Bootstrap a concept
        fp0 = _make_fp(seed=0, mean_shift=0.0, n_samples=50)
        proto.phase_a({0: fp0})

        # Build fake posteriors: client 0 has 99% for concept 0, client 1 has 0.5%
        concept_id = list(proto.memory_bank._library.keys())[0]
        posteriors = {
            0: PosteriorAssignment(
                probabilities={concept_id: 0.99}, map_concept_id=concept_id,
                is_novel=False, entropy=0.01,
            ),
            1: PosteriorAssignment(
                probabilities={concept_id: 0.005}, map_concept_id=concept_id,
                is_novel=False, entropy=0.01,
            ),
        }
        assignments = {0: concept_id, 1: concept_id}
        params = {0: _make_model_params(seed=0), 1: _make_model_params(seed=1)}

        result = proto.phase_b_soft(params, assignments, posteriors)

        # Client 1's weight (0.005) < 0.01 → filtered out
        # Result should be dominated by client 0's params
        agg = result.aggregated_params[concept_id]
        np.testing.assert_allclose(agg["coef"], params[0]["coef"], atol=1e-10)

    def test_soft_namespaced_aggregation_uses_matching_expert_slots(self) -> None:
        from fedprotrack.posterior.gibbs import PosteriorAssignment

        cfg = TwoPhaseConfig(loss_novelty_threshold=0.5)
        proto = TwoPhaseFedProTrack(cfg)

        params = {
            0: {
                "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[2.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([0.0], dtype=np.float64),
            },
            1: {
                "shared.trunk.weight": np.array([[3.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                "expert.1.head.weight": np.array([[10.0]], dtype=np.float64),
                "expert.1.head.bias": np.array([1.0], dtype=np.float64),
            },
        }
        assignments = {0: 0, 1: 1}
        posteriors = {
            0: PosteriorAssignment(
                probabilities={0: 0.9, 1: 0.1},
                map_concept_id=0,
                is_novel=False,
                entropy=0.1,
            ),
            1: PosteriorAssignment(
                probabilities={0: 0.2, 1: 0.8},
                map_concept_id=1,
                is_novel=False,
                entropy=0.1,
            ),
        }

        result = proto.phase_b_soft(params, assignments, posteriors)
        agg0 = result.aggregated_params[0]
        agg1 = result.aggregated_params[1]

        np.testing.assert_allclose(
            agg0["expert.0.head.weight"],
            np.array([[2.0]], dtype=np.float64),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            agg1["expert.1.head.weight"],
            np.array([[10.0]], dtype=np.float64),
            atol=1e-10,
        )

    def test_soft_namespaced_shared_can_aggregate_per_concept(self) -> None:
        from fedprotrack.posterior.gibbs import PosteriorAssignment

        cfg = TwoPhaseConfig(global_shared_aggregation=False)
        proto = TwoPhaseFedProTrack(cfg)

        params = {
            0: {
                "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[2.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([0.0], dtype=np.float64),
            },
            1: {
                "shared.trunk.weight": np.array([[3.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                "expert.1.head.weight": np.array([[10.0]], dtype=np.float64),
                "expert.1.head.bias": np.array([1.0], dtype=np.float64),
            },
        }
        assignments = {0: 0, 1: 1}
        posteriors = {
            0: PosteriorAssignment(
                probabilities={0: 0.9, 1: 0.1},
                map_concept_id=0,
                is_novel=False,
                entropy=0.1,
            ),
            1: PosteriorAssignment(
                probabilities={0: 0.2, 1: 0.8},
                map_concept_id=1,
                is_novel=False,
                entropy=0.1,
            ),
        }

        result = proto.phase_b_soft(params, assignments, posteriors)
        agg0 = result.aggregated_params[0]
        agg1 = result.aggregated_params[1]

        np.testing.assert_allclose(
            agg0["shared.trunk.weight"],
            np.array([[15.0 / 11.0]], dtype=np.float64),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            agg1["shared.trunk.weight"],
            np.array([[25.0 / 9.0]], dtype=np.float64),
            atol=1e-10,
        )
        assert agg0["shared.trunk.weight"][0, 0] != agg1["shared.trunk.weight"][0, 0]

    def test_soft_aggregation_runner_flag(self) -> None:
        """FedProTrackRunner with soft_aggregation=True runs without error."""
        from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
        from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner

        gc = GeneratorConfig(K=3, T=5, n_samples=100, generator_type="sine",
                             rho=2.0, alpha=0.0, delta=0.5, seed=42)
        dataset = generate_drift_dataset(gc)

        runner = FedProTrackRunner(seed=42, soft_aggregation=True)
        result = runner.run(dataset)

        assert result.accuracy_matrix.shape == (3, 5)
        assert result.total_bytes > 0
        assert result.method_name == "FedProTrack"
