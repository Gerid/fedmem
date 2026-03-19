from __future__ import annotations

import run_adapter_research_loop as loop


def _summary(
    *,
    mean_final: float,
    mean_phase3: float,
    mean_recovery_next_fed: float,
    mean_bytes: float,
    mean_spawned: float = 0.0,
    mean_active: float = 1.0,
    mean_assignment_switch_rate: float = 0.0,
    mean_multi_route_rate: float = 0.0,
    mean_shared_drift_norm: float = 0.12,
) -> dict[str, object]:
    return {
        "mean_final": mean_final,
        "mean_phase3": mean_phase3,
        "mean_recovery_next_fed": mean_recovery_next_fed,
        "mean_bytes": mean_bytes,
        "mean_spawned": mean_spawned,
        "mean_active": mean_active,
        "mean_assignment_switch_rate": mean_assignment_switch_rate,
        "mean_multi_route_rate": mean_multi_route_rate,
        "mean_shared_drift_norm": mean_shared_drift_norm,
    }


class TestFailureProfile:
    def test_detects_routing_collapse(self) -> None:
        profile = loop._failure_profile(
            _summary(
                mean_final=0.64,
                mean_phase3=0.52,
                mean_recovery_next_fed=0.49,
                mean_bytes=15_000_000.0,
            )
        )
        assert profile["routing_collapsed"] is True
        assert profile["multi_route_dead"] is True


class TestCandidateCatalog:
    def test_prefers_pre_adapter_and_hybrid_when_collapsed(self) -> None:
        catalog = loop._candidate_catalog(
            _summary(
                mean_final=0.64,
                mean_phase3=0.52,
                mean_recovery_next_fed=0.49,
                mean_bytes=15_000_000.0,
            )
        )
        sources = [str(variant["fingerprint_source"]) for variant in catalog]
        assert sources[0] == "pre_adapter_embed"
        assert "hybrid_raw_pre_adapter" in sources
        assert "model_embed" not in sources


class TestAcceptanceGate:
    def test_accepts_candidate_that_weakly_dominates_champion(self) -> None:
        keep, reason = loop._should_accept_candidate(
            _summary(
                mean_final=0.66,
                mean_phase3=0.56,
                mean_recovery_next_fed=0.47,
                mean_bytes=16_000_000.0,
            ),
            _summary(
                mean_final=0.67,
                mean_phase3=0.57,
                mean_recovery_next_fed=0.48,
                mean_bytes=15_000_000.0,
            ),
            _summary(
                mean_final=0.74,
                mean_phase3=0.62,
                mean_recovery_next_fed=0.69,
                mean_bytes=8_000_000.0,
            ),
        )
        assert keep is True
        assert "dominates" in reason

    def test_rejects_candidate_that_breaks_linear_bytes_cap(self) -> None:
        keep, reason = loop._should_accept_candidate(
            _summary(
                mean_final=0.66,
                mean_phase3=0.56,
                mean_recovery_next_fed=0.47,
                mean_bytes=16_000_000.0,
            ),
            _summary(
                mean_final=0.69,
                mean_phase3=0.58,
                mean_recovery_next_fed=0.50,
                mean_bytes=11_000_000.0,
            ),
            _summary(
                mean_final=0.74,
                mean_phase3=0.62,
                mean_recovery_next_fed=0.69,
                mean_bytes=4_000_000.0,
            ),
        )
        assert keep is False
        assert "bytes cap" in reason
