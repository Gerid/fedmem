"""End-to-end FedProTrack runner using the two-phase protocol.

Orchestrates a complete federated concept drift simulation:
  1. Clients build fingerprints and train local models
  2. Phase A: fingerprint exchange for concept identification
  3. Phase B: model aggregation within concept clusters
  4. Metrics collection and communication byte accounting
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..baselines.comm_tracker import model_bytes
from ..concept_tracker.fingerprint import ConceptFingerprint
from ..drift_detector import BaseDriftDetector
from ..drift_generator import DriftDataset
from ..federation import merge_param_namespaces, split_param_namespaces
from ..metrics.concept_metrics import (
    assignment_switch_rate as compute_assignment_switch_rate,
    avg_clients_per_concept as compute_avg_clients_per_concept,
    memory_reuse_rate as compute_memory_reuse_rate,
    routing_consistency as compute_routing_consistency,
    singleton_group_ratio as compute_singleton_group_ratio,
)
from ..metrics.experiment_log import ExperimentLog
from ..models import (
    TorchFactorizedAdapterClassifier,
    TorchFeatureAdapterClassifier,
    TorchLinearClassifier,
)
from .gibbs import PosteriorAssignment
from .two_phase_protocol import PhaseAResult, TwoPhaseConfig, TwoPhaseFedProTrack


def _make_detector(name: str) -> BaseDriftDetector:
    """Create a drift detector by name."""
    from ..drift_detector import (
        ADWINDetector,
        KSWINDetector,
        NoDriftDetector,
        PageHinkleyDetector,
    )

    detectors: dict[str, type] = {
        "ADWIN": ADWINDetector,
        "PageHinkley": PageHinkleyDetector,
        "KSWIN": KSWINDetector,
        "NoDrift": NoDriftDetector,
    }
    if name not in detectors:
        raise ValueError(f"Unknown detector: {name}. Choose from {list(detectors)}")
    return detectors[name]()


def _infer_n_features(generator_type: str, dataset: DriftDataset | None = None) -> int:
    """Infer feature dimensionality from generator type or dataset.

    Parameters
    ----------
    generator_type : str
    dataset : DriftDataset, optional
        If provided, infer from first data sample.

    Returns
    -------
    int
    """
    if dataset is not None and dataset.data:
        first_key = next(iter(dataset.data))
        X, _ = dataset.data[first_key]
        return X.shape[1]
    return {"sine": 2, "sea": 3, "circle": 2}.get(generator_type, 2)


def _infer_n_classes(dataset: DriftDataset) -> int:
    """Infer number of class labels from a dataset."""
    all_labels: set[int] = set()
    for _, y in dataset.data.values():
        all_labels.update(int(v) for v in np.unique(y))
    if not all_labels:
        return 2
    return max(all_labels) + 1


def _infer_fingerprint_n_classes(dataset: DriftDataset) -> int:
    """Infer label cardinality for fingerprinting, allowing an alternate label view."""
    fp_labels = getattr(dataset, "fingerprint_labels", None)
    if not isinstance(fp_labels, dict) or not fp_labels:
        return _infer_n_classes(dataset)

    all_labels: set[int] = set()
    for y in fp_labels.values():
        all_labels.update(int(v) for v in np.unique(y))
    if not all_labels:
        return _infer_n_classes(dataset)
    return max(all_labels) + 1


def _make_model(
    model_type: str,
    *,
    n_features: int,
    n_classes: int,
    lr: float,
    n_epochs: int,
    seed: int,
    hidden_dim: int,
    adapter_dim: int,
):
    """Instantiate the configured client model."""
    if model_type == "linear":
        return TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
    if model_type == "feature_adapter":
        return TorchFeatureAdapterClassifier(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
    if model_type == "factorized_adapter":
        return TorchFactorizedAdapterClassifier(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed,
        )
    raise ValueError(
        "Unknown model_type: "
        f"{model_type}. Choose from ['linear', 'feature_adapter', 'factorized_adapter']"
    )


def _is_feature_adapter_model(model) -> bool:
    return isinstance(model, (TorchFeatureAdapterClassifier, TorchFactorizedAdapterClassifier))


def _is_factorized_adapter_model(model) -> bool:
    return isinstance(model, TorchFactorizedAdapterClassifier)


def _is_namespaced_routed_model(model) -> bool:
    return _is_feature_adapter_model(model)


def _model_predict(
    model,
    X: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
) -> np.ndarray:
    if _is_namespaced_routed_model(model):
        return model.predict(X, slot_id=slot_id, slot_weights=slot_weights)
    return model.predict(X)


def _model_predict_loss(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
) -> float:
    if _is_namespaced_routed_model(model):
        return model.predict_loss(X, y, slot_id=slot_id, slot_weights=slot_weights)
    return model.predict_loss(X, y)


def _model_embed(
    model,
    X: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
    representation: str = "post_adapter",
) -> np.ndarray:
    if _is_namespaced_routed_model(model):
        return model.embed(
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation=representation,
        )
    raise ValueError("model_embed requires a routed model with embed() support")


def _model_fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
    update_shared: bool = True,
    update_private: bool = True,
    update_head: bool = True,
) -> None:
    if _is_factorized_adapter_model(model):
        model.fit(
            X,
            y,
            slot_id=slot_id,
            slot_weights=slot_weights,
            update_shared=update_shared,
            update_private=update_private,
            update_head=update_head,
        )
    elif _is_namespaced_routed_model(model):
        model.fit(
            X,
            y,
            slot_id=slot_id,
            slot_weights=slot_weights,
            update_shared=update_shared,
        )
    else:
        model.fit(X, y)


def _model_partial_fit(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
    update_shared: bool = True,
    update_private: bool = True,
    update_head: bool = True,
) -> None:
    if _is_factorized_adapter_model(model):
        model.partial_fit(
            X,
            y,
            slot_id=slot_id,
            slot_weights=slot_weights,
            update_shared=update_shared,
            update_private=update_private,
            update_head=update_head,
        )
    elif _is_namespaced_routed_model(model):
        model.partial_fit(
            X,
            y,
            slot_id=slot_id,
            slot_weights=slot_weights,
            update_shared=update_shared,
        )
    else:
        model.partial_fit(X, y)


def _model_get_params(model, *, slot_id: int = 0) -> dict[str, np.ndarray]:
    if _is_namespaced_routed_model(model):
        return model.get_params(slot_id=slot_id)
    return model.get_params()


def _model_set_params(model, params: dict[str, np.ndarray]) -> None:
    model.set_params(params)


def _model_blend_params(
    model,
    params: dict[str, np.ndarray],
    *,
    alpha: float,
) -> None:
    model.blend_params(params, alpha=alpha)


def _normalise_slot_weights(
    slot_id: int,
    slot_weights: dict[int, float] | None = None,
) -> dict[int, float]:
    if not slot_weights:
        return {int(slot_id): 1.0}
    filtered = {
        int(current_slot): float(weight)
        for current_slot, weight in slot_weights.items()
        if float(weight) > 0.01
    }
    if not filtered:
        return {int(slot_id): 1.0}
    total = float(sum(filtered.values()))
    return {
        current_slot: weight / total
        for current_slot, weight in filtered.items()
    }


def _select_weighted_training_slots(
    slot_id: int,
    slot_weights: dict[int, float] | None = None,
    *,
    top_k: int | None = None,
    min_secondary_weight: float = 0.0,
) -> dict[int, float]:
    normalized = _normalise_slot_weights(slot_id, slot_weights)
    if top_k is None and min_secondary_weight <= 0.0:
        return normalized

    ranked = sorted(
        normalized.items(),
        key=lambda item: (-float(item[1]), int(item[0])),
    )
    k = len(ranked) if top_k is None else max(1, int(top_k))
    selected = ranked[:k]
    if len(selected) >= 2 and float(selected[1][1]) >= float(min_secondary_weight):
        total = float(sum(weight for _, weight in selected))
        return {
            int(current_slot): float(weight) / total
            for current_slot, weight in selected
        }
    map_slot = int(selected[0][0]) if selected else int(slot_id)
    return {map_slot: 1.0}


def _collect_slot_anchor_payloads(
    model,
    *,
    slot_ids: list[int],
    memory_bank,
) -> dict[int, dict[str, np.ndarray]]:
    anchor_payloads: dict[int, dict[str, np.ndarray]] = {}
    for slot_id in slot_ids:
        stored = memory_bank.get_model_params(int(slot_id))
        if stored is not None:
            anchor_payloads[int(slot_id)] = {
                key: value.copy() for key, value in stored.items()
            }
            continue
        anchor_payloads[int(slot_id)] = _model_get_params(model, slot_id=int(slot_id))
    return anchor_payloads


def _warm_start_factorized_slots(
    model,
    *,
    shared_slot_id: int,
    slot_ids: list[int],
    anchor_payloads: dict[int, dict[str, np.ndarray]],
) -> None:
    shared_payload, _, other_payload = split_param_namespaces(
        _model_get_params(model, slot_id=int(shared_slot_id))
    )
    expert_payload: dict[str, np.ndarray] = {}
    for slot_id in slot_ids:
        slot_payload = anchor_payloads.get(int(slot_id))
        if slot_payload is None:
            slot_payload = _model_get_params(model, slot_id=int(slot_id))
        _, slot_expert, _ = split_param_namespaces(slot_payload)
        expert_payload.update({
            key: value.copy() for key, value in slot_expert.items()
        })
    merged = merge_param_namespaces(
        shared=shared_payload if shared_payload else None,
        expert=expert_payload if expert_payload else None,
        other=other_payload if other_payload else None,
    )
    if merged:
        _model_set_params(model, merged)


def _apply_factorized_slot_anchor(
    model,
    *,
    slot_ids: list[int],
    primary_slot_id: int,
    anchor_payloads: dict[int, dict[str, np.ndarray]],
    primary_anchor_alpha: float,
    secondary_anchor_alpha: float,
) -> None:
    shared_payload, _, other_payload = split_param_namespaces(
        _model_get_params(model, slot_id=int(primary_slot_id))
    )
    expert_payload: dict[str, np.ndarray] = {}
    for slot_id in slot_ids:
        current_payload = _model_get_params(model, slot_id=int(slot_id))
        _, current_expert, _ = split_param_namespaces(current_payload)
        _, anchor_expert, _ = split_param_namespaces(
            anchor_payloads.get(int(slot_id), current_payload)
        )
        alpha = (
            float(primary_anchor_alpha)
            if int(slot_id) == int(primary_slot_id)
            else float(secondary_anchor_alpha)
        )
        for key, current_value in current_expert.items():
            anchor_value = anchor_expert.get(key)
            if anchor_value is None or alpha <= 0.0:
                expert_payload[key] = current_value.copy()
                continue
            expert_payload[key] = (
                alpha * anchor_value + (1.0 - alpha) * current_value
            )
    merged = merge_param_namespaces(
        shared=shared_payload if shared_payload else None,
        expert=expert_payload if expert_payload else None,
        other=other_payload if other_payload else None,
    )
    if merged:
        _model_set_params(model, merged)


def _run_factorized_primary_consolidation(
    model,
    X: np.ndarray,
    y: np.ndarray,
    *,
    slot_id: int,
    steps: int,
    mode: str = "full",
) -> None:
    update_private = mode != "head_only"
    for _ in range(int(steps)):
        _model_partial_fit(
            model,
            X,
            y,
            slot_id=slot_id,
            slot_weights=None,
            update_shared=False,
            update_private=update_private,
            update_head=True,
        )


def _shared_drift_norm(
    before: dict[str, np.ndarray],
    after: dict[str, np.ndarray],
) -> float:
    before_shared, _, _ = split_param_namespaces(before)
    after_shared, _, _ = split_param_namespaces(after)
    if not before_shared or not after_shared:
        return 0.0
    squared_norm = 0.0
    for key, before_value in before_shared.items():
        after_value = after_shared.get(key)
        if after_value is None:
            continue
        diff = after_value - before_value
        squared_norm += float(np.sum(diff * diff))
    return float(np.sqrt(squared_norm))


def _build_fingerprint_features(
    model,
    X: np.ndarray,
    *,
    slot_id: int = 0,
    slot_weights: dict[int, float] | None = None,
    fingerprint_source: str,
    bootstrap_hidden_dim: int | None = None,
    use_bootstrap_raw: bool = False,
) -> np.ndarray:
    raw = np.asarray(X, dtype=np.float32)

    def _center_hidden(hidden: np.ndarray) -> np.ndarray:
        hidden = np.asarray(hidden, dtype=np.float32)
        if hidden.size == 0:
            return hidden
        return (hidden - hidden.mean(axis=0, keepdims=True)).astype(np.float32, copy=False)

    def _scale_hidden(hidden: np.ndarray, scale: float) -> np.ndarray:
        return (np.asarray(hidden, dtype=np.float32) * np.float32(scale)).astype(
            np.float32,
            copy=False,
        )

    if fingerprint_source == "raw_input":
        return raw
    if fingerprint_source == "model_embed":
        return _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="post_adapter",
        )
    if fingerprint_source == "pre_adapter_embed":
        return _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
    if fingerprint_source == "hybrid_raw_pre_adapter":
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        return np.concatenate([raw, pre_adapter], axis=1).astype(np.float32, copy=False)
    if fingerprint_source == "weighted_hybrid_raw_pre_adapter":
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        return np.concatenate([raw, pre_adapter], axis=1).astype(np.float32, copy=False)
    if fingerprint_source == "centered_hybrid_raw_pre_adapter":
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        centered = _center_hidden(pre_adapter)
        return np.concatenate([raw, centered], axis=1).astype(np.float32, copy=False)
    if fingerprint_source == "attenuated_hybrid_raw_pre_adapter":
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        attenuated = _scale_hidden(pre_adapter, 0.25)
        return np.concatenate([raw, attenuated], axis=1).astype(np.float32, copy=False)
    if fingerprint_source == "double_raw_hybrid_pre_adapter":
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        return np.concatenate([raw, raw, pre_adapter], axis=1).astype(
            np.float32,
            copy=False,
        )
    if fingerprint_source == "bootstrap_raw_hybrid_pre_adapter":
        if bootstrap_hidden_dim is None:
            raise ValueError(
                "bootstrap_hidden_dim is required for bootstrap_raw_hybrid_pre_adapter"
            )
        if use_bootstrap_raw:
            zeros = np.zeros((len(raw), bootstrap_hidden_dim), dtype=np.float32)
            return np.concatenate([raw, zeros], axis=1).astype(np.float32, copy=False)
        pre_adapter = _model_embed(
            model,
            X,
            slot_id=slot_id,
            slot_weights=slot_weights,
            representation="pre_adapter",
        )
        return np.concatenate([raw, pre_adapter], axis=1).astype(np.float32, copy=False)
    raise ValueError(
        "fingerprint_source must be one of "
        "['raw_input', 'model_embed', 'pre_adapter_embed', "
        "'hybrid_raw_pre_adapter', 'weighted_hybrid_raw_pre_adapter', "
        "'centered_hybrid_raw_pre_adapter', "
        "'attenuated_hybrid_raw_pre_adapter', "
        "'double_raw_hybrid_pre_adapter', "
        "'bootstrap_raw_hybrid_pre_adapter']"
    )


def _fingerprint_feature_groups(
    fingerprint_source: str,
    *,
    n_features: int,
    hidden_dim: int,
) -> tuple[tuple[int, int, float], ...] | None:
    if fingerprint_source == "weighted_hybrid_raw_pre_adapter":
        return (
            (0, n_features, 0.75),
            (n_features, n_features + hidden_dim, 0.25),
        )
    return None


def _summarize_phase_a_round(
    *,
    timestep: int,
    phase_a_result: PhaseAResult,
    active_after: int,
) -> dict[str, object]:
    fp_losses = list(phase_a_result.client_fp_losses.values())
    thresholds = list(phase_a_result.client_effective_thresholds.values())
    map_probs = list(phase_a_result.client_map_probabilities.values())
    fp_gaps = [
        loss - phase_a_result.client_effective_thresholds[client_id]
        for client_id, loss in phase_a_result.client_fp_losses.items()
        if client_id in phase_a_result.client_effective_thresholds
    ]
    over_threshold = [
        float(
            loss > phase_a_result.client_effective_thresholds[client_id]
        )
        for client_id, loss in phase_a_result.client_fp_losses.items()
        if client_id in phase_a_result.client_effective_thresholds
    ]

    return {
        "t": int(timestep),
        "library_size_before": int(phase_a_result.library_size_before),
        "active_after": int(active_after),
        "spawned": int(phase_a_result.spawned),
        "merged": int(phase_a_result.merged),
        "n_clients_logged": int(len(fp_losses)),
        "mean_fp_loss": float(np.mean(fp_losses)) if fp_losses else None,
        "min_fp_loss": float(np.min(fp_losses)) if fp_losses else None,
        "max_fp_loss": float(np.max(fp_losses)) if fp_losses else None,
        "mean_effective_threshold": (
            float(np.mean(thresholds)) if thresholds else None
        ),
        "mean_fp_gap": float(np.mean(fp_gaps)) if fp_gaps else None,
        "over_threshold_rate": (
            float(np.mean(over_threshold)) if over_threshold else None
        ),
        "mean_map_prob": float(np.mean(map_probs)) if map_probs else None,
    }


def _compose_routed_payload(
    candidate_payloads: dict[int, dict[str, np.ndarray]],
    slot_weights: dict[int, float],
) -> tuple[dict[str, np.ndarray], dict[int, float]]:
    """Compose a client read payload from multiple slot states."""
    filtered = {
        int(slot_id): float(weight)
        for slot_id, weight in slot_weights.items()
        if float(weight) > 0.0 and int(slot_id) in candidate_payloads
    }
    if not filtered:
        return {}, {}

    total = float(sum(filtered.values()))
    normalized = {
        slot_id: weight / total
        for slot_id, weight in filtered.items()
    }

    shared_sums: dict[str, np.ndarray] = {}
    shared_weight_sums: dict[str, float] = {}
    expert_params: dict[str, np.ndarray] = {}
    other_params: dict[str, np.ndarray] = {}
    for slot_id in sorted(normalized):
        shared, expert, other = split_param_namespaces(candidate_payloads[slot_id])
        slot_weight = normalized[slot_id]
        for key, value in shared.items():
            weighted_value = slot_weight * value
            if key in shared_sums:
                shared_sums[key] = shared_sums[key] + weighted_value
                shared_weight_sums[key] += slot_weight
            else:
                shared_sums[key] = weighted_value.copy()
                shared_weight_sums[key] = slot_weight
        expert_params.update({key: value.copy() for key, value in expert.items()})
        for key, value in other.items():
            other_params.setdefault(key, value.copy())

    shared_params = {
        key: value / shared_weight_sums[key]
        for key, value in shared_sums.items()
        if shared_weight_sums[key] > 0.0
    }

    return (
        merge_param_namespaces(
            shared=shared_params,
            expert=expert_params,
            other=other_params,
        ),
        normalized,
    )


def _prepare_routed_read_weights(
    slot_weights: dict[int, float] | None,
    *,
    top_k: int | None = None,
    temperature: float = 1.0,
    only_on_ambiguity: bool = False,
    min_entropy: float | None = None,
    min_secondary_weight: float = 0.0,
    max_primary_gap: float | None = None,
) -> dict[int, float] | None:
    if not slot_weights:
        return None
    filtered = {
        int(slot_id): float(weight)
        for slot_id, weight in slot_weights.items()
        if float(weight) > 0.0
    }
    if not filtered:
        return None
    ranked = sorted(
        filtered.items(),
        key=lambda item: (-item[1], item[0]),
    )
    if not ranked:
        return None
    if float(temperature) <= 0.0:
        raise ValueError("routed_read_temperature must be > 0")
    base_values = np.asarray([weight for _, weight in ranked], dtype=np.float64)
    base_total = float(base_values.sum())
    if base_total <= 0.0:
        return None
    normalized_base = {
        int(slot_id): float(value / base_total)
        for (slot_id, _), value in zip(ranked, base_values, strict=False)
    }
    if only_on_ambiguity:
        ambiguous = False
        if len(ranked) >= 2:
            if min_entropy is not None:
                probs = np.asarray(list(normalized_base.values()), dtype=np.float64)
                entropy = float(
                    -(probs * np.log(np.clip(probs, 1e-12, None))).sum()
                    / np.log(float(len(probs)))
                )
                ambiguous = ambiguous or entropy >= float(min_entropy)
            if float(min_secondary_weight) > 0.0 or max_primary_gap is not None:
                top1 = float(ranked[0][1])
                top2 = float(ranked[1][1])
                gap_ok = max_primary_gap is None or (top1 - top2) <= float(max_primary_gap)
                ambiguous = ambiguous or (
                    top2 >= float(min_secondary_weight) and gap_ok
                )
            if min_entropy is None and float(min_secondary_weight) <= 0.0 and max_primary_gap is None:
                ambiguous = True
        if not ambiguous:
            return normalized_base
    if top_k is not None:
        ranked = ranked[: int(top_k)]
    if not ranked:
        return None
    values = np.asarray([weight for _, weight in ranked], dtype=np.float64)
    if float(temperature) != 1.0:
        values = np.power(np.clip(values, 1e-12, None), 1.0 / float(temperature))
    total = float(values.sum())
    if total <= 0.0:
        return None
    return {
        int(slot_id): float(value / total)
        for (slot_id, _), value in zip(ranked, values, strict=False)
    }


@dataclass
class FedProTrackResult:
    """Complete result from a FedProTrack simulation.

    Parameters
    ----------
    accuracy_matrix : np.ndarray
        Shape (K, T). Per-client per-step classification accuracy.
    predicted_concept_matrix : np.ndarray
        Shape (K, T). Predicted concept IDs.
    true_concept_matrix : np.ndarray
        Shape (K, T). Ground-truth concept IDs.
    total_bytes : float
        Total communication cost (Phase A + Phase B).
    phase_a_bytes : float
        Communication cost for Phase A only.
    phase_b_bytes : float
        Communication cost for Phase B only.
    mean_accuracy : float
        Mean accuracy across all clients and steps.
    final_accuracy : float
        Mean accuracy at the last time step.
    method_name : str
        Human-readable name.
    """

    accuracy_matrix: np.ndarray
    predicted_concept_matrix: np.ndarray
    true_concept_matrix: np.ndarray
    total_bytes: float
    phase_a_bytes: float
    phase_b_bytes: float
    mean_accuracy: float
    final_accuracy: float
    spawned_concepts: int = 0
    merged_concepts: int = 0
    pruned_concepts: int = 0
    active_concepts: int = 0
    soft_assignments: np.ndarray | None = None
    assignment_switch_rate: float | None = None
    avg_clients_per_concept: float | None = None
    singleton_group_ratio: float | None = None
    memory_reuse_rate: float | None = None
    routing_consistency: float | None = None
    shared_drift_norm: float | None = None
    expert_update_coverage: float | None = None
    multi_route_rate: float | None = None
    phase_a_round_diagnostics: list[dict[str, object]] = field(default_factory=list)
    method_name: str = "FedProTrack"

    def to_experiment_log(self) -> ExperimentLog:
        """Convert to ExperimentLog for unified metrics computation.

        Returns
        -------
        ExperimentLog
        """
        return ExperimentLog(
            ground_truth=self.true_concept_matrix,
            predicted=self.predicted_concept_matrix,
            accuracy_curve=self.accuracy_matrix,
            total_bytes=self.total_bytes if self.total_bytes > 0 else None,
            soft_assignments=self.soft_assignments,
            method_name=self.method_name,
        )


class FedProTrackRunner:
    """End-to-end FedProTrack simulation runner.

    Parameters
    ----------
    config : TwoPhaseConfig
        Two-phase protocol configuration.
    federation_every : int
        Run federation (Phase A + Phase B) every this many steps.
    detector_name : str
        Drift detector name ("ADWIN", "PageHinkley", "KSWIN", "NoDrift").
    seed : int
        Base random seed.
    event_triggered : bool
        If True, Phase A only runs when at least one client detects drift
        (or at the first step). This reduces communication by skipping
        fingerprint exchange during stable periods. Phase B still runs
        on the ``federation_every`` schedule when assignments exist.
    soft_aggregation : bool
        If True, use posterior-weighted (soft) Phase B aggregation instead
        of hard MAP assignment. Each client's model contributes to multiple
        concept clusters proportional to its posterior probability. This
        reduces noise when posteriors are ambiguous. Default False.
    blend_alpha : float
        Weight for momentum blend that pulls local model params toward the
        previous step's params after training. 0.0 disables it. Default 0.5.
    auto_scale : bool
        If True, automatically scale ``loss_novelty_threshold``,
        ``merge_threshold``, ``max_concepts``, and ``blend_alpha`` based
        on feature dimensionality. Higher-dimensional data (e.g. MNIST
        with 20 PCA features) gets higher novelty thresholds and lower
        merge thresholds to prevent over-spawning. Default False.
    routed_local_training : bool
        If True, feature-adapter clients reuse routed slot weights during
        local training instead of updating only the MAP slot. Experimental
        and default False.
    skip_last_federation_round : bool
        If True, do not run or charge a federation round after the final
        evaluation step. This keeps communication accounting aligned with
        matched-budget baselines because there is no future prediction to
        benefit from the final sync. Default True.
    """

    def __init__(
        self,
        config: TwoPhaseConfig | None = None,
        federation_every: int = 1,
        detector_name: str = "ADWIN",
        seed: int = 42,
        event_triggered: bool = False,
        soft_aggregation: bool = False,
        blend_alpha: float = 0.5,
        auto_scale: bool = False,
        routed_local_training: bool = False,
        skip_last_federation_round: bool = True,
        lr: float = 0.1,
        n_epochs: int = 1,
        loss_validation: bool = False,
        loss_val_top_k: int = 3,
        dormant_recall: bool = False,
        model_type: str = "linear",
        hidden_dim: int = 64,
        adapter_dim: int = 16,
        fingerprint_source: str = "raw_input",
        expert_update_policy: str | None = None,
        shared_update_policy: str = "always",
        routed_write_top_k: int | None = None,
        routed_write_min_secondary_weight: float = 0.0,
        routed_read_top_k: int | None = None,
        routed_read_temperature: float = 1.0,
        routed_read_only_on_ambiguity: bool = False,
        routed_read_min_entropy: float | None = None,
        routed_read_min_secondary_weight: float = 0.0,
        routed_read_max_primary_gap: float | None = None,
        factorized_slot_preserving: bool = False,
        factorized_primary_anchor_alpha: float = 0.25,
        factorized_secondary_anchor_alpha: float = 0.75,
        factorized_primary_consolidation_steps: int = 0,
        factorized_primary_consolidation_mode: str = "full",
    ):
        self.config = config or TwoPhaseConfig()
        self.federation_every = federation_every
        self.detector_name = detector_name
        self.seed = seed
        self.event_triggered = event_triggered
        self.soft_aggregation = soft_aggregation
        self.blend_alpha = blend_alpha
        self.auto_scale = auto_scale
        self.routed_local_training = routed_local_training
        self.skip_last_federation_round = skip_last_federation_round
        self.lr = lr
        self.n_epochs = n_epochs
        self.loss_validation = loss_validation
        self.loss_val_top_k = loss_val_top_k
        self.dormant_recall = dormant_recall
        self.model_type = model_type
        self.hidden_dim = hidden_dim
        self.adapter_dim = adapter_dim
        self.fingerprint_source = fingerprint_source
        self.expert_update_policy = (
            "posterior_weighted"
            if expert_update_policy is None and routed_local_training
            else (expert_update_policy or "map_only")
        )
        self.shared_update_policy = shared_update_policy
        self.routed_write_top_k = routed_write_top_k
        self.routed_write_min_secondary_weight = routed_write_min_secondary_weight
        self.routed_read_top_k = routed_read_top_k
        self.routed_read_temperature = routed_read_temperature
        self.routed_read_only_on_ambiguity = routed_read_only_on_ambiguity
        self.routed_read_min_entropy = routed_read_min_entropy
        self.routed_read_min_secondary_weight = routed_read_min_secondary_weight
        self.routed_read_max_primary_gap = routed_read_max_primary_gap
        self.factorized_slot_preserving = factorized_slot_preserving
        self.factorized_primary_anchor_alpha = factorized_primary_anchor_alpha
        self.factorized_secondary_anchor_alpha = factorized_secondary_anchor_alpha
        self.factorized_primary_consolidation_steps = factorized_primary_consolidation_steps
        self.factorized_primary_consolidation_mode = factorized_primary_consolidation_mode

    def run(self, dataset: DriftDataset) -> FedProTrackResult:
        """Execute the full FedProTrack simulation.

        Parameters
        ----------
        dataset : DriftDataset
            Pre-generated dataset with concept matrix and data.

        Returns
        -------
        FedProTrackResult
        """
        gc = dataset.config
        K, T = gc.K, gc.T
        n_features = _infer_n_features(gc.generator_type, dataset)
        model_n_classes = _infer_n_classes(dataset)
        fingerprint_n_classes = _infer_fingerprint_n_classes(dataset)
        fingerprint_labels = getattr(dataset, "fingerprint_labels", None)
        if self.fingerprint_source not in {
            "raw_input",
            "model_embed",
            "pre_adapter_embed",
            "hybrid_raw_pre_adapter",
            "weighted_hybrid_raw_pre_adapter",
            "centered_hybrid_raw_pre_adapter",
            "attenuated_hybrid_raw_pre_adapter",
            "double_raw_hybrid_pre_adapter",
            "bootstrap_raw_hybrid_pre_adapter",
        }:
            raise ValueError(
                "fingerprint_source must be one of "
                "['raw_input', 'model_embed', 'pre_adapter_embed', "
                "'hybrid_raw_pre_adapter', 'weighted_hybrid_raw_pre_adapter', "
                "'centered_hybrid_raw_pre_adapter', "
                "'attenuated_hybrid_raw_pre_adapter', "
                "'double_raw_hybrid_pre_adapter', "
                "'bootstrap_raw_hybrid_pre_adapter']"
            )
        if self.expert_update_policy not in {"map_only", "posterior_weighted"}:
            raise ValueError(
                "expert_update_policy must be one of ['map_only', 'posterior_weighted']"
            )
        if self.shared_update_policy not in {"always", "freeze_on_multiroute"}:
            raise ValueError(
                "shared_update_policy must be one of ['always', 'freeze_on_multiroute']"
            )
        if self.routed_write_top_k is not None and int(self.routed_write_top_k) < 1:
            raise ValueError("routed_write_top_k must be >= 1 when provided")
        if float(self.routed_write_min_secondary_weight) < 0.0:
            raise ValueError("routed_write_min_secondary_weight must be >= 0")
        if self.routed_read_top_k is not None and int(self.routed_read_top_k) < 1:
            raise ValueError("routed_read_top_k must be >= 1 when provided")
        if float(self.routed_read_temperature) <= 0.0:
            raise ValueError("routed_read_temperature must be > 0")
        if self.routed_read_min_entropy is not None and not 0.0 <= float(self.routed_read_min_entropy) <= 1.0:
            raise ValueError("routed_read_min_entropy must be in [0, 1] when provided")
        if float(self.routed_read_min_secondary_weight) < 0.0:
            raise ValueError("routed_read_min_secondary_weight must be >= 0")
        if self.routed_read_max_primary_gap is not None and float(self.routed_read_max_primary_gap) < 0.0:
            raise ValueError("routed_read_max_primary_gap must be >= 0 when provided")
        if not 0.0 <= float(self.factorized_primary_anchor_alpha) <= 1.0:
            raise ValueError("factorized_primary_anchor_alpha must be in [0, 1]")
        if not 0.0 <= float(self.factorized_secondary_anchor_alpha) <= 1.0:
            raise ValueError("factorized_secondary_anchor_alpha must be in [0, 1]")
        if int(self.factorized_primary_consolidation_steps) < 0:
            raise ValueError("factorized_primary_consolidation_steps must be >= 0")
        if self.factorized_primary_consolidation_mode not in {"full", "head_only"}:
            raise ValueError(
                "factorized_primary_consolidation_mode must be one of ['full', 'head_only']"
            )
        fingerprint_features = n_features
        fingerprint_feature_groups = _fingerprint_feature_groups(
            self.fingerprint_source,
            n_features=n_features,
            hidden_dim=self.hidden_dim,
        )
        if self.fingerprint_source in {
            "model_embed",
            "pre_adapter_embed",
            "hybrid_raw_pre_adapter",
            "weighted_hybrid_raw_pre_adapter",
            "centered_hybrid_raw_pre_adapter",
            "attenuated_hybrid_raw_pre_adapter",
            "double_raw_hybrid_pre_adapter",
            "bootstrap_raw_hybrid_pre_adapter",
        }:
            if self.model_type != "feature_adapter":
                raise ValueError(
                    "embed-based fingerprint_source values require "
                    "model_type='feature_adapter'"
                )
            if self.fingerprint_source in {
                "hybrid_raw_pre_adapter",
                "weighted_hybrid_raw_pre_adapter",
                "centered_hybrid_raw_pre_adapter",
                "attenuated_hybrid_raw_pre_adapter",
                "double_raw_hybrid_pre_adapter",
                "bootstrap_raw_hybrid_pre_adapter",
            }:
                fingerprint_features = n_features + self.hidden_dim
                if self.fingerprint_source == "double_raw_hybrid_pre_adapter":
                    fingerprint_features = n_features * 2 + self.hidden_dim
            else:
                fingerprint_features = self.hidden_dim

        # --- Dimension-adaptive scaling ---
        # Higher-dimensional features produce noisier fingerprints, requiring
        # more conservative spawning (higher loss_novelty_threshold), more
        # aggressive merging (lower merge_threshold), tighter concept caps,
        # and less momentum blending.
        loss_novelty_th = self.config.loss_novelty_threshold
        merge_th = self.config.merge_threshold
        max_concepts = self.config.max_concepts
        blend_alpha = self.blend_alpha

        if self.auto_scale and n_features > 3:
            # Scale factor: 1.0 at dim=2, ~10.0 at dim=20
            scale = n_features / 2.0
            loss_novelty_th = max(loss_novelty_th, 0.02 * scale)
            merge_th = min(merge_th, max(0.80, 0.98 - 0.01 * (n_features - 2)))
            max_concepts = min(max_concepts, max(6, 20 - n_features // 2))
            blend_alpha = max(0.0, blend_alpha - 0.05 * (n_features - 2))

        # Update config dimensions
        cfg = TwoPhaseConfig(
            omega=self.config.omega,
            kappa=self.config.kappa,
            novelty_threshold=self.config.novelty_threshold,
            loss_novelty_threshold=loss_novelty_th,
            sticky_dampening=self.config.sticky_dampening,
            sticky_posterior_gate=self.config.sticky_posterior_gate,
            model_loss_weight=self.config.model_loss_weight,
            post_spawn_merge=self.config.post_spawn_merge,
            merge_threshold=merge_th,
            merge_min_support=self.config.merge_min_support,
            min_count=self.config.min_count,
            max_concepts=max_concepts,
            max_spawn_clusters_per_round=self.config.max_spawn_clusters_per_round,
            novelty_hysteresis_rounds=self.config.novelty_hysteresis_rounds,
            merge_every=self.config.merge_every,
            shrink_every=self.config.shrink_every,
            key_mode=self.config.key_mode,
            key_ema_decay=self.config.key_ema_decay,
            key_style_weight=self.config.key_style_weight,
            key_semantic_weight=self.config.key_semantic_weight,
            key_prototype_weight=self.config.key_prototype_weight,
            global_shared_aggregation=self.config.global_shared_aggregation,
            entropy_freeze_threshold=self.config.entropy_freeze_threshold,
            adaptive_addressing=self.config.adaptive_addressing,
            addressing_min_round_interval=self.config.addressing_min_round_interval,
            addressing_drift_threshold=self.config.addressing_drift_threshold,
            n_features=fingerprint_features,
            n_classes=fingerprint_n_classes,
        )

        protocol = TwoPhaseFedProTrack(cfg)

        # Enable dormant model preservation when dormant recall is active
        if self.dormant_recall:
            protocol.memory_bank.config.preserve_dormant_models = True

        # Per-client state
        detectors = [_make_detector(self.detector_name) for _ in range(K)]
        models = [
            _make_model(
                self.model_type,
                n_features=n_features,
                n_classes=model_n_classes,
                lr=self.lr,
                n_epochs=self.n_epochs,
                seed=self.seed + k,
                hidden_dim=self.hidden_dim,
                adapter_dim=self.adapter_dim,
            )
            for k in range(K)
        ]
        model_params: list[dict[str, np.ndarray]] = [{}] * K
        client_slot_weights: list[dict[int, float] | None] = [None] * K
        client_read_slot_weights: list[dict[int, float] | None] = [None] * K

        # Results
        accuracy_matrix = np.zeros((K, T), dtype=np.float64)
        predicted_matrix = np.zeros((K, T), dtype=np.int32)

        # Collect posteriors for soft_assignments: list of (t, posteriors_dict)
        posteriors_log: list[tuple[int, dict[int, PosteriorAssignment]]] = []
        phase_a_round_diagnostics: list[dict[str, object]] = []

        phase_a_bytes = 0.0
        phase_b_bytes = 0.0
        total_spawned = 0
        total_merged = 0
        total_pruned = 0
        prev_assignments: dict[int, int] | None = None
        old_assignments: dict[int, int] | None = None
        last_a_result: PhaseAResult | None = None
        local_update_events = 0
        multi_route_events = 0
        expert_update_total = 0.0
        shared_drift_total = 0.0
        shared_drift_events = 0

        for t in range(T):
            # Per-step fingerprints: built fresh each step from current
            # data only, avoiding cross-concept contamination.
            step_fingerprints = [
                ConceptFingerprint(
                    fingerprint_features,
                    fingerprint_n_classes,
                    feature_groups=fingerprint_feature_groups,
                )
                for _ in range(K)
            ]

            # --- Per-client: predict, detect, fingerprint, train ---
            any_drift = False
            for k in range(K):
                X, y = dataset.data[(k, t)]
                mid = len(X) // 2
                X_test, y_test = X[:mid], y[:mid]
                X_train, y_train = X[mid:], y[mid:]
                current_slot_id = (
                    prev_assignments.get(k, 0) if prev_assignments is not None else 0
                )

                # 1. Prequential prediction (on GPU via TorchLinearClassifier)
                y_pred = _model_predict(
                    models[k],
                    X_test,
                    slot_id=current_slot_id,
                    slot_weights=client_read_slot_weights[k],
                )

                acc = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.0
                accuracy_matrix[k, t] = acc

                # 2. Drift detection
                errors = (y_pred != y_test).astype(float)
                is_drift = False
                for e in errors:
                    result = detectors[k].update(e)
                    if result.is_drift:
                        is_drift = True
                        detectors[k].reset()
                        break
                if is_drift:
                    any_drift = True

                training_slot_weights = None
                update_shared = True
                shared_before_local: dict[str, np.ndarray] | None = None
                slot_anchor_payloads: dict[int, dict[str, np.ndarray]] | None = None
                effective_training_weights = {int(current_slot_id): 1.0}
                if _is_namespaced_routed_model(models[k]):
                    if self.expert_update_policy == "posterior_weighted":
                        training_slot_weights = _select_weighted_training_slots(
                            current_slot_id,
                            client_slot_weights[k],
                            top_k=self.routed_write_top_k,
                            min_secondary_weight=self.routed_write_min_secondary_weight,
                        )
                    effective_training_weights = _normalise_slot_weights(
                        current_slot_id,
                        training_slot_weights,
                    )
                    if (
                        self.shared_update_policy == "freeze_on_multiroute"
                        and len(effective_training_weights) > 1
                    ):
                        update_shared = False
                    if (
                        self.factorized_slot_preserving
                        and _is_factorized_adapter_model(models[k])
                        and len(effective_training_weights) > 1
                    ):
                        routed_slot_ids = [
                            int(slot_id)
                            for slot_id in sorted(effective_training_weights)
                        ]
                        slot_anchor_payloads = _collect_slot_anchor_payloads(
                            models[k],
                            slot_ids=routed_slot_ids,
                            memory_bank=protocol.memory_bank,
                        )
                        _warm_start_factorized_slots(
                            models[k],
                            shared_slot_id=current_slot_id,
                            slot_ids=routed_slot_ids,
                            anchor_payloads=slot_anchor_payloads,
                        )
                    shared_before_local = _model_get_params(
                        models[k],
                        slot_id=current_slot_id,
                    )
                    local_update_events += 1
                    expert_update_total += float(len(effective_training_weights))
                    if len(effective_training_weights) > 1:
                        multi_route_events += 1

                # 3. Local training on GPU
                if self.n_epochs > 1:
                    _model_fit(
                        models[k],
                        X_train,
                        y_train,
                        slot_id=current_slot_id,
                        slot_weights=training_slot_weights,
                        update_shared=update_shared,
                    )
                else:
                    _model_partial_fit(
                        models[k],
                        X_train,
                        y_train,
                        slot_id=current_slot_id,
                        slot_weights=training_slot_weights,
                        update_shared=update_shared,
                    )

                if (
                    slot_anchor_payloads is not None
                    and _is_factorized_adapter_model(models[k])
                ):
                    _apply_factorized_slot_anchor(
                        models[k],
                        slot_ids=[
                            int(slot_id)
                            for slot_id in sorted(effective_training_weights)
                        ],
                        primary_slot_id=current_slot_id,
                        anchor_payloads=slot_anchor_payloads,
                        primary_anchor_alpha=self.factorized_primary_anchor_alpha,
                        secondary_anchor_alpha=self.factorized_secondary_anchor_alpha,
                    )
                    if int(self.factorized_primary_consolidation_steps) > 0:
                        _run_factorized_primary_consolidation(
                            models[k],
                            X_train,
                            y_train,
                            slot_id=current_slot_id,
                            steps=int(self.factorized_primary_consolidation_steps),
                            mode=self.factorized_primary_consolidation_mode,
                        )

                if _is_namespaced_routed_model(models[k]) and shared_before_local is not None:
                    shared_after_local = _model_get_params(
                        models[k],
                        slot_id=current_slot_id,
                    )
                    shared_drift_total += _shared_drift_norm(
                        shared_before_local,
                        shared_after_local,
                    )
                    shared_drift_events += 1

                # 4. Build per-step fingerprint (fresh, no accumulation)
                y_fingerprint = y_train
                if isinstance(fingerprint_labels, dict) and (k, t) in fingerprint_labels:
                    y_fingerprint = fingerprint_labels[(k, t)][mid:]
                fingerprint_X = _build_fingerprint_features(
                    models[k],
                    X_train,
                    slot_id=current_slot_id,
                    slot_weights=client_read_slot_weights[k],
                    fingerprint_source=self.fingerprint_source,
                    bootstrap_hidden_dim=(
                        self.hidden_dim
                        if self.fingerprint_source
                        == "bootstrap_raw_hybrid_pre_adapter"
                        else None
                    ),
                    use_bootstrap_raw=(
                        self.fingerprint_source
                        == "bootstrap_raw_hybrid_pre_adapter"
                        and protocol.memory_bank.n_concepts <= 1
                    ),
                )
                step_fingerprints[k].update(fingerprint_X, y_fingerprint)

                if (
                    blend_alpha > 0.0
                    and not is_drift
                    and model_params[k]
                ):
                    # Warm-start with momentum only when no drift detected
                    _model_blend_params(
                        models[k],
                        model_params[k],
                        alpha=blend_alpha,
                    )

                model_params[k] = _model_get_params(
                    models[k],
                    slot_id=current_slot_id,
                )

            # --- Federation ---
            is_federation_step = (t + 1) % self.federation_every == 0
            if self.skip_last_federation_round and t == T - 1:
                is_federation_step = False
            if is_federation_step:
                # Decide whether to run Phase A
                # In event-triggered mode: only run Phase A when drift
                # detected or no assignments exist yet (bootstrap)
                run_phase_a = True
                if self.event_triggered and prev_assignments is not None:
                    run_phase_a = any_drift

                if run_phase_a:
                    # Phase A: fingerprint exchange (per-step fingerprints)
                    client_model_losses = {
                        k: 1.0 - accuracy_matrix[k, t] for k in range(K)
                    }
                    client_fps = {k: step_fingerprints[k] for k in range(K)}
                    last_a_result = protocol.phase_a(
                        client_fps, prev_assignments, client_model_losses,
                    )
                    phase_a_bytes += last_a_result.total_bytes
                    total_spawned += last_a_result.spawned
                    total_merged += last_a_result.merged
                    total_pruned += last_a_result.pruned
                    old_assignments = prev_assignments
                    prev_assignments = last_a_result.assignments
                    posteriors_log.append((t, last_a_result.posteriors))
                    phase_a_round_diagnostics.append(
                        _summarize_phase_a_round(
                            timestep=t,
                            phase_a_result=last_a_result,
                            active_after=len(set(prev_assignments.values()))
                            if prev_assignments
                            else 0,
                        )
                    )

                    # --- Loss-based concept validation ---
                    # After fingerprint-based assignment, validate by
                    # evaluating candidate models on local data. This
                    # corrects fingerprint errors in high-dimensional or
                    # class-subset-heterogeneous settings.
                    if (
                        self.loss_validation
                        and protocol.memory_bank.n_concepts >= 2
                    ):
                        all_cids = list(
                            protocol.memory_bank.concept_library.keys()
                        )
                        # Collect candidate models (top-K by posterior)
                        candidate_models: dict[int, dict[str, np.ndarray]] = {}
                        for cid in all_cids:
                            m = protocol.memory_bank.get_model_params(cid)
                            if m is not None:
                                candidate_models[cid] = m

                        if len(candidate_models) >= 2:
                            val_bytes_down = 0.0
                            for k in range(K):
                                X_k, y_k = dataset.data[(k, t)]
                                mid = len(X_k) // 2
                                X_val, y_val = X_k[:mid], y_k[:mid]

                                # Pick top-K candidates by posterior
                                post = last_a_result.posteriors.get(k)
                                if post is not None:
                                    ranked = sorted(
                                        post.probabilities.items(),
                                        key=lambda x: -x[1],
                                    )
                                    top_cids = [
                                        c for c, _ in ranked[:self.loss_val_top_k]
                                        if c in candidate_models
                                    ]
                                else:
                                    top_cids = list(candidate_models.keys())[
                                        :self.loss_val_top_k
                                    ]

                                # Ensure current assignment is included
                                cur_cid = prev_assignments[k]
                                if cur_cid not in top_cids and cur_cid in candidate_models:
                                    top_cids.append(cur_cid)

                                # Evaluate each candidate
                                best_cid = cur_cid
                                best_acc = -1.0
                                probe = _make_model(
                                    self.model_type,
                                    n_features=n_features,
                                    n_classes=model_n_classes,
                                    lr=self.lr,
                                    n_epochs=1,
                                    seed=self.seed,
                                    hidden_dim=self.hidden_dim,
                                    adapter_dim=self.adapter_dim,
                                )
                                for cid in top_cids:
                                    _model_set_params(probe, candidate_models[cid])
                                    preds = _model_predict(
                                        probe,
                                        X_val,
                                        slot_id=cid,
                                        slot_weights={cid: 1.0}
                                        if _is_namespaced_routed_model(probe)
                                        else None,
                                    )
                                    acc = float(np.mean(preds == y_val))
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_cid = cid

                                # Override if different
                                if best_cid != prev_assignments[k]:
                                    prev_assignments[k] = best_cid

                                # Comm cost: download top-K models
                                for cid in top_cids:
                                    val_bytes_down += model_bytes(
                                        candidate_models[cid],
                                        precision_bits=32,
                                    )

                            phase_a_bytes += val_bytes_down

                    # --- Server-side model probing (Insight 3: φ quality) ---
                    # Instead of relying on fingerprint matching (noisy in
                    # high dim), each client uploads a small validation batch.
                    # The server evaluates ALL stored concept models on this
                    # batch and returns the best-matching concept ID.
                    #
                    # Communication: client uploads n_probe samples (cheap),
                    # server evaluates locally, returns concept ID (4 bytes).
                    # Much cheaper than IFCA (no model download) but uses
                    # the model-prediction signal (stronger than fingerprint).
                    if self.dormant_recall:
                        all_stored = dict(protocol.memory_bank._model_store)
                        if len(all_stored) >= 2:
                            n_probe = 30  # samples per client
                            probe_bytes_up = 0.0
                            probe = _make_model(
                                self.model_type,
                                n_features=n_features,
                                n_classes=model_n_classes,
                                lr=self.lr,
                                n_epochs=1,
                                seed=self.seed,
                                hidden_dim=self.hidden_dim,
                                adapter_dim=self.adapter_dim,
                            )
                            for k in range(K):
                                X_k, y_k = dataset.data[(k, t)]
                                mid = len(X_k) // 2
                                X_val = X_k[:min(n_probe, mid)]
                                y_val = y_k[:min(n_probe, mid)]

                                # Server evaluates all stored models
                                best_cid = prev_assignments[k]
                                best_acc = -1.0
                                for cid, params in all_stored.items():
                                    _model_set_params(probe, params)
                                    preds = _model_predict(
                                        probe,
                                        X_val,
                                        slot_id=cid,
                                        slot_weights={cid: 1.0}
                                        if _is_namespaced_routed_model(probe)
                                        else None,
                                    )
                                    acc = float(np.mean(preds == y_val))
                                    if acc > best_acc:
                                        best_acc = acc
                                        best_cid = cid

                                prev_assignments[k] = best_cid

                                # Comm: upload probe samples (features + labels)
                                probe_bytes_up += (
                                    len(X_val) * n_features * 4  # features
                                    + len(y_val) * 4  # labels
                                )

                            # Download: just concept ID per client (4 bytes)
                            phase_a_bytes += probe_bytes_up + K * 4.0

                # Record concept assignments
                for k in range(K):
                    predicted_matrix[k, t] = (
                        prev_assignments.get(k, 0) if prev_assignments else 0
                    )

                # Concept-model warm-start on concept SWITCHES only:
                # When a client switches to a different concept (recurrence),
                # restore the stored model for that concept before Phase B.
                # This prevents a freshly switched feature-adapter client
                # from uploading the previous slot's expert into the new
                # concept cluster.
                if prev_assignments and old_assignments is not None:
                    for k in range(K):
                        new_cid = prev_assignments.get(k)
                        old_cid = old_assignments.get(k)
                        if (
                            new_cid is not None
                            and old_cid is not None
                            and new_cid != old_cid
                        ):
                            stored = protocol.memory_bank.get_model_params(new_cid)
                            if stored is not None:
                                _model_set_params(models[k], stored)
                                model_params[k] = {
                                    key: arr.copy()
                                    for key, arr in stored.items()
                                }
                                if _is_namespaced_routed_model(models[k]):
                                    client_slot_weights[k] = {int(new_cid): 1.0}
                                    client_read_slot_weights[k] = {int(new_cid): 1.0}

                # Phase B: model aggregation (always runs on federation steps
                # when assignments exist, even if Phase A was skipped)
                if prev_assignments:
                    client_p = {k: model_params[k] for k in range(K) if model_params[k]}
                    if client_p:
                        if self.soft_aggregation and last_a_result is not None:
                            b_result = protocol.phase_b_soft(
                                client_p, prev_assignments,
                                last_a_result.posteriors,
                            )
                        else:
                            b_result = protocol.phase_b(client_p, prev_assignments)
                        phase_b_bytes += b_result.total_bytes

                        # Distribute aggregated models (load back to GPU)
                        for k in range(K):
                            cid = prev_assignments.get(k)
                            if cid is not None and cid in b_result.aggregated_params:
                                if _is_namespaced_routed_model(models[k]):
                                    routing_weights = (
                                        last_a_result.posteriors.get(k).probabilities
                                        if last_a_result is not None
                                        and k in last_a_result.posteriors
                                        else {cid: 1.0}
                                    )
                                    candidate_payloads: dict[int, dict[str, np.ndarray]] = {}
                                    for slot_id in routing_weights:
                                        params = b_result.aggregated_params.get(slot_id)
                                        if params is None:
                                            params = protocol.memory_bank.get_model_params(slot_id)
                                        if params is not None:
                                            candidate_payloads[int(slot_id)] = params
                                    filtered_write_weights = {
                                        int(slot_id): float(weight)
                                        for slot_id, weight in routing_weights.items()
                                        if float(weight) > 0.0
                                        and int(slot_id) in candidate_payloads
                                    }
                                    read_weights = _prepare_routed_read_weights(
                                        filtered_write_weights or routing_weights,
                                        top_k=self.routed_read_top_k,
                                        temperature=self.routed_read_temperature,
                                        only_on_ambiguity=self.routed_read_only_on_ambiguity,
                                        min_entropy=self.routed_read_min_entropy,
                                        min_secondary_weight=self.routed_read_min_secondary_weight,
                                        max_primary_gap=self.routed_read_max_primary_gap,
                                    )
                                    agg_p, filtered_weights = _compose_routed_payload(
                                        candidate_payloads,
                                        read_weights or routing_weights,
                                    )
                                    if not agg_p:
                                        agg_p = {
                                            key: arr.copy()
                                            for key, arr in b_result.aggregated_params[cid].items()
                                        }
                                        filtered_write_weights = {int(cid): 1.0}
                                        filtered_weights = {cid: 1.0}
                                    model_params[k] = agg_p
                                    client_slot_weights[k] = (
                                        filtered_write_weights if filtered_write_weights else filtered_weights
                                    )
                                    client_read_slot_weights[k] = filtered_weights
                                else:
                                    model_params[k] = {
                                        key: arr.copy()
                                        for key, arr in b_result.aggregated_params[cid].items()
                                    }
                                    client_slot_weights[k] = None
                                    client_read_slot_weights[k] = None
                                _model_set_params(models[k], model_params[k])
            else:
                # No federation this step — carry forward previous assignments
                if prev_assignments:
                    for k in range(K):
                        predicted_matrix[k, t] = prev_assignments.get(k, 0)

        total_bytes = phase_a_bytes + phase_b_bytes

        # Build soft_assignments (K, T, C_max) from collected posteriors
        soft_assignments: np.ndarray | None = None
        if posteriors_log:
            # Collect all concept IDs seen across all posteriors
            all_cids: set[int] = set()
            for _, post_dict in posteriors_log:
                for pa in post_dict.values():
                    all_cids.update(pa.probabilities.keys())
            if all_cids:
                cid_list = sorted(all_cids)
                cid_to_idx = {c: i for i, c in enumerate(cid_list)}
                C = len(cid_list)
                soft_assignments = np.zeros((K, T, C), dtype=np.float64)
                for t_step, post_dict in posteriors_log:
                    for k, pa in post_dict.items():
                        for cid, prob in pa.probabilities.items():
                            if cid in cid_to_idx:
                                soft_assignments[k, t_step, cid_to_idx[cid]] = prob

        result_assignment_switch_rate = compute_assignment_switch_rate(predicted_matrix)
        result_avg_clients_per_concept = compute_avg_clients_per_concept(predicted_matrix)
        result_singleton_group_ratio = compute_singleton_group_ratio(predicted_matrix)
        result_memory_reuse_rate = compute_memory_reuse_rate(
            dataset.concept_matrix,
            predicted_matrix,
        )
        result_routing_consistency = compute_routing_consistency(
            soft_assignments,
            predicted_matrix,
        )
        result_shared_drift_norm = (
            shared_drift_total / shared_drift_events
            if shared_drift_events > 0
            else None
        )
        result_expert_update_coverage = (
            expert_update_total / local_update_events
            if local_update_events > 0
            else None
        )
        result_multi_route_rate = (
            multi_route_events / local_update_events
            if local_update_events > 0
            else None
        )

        return FedProTrackResult(
            accuracy_matrix=accuracy_matrix,
            predicted_concept_matrix=predicted_matrix,
            true_concept_matrix=dataset.concept_matrix,
            total_bytes=total_bytes,
            phase_a_bytes=phase_a_bytes,
            phase_b_bytes=phase_b_bytes,
            mean_accuracy=float(accuracy_matrix.mean()),
            final_accuracy=float(accuracy_matrix[:, -1].mean()),
            spawned_concepts=total_spawned,
            merged_concepts=total_merged,
            pruned_concepts=total_pruned,
            active_concepts=protocol.memory_bank.n_concepts,
            soft_assignments=soft_assignments,
            assignment_switch_rate=result_assignment_switch_rate,
            avg_clients_per_concept=result_avg_clients_per_concept,
            singleton_group_ratio=result_singleton_group_ratio,
            memory_reuse_rate=result_memory_reuse_rate,
            routing_consistency=result_routing_consistency,
            shared_drift_norm=result_shared_drift_norm,
            expert_update_coverage=result_expert_update_coverage,
            multi_route_rate=result_multi_route_rate,
            phase_a_round_diagnostics=phase_a_round_diagnostics,
        )
