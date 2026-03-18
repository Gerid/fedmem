from __future__ import annotations

import numpy as np

from fedprotrack.federation import NamespacedExpertAggregator


class TestNamespacedExpertAggregator:
    def test_aggregate_namespaced_splits_shared_and_experts(self) -> None:
        agg = NamespacedExpertAggregator()
        client_params = [
            {
                "shared.trunk.weight": np.array([[1.0, 2.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[1.0, 1.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([0.0], dtype=np.float64),
            },
            {
                "shared.trunk.weight": np.array([[3.0, 4.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                "expert.1.head.weight": np.array([[5.0, 5.0]], dtype=np.float64),
                "expert.1.head.bias": np.array([1.0], dtype=np.float64),
            },
        ]
        result = agg.aggregate_namespaced(client_params)
        np.testing.assert_allclose(
            result.shared_params["shared.trunk.weight"],
            np.array([[2.0, 3.0]], dtype=np.float64),
        )
        assert 0 in result.expert_params
        assert 1 in result.expert_params

    def test_routed_weights_affect_slot_aggregation(self) -> None:
        agg = NamespacedExpertAggregator()
        client_params = [
            {
                "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[2.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([0.0], dtype=np.float64),
            },
            {
                "shared.trunk.weight": np.array([[3.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([2.0], dtype=np.float64),
                "expert.0.head.weight": np.array([[10.0]], dtype=np.float64),
                "expert.0.head.bias": np.array([2.0], dtype=np.float64),
            },
        ]
        result = agg.aggregate_namespaced(
            client_params,
            expert_weights=[{0: 1.0}, {0: 0.25}],
        )
        np.testing.assert_allclose(
            result.expert_params[0]["expert.0.head.weight"],
            np.array([[3.6]], dtype=np.float64),
            atol=1e-8,
        )

    def test_flat_aggregate_returns_namespaced_dict(self) -> None:
        agg = NamespacedExpertAggregator()
        merged = agg.aggregate([
            {
                "shared.trunk.weight": np.array([[1.0]], dtype=np.float64),
                "shared.trunk.bias": np.array([0.0], dtype=np.float64),
                "expert.2.head.weight": np.array([[4.0]], dtype=np.float64),
                "expert.2.head.bias": np.array([1.0], dtype=np.float64),
            }
        ])
        assert "shared.trunk.weight" in merged
        assert "expert.2.head.weight" in merged
