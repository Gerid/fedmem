from __future__ import annotations

import numpy as np

from fedprotrack.models import TorchFeatureAdapterClassifier


def _toy_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    X = rng.randn(64, 6).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.int64)
    return X, y


class TestTorchFeatureAdapterClassifier:
    def test_namespaced_payload_contains_shared_and_expert_keys(self) -> None:
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=10,
            adapter_dim=4,
            n_epochs=1,
            seed=7,
        )
        params = model.get_params()
        assert "shared.trunk.weight" in params
        assert "shared.trunk.bias" in params
        assert any(key.startswith("expert.0.") for key in params)

    def test_round_trip_params_preserves_predictions(self) -> None:
        X, y = _toy_data()
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=12,
            adapter_dim=4,
            n_epochs=2,
            seed=11,
        )
        model.fit(X, y, slot_id=2)
        params = model.get_params()

        clone = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=12,
            adapter_dim=4,
            n_epochs=2,
            seed=99,
        )
        clone.set_params(params)
        np.testing.assert_array_equal(
            model.predict(X, slot_id=2),
            clone.predict(X, slot_id=2),
        )

    def test_predict_with_weighted_expert_mixture(self) -> None:
        X, y = _toy_data(seed=3)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=8,
            adapter_dim=3,
            n_epochs=1,
            seed=13,
        )
        model.partial_fit(X, y, slot_id=0)
        model.partial_fit(X, y, slot_id=1)
        preds = model.predict(X, slot_weights={0: 0.75, 1: 0.25})
        assert preds.shape == (len(X),)
        assert set(np.unique(preds)).issubset({0, 1})
