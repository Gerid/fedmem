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

    def test_embed_returns_hidden_dim_for_single_slot(self) -> None:
        X, y = _toy_data(seed=4)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=9,
            adapter_dim=3,
            n_epochs=1,
            seed=19,
        )
        model.partial_fit(X, y, slot_id=0)
        hidden = model.embed(X, slot_id=0)
        assert hidden.shape == (len(X), 9)
        assert hidden.dtype == np.float32

    def test_embed_supports_weighted_slot_mixture(self) -> None:
        X, y = _toy_data(seed=9)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=7,
            adapter_dim=3,
            n_epochs=1,
            seed=23,
        )
        model.partial_fit(X, y, slot_id=0)
        model.partial_fit(X, y, slot_id=1)
        hidden0 = model.embed(X, slot_id=0)
        hidden1 = model.embed(X, slot_id=1)
        mixed = model.embed(X, slot_weights={0: 0.25, 1: 0.75})
        np.testing.assert_allclose(mixed, 0.25 * hidden0 + 0.75 * hidden1, atol=1e-6)

    def test_pre_adapter_embed_ignores_slot_specific_mixture(self) -> None:
        X, y = _toy_data(seed=10)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=7,
            adapter_dim=3,
            n_epochs=1,
            seed=31,
        )
        model.partial_fit(X, y, slot_id=0)
        model.partial_fit(X, y, slot_id=1)
        pre_slot0 = model.embed(X, slot_id=0, representation="pre_adapter")
        pre_mixed = model.embed(
            X,
            slot_weights={0: 0.1, 1: 0.9},
            representation="pre_adapter",
        )
        np.testing.assert_allclose(pre_slot0, pre_mixed, atol=1e-6)

    def test_weighted_training_updates_multiple_slots(self) -> None:
        X, y = _toy_data(seed=5)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=8,
            adapter_dim=3,
            n_epochs=1,
            seed=17,
        )
        model.partial_fit(X, y, slot_id=0)
        params_before = model.get_params(slot_id=0)
        params_before_slot1 = model.get_params(slot_id=1)

        model.partial_fit(X, y, slot_id=0, slot_weights={0: 0.4, 1: 0.6})
        params_after_slot0 = model.get_params(slot_id=0)
        params_after_slot1 = model.get_params(slot_id=1)

        assert not np.allclose(
            params_before["expert.0.head.weight"],
            params_after_slot0["expert.0.head.weight"],
        )
        assert not np.allclose(
            params_before_slot1["expert.1.head.weight"],
            params_after_slot1["expert.1.head.weight"],
        )

    def test_weighted_training_can_freeze_shared_params(self) -> None:
        X, y = _toy_data(seed=12)
        model = TorchFeatureAdapterClassifier(
            n_features=6,
            n_classes=2,
            hidden_dim=8,
            adapter_dim=3,
            n_epochs=1,
            seed=29,
        )
        model.partial_fit(X, y, slot_id=0)
        model.partial_fit(X, y, slot_id=1)
        params_before = model.get_params(slot_id=0)
        params_before_slot1 = model.get_params(slot_id=1)

        model.partial_fit(
            X,
            y,
            slot_id=0,
            slot_weights={0: 0.4, 1: 0.6},
            update_shared=False,
        )
        params_after_slot0 = model.get_params(slot_id=0)
        params_after_slot1 = model.get_params(slot_id=1)

        np.testing.assert_allclose(
            params_before["shared.trunk.weight"],
            params_after_slot0["shared.trunk.weight"],
            atol=1e-10,
        )
        np.testing.assert_allclose(
            params_before["shared.trunk.bias"],
            params_after_slot0["shared.trunk.bias"],
            atol=1e-10,
        )
        assert not np.allclose(
            params_before["expert.0.head.weight"],
            params_after_slot0["expert.0.head.weight"],
        )
        assert not np.allclose(
            params_before_slot1["expert.1.head.weight"],
            params_after_slot1["expert.1.head.weight"],
        )
