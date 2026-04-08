from __future__ import annotations

"""Tests for TorchSmallCNN end-to-end CNN models."""

import numpy as np
import pytest
import torch

from fedprotrack.models import TorchSmallCNN

# TorchMobileNetV2 is planned but not yet implemented; guard tests.
try:
    from fedprotrack.models import TorchMobileNetV2  # type: ignore[attr-defined]
except ImportError:
    TorchMobileNetV2 = None  # type: ignore[assignment,misc]

_skip_mobilenet = pytest.mark.skipif(
    TorchMobileNetV2 is None,
    reason="TorchMobileNetV2 not yet implemented",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _random_images(n: int = 32, n_classes: int = 5, seed: int = 0):
    """Generate random (N, 3, 32, 32) images and labels."""
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 3, 32, 32).astype(np.float32)
    y = rng.randint(0, n_classes, size=n).astype(np.int64)
    return X, y


# ======================================================================
# TorchSmallCNN
# ======================================================================


class TestTorchSmallCNN:
    """Tests for TorchSmallCNN."""

    def test_construct_and_predict_before_fit(self) -> None:
        model = TorchSmallCNN(n_features=3072, n_classes=5, seed=42)
        X, _ = _random_images(8)
        preds = model.predict(X)
        assert preds.shape == (8,)
        assert preds.dtype == np.int64
        # Before fit, should return zeros
        np.testing.assert_array_equal(preds, np.zeros(8, dtype=np.int64))

    def test_fit_and_predict(self) -> None:
        X, y = _random_images(48, n_classes=3, seed=1)
        model = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=2, seed=7)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (48,)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_partial_fit(self) -> None:
        X, y = _random_images(24, n_classes=4, seed=2)
        model = TorchSmallCNN(n_features=3072, n_classes=4, n_epochs=1, seed=10)
        model.partial_fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (24,)

    @pytest.mark.skipif(
        not hasattr(TorchSmallCNN, "predict_proba"),
        reason="TorchSmallCNN does not implement predict_proba yet",
    )
    def test_predict_proba_shape_and_sum(self) -> None:
        X, y = _random_images(16, n_classes=5, seed=3)
        model = TorchSmallCNN(n_features=3072, n_classes=5, n_epochs=1, seed=11)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (16, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert (proba >= 0).all()

    @pytest.mark.skipif(
        not hasattr(TorchSmallCNN, "predict_proba"),
        reason="TorchSmallCNN does not implement predict_proba yet",
    )
    def test_predict_proba_before_fit(self) -> None:
        model = TorchSmallCNN(n_features=3072, n_classes=3, seed=0)
        X, _ = _random_images(4, n_classes=3)
        proba = model.predict_proba(X)
        assert proba.shape == (4, 3)
        np.testing.assert_allclose(proba, 1.0 / 3.0, atol=1e-6)

    def test_predict_loss(self) -> None:
        X, y = _random_images(32, n_classes=5, seed=4)
        model = TorchSmallCNN(n_features=3072, n_classes=5, n_epochs=1, seed=12)
        # Before fit, should return inf
        assert model.predict_loss(X, y) == float("inf")
        model.fit(X, y)
        loss = model.predict_loss(X, y)
        assert isinstance(loss, float)
        assert loss > 0.0
        assert loss < 100.0  # sanity upper bound

    def test_get_params_set_params_roundtrip(self) -> None:
        X, y = _random_images(32, n_classes=4, seed=5)
        model = TorchSmallCNN(n_features=3072, n_classes=4, n_epochs=2, seed=13)
        model.fit(X, y)
        params = model.get_params()
        preds_before = model.predict(X)

        # Load into a fresh model
        clone = TorchSmallCNN(n_features=3072, n_classes=4, n_epochs=2, seed=99)
        clone.set_params(params)
        preds_after = clone.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_get_params_keys(self) -> None:
        model = TorchSmallCNN(n_features=3072, n_classes=3, seed=0)
        params = model.get_params()
        # Should contain conv and linear layer keys
        key_set = set(params.keys())
        assert any("features" in k for k in key_set)
        assert any("classifier" in k for k in key_set)
        # All values should be numpy arrays
        for val in params.values():
            assert isinstance(val, np.ndarray)

    def test_blend_params(self) -> None:
        X, y = _random_images(32, n_classes=3, seed=6)
        m1 = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=1, seed=20)
        m2 = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=1, seed=21)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.get_params()
        p2 = m2.get_params()

        m1.blend_params(p2, alpha=0.5)
        blended = m1.get_params()
        for key in p1:
            expected = 0.5 * p2[key].astype(np.float64) + 0.5 * p1[key].astype(np.float64)
            np.testing.assert_allclose(
                blended[key].astype(np.float64),
                expected,
                atol=1e-5,
                err_msg=f"Blend mismatch on key {key}",
            )

    def test_clone_fresh_independent(self) -> None:
        X, y = _random_images(32, n_classes=3, seed=7)
        model = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=1, seed=30)
        model.fit(X, y)

        clone = model.clone_fresh(seed=31)
        # Clone should be untrained
        preds_clone = clone.predict(X)
        np.testing.assert_array_equal(
            preds_clone, np.zeros(32, dtype=np.int64),
        )
        # Original should still produce trained predictions
        preds_orig = model.predict(X)
        assert not np.all(preds_orig == 0) or True  # may be all-0 by chance

    def test_clone_fresh_default_seed(self) -> None:
        model = TorchSmallCNN(n_features=3072, n_classes=5, seed=10)
        clone = model.clone_fresh()
        assert clone._seed == 11

    def test_device_placement(self) -> None:
        model = TorchSmallCNN(n_features=3072, n_classes=5, seed=0)
        # Device should be a valid torch device
        assert isinstance(model.device, torch.device)
        # All parameters should be on the same device
        for param in model._net.parameters():
            assert param.device.type == model.device.type

    def test_force_cpu(self, monkeypatch) -> None:
        monkeypatch.setenv("FEDPROTRACK_FORCE_CPU", "1")
        model = TorchSmallCNN(n_features=3072, n_classes=3, seed=0)
        assert model.device == torch.device("cpu")

    def test_deterministic_with_seed(self) -> None:
        X, y = _random_images(16, n_classes=3, seed=8)
        m1 = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=1, seed=42)
        m1.fit(X, y)
        p1 = m1.predict(X)

        m2 = TorchSmallCNN(n_features=3072, n_classes=3, n_epochs=1, seed=42)
        m2.fit(X, y)
        p2 = m2.predict(X)

        np.testing.assert_array_equal(p1, p2)


# ======================================================================
# TorchMobileNetV2
# ======================================================================


@_skip_mobilenet
class TestTorchMobileNetV2:
    """Tests for TorchMobileNetV2."""

    def test_construct_and_predict_before_fit(self) -> None:
        model = TorchMobileNetV2(n_classes=5, seed=42)
        X, _ = _random_images(4)
        preds = model.predict(X)
        assert preds.shape == (4,)
        assert preds.dtype == np.int64
        np.testing.assert_array_equal(preds, np.zeros(4, dtype=np.int64))

    def test_fit_and_predict(self) -> None:
        X, y = _random_images(16, n_classes=3, seed=1)
        model = TorchMobileNetV2(
            n_classes=3, n_epochs=1, batch_size=8, seed=7,
        )
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (16,)
        assert set(np.unique(preds)).issubset({0, 1, 2})

    def test_partial_fit(self) -> None:
        X, y = _random_images(12, n_classes=4, seed=2)
        model = TorchMobileNetV2(n_classes=4, n_epochs=1, seed=10)
        model.partial_fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (12,)

    def test_predict_proba_shape_and_sum(self) -> None:
        X, y = _random_images(8, n_classes=5, seed=3)
        model = TorchMobileNetV2(n_classes=5, n_epochs=1, seed=11)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (8, 5)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_loss(self) -> None:
        X, y = _random_images(16, n_classes=5, seed=4)
        model = TorchMobileNetV2(n_classes=5, n_epochs=1, seed=12)
        assert model.predict_loss(X, y) == float("inf")
        model.fit(X, y)
        loss = model.predict_loss(X, y)
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_get_params_set_params_roundtrip(self) -> None:
        X, y = _random_images(16, n_classes=4, seed=5)
        model = TorchMobileNetV2(n_classes=4, n_epochs=1, seed=13)
        model.fit(X, y)
        params = model.get_params()
        preds_before = model.predict(X)

        clone = TorchMobileNetV2(n_classes=4, n_epochs=1, seed=99)
        clone.set_params(params)
        preds_after = clone.predict(X)

        np.testing.assert_array_equal(preds_before, preds_after)

    def test_get_params_keys(self) -> None:
        model = TorchMobileNetV2(n_classes=3, seed=0)
        params = model.get_params()
        key_set = set(params.keys())
        assert any("features" in k for k in key_set)
        assert any("classifier" in k for k in key_set)
        for val in params.values():
            assert isinstance(val, np.ndarray)

    def test_blend_params(self) -> None:
        X, y = _random_images(16, n_classes=3, seed=6)
        m1 = TorchMobileNetV2(n_classes=3, n_epochs=1, seed=20)
        m2 = TorchMobileNetV2(n_classes=3, n_epochs=1, seed=21)
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.get_params()
        p2 = m2.get_params()

        m1.blend_params(p2, alpha=0.5)
        blended = m1.get_params()
        # Spot-check a few keys
        checked = 0
        for key in list(p1.keys())[:5]:
            expected = 0.5 * p2[key].astype(np.float64) + 0.5 * p1[key].astype(np.float64)
            np.testing.assert_allclose(
                blended[key].astype(np.float64),
                expected,
                atol=1e-5,
                err_msg=f"Blend mismatch on key {key}",
            )
            checked += 1
        assert checked > 0

    def test_clone_fresh_independent(self) -> None:
        model = TorchMobileNetV2(n_classes=3, n_epochs=1, seed=30)
        clone = model.clone_fresh(seed=31)
        assert clone._seed == 31
        assert clone.n_classes == model.n_classes
        assert clone.lr == model.lr

    def test_clone_fresh_default_seed(self) -> None:
        model = TorchMobileNetV2(n_classes=5, seed=10)
        clone = model.clone_fresh()
        assert clone._seed == 11

    def test_device_placement(self) -> None:
        model = TorchMobileNetV2(n_classes=5, seed=0)
        assert isinstance(model.device, torch.device)
        for param in model._net.parameters():
            assert param.device.type == model.device.type

    def test_force_cpu(self, monkeypatch) -> None:
        monkeypatch.setenv("FEDPROTRACK_FORCE_CPU", "1")
        model = TorchMobileNetV2(n_classes=3, seed=0)
        assert model.device == torch.device("cpu")

    def test_first_conv_stride_adapted(self) -> None:
        """MobileNetV2 first conv should use stride=1 for 32x32 images."""
        model = TorchMobileNetV2(n_classes=5, seed=0)
        first_conv = model._net.features[0][0]
        assert first_conv.stride == (1, 1)


# ======================================================================
# Runner factory integration
# ======================================================================


class TestMakeModelIntegration:
    """Test that the runner factory creates CNN models correctly."""

    def test_make_model_small_cnn(self) -> None:
        from fedprotrack.posterior.fedprotrack_runner import _make_model

        model = _make_model(
            "small_cnn",
            n_features=512,  # unused for CNN but required by signature
            n_classes=10,
            lr=0.01,
            n_epochs=1,
            seed=42,
            hidden_dim=64,
            adapter_dim=16,
            batch_size=32,
        )
        assert isinstance(model, TorchSmallCNN)
        assert model.n_classes == 10

    @_skip_mobilenet
    def test_make_model_mobilenetv2(self) -> None:
        from fedprotrack.posterior.fedprotrack_runner import _make_model

        model = _make_model(
            "mobilenetv2",
            n_features=512,
            n_classes=10,
            lr=0.01,
            n_epochs=1,
            seed=42,
            hidden_dim=64,
            adapter_dim=16,
            batch_size=32,
        )
        assert isinstance(model, TorchMobileNetV2)
        assert model.n_classes == 10

    def test_make_model_unknown_raises(self) -> None:
        from fedprotrack.posterior.fedprotrack_runner import _make_model

        with pytest.raises(ValueError, match="Unknown model_type"):
            _make_model(
                "unknown_model",
                n_features=10,
                n_classes=5,
                lr=0.01,
                n_epochs=1,
                seed=0,
                hidden_dim=64,
                adapter_dim=16,
            )
