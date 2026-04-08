from __future__ import annotations

"""Tests for the model factory and multi-architecture support."""

import numpy as np
import pytest

from fedprotrack.models.factory import MODEL_REGISTRY, create_model


class TestCreateModelLinear:
    """Tests for creating linear models via factory."""

    def test_create_linear_model(self) -> None:
        model = create_model("linear", 64, 10)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "get_params")
        assert hasattr(model, "set_params")

    def test_linear_model_fit_predict(self) -> None:
        model = create_model("linear", 4, 3, seed=42)
        X = np.random.RandomState(0).randn(20, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (20,)
        assert preds.dtype == np.int64

    def test_linear_model_params_roundtrip(self) -> None:
        model = create_model("linear", 4, 3, seed=42)
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)
        model.fit(X, y)
        params = model.get_params()
        model2 = create_model("linear", 4, 3, seed=99)
        model2.set_params(params)
        preds1 = model.predict(X)
        preds2 = model2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)


class TestCreateModelSmallCNN:
    """Tests for creating small_cnn models via factory."""

    def test_create_small_cnn_model(self) -> None:
        model = create_model("small_cnn", 3072, 10)
        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "get_params")
        assert hasattr(model, "set_params")

    def test_small_cnn_fit_predict(self) -> None:
        model = create_model("small_cnn", 3072, 5, seed=42)
        # 3 channels * 32 * 32 = 3072
        X = np.random.RandomState(0).randn(8, 3072).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (8,)
        assert preds.dtype == np.int64

    def test_small_cnn_params_roundtrip(self) -> None:
        model = create_model("small_cnn", 3072, 5, seed=42)
        X = np.random.RandomState(0).randn(8, 3072).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
        model.fit(X, y)
        params = model.get_params()
        model2 = create_model("small_cnn", 3072, 5, seed=99)
        model2.set_params(params)
        preds1 = model.predict(X)
        preds2 = model2.predict(X)
        np.testing.assert_array_equal(preds1, preds2)

    def test_small_cnn_predict_loss(self) -> None:
        model = create_model("small_cnn", 3072, 5, seed=42)
        X = np.random.RandomState(0).randn(8, 3072).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
        model.fit(X, y)
        loss = model.predict_loss(X, y)
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_small_cnn_blend_params(self) -> None:
        model1 = create_model("small_cnn", 3072, 5, seed=42)
        model2 = create_model("small_cnn", 3072, 5, seed=99)
        X = np.random.RandomState(0).randn(8, 3072).astype(np.float32)
        y = np.array([0, 1, 2, 3, 4, 0, 1, 2], dtype=np.int64)
        model1.fit(X, y)
        model2.fit(X, y)
        other_params = model2.get_params()
        model1.blend_params(other_params, alpha=0.5)
        preds = model1.predict(X)
        assert preds.shape == (8,)


class TestFactoryErrors:
    """Tests for factory error handling."""

    def test_unknown_model_type_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown model_type"):
            create_model("nonexistent_model", 64, 10)

    def test_registry_has_linear(self) -> None:
        assert "linear" in MODEL_REGISTRY

    def test_registry_has_small_cnn(self) -> None:
        # Trigger lazy registration
        create_model("small_cnn", 64, 10)
        assert "small_cnn" in MODEL_REGISTRY


class TestBaselineModelTypePassthrough:
    """Test that baseline runner functions accept model_type without error."""

    def test_cfl_client_accepts_model_type(self) -> None:
        from fedprotrack.baselines.cfl import CFLClient

        client = CFLClient(0, 4, 3, seed=42, model_type="linear")
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (10,)

    def test_ifca_client_accepts_model_type(self) -> None:
        from fedprotrack.baselines.ifca import IFCAClient

        client = IFCAClient(0, 4, 3, seed=42, model_type="linear")
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (10,)

    def test_flash_client_accepts_model_type(self) -> None:
        from fedprotrack.baselines.flash import FlashClient

        client = FlashClient(0, 4, 3, seed=42, model_type="linear")
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (10,)

    def test_compressed_fedavg_client_accepts_model_type(self) -> None:
        from fedprotrack.baselines.compressed_fedavg import CompressedFedAvgClient

        client = CompressedFedAvgClient(0, 4, 3, seed=42, model_type="linear")
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (10,)

    def test_fedem_client_accepts_model_type(self) -> None:
        from fedprotrack.baselines.fedem import FedEMClient

        client = FedEMClient(0, 4, 3, seed=42, model_type="linear")
        X = np.random.RandomState(0).randn(10, 4).astype(np.float32)
        y = np.array([0, 1, 2] * 3 + [0], dtype=np.int64)
        client.fit(X, y)
        preds = client.predict(X)
        assert preds.shape == (10,)
