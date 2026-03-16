import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset


def test_end_to_end_small():
    """End-to-end test with a small configuration."""
    config = GeneratorConfig(
        K=3, T=4, n_samples=50, rho=2.0, alpha=0.5, delta=0.5,
        generator_type="sine", seed=42,
    )
    dataset = generate_drift_dataset(config)

    assert dataset.concept_matrix.shape == (3, 4)
    assert len(dataset.data) == 3 * 4
    for (k, t), (X, y) in dataset.data.items():
        assert X.shape == (50, 2)
        assert y.shape == (50,)


def test_save_and_load(tmp_path):
    config = GeneratorConfig(
        K=2, T=3, n_samples=20, rho=2.0, alpha=0.0, delta=0.7,
        generator_type="sine", seed=42,
    )
    dataset = generate_drift_dataset(config)
    out_dir = dataset.save(base_dir=tmp_path)

    # Check files exist
    assert (out_dir / "config.json").exists()
    assert (out_dir / "concept_matrix.npy").exists()
    assert (out_dir / "concept_matrix.csv").exists()
    assert (out_dir / "concept_specs.json").exists()

    # Check data files
    for k in range(2):
        for t in range(3):
            assert (out_dir / "data" / f"client_{k:02d}_step_{t:02d}.npz").exists()

    # Verify concept matrix roundtrip
    loaded = np.load(out_dir / "concept_matrix.npy")
    np.testing.assert_array_equal(loaded, dataset.concept_matrix)

    # Verify config roundtrip
    loaded_config = GeneratorConfig.from_json(out_dir / "config.json")
    assert loaded_config.K == config.K
    assert loaded_config.seed == config.seed


def test_sea_generator():
    config = GeneratorConfig(
        K=2, T=3, n_samples=30, rho=2.0, alpha=0.5, delta=0.5,
        generator_type="sea", seed=42,
    )
    dataset = generate_drift_dataset(config)
    for (k, t), (X, y) in dataset.data.items():
        assert X.shape == (30, 3)


def test_concept_count_matches_rho():
    """n_concepts from config should match what's used in the matrix."""
    config = GeneratorConfig(K=5, T=10, rho=5.0, alpha=0.0, seed=42)
    assert config.n_concepts == 2  # round(10/5) = 2
    dataset = generate_drift_dataset(config)
    unique = np.unique(dataset.concept_matrix)
    assert len(unique) <= 2
