import numpy as np
import pytest

from fedprotrack.drift_generator.data_streams import (
    ConceptSpec,
    generate_samples,
    make_concept_specs,
)


def test_make_specs_sine():
    specs = make_concept_specs(n_concepts=5, generator_type="sine", delta=0.7)
    assert len(specs) == 5
    assert specs[0].noise_scale == pytest.approx(0.3)
    # Variants cycle 0-3
    assert [s.variant for s in specs] == [0, 1, 2, 3, 0]


def test_make_specs_sea():
    specs = make_concept_specs(n_concepts=3, generator_type="sea", delta=1.0)
    assert len(specs) == 3
    assert specs[0].noise_scale == pytest.approx(0.0)


def test_generate_samples_sine_shape():
    spec = ConceptSpec(concept_id=0, generator_type="sine", variant=0, noise_scale=0.0)
    X, y = generate_samples(spec, n_samples=100, seed=42)
    assert X.shape == (100, 2)
    assert y.shape == (100,)
    assert set(np.unique(y)).issubset({0, 1})


def test_generate_samples_sea_shape():
    spec = ConceptSpec(concept_id=0, generator_type="sea", variant=0, noise_scale=0.0)
    X, y = generate_samples(spec, n_samples=100, seed=42)
    assert X.shape == (100, 3)
    assert y.shape == (100,)


def test_generate_samples_circle_shape():
    spec = ConceptSpec(concept_id=0, generator_type="circle", variant=0, noise_scale=0.0)
    X, y = generate_samples(spec, n_samples=100, seed=42)
    assert X.shape == (100, 2)
    assert y.shape == (100,)


def test_noise_increases_with_lower_delta():
    """Lower delta should add more noise, making classification harder."""
    spec_clean = ConceptSpec(concept_id=0, generator_type="sine", variant=0, noise_scale=0.0)
    spec_noisy = ConceptSpec(concept_id=0, generator_type="sine", variant=0, noise_scale=0.9)

    X_clean, _ = generate_samples(spec_clean, n_samples=500, seed=42)
    X_noisy, _ = generate_samples(spec_noisy, n_samples=500, seed=42)

    # Noisy data should have larger variance
    assert X_noisy.std() > X_clean.std()


def test_reproducibility():
    spec = ConceptSpec(concept_id=0, generator_type="sine", variant=0, noise_scale=0.3)
    X1, y1 = generate_samples(spec, n_samples=50, seed=42)
    X2, y2 = generate_samples(spec, n_samples=50, seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
