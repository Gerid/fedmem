import numpy as np
import pytest

from fedprotrack.drift_generator.concept_matrix import (
    generate_concept_matrix,
    _generate_markov_sequence,
)


def test_matrix_shape():
    matrix = generate_concept_matrix(K=5, T=8, n_concepts=3, alpha=0.5, seed=42)
    assert matrix.shape == (5, 8)
    assert matrix.dtype == np.int32


def test_sync_all_rows_identical():
    """With alpha=0, all clients should have the same drift schedule."""
    matrix = generate_concept_matrix(K=10, T=10, n_concepts=3, alpha=0.0, seed=42)
    for k in range(1, 10):
        np.testing.assert_array_equal(matrix[0], matrix[k])


def test_async_rows_differ():
    """With alpha=1, clients should generally have different schedules."""
    matrix = generate_concept_matrix(K=10, T=10, n_concepts=3, alpha=1.0, seed=42)
    # Not all rows identical (with very high probability)
    all_same = all(np.array_equal(matrix[0], matrix[k]) for k in range(1, 10))
    assert not all_same


def test_single_concept():
    """With n_concepts=1, the matrix should be all zeros (but we require n_concepts>=2)."""
    # When T/rho is very large, n_concepts is small
    matrix = generate_concept_matrix(K=5, T=10, n_concepts=2, alpha=0.0, seed=42)
    # Should only use concepts 0 and 1
    assert set(np.unique(matrix)).issubset({0, 1})


def test_concept_ids_in_range():
    matrix = generate_concept_matrix(K=10, T=10, n_concepts=5, alpha=0.5, seed=42)
    assert matrix.min() >= 0
    assert matrix.max() < 5


def test_reproducibility():
    m1 = generate_concept_matrix(K=5, T=8, n_concepts=3, alpha=0.5, seed=123)
    m2 = generate_concept_matrix(K=5, T=8, n_concepts=3, alpha=0.5, seed=123)
    np.testing.assert_array_equal(m1, m2)


def test_markov_sequence_length():
    rng = np.random.default_rng(42)
    seq = _generate_markov_sequence(T=20, n_concepts=4, rng=rng)
    assert len(seq) == 20
    assert seq.min() >= 0
    assert seq.max() < 4
