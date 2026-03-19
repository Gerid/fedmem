from __future__ import annotations

import numpy as np
import pytest

from fedprotrack.posterior.routing_similarity import (
    normalize_mass,
    pairwise_cosine_cost,
    pairwise_euclidean_cost,
    prototype_cost_matrix,
    prototype_transport_cost,
    prototype_transport_similarity,
    sinkhorn_transport_cost,
    sinkhorn_transport_plan,
    sinkhorn_transport_similarity,
)


def test_sinkhorn_cost_increases_under_uniform_cost_shift() -> None:
    cost = np.array(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    mass = np.array([0.5, 0.5], dtype=np.float64)
    base = sinkhorn_transport_cost(cost, mass, mass, reg=0.2, max_iter=200)
    shifted = sinkhorn_transport_cost(
        cost + 0.25,
        mass,
        mass,
        reg=0.2,
        max_iter=200,
    )
    assert shifted > base


def test_sinkhorn_plan_shape_and_marginals() -> None:
    cost = np.array(
        [
            [0.1, 0.4, 0.8],
            [0.2, 0.3, 0.5],
        ],
        dtype=np.float64,
    )
    source_mass = np.array([0.7, 0.3], dtype=np.float64)
    target_mass = np.array([0.2, 0.5, 0.3], dtype=np.float64)

    plan = sinkhorn_transport_plan(
        cost,
        source_mass,
        target_mass,
        reg=0.3,
        max_iter=300,
    )

    assert plan.shape == cost.shape
    np.testing.assert_allclose(plan.sum(axis=1), normalize_mass(source_mass), atol=1e-3)
    np.testing.assert_allclose(plan.sum(axis=0), normalize_mass(target_mass), atol=1e-3)

    similarity = sinkhorn_transport_similarity(
        cost,
        source_mass,
        target_mass,
        reg=0.3,
        max_iter=300,
    )
    assert 0.0 < similarity <= 1.0


def test_prototype_transport_similarity_prefers_nearby_prototypes() -> None:
    source = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    near = np.array(
        [
            [0.1, -0.1],
            [0.9, 1.1],
        ],
        dtype=np.float64,
    )
    far = np.array(
        [
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=np.float64,
    )

    sim_near = prototype_transport_similarity(source, near, metric="euclidean")
    sim_far = prototype_transport_similarity(source, far, metric="euclidean")

    assert 0.0 <= sim_far < sim_near <= 1.0


def test_pairwise_cost_shapes_are_stable() -> None:
    source = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [0.0, 1.0, 0.0],
            [2.0, 0.0, 1.0],
            [1.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    euclidean = pairwise_euclidean_cost(source, target)
    cosine = pairwise_cosine_cost(source, target)
    proto = prototype_cost_matrix(source, target, metric="sqeuclidean")

    assert euclidean.shape == (2, 3)
    assert cosine.shape == (2, 3)
    assert proto.shape == (2, 3)
    assert np.all(np.isfinite(euclidean))
    assert np.all(np.isfinite(cosine))
    assert np.all(np.isfinite(proto))


def test_invalid_parameters_raise() -> None:
    cost = np.array([[0.0, 1.0]], dtype=np.float64)
    mass = np.array([1.0], dtype=np.float64)

    with pytest.raises(ValueError, match="reg must be > 0"):
        sinkhorn_transport_plan(cost, mass, np.array([1.0, 0.0]), reg=0.0)

    with pytest.raises(ValueError, match="source_mass must be non-negative"):
        sinkhorn_transport_cost(cost, np.array([-1.0]), np.array([1.0, 0.0]))

    with pytest.raises(ValueError, match="target_mass must have positive total mass"):
        sinkhorn_transport_cost(cost, mass, np.array([0.0, 0.0]))

    with pytest.raises(ValueError, match="cost shape"):
        sinkhorn_transport_plan(
            np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
        )

    with pytest.raises(ValueError, match="temperature must be > 0"):
        sinkhorn_transport_similarity(cost, mass, np.array([1.0, 0.0]), temperature=0.0)

    with pytest.raises(ValueError, match="metric must be one of"):
        prototype_cost_matrix(
            np.array([[0.0, 1.0]], dtype=np.float64),
            np.array([[1.0, 0.0]], dtype=np.float64),
            metric="bad",
        )

    with pytest.raises(ValueError, match="must be a 2-D array"):
        pairwise_euclidean_cost(np.array([1.0, 2.0], dtype=np.float64), np.array([[1.0, 2.0]], dtype=np.float64))


def test_prototype_transport_cost_matches_similarity_bounds() -> None:
    source = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )
    target = np.array(
        [
            [0.05, -0.05],
            [1.05, 0.95],
        ],
        dtype=np.float64,
    )
    cost = prototype_transport_cost(source, target, metric="euclidean")
    similarity = prototype_transport_similarity(source, target, metric="euclidean")

    assert cost >= 0.0
    assert 0.0 < similarity <= 1.0
