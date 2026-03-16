"""Data stream generation for federated concept drift.

Each concept ID maps to a parameterized data-generating distribution.
The separability parameter delta controls how distinguishable concepts are
via additive noise: delta=1.0 → perfectly separable, delta→0 → nearly random.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from river.datasets import synth


@dataclass
class ConceptSpec:
    """Specification for a single concept's data generator."""

    concept_id: int
    generator_type: str  # "sine", "sea", "circle"
    variant: int         # generator variant index
    noise_scale: float   # noise magnitude (derived from delta)


def make_concept_specs(
    n_concepts: int, generator_type: str, delta: float
) -> list[ConceptSpec]:
    """Create concept specifications mapping each concept ID to a generator variant.

    Parameters
    ----------
    n_concepts : int
        Number of distinct concepts.
    generator_type : str
        One of "sine", "sea", "circle".
    delta : float
        Separability in (0, 1]. Higher → more separable (less noise).

    Returns
    -------
    specs : list of ConceptSpec
    """
    noise_scale = 1.0 - delta  # delta=1 → noise=0, delta=0.1 → noise=0.9

    if generator_type == "sine":
        n_variants = 4
    elif generator_type == "sea":
        n_variants = 4
    elif generator_type == "circle":
        n_variants = 1  # circle has no built-in variants, we shift center
    else:
        raise ValueError(f"Unknown generator_type: {generator_type}")

    specs = []
    for c in range(n_concepts):
        variant = c % n_variants
        specs.append(ConceptSpec(
            concept_id=c,
            generator_type=generator_type,
            variant=variant,
            noise_scale=noise_scale,
        ))
    return specs


def generate_samples(
    spec: ConceptSpec, n_samples: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (X, y) arrays for a single concept.

    Parameters
    ----------
    spec : ConceptSpec
        Concept specification.
    n_samples : int
        Number of samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,) with int labels
    """
    if spec.generator_type == "sine":
        gen = synth.Sine(
            classification_function=spec.variant,
            seed=seed,
            has_noise=False,
        )
        n_features = 2
    elif spec.generator_type == "sea":
        gen = synth.SEA(
            variant=spec.variant,
            seed=seed,
            noise=0.0,
        )
        n_features = 3
    elif spec.generator_type == "circle":
        gen = _make_circle_generator(spec.variant, seed)
        n_features = 2
    else:
        raise ValueError(f"Unknown generator_type: {spec.generator_type}")

    X = np.empty((n_samples, n_features), dtype=np.float64)
    y = np.empty(n_samples, dtype=np.int32)

    for i, (x_dict, label) in enumerate(gen):
        if i >= n_samples:
            break
        X[i] = [x_dict[k] for k in sorted(x_dict.keys())]
        y[i] = int(label)

    # Add noise controlled by delta (via noise_scale)
    if spec.noise_scale > 0:
        rng = np.random.default_rng(seed + 99999)
        noise = rng.normal(0, spec.noise_scale, size=X.shape)
        X = X + noise

    return X, y


def _make_circle_generator(variant: int, seed: int):
    """Create a circle-variant generator by shifting the decision boundary.

    For circle, we use Sine generator with function 0 as the base,
    then apply a spatial shift per variant to create distinct concepts.
    Since river doesn't have a built-in circle with variants, we use
    a simple 2D circular decision boundary generator.
    """
    # Use a custom generator for circle variants
    return _CircleGenerator(variant=variant, seed=seed)


class _CircleGenerator:
    """Simple 2D circular decision boundary generator with variant-based shifts."""

    def __init__(self, variant: int, seed: int):
        self.variant = variant
        self.rng = np.random.default_rng(seed)
        # Shift center based on variant
        self.center_x = 0.5 + variant * 0.3
        self.center_y = 0.5
        self.radius = 0.3

    def __iter__(self):
        while True:
            x0 = self.rng.random()
            x1 = self.rng.random()
            dist = np.sqrt((x0 - self.center_x) ** 2 + (x1 - self.center_y) ** 2)
            label = 1 if dist <= self.radius else 0
            yield {0: x0, 1: x1}, label
