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
    elif generator_type in ("gaussian_linear", "gaussian_anisotropic"):
        n_variants = n_concepts  # each concept gets a unique weight vector
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
    if spec.generator_type in ("gaussian_linear", "gaussian_anisotropic"):
        return _generate_gaussian_samples(spec, n_samples, seed)

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


def _generate_gaussian_samples(
    spec: ConceptSpec,
    n_samples: int,
    seed: int,
    d: int = 20,
    sigma: float = 1.0,
    r_signal: int = 5,
    signal_eigenvalue: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate Gaussian linear regression data for theory validation.

    Parameters
    ----------
    spec : ConceptSpec
        ``generator_type`` is ``"gaussian_linear"`` (isotropic, X ~ N(0, I_d))
        or ``"gaussian_anisotropic"`` (spiked, first ``r_signal`` eigenvalues
        are ``signal_eigenvalue``, rest are 1).
        ``variant`` is the concept ID (used to generate per-concept weight).
        ``noise_scale`` is ``1 - delta``, controlling concept separation.
    n_samples : int
    seed : int

    Returns
    -------
    X : np.ndarray of shape ``(n_samples, d)``
    y : np.ndarray of shape ``(n_samples,)`` with binary labels {0, 1}
    """
    rng = np.random.default_rng(seed)

    # Generate per-concept weight vector (deterministic from concept ID).
    w_rng = np.random.default_rng(42 + spec.concept_id * 7919)
    w_star = w_rng.normal(0, 1, size=d)
    # Scale by delta: higher delta = more separation between concepts.
    delta = 1.0 - spec.noise_scale
    w_star *= delta * 3.0  # scale so concepts are well-separated at delta=1

    if spec.generator_type == "gaussian_linear":
        # Isotropic: X ~ N(0, I_d)
        X = rng.normal(0, 1, size=(n_samples, d))
    elif spec.generator_type == "gaussian_anisotropic":
        # Spiked covariance: first r_signal dims have large eigenvalue.
        X = rng.normal(0, 1, size=(n_samples, d))
        X[:, :r_signal] *= np.sqrt(signal_eigenvalue)
        # Concept separation lives in the signal subspace.
        w_star[r_signal:] *= 0.1
    else:
        raise ValueError(f"Unexpected: {spec.generator_type}")

    # Linear regression: y = sign(<w*, x> + noise)
    logits = X @ w_star + rng.normal(0, sigma, size=n_samples)
    y = (logits > 0).astype(np.int32)

    return X.astype(np.float64), y


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
