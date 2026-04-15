"""Rotating MNIST dataset for federated concept drift experiments (E6).

Concepts correspond to discrete rotation angles applied to MNIST digits.
Clients experience concept drift by switching between rotation angles
according to the same (rho, alpha, delta) concept matrix used for
synthetic experiments.

The feature representation uses PCA-reduced pixel values to keep
dimensionality compatible with linear classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from ..drift_generator.configs import GeneratorConfig
from ..drift_generator.data_streams import ConceptSpec
from ..drift_generator.generator import DriftDataset
from ..drift_generator.concept_matrix import generate_concept_matrix


@dataclass
class RotatingMNISTConfig:
    """Configuration for Rotating MNIST dataset generation.

    Parameters
    ----------
    K : int
        Number of federated clients.
    T : int
        Number of time steps.
    n_samples : int
        Samples per client per time step.
    rho : float
        Recurrence period for concept cycling.
    alpha : float
        Asynchrony parameter (0 = synchronous, 1 = fully async).
    delta : float
        Concept separability (mapped to rotation angle range).
    n_concepts : int
        Number of distinct rotation angles.
    n_features : int
        PCA dimensions for feature representation.
    seed : int
        Random seed.
    """

    K: int = 5
    T: int = 10
    n_samples: int = 200
    rho: float = 5.0
    alpha: float = 0.5
    delta: float = 0.5
    n_concepts: int = 4
    n_features: int = 20
    seed: int = 42


def _load_mnist() -> tuple[np.ndarray, np.ndarray]:
    """Load MNIST training data using torchvision.

    Returns
    -------
    images : np.ndarray
        Shape (N, 28, 28), float32 in [0, 1].
    labels : np.ndarray
        Shape (N,), int.
    """
    from torchvision import datasets, transforms

    # Download to a standard cache directory
    mnist = datasets.MNIST(
        root=".mnist_cache", train=True, download=True,
        transform=transforms.ToTensor(),
    )

    images = mnist.data.numpy().astype(np.float32) / 255.0
    labels = mnist.targets.numpy()
    return images, labels


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a 28x28 image by the given angle (in degrees).

    Uses scipy.ndimage for rotation with bilinear interpolation.

    Parameters
    ----------
    image : np.ndarray
        Shape (28, 28).
    angle_deg : float
        Rotation angle in degrees.

    Returns
    -------
    np.ndarray
        Rotated image, same shape.
    """
    from scipy.ndimage import rotate

    rotated = rotate(image, angle_deg, reshape=False, order=1, mode="constant", cval=0.0)
    return rotated.astype(np.float32)


def _rotation_angles(n_concepts: int, delta: float) -> list[float]:
    """Generate rotation angles for concepts.

    Higher delta means more separation between concepts
    (larger angle differences).

    Parameters
    ----------
    n_concepts : int
    delta : float
        In [0, 1]. Controls max rotation range.

    Returns
    -------
    list[float]
        Rotation angles in degrees.
    """
    # Map delta to angle range: delta=0.1 -> 18 deg, delta=1.0 -> 180 deg
    max_angle = delta * 180.0
    if n_concepts == 1:
        return [0.0]
    step = max_angle / (n_concepts - 1)
    return [i * step for i in range(n_concepts)]


def generate_rotating_mnist_dataset(
    config: RotatingMNISTConfig | None = None,
) -> DriftDataset:
    """Generate a Rotating MNIST dataset compatible with FedProTrack pipeline.

    Each concept corresponds to a different rotation angle applied to
    MNIST digits. The concept matrix follows the same (rho, alpha) recurrence
    structure as synthetic experiments.

    Parameters
    ----------
    config : RotatingMNISTConfig, optional

    Returns
    -------
    DriftDataset
        Compatible with all existing runners and metrics.
    """
    if config is None:
        config = RotatingMNISTConfig()

    rng = np.random.RandomState(config.seed)

    # Load MNIST
    images, labels = _load_mnist()

    # Build concept matrix using the same machinery as synthetic experiments
    gen_config = GeneratorConfig(
        K=config.K,
        T=config.T,
        n_samples=config.n_samples,
        rho=config.rho,
        alpha=config.alpha,
        delta=config.delta,
        generator_type="sine",  # placeholder for interface compat
        seed=config.seed,
    )
    concept_matrix = generate_concept_matrix(
        K=config.K, T=config.T,
        n_concepts=config.n_concepts,
        alpha=config.alpha,
        seed=config.seed,
    )

    # Rotation angles for each concept
    angles = _rotation_angles(config.n_concepts, config.delta)

    # Pre-rotate all images for each angle and fit PCA
    unique_concepts = sorted(set(concept_matrix.flatten()))
    rotated_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    for concept_id in unique_concepts:
        angle = angles[concept_id % len(angles)]
        # Rotate all images
        rot_imgs = np.array([_rotate_image(img, angle) for img in images])
        # Flatten to (N, 784)
        flat = rot_imgs.reshape(len(rot_imgs), -1)
        rotated_pools[concept_id] = (flat, labels)

    # Fit PCA on a combined sample from all concepts
    combined = np.vstack([
        pool[0][rng.choice(len(pool[0]), min(2000, len(pool[0])), replace=False)]
        for pool in rotated_pools.values()
    ])
    pca = PCA(n_components=config.n_features, random_state=config.seed)
    pca.fit(combined)

    # Transform all pools
    pca_pools: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for concept_id, (flat, lab) in rotated_pools.items():
        pca_flat = pca.transform(flat).astype(np.float32)
        pca_pools[concept_id] = (pca_flat, lab)

    # Build data dict: (k, t) -> (X, y) with binary labels
    # Binary task: even vs odd digit
    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(config.K):
        for t in range(config.T):
            concept_id = int(concept_matrix[k, t])
            pool_X, pool_y = pca_pools[concept_id]
            n_pool = len(pool_X)

            # Sample n_samples with replacement
            idx = rng.choice(n_pool, config.n_samples, replace=True)
            X = pool_X[idx]
            y = (pool_y[idx] % 2).astype(np.int32)  # even=0, odd=1
            data[(k, t)] = (X, y)

    # Build concept specs (minimal, for interface compatibility)
    concept_specs = [
        ConceptSpec(
            concept_id=cid,
            generator_type="rotating_mnist",
            variant=cid,
            noise_scale=0.0,
        )
        for cid in unique_concepts
    ]

    return DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=gen_config,
        concept_specs=concept_specs,
    )
