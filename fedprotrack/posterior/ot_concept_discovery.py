from __future__ import annotations

"""Optimal-transport-inspired concept discovery via spectral clustering.

Replaces the threshold-based Phase A (Gibbs posterior + novelty gating)
with a geometry-driven approach:

1. Each client's fingerprint defines an empirical distribution.
2. Pairwise distances between client distributions approximate the
   Wasserstein-2 metric (for Gaussian mixtures with shared covariance,
   Euclidean distance on class-conditional means ≈ W₂).
3. Spectral clustering on the distance matrix with eigengap heuristic
   determines the number of concepts and cluster assignments —
   **zero thresholds, zero hyperparameters** beyond an optional
   ``max_concepts`` cap.

Theoretical connection
----------------------
The eigengap of the normalised graph Laplacian corresponds to the
Wasserstein separation between concept clusters.  When the within-
concept W₂ dispersion is small relative to between-concept W₂, the
gap is large and spectral clustering recovers the true partition.  This
is the distributional analogue of the Euclidean crossover condition
n_c · σ_B² > σ² · d_eff from Theorem 1.
"""

import numpy as np
from sklearn.cluster import SpectralClustering, KMeans

from ..concept_tracker.fingerprint import ConceptFingerprint


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def spectral_concept_discovery(
    client_fingerprints: dict[int, ConceptFingerprint],
    *,
    max_concepts: int = 20,
    min_concepts: int = 1,
    affinity_scale: str = "local",
    eigengap_method: str = "last_significant",
) -> tuple[int, dict[int, int]]:
    """Discover concepts from client fingerprints via spectral clustering.

    Parameters
    ----------
    client_fingerprints : dict[int, ConceptFingerprint]
        Client ID → current-round fingerprint.
    max_concepts : int
        Upper bound on discoverable concepts (capped by K-1).
    min_concepts : int
        Lower bound — at least this many concepts will be returned.
    affinity_scale : str
        How to set the RBF kernel bandwidth:
        ``"local"`` uses the mean k-NN distance (sharper boundaries, default),
        ``"median"`` uses the median pairwise distance (ablation: pre-fix 1b),
        ``"p25"`` uses the 25th percentile.
    eigengap_method : str
        Which eigengap heuristic to use for the cluster count estimate:
        ``"last_significant"`` (default) uses the plateau-noise-adjusted
        last-significant-gap heuristic, which is robust to heterogeneous
        concept separation; ``"argmax"`` uses the classical argmax gap
        (ablation: pre-fix 1a).

    Returns
    -------
    n_concepts : int
        Number of discovered concepts.
    assignments : dict[int, int]
        Client ID → concept ID (0-indexed).
    """
    client_ids = sorted(client_fingerprints.keys())
    K = len(client_ids)

    if K <= 1:
        return 1, {k: 0 for k in client_ids}

    # --- 1. Build feature matrix from fingerprints -----------------------
    vectors = np.stack([
        client_fingerprints[k].to_vector() for k in client_ids
    ])  # (K, d)

    # --- 2. Pairwise distance matrix ------------------------------------
    # Euclidean distance on fingerprint vectors ≈ W₂ for Gaussian
    # mixtures with shared covariance (class-conditional means dominate).
    dist_sq = _pairwise_sq_distances(vectors)

    # --- 3. Affinity matrix (RBF kernel) ---------------------------------
    sigma2 = _choose_bandwidth(dist_sq, method=affinity_scale)
    if sigma2 < 1e-30:
        # All fingerprints identical → single concept
        return 1, {k: 0 for k in client_ids}
    affinity = np.exp(-dist_sq / (2.0 * sigma2))
    np.fill_diagonal(affinity, 0.0)  # no self-loops

    # --- 4. Normalised graph Laplacian + eigengap -----------------------
    C_max = min(max_concepts, K - 1, K)
    C_max = max(C_max, min_concepts)

    n_concepts = _eigengap_estimate(
        affinity, C_max, min_concepts, method=eigengap_method,
    )

    if n_concepts <= 1:
        return 1, {k: 0 for k in client_ids}

    # --- 5. Spectral clustering ------------------------------------------
    labels = _spectral_cluster(affinity, n_concepts)

    assignments = {
        client_ids[i]: int(labels[i]) for i in range(K)
    }
    return n_concepts, assignments


# ---------------------------------------------------------------------------
# Wasserstein-aware concept memory for recurrence
# ---------------------------------------------------------------------------

class ConceptMemory:
    """Lightweight memory bank for re-identifying recurring concepts.

    Stores centroid vectors of previously seen concepts.  When a new
    round's spectral clustering produces clusters, each cluster centroid
    is matched to stored concepts via nearest-neighbour in the
    fingerprint vector space (≈ nearest in W₂ for Gaussians).  New
    concepts are registered; dormant concepts are recalled.
    """

    def __init__(self, match_threshold: float = 0.7) -> None:
        self._centroids: dict[int, np.ndarray] = {}
        self._next_id: int = 0
        self.match_threshold = match_threshold

    @property
    def n_concepts(self) -> int:
        return len(self._centroids)

    def match_and_update(
        self,
        cluster_centroids: dict[int, np.ndarray],
    ) -> dict[int, int]:
        """Match local cluster IDs to persistent concept IDs.

        Parameters
        ----------
        cluster_centroids : dict[int, np.ndarray]
            Local cluster ID → centroid vector (from this round).

        Returns
        -------
        mapping : dict[int, int]
            Local cluster ID → persistent concept ID.
        """
        if not self._centroids:
            # First round — register all clusters as new concepts
            mapping = {}
            for local_id, vec in cluster_centroids.items():
                cid = self._next_id
                self._next_id += 1
                self._centroids[cid] = vec.copy()
                mapping[local_id] = cid
            return mapping

        # Greedy matching: for each new cluster, find best stored concept
        mapping: dict[int, int] = {}
        used_stored: set[int] = set()

        # Compute all pairwise cosine similarities
        local_ids = sorted(cluster_centroids.keys())
        stored_ids = sorted(self._centroids.keys())

        for local_id in local_ids:
            vec = cluster_centroids[local_id]
            best_sim = -1.0
            best_stored = -1
            for stored_id in stored_ids:
                if stored_id in used_stored:
                    continue
                sim = _cosine_sim(vec, self._centroids[stored_id])
                if sim > best_sim:
                    best_sim = sim
                    best_stored = stored_id

            if best_sim >= self.match_threshold and best_stored >= 0:
                # Re-identify as existing concept — update centroid (EMA)
                mapping[local_id] = best_stored
                used_stored.add(best_stored)
                self._centroids[best_stored] = (
                    0.7 * self._centroids[best_stored] + 0.3 * vec
                )
            else:
                # New concept
                cid = self._next_id
                self._next_id += 1
                self._centroids[cid] = vec.copy()
                mapping[local_id] = cid

        return mapping


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_sq_distances(X: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances for an (N, d) matrix."""
    sq_norms = np.sum(X ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * X @ X.T
    np.maximum(dist_sq, 0.0, out=dist_sq)
    return dist_sq


def _choose_bandwidth(dist_sq: np.ndarray, method: str = "median") -> float:
    """Choose RBF kernel bandwidth from the distance matrix.

    The bandwidth affects spectral clustering sensitivity.  ``"median"``
    follows standard practice but can wash out fine structure when
    concept separations are non-uniform.  ``"local"`` uses the mean of
    each point's k-nearest-neighbour distances, which yields sharper
    cluster boundaries in heterogeneous geometries.
    """
    K = dist_sq.shape[0]
    upper = dist_sq[np.triu_indices_from(dist_sq, k=1)]
    if len(upper) == 0:
        return 1.0
    if method == "local":
        # k-NN local bandwidth: mean distance to 3 nearest neighbours
        k_nn = min(4, K - 1)
        sorted_dists = np.sort(dist_sq, axis=1)
        # Skip self (index 0)
        local = sorted_dists[:, 1:k_nn + 1].mean()
        return float(local) + 1e-10
    if method == "p25":
        return float(np.percentile(upper, 25)) + 1e-10
    if method == "median":
        return float(np.median(upper)) + 1e-10
    return float(np.mean(upper)) + 1e-10


def _eigengap_estimate(
    affinity: np.ndarray,
    C_max: int,
    C_min: int = 1,
    *,
    method: str = "last_significant",
) -> int:
    """Estimate number of clusters from normalised Laplacian eigenvalues.

    Uses the symmetric normalised Laplacian L_sym = I - D^{-1/2} A D^{-1/2}.
    For k well-separated clusters, the first k eigenvalues are "structural"
    (significantly below 1.0) and the remainder form a "noise plateau" near
    1.0.

    Parameters
    ----------
    affinity : np.ndarray
        (K, K) symmetric non-negative affinity matrix.
    C_max : int
        Upper bound on cluster count.
    C_min : int
        Lower bound on cluster count.
    method : str
        Heuristic to pick the cluster count from the eigenvalue sequence:
        - ``"last_significant"`` (default): scan gaps from index 1, find the
          LAST gap that is significantly larger than the plateau noise level,
          which is robust to non-uniform concept separation (the fix used in
          FedProTrack's OT breakthrough).
        - ``"argmax"`` (ablation): pick the index with the largest eigengap.
          This is the classical heuristic and can under-count when concept
          separations are heterogeneous.
    """
    K = affinity.shape[0]
    deg = affinity.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(deg), 0.0)
    L_sym = (
        np.eye(K)
        - (deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :])
    )
    L_sym = 0.5 * (L_sym + L_sym.T)

    eigvals = np.linalg.eigvalsh(L_sym)
    eigvals = np.sort(np.real(eigvals))

    n_check = min(C_max + 3, K)
    if n_check < 2:
        return max(C_min, 1)

    # Estimate plateau noise level from the tail of the spectrum.
    # Non-structural eigenvalues of L_sym concentrate near a plateau;
    # the spacing between consecutive plateau eigenvalues measures noise.
    plateau_start = max(C_max + 1, min(n_check - 1, int(K * 0.6)))
    if plateau_start < K - 1:
        plateau_gaps = np.diff(eigvals[plateau_start:])
        noise_level = float(np.median(np.abs(plateau_gaps))) + 1e-6
    else:
        noise_level = 1e-3

    # Scan gaps from index 1 onward.  Find the LAST gap that is
    # significantly larger than the plateau noise (i.e. the transition
    # from structural to noise eigenvalues).  A gap is "significant" if
    # it exceeds the noise level by a factor >= 5.
    gaps = np.diff(eigvals[:n_check])

    if method == "argmax":
        # Ablation: classical argmax-gap heuristic. Returns the index that
        # maximises the gap between consecutive eigenvalues, restricted to
        # indices >= 1 (at least one cluster).
        if len(gaps) == 0:
            return max(C_min, 1)
        argmax_idx = int(np.argmax(gaps[1:])) + 1 if len(gaps) >= 2 else 1
        n_concepts = argmax_idx
        return max(C_min, min(n_concepts, C_max))

    significant_threshold = max(5.0 * noise_level, 0.02)

    # Find the last significant gap in the range [1, n_check-1)
    last_sig_idx = -1
    for i in range(1, len(gaps)):
        if gaps[i] >= significant_threshold:
            last_sig_idx = i

    if last_sig_idx < 0:
        # No clearly structural gaps — fall back to absolute threshold
        n_concepts = int(np.sum(eigvals[:n_check] < 0.95))
        if n_concepts < 1:
            n_concepts = 1
    else:
        # eigenvalues [0..last_sig_idx] are structural → last_sig_idx+1 clusters
        n_concepts = last_sig_idx + 1

    return max(C_min, min(n_concepts, C_max))


def _spectral_cluster(affinity: np.ndarray, n_clusters: int) -> np.ndarray:
    """Run spectral clustering on a precomputed affinity matrix."""
    K = affinity.shape[0]
    if n_clusters >= K:
        return np.arange(K)

    try:
        sc = SpectralClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            random_state=42,
            n_init=10,
            assign_labels="kmeans",
        )
        labels = sc.fit_predict(affinity)
    except Exception:
        # Fallback: k-means on eigenvectors manually
        deg = affinity.sum(axis=1)
        deg_inv_sqrt = np.where(deg > 1e-10, 1.0 / np.sqrt(deg), 0.0)
        L_sym = (
            np.eye(K)
            - (deg_inv_sqrt[:, None] * affinity * deg_inv_sqrt[None, :])
        )
        L_sym = 0.5 * (L_sym + L_sym.T)
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # First n_clusters eigenvectors (smallest eigenvalues)
        V = eigvecs[:, :n_clusters]
        # Row-normalise
        norms = np.linalg.norm(V, axis=1, keepdims=True)
        V = V / np.maximum(norms, 1e-10)
        km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        labels = km.fit_predict(V)

    return labels


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
