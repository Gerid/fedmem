from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ExperimentLog:
    """Container for a single experiment's predictions and metadata.

    Parameters
    ----------
    ground_truth : np.ndarray
        Shape (K, T), dtype int32. Ground-truth concept IDs per client per step.
    predicted : np.ndarray
        Shape (K, T), dtype int32. Predicted concept IDs per client per step.
    soft_assignments : np.ndarray or None
        Shape (K, T, C), dtype float64. Soft assignment probabilities.
    accuracy_curve : np.ndarray or None
        Shape (K, T), dtype float64. Per-step classification accuracy.
    total_bytes : float or None
        Total communication cost in bytes.
    method_name : str
        Human-readable name for the method (e.g. "FedAvg", "ConceptAwareFedAvg").
    """

    ground_truth: np.ndarray
    predicted: np.ndarray
    soft_assignments: np.ndarray | None = None
    accuracy_curve: np.ndarray | None = None
    total_bytes: float | None = None
    method_name: str = "unknown"

    def __post_init__(self) -> None:
        """Validate that ground_truth and predicted have the same shape."""
        if self.ground_truth.shape != self.predicted.shape:
            raise ValueError(
                f"ground_truth shape {self.ground_truth.shape} does not match "
                f"predicted shape {self.predicted.shape}"
            )
        if self.ground_truth.ndim != 2:
            raise ValueError(
                f"ground_truth must be 2-D (K, T), got shape {self.ground_truth.shape}"
            )
        # Ensure correct dtypes
        self.ground_truth = self.ground_truth.astype(np.int32)
        self.predicted = self.predicted.astype(np.int32)
        if self.soft_assignments is not None:
            self.soft_assignments = self.soft_assignments.astype(np.float64)
        if self.accuracy_curve is not None:
            self.accuracy_curve = self.accuracy_curve.astype(np.float64)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, path: str | Path) -> None:
        """Save ground_truth, predicted, and scalar metadata to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path (will be overwritten if it exists).
        """
        path = Path(path)
        payload: dict = {
            "method_name": self.method_name,
            "total_bytes": self.total_bytes,
            "shape_K": int(self.ground_truth.shape[0]),
            "shape_T": int(self.ground_truth.shape[1]),
            "ground_truth": self.ground_truth.tolist(),
            "predicted": self.predicted.tolist(),
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentLog:
        """Load an ExperimentLog from a JSON file (no soft_assignments/accuracy_curve).

        Parameters
        ----------
        path : str or Path
            JSON file previously created by :meth:`to_json`.

        Returns
        -------
        ExperimentLog
        """
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            ground_truth=np.array(payload["ground_truth"], dtype=np.int32),
            predicted=np.array(payload["predicted"], dtype=np.int32),
            total_bytes=payload.get("total_bytes"),
            method_name=payload.get("method_name", "unknown"),
        )

    def save(self, output_dir: str | Path) -> None:
        """Persist the log to *output_dir*.

        Creates the following files inside *output_dir*:

        * ``log.json`` — ground_truth, predicted, and scalar metadata.
        * ``arrays.npz`` — soft_assignments and accuracy_curve (if present).

        Parameters
        ----------
        output_dir : str or Path
            Directory to write into (created if it does not exist).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.to_json(output_dir / "log.json")

        # Save optional dense arrays in a single compressed archive.
        arrays: dict[str, np.ndarray] = {}
        if self.soft_assignments is not None:
            arrays["soft_assignments"] = self.soft_assignments
        if self.accuracy_curve is not None:
            arrays["accuracy_curve"] = self.accuracy_curve
        if arrays:
            np.savez_compressed(output_dir / "arrays.npz", **arrays)

    @classmethod
    def load(cls, output_dir: str | Path) -> ExperimentLog:
        """Load an ExperimentLog previously saved with :meth:`save`.

        Parameters
        ----------
        output_dir : str or Path
            Directory written by :meth:`save`.

        Returns
        -------
        ExperimentLog
        """
        output_dir = Path(output_dir)
        log = cls.from_json(output_dir / "log.json")

        npz_path = output_dir / "arrays.npz"
        if npz_path.exists():
            npz = np.load(npz_path)
            if "soft_assignments" in npz:
                log.soft_assignments = npz["soft_assignments"].astype(np.float64)
            if "accuracy_curve" in npz:
                log.accuracy_curve = npz["accuracy_curve"].astype(np.float64)

        return log


# ---------------------------------------------------------------------------


@dataclass
class MetricsResult:
    """Aggregated evaluation metrics for a single experiment run.

    Parameters
    ----------
    concept_re_id_accuracy : float
        Global fraction of (k, t) cells where the aligned predicted concept
        matches the ground truth.
    assignment_entropy : float
        Mean entropy of the concept assignment distribution.
    wrong_memory_reuse_rate : float
        Fraction of (k, t) cells where the aligned prediction is wrong
        (1 − concept_re_id_accuracy).
    worst_window_dip : float or None
        Largest accuracy drop inside any sliding evaluation window.
    worst_window_recovery : int or None
        Number of steps required to recover from the worst window dip.
    budget_normalized_score : float or None
        Accuracy-AUC divided by communication cost, normalised to [0, 1].
    per_client_re_id : np.ndarray
        Shape (K,). Per-client concept re-identification accuracy.
    per_timestep_re_id : np.ndarray
        Shape (T,). Per-timestep concept re-identification accuracy.
    """

    concept_re_id_accuracy: float
    assignment_entropy: float
    wrong_memory_reuse_rate: float
    worst_window_dip: float | None
    worst_window_recovery: int | None
    budget_normalized_score: float | None
    per_client_re_id: np.ndarray
    per_timestep_re_id: np.ndarray
    final_accuracy: float | None = None
    accuracy_auc: float | None = None

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation.

        Returns
        -------
        dict
            All scalar fields are kept as Python scalars; numpy arrays are
            converted to plain Python lists.
        """
        return {
            "concept_re_id_accuracy": float(self.concept_re_id_accuracy),
            "assignment_entropy": float(self.assignment_entropy),
            "wrong_memory_reuse_rate": float(self.wrong_memory_reuse_rate),
            "worst_window_dip": (
                float(self.worst_window_dip)
                if self.worst_window_dip is not None
                else None
            ),
            "worst_window_recovery": (
                int(self.worst_window_recovery)
                if self.worst_window_recovery is not None
                else None
            ),
            "budget_normalized_score": (
                float(self.budget_normalized_score)
                if self.budget_normalized_score is not None
                else None
            ),
            "per_client_re_id": self.per_client_re_id.tolist(),
            "per_timestep_re_id": self.per_timestep_re_id.tolist(),
            "final_accuracy": (
                float(self.final_accuracy)
                if self.final_accuracy is not None
                else None
            ),
            "accuracy_auc": (
                float(self.accuracy_auc)
                if self.accuracy_auc is not None
                else None
            ),
        }

    def to_json(self, path: str | Path) -> None:
        """Serialise to a JSON file.

        Parameters
        ----------
        path : str or Path
            Destination file path.
        """
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
