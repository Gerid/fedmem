"""Budget frontier analysis including FedProTrack.

Extends the existing budget sweep to include FedProTrack alongside
FedAvg-Full, FedProto, and TrackedSummary for a complete comparison.
"""

from __future__ import annotations

from pathlib import Path

from ..baselines.budget_sweep import BudgetPoint, run_budget_sweep
from ..drift_generator import DriftDataset
from ..metrics.budget_metrics import compute_accuracy_auc
from ..metrics.visualization import plot_budget_frontier
from ..posterior.fedprotrack_runner import FedProTrackRunner
from ..posterior.two_phase_protocol import TwoPhaseConfig


def run_fedprotrack_budget_points(
    dataset: DriftDataset,
    federation_every_values: list[int] | None = None,
    config: TwoPhaseConfig | None = None,
    seed: int = 42,
) -> list[BudgetPoint]:
    """Run FedProTrack at multiple federation frequencies.

    Parameters
    ----------
    dataset : DriftDataset
    federation_every_values : list[int], optional
        Default [1, 2, 5, 10].
    config : TwoPhaseConfig, optional
    seed : int

    Returns
    -------
    list[BudgetPoint]
    """
    if federation_every_values is None:
        federation_every_values = [1, 2, 5, 10]

    points: list[BudgetPoint] = []
    for fe in federation_every_values:
        runner = FedProTrackRunner(
            config=config,
            federation_every=fe,
            seed=seed,
        )
        result = runner.run(dataset)
        auc = compute_accuracy_auc(result.accuracy_matrix)
        points.append(BudgetPoint(
            method_name="FedProTrack",
            federation_every=fe,
            total_bytes=result.total_bytes,
            accuracy_auc=auc,
        ))

    return points


def generate_full_budget_frontier(
    dataset: DriftDataset,
    output_path: Path | str,
    federation_every_values: list[int] | None = None,
    similarity_threshold: float = 0.5,
    seed: int = 42,
) -> Path:
    """Generate a budget frontier plot with all 4 methods.

    Parameters
    ----------
    dataset : DriftDataset
    output_path : Path or str
    federation_every_values : list[int], optional
    similarity_threshold : float
    seed : int

    Returns
    -------
    Path
        Path to saved figure.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if federation_every_values is None:
        federation_every_values = [1, 2, 5, 10]

    # Existing baselines
    baseline_points = run_budget_sweep(
        dataset, federation_every_values, similarity_threshold,
    )

    # FedProTrack
    fpt_points = run_fedprotrack_budget_points(
        dataset, federation_every_values, seed=seed,
    )

    # Group by method
    method_points: dict[str, list[BudgetPoint]] = {}
    for p in baseline_points + fpt_points:
        if p.method_name not in method_points:
            method_points[p.method_name] = []
        method_points[p.method_name].append(p)

    plot_budget_frontier(
        method_points,
        output_path=output_path,
        title="Budget Frontier: All Methods",
    )

    return output_path
