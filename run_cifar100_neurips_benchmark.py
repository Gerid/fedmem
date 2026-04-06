from __future__ import annotations

"""NeurIPS-scale CIFAR-100 federated concept drift benchmark.

Provides a publication-grade experiment entry script matching top-conference
FL benchmarks:
  - K = 20--100 clients, T = 100--200 rounds, E = 5 local epochs
  - Partial client participation (20% for large-scale)
  - Multiple drift types: sudden, incremental, recurrent
  - Multi-seed runs with aggregated statistics and error bars
  - Communication budget tracking for all methods
  - Predefined presets: small / medium / large / full

Example usage
-------------
Quick development check::

    python run_cifar100_neurips_benchmark.py --quick

Medium preset for paper main table (K=20, T=200, 5 seeds)::

    python run_cifar100_neurips_benchmark.py --preset medium

Large-scale scalability experiment (K=100, T=200, 20% participation)::

    python run_cifar100_neurips_benchmark.py --preset large

Selective method groups::

    python run_cifar100_neurips_benchmark.py --methods core,cluster --K 20 --T 200
"""

import argparse
import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Ensure unbuffered output for experiment logs
# ---------------------------------------------------------------------------
os.environ.setdefault("PYTHONUNBUFFERED", "1")

# ---------------------------------------------------------------------------
# Lazy imports -- wrapped so we get a clear error message if missing
# ---------------------------------------------------------------------------

from fedprotrack.baselines.runners import (
    MethodResult,
    run_apfl_full,
    run_atp_full,
    run_cfl_full,
    run_compressed_fedavg_full,
    run_fedccfa_full,
    run_feddrift_full,
    run_fedem_full,
    run_fedproto_full,
    run_fedprox_full,
    run_fedrc_full,
    run_fesem_full,
    run_flash_full,
    run_flux_full,
    run_flux_prior_full,
    run_ifca_full,
    run_pfedme_full,
    run_tracked_summary_full,
)
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_local_only,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import (
    canonical_method_name,
    dedupe_method_names,
    identity_metrics_valid,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_GROUPS: dict[str, list[str]] = {
    "core": ["FedProTrack", "FedAvg", "Oracle", "IFCA"],
    "drift": ["FedDrift", "Flash", "FedCCFA"],
    "cluster": ["CFL", "FedRC", "FedEM", "FeSEM"],
    "pfl": ["FedProx", "pFedMe", "APFL", "ATP"],
    "extra": ["TrackedSummary", "FLUX", "FLUX-prior", "CompressedFedAvg", "LocalOnly"],
    "quick": ["FedProTrack", "FedAvg", "Oracle", "IFCA", "FedRC"],
}
METHOD_GROUPS["all"] = (
    METHOD_GROUPS["core"]
    + METHOD_GROUPS["drift"]
    + METHOD_GROUPS["cluster"]
    + METHOD_GROUPS["pfl"]
    + METHOD_GROUPS["extra"]
)

PRESETS: dict[str, dict[str, object]] = {
    "small": {
        "K": 20,
        "T": 100,
        "participation": 1.0,
        "n_seeds": 3,
        "seeds": "42,43,44",
    },
    "medium": {
        "K": 20,
        "T": 200,
        "participation": 1.0,
        "n_seeds": 5,
        "seeds": "42,43,44,45,46",
    },
    "large": {
        "K": 100,
        "T": 200,
        "participation": 0.2,
        "n_seeds": 5,
        "seeds": "42,43,44,45,46",
    },
}

FPT_MODE_NAMES: dict[str, str] = {
    "base": "FedProTrack-base",
    "auto": "FedProTrack-auto",
    "calibrated": "FedProTrack-calibrated",
    "hybrid": "FedProTrack-hybrid",
    "hybrid-proto": "FedProTrack-hybrid-proto",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
    """Convert a generic baseline result to an ExperimentLog.

    Parameters
    ----------
    method_name : str
        Human-readable method label.
    result : object
        A result object that may have ``to_experiment_log``,
        ``accuracy_matrix``, and ``predicted_concept_matrix`` attributes.
    ground_truth : np.ndarray
        Shape (K, T) ground-truth concept matrix.

    Returns
    -------
    ExperimentLog
    """
    total_bytes = getattr(result, "total_bytes", None)
    if total_bytes is not None and float(total_bytes) <= 0.0:
        total_bytes = None

    if hasattr(result, "to_experiment_log"):
        try:
            return result.to_experiment_log(ground_truth)
        except TypeError:
            return result.to_experiment_log()

    return ExperimentLog(
        ground_truth=ground_truth,
        predicted=np.asarray(
            getattr(result, "predicted_concept_matrix"), dtype=np.int32
        ),
        accuracy_curve=np.asarray(
            getattr(result, "accuracy_matrix"), dtype=np.float64
        ),
        total_bytes=total_bytes,
        method_name=method_name,
    )


def _resolve_methods(methods_arg: str) -> list[str]:
    """Expand group names and comma-separated method lists.

    Parameters
    ----------
    methods_arg : str
        Comma-separated list of method names or group names.

    Returns
    -------
    list[str]
        Deduplicated ordered list of method names.
    """
    tokens = [tok.strip() for tok in methods_arg.split(",") if tok.strip()]
    expanded: list[str] = []
    for tok in tokens:
        if tok in METHOD_GROUPS:
            expanded.extend(METHOD_GROUPS[tok])
        else:
            expanded.append(tok)
    # Deduplicate preserving order
    seen: set[str] = set()
    result: list[str] = []
    for name in expanded:
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _select_active_clients(
    K: int,
    participation: float,
    t: int,
    seed: int,
) -> list[int]:
    """Deterministically sample active clients for round *t*.

    Parameters
    ----------
    K : int
        Total number of clients.
    participation : float
        Fraction of clients active per round (1.0 = full participation).
    t : int
        Current round index.
    seed : int
        Base random seed.

    Returns
    -------
    list[int]
        Sorted list of active client indices.
    """
    if participation >= 1.0:
        return list(range(K))
    n_active = max(1, math.ceil(K * participation))
    rng = np.random.RandomState(seed + t * 7919)
    active = sorted(rng.choice(K, size=n_active, replace=False).tolist())
    return active


# ---------------------------------------------------------------------------
# Method builder
# ---------------------------------------------------------------------------

def _build_methods(
    dataset,
    exp_cfg: ExperimentConfig,
    *,
    federation_every: int,
    fpt_lr: float,
    fpt_epochs: int,
    fpt_mode: str,
    n_epochs: int,
    lr: float,
    seed: int,
):
    """Build a dict of method-name -> callable returning a result.

    Parameters
    ----------
    dataset : DriftDataset
        The loaded dataset to run on.
    exp_cfg : ExperimentConfig
        Experiment config (wraps generator config).
    federation_every : int
        Federation cadence.
    fpt_lr : float
        Learning rate for FedProTrack.
    fpt_epochs : int
        Local epochs for FedProTrack.
    fpt_mode : str
        FedProTrack mode (base / calibrated / hybrid / ...).
    n_epochs : int
        Local epochs matched across all baselines.
    lr : float
        Learning rate matched across all baselines.
    seed : int
        Run seed.

    Returns
    -------
    dict[str, callable]
    """
    fpt_kwargs: dict[str, object] = {
        "auto_scale": fpt_mode == "auto",
        "similarity_calibration": fpt_mode in {
            "calibrated", "hybrid", "hybrid-proto",
        },
        "model_signature_weight": (
            0.55 if fpt_mode in {"hybrid", "hybrid-proto"} else 0.0
        ),
        "model_signature_dim": 8,
        "update_ot_weight": 0.0,
        "update_ot_dim": 4,
        "labelwise_proto_weight": 0.0,
        "labelwise_proto_dim": 4,
        "prototype_alignment_mix": (
            0.25 if fpt_mode == "hybrid-proto" else 0.0
        ),
        "prototype_alignment_early_rounds": 0,
        "prototype_alignment_early_mix": 0.0,
        "prototype_prealign_early_rounds": 0,
        "prototype_prealign_early_mix": 0.0,
        "prototype_subgroup_early_rounds": 0,
        "prototype_subgroup_early_mix": 0.0,
        "prototype_subgroup_min_clients": 3,
        "prototype_subgroup_similarity_gate": 0.8,
    }

    fpt_name = FPT_MODE_NAMES.get(fpt_mode, f"FedProTrack-{fpt_mode}")
    n_concepts = int(dataset.concept_matrix.max()) + 1

    methods: dict[str, object] = {
        fpt_name: lambda: FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0,
                kappa=0.7,
                novelty_threshold=0.25,
                loss_novelty_threshold=0.15,
                sticky_dampening=1.5,
                sticky_posterior_gate=0.35,
                merge_threshold=0.85,
                min_count=5.0,
                max_concepts=max(6, n_concepts + 3),
                merge_every=2,
                shrink_every=6,
            ),
            federation_every=federation_every,
            detector_name="ADWIN",
            seed=seed,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            **fpt_kwargs,
        ).run(dataset),
        "LocalOnly": lambda: run_local_only(exp_cfg, dataset=dataset),
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        ),
        "Oracle": lambda: run_oracle_baseline(exp_cfg, dataset=dataset),
        "FedProto": lambda: run_fedproto_full(
            dataset, federation_every=federation_every,
        ),
        "IFCA": lambda: run_ifca_full(
            dataset,
            federation_every=federation_every,
            n_clusters=max(4, n_concepts),
            lr=lr,
            n_epochs=n_epochs,
        ),
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every,
        ),
        "FeSEM": lambda: run_fesem_full(
            dataset, federation_every=federation_every,
        ),
        "FedRC": lambda: run_fedrc_full(
            dataset, federation_every=federation_every,
        ),
        "FedEM": lambda: run_fedem_full(
            dataset, federation_every=federation_every,
        ),
        "FedCCFA": lambda: run_fedccfa_full(
            dataset, federation_every=federation_every,
        ),
        "FedDrift": lambda: run_feddrift_full(
            dataset, federation_every=federation_every,
        ),
        "Flash": lambda: run_flash_full(
            dataset, federation_every=federation_every,
        ),
        "TrackedSummary": lambda: run_tracked_summary_full(
            dataset, federation_every=federation_every,
        ),
        "pFedMe": lambda: run_pfedme_full(
            dataset, federation_every=federation_every,
        ),
        "APFL": lambda: run_apfl_full(
            dataset, federation_every=federation_every,
        ),
        "ATP": lambda: run_atp_full(
            dataset, federation_every=federation_every,
        ),
        "FLUX": lambda: run_flux_full(
            dataset, federation_every=federation_every,
        ),
        "FLUX-prior": lambda: run_flux_prior_full(
            dataset, federation_every=federation_every,
        ),
        "CompressedFedAvg": lambda: run_compressed_fedavg_full(
            dataset, federation_every=federation_every,
        ),
        "FedProx": lambda: run_fedprox_full(
            dataset, federation_every=federation_every,
        ),
    }

    # Map canonical "FedProTrack" to the labelled variant
    if "FedProTrack" not in methods:
        methods["FedProTrack"] = methods[fpt_name]

    return methods, fpt_name


# ---------------------------------------------------------------------------
# Single-seed runner
# ---------------------------------------------------------------------------

def _run_single_seed(
    seed: int,
    method_names: list[str],
    *,
    K: int,
    T: int,
    n_samples: int,
    rho: float,
    alpha: float,
    delta: float,
    n_features: int,
    samples_per_coarse_class: int,
    batch_size: int,
    n_workers: int,
    data_root: str,
    feature_cache_dir: str,
    feature_seed: int,
    federation_every: int,
    lr: float,
    n_epochs: int,
    fpt_lr: float,
    fpt_epochs: int,
    fpt_mode: str,
    fpt_name: str,
    participation: float,
) -> list[dict]:
    """Run all methods for one seed and return result rows.

    Parameters
    ----------
    seed : int
        Random seed for this run.
    method_names : list[str]
        Ordered list of method names to evaluate.
    K, T, ... : various
        Dataset and training hyperparameters.

    Returns
    -------
    list[dict]
        One dict per method with metrics and metadata.
    """
    os.environ.setdefault("FEDPROTRACK_GPU_THRESHOLD", "0")

    dataset_cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=T,
        n_samples=n_samples,
        rho=rho,
        alpha=alpha,
        delta=delta,
        n_features=n_features,
        samples_per_coarse_class=samples_per_coarse_class,
        batch_size=batch_size,
        n_workers=n_workers,
        data_root=data_root,
        feature_cache_dir=feature_cache_dir,
        feature_seed=feature_seed,
        seed=seed,
    )

    dataset = generate_cifar100_recurrence_dataset(dataset_cfg)
    ground_truth = dataset.concept_matrix
    n_concepts = int(ground_truth.max()) + 1

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=federation_every,
    )

    methods, _ = _build_methods(
        dataset,
        exp_cfg,
        federation_every=federation_every,
        fpt_lr=fpt_lr,
        fpt_epochs=fpt_epochs,
        fpt_mode=fpt_mode,
        n_epochs=n_epochs,
        lr=lr,
        seed=seed,
    )

    rows: list[dict] = []
    for method_name in method_names:
        # Resolve the display name -> callable
        lookup_name = method_name
        if method_name == "FedProTrack":
            lookup_name = fpt_name
        if lookup_name not in methods and method_name in methods:
            lookup_name = method_name

        if lookup_name not in methods:
            print(
                f"  [SKIP] {method_name}: not available in method registry",
                flush=True,
            )
            continue

        t0 = time.time()
        try:
            result = methods[lookup_name]()
            canonical = canonical_method_name(
                "FedProTrack" if method_name.startswith("FedProTrack") else method_name
            )
            log = _make_log(method_name, result, ground_truth)
            metrics = compute_all_metrics(
                log,
                identity_capable=identity_metrics_valid(canonical),
            )

            total_bytes = float(getattr(result, "total_bytes", 0.0) or 0.0)

            # Build per-round accuracy curve for CSV output
            acc_matrix = np.asarray(
                getattr(result, "accuracy_matrix", log.accuracy_curve),
                dtype=np.float64,
            )
            mean_curve = acc_matrix.mean(axis=0)

            row = {
                "method": method_name,
                "seed": seed,
                "K": K,
                "T": T,
                "participation": participation,
                "n_epochs": n_epochs,
                "final_accuracy": metrics.final_accuracy,
                "accuracy_auc": metrics.accuracy_auc,
                "concept_re_id_accuracy": metrics.concept_re_id_accuracy,
                "assignment_switch_rate": metrics.assignment_switch_rate,
                "avg_clients_per_concept": metrics.avg_clients_per_concept,
                "singleton_group_ratio": metrics.singleton_group_ratio,
                "memory_reuse_rate": metrics.memory_reuse_rate,
                "routing_consistency": metrics.routing_consistency,
                "wrong_memory_reuse_rate": metrics.wrong_memory_reuse_rate,
                "assignment_entropy": metrics.assignment_entropy,
                "total_bytes": total_bytes,
                "wall_clock_s": time.time() - t0,
                "n_concepts": n_concepts,
                "status": "ok",
                "mean_accuracy_curve": mean_curve.tolist(),
            }
            rows.append(row)

            reid_str = (
                f"{metrics.concept_re_id_accuracy:.4f}"
                if metrics.concept_re_id_accuracy is not None
                else "--"
            )
            print(
                f"  {method_name:24s} seed={seed} "
                f"acc={metrics.final_accuracy:.4f} "
                f"auc={metrics.accuracy_auc:.4f} "
                f"reid={reid_str} "
                f"bytes={total_bytes:.0f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

        except Exception as exc:
            elapsed = time.time() - t0
            rows.append({
                "method": method_name,
                "seed": seed,
                "K": K,
                "T": T,
                "participation": participation,
                "n_epochs": n_epochs,
                "final_accuracy": None,
                "accuracy_auc": None,
                "concept_re_id_accuracy": None,
                "assignment_switch_rate": None,
                "avg_clients_per_concept": None,
                "singleton_group_ratio": None,
                "memory_reuse_rate": None,
                "routing_consistency": None,
                "wrong_memory_reuse_rate": None,
                "assignment_entropy": None,
                "total_bytes": None,
                "wall_clock_s": elapsed,
                "n_concepts": None,
                "status": "failed",
                "mean_accuracy_curve": [],
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(
                f"  {method_name:24s} seed={seed} FAILED: "
                f"{type(exc).__name__}: {exc} ({elapsed:.1f}s)",
                flush=True,
            )

    return rows


# ---------------------------------------------------------------------------
# Aggregation and output
# ---------------------------------------------------------------------------

_RAW_CSV_FIELDS = [
    "method",
    "seed",
    "K",
    "T",
    "participation",
    "n_epochs",
    "final_accuracy",
    "accuracy_auc",
    "concept_re_id_accuracy",
    "assignment_switch_rate",
    "avg_clients_per_concept",
    "singleton_group_ratio",
    "memory_reuse_rate",
    "routing_consistency",
    "wrong_memory_reuse_rate",
    "assignment_entropy",
    "total_bytes",
    "wall_clock_s",
    "n_concepts",
    "status",
]

_SUMMARY_CSV_FIELDS = [
    "method",
    "n_runs",
    "mean_acc",
    "std_acc",
    "mean_auc",
    "std_auc",
    "mean_reid",
    "std_reid",
    "mean_total_bytes",
    "mean_wall_clock_s",
]


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    """Write per-seed raw results CSV.

    Parameters
    ----------
    rows : list[dict]
    path : Path
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_RAW_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _RAW_CSV_FIELDS})


def _aggregate(rows: list[dict]) -> dict[str, dict[str, object]]:
    """Aggregate per-seed rows into per-method summary statistics.

    Parameters
    ----------
    rows : list[dict]
        Raw per-seed result rows.

    Returns
    -------
    dict[str, dict]
        Keyed by method name, values are summary statistics.
    """
    summary: dict[str, dict[str, object]] = {}
    ok_rows = [r for r in rows if r.get("status") == "ok"]

    for method in sorted({r["method"] for r in ok_rows}):
        method_rows = [r for r in ok_rows if r["method"] == method]
        n_runs = len(method_rows)

        def _safe_mean(key: str) -> float | None:
            vals = [float(r[key]) for r in method_rows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        def _safe_std(key: str) -> float | None:
            vals = [float(r[key]) for r in method_rows if r.get(key) is not None]
            return float(np.std(vals)) if len(vals) > 1 else None

        entry: dict[str, object] = {
            "n_runs": n_runs,
            "mean_acc": _safe_mean("final_accuracy"),
            "std_acc": _safe_std("final_accuracy"),
            "mean_auc": _safe_mean("accuracy_auc"),
            "std_auc": _safe_std("accuracy_auc"),
            "mean_reid": _safe_mean("concept_re_id_accuracy"),
            "std_reid": _safe_std("concept_re_id_accuracy"),
            "mean_total_bytes": _safe_mean("total_bytes"),
            "mean_wall_clock_s": _safe_mean("wall_clock_s"),
        }

        # Extra identity metrics for the JSON summary
        for extra_key in (
            "assignment_switch_rate",
            "avg_clients_per_concept",
            "singleton_group_ratio",
            "memory_reuse_rate",
            "routing_consistency",
            "wrong_memory_reuse_rate",
            "assignment_entropy",
        ):
            m = _safe_mean(extra_key)
            if m is not None:
                entry[f"mean_{extra_key}"] = m

        # Mean accuracy curves across seeds
        curves = [
            r["mean_accuracy_curve"]
            for r in method_rows
            if r.get("mean_accuracy_curve")
        ]
        if curves and all(len(c) == len(curves[0]) for c in curves):
            entry["mean_accuracy_curve"] = (
                np.mean(curves, axis=0).tolist()
            )

        summary[method] = entry

    return summary


def _write_summary_csv(summary: dict[str, dict], path: Path) -> None:
    """Write aggregated summary CSV.

    Parameters
    ----------
    summary : dict[str, dict]
    path : Path
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for method, entry in summary.items():
            row = {"method": method}
            row.update(entry)
            writer.writerow({k: row.get(k, "") for k in _SUMMARY_CSV_FIELDS})


def _print_summary_table(summary: dict[str, dict]) -> None:
    """Print a formatted summary table to stdout.

    Parameters
    ----------
    summary : dict[str, dict]
    """
    header = (
        f"{'Method':24s} {'Acc':>12s} {'AUC':>12s} "
        f"{'Re-ID':>12s} {'Bytes':>14s} {'Time(s)':>10s}"
    )
    print("\n" + "=" * len(header), flush=True)
    print("BENCHMARK SUMMARY", flush=True)
    print("=" * len(header), flush=True)
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for method, entry in summary.items():
        acc = entry.get("mean_acc")
        acc_std = entry.get("std_acc")
        auc = entry.get("mean_auc")
        auc_std = entry.get("std_auc")
        reid = entry.get("mean_reid")
        reid_std = entry.get("std_reid")
        total_b = entry.get("mean_total_bytes")
        wall = entry.get("mean_wall_clock_s")

        def _fmt(val, std, width=12):
            if val is None:
                return "--".rjust(width)
            if std is not None:
                return f"{val:.4f}+/-{std:.4f}".rjust(width)
            return f"{val:.4f}".rjust(width)

        acc_str = _fmt(acc, acc_std)
        auc_str = _fmt(auc, auc_std)
        reid_str = _fmt(reid, reid_std)
        bytes_str = f"{total_b:.0f}".rjust(14) if total_b is not None else "--".rjust(14)
        wall_str = f"{wall:.1f}".rjust(10) if wall is not None else "--".rjust(10)

        print(
            f"{method:24s} {acc_str} {auc_str} {reid_str} {bytes_str} {wall_str}",
            flush=True,
        )

    print("=" * len(header), flush=True)


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description=(
            "NeurIPS-scale CIFAR-100 federated concept drift benchmark. "
            "Runs multiple methods across seeds and reports aggregated "
            "accuracy, concept re-identification, and communication cost."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Presets:\n"
            "  small   K=20,  T=100, participation=1.0, 3 seeds\n"
            "  medium  K=20,  T=200, participation=1.0, 5 seeds\n"
            "  large   K=100, T=200, participation=0.2, 5 seeds\n"
            "  full    run medium + large sequentially\n"
        ),
    )

    # -- Scale parameters --
    parser.add_argument("--K", type=int, default=20, help="Number of clients (default: 20)")
    parser.add_argument("--T", type=int, default=200, help="Number of federation rounds (default: 200)")
    parser.add_argument(
        "--n-epochs", type=int, default=5,
        help="Local training epochs, matched across all methods (default: 5)",
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Local learning rate (default: 0.05)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument(
        "--participation", type=float, default=1.0,
        help="Fraction of clients active per round, 1.0 = full (default: 1.0)",
    )

    # -- Seeds --
    parser.add_argument("--n-seeds", type=int, default=3, help="Number of seeds (default: 3)")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seed list (default: 42,43,44)")

    # -- Dataset --
    parser.add_argument("--rho", type=float, default=3.0, help="Recurrence frequency (default: 3.0)")
    parser.add_argument("--alpha", type=float, default=0.75, help="Asynchrony level (default: 0.75)")
    parser.add_argument("--delta", type=float, default=0.85, help="Concept separability (default: 0.85)")
    parser.add_argument("--n-samples", type=int, default=400, help="Samples per (client, round) (default: 400)")
    parser.add_argument("--n-features", type=int, default=128, help="PCA feature dimension (default: 128)")
    parser.add_argument(
        "--samples-per-coarse-class", type=int, default=120,
        help="Source images per coarse class (default: 120)",
    )
    parser.add_argument(
        "--drift-type", default="recurrent",
        choices=["sudden", "incremental", "recurrent"],
        help="Drift type (default: recurrent)",
    )

    # -- Methods --
    parser.add_argument(
        "--methods", default="all",
        help=(
            "Comma-separated method names or group names: "
            "core, drift, cluster, pfl, extra, quick, all (default: all)"
        ),
    )
    parser.add_argument(
        "--fpt-mode", default="calibrated",
        choices=["base", "auto", "calibrated", "hybrid", "hybrid-proto"],
        help="FedProTrack routing mode (default: calibrated)",
    )
    parser.add_argument("--fpt-lr", type=float, default=None, help="FedProTrack lr (default: same as --lr)")
    parser.add_argument("--fpt-epochs", type=int, default=None, help="FedProTrack epochs (default: same as --n-epochs)")

    # -- Federation --
    parser.add_argument("--federation-every", type=int, default=1, help="Federation cadence (default: 1)")

    # -- Infra --
    parser.add_argument("--data-root", default=".cifar100_cache", help="CIFAR-100 download cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache", help="Feature cache directory")
    parser.add_argument("--feature-seed", type=int, default=2718, help="Feature extraction seed")
    parser.add_argument("--n-workers", type=int, default=0, help="DataLoader workers (default: 0)")
    parser.add_argument("--out-dir", default="results_neurips_benchmark", help="Output directory")

    # -- Presets --
    parser.add_argument(
        "--preset", default=None,
        choices=["small", "medium", "large", "full"],
        help="Predefined configuration preset (overrides K/T/seeds/participation)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick development mode: K=10, T=40, 1 seed, core methods only",
    )

    return parser.parse_args()


def _apply_drift_type(args: argparse.Namespace) -> None:
    """Adjust rho based on the requested drift type.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args (modified in place).
    """
    if args.drift_type == "sudden":
        # Single concept switch at T/2 => 2 concepts => rho = T/2
        args.rho = float(args.T) / 2.0
        args.alpha = 0.0  # synchronous for sudden drift
    elif args.drift_type == "incremental":
        # Gradual transition => moderate recurrence, high alpha for async
        args.rho = float(args.T) / 4.0
        args.alpha = 0.9
    # "recurrent" keeps user-specified rho and alpha


def _run_preset(preset_name: str, args: argparse.Namespace) -> list[dict]:
    """Run a single preset configuration and return all raw rows.

    Parameters
    ----------
    preset_name : str
        One of "small", "medium", "large".
    args : argparse.Namespace
        Parsed CLI arguments (used for non-preset fields).

    Returns
    -------
    list[dict]
        All per-seed result rows from this preset run.
    """
    preset = PRESETS[preset_name]
    K = int(preset["K"])
    T = int(preset["T"])
    participation = float(preset["participation"])
    seeds = [int(s.strip()) for s in str(preset["seeds"]).split(",") if s.strip()]

    fpt_lr = args.fpt_lr if args.fpt_lr is not None else args.lr
    fpt_epochs = args.fpt_epochs if args.fpt_epochs is not None else args.n_epochs
    fpt_name = FPT_MODE_NAMES.get(args.fpt_mode, f"FedProTrack-{args.fpt_mode}")
    method_names = _resolve_methods(args.methods)
    method_names = dedupe_method_names(method_names)

    print(
        f"\n{'='*60}\n"
        f"Preset: {preset_name} | K={K}, T={T}, "
        f"participation={participation}, seeds={seeds}\n"
        f"Methods: {method_names}\n"
        f"{'='*60}",
        flush=True,
    )

    # Warm feature cache once
    warm_cfg = CIFAR100RecurrenceConfig(
        K=K,
        T=T,
        n_samples=args.n_samples,
        rho=args.rho,
        alpha=args.alpha,
        delta=args.delta,
        n_features=args.n_features,
        samples_per_coarse_class=args.samples_per_coarse_class,
        batch_size=args.batch_size,
        n_workers=args.n_workers,
        data_root=args.data_root,
        feature_cache_dir=args.feature_cache_dir,
        feature_seed=args.feature_seed,
        seed=seeds[0],
    )
    print("Warming feature cache...", flush=True)
    prepare_cifar100_recurrence_feature_cache(warm_cfg)

    all_rows: list[dict] = []
    for seed in seeds:
        print(f"\n--- Seed {seed} ---", flush=True)
        seed_rows = _run_single_seed(
            seed=seed,
            method_names=method_names,
            K=K,
            T=T,
            n_samples=args.n_samples,
            rho=args.rho,
            alpha=args.alpha,
            delta=args.delta,
            n_features=args.n_features,
            samples_per_coarse_class=args.samples_per_coarse_class,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            data_root=args.data_root,
            feature_cache_dir=args.feature_cache_dir,
            feature_seed=args.feature_seed,
            federation_every=args.federation_every,
            lr=args.lr,
            n_epochs=args.n_epochs,
            fpt_lr=fpt_lr,
            fpt_epochs=fpt_epochs,
            fpt_mode=args.fpt_mode,
            fpt_name=fpt_name,
            participation=participation,
        )
        all_rows.extend(seed_rows)

    return all_rows


def main() -> None:
    """Entry point for the NeurIPS benchmark script."""
    args = _parse_args()

    # Apply drift type adjustments
    _apply_drift_type(args)

    # Apply quick mode
    if args.quick:
        args.K = 10
        args.T = 40
        args.seeds = "42"
        args.n_seeds = 1
        args.methods = "quick"

    # Ensure fpt_lr and fpt_epochs default to matched values
    if args.fpt_lr is None:
        args.fpt_lr = args.lr
    if args.fpt_epochs is None:
        args.fpt_epochs = args.n_epochs

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.time()

    # Handle 'full' preset: run medium + large sequentially
    if args.preset == "full":
        all_rows: list[dict] = []
        for sub_preset in ("medium", "large"):
            rows = _run_preset(sub_preset, args)
            all_rows.extend(rows)
            # Write intermediate results
            sub_dir = out_dir / sub_preset
            sub_dir.mkdir(parents=True, exist_ok=True)
            _write_raw_csv(rows, sub_dir / "raw_results.csv")
            sub_summary = _aggregate(rows)
            with open(sub_dir / "summary.json", "w", encoding="utf-8") as f:
                json.dump(sub_summary, f, indent=2, default=str)
            _write_summary_csv(sub_summary, sub_dir / "summary.csv")
            _print_summary_table(sub_summary)

        # Write combined results
        _write_raw_csv(all_rows, out_dir / "raw_results_all.csv")
        combined_summary = _aggregate(all_rows)
        with open(out_dir / "summary_all.json", "w", encoding="utf-8") as f:
            json.dump(combined_summary, f, indent=2, default=str)

    elif args.preset is not None:
        # Single named preset
        all_rows = _run_preset(args.preset, args)
        _write_raw_csv(all_rows, out_dir / "raw_results.csv")
        summary = _aggregate(all_rows)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        _write_summary_csv(summary, out_dir / "summary.csv")
        _print_summary_table(summary)

    else:
        # Free-form arguments (no preset)
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        # Trim seeds to n_seeds
        seeds = seeds[: args.n_seeds]

        method_names = _resolve_methods(args.methods)
        method_names = dedupe_method_names(method_names)
        fpt_name = FPT_MODE_NAMES.get(args.fpt_mode, f"FedProTrack-{args.fpt_mode}")

        print(
            f"\nNeurIPS Benchmark: K={args.K}, T={args.T}, "
            f"participation={args.participation}, "
            f"n_epochs={args.n_epochs}, lr={args.lr}, "
            f"drift_type={args.drift_type}, "
            f"fpt_mode={args.fpt_mode}\n"
            f"Seeds: {seeds}\n"
            f"Methods ({len(method_names)}): {method_names}",
            flush=True,
        )

        # Warm feature cache
        warm_cfg = CIFAR100RecurrenceConfig(
            K=args.K,
            T=args.T,
            n_samples=args.n_samples,
            rho=args.rho,
            alpha=args.alpha,
            delta=args.delta,
            n_features=args.n_features,
            samples_per_coarse_class=args.samples_per_coarse_class,
            batch_size=args.batch_size,
            n_workers=args.n_workers,
            data_root=args.data_root,
            feature_cache_dir=args.feature_cache_dir,
            feature_seed=args.feature_seed,
            seed=seeds[0],
        )
        print("Warming feature cache...", flush=True)
        prepare_cifar100_recurrence_feature_cache(warm_cfg)

        all_rows = []
        for seed in seeds:
            print(f"\n--- Seed {seed} ---", flush=True)
            seed_rows = _run_single_seed(
                seed=seed,
                method_names=method_names,
                K=args.K,
                T=args.T,
                n_samples=args.n_samples,
                rho=args.rho,
                alpha=args.alpha,
                delta=args.delta,
                n_features=args.n_features,
                samples_per_coarse_class=args.samples_per_coarse_class,
                batch_size=args.batch_size,
                n_workers=args.n_workers,
                data_root=args.data_root,
                feature_cache_dir=args.feature_cache_dir,
                feature_seed=args.feature_seed,
                federation_every=args.federation_every,
                lr=args.lr,
                n_epochs=args.n_epochs,
                fpt_lr=args.fpt_lr,
                fpt_epochs=args.fpt_epochs,
                fpt_mode=args.fpt_mode,
                fpt_name=fpt_name,
                participation=args.participation,
            )
            all_rows.extend(seed_rows)

        _write_raw_csv(all_rows, out_dir / "raw_results.csv")
        summary = _aggregate(all_rows)
        with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        _write_summary_csv(summary, out_dir / "summary.csv")
        _print_summary_table(summary)

    # Save full config for reproducibility
    config_dump = {
        "K": args.K,
        "T": args.T,
        "n_epochs": args.n_epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "participation": args.participation,
        "rho": args.rho,
        "alpha": args.alpha,
        "delta": args.delta,
        "n_features": args.n_features,
        "samples_per_coarse_class": args.samples_per_coarse_class,
        "drift_type": args.drift_type,
        "fpt_mode": args.fpt_mode,
        "fpt_lr": args.fpt_lr,
        "fpt_epochs": args.fpt_epochs,
        "federation_every": args.federation_every,
        "preset": args.preset,
        "quick": args.quick,
        "seeds": args.seeds,
        "n_seeds": args.n_seeds,
        "methods": args.methods,
        "feature_seed": args.feature_seed,
        "n_samples": args.n_samples,
    }
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_dump, f, indent=2)

    total_elapsed = time.time() - overall_t0
    n_ok = sum(1 for r in all_rows if r.get("status") == "ok")
    n_fail = sum(1 for r in all_rows if r.get("status") == "failed")
    print(
        f"\nBenchmark complete: {n_ok} succeeded, {n_fail} failed, "
        f"total time {total_elapsed:.1f}s\n"
        f"Results: {out_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
