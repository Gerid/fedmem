"""Phase 4 comprehensive analysis suite for FedProTrack (NeurIPS 2026).

Stages (each can be skipped with --skip-<stage>):

  Stage 1: Re-run E5 synthetic grid, save per-setting CSV
  Stage 2: Re-run E6 Rotating MNIST grid, save per-setting CSV
  Stage 3: E5 conditional analysis (from CSV, no re-run needed after Stage 1)
  Stage 4: E6 stability analysis (from CSV, no re-run needed after Stage 2)
  Stage 5: Case studies — trajectory plots for 6 representative settings
  Stage 6: Component ablation on 18 anchor settings
  Stage 7: Hyperparameter robustness sweeps
  Stage 8: Statistical significance (expanded seeds on key settings)
  Stage 9: E4 byte breakdown + lightweight variants

Usage:
    PYTHONUNBUFFERED=1 conda run -n base python run_phase4_analysis.py \\
        --results-dir results_phase4 \\
        [--skip-stage1] [--skip-stage2] ...

If raw CSVs already exist (from a previous run), use --skip-stage1 --skip-stage2
to jump directly to analysis stages.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_local_only,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.baselines.runners import (
    run_compressed_fedavg_full,
    run_feddrift_full,
    run_fedproto_full,
    run_flash_full,
    run_ifca_full,
    run_tracked_summary_full,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog, MetricsResult
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner, FedProTrackResult
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.experiments.phase4_analysis import (
    AnchorSetting,
    COMPONENT_ABLATIONS,
    DEFAULT_ANCHORS,
    conditional_analysis_e5,
    e4_byte_breakdown,
    load_raw_csv,
    plot_ablation_table,
    plot_case_study,
    plot_hyperparam_robustness,
    stability_analysis_e6,
    statistical_significance,
)

# Real data imports (may fail if MNIST not available)
try:
    from fedprotrack.real_data.rotating_mnist import (
        RotatingMNISTConfig,
        generate_rotating_mnist_dataset,
    )
    HAS_MNIST = True
except ImportError:
    HAS_MNIST = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

ALL_E5_METHODS = [
    "FedProTrack", "LocalOnly", "FedAvg", "Oracle", "FedProto",
    "TrackedSummary", "Flash", "FedDrift", "IFCA", "CompressedFedAvg",
]

E6_METHODS = ["FedProTrack", "FedAvg", "IFCA", "FedProto"]


def _make_log(name: str, acc: np.ndarray, pred: np.ndarray,
              gt: np.ndarray, total_bytes: float = 0.0) -> ExperimentLog:
    return ExperimentLog(
        ground_truth=gt, predicted=pred, accuracy_curve=acc,
        total_bytes=total_bytes if total_bytes > 0 else None,
        method_name=name,
    )


def _metrics_to_row(
    mr: MetricsResult,
    method: str,
    gen_cfg: GeneratorConfig | None = None,
    seed: int = 0,
    extra: dict | None = None,
) -> dict:
    """Convert MetricsResult to a flat dict row for CSV."""
    row: dict = {}
    if gen_cfg is not None:
        row["generator_type"] = gen_cfg.generator_type
        row["rho"] = gen_cfg.rho
        row["alpha"] = gen_cfg.alpha
        row["delta"] = gen_cfg.delta
        row["K"] = gen_cfg.K
        row["T"] = gen_cfg.T
    if extra:
        row.update(extra)
    row["seed"] = seed
    row["method"] = method
    row["concept_re_id_accuracy"] = (
        round(float(mr.concept_re_id_accuracy), 6)
        if mr.concept_re_id_accuracy is not None else ""
    )
    row["assignment_entropy"] = (
        round(float(mr.assignment_entropy), 6)
        if mr.assignment_entropy is not None else ""
    )
    row["wrong_memory_reuse_rate"] = (
        round(float(mr.wrong_memory_reuse_rate), 6)
        if mr.wrong_memory_reuse_rate is not None else ""
    )
    row["worst_window_dip"] = (
        round(float(mr.worst_window_dip), 6)
        if mr.worst_window_dip is not None else ""
    )
    row["worst_window_recovery"] = (
        mr.worst_window_recovery if mr.worst_window_recovery is not None else ""
    )
    row["budget_normalized_score"] = (
        round(float(mr.budget_normalized_score), 6)
        if mr.budget_normalized_score is not None else ""
    )
    row["final_accuracy"] = (
        round(float(mr.final_accuracy), 6)
        if mr.final_accuracy is not None else ""
    )
    row["accuracy_auc"] = (
        round(float(mr.accuracy_auc), 6)
        if mr.accuracy_auc is not None else ""
    )
    return row


RAW_CSV_HEADER = [
    "generator_type", "rho", "alpha", "delta", "K", "T", "seed", "method",
    "concept_re_id_accuracy", "assignment_entropy", "wrong_memory_reuse_rate",
    "worst_window_dip", "worst_window_recovery", "budget_normalized_score",
    "final_accuracy", "accuracy_auc",
]


def _write_raw_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_HEADER,
                                extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Stage 1: E5 Synthetic Grid (re-run to save per-setting data)
# ---------------------------------------------------------------------------

def run_e5_grid(
    generators: list[str],
    seeds: list[int],
    results_dir: Path,
    quick: bool = False,
) -> Path:
    """Run E5 synthetic grid and save per-setting CSV."""
    if quick:
        rho_values = [5.0]
        alpha_values = [0.0, 0.5, 1.0]
        delta_values = [0.3, 0.7]
        K, T, n_samples = 5, 8, 200
    else:
        rho_values = [2.0, 5.0, 10.0]
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        delta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        K, T, n_samples = 10, 20, 500

    csv_path = results_dir / "raw_e5.csv"
    rows: list[dict] = []

    grid = []
    for gen_type in generators:
        for rho in rho_values:
            for alpha in alpha_values:
                for delta in delta_values:
                    for seed in seeds:
                        cfg = GeneratorConfig(
                            K=K, T=T, n_samples=n_samples,
                            rho=rho, alpha=alpha, delta=delta,
                            generator_type=gen_type, seed=seed,
                        )
                        grid.append((cfg, seed))

    n_total = len(grid)
    print(f"\n[Stage 1] Running E5: {n_total} settings x {len(ALL_E5_METHODS)} methods")

    for i, (gen_cfg, seed) in enumerate(grid):
        tag = (f"{gen_cfg.generator_type}_r{gen_cfg.rho}_a{gen_cfg.alpha}"
               f"_d{gen_cfg.delta}_s{seed}")
        try:
            dataset = generate_drift_dataset(gen_cfg)
            gt = dataset.concept_matrix

            # FedProTrack
            fpt_runner = FedProTrackRunner(config=TwoPhaseConfig(), seed=seed)
            fpt_result = fpt_runner.run(dataset)
            fpt_log = fpt_result.to_experiment_log()
            mr = compute_all_metrics(fpt_log, identity_capable=True)
            rows.append(_metrics_to_row(mr, "FedProTrack", gen_cfg, seed))

            # LocalOnly
            exp_cfg = ExperimentConfig(generator_config=gen_cfg)
            lo = run_local_only(exp_cfg, dataset=dataset)
            lo_log = _make_log("LocalOnly", lo.accuracy_matrix,
                               lo.predicted_concept_matrix, gt)
            mr = compute_all_metrics(lo_log, identity_capable=False)
            rows.append(_metrics_to_row(mr, "LocalOnly", gen_cfg, seed))

            # FedAvg
            fa = run_fedavg_baseline(exp_cfg, dataset=dataset)
            fa_log = _make_log("FedAvg", fa.accuracy_matrix,
                               fa.predicted_concept_matrix, gt)
            mr = compute_all_metrics(fa_log, identity_capable=False)
            rows.append(_metrics_to_row(mr, "FedAvg", gen_cfg, seed))

            # Oracle
            oracle = run_oracle_baseline(exp_cfg, dataset=dataset)
            oracle_log = _make_log("Oracle", oracle.accuracy_matrix,
                                   oracle.predicted_concept_matrix, gt)
            mr = compute_all_metrics(oracle_log, identity_capable=True)
            rows.append(_metrics_to_row(mr, "Oracle", gen_cfg, seed))

            # FedProto
            fp = run_fedproto_full(dataset)
            mr = compute_all_metrics(fp.to_experiment_log(gt), identity_capable=False)
            rows.append(_metrics_to_row(mr, "FedProto", gen_cfg, seed))

            # TrackedSummary
            ts = run_tracked_summary_full(dataset)
            mr = compute_all_metrics(ts.to_experiment_log(gt), identity_capable=True)
            rows.append(_metrics_to_row(mr, "TrackedSummary", gen_cfg, seed))

            # Flash
            fl = run_flash_full(dataset)
            mr = compute_all_metrics(fl.to_experiment_log(gt), identity_capable=False)
            rows.append(_metrics_to_row(mr, "Flash", gen_cfg, seed))

            # FedDrift
            fd = run_feddrift_full(dataset)
            mr = compute_all_metrics(fd.to_experiment_log(gt), identity_capable=True)
            rows.append(_metrics_to_row(mr, "FedDrift", gen_cfg, seed))

            # IFCA
            ifca = run_ifca_full(dataset)
            mr = compute_all_metrics(ifca.to_experiment_log(gt), identity_capable=True)
            rows.append(_metrics_to_row(mr, "IFCA", gen_cfg, seed))

            # CompressedFedAvg
            cfed = run_compressed_fedavg_full(dataset)
            mr = compute_all_metrics(cfed.to_experiment_log(gt), identity_capable=False)
            rows.append(_metrics_to_row(mr, "CompressedFedAvg", gen_cfg, seed))

            reid_fpt = [r for r in rows if r["method"] == "FedProTrack"
                        and r.get("concept_re_id_accuracy")]
            reid_ifca = [r for r in rows if r["method"] == "IFCA"
                         and r.get("concept_re_id_accuracy")]
            if reid_fpt and reid_ifca:
                fpt_v = reid_fpt[-1]["concept_re_id_accuracy"]
                ifca_v = reid_ifca[-1]["concept_re_id_accuracy"]
                print(f"  [{i+1}/{n_total}] {tag} -> FPT={fpt_v} IFCA={ifca_v}",
                      flush=True)
            else:
                print(f"  [{i+1}/{n_total}] {tag} -> done", flush=True)

        except Exception as e:
            print(f"  [{i+1}/{n_total}] {tag} ERROR: {e}", flush=True)

        # Incremental save every 50 settings
        if (i + 1) % 50 == 0:
            _write_raw_csv(rows, csv_path)

    _write_raw_csv(rows, csv_path)
    print(f"  E5 raw data saved: {csv_path} ({len(rows)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Stage 2: E6 Rotating MNIST Grid
# ---------------------------------------------------------------------------

def run_e6_grid(
    seeds: list[int],
    results_dir: Path,
    quick: bool = False,
) -> Path:
    """Run E6 Rotating MNIST grid and save per-setting CSV."""
    if not HAS_MNIST:
        print("[Stage 2] SKIP: Rotating MNIST not available")
        return results_dir / "raw_e6.csv"

    if quick:
        rho_values = [5.0]
        alpha_values = [0.0, 0.5, 1.0]
        delta_values = [0.3, 0.7]
        K, T, n_samples, n_features = 5, 8, 200, 10
    else:
        rho_values = [2.0, 5.0, 10.0]
        alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        delta_values = [0.1, 0.3, 0.5, 0.7, 1.0]
        K, T, n_samples, n_features = 5, 10, 200, 20

    csv_path = results_dir / "raw_e6.csv"
    rows: list[dict] = []

    grid = []
    for rho in rho_values:
        for alpha in alpha_values:
            for delta in delta_values:
                for seed in seeds:
                    rm_cfg = RotatingMNISTConfig(
                        K=K, T=T, n_samples=n_samples,
                        rho=rho, alpha=alpha, delta=delta,
                        n_features=n_features, seed=seed,
                    )
                    # Build a GeneratorConfig-like for CSV output
                    gen_cfg = GeneratorConfig(
                        K=K, T=T, n_samples=n_samples,
                        rho=rho, alpha=alpha, delta=delta,
                        generator_type="rotating_mnist", seed=seed,
                    )
                    grid.append((rm_cfg, gen_cfg, seed))

    n_total = len(grid)
    print(f"\n[Stage 2] Running E6: {n_total} settings x {len(E6_METHODS)} methods")

    for i, (rm_cfg, gen_cfg, seed) in enumerate(grid):
        tag = f"rmnist_r{rm_cfg.rho}_a{rm_cfg.alpha}_d{rm_cfg.delta}_s{seed}"
        try:
            ds = generate_rotating_mnist_dataset(rm_cfg)
            gt = ds.concept_matrix

            # FedProTrack
            fpt_runner = FedProTrackRunner(config=TwoPhaseConfig(), seed=seed)
            fpt_result = fpt_runner.run(ds)
            mr = compute_all_metrics(fpt_result.to_experiment_log(),
                                     identity_capable=True)
            rows.append(_metrics_to_row(mr, "FedProTrack", gen_cfg, seed))

            # FedAvg
            exp_cfg = ExperimentConfig(generator_config=ds.config)
            fa = run_fedavg_baseline(exp_cfg, dataset=ds)
            fa_log = _make_log("FedAvg", fa.accuracy_matrix,
                               fa.predicted_concept_matrix, gt)
            mr = compute_all_metrics(fa_log, identity_capable=False)
            rows.append(_metrics_to_row(mr, "FedAvg", gen_cfg, seed))

            # IFCA
            ifca = run_ifca_full(ds)
            mr = compute_all_metrics(ifca.to_experiment_log(gt),
                                     identity_capable=True)
            rows.append(_metrics_to_row(mr, "IFCA", gen_cfg, seed))

            # FedProto
            fp = run_fedproto_full(ds)
            mr = compute_all_metrics(fp.to_experiment_log(gt),
                                     identity_capable=False)
            rows.append(_metrics_to_row(mr, "FedProto", gen_cfg, seed))

            print(f"  [{i+1}/{n_total}] {tag} -> done", flush=True)

        except Exception as e:
            print(f"  [{i+1}/{n_total}] {tag} ERROR: {e}", flush=True)

        if (i + 1) % 50 == 0:
            _write_raw_csv(rows, csv_path)

    _write_raw_csv(rows, csv_path)
    print(f"  E6 raw data saved: {csv_path} ({len(rows)} rows)")
    return csv_path


# ---------------------------------------------------------------------------
# Stage 5: Case Studies
# ---------------------------------------------------------------------------

CASE_STUDY_SETTINGS = [
    # (generator, rho, alpha, delta, label)
    ("sine", 10.0, 0.0, 0.5, "SINE easy: high rho, synchronous"),
    ("sine", 5.0, 0.5, 0.7, "SINE medium: mixed drift"),
    ("sine", 2.0, 1.0, 0.5, "SINE hard: low rho, fully async"),
    ("sea", 5.0, 0.25, 0.5, "SEA: moderate with some sync"),
    ("circle", 5.0, 0.5, 0.5, "CIRCLE: moderate baseline"),
    ("circle", 2.0, 0.75, 1.0, "CIRCLE: hard async, max delta"),
]


def run_case_studies(
    results_dir: Path,
    quick: bool = False,
    seed: int = 42,
) -> None:
    """Run 6 representative case studies and plot trajectories."""
    case_dir = results_dir / "case_studies"
    case_dir.mkdir(parents=True, exist_ok=True)

    K = 5 if quick else 10
    T = 8 if quick else 20
    n_samples = 200 if quick else 500

    for gen_type, rho, alpha, delta, label in CASE_STUDY_SETTINGS:
        print(f"  Case: {label}")
        gen_cfg = GeneratorConfig(
            K=K, T=T, n_samples=n_samples,
            rho=rho, alpha=alpha, delta=delta,
            generator_type=gen_type, seed=seed,
        )
        dataset = generate_drift_dataset(gen_cfg)
        gt = dataset.concept_matrix

        # Run FedProTrack
        fpt_runner = FedProTrackRunner(config=TwoPhaseConfig(), seed=seed)
        fpt_result = fpt_runner.run(dataset)
        fpt_log = fpt_result.to_experiment_log()
        fpt_metrics = compute_all_metrics(fpt_log, identity_capable=True)

        # Per-timestep metrics from MetricsResult
        per_t_entropy = None
        per_t_wrong = None
        if fpt_metrics.per_timestep_re_id is not None:
            per_t_wrong = 1.0 - fpt_metrics.per_timestep_re_id

        safe_label = label.replace(":", "").replace(" ", "_").replace(",", "")[:40]

        plot_case_study(
            fpt_result.accuracy_matrix,
            fpt_result.predicted_concept_matrix,
            gt,
            per_t_entropy,
            per_t_wrong,
            "FedProTrack",
            label,
            case_dir / f"case_{safe_label}_fpt.png",
        )

        # Also run IFCA for comparison
        ifca_result = run_ifca_full(dataset)
        ifca_log = ifca_result.to_experiment_log(gt)
        ifca_metrics = compute_all_metrics(ifca_log, identity_capable=True)

        ifca_per_t_wrong = None
        if ifca_metrics.per_timestep_re_id is not None:
            ifca_per_t_wrong = 1.0 - ifca_metrics.per_timestep_re_id

        plot_case_study(
            ifca_result.accuracy_matrix,
            ifca_result.predicted_concept_matrix,
            gt,
            None,
            ifca_per_t_wrong,
            "IFCA",
            label,
            case_dir / f"case_{safe_label}_ifca.png",
        )

    print(f"  Case studies saved to {case_dir}/")


# ---------------------------------------------------------------------------
# Stage 6: Component Ablation on Anchor Settings
# ---------------------------------------------------------------------------

def run_component_ablation(
    results_dir: Path,
    anchors: list[AnchorSetting] | None = None,
    quick: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """Run component ablation on anchor settings."""
    abl_dir = results_dir / "ablation"
    abl_dir.mkdir(parents=True, exist_ok=True)

    if anchors is None:
        anchors = DEFAULT_ANCHORS
    if seeds is None:
        seeds = [42, 123, 456]

    K = 5 if quick else 10
    T = 8 if quick else 20
    n_samples = 200 if quick else 500

    # results[anchor_label][variant_name] = {metric: mean_value}
    all_results: dict[str, dict[str, dict[str, float]]] = {}
    raw_rows: list[dict] = []

    n_total = len(anchors) * len(COMPONENT_ABLATIONS) * len(seeds)
    print(f"\n[Stage 6] Component ablation: {len(anchors)} anchors x "
          f"{len(COMPONENT_ABLATIONS)} variants x {len(seeds)} seeds = {n_total}")

    idx = 0
    for anchor in anchors:
        anchor_results: dict[str, dict[str, list[float]]] = {}

        for variant_name, overrides in COMPONENT_ABLATIONS.items():
            metric_lists: dict[str, list[float]] = {}

            for seed in seeds:
                idx += 1
                gen_cfg = GeneratorConfig(
                    K=K, T=T, n_samples=n_samples,
                    rho=anchor.rho, alpha=anchor.alpha, delta=anchor.delta,
                    generator_type=anchor.generator_type, seed=seed,
                )

                overrides = dict(overrides)
                phase_a_only = overrides.pop("_phase_a_only", False)

                cfg_kwargs = dict(overrides)
                two_cfg = TwoPhaseConfig(**cfg_kwargs)

                try:
                    dataset = generate_drift_dataset(gen_cfg)
                    runner = FedProTrackRunner(
                        config=two_cfg,
                        federation_every=999 if phase_a_only else 1,
                        seed=seed,
                    )
                    result = runner.run(dataset)
                    log = result.to_experiment_log()
                    mr = compute_all_metrics(log, identity_capable=True)

                    for m_name in ["concept_re_id_accuracy",
                                   "wrong_memory_reuse_rate",
                                   "final_accuracy", "assignment_entropy",
                                   "accuracy_auc"]:
                        val = getattr(mr, m_name, None)
                        if val is not None:
                            metric_lists.setdefault(m_name, []).append(float(val))

                    row = _metrics_to_row(mr, f"FPT_{variant_name}", gen_cfg, seed)
                    row["variant"] = variant_name
                    row["anchor"] = anchor.label
                    raw_rows.append(row)

                    print(f"  [{idx}/{n_total}] {anchor.label} | {variant_name} "
                          f"| s={seed} -> "
                          f"reid={mr.concept_re_id_accuracy:.3f}" if
                          mr.concept_re_id_accuracy is not None else
                          f"  [{idx}/{n_total}] {anchor.label} | {variant_name} "
                          f"| s={seed} -> done",
                          flush=True)

                except Exception as e:
                    print(f"  [{idx}/{n_total}] {anchor.label} | {variant_name} "
                          f"| s={seed} ERROR: {e}", flush=True)

            # Average across seeds
            anchor_results[variant_name] = {
                m: float(np.mean(v)) for m, v in metric_lists.items()
            }

        all_results[anchor.label] = anchor_results

    # Plot and save
    plot_ablation_table(all_results, abl_dir)

    # Save raw CSV
    header = RAW_CSV_HEADER + ["variant", "anchor"]
    with open(abl_dir / "ablation_raw.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        writer.writeheader()
        for r in raw_rows:
            writer.writerow(r)

    # Save aggregated JSON
    with open(abl_dir / "ablation_aggregated.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"  Ablation results saved to {abl_dir}/")


# ---------------------------------------------------------------------------
# Stage 7: Hyperparameter Robustness
# ---------------------------------------------------------------------------

def run_hyperparam_robustness(
    results_dir: Path,
    quick: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """Run local sweeps around optimal hyperparameter values."""
    rob_dir = results_dir / "robustness"
    rob_dir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [42, 123, 456]

    K = 5 if quick else 10
    T = 8 if quick else 20
    n_samples = 200 if quick else 500

    # Sweep definitions: param_name -> (values, default_idx)
    sweeps: dict[str, tuple[list[float], dict]] = {
        "loss_novelty_threshold": (
            [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.1],
            {},
        ),
        "sticky_dampening": (
            [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0],
            {},
        ),
        "omega": (
            [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
            {},
        ),
        "merge_threshold": (
            [0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
            {},
        ),
        "kappa": (
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            {},
        ),
    }

    # Test on 3 representative settings
    test_settings = [
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.5, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=10.0, alpha=0.0, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=2.0, alpha=0.75, delta=0.5,
                        generator_type="sine", seed=42),
    ]

    all_results: dict[str, list[tuple[float, dict[str, float]]]] = {}

    for param_name, (values, _) in sweeps.items():
        print(f"  Sweeping {param_name}: {values}")
        sweep_results: list[tuple[float, dict[str, float]]] = []

        for val in values:
            metric_lists: dict[str, list[float]] = {}

            for gen_cfg in test_settings:
                for seed in seeds:
                    cfg_kwargs = {param_name: val}
                    two_cfg = TwoPhaseConfig(**cfg_kwargs)

                    gc = GeneratorConfig(
                        K=gen_cfg.K, T=gen_cfg.T, n_samples=gen_cfg.n_samples,
                        rho=gen_cfg.rho, alpha=gen_cfg.alpha, delta=gen_cfg.delta,
                        generator_type=gen_cfg.generator_type, seed=seed,
                    )

                    try:
                        dataset = generate_drift_dataset(gc)
                        runner = FedProTrackRunner(config=two_cfg, seed=seed)
                        result = runner.run(dataset)
                        log = result.to_experiment_log()
                        mr = compute_all_metrics(log, identity_capable=True)

                        for m_name in ["concept_re_id_accuracy",
                                       "wrong_memory_reuse_rate",
                                       "final_accuracy"]:
                            v = getattr(mr, m_name, None)
                            if v is not None:
                                metric_lists.setdefault(m_name, []).append(float(v))
                    except Exception as e:
                        print(f"    {param_name}={val} s={seed}: {e}", flush=True)

            avg_metrics = {m: float(np.mean(v)) for m, v in metric_lists.items()}
            sweep_results.append((val, avg_metrics))
            print(f"    {param_name}={val}: "
                  f"reid={avg_metrics.get('concept_re_id_accuracy', 'N/A'):.4f}",
                  flush=True)

        all_results[param_name] = sweep_results

    plot_hyperparam_robustness(all_results, rob_dir)
    print(f"  Robustness results saved to {rob_dir}/")


# ---------------------------------------------------------------------------
# Stage 8: Statistical Significance (expanded seeds)
# ---------------------------------------------------------------------------

def run_significance_tests(
    results_dir: Path,
    quick: bool = False,
) -> None:
    """Run expanded-seed significance tests on key settings."""
    sig_dir = results_dir / "significance"
    sig_dir.mkdir(parents=True, exist_ok=True)

    if quick:
        seeds = list(range(42, 62))  # 20 seeds
        K, T, n_samples = 5, 8, 200
    else:
        seeds = list(range(42, 62))  # 20 seeds
        K, T, n_samples = 10, 20, 500

    # Key settings: E1 gate (SINE), hard case, E6-like
    key_settings = [
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.5, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=10.0, alpha=0.0, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=2.0, alpha=0.75, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.5, delta=0.5,
                        generator_type="sea", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.5, delta=0.5,
                        generator_type="circle", seed=42),
    ]

    rows: list[dict] = []
    n_total = len(key_settings) * len(seeds)
    print(f"\n[Stage 8] Significance: {len(key_settings)} settings x "
          f"{len(seeds)} seeds = {n_total}")

    idx = 0
    for base_cfg in key_settings:
        for seed in seeds:
            idx += 1
            gen_cfg = GeneratorConfig(
                K=base_cfg.K, T=base_cfg.T, n_samples=base_cfg.n_samples,
                rho=base_cfg.rho, alpha=base_cfg.alpha, delta=base_cfg.delta,
                generator_type=base_cfg.generator_type, seed=seed,
            )

            try:
                dataset = generate_drift_dataset(gen_cfg)
                gt = dataset.concept_matrix

                # FedProTrack
                fpt = FedProTrackRunner(config=TwoPhaseConfig(), seed=seed)
                fpt_res = fpt.run(dataset)
                mr = compute_all_metrics(fpt_res.to_experiment_log(),
                                         identity_capable=True)
                rows.append(_metrics_to_row(mr, "FedProTrack", gen_cfg, seed))

                # IFCA
                ifca = run_ifca_full(dataset)
                mr = compute_all_metrics(ifca.to_experiment_log(gt),
                                         identity_capable=True)
                rows.append(_metrics_to_row(mr, "IFCA", gen_cfg, seed))

                # FedProto
                fp = run_fedproto_full(dataset)
                mr = compute_all_metrics(fp.to_experiment_log(gt),
                                         identity_capable=False)
                rows.append(_metrics_to_row(mr, "FedProto", gen_cfg, seed))

                # FedAvg
                exp_cfg = ExperimentConfig(generator_config=gen_cfg)
                fa = run_fedavg_baseline(exp_cfg, dataset=dataset)
                fa_log = _make_log("FedAvg", fa.accuracy_matrix,
                                   fa.predicted_concept_matrix, gt)
                mr = compute_all_metrics(fa_log, identity_capable=False)
                rows.append(_metrics_to_row(mr, "FedAvg", gen_cfg, seed))

                print(f"  [{idx}/{n_total}] "
                      f"{gen_cfg.generator_type}_r{gen_cfg.rho}_a{gen_cfg.alpha}"
                      f"_d{gen_cfg.delta}_s{seed} -> done", flush=True)

            except Exception as e:
                print(f"  [{idx}/{n_total}] ERROR: {e}", flush=True)

    # Save raw and run tests
    csv_path = sig_dir / "significance_raw.csv"
    _write_raw_csv(rows, csv_path)

    # Run paired tests
    for metric in ["concept_re_id_accuracy", "final_accuracy"]:
        for competitor in ["IFCA", "FedProto", "FedAvg"]:
            result = statistical_significance(
                rows, sig_dir,
                method_a="FedProTrack", method_b=competitor,
                metric=metric,
            )
            p = result.get("p_value", "N/A")
            d = result.get("cohens_d", "N/A")
            print(f"  FedProTrack vs {competitor} ({metric}): "
                  f"p={p}, d={d}", flush=True)

    print(f"  Significance results saved to {sig_dir}/")


# ---------------------------------------------------------------------------
# Stage 9: E4 Byte Breakdown + Lightweight Variants
# ---------------------------------------------------------------------------

def run_e4_analysis(
    results_dir: Path,
    quick: bool = False,
    seeds: list[int] | None = None,
) -> None:
    """E4 byte breakdown and event-triggered / top-M variants."""
    e4_dir = results_dir / "e4_analysis"
    e4_dir.mkdir(parents=True, exist_ok=True)

    if seeds is None:
        seeds = [42, 123, 456]

    K = 5 if quick else 10
    T = 8 if quick else 20
    n_samples = 200 if quick else 500

    # Test on representative settings
    test_cfgs = [
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.5, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=0.0, delta=0.5,
                        generator_type="sine", seed=42),
        GeneratorConfig(K=K, T=T, n_samples=n_samples,
                        rho=5.0, alpha=1.0, delta=0.5,
                        generator_type="sine", seed=42),
    ]

    # --- Byte breakdown for each method variant ---
    print("\n[Stage 9] E4 byte breakdown + lightweight variants")

    # Collect per-variant stats
    variants = {
        "FedProTrack": {"event_triggered": False},
        "FedProTrack-ET": {"event_triggered": True},
    }

    byte_stats: dict[str, dict[str, float]] = {}
    accuracy_stats: dict[str, list[float]] = {}

    for variant_name, variant_kwargs in variants.items():
        total_phase_a = []
        total_phase_b = []
        total_bytes_list = []
        reids = []
        accs = []

        for gen_cfg in test_cfgs:
            for seed in seeds:
                gc = GeneratorConfig(
                    K=gen_cfg.K, T=gen_cfg.T, n_samples=gen_cfg.n_samples,
                    rho=gen_cfg.rho, alpha=gen_cfg.alpha, delta=gen_cfg.delta,
                    generator_type=gen_cfg.generator_type, seed=seed,
                )
                dataset = generate_drift_dataset(gc)
                runner = FedProTrackRunner(
                    config=TwoPhaseConfig(), seed=seed,
                    event_triggered=variant_kwargs.get("event_triggered", False),
                )
                result = runner.run(dataset)
                total_phase_a.append(result.phase_a_bytes)
                total_phase_b.append(result.phase_b_bytes)
                total_bytes_list.append(result.total_bytes)

                log = result.to_experiment_log()
                mr = compute_all_metrics(log, identity_capable=True)
                if mr.concept_re_id_accuracy is not None:
                    reids.append(mr.concept_re_id_accuracy)
                if mr.final_accuracy is not None:
                    accs.append(mr.final_accuracy)

        byte_stats[variant_name] = {
            "phase_a_bytes": float(np.mean(total_phase_a)),
            "phase_b_bytes": float(np.mean(total_phase_b)),
            "total_bytes": float(np.mean(total_bytes_list)),
        }
        accuracy_stats[variant_name] = reids

        print(f"  {variant_name}: "
              f"phase_a={np.mean(total_phase_a):.0f}, "
              f"phase_b={np.mean(total_phase_b):.0f}, "
              f"total={np.mean(total_bytes_list):.0f}, "
              f"reid={np.mean(reids):.4f}" if reids else
              f"  {variant_name}: no reid data",
              flush=True)

    # Also collect baselines for comparison
    for gen_cfg in test_cfgs[:1]:
        for seed in seeds[:1]:
            gc = GeneratorConfig(
                K=gen_cfg.K, T=gen_cfg.T, n_samples=gen_cfg.n_samples,
                rho=gen_cfg.rho, alpha=gen_cfg.alpha, delta=gen_cfg.delta,
                generator_type=gen_cfg.generator_type, seed=seed,
            )
            dataset = generate_drift_dataset(gc)

            # IFCA bytes
            ifca = run_ifca_full(dataset)
            byte_stats["IFCA"] = {
                "phase_a_bytes": 0.0,
                "phase_b_bytes": ifca.total_bytes,
                "total_bytes": ifca.total_bytes,
            }

            # FedProto bytes
            fp = run_fedproto_full(dataset)
            byte_stats["FedProto"] = {
                "phase_a_bytes": fp.total_bytes,
                "phase_b_bytes": 0.0,
                "total_bytes": fp.total_bytes,
            }

    e4_byte_breakdown(byte_stats, e4_dir)

    # Save detailed stats
    with open(e4_dir / "e4_detailed_stats.json", "w") as f:
        json.dump({
            "byte_stats": byte_stats,
            "accuracy_stats": {k: [round(v, 4) for v in vals]
                               for k, vals in accuracy_stats.items()},
        }, f, indent=2)

    print(f"  E4 analysis saved to {e4_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Analysis Suite")
    parser.add_argument("--results-dir", default="results_phase4",
                        help="Output directory")
    parser.add_argument("--generators", default="sine,sea,circle",
                        help="Comma-separated generators for E5")
    parser.add_argument("--seeds", default="42,123,456,789,1024",
                        help="Comma-separated seeds")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced grid for fast testing")

    # Skip flags for each stage
    parser.add_argument("--skip-stage1", action="store_true",
                        help="Skip E5 grid (use existing raw_e5.csv)")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="Skip E6 grid (use existing raw_e6.csv)")
    parser.add_argument("--skip-stage3", action="store_true",
                        help="Skip E5 conditional analysis")
    parser.add_argument("--skip-stage4", action="store_true",
                        help="Skip E6 stability analysis")
    parser.add_argument("--skip-stage5", action="store_true",
                        help="Skip case studies")
    parser.add_argument("--skip-stage6", action="store_true",
                        help="Skip component ablation")
    parser.add_argument("--skip-stage7", action="store_true",
                        help="Skip hyperparameter robustness")
    parser.add_argument("--skip-stage8", action="store_true",
                        help="Skip statistical significance")
    parser.add_argument("--skip-stage9", action="store_true",
                        help="Skip E4 analysis")

    # Only flags (run specific stages only)
    parser.add_argument("--only", default="",
                        help="Comma-separated stage numbers to run ONLY")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    generators = [g.strip() for g in args.generators.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Determine which stages to run
    if args.only:
        only_stages = {int(s.strip()) for s in args.only.split(",")}
        skip = {i for i in range(1, 10) if i not in only_stages}
    else:
        skip = set()
        for i in range(1, 10):
            if getattr(args, f"skip_stage{i}", False):
                skip.add(i)

    print(f"Results dir: {results_dir}")
    print(f"Generators: {generators}")
    print(f"Seeds: {seeds}")
    print(f"Quick: {args.quick}")
    print(f"Stages to run: {sorted(set(range(1,10)) - skip)}")

    t_start = time.time()

    # ===== Stage 1: E5 Grid =====
    e5_csv = results_dir / "raw_e5.csv"
    if 1 not in skip:
        e5_csv = run_e5_grid(generators, seeds, results_dir, args.quick)
    elif not e5_csv.exists():
        print(f"WARNING: {e5_csv} not found. Stage 3 will fail.")

    # ===== Stage 2: E6 Grid =====
    e6_csv = results_dir / "raw_e6.csv"
    if 2 not in skip:
        e6_csv = run_e6_grid(seeds, results_dir, args.quick)
    elif not e6_csv.exists():
        print(f"WARNING: {e6_csv} not found. Stage 4 will fail.")

    # ===== Stage 3: E5 Conditional Analysis =====
    if 3 not in skip and e5_csv.exists():
        print("\n[Stage 3] E5 conditional analysis...")
        e5_rows = load_raw_csv(e5_csv)
        cond_dir = results_dir / "conditional_e5"
        summary = conditional_analysis_e5(e5_rows, cond_dir)
        win_cond = summary.get("win_conditions", {})
        for comp, data in win_cond.items():
            n_wins = len(data.get("wins", []))
            n_losses = len(data.get("losses", []))
            print(f"  vs {comp}: {n_wins} wins, {n_losses} losses")
        print(f"  E5 conditional analysis saved to {cond_dir}/")

    # ===== Stage 4: E6 Stability Analysis =====
    if 4 not in skip and e6_csv.exists():
        print("\n[Stage 4] E6 stability analysis...")
        e6_rows = load_raw_csv(e6_csv)
        stab_dir = results_dir / "stability_e6"
        stability_analysis_e6(e6_rows, stab_dir)
        print(f"  E6 stability analysis saved to {stab_dir}/")

    # ===== Stage 5: Case Studies =====
    if 5 not in skip:
        print("\n[Stage 5] Case studies...")
        run_case_studies(results_dir, args.quick)

    # ===== Stage 6: Component Ablation =====
    if 6 not in skip:
        print("\n[Stage 6] Component ablation...")
        run_component_ablation(results_dir, quick=args.quick, seeds=seeds[:3])

    # ===== Stage 7: Hyperparameter Robustness =====
    if 7 not in skip:
        print("\n[Stage 7] Hyperparameter robustness...")
        run_hyperparam_robustness(results_dir, args.quick, seeds=seeds[:3])

    # ===== Stage 8: Statistical Significance =====
    if 8 not in skip:
        print("\n[Stage 8] Statistical significance...")
        run_significance_tests(results_dir, args.quick)

    # ===== Stage 9: E4 Analysis =====
    if 9 not in skip:
        print("\n[Stage 9] E4 byte breakdown...")
        run_e4_analysis(results_dir, args.quick, seeds=seeds[:3])

    # ===== Stage 3+4 extras: E5 + E6 significance from raw CSVs =====
    if 3 not in skip and 4 not in skip and e5_csv.exists() and e6_csv.exists():
        print("\n[Extra] Cross-dataset significance...")
        sig_dir = results_dir / "significance"
        sig_dir.mkdir(parents=True, exist_ok=True)

        e5_rows = load_raw_csv(e5_csv)
        for metric in ["concept_re_id_accuracy", "final_accuracy"]:
            for comp in ["IFCA", "FedProto"]:
                r = statistical_significance(
                    e5_rows, sig_dir,
                    method_a="FedProTrack", method_b=comp, metric=metric,
                )
                print(f"  E5 FPT vs {comp} ({metric}): "
                      f"p={r.get('p_value')}, d={r.get('cohens_d')}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Phase 4 analysis completed in {elapsed:.1f}s")
    print(f"All results saved to {results_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
