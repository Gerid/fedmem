"""Resume Phase 4 pipeline from where Stage 1 was interrupted.

Appends missing SEA/CIRCLE settings to raw_e5.csv, then runs stages 2-9.
Uses ProcessPoolExecutor for parallel Stage 1 execution (each worker forces
CPU + single-thread BLAS to avoid GPU contention).
"""
from __future__ import annotations

import csv
import os
import sys
import time
import threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

N_WORKERS = max(1, (os.cpu_count() or 4) - 4)

from fedprotrack.drift_generator import GeneratorConfig, generate_drift_dataset
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline, run_local_only, run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.baselines.runners import (
    run_compressed_fedavg_full, run_feddrift_full, run_fedproto_full,
    run_flash_full, run_ifca_full, run_tracked_summary_full,
)
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.experiments.phase4_analysis import (
    conditional_analysis_e5, load_raw_csv, stability_analysis_e6,
    statistical_significance,
)

# Re-use helpers from run_phase4_analysis
from run_phase4_analysis import (
    RAW_CSV_HEADER, _make_log, _metrics_to_row, _write_raw_csv,
    run_e6_grid, run_case_studies, run_component_ablation,
    run_hyperparam_robustness, run_significance_tests, run_e4_analysis,
)

RESULTS_DIR = Path("results_phase4")
SEEDS = [42, 123, 456, 789, 1024]
K, T, N_SAMPLES = 10, 20, 500
RHO_VALUES = [2.0, 5.0, 10.0]
ALPHA_VALUES = [0.0, 0.25, 0.5, 0.75, 1.0]
DELTA_VALUES = [0.1, 0.3, 0.5, 0.7, 1.0]


def load_existing_keys(csv_path: Path) -> set[tuple]:
    """Load (generator_type, rho, alpha, delta, seed) keys from existing CSV."""
    keys = set()
    if not csv_path.exists():
        return keys
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (
                row["generator_type"],
                float(row["rho"]),
                float(row["alpha"]),
                float(row["delta"]),
                int(row["seed"]),
            )
            keys.add(key)
    return keys


def _worker_init() -> None:
    """Per-worker initialiser: force CPU and single-thread BLAS."""
    os.environ["FEDPROTRACK_FORCE_CPU"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _worker_run_setting(args: tuple) -> tuple[str, list[dict]]:
    """Top-level picklable function executed in each worker process."""
    _worker_init()
    gen_cfg, seed, tag = args
    rows = run_single_e5_setting(gen_cfg, seed)
    return tag, rows


def run_single_e5_setting(gen_cfg: GeneratorConfig, seed: int) -> list[dict]:
    """Run all 10 methods on one E5 setting, return rows."""
    rows = []
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

    return rows


def resume_stage1():
    """Complete Stage 1 by running only missing settings (parallel)."""
    csv_path = RESULTS_DIR / "raw_e5.csv"
    existing_keys = load_existing_keys(csv_path)
    print(f"Existing settings in CSV: {len(existing_keys)}")

    # Build remaining grid
    remaining = []
    for gen_type in ["sine", "sea", "circle"]:
        for rho in RHO_VALUES:
            for alpha in ALPHA_VALUES:
                for delta in DELTA_VALUES:
                    for seed in SEEDS:
                        key = (gen_type, rho, alpha, delta, seed)
                        if key not in existing_keys:
                            cfg = GeneratorConfig(
                                K=K, T=T, n_samples=N_SAMPLES,
                                rho=rho, alpha=alpha, delta=delta,
                                generator_type=gen_type, seed=seed,
                            )
                            tag = f"{gen_type}_r{rho}_a{alpha}_d{delta}_s{seed}"
                            remaining.append((cfg, seed, tag))

    n_total = len(remaining)
    print(f"Remaining settings to run: {n_total}")
    print(f"Using {N_WORKERS} workers (cpu_count - 4)")

    if n_total == 0:
        print("Stage 1 already complete!")
        return

    # Lock for thread-safe CSV append
    csv_lock = threading.Lock()
    pending_rows: list[dict] = []
    completed = 0
    flush_every = 25

    def _flush_rows() -> None:
        nonlocal pending_rows
        if pending_rows:
            _append_rows(csv_path, pending_rows)
            pending_rows = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {
            pool.submit(_worker_run_setting, task): task
            for task in remaining
        }

        for future in as_completed(futures):
            task = futures[future]
            tag = task[2]
            completed += 1
            try:
                result_tag, rows = future.result()
                pending_rows.extend(rows)

                fpt_row = [r for r in rows if r["method"] == "FedProTrack"]
                ifca_row = [r for r in rows if r["method"] == "IFCA"]
                fpt_v = fpt_row[0]["concept_re_id_accuracy"] if fpt_row else "?"
                ifca_v = ifca_row[0]["concept_re_id_accuracy"] if ifca_row else "?"
                print(f"  [{completed}/{n_total}] {result_tag} -> FPT={fpt_v} IFCA={ifca_v}",
                      flush=True)
            except Exception as e:
                print(f"  [{completed}/{n_total}] {tag} ERROR: {e}", flush=True)

            # Flush every 25 completed settings
            if completed % flush_every == 0:
                _flush_rows()

    # Final flush
    _flush_rows()

    total_after = len(load_existing_keys(csv_path))
    print(f"Stage 1 complete: {total_after} unique settings in {csv_path}")


def _append_rows(csv_path: Path, rows: list[dict]) -> None:
    """Append rows to existing CSV."""
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_HEADER,
                                extrasaction="ignore")
        for r in rows:
            writer.writerow(r)


def main():
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: Complete remaining E5 settings
    print("=" * 60)
    print("RESUMING PHASE 4 PIPELINE")
    print("=" * 60)
    resume_stage1()

    # Stage 2: E6 Grid
    print("\n" + "=" * 60)
    print("STAGE 2: E6 Rotating MNIST")
    print("=" * 60)
    e6_csv = run_e6_grid(SEEDS, RESULTS_DIR, quick=False)

    # Stage 3: E5 Conditional Analysis
    e5_csv = RESULTS_DIR / "raw_e5.csv"
    if e5_csv.exists():
        print("\n[Stage 3] E5 conditional analysis...")
        e5_rows = load_raw_csv(e5_csv)
        cond_dir = RESULTS_DIR / "conditional_e5"
        summary = conditional_analysis_e5(e5_rows, cond_dir)
        win_cond = summary.get("win_conditions", {})
        for comp, data in win_cond.items():
            n_wins = len(data.get("wins", []))
            n_losses = len(data.get("losses", []))
            print(f"  vs {comp}: {n_wins} wins, {n_losses} losses")
        print(f"  Saved to {cond_dir}/")

    # Stage 4: E6 Stability Analysis
    e6_csv = RESULTS_DIR / "raw_e6.csv"
    if e6_csv.exists():
        print("\n[Stage 4] E6 stability analysis...")
        e6_rows = load_raw_csv(e6_csv)
        stab_dir = RESULTS_DIR / "stability_e6"
        stability_analysis_e6(e6_rows, stab_dir)
        print(f"  Saved to {stab_dir}/")

    # Stage 5: Case Studies
    print("\n[Stage 5] Case studies...")
    run_case_studies(RESULTS_DIR, quick=False)

    # Stage 6: Component Ablation
    print("\n[Stage 6] Component ablation...")
    run_component_ablation(RESULTS_DIR, quick=False, seeds=SEEDS[:3])

    # Stage 7: Hyperparameter Robustness
    print("\n[Stage 7] Hyperparameter robustness...")
    run_hyperparam_robustness(RESULTS_DIR, quick=False, seeds=SEEDS[:3])

    # Stage 8: Statistical Significance
    print("\n[Stage 8] Statistical significance...")
    run_significance_tests(RESULTS_DIR, quick=False)

    # Stage 9: E4 Analysis
    print("\n[Stage 9] E4 byte breakdown...")
    run_e4_analysis(RESULTS_DIR, quick=False, seeds=SEEDS[:3])

    # Cross-dataset significance
    if e5_csv.exists() and e6_csv.exists():
        print("\n[Extra] Cross-dataset significance...")
        sig_dir = RESULTS_DIR / "significance"
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
    print(f"Phase 4 pipeline completed in {elapsed:.1f}s")
    print(f"All results saved to {RESULTS_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
