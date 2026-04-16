from __future__ import annotations

"""Oracle intervention experiments for causal decomposition of FPT's performance gap.

Runs FedProTrack with controlled oracle overrides to isolate which failure mode
(identity inference, over-spawning, bank contamination, or aggregation) is
responsible for the gap to CFL/Oracle.

Interventions:
  1. FPT-base: Normal FedProTrack (fingerprint-based Phase A)
  2. FPT-OracleAssign: Oracle concept assignments at Phase A, but FPT's
     memory bank / model storage / aggregation pipeline
  3. FPT-CapSpawn: FPT but max_concepts capped at ground-truth count
  4. FPT-OracleAssign+CapSpawn: Both oracle assignment and capped concepts
  5. Oracle: Ground-truth federated aggregation (existing baseline)
  6. FedAvg: Global aggregation baseline
  7. CFL: Clustered FL baseline
"""

import argparse
import copy
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import (
    run_fedavg_baseline,
    run_oracle_baseline,
)
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.experiments.method_registry import identity_metrics_valid
from fedprotrack.metrics import compute_all_metrics
from fedprotrack.metrics.experiment_log import ExperimentLog
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def _make_log(
    method_name: str,
    result: object,
    ground_truth: np.ndarray,
) -> ExperimentLog:
    """Convert a method result into a unified ExperimentLog."""
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


class OracleAssignmentRunner:
    """Wrapper around FedProTrackRunner that injects oracle concept assignments.

    After Phase A computes its (noisy) fingerprint-based assignments,
    this wrapper replaces them with ground-truth concept IDs from the
    dataset's concept_matrix.  Everything else (memory bank, model
    storage, aggregation) proceeds normally, so the experiment isolates
    the effect of perfect identity inference.
    """

    def __init__(
        self,
        base_runner: FedProTrackRunner,
        dataset_concept_matrix: np.ndarray,
    ) -> None:
        self._base = base_runner
        self._gt = dataset_concept_matrix  # (K, T)

    def run(self, dataset):
        """Monkey-patch Phase A to return oracle assignments, then run."""
        from fedprotrack.posterior.gibbs import PosteriorAssignment
        from fedprotrack.posterior.memory_bank import MemorySlot
        from fedprotrack.baselines.comm_tracker import fingerprint_bytes
        from fedprotrack.posterior.two_phase_protocol import (
            PhaseAResult,
            TwoPhaseFedProTrack,
        )

        original_phase_a = TwoPhaseFedProTrack.phase_a
        gt_matrix = self._gt
        federation_every = self._base.federation_every
        _state = {"fed_round": 0}

        def _oracle_phase_a(protocol_self, client_fps, prev_assignments,
                            client_model_losses=None,
                            client_signatures=None,
                            client_update_signatures=None,
                            client_batch_prototype_signatures=None):
            """Construct PhaseAResult directly from ground truth — no spawning."""
            K = len(client_fps)
            t = _state["fed_round"] * federation_every
            T = gt_matrix.shape[1]

            fp_bytes = fingerprint_bytes(
                protocol_self.config.n_features,
                protocol_self.config.n_classes,
                precision_bits=16,
                include_global_mean=False,
            )
            bytes_up = K * fp_bytes

            assignments: dict[int, int] = {}
            posteriors: dict[int, PosteriorAssignment] = {}

            for k in range(K):
                true_cid = int(gt_matrix[k, min(t, T - 1)])
                assignments[k] = true_cid
                posteriors[k] = PosteriorAssignment(
                    probabilities={true_cid: 1.0},
                    map_concept_id=true_cid,
                    is_novel=False,
                    entropy=0.0,
                )

                # Ensure concept slot exists in memory bank (clean creation)
                if true_cid not in protocol_self.memory_bank._library:
                    fp_copy = copy.deepcopy(client_fps[k])
                    protocol_self.memory_bank._library[true_cid] = fp_copy
                    protocol_self.memory_bank._slots[true_cid] = MemorySlot(
                        slot_id=true_cid,
                        center_key=fp_copy,
                        semantic_anchor_set=fp_copy,
                    )
                    if protocol_self.memory_bank._next_id <= true_cid:
                        protocol_self.memory_bank._next_id = true_cid + 1
                else:
                    # Absorb fingerprint into existing slot (clean update)
                    protocol_self.memory_bank.absorb_fingerprint(
                        true_cid, client_fps[k],
                    )

            _state["fed_round"] += 1

            return PhaseAResult(
                assignments=assignments,
                posteriors=posteriors,
                bytes_up=bytes_up,
                bytes_down=K * 4.0,
                library_size_before=protocol_self.memory_bank.n_concepts,
            )

        TwoPhaseFedProTrack.phase_a = _oracle_phase_a
        try:
            result = self._base.run(dataset)
        finally:
            TwoPhaseFedProTrack.phase_a = original_phase_a

        return result


def _run_oracle_no_warmstart(
    dataset,
    config,
    *,
    federation_every: int,
    seed: int,
    lr: float,
    n_epochs: int,
):
    """FPT-OracleAssign but with recall_concept_model patched to return None."""
    from fedprotrack.posterior.two_phase_protocol import TwoPhaseFedProTrack

    original_recall = TwoPhaseFedProTrack.recall_concept_model

    def _no_recall(self, concept_id):
        return None

    TwoPhaseFedProTrack.recall_concept_model = _no_recall
    try:
        runner = FedProTrackRunner(
            config=config,
            federation_every=federation_every,
            detector_name="ADWIN",
            seed=seed,
            lr=lr,
            n_epochs=n_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        )
        result = OracleAssignmentRunner(runner, dataset.concept_matrix).run(dataset)
    finally:
        TwoPhaseFedProTrack.recall_concept_model = original_recall
    return result


def _run_simple_oracle_per_client(
    dataset,
    *,
    federation_every: int,
    lr: float,
    n_epochs: int,
    seed: int,
):
    """Oracle aggregation with per-client models (same setup as FPT runner)."""
    from fedprotrack.baselines.comm_tracker import model_bytes as _model_bytes
    from fedprotrack.experiment.runner import ExperimentResult
    from fedprotrack.evaluation.metrics import ConceptTrackingMetrics
    from fedprotrack.models import TorchLinearClassifier

    K, T = dataset.concept_matrix.shape
    n_features = dataset.data[(0, 0)][0].shape[1]
    all_labels: set[int] = set()
    for (k_i, t_i), (_, y_i) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y_i))
    n_classes = max(all_labels) + 1

    # Per-client models (same init as FPT runner)
    models = [
        TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed + k,
        )
        for k in range(K)
    ]
    acc_matrix = np.zeros((K, T), dtype=np.float64)
    total_bytes = 0.0

    for t in range(T):
        uploads_by_concept: dict[int, list[dict[str, np.ndarray]]] = {}
        for k in range(K):
            X, y = dataset.data[(k, t)]
            mid = len(X) // 2
            X_test, y_test = X[:mid], y[:mid]
            X_train, y_train = X[mid:], y[mid:]

            y_pred = models[k].predict(X_test)
            acc_matrix[k, t] = float(np.mean(y_pred == y_test))

            models[k].fit(X_train, y_train)
            true_concept = int(dataset.concept_matrix[k, t])
            uploads_by_concept.setdefault(true_concept, []).append(
                (k, models[k].get_params())
            )

        if (t + 1) % federation_every == 0 and t < T - 1:
            for concept_id, client_uploads in uploads_by_concept.items():
                params_list = [p for _, p in client_uploads]
                aggregated = {
                    key: np.mean(np.stack([p[key] for p in params_list]), axis=0)
                    for key in params_list[0]
                }
                for k_c, _ in client_uploads:
                    models[k_c].set_params(aggregated)
                n_c = len(client_uploads)
                total_bytes += n_c * _model_bytes(aggregated) * 2

    return ExperimentResult(
        config=None,
        method_name="SimpleOracle-PerClient",
        accuracy_matrix=acc_matrix,
        concept_tracking_accuracy=1.0,
        predicted_concept_matrix=dataset.concept_matrix.copy(),
        true_concept_matrix=dataset.concept_matrix,
        mean_accuracy=float(acc_matrix.mean()),
        final_accuracy=float(acc_matrix[:, -1].mean()),
        forgetting=0.0,
        backward_transfer=0.0,
        total_bytes=total_bytes,
        tracking_metrics=ConceptTrackingMetrics(),
    )


def _build_methods(
    dataset,
    exp_cfg: ExperimentConfig,
    *,
    federation_every: int,
    fpt_lr: float,
    fpt_epochs: int,
) -> dict[str, object]:
    """Build callable map for all methods under test."""
    n_concepts_true = int(dataset.concept_matrix.max()) + 1
    seed = int(dataset.config.seed)

    def _fpt_config(**overrides):
        defaults = dict(
            omega=2.0,
            kappa=0.7,
            novelty_threshold=0.25,
            loss_novelty_threshold=0.02,
            sticky_dampening=1.0,
            sticky_posterior_gate=0.35,
            merge_threshold=0.80,
            min_count=5.0,
            max_concepts=6,
            merge_every=2,
            shrink_every=6,
        )
        defaults.update(overrides)
        return TwoPhaseConfig(**defaults)

    def _fpt_runner(**config_overrides):
        return FedProTrackRunner(
            config=_fpt_config(**config_overrides),
            federation_every=federation_every,
            detector_name="ADWIN",
            seed=seed,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            soft_aggregation=True,
            blend_alpha=0.0,
            similarity_calibration=True,
        )

    return {
        # 1. Normal FedProTrack
        "FPT-base": lambda: _fpt_runner().run(dataset),

        # 2. Oracle concept assignments, FPT pipeline (soft agg)
        "FPT-OracleAssign": lambda: OracleAssignmentRunner(
            _fpt_runner(),
            dataset.concept_matrix,
        ).run(dataset),

        # 3. Oracle + no warm-start (patch recall_concept_model to disable)
        "FPT-OracleNoWarm": lambda: _run_oracle_no_warmstart(
            dataset,
            _fpt_config(),
            federation_every=federation_every,
            seed=seed,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
        ),

        # 4. FPT with concept count capped at ground truth
        "FPT-CapSpawn": lambda: _fpt_runner(
            max_concepts=n_concepts_true,
        ).run(dataset),

        # 5. Full Oracle baseline
        "Oracle": lambda: run_oracle_baseline(
            exp_cfg,
            dataset=dataset,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            seed=seed,
        ),

        # 6. FedAvg
        "FedAvg": lambda: run_fedavg_baseline(
            exp_cfg,
            dataset=dataset,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            seed=seed,
        ),

        # 7. CFL
        "CFL": lambda: run_cfl_full(
            dataset, federation_every=federation_every,
        ),

        # 8. Simple Oracle with per-client models (same init as FPT)
        "SimpleOracle-PerClient": lambda: _run_simple_oracle_per_client(
            dataset,
            federation_every=federation_every,
            lr=fpt_lr,
            n_epochs=fpt_epochs,
            seed=seed,
        ),

    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Oracle intervention experiments for FPT causal decomposition"
    )
    parser.add_argument("--results-dir", default="tmp/oracle_interventions")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44],
    )
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=800)
    parser.add_argument("--rho", type=float, default=7.5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--delta", type=float, default=0.85)
    parser.add_argument("--n-features", type=int, default=128)
    parser.add_argument("--samples-per-coarse-class", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-workers", type=int, default=0)
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--fpt-lr", type=float, default=0.05)
    parser.add_argument("--fpt-epochs", type=int, default=5)
    parser.add_argument("--label-split", default="disjoint")
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--data-root", default=".cifar100_cache")
    parser.add_argument("--feature-cache-dir", default=".feature_cache")
    parser.add_argument("--feature-seed", type=int, default=2718)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        print(
            f"\n{'='*70}\n"
            f"seed={seed}, K={args.K}, T={args.T}, "
            f"label_split={args.label_split}\n"
            f"{'='*70}",
            flush=True,
        )

        dataset_cfg = CIFAR100RecurrenceConfig(
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
            seed=seed,
            label_split=args.label_split,
            min_group_size=args.min_group_size,
        )

        print("Preparing feature cache...", flush=True)
        prepare_cifar100_recurrence_feature_cache(dataset_cfg)
        dataset = generate_cifar100_recurrence_dataset(dataset_cfg)

        K, T = dataset.concept_matrix.shape
        n_concepts = int(dataset.concept_matrix.max()) + 1
        print(f"  Concept matrix: K={K}, T={T}, n_concepts={n_concepts}")

        exp_cfg = ExperimentConfig(
            generator_config=dataset.config,
            federation_every=args.federation_every,
        )

        methods = _build_methods(
            dataset,
            exp_cfg,
            federation_every=args.federation_every,
            fpt_lr=args.fpt_lr,
            fpt_epochs=args.fpt_epochs,
        )

        for method_name, run_fn in methods.items():
            print(f"  Running {method_name}...", end=" ", flush=True)
            t0 = time.time()
            try:
                result = run_fn()
                log = _make_log(method_name, result, dataset.concept_matrix)
                metrics = compute_all_metrics(
                    log,
                    identity_capable=identity_metrics_valid(method_name),
                )
                elapsed = time.time() - t0

                fa = getattr(metrics, "final_accuracy", None) or 0.0
                reid = getattr(metrics, "concept_re_id_accuracy", None) or 0.0
                auc = getattr(metrics, "accuracy_auc", None) or 0.0
                total_bytes = getattr(log, "total_bytes", None) or 0.0

                spawned = getattr(result, "spawned_concepts", None) or 0
                active = getattr(result, "active_concepts", None) or 0

                row = {
                    "seed": seed,
                    "method": method_name,
                    "final_acc": round(float(fa), 4),
                    "re_id": round(float(reid), 4),
                    "auc": round(float(auc), 4),
                    "total_bytes": int(total_bytes),
                    "spawned_concepts": spawned,
                    "active_concepts": active,
                    "elapsed_s": round(elapsed, 1),
                }
                all_rows.append(row)
                print(
                    f"acc={row['final_acc']:.3f}  "
                    f"re_id={row['re_id']:.3f}  "
                    f"spawned={spawned}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
            except Exception:
                print("FAILED", flush=True)
                traceback.print_exc()

    # ---- Save results ----
    csv_path = results_dir / "oracle_interventions.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to {csv_path}", flush=True)

    # ---- Compute summary ----
    method_names = list(dict.fromkeys(r["method"] for r in all_rows))
    summary_rows = []
    for method in method_names:
        method_rows = [r for r in all_rows if r["method"] == method]
        accs = [r["final_acc"] for r in method_rows]
        reids = [r["re_id"] for r in method_rows]
        aucs = [r["auc"] for r in method_rows]
        spawned = [r["spawned_concepts"] for r in method_rows]

        summary_rows.append({
            "method": method,
            "mean_final_acc": round(float(np.mean(accs)), 4),
            "std_final_acc": round(float(np.std(accs)), 4),
            "mean_re_id": round(float(np.mean(reids)), 4),
            "mean_auc": round(float(np.mean(aucs)), 4),
            "mean_total_bytes": float(np.mean([r["total_bytes"] for r in method_rows])),
            "mean_spawned": float(np.mean(spawned)),
        })

    # ---- Causal decomposition ----
    method_acc = {s["method"]: s["mean_final_acc"] for s in summary_rows}
    print("\n" + "=" * 80)
    print("CAUSAL DECOMPOSITION")
    print("=" * 80)
    print(f"{'Method':<25} {'Acc':>8} {'Gap to Oracle':>15} {'Re-ID':>8} {'Spawned':>10}")
    print("-" * 75)
    oracle_acc = method_acc.get("Oracle", 0.0)
    for s in sorted(summary_rows, key=lambda x: -x["mean_final_acc"]):
        gap = s["mean_final_acc"] - oracle_acc
        print(
            f"{s['method']:<25} "
            f"{s['mean_final_acc']:>8.4f} "
            f"{gap:>+15.4f} "
            f"{s['mean_re_id']:>8.4f} "
            f"{s['mean_spawned']:>10.1f}"
        )

    # Decomposition analysis
    fpt_base = method_acc.get("FPT-base", 0.0)
    fpt_oracle = method_acc.get("FPT-OracleAssign", 0.0)
    fpt_oracle_hard = method_acc.get("FPT-OracleHard", 0.0)
    fpt_cap = method_acc.get("FPT-CapSpawn", 0.0)

    print("\n--- Gap Attribution ---")
    total_gap = oracle_acc - fpt_base
    assign_fix = fpt_oracle - fpt_base
    hard_fix = fpt_oracle_hard - fpt_base
    spawn_fix = fpt_cap - fpt_base
    pipeline_gap = oracle_acc - fpt_oracle

    print(f"Total gap (Oracle - FPT-base):        {total_gap:+.4f}")
    print(f"Effect of oracle assignment (soft):    {assign_fix:+.4f} ({assign_fix/max(total_gap,1e-6)*100:.0f}%)")
    print(f"Effect of oracle assignment (hard):    {hard_fix:+.4f} ({hard_fix/max(total_gap,1e-6)*100:.0f}%)")
    print(f"Effect of capping spawn count:         {spawn_fix:+.4f} ({spawn_fix/max(total_gap,1e-6)*100:.0f}%)")
    print(f"Pipeline gap (Oracle - FPT-OracleAssign): {pipeline_gap:+.4f}")
    print(f"  -> This is the FPT pipeline overhead with PERFECT assignments")

    # Save JSON summary
    json_path = results_dir / "summary.json"
    summary = {
        "experiment": "oracle_interventions",
        "label_split": args.label_split,
        "K": args.K,
        "T": args.T,
        "seeds": args.seeds,
        "federation_every": args.federation_every,
        "fpt_lr": args.fpt_lr,
        "fpt_epochs": args.fpt_epochs,
        "summary": summary_rows,
        "decomposition": {
            "total_gap_oracle_vs_fpt": round(total_gap, 4),
            "effect_oracle_assign_soft": round(assign_fix, 4),
            "effect_oracle_assign_hard": round(hard_fix, 4),
            "effect_cap_spawn": round(spawn_fix, 4),
            "pipeline_gap": round(pipeline_gap, 4),
        },
        "raw_rows": all_rows,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nJSON summary saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
