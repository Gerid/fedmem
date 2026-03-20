from __future__ import annotations

"""Diagnose the accuracy impact of correct vs wrong concept assignments.

For FPT on disjoint labels, computes accuracy conditioned on whether the
concept assignment (after Hungarian alignment) is correct or wrong.
Also shows FedAvg and Oracle accuracy at the same cells for comparison.
"""

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.metrics.hungarian import align_predictions
from fedprotrack.posterior.fedprotrack_runner import FedProTrackRunner
from fedprotrack.posterior.two_phase_protocol import TwoPhaseConfig
from fedprotrack.real_data import (
    CIFAR100RecurrenceConfig,
    generate_cifar100_recurrence_dataset,
    prepare_cifar100_recurrence_feature_cache,
)


def main() -> None:
    seeds = [42, 43, 44, 45, 46]

    # Accumulators across seeds
    all_fpt_correct_acc = []
    all_fpt_wrong_acc = []
    all_fedavg_at_correct = []
    all_fedavg_at_wrong = []
    all_oracle_at_correct = []
    all_oracle_at_wrong = []
    all_cfl_at_correct = []
    all_cfl_at_wrong = []
    all_n_correct = []
    all_n_wrong = []

    for seed in seeds:
        print(f"\n{'='*60}\nseed={seed}\n{'='*60}", flush=True)

        cfg = CIFAR100RecurrenceConfig(
            K=4, T=12, n_samples=200, rho=3.0, alpha=0.75, delta=0.9,
            n_features=64, samples_per_coarse_class=30, batch_size=128,
            n_workers=0, data_root=".cifar100_cache",
            feature_cache_dir=".feature_cache", feature_seed=2718,
            seed=seed, label_split="disjoint",
        )
        prepare_cifar100_recurrence_feature_cache(cfg)
        ds = generate_cifar100_recurrence_dataset(cfg)
        n_true = int(ds.concept_matrix.max()) + 1
        exp_cfg = ExperimentConfig(
            generator_config=ds.config, federation_every=2
        )

        # Run all methods
        fpt_result = FedProTrackRunner(
            config=TwoPhaseConfig(
                omega=2.0, kappa=0.7, novelty_threshold=0.25,
                loss_novelty_threshold=0.15, sticky_dampening=1.5,
                sticky_posterior_gate=0.35, merge_threshold=0.85,
                min_count=5.0, max_concepts=max(6, n_true + 3),
                merge_every=2, shrink_every=6,
            ),
            federation_every=2, detector_name="ADWIN",
            seed=seed, lr=0.05, n_epochs=5,
            soft_aggregation=True, blend_alpha=0.0,
        ).run(ds)

        fedavg_result = run_fedavg_baseline(exp_cfg, dataset=ds)
        oracle_result = run_oracle_baseline(exp_cfg, dataset=ds)
        cfl_result = run_cfl_full(ds, federation_every=2)

        # Get accuracy matrices
        fpt_acc = fpt_result.accuracy_matrix  # (K, T)
        fedavg_acc = np.asarray(fedavg_result.accuracy_matrix, dtype=np.float64)
        oracle_acc = np.asarray(oracle_result.accuracy_matrix, dtype=np.float64)
        cfl_acc = np.asarray(cfl_result.accuracy_matrix, dtype=np.float64)

        # Hungarian alignment of FPT predictions
        gt = ds.concept_matrix
        pred = fpt_result.predicted_concept_matrix
        aligned, mapping = align_predictions(gt, pred)

        correct_mask = (aligned == gt)  # (K, T) boolean
        wrong_mask = ~correct_mask

        n_correct = int(correct_mask.sum())
        n_wrong = int(wrong_mask.sum())
        n_total = gt.size

        print(f"Correct cells: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
        print(f"Wrong cells:   {n_wrong}/{n_total} ({n_wrong/n_total:.1%})")

        # Conditioned accuracy
        fpt_correct = float(fpt_acc[correct_mask].mean()) if n_correct > 0 else 0.0
        fpt_wrong = float(fpt_acc[wrong_mask].mean()) if n_wrong > 0 else 0.0
        fedavg_correct = float(fedavg_acc[correct_mask].mean()) if n_correct > 0 else 0.0
        fedavg_wrong = float(fedavg_acc[wrong_mask].mean()) if n_wrong > 0 else 0.0
        oracle_correct = float(oracle_acc[correct_mask].mean()) if n_correct > 0 else 0.0
        oracle_wrong = float(oracle_acc[wrong_mask].mean()) if n_wrong > 0 else 0.0
        cfl_correct = float(cfl_acc[correct_mask].mean()) if n_correct > 0 else 0.0
        cfl_wrong = float(cfl_acc[wrong_mask].mean()) if n_wrong > 0 else 0.0

        print(f"\n{'Method':<12} {'All':>8} {'Correct':>8} {'Wrong':>8} {'Gap':>8}")
        print("-" * 48)
        print(f"{'FPT':<12} {fpt_acc.mean():>8.3f} {fpt_correct:>8.3f} {fpt_wrong:>8.3f} {fpt_correct-fpt_wrong:>+8.3f}")
        print(f"{'FedAvg':<12} {fedavg_acc.mean():>8.3f} {fedavg_correct:>8.3f} {fedavg_wrong:>8.3f} {fedavg_correct-fedavg_wrong:>+8.3f}")
        print(f"{'CFL':<12} {cfl_acc.mean():>8.3f} {cfl_correct:>8.3f} {cfl_wrong:>8.3f} {cfl_correct-cfl_wrong:>+8.3f}")
        print(f"{'Oracle':<12} {oracle_acc.mean():>8.3f} {oracle_correct:>8.3f} {oracle_wrong:>8.3f} {oracle_correct-oracle_wrong:>+8.3f}")

        # Show which cells are wrong and what the accuracy looks like
        print(f"\nPer-cell detail (correct=C, wrong=W):")
        print(f"  Ground truth:\n  {gt}")
        print(f"  FPT aligned:\n  {aligned}")
        labels = np.where(correct_mask, "C", "W")
        print(f"  Status:\n  {labels}")
        print(f"  FPT accuracy:\n  {np.round(fpt_acc, 2)}")
        print(f"  FedAvg accuracy:\n  {np.round(fedavg_acc, 2)}")

        all_fpt_correct_acc.append(fpt_correct)
        all_fpt_wrong_acc.append(fpt_wrong)
        all_fedavg_at_correct.append(fedavg_correct)
        all_fedavg_at_wrong.append(fedavg_wrong)
        all_oracle_at_correct.append(oracle_correct)
        all_oracle_at_wrong.append(oracle_wrong)
        all_cfl_at_correct.append(cfl_correct)
        all_cfl_at_wrong.append(cfl_wrong)
        all_n_correct.append(n_correct)
        all_n_wrong.append(n_wrong)

    # Aggregate summary
    print(f"\n{'='*70}")
    print(f"AGGREGATE SUMMARY ACROSS {len(seeds)} SEEDS")
    print(f"{'='*70}")
    mean_pct_correct = np.mean(all_n_correct) / (np.mean(all_n_correct) + np.mean(all_n_wrong))
    print(f"Mean correct cells: {mean_pct_correct:.1%}")
    print(f"\n{'Method':<12} {'@Correct':>10} {'@Wrong':>10} {'Gap':>10}")
    print("-" * 48)
    print(f"{'FPT':<12} {np.mean(all_fpt_correct_acc):>10.3f} {np.mean(all_fpt_wrong_acc):>10.3f} {np.mean(all_fpt_correct_acc)-np.mean(all_fpt_wrong_acc):>+10.3f}")
    print(f"{'FedAvg':<12} {np.mean(all_fedavg_at_correct):>10.3f} {np.mean(all_fedavg_at_wrong):>10.3f} {np.mean(all_fedavg_at_correct)-np.mean(all_fedavg_at_wrong):>+10.3f}")
    print(f"{'CFL':<12} {np.mean(all_cfl_at_correct):>10.3f} {np.mean(all_cfl_at_wrong):>10.3f} {np.mean(all_cfl_at_correct)-np.mean(all_cfl_at_wrong):>+10.3f}")
    print(f"{'Oracle':<12} {np.mean(all_oracle_at_correct):>10.3f} {np.mean(all_oracle_at_wrong):>10.3f} {np.mean(all_oracle_at_correct)-np.mean(all_oracle_at_wrong):>+10.3f}")

    print(f"\nINTERPRETATION:")
    fpt_c = np.mean(all_fpt_correct_acc)
    fpt_w = np.mean(all_fpt_wrong_acc)
    fedavg_all = np.mean([np.mean([all_fedavg_at_correct[i], all_fedavg_at_wrong[i]]) for i in range(len(seeds))])
    print(f"  FPT@correct ({fpt_c:.3f}) vs Oracle@correct ({np.mean(all_oracle_at_correct):.3f}): ", end="")
    print(f"correct assignments {'help' if fpt_c > fedavg_all else 'do NOT help enough'}")
    print(f"  FPT@wrong ({fpt_w:.3f}) vs FedAvg@wrong ({np.mean(all_fedavg_at_wrong):.3f}): ", end="")
    print(f"wrong assignments {'hurt less than FedAvg' if fpt_w > np.mean(all_fedavg_at_wrong) else 'hurt MORE than FedAvg'}")
    print(f"  FedAvg overall ({fedavg_all:.3f}): {'wrong assignments destroy FPT advantage' if fpt_w < fedavg_all * 0.5 else 'wrong assignments moderate'}")


if __name__ == "__main__":
    main()
