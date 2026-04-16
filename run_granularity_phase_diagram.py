from __future__ import annotations

"""Granularity phase diagram: sweep (K, C, concept_separation) to find
when concept-level aggregation beats global aggregation.

Compares three methods on synthetic data:
  - FedAvg (global aggregation, no concept awareness)
  - Oracle (grouped FedAvg with true concept IDs)
  - CFL (clustered federated learning, discovers clusters from updates)

Outputs to ``tmp/granularity_phase_diagram/``.
"""

import argparse
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np

from fedprotrack.baselines.runners import run_cfl_full
from fedprotrack.drift_generator.concept_matrix import generate_concept_matrix
from fedprotrack.drift_generator.configs import GeneratorConfig
from fedprotrack.drift_generator.data_streams import ConceptSpec, generate_samples
from fedprotrack.drift_generator.generator import DriftDataset
from fedprotrack.experiment.baselines import run_fedavg_baseline, run_oracle_baseline
from fedprotrack.experiment.runner import ExperimentConfig
from fedprotrack.models import TorchLinearClassifier


def _make_diverse_concept_specs(
    n_concepts: int,
    delta: float,
) -> list[ConceptSpec]:
    """Build concept specs that stay distinct even when n_concepts > 4.

    Cycles through sine variants 0-3 and sea variants 0-3 to provide up
    to 8 truly distinct data-generating distributions.  Beyond 8 the
    specs wrap, but the sweep parameters stay within [2, 8].

    Parameters
    ----------
    n_concepts : int
        Number of distinct concepts needed.
    delta : float
        Separability in (0, 1].  Higher means less noise.

    Returns
    -------
    list[ConceptSpec]
    """
    noise_scale = 1.0 - delta
    pool: list[tuple[str, int]] = []
    # 4 sine variants + 4 sea variants = 8 distinct generators
    for v in range(4):
        pool.append(("sine", v))
    for v in range(4):
        pool.append(("sea", v))

    specs: list[ConceptSpec] = []
    for c in range(n_concepts):
        gen_type, variant = pool[c % len(pool)]
        specs.append(ConceptSpec(
            concept_id=c,
            generator_type=gen_type,
            variant=variant,
            noise_scale=noise_scale,
        ))
    return specs


def _infer_n_features_from_spec(spec: ConceptSpec) -> int:
    return {"sine": 2, "sea": 3, "circle": 2}[spec.generator_type]


def _generate_phase_diagram_dataset(
    K: int,
    T: int,
    C: int,
    delta: float,
    alpha: float,
    n_samples: int,
    seed: int,
) -> DriftDataset:
    """Generate a synthetic DriftDataset with exactly C concepts.

    Parameters
    ----------
    K : int
        Number of clients.
    T : int
        Number of time steps.
    C : int
        Number of distinct concepts.
    delta : float
        Concept separability in (0, 1].
    alpha : float
        Asynchrony level in [0, 1].
    n_samples : int
        Samples per (client, time-step).
    seed : int
        Random seed.

    Returns
    -------
    DriftDataset
    """
    matrix = generate_concept_matrix(
        K=K, T=T, n_concepts=C, alpha=alpha, seed=seed,
    )

    specs = _make_diverse_concept_specs(C, delta)

    # When mixing sine (2-d) and sea (3-d) generators we must project
    # everything into a shared feature space.  Pad 2-d samples with a
    # constant zero column so all data is 3-d.
    max_dim = max(_infer_n_features_from_spec(s) for s in specs)

    data: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for k in range(K):
        for t in range(T):
            concept_id = int(matrix[k, t])
            sample_seed = seed + k * T + t + 10000
            X, y = generate_samples(specs[concept_id], n_samples, sample_seed)
            if X.shape[1] < max_dim:
                pad_width = max_dim - X.shape[1]
                X = np.hstack([X, np.zeros((X.shape[0], pad_width), dtype=X.dtype)])
            data[(k, t)] = (X, y)

    # Build a GeneratorConfig that the baselines can inspect.
    # rho is chosen so that config.n_concepts == C.  Since
    # n_concepts = max(2, round(T / rho)), we set rho = T / C.
    rho = T / C if C < T else 1.0
    cfg = GeneratorConfig(
        K=K, T=T, n_samples=n_samples,
        rho=rho, alpha=alpha, delta=delta,
        generator_type="sine",  # nominal; actual data is mixed
        seed=seed,
    )

    return DriftDataset(
        concept_matrix=matrix,
        data=data,
        config=cfg,
        concept_specs=specs,
    )


def _per_concept_accuracy(
    dataset: DriftDataset,
    n_features: int,
    n_classes: int,
    seed: int,
    n_epochs: int = 5,
    lr: float = 0.1,
) -> float:
    """Train one model per concept (no federation) to measure intrinsic
    task difficulty.

    Returns the mean accuracy across all concepts, averaged over all
    (client, time-step) cells belonging to each concept.  This isolates
    'how hard is each concept to learn' from 'how well does aggregation
    work'.
    """
    K = dataset.config.K
    T = dataset.config.T
    concept_ids = sorted(set(int(v) for v in np.unique(dataset.concept_matrix)))

    accs_per_concept: dict[int, list[float]] = {c: [] for c in concept_ids}

    for cid in concept_ids:
        model = TorchLinearClassifier(
            n_features=n_features,
            n_classes=n_classes,
            lr=lr,
            n_epochs=n_epochs,
            seed=seed + cid * 1000,
        )
        cells = list(zip(*np.where(dataset.concept_matrix == cid)))
        # Train on first 80% of cells, test on last 20%
        if len(cells) < 2:
            accs_per_concept[cid].append(0.5)
            continue

        # Collect all data for this concept
        all_X, all_y = [], []
        for k, t in cells:
            X, y = dataset.data[(k, t)]
            all_X.append(X)
            all_y.append(y)
        all_X = np.concatenate(all_X)
        all_y = np.concatenate(all_y)

        n = len(all_X)
        split = max(1, int(n * 0.8))
        X_train, y_train = all_X[:split], all_y[:split]
        X_test, y_test = all_X[split:], all_y[split:]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = float(np.mean(y_pred == y_test)) if len(y_test) > 0 else 0.5
        accs_per_concept[cid].append(acc)

    return float(np.mean([np.mean(v) for v in accs_per_concept.values() if v]))


def _run_one_point(
    K: int,
    T: int,
    C: int,
    delta: float,
    alpha: float,
    n_samples: int,
    seed: int,
    federation_every: int,
    lr: float,
    n_epochs: int,
    cfl_warmup: int,
) -> dict[str, object]:
    """Run all three methods for one (K, C, delta, seed) point.

    Returns a dict with all recorded metrics.
    """
    dataset = _generate_phase_diagram_dataset(
        K=K, T=T, C=C, delta=delta, alpha=alpha,
        n_samples=n_samples, seed=seed,
    )

    # Infer dims from data
    X0, y0 = dataset.data[(0, 0)]
    n_features = X0.shape[1]
    all_labels: set[int] = set()
    for (_, _val), (_, y) in dataset.data.items():
        all_labels.update(int(v) for v in np.unique(y))
    n_classes = max(all_labels) + 1

    # Per-concept accuracy (task difficulty control)
    concept_acc = _per_concept_accuracy(
        dataset, n_features, n_classes, seed,
        n_epochs=n_epochs, lr=lr,
    )

    exp_cfg = ExperimentConfig(
        generator_config=dataset.config,
        federation_every=federation_every,
    )

    results: dict[str, dict[str, float]] = {}

    # --- FedAvg ---
    try:
        fedavg_res = run_fedavg_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        )
        results["FedAvg"] = {
            "mean_acc": float(fedavg_res.mean_accuracy),
            "final_acc": float(fedavg_res.final_accuracy),
            "total_bytes": float(fedavg_res.total_bytes or 0),
        }
    except Exception:
        traceback.print_exc()
        results["FedAvg"] = {"mean_acc": 0.0, "final_acc": 0.0, "total_bytes": 0.0}

    # --- Oracle ---
    try:
        oracle_res = run_oracle_baseline(
            exp_cfg, dataset=dataset, lr=lr, n_epochs=n_epochs, seed=seed,
        )
        results["Oracle"] = {
            "mean_acc": float(oracle_res.mean_accuracy),
            "final_acc": float(oracle_res.final_accuracy),
            "total_bytes": float(oracle_res.total_bytes or 0),
        }
    except Exception:
        traceback.print_exc()
        results["Oracle"] = {"mean_acc": 0.0, "final_acc": 0.0, "total_bytes": 0.0}

    # --- CFL ---
    try:
        cfl_res = run_cfl_full(
            dataset,
            federation_every=federation_every,
            warmup_rounds=cfl_warmup,
            max_clusters=max(C + 2, 8),
        )
        results["CFL"] = {
            "mean_acc": float(cfl_res.accuracy_matrix.mean()),
            "final_acc": float(cfl_res.accuracy_matrix[:, -1].mean()),
            "total_bytes": float(cfl_res.total_bytes),
        }
    except Exception:
        traceback.print_exc()
        results["CFL"] = {"mean_acc": 0.0, "final_acc": 0.0, "total_bytes": 0.0}

    # Oracle advantage = Oracle_acc - FedAvg_acc
    oracle_advantage = results["Oracle"]["final_acc"] - results["FedAvg"]["final_acc"]

    # Theoretical crossover prediction: when K/C is large AND delta is
    # high, concept-level aggregation should dominate because each
    # concept group has enough clients to average noise away AND the
    # concepts are different enough that global averaging hurts.
    kc_ratio = K / C
    theory_predicts_oracle_wins = kc_ratio >= 2.0 and delta >= 0.5

    return {
        "K": K,
        "C": C,
        "delta": delta,
        "seed": seed,
        "K_over_C": round(kc_ratio, 2),
        "concept_acc": round(concept_acc, 4),
        "fedavg_mean_acc": round(results["FedAvg"]["mean_acc"], 4),
        "fedavg_final_acc": round(results["FedAvg"]["final_acc"], 4),
        "oracle_mean_acc": round(results["Oracle"]["mean_acc"], 4),
        "oracle_final_acc": round(results["Oracle"]["final_acc"], 4),
        "cfl_mean_acc": round(results["CFL"]["mean_acc"], 4),
        "cfl_final_acc": round(results["CFL"]["final_acc"], 4),
        "oracle_advantage": round(oracle_advantage, 4),
        "theory_predicts_oracle": theory_predicts_oracle_wins,
        "fedavg_bytes": int(results["FedAvg"]["total_bytes"]),
        "oracle_bytes": int(results["Oracle"]["total_bytes"]),
        "cfl_bytes": int(results["CFL"]["total_bytes"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Granularity phase diagram: sweep (K, C, separation) to find "
            "when concept-level aggregation beats global aggregation."
        ),
    )
    parser.add_argument("--results-dir", default="tmp/granularity_phase_diagram")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--federation-every", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--cfl-warmup", type=int, default=5)
    parser.add_argument(
        "--K-values", type=int, nargs="+", default=[10, 20, 40],
        help="Client counts to sweep",
    )
    parser.add_argument(
        "--C-values", type=int, nargs="+", default=[2, 3, 5, 8],
        help="Concept counts to sweep",
    )
    parser.add_argument(
        "--delta-values", type=float, nargs="+",
        default=[0.2, 0.4, 0.6, 0.8, 1.0],
        help="Concept separation levels (delta) to sweep",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []

    total_points = (
        len(args.K_values)
        * len(args.C_values)
        * len(args.delta_values)
        * len(args.seeds)
    )
    point_idx = 0

    for K in args.K_values:
        for C in args.C_values:
            for delta in args.delta_values:
                for seed in args.seeds:
                    point_idx += 1
                    print(
                        f"\n[{point_idx}/{total_points}] "
                        f"K={K}, C={C}, delta={delta:.2f}, seed={seed}",
                        flush=True,
                    )
                    t0 = time.time()
                    try:
                        row = _run_one_point(
                            K=K,
                            T=args.T,
                            C=C,
                            delta=delta,
                            alpha=args.alpha,
                            n_samples=args.n_samples,
                            seed=seed,
                            federation_every=args.federation_every,
                            lr=args.lr,
                            n_epochs=args.n_epochs,
                            cfl_warmup=args.cfl_warmup,
                        )
                        elapsed = time.time() - t0
                        row["elapsed_s"] = round(elapsed, 1)
                        all_rows.append(row)
                        print(
                            f"  FedAvg={row['fedavg_final_acc']:.3f}  "
                            f"Oracle={row['oracle_final_acc']:.3f}  "
                            f"CFL={row['cfl_final_acc']:.3f}  "
                            f"advantage={row['oracle_advantage']:+.3f}  "
                            f"concept_acc={row['concept_acc']:.3f}  "
                            f"({elapsed:.1f}s)",
                            flush=True,
                        )
                    except Exception:
                        print("  FAILED", flush=True)
                        traceback.print_exc()

    # ---- Save CSV ----
    csv_path = results_dir / "phase_diagram_results.csv"
    if all_rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nCSV saved to {csv_path}", flush=True)

    # ---- Summary tables ----
    print("\n" + "=" * 90)
    print("GRANULARITY PHASE DIAGRAM: Oracle advantage (final_acc) averaged across seeds")
    print("=" * 90)

    # Group by (K, C, delta) and average across seeds
    from collections import defaultdict

    grouped: dict[tuple[int, int, float], list[dict]] = defaultdict(list)
    for row in all_rows:
        key = (row["K"], row["C"], row["delta"])
        grouped[key].append(row)

    print(
        f"{'K':>4}  {'C':>3}  {'K/C':>5}  {'delta':>6}  "
        f"{'FedAvg':>8}  {'Oracle':>8}  {'CFL':>8}  "
        f"{'OracAdv':>9}  {'ConceptAcc':>10}  {'TheoryWins':>10}"
    )
    print("-" * 90)

    summary_rows: list[dict[str, object]] = []
    for (K, C, delta) in sorted(grouped.keys()):
        rows = grouped[(K, C, delta)]
        avg = lambda key: float(np.mean([r[key] for r in rows]))
        fa = avg("fedavg_final_acc")
        oa = avg("oracle_final_acc")
        ca = avg("cfl_final_acc")
        adv = avg("oracle_advantage")
        cacc = avg("concept_acc")
        theory = rows[0]["theory_predicts_oracle"]
        kc = round(K / C, 2)

        print(
            f"{K:>4}  {C:>3}  {kc:>5.1f}  {delta:>6.2f}  "
            f"{fa:>8.4f}  {oa:>8.4f}  {ca:>8.4f}  "
            f"{adv:>+9.4f}  {cacc:>10.4f}  {str(theory):>10}"
        )
        summary_rows.append({
            "K": K,
            "C": C,
            "K_over_C": kc,
            "delta": delta,
            "fedavg_final_acc": round(fa, 4),
            "oracle_final_acc": round(oa, 4),
            "cfl_final_acc": round(ca, 4),
            "oracle_advantage": round(adv, 4),
            "concept_acc": round(cacc, 4),
            "theory_predicts_oracle": theory,
        })

    # ---- Crossover analysis ----
    print("\n" + "=" * 90)
    print("CROSSOVER ANALYSIS: Where Oracle advantage crosses zero")
    print("=" * 90)
    for K in args.K_values:
        for C in args.C_values:
            advs = []
            for delta in sorted(args.delta_values):
                key = (K, C, delta)
                if key in grouped:
                    adv = float(np.mean([r["oracle_advantage"] for r in grouped[key]]))
                    advs.append((delta, adv))
            if advs:
                sign_changes = []
                for i in range(1, len(advs)):
                    if advs[i - 1][1] * advs[i][1] < 0:
                        # Linear interpolation to find crossover delta
                        d0, a0 = advs[i - 1]
                        d1, a1 = advs[i]
                        crossover = d0 + (d1 - d0) * (-a0) / (a1 - a0)
                        sign_changes.append(round(crossover, 3))
                if sign_changes:
                    print(
                        f"  K={K}, C={C} (K/C={K/C:.1f}): "
                        f"crossover at delta ~ {sign_changes}"
                    )
                else:
                    direction = "always positive" if advs[-1][1] > 0 else "always negative"
                    print(f"  K={K}, C={C} (K/C={K/C:.1f}): {direction}")

    # ---- Save summary JSON ----
    summary = {
        "experiment": "granularity_phase_diagram",
        "T": args.T,
        "n_samples": args.n_samples,
        "alpha": args.alpha,
        "seeds": args.seeds,
        "K_values": args.K_values,
        "C_values": args.C_values,
        "delta_values": args.delta_values,
        "federation_every": args.federation_every,
        "lr": args.lr,
        "n_epochs": args.n_epochs,
        "cfl_warmup": args.cfl_warmup,
        "summary": summary_rows,
        "raw_rows": all_rows,
    }
    json_path = results_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary JSON saved to {json_path}", flush=True)


if __name__ == "__main__":
    main()
