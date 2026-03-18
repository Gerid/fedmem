"""CLI entry point for the drift generator and metrics pipeline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .drift_generator import GeneratorConfig, generate_drift_dataset
from .metrics import (
    ExperimentLog,
    PhaseDiagramData,
    build_phase_diagram,
    compute_all_metrics,
    load_results_from_dir,
    plot_phase_diagram,
)
from .metrics.visualization import plot_budget_frontier


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a single drift dataset."""
    config = GeneratorConfig(
        K=args.K,
        T=args.T,
        n_samples=args.n_samples,
        rho=args.rho,
        alpha=args.alpha,
        delta=args.delta,
        generator_type=args.generator,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Generating: {config.dir_name}")
    print(f"  K={config.K}, T={config.T}, n_concepts={config.n_concepts}")

    dataset = generate_drift_dataset(config)
    out_dir = dataset.save()
    print(f"  Saved to: {out_dir}")

    # Auto-visualize
    _save_heatmap(dataset.concept_matrix, out_dir / "concept_matrix.png", config.dir_name)
    print(f"  Heatmap saved to: {out_dir / 'concept_matrix.png'}")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Generate datasets for all parameter combinations."""
    rho_vals = _parse_float_list(args.rho_list)
    alpha_vals = _parse_float_list(args.alpha_list)
    delta_vals = _parse_float_list(args.delta_list)

    combos = list(product(rho_vals, alpha_vals, delta_vals))
    print(f"Sweep: {len(combos)} configurations")

    for i, (rho, alpha, delta) in enumerate(combos, 1):
        config = GeneratorConfig(
            K=args.K,
            T=args.T,
            n_samples=args.n_samples,
            rho=rho,
            alpha=alpha,
            delta=delta,
            generator_type=args.generator,
            seed=args.seed,
            output_dir=args.output_dir,
        )
        print(f"  [{i}/{len(combos)}] {config.dir_name}")
        dataset = generate_drift_dataset(config)
        out_dir = dataset.save()
        _save_heatmap(dataset.concept_matrix, out_dir / "concept_matrix.png", config.dir_name)

    print("Sweep complete.")


def cmd_visualize(args: argparse.Namespace) -> None:
    """Visualize a concept matrix from a saved dataset."""
    input_dir = Path(args.input_dir)
    matrix = np.load(input_dir / "concept_matrix.npy")
    title = input_dir.name
    out_path = input_dir / "concept_matrix.png"
    _save_heatmap(matrix, out_path, title)
    print(f"Heatmap saved to: {out_path}")

    if args.show:
        plt.show()


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Evaluate concept identity inference predictions against ground truth."""
    gt_dir = Path(args.ground_truth_dir)
    gt_matrix = np.load(gt_dir / "concept_matrix.npy").astype("int32")
    K, T = gt_matrix.shape

    if args.baseline == "fedavg":
        predicted = np.zeros((K, T), dtype="int32")
        method_name = "FedAvg"
        print("Using FedAvg baseline: all clients assigned to concept 0")
    elif args.predictions is not None:
        predicted = np.load(args.predictions).astype("int32")
        method_name = args.method_name or Path(args.predictions).stem
    else:
        print("Error: provide --predictions <path.npy> or --baseline fedavg")
        import sys; sys.exit(1)

    accuracy_curve = None
    if args.accuracy_curve is not None:
        accuracy_curve = np.load(args.accuracy_curve).astype("float64")

    log = ExperimentLog(
        ground_truth=gt_matrix, predicted=predicted,
        accuracy_curve=accuracy_curve, total_bytes=args.total_bytes,
        method_name=method_name,
    )
    result = compute_all_metrics(log)

    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  Method : {log.method_name}")
    print(f"  Dataset: {gt_dir.name}")
    print(sep)
    if result.concept_re_id_accuracy is not None:
        print(f"  concept_re_id_accuracy   : {result.concept_re_id_accuracy:.4f}")
    else:
        print(f"  concept_re_id_accuracy   : --  (method does not track identity)")
    if result.assignment_entropy is not None:
        print(f"  assignment_entropy       : {result.assignment_entropy:.4f}")
    else:
        print(f"  assignment_entropy       : --  (method does not track identity)")
    if result.assignment_switch_rate is not None:
        print(f"  assignment_switch_rate   : {result.assignment_switch_rate:.4f}")
        print(f"  avg_clients_per_concept  : {result.avg_clients_per_concept:.4f}")
        print(f"  singleton_group_ratio    : {result.singleton_group_ratio:.4f}")
        print(f"  memory_reuse_rate        : {result.memory_reuse_rate:.4f}")
        print(f"  routing_consistency      : {result.routing_consistency:.4f}")
    else:
        print(f"  assignment_switch_rate   : --  (method does not track identity)")
    if result.wrong_memory_reuse_rate is not None:
        print(f"  wrong_memory_reuse_rate  : {result.wrong_memory_reuse_rate:.4f}")
    else:
        print(f"  wrong_memory_reuse_rate  : --  (method does not track identity)")
    if result.worst_window_dip is not None:
        print(f"  worst_window_dip         : {result.worst_window_dip:.4f}")
        print(f"  worst_window_recovery    : {result.worst_window_recovery} steps")
    if result.budget_normalized_score is not None:
        print(f"  budget_normalized_score  : {result.budget_normalized_score:.6f}")
    print(sep)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_json(out_dir / "metrics.json")
    log.save(out_dir / "log")
    print(f"\n  Metrics saved to: {out_dir / 'metrics.json'}")


def cmd_phase_diagram(args: argparse.Namespace) -> None:
    """Build and render a phase diagram from sweep results."""
    results_dir = Path(args.results_dir)
    method_name = args.method_name or "unknown"
    print(f"Loading results from: {results_dir}")
    results = load_results_from_dir(results_dir, method_name=method_name)
    if not results:
        print(f"No results found in {results_dir}. Subdirectories must contain config.json + metrics.json.")
        import sys; sys.exit(1)
    print(f"  Found {len(results)} result(s)")
    diagram = build_phase_diagram(
        results=results, row_param=args.row, col_param=args.col,
        metric_name=args.metric, method_name=method_name,
    )
    out_path = (
        Path(args.output) if args.output
        else results_dir / f"phase_{args.row}_x_{args.col}_{args.metric}.png"
    )
    fig = plot_phase_diagram(diagram, output_path=out_path)
    plt.close(fig)
    print(f"  Phase diagram saved to: {out_path}")
    npz_path = out_path.with_suffix(".npz")
    diagram.to_npz(npz_path)
    print(f"  Phase diagram data saved to: {npz_path}")


def cmd_budget_sweep(args: argparse.Namespace) -> None:
    """Run matched-budget sweep across the baseline suite."""
    from .drift_generator.generator import DriftDataset
    from .baselines.budget_sweep import BudgetPoint, run_budget_sweep, find_crossover_points
    import json

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {dataset_dir}")
    concept_matrix = np.load(dataset_dir / "concept_matrix.npy").astype("int32")
    K, T = concept_matrix.shape

    data: dict = {}
    data_dir = dataset_dir / "data"
    for k in range(K):
        for t in range(T):
            npz_path = data_dir / f"client{k:03d}_step{t:03d}.npz"
            if npz_path.exists():
                arr = np.load(npz_path)
                data[(k, t)] = (arr["X"], arr["y"])
            else:
                raise FileNotFoundError(f"Missing data file: {npz_path}")

    config = GeneratorConfig.from_json(dataset_dir / "config.json")
    dataset = DriftDataset(
        concept_matrix=concept_matrix,
        data=data,
        config=config,
        concept_specs=[],
    )

    # Parse federation_every list
    fe_list = [int(x) for x in args.federation_every_list.split(",")]
    print(f"federation_every values: {fe_list}")

    # Run sweep
    print("Running budget sweep (3 methods × federation_every values)...")
    points = run_budget_sweep(
        dataset=dataset,
        federation_every_values=fe_list,
        similarity_threshold=args.similarity_threshold,
    )

    # Save JSON
    json_path = out_dir / "budget_points.json"
    records = [
        {
            "method_name": p.method_name,
            "federation_every": p.federation_every,
            "total_bytes": p.total_bytes,
            "accuracy_auc": p.accuracy_auc,
        }
        for p in points
    ]
    json_path.write_text(json.dumps(records, indent=2))
    print(f"  Budget points saved to: {json_path}")

    # Group by method
    method_points: dict[str, list[BudgetPoint]] = {}
    for p in points:
        method_points.setdefault(p.method_name, []).append(p)

    # Find crossover points
    method_names = list(method_points.keys())
    for i in range(len(method_names)):
        for j in range(i + 1, len(method_names)):
            mn_a, mn_b = method_names[i], method_names[j]
            crossovers = find_crossover_points(method_points[mn_a], method_points[mn_b])
            if crossovers:
                print(f"  Crossover {mn_a} vs {mn_b}:")
                for bytes_val, auc_val in crossovers:
                    print(f"    @ {bytes_val:.0f} bytes, AUC={auc_val:.4f}")
            else:
                print(f"  No crossover found: {mn_a} vs {mn_b}")

    # Plot frontier
    png_path = out_dir / "budget_frontier.png"
    fig = plot_budget_frontier(method_points, output_path=png_path)
    plt.close(fig)
    print(f"  Budget frontier plot saved to: {png_path}")


def _save_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    """Save concept matrix as a heatmap."""
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 0.8), max(4, matrix.shape[0] * 0.5)))
    im = ax.imshow(matrix, aspect="auto", cmap="tab10", interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Client")
    ax.set_title(f"Concept Matrix: {title}")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    fig.colorbar(im, ax=ax, label="Concept ID")

    # Annotate cells with concept IDs
    for k in range(matrix.shape[0]):
        for t in range(matrix.shape[1]):
            ax.text(t, k, str(matrix[k, t]), ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _parse_float_list(s: str) -> list[float]:
    """Parse comma-separated float list, supporting 'inf'."""
    vals = []
    for v in s.split(","):
        v = v.strip()
        if v.lower() == "inf":
            vals.append(math.inf)
        else:
            vals.append(float(v))
    return vals


def _parse_rho(s: str) -> float:
    """Parse rho value, supporting 'inf'."""
    if s.lower() == "inf":
        return math.inf
    return float(s)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fedprotrack",
        description="FedProTrack: Distributed Drift Generator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--K", type=int, default=10, help="Number of clients")
        p.add_argument("--T", type=int, default=10, help="Number of time steps")
        p.add_argument("--n-samples", type=int, default=500, help="Samples per (client, step)")
        p.add_argument("--generator", type=str, default="sine", choices=["sine", "sea", "circle"])
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--output-dir", type=str, default="outputs")

    # generate
    gen_parser = sub.add_parser("generate", help="Generate a single drift dataset")
    add_common(gen_parser)
    gen_parser.add_argument("--rho", type=_parse_rho, default=5.0, help="Recurrence (use 'inf' for no reuse)")
    gen_parser.add_argument("--alpha", type=float, default=0.5, help="Asynchrony [0,1]")
    gen_parser.add_argument("--delta", type=float, default=0.5, help="Separability (0,1]")

    # sweep
    sweep_parser = sub.add_parser("sweep", help="Sweep over parameter combinations")
    add_common(sweep_parser)
    sweep_parser.add_argument("--rho-list", type=str, default="2,5,10,inf", help="Comma-separated rho values")
    sweep_parser.add_argument("--alpha-list", type=str, default="0,0.5,1.0", help="Comma-separated alpha values")
    sweep_parser.add_argument("--delta-list", type=str, default="0.1,0.5,1.0", help="Comma-separated delta values")

    # visualize
    viz_parser = sub.add_parser("visualize", help="Visualize a concept matrix")
    viz_parser.add_argument("input_dir", type=str, help="Path to saved dataset directory")
    viz_parser.add_argument("--show", action="store_true", help="Show plot interactively")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate identity inference predictions")
    ev.add_argument("ground_truth_dir", type=str, help="Directory with concept_matrix.npy")
    ev.add_argument("--predictions", type=str, default=None, help="Path to predicted .npy")
    ev.add_argument("--baseline", type=str, choices=["fedavg"], default=None)
    ev.add_argument("--method-name", type=str, default=None)
    ev.add_argument("--accuracy-curve", type=str, default=None)
    ev.add_argument("--total-bytes", type=float, default=None)
    ev.add_argument("--output-dir", type=str, default="results")

    # phase-diagram
    pd = sub.add_parser("phase-diagram", help="Build phase diagram from sweep results")
    pd.add_argument("results_dir", type=str)
    pd.add_argument("--row", type=str, default="rho")
    pd.add_argument("--col", type=str, default="delta")
    pd.add_argument("--metric", type=str, default="concept_re_id_accuracy",
        choices=["concept_re_id_accuracy","assignment_entropy","assignment_switch_rate",
                 "avg_clients_per_concept","singleton_group_ratio","memory_reuse_rate",
                 "routing_consistency","wrong_memory_reuse_rate",
                 "worst_window_dip","worst_window_recovery","budget_normalized_score"])
    pd.add_argument("--method-name", type=str, default=None)
    pd.add_argument("--output", type=str, default=None)

    # budget-sweep
    bs = sub.add_parser("budget-sweep", help="Matched-budget comparison of baseline methods")
    bs.add_argument("--dataset-dir", type=str, required=True, help="Path to saved DriftDataset directory")
    bs.add_argument("--federation-every-list", type=str, default="1,2,5,10",
                    help="Comma-separated federation_every values (default: 1,2,5,10)")
    bs.add_argument("--similarity-threshold", type=float, default=0.5,
                    help="TrackedSummary clustering threshold (default: 0.5)")
    bs.add_argument("--output-dir", type=str, default="results/budget_sweep")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "sweep":
        cmd_sweep(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "phase-diagram":
        cmd_phase_diagram(args)
    elif args.command == "budget-sweep":
        cmd_budget_sweep(args)

if __name__ == "__main__":
    main()
