"""Extract FPT-OT + all baselines across 5 (K, rho) configs.

Final-round accuracy = last element of mean_accuracy_curve.
Aggregate across 3 seeds per config, then cross-config mean.
"""
from __future__ import annotations
import json
import statistics as stats
from pathlib import Path

CONFIGS = [
    ("K20_rho17", 20, 17, 6),
    ("K20_rho25", 20, 25, 4),
    ("K20_rho33", 20, 33, 3),
    ("K40_rho25", 40, 25, 4),
    ("K40_rho33", 40, 33, 3),
]

ROOT = Path("E:/fedprotrack/tmp")


def file_main(tag: str) -> Path | None:
    p = ROOT / f"runpod_neurips_{tag}.json"
    return p if p.exists() else None


def file_cluster(tag: str) -> Path | None:
    for o in [
        ROOT / f"runpod_cluster_{tag}_methods_cluster.json",
        ROOT / f"runpod_cluster_{tag}_saved.json",
    ]:
        if o.exists():
            return o
    return None


def per_seed_final(summary: dict, method: str) -> float | None:
    if method not in summary:
        return None
    curve = summary[method].get("mean_accuracy_curve")
    if not curve:
        return None
    return float(curve[-1])


def collect(path: Path) -> dict[str, list[float]]:
    with open(path) as f:
        d = json.load(f)
    out: dict[str, list[float]] = {}
    for seed_key, seed_data in d.items():
        if not isinstance(seed_data, dict):
            continue
        res = seed_data.get("results", {})
        summ = res.get("summary.json", {})
        if not summ:
            continue
        for m in summ.keys():
            final = per_seed_final(summ, m)
            if final is None:
                continue
            out.setdefault(m, []).append(final)
    return out


def aggregate(tag: str) -> dict:
    methods: dict[str, list[float]] = {}
    # Main neurips file → FedAvg, IFCA, FedProTrack (FPT-OT), Oracle
    main = file_main(tag)
    if main:
        for m, vals in collect(main).items():
            methods.setdefault(m, []).extend(vals)
    # Cluster file → CFL, FedEM, FedRC
    clust = file_cluster(tag)
    if clust:
        for m, vals in collect(clust).items():
            methods.setdefault(m, []).extend(vals)
    out = {}
    for m, vals in methods.items():
        out[m] = {
            "mean": stats.mean(vals),
            "std": stats.pstdev(vals) if len(vals) > 1 else 0.0,
            "n": len(vals),
        }
    return out


def main():
    all_cfg = {}
    for tag, K, rho, C in CONFIGS:
        agg = aggregate(tag)
        all_cfg[tag] = {"K": K, "rho": rho, "C": C, "methods": agg}
        print(f"\n=== {tag} (K={K}, rho={rho}, C={C}) ===")
        order = [
            "FedAvg", "Oracle", "IFCA", "FedEM", "FedRC", "CFL",
            "FedAvg-FPTTrain", "FedProTrack",
        ]
        for m in order:
            if m not in agg:
                continue
            st = agg[m]
            label = "FPT-OT" if m == "FedProTrack" else m
            print(f"  {label:18s}  {st['mean']:.4f} ± {st['std']:.4f}  (n={st['n']})")

    # Cross-config means
    print("\n" + "=" * 60)
    print("CROSS-CONFIG MEAN (average across 5 configs × 3 seeds = 15 runs per method)")
    print("=" * 60)
    method_vals: dict[str, list[float]] = {}
    for entry in all_cfg.values():
        for m, st in entry["methods"].items():
            method_vals.setdefault(m, []).append(st["mean"])
    order = [
        "FedProTrack", "Oracle", "CFL", "IFCA", "FedEM", "FedRC",
        "FedAvg-FPTTrain", "FedAvg",
    ]
    for m in order:
        if m not in method_vals:
            continue
        vals = method_vals[m]
        label = "FPT-OT" if m == "FedProTrack" else m
        print(f"  {label:18s}  cross-cfg mean={stats.mean(vals):.4f}  "
              f"std_across_cfg={stats.pstdev(vals) if len(vals) > 1 else 0.0:.4f}  n_cfgs={len(vals)}")

    # Save
    with open("E:/fedprotrack/tmp/flagship_fptot_aggregated.json", "w") as f:
        json.dump(all_cfg, f, indent=2)
    print("\nSaved → tmp/flagship_fptot_aggregated.json")


if __name__ == "__main__":
    main()
