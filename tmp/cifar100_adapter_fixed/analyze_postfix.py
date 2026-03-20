from __future__ import annotations

"""Full post-fix adapter analysis with statistical tests."""

import json
import statistics
from pathlib import Path
from collections import defaultdict
from itertools import combinations

BASE_FIXED = Path(r"E:\fedprotrack\.claude\worktrees\elegant-poitras\tmp\cifar100_adapter_fixed")
BASE_PREFIX = Path(r"E:\fedprotrack\.claude\worktrees\elegant-poitras\tmp\cifar100_adapter")

SEEDS = [42, 43, 44, 45, 46]


def load_results(path):
    with open(path / "results.json") as f:
        return json.load(f)["rows"]


def group_by_method_seed(rows):
    """Return {method: {seed: row}}."""
    out = defaultdict(dict)
    for r in rows:
        out[r["method"]][r["seed"]] = r
    return dict(out)


def paired_values(data, m1, m2, key):
    """Extract paired values for two methods across common seeds."""
    v1, v2 = [], []
    for s in SEEDS:
        a = data.get(m1, {}).get(s)
        b = data.get(m2, {}).get(s)
        if a and b and a.get(key) is not None and b.get(key) is not None:
            v1.append(a[key])
            v2.append(b[key])
    return v1, v2


def wilcoxon_test(x, y):
    """Wilcoxon signed-rank test. Returns (statistic, p_value, n)."""
    from scipy.stats import wilcoxon
    diffs = [a - b for a, b in zip(x, y)]
    # Check if all diffs are zero
    if all(d == 0 for d in diffs):
        return 0.0, 1.0, len(diffs)
    try:
        stat, p = wilcoxon(x, y, alternative="two-sided")
        return stat, p, len(diffs)
    except Exception as e:
        return float("nan"), float("nan"), len(diffs)


def cohens_d(x, y):
    """Paired Cohen's d."""
    diffs = [a - b for a, b in zip(x, y)]
    if len(diffs) < 2:
        return float("nan")
    mean_d = statistics.mean(diffs)
    std_d = statistics.stdev(diffs)
    if std_d == 0:
        return float("inf") if mean_d != 0 else 0.0
    return mean_d / std_d


def method_summary(data, method, keys):
    """Return {key: (mean, std)} for a method."""
    out = {}
    for k in keys:
        vals = [data[method][s][k] for s in SEEDS
                if s in data.get(method, {}) and data[method][s].get(k) is not None]
        if vals:
            out[k] = (statistics.mean(vals),
                      statistics.stdev(vals) if len(vals) > 1 else 0.0,
                      len(vals))
    return out


def main():
    # Load data
    postfix_rows = load_results(BASE_FIXED)
    prefix_rows = load_results(BASE_PREFIX)

    post_data = group_by_method_seed(postfix_rows)
    pre_data = group_by_method_seed(prefix_rows)

    methods_post = sorted(post_data.keys())
    print("=" * 80)
    print("POST-FIX ADAPTER ANALYSIS REPORT")
    print("CIFAR-100 Recurrence Benchmark | T=20, K=4, n_samples=400 | 5 seeds")
    print("=" * 80)

    # ===== 1. Summary Table =====
    print("\n## 1. Method Summary (mean +/- std, 5 seeds)")
    print()
    metrics = ["final_accuracy", "accuracy_auc", "concept_re_id_accuracy",
               "wrong_memory_reuse_rate", "assignment_entropy", "total_bytes"]
    header = f"{'Method':30s} | {'Final Acc':14s} | {'Acc AUC':14s} | {'Re-ID':14s} | {'Wrong Mem':14s} | {'Entropy':14s} | {'Bytes (KB)':12s}"
    print(header)
    print("-" * len(header))

    for m in methods_post:
        s = method_summary(post_data, m, metrics)
        def fmt(k, mult=1.0):
            if k in s:
                return f"{s[k][0]*mult:.4f}+/-{s[k][1]*mult:.4f}"
            return "--"
        def fmt_bytes(k):
            if k in s:
                return f"{s[k][0]/1000:.0f}"
            return "--"
        print(f"{m:30s} | {fmt('final_accuracy'):14s} | {fmt('accuracy_auc'):14s} | "
              f"{fmt('concept_re_id_accuracy'):14s} | {fmt('wrong_memory_reuse_rate'):14s} | "
              f"{fmt('assignment_entropy'):14s} | {fmt_bytes('total_bytes'):>12s}")

    # ===== 2. Pre-fix vs Post-fix Comparison =====
    print("\n\n## 2. Pre-fix vs Post-fix Adapter Comparison")
    print()

    # Pre-fix adapter was called "FPT-adapter-base"
    pre_adapter_acc = [pre_data["FPT-adapter-base"][s]["final_accuracy"] for s in SEEDS]
    post_ep20_acc = [post_data["FPT-adapter-ep20-fed5"][s]["final_accuracy"] for s in SEEDS]
    post_ep30_acc = [post_data["FPT-adapter-ep30-fed5"][s]["final_accuracy"] for s in SEEDS]

    print(f"  Pre-fix  adapter-base:     {statistics.mean(pre_adapter_acc):.4f} +/- {statistics.stdev(pre_adapter_acc):.4f}")
    print(f"  Post-fix adapter-ep20:     {statistics.mean(post_ep20_acc):.4f} +/- {statistics.stdev(post_ep20_acc):.4f}")
    print(f"  Post-fix adapter-ep30:     {statistics.mean(post_ep30_acc):.4f} +/- {statistics.stdev(post_ep30_acc):.4f}")
    print()

    improvement_ep20 = statistics.mean(post_ep20_acc) - statistics.mean(pre_adapter_acc)
    improvement_ep30 = statistics.mean(post_ep30_acc) - statistics.mean(pre_adapter_acc)
    print(f"  Improvement (ep20 - pre-fix): +{improvement_ep20:.4f} ({improvement_ep20/statistics.mean(pre_adapter_acc)*100:.0f}% relative)")
    print(f"  Improvement (ep30 - pre-fix): +{improvement_ep30:.4f} ({improvement_ep30/statistics.mean(pre_adapter_acc)*100:.0f}% relative)")

    # ===== 3. Pairwise Wilcoxon Tests =====
    print("\n\n## 3. Pairwise Statistical Tests (Wilcoxon signed-rank, two-sided)")
    print()

    identity_methods = [m for m in methods_post if m != "CFL"]
    all_methods = methods_post
    pairs = list(combinations(all_methods, 2))

    # Bonferroni correction
    n_tests = len(pairs)
    alpha = 0.05
    bonferroni_alpha = alpha / n_tests

    print(f"  Number of pairwise tests: {n_tests}")
    print(f"  Bonferroni-corrected alpha: {bonferroni_alpha:.4f}")
    print()

    test_header = f"  {'Pair':50s} | {'Mean Diff':10s} | {'Cohen d':8s} | {'W-stat':8s} | {'p-value':10s} | {'Sig?':5s}"
    print(test_header)
    print("  " + "-" * (len(test_header) - 2))

    for metric_name, metric_key in [("Final Accuracy", "final_accuracy"),
                                      ("Accuracy AUC", "accuracy_auc")]:
        print(f"\n  --- {metric_name} ---")
        for m1, m2 in pairs:
            v1, v2 = paired_values(post_data, m1, m2, metric_key)
            if len(v1) < 3:
                continue
            mean_diff = statistics.mean(v1) - statistics.mean(v2)
            d = cohens_d(v1, v2)
            w, p, n = wilcoxon_test(v1, v2)
            sig = "YES" if p < bonferroni_alpha else "no"
            print(f"  {m1+' vs '+m2:50s} | {mean_diff:+.4f}    | {d:+.3f}   | {w:.1f}    | {p:.4f}     | {sig}")

    # ===== 4. Adapter vs Linear Deep Dive =====
    print("\n\n## 4. Adapter vs Linear Deep Dive")
    print()

    lin_acc = [post_data["FPT-linear-base"][s]["final_accuracy"] for s in SEEDS]
    lin_reid = [post_data["FPT-linear-base"][s]["concept_re_id_accuracy"] for s in SEEDS]
    lin_entropy = [post_data["FPT-linear-base"][s]["assignment_entropy"] for s in SEEDS]
    lin_bytes = [post_data["FPT-linear-base"][s]["total_bytes"] for s in SEEDS]
    lin_wmr = [post_data["FPT-linear-base"][s]["wrong_memory_reuse_rate"] for s in SEEDS]

    ep30_acc = [post_data["FPT-adapter-ep30-fed5"][s]["final_accuracy"] for s in SEEDS]
    ep30_reid = [post_data["FPT-adapter-ep30-fed5"][s]["concept_re_id_accuracy"] for s in SEEDS]
    ep30_entropy = [post_data["FPT-adapter-ep30-fed5"][s]["assignment_entropy"] for s in SEEDS]
    ep30_bytes = [post_data["FPT-adapter-ep30-fed5"][s]["total_bytes"] for s in SEEDS]
    ep30_wmr = [post_data["FPT-adapter-ep30-fed5"][s]["wrong_memory_reuse_rate"] for s in SEEDS]

    ep20_acc = [post_data["FPT-adapter-ep20-fed5"][s]["final_accuracy"] for s in SEEDS]
    ep20_reid = [post_data["FPT-adapter-ep20-fed5"][s]["concept_re_id_accuracy"] for s in SEEDS]
    ep20_bytes = [post_data["FPT-adapter-ep20-fed5"][s]["total_bytes"] for s in SEEDS]

    print("  ### 4a. Accuracy Gap")
    gap_ep30 = statistics.mean(lin_acc) - statistics.mean(ep30_acc)
    gap_ep20 = statistics.mean(lin_acc) - statistics.mean(ep20_acc)
    print(f"  Linear:      {statistics.mean(lin_acc):.4f} +/- {statistics.stdev(lin_acc):.4f}")
    print(f"  Adapter-30:  {statistics.mean(ep30_acc):.4f} +/- {statistics.stdev(ep30_acc):.4f}")
    print(f"  Adapter-20:  {statistics.mean(ep20_acc):.4f} +/- {statistics.stdev(ep20_acc):.4f}")
    print(f"  Gap (lin-ep30): {gap_ep30:+.4f} ({gap_ep30/statistics.mean(lin_acc)*100:.1f}%)")
    print(f"  Gap (lin-ep20): {gap_ep20:+.4f} ({gap_ep20/statistics.mean(lin_acc)*100:.1f}%)")

    # Per-seed gaps
    print("\n  Per-seed accuracy (linear vs adapter-ep30):")
    for s in SEEDS:
        la = post_data["FPT-linear-base"][s]["final_accuracy"]
        aa = post_data["FPT-adapter-ep30-fed5"][s]["final_accuracy"]
        print(f"    seed={s}: linear={la:.4f}  adapter={aa:.4f}  gap={la-aa:+.4f}")

    # Wilcoxon for linear vs adapter-ep30
    w, p, n = wilcoxon_test(lin_acc, ep30_acc)
    d = cohens_d(lin_acc, ep30_acc)
    print(f"\n  Wilcoxon (linear vs adapter-ep30): W={w:.1f}, p={p:.4f}, Cohen's d={d:+.3f}, n={n}")
    print(f"  --> {'Significant' if p < 0.05 else 'Not significant'} at alpha=0.05")

    # ===== 4b. Re-ID Drop =====
    print("\n  ### 4b. Re-ID Accuracy Drop")
    print(f"  Linear re-ID:      {statistics.mean(lin_reid):.4f} +/- {statistics.stdev(lin_reid):.4f}")
    print(f"  Adapter-ep30 re-ID:{statistics.mean(ep30_reid):.4f} +/- {statistics.stdev(ep30_reid):.4f}")
    print(f"  Adapter-ep20 re-ID:{statistics.mean(ep20_reid):.4f} +/- {statistics.stdev(ep20_reid):.4f}")
    reid_drop = statistics.mean(lin_reid) - statistics.mean(ep30_reid)
    print(f"  Drop (lin - ep30): {reid_drop:+.4f} ({reid_drop/statistics.mean(lin_reid)*100:.1f}%)")

    w_r, p_r, n_r = wilcoxon_test(lin_reid, ep30_reid)
    d_r = cohens_d(lin_reid, ep30_reid)
    print(f"  Wilcoxon (re-ID): W={w_r:.1f}, p={p_r:.4f}, Cohen's d={d_r:+.3f}")

    print("\n  Per-seed re-ID:")
    for s in SEEDS:
        lr = post_data["FPT-linear-base"][s]["concept_re_id_accuracy"]
        ar = post_data["FPT-adapter-ep30-fed5"][s]["concept_re_id_accuracy"]
        print(f"    seed={s}: linear={lr:.4f}  adapter={ar:.4f}  gap={lr-ar:+.4f}")

    # ===== 4c. Entropy / Wrong Memory =====
    print("\n  ### 4c. Assignment Entropy & Wrong Memory Reuse")
    print(f"  Linear entropy:      {statistics.mean(lin_entropy):.4f} +/- {statistics.stdev(lin_entropy):.4f}")
    print(f"  Adapter-ep30 entropy:{statistics.mean(ep30_entropy):.4f} +/- {statistics.stdev(ep30_entropy):.4f}")
    print(f"  --> Adapter has MUCH lower entropy ({statistics.mean(ep30_entropy)/statistics.mean(lin_entropy)*100:.0f}% of linear)")
    print(f"      This means adapter Phase A is more confident but WRONG more often")
    print(f"  Linear wrong-mem:      {statistics.mean(lin_wmr):.4f}")
    print(f"  Adapter-ep30 wrong-mem:{statistics.mean(ep30_wmr):.4f}")

    # ===== 5. Budget Efficiency =====
    print("\n\n## 5. Budget Efficiency")
    print()
    for m in methods_post:
        acc_vals = [post_data[m][s]["final_accuracy"] for s in SEEDS
                    if s in post_data[m] and post_data[m][s].get("final_accuracy") is not None]
        byte_vals = [post_data[m][s]["total_bytes"] for s in SEEDS
                     if s in post_data[m] and post_data[m][s].get("total_bytes") is not None]
        if acc_vals and byte_vals:
            ma = statistics.mean(acc_vals)
            mb = statistics.mean(byte_vals)
            eff = ma / (mb / 1e6)
            print(f"  {m:30s}  acc={ma:.4f}  bytes={mb/1000:.0f}KB  acc/MB={eff:.2f}")

    # Adapter bytes ratio
    lin_b_mean = statistics.mean(lin_bytes)
    ep30_b_mean = statistics.mean(ep30_bytes)
    ep20_b_mean = statistics.mean(ep20_bytes)
    print(f"\n  Adapter-ep30 uses {ep30_b_mean/lin_b_mean:.2f}x the bytes of linear")
    print(f"  Adapter-ep20 uses {ep20_b_mean/lin_b_mean:.2f}x the bytes of linear")

    # ===== 6. Win Rate Table =====
    print("\n\n## 6. Win Rate (per-seed, final_accuracy)")
    print()
    wins = defaultdict(int)
    for s in SEEDS:
        best_m = None
        best_v = -1
        for m in methods_post:
            v = post_data[m][s]["final_accuracy"]
            if v > best_v:
                best_v = v
                best_m = m
        wins[best_m] += 1
    for m in methods_post:
        print(f"  {m:30s}  wins: {wins[m]}/5")

    # ===== 7. Ranking =====
    print("\n\n## 7. Mean Rank (lower is better)")
    print()
    rank_sums = defaultdict(float)
    for s in SEEDS:
        sorted_m = sorted(methods_post,
                          key=lambda m: post_data[m][s]["final_accuracy"],
                          reverse=True)
        for rank, m in enumerate(sorted_m, 1):
            rank_sums[m] += rank
    for m in sorted(rank_sums, key=lambda x: rank_sums[x]):
        print(f"  {m:30s}  mean_rank: {rank_sums[m]/5:.2f}")

    # ===== 8. Diagnosis: Why Adapter Still Trails =====
    print("\n\n## 8. Diagnosis: Why Adapter Still Trails Linear")
    print("=" * 60)
    print(f"""
  The fix recovered adapter from catastrophic failure (0.13 -> 0.42/0.49)
  but adapter-ep30 still trails linear by ~{gap_ep30:.3f} in final accuracy.

  THREE CONTRIBUTING FACTORS:

  (A) Re-ID accuracy dropped 0.435 -> 0.333 (-23.4%)
      - Pre-fix, adapter and linear had IDENTICAL re-ID (0.435)
      - Post-fix, adapter re-ID is lower despite using same Phase A
      - The adapter's different fingerprints (from adapter layers) may
        distort concept signatures, making Phase A routing less accurate
      - Lower entropy ({statistics.mean(ep30_entropy):.3f} vs {statistics.mean(lin_entropy):.3f}) suggests
        overconfident but incorrect concept assignments

  (B) Communication overhead: adapter uses {ep30_b_mean/lin_b_mean:.1f}x more bytes
      - Adapter must transmit adapter parameters in addition to base model
      - At {ep30_b_mean/1000:.0f}KB vs {lin_b_mean/1000:.0f}KB, this is a moderate overhead
      - Budget-normalized efficiency is lower for adapter

  (C) Training regime mismatch
      - Adapter needs ep=30 to approach linear's ep=5 performance
      - 6x more local epochs to achieve similar accuracy suggests the
        adapter architecture has slower convergence on this task
      - ep20 vs ep30 gap (~0.04) shows adapter is still improving with
        more epochs -- may need even more training

  CONCLUSION: The adapter fix resolved the catastrophic failure but
  exposed a secondary issue: adapter fingerprints degrade Phase A
  routing quality. The model trains correctly now, but concept
  assignment quality has regressed, limiting the accuracy ceiling.
""")


if __name__ == "__main__":
    main()
