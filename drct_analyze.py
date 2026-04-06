from __future__ import annotations
import csv
import numpy as np
from scipy import stats

rows = list(csv.DictReader(open("drct_falsification_timeseries.csv")))
trained = [r for r in rows if int(r["epoch"]) >= 2]  # skip untrained + transient epoch 1

by_tag: dict[str, list[dict]] = {}
for r in trained:
    by_tag.setdefault(r["tag"], []).append(
        {"r_sigma": float(r["r_sigma"]), "r_g": float(r["r_g"]), "rho": float(r["rho"])}
    )

print("=== Claim 1: r^G vs r_Sigma (paired, all trained epochs, all concepts) ===")
all_rs = np.array([v["r_sigma"] for tag in by_tag for v in by_tag[tag]])
all_rg = np.array([v["r_g"] for tag in by_tag for v in by_tag[tag]])
print(f"n={len(all_rs)}  mean r_sigma={all_rs.mean():.3f}  mean r_g={all_rg.mean():.3f}")
diff = all_rg - all_rs
print(f"paired diff mean={diff.mean():.3f}  std={diff.std(ddof=1):.3f}  "
      f"relative={(all_rg/all_rs).mean():.3f} (mean of r_g/r_sigma)")
t, p = stats.ttest_rel(all_rg, all_rs)
print(f"paired t-test: t={t:.2f}  p={p:.2e}")
w, p = stats.wilcoxon(all_rg, all_rs)
print(f"wilcoxon signed-rank: W={w:.1f}  p={p:.2e}")

print()
print("=== Claim 2: temporal stability of rho within concept ===")
for tag, vs in by_tag.items():
    rhos = np.array([v["rho"] for v in vs])
    cv = rhos.std(ddof=1) / rhos.mean()
    print(f"[{tag}] n={len(rhos)}  rho mean={rhos.mean():.3f}  std={rhos.std(ddof=1):.3f}  "
          f"CV={cv:.3f}  range=[{rhos.min():.3f},{rhos.max():.3f}]")
    if len(rhos) >= 6:
        early, late = rhos[:3], rhos[-3:]
        t, p = stats.ttest_ind(late, early, equal_var=False)
        print(f"       early(mean={early.mean():.3f}) vs late(mean={late.mean():.3f})  "
              f"Welch t={t:.2f}  p={p:.3f}")

print()
print("=== Claim 3: cross-concept discrimination (ANOVA on last-3 rho) ===")
groups, tags = [], []
for tag, vs in by_tag.items():
    rhos = np.array([v["rho"] for v in vs[-3:]])
    groups.append(rhos); tags.append(tag)
    print(f"[{tag}] last3 rho: {rhos.round(3)}")
f, p = stats.f_oneway(*groups)
print(f"one-way ANOVA: F={f:.2f}  p={p:.3f}")
for i in range(len(groups)):
    for j in range(i + 1, len(groups)):
        t, p = stats.ttest_ind(groups[i], groups[j], equal_var=False)
        print(f"  {tags[i]} vs {tags[j]}: Welch t={t:.2f}  p={p:.3f}")

print()
print("=== Batch-level noise floor at fixed model (experiment B) ===")
brows = list(csv.DictReader(open("drct_falsification_batch_var.csv")))
brho = np.array([float(r["rho"]) for r in brows])
print(f"n={len(brho)}  mean={brho.mean():.3f}  std={brho.std(ddof=1):.3f}  "
      f"CV={brho.std(ddof=1)/brho.mean():.3f}")

# Noise floor vs cross-concept effect size.
print()
print("=== Effect-size comparison ===")
pooled_within_std = np.sqrt(np.mean([g.var(ddof=1) for g in groups]))
means = np.array([g.mean() for g in groups])
between_std = means.std(ddof=1)
print(f"pooled within-concept std(rho, last3)={pooled_within_std:.3f}")
print(f"between-concept std of mean rho={between_std:.3f}")
print(f"batch-noise CV={brho.std(ddof=1)/brho.mean():.3f}")
print(f"between-concept CV over means={between_std/means.mean():.3f}")
