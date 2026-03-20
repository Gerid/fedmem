from __future__ import annotations
import json, numpy as np

with open('tmp/cifar100_adapter/results.json') as f:
    data = json.load(f)

rows = data['rows']
methods = ['FPT-linear-base', 'FPT-adapter-base', 'FPT-adapter-hybrid-proto', 'CFL', 'IFCA']
print("=== Adapter Experiment (T=20, N=400, 5 seeds) ===")
for m in methods:
    subset = [r for r in rows if r['method'] == m]
    accs = [r['final_accuracy'] for r in subset]
    aucs = [r['accuracy_auc'] for r in subset]
    reids = [r['concept_re_id_accuracy'] for r in subset if r['concept_re_id_accuracy'] is not None]
    bytes_ = [r['total_bytes'] for r in subset]
    reid_str = f"{np.mean(reids):.4f}" if reids else "N/A"
    print(f"  {m}: acc={np.mean(accs):.4f}+/-{np.std(accs):.4f}, auc={np.mean(aucs):.4f}, re-id={reid_str}, bytes={np.mean(bytes_):.0f}")
