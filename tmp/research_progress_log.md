# CIFAR-100 Research Progress Log

> Auto-updated by agent team. Check this file for overnight progress.
> Last updated: 2026-03-20 09:20

---

## CRITICAL CORRECTIONS (Deep Diagnosis Phase)

### 之前大量结论被推翻，原因是两个 bug + 一个指标假象

**Bug 1: Oracle 严重欠训练**
- Oracle 用 n_epochs=1 + partial_fit（1 次梯度更新），其他方法用 n_epochs=5~10 + fit
- 修复后 Oracle: 0.15 → 0.458 (T=6 smoke), 0.12 → 0.619 (T=20 single seed)
- 影响: H4 "Oracle 永远不如 CFL" 的结论无效——89.5% 的差距来自 epoch 不匹配

**Bug 2: Adapter 零初始化缺失**（已修复，这个是真实 bug）
- _AdapterBlock 的 up 投影用默认 Kaiming 初始化而非零初始化
- 修复后 adapter: 0.136 → 0.416

**指标假象: Adapter entropy "坍缩"到 0.170 不是后验坍缩**
- Adapter 用 federation_every=5（4 次联邦），linear 用 federation_every=2（10 次联邦）
- assignment_entropy 对 ALL K*T cells 取平均，包括零填充的非联邦步
- Adapter 20% 的 cell 有值 → 平均 entropy = 0.170; Linear 50% → 0.556
- **每联邦步的真实 entropy: adapter=0.858, linear=1.128, 差距远小于表面**
- 控制实验证明: 同样 fed=5 时，adapter 和 linear entropy 完全相同 (0.171674)
- 4 轮实验（H2, H2c, H2d, H2h）全在追一个幻影

### 之前被错误推翻的假说
- ~~"Adapter 与 loss-based 概念检测结构性不兼容"~~ → 不对，是 federation_every 不同
- ~~"概念特化在 CIFAR-100 上有害"~~ → 部分正确但原因不同（见下）
- ~~"Oracle 在所有 n_samples 都只有 0.12"~~ → Oracle 实现 bug

### 真正成立的发现
1. **CIFAR-100 recurrence 的所有概念共享相同的 20 类标签分布** — 概念只在视觉风格上不同（灰度、模糊、色调等），不在类别分布上不同
2. **因此概念特化确实没有优势** — 但原因是数据集设计（标签无差异），不是"概念特化本身有害"
3. **55% 的 (k,t) 位置是 singleton**（某概念只有 1 个客户端），Oracle 在这些位置 = LocalOnly
4. **匹配 n_epochs 后 Oracle 与 CFL 差距仅 0.44%** — 几乎不存在的差距
5. **FedProTrack re-ID 优势 (2-3x vs IFCA) 是真实的**，在所有 T 值上成立
6. **Adapter 零初始化是真实 bug** — 修复有效

---

## 当前实验历史

### Round 1-6: 见上方修正（很多结论需要重新解读）

### Round 7: Oracle 修复 + 重跑 smoke
- Oracle (fixed, ep=5): 从 rank #20 升到 rank #7, acc=0.458
- CFL 仍领先 (0.625)，但差距大部分来自 n_epochs 不匹配

### Round 8: 深层诊断（当前）
- **Scientist**: Entropy 0.170 是 federation_every 不同导致的指标稀释，不是后验坍缩
- **Analyst**: Oracle-CFL 差距 89.5% 来自 epoch 不匹配；所有概念共享相同标签分布

---

## 接下来需要做的
1. ~~**重新设计 CIFAR-100 实验** — 概念需要有不同的标签分布才能测试概念感知方法~~ DONE (label_split="disjoint")
2. ~~**修复 entropy 指标** — 只在联邦步计算，不对零填充 cell 取平均~~ DONE
3. **实现 posterior-gated fallback** (H6) — 不确定时回退到全局模型
4. **重新评估论文叙事** — misattribution cost model 是新的理论贡献

## 教训
- 看到"byte-identical 结果"要追问代码路径，不要接受"结构性不兼容"
- 检查 baseline 实现是否正确，再下结论
- 不同方法用不同超参比较时，指标可能不可比

---

## Round 9: H5 + Phase Diagram (2026-03-20)

### Infrastructure improvements (committed)
1. **`_infer_n_features` fix in baselines.py**: `gc.generator_type` crashes on CIFAR configs;
   now uses `getattr(gc, "generator_type", None)` so FedAvg/Oracle baselines work with
   CIFAR100RecurrenceConfig.
2. **`generate_concept_matrix_low_singleton`** integrated into
   `fedprotrack/drift_generator/concept_matrix.py` (moved from standalone script).
3. **`min_group_size` parameter** added to `CIFAR100RecurrenceConfig`. When >1,
   post-processes concept matrix column-by-column to ensure every concept at each
   time step has at least that many clients. Eliminates singletons.
4. Updated `run_fpt_advantage.py` and `run_phase_diagram.py` with `--min-group-size` flag.

### H5: REFUTED (5 seeds, K=10, T=30, disjoint, min_group_size=2)

| Method | Final Acc | Re-ID |
|--------|-----------|-------|
| Oracle | 0.789 | 1.000 |
| CFL | 0.788 | N/A |
| FedAvg | 0.734 | N/A |
| **FPT** | **0.682** | **0.739** |
| IFCA | 0.330 | 0.939 |

**FPT trails CFL by 10.5% and even FedAvg by 5.2%.** Root cause: 27% concept
misattribution rate. With disjoint labels, wrong routing = predictions on wrong
classes = ~0% accuracy on those samples.

**Misattribution cost model**: E[acc] = 0.73 * 0.80 + 0.27 * 0.0 = 0.584.
Observed: 0.588. Near-exact match.

**Over-spawning is NOT the cause**: tightening max_concepts from 6 to 4 produces
identical accuracy. All configs converge to 4 active concepts.

### Phase Diagram (3 splits x 3 seeds x 5 methods)

| Method | shared | overlapping | disjoint |
|--------|--------|-------------|----------|
| Oracle | 0.484 | 0.648 | 0.796 |
| CFL | 0.558 | 0.647 | 0.791 |
| FedAvg | 0.507 | 0.587 | 0.739 |
| FPT | 0.462 | 0.541 | 0.676 |
| IFCA | 0.298 | 0.324 | 0.344 |

**Key finding**: FPT's accuracy gap vs CFL is ~10% at ALL heterogeneity levels.
The gap does not shrink with more heterogeneity. This is structural.

**FPT re-ID is label-agnostic**: 0.731 (shared), 0.716 (overlap), 0.729 (disjoint).
Concept tracking works independently of label composition.

### Critical diagnostic: Gibbs posterior is the bottleneck, not fingerprints

A simple KNN classifier on raw data fingerprints achieves **1.000 re-ID** on the
exact same data where FedProTrack's Gibbs posterior achieves only 0.723. The
fingerprints are perfectly discriminative (within-concept cosine sim 0.999,
across-concept 0.602). The posterior inference is throwing away 27.7% of the
available information.

Blend alpha sweep: increasing blend_alpha from 0.0 to 0.8 monotonically decreases
accuracy (0.671 -> 0.613) because it increases model inertia, not fallback.
Re-ID stays constant at 0.723 across all blend values.

### Next step: H6 (posterior-gated fallback)
When the posterior is uncertain (high entropy), fall back to global FedAvg model.
Breakeven re-ID for disjoint = 0.79; FPT has 0.73. Uncertain assignments should
use the global model to avoid misattribution penalty.

### Alternative next step: Fix the posterior inference directly
Since fingerprints are perfectly discriminative but the Gibbs posterior only achieves
0.723 re-ID, the most impactful improvement would be fixing how the posterior
processes fingerprints. The `similarity_calibration` feature was meant to help but
the calibration may not be aggressive enough.
