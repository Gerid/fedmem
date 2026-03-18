# CIFAR-100 实验发现与诊断报告

> 最后更新: 2026-03-18
> 状态: 诊断完成，待决策下一步方向

---

## 1. 实验背景

目标：验证 FedProTrack 在真实高维数据（CIFAR-100, ResNet-18 提取 128 维特征）上的有效性。

### 数据设置
- **特征**：ResNet-18 pretrained → 128 维 feature vector（torchvision 提取，非端到端训练）
- **概念构造**：将 CIFAR-100 的 100 个类分成 N 个不重叠子集（每个子集 5-20 个类），每个子集 = 1 个概念，类内标签重映射到 0..n_classes_per_concept
- **模型**：线性分类器（128→n_classes），SGD 训练

---

## 2. 关键发现

### 发现 1: 训练预算公平性问题（已修复）
- **问题**：初始 CIFAR-100 比较中 IFCA 使用 lr=0.01/epochs=1，而 FPT 使用 lr=0.05/epochs=5，训练预算不对等
- **修复**：统一 lr=0.05, epochs=5
- **影响**：修复后 IFCA 性能大幅提升，FPT 原本的优势消失

### 发现 2: 通信预算公平性
- **问题**：FPT 每轮只传 fingerprint（~1KB），IFCA 传全部 cluster 模型（~20KB/cluster）
- **正确比较**：应在 matched total bytes 下比较（budget frontier）
- **结果**：FPT 在低预算区间有优势（同等 bytes 下精度更高），但这主要来自"少聚合=少负干扰"，而非概念感知能力

### 发现 3: FedAvg 在高频通信下反而最好
| 方法 | FinalAcc | Bytes |
|------|----------|-------|
| FedAvg f=1 | **0.424** | 1.7M |
| FPT f=1 | 0.398 | 2.3M |
| IFCA f=1 | 0.413 | 5.2M |
| LocalOnly | 0.364 | 0 |

**解读**：概念异质性不够强时，FedAvg 的全局聚合反而最优。概念感知方法的"精准聚合"收益被"错误分配"的代价抵消。

### 发现 4: Oracle 上界很低
- Oracle f=2 精度 0.862 vs LocalOnly 0.726 → 聚合收益仅 +13.6pp
- K=12 clients, 8 concepts → 同概念平均 1.5 个 client → 聚合数据量增益有限
- **含义**：即使完美的概念感知也只能提升 ~14pp，而错误的概念分配可能抹掉全部收益

---

## 3. Recurrence-with-Gap 实验（FPT 理论优势场景）

### 实验设计
- K=12 clients, T=30 steps, 8 total concepts
- Phase 1 (t=0-9): concepts 0,1,2 active
- Phase 2 (t=10-19): concepts 3,4,5 active (0,1,2 dormant)
- Phase 3 (t=20-29): concepts 0,1,2 RECUR
- 核心指标: Recovery@t=20（概念复现时的即时恢复能力）

### 结果
| Method | Final | Phase3 | Recov@20 | Bytes |
|--------|-------|--------|----------|-------|
| LocalOnly | 0.726 | 0.652 | 0.314 | 0 |
| FedAvg f=2 | 0.598 | 0.481 | 0.264 | 867K |
| IFCA-3 f=2 | 0.759 | 0.683 | 0.258 | 1.7M |
| IFCA-8 f=2 | **0.806** | **0.725** | 0.257 | 3.9M |
| FPT f=2 | 0.776 | 0.680 | 0.300 | 1.1M |
| Oracle f=2 | 0.862 | 0.862 | **0.846** | 858K |

### 关键观察
1. **Recovery@t=20 所有方法都很低**（0.26-0.31），Oracle 0.85。说明没有任何方法成功利用了概念复现的记忆
2. **IFCA-8 最终精度最高**（0.806），因为 8 clusters 覆盖了所有概念，loss-based selection 更准确
3. **FPT 通信量最低**（1.1M vs IFCA-8 的 3.9M），但精度不如 IFCA-8

---

## 4. 根因分析

### 4.1 Over-spawning（核心问题）
- FPT 在 30 步内 spawn 了 9-16 个概念，实际只有 8 个
- Merge 几乎没有触发（0-2 次）
- 记忆库被噪声概念污染，matching 准确率极低

### 4.2 为什么 over-spawn？
在 128 维空间中，ConceptFingerprint 的 Hellinger similarity 不够稳定：
- 每步仅 200 个样本，5 个类 → 每类 40 个样本
- 128 维的 class-conditional mean 估计方差大
- 相邻时步的 fingerprint 相似度波动 → 频繁被判为"novel concept"

### 4.3 三种失败模式映射（Insight 2）
| 失败模式 | 在 CIFAR-100 上的表现 | 严重程度 |
|----------|----------------------|----------|
| 识别失败（δ_id 高）| fingerprint matching 在 128d 中不准 | **主导** |
| 覆盖失败（p_cov 低）| 概念被 shrink 或被 spawn 挤掉 | 中等 |
| 协议失败（ε_gate 大）| Phase A bytes 占比不高 | 较轻 |

### 4.4 尝试过的修复及结果
| 修复 | 效果 | 原因 |
|------|------|------|
| Dormant model preservation | 无效 | 概念没被 prune，是 over-spawn 问题 |
| Server-side model probing | 更差 | 10+ 噪声概念中选错概率更高 |
| Tight spawning (loss_novelty=0.5) | 无效 | 仍然 spawn 14-16 次 |
| Aggressive merging (merge_th=0.6) | 微弱 | merge 只 0-2 次 |

---

## 5. 结论与方向

### 核心结论
> **FPT 的 fingerprint-based concept tracking 在高维（128d）真实数据上不 work。**
>
> 这不是调参问题，是 representation quality（φ）的根本性不足。在 φ 质量跨越相变点之前，记忆系统性有害。

### Insight 1 验证
- 相变点确实存在：re-ID 精度在 CIFAR-100 上远低于阈值，记忆有害
- 在合成数据（2D SINE）上 re-ID=0.797 > 阈值，记忆有益

### Insight 3 启示
要让 FPT 在高维数据上 work，需要：
1. **学习低维概念 embedding**（contrastive learning / autoencoder）替代手工 fingerprint
2. 或**用 model prediction 作为概念信号**（但这趋近于 IFCA）
3. 或**承认适用域限制**：FPT 适用于 φ 质量足够高的场景（低维/强概念差异）

### 论文叙事建议
1. **不要声称 FPT 在所有场景下优于 IFCA**
2. 强调理论贡献：相变点理论 + 三种失败模式分类
3. 合成数据验证理论预测
4. CIFAR-100 作为"失败模式诊断"的案例研究——展示当 φ 质量不足时理论预测的失败模式如何精确对应实验结果
5. 指出 learned φ 作为 future work

### FPT 真正的差异化价值（vs IFCA）
| 维度 | IFCA | FPT (理论) | FPT (CIFAR-100 实际) |
|------|------|-----------|---------------------|
| 概念匹配 | loss-based (强) | fingerprint (弱) | 失败 |
| 通信量 | O(K_clusters × model) | O(fingerprint) | 优势 |
| 概念数动态调整 | 固定 | 动态 spawn/merge | over-spawn |
| 概念复现记忆 | 无 | 有 (理论) | 未能触发 |
| 适用域 | 广 | 窄 (需要好的 φ) | — |

---

## 6. 代码变更记录

### 新增/修改文件
- `fedprotrack/posterior/memory_bank.py`: 添加 `preserve_dormant_models` 配置项
- `fedprotrack/posterior/fedprotrack_runner.py`: 添加 `dormant_recall` 参数 + server-side model probing 逻辑
- `run_cifar100_recurrence_gap.py`: recurrence-with-gap 实验脚本
- `run_cifar100_budget_matched.py`: budget frontier 比较脚本

### 测试状态
- 72/72 相关测试通过（test_fedprotrack_runner + test_posterior + test_two_phase）

---

## 7. 待决策

- [ ] 方向 A: 实现 learned φ（contrastive concept embedding），让 FPT 在高维数据上 work
- [ ] 方向 B: 承认适用域限制，论文聚焦理论 + 合成数据 + 诊断框架
- [ ] 方向 C: 混合方案——Phase A 用 IFCA 式 loss-based selection + FPT 的 dormant memory，看是否能兼得两家之长
