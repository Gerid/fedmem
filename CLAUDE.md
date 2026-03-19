# CLAUDE.md

Compatibility note: public Phase 3 / E6 / CIFAR comparison entrypoints now expose explicit `FedProTrack` variants (`legacy`, `plan_c_linear`, `plan_c_feature_adapter`) and default to `legacy` so `main` preserves the original FedProTrack semantics.
Plan C note: recurrence / overlap research entrypoints may still opt into the Plan C or adapter presets explicitly; these are no longer the silent default for the public benchmark scripts.
Recurrence benchmark note: recurrence-style CIFAR subset ablations may explicitly set `global_shared_aggregation=False` for feature-adapter runs; this is a benchmark-specific knob, not a global preset default.
Feature-adapter note: routed local training is available as an experimental ablation knob and is off by default; do not treat it as a validated default path.
Stage 2 model note: a prototype `factorized_adapter` model type now exists in the shared model stack and runner, and recurrence / overlap registries expose it as the opt-in `factorized_local_shared`, `factorized_weighted_freeze`, `factorized_anchor_freeze`, and `factorized_top2_freeze` Stage 2 paths; current benchmark defaults still remain on the feature-adapter path.
Stage 1 screening note: `run_cifar100_recurrence_gap.py` now implements the adapter rescue screening workflow (`seed=42` gate, 3-seed decision table, root-cause summary) and writes `results.json`, `recurrence_selection_table.json/csv`, and `root_cause_summary.txt`.
Overlap veto note: `run_cifar100_label_overlap.py` now compares selected adapter variants on overlap points and writes `raw_results.csv`, `overlap_veto_table.json/csv`, and `root_cause_summary.txt`.
Runner policy note: adapter rescue experiments now use explicit `fingerprint_source`, `expert_update_policy`, and `shared_update_policy` knobs plus `shared_drift_norm`, `expert_update_coverage`, and `multi_route_rate` diagnostics; factorized Stage 2 runs additionally support `factorized_slot_preserving` with primary/secondary expert anchor alphas.
Factorized consolidation note: Stage 2 factorized ablations may also set `factorized_primary_consolidation_steps` to run a short primary-slot-only local tail after multiroute anchor/freeze updates.
Head-only consolidation note: factorized consolidation ablations may set `factorized_primary_consolidation_mode="head_only"` to restrict that tail to the primary slot head while keeping shared/private frozen.
Sharpened-read note: Stage 2 factorized ablations may also expose `routed_read_top_k` / `routed_read_temperature` to sharpen routed read-time mixtures without changing the write path.
Conditional sharpened-read note: Stage 2 factorized ablations may further gate read sharpening with `routed_read_only_on_ambiguity`, `routed_read_min_entropy`, `routed_read_min_secondary_weight`, and `routed_read_max_primary_gap` so only ambiguous posterior mixtures are sharpened.
Phase A hysteresis note: Stage 2 factorized ablations may also set `novelty_hysteresis_rounds` to require repeated novel evidence before a new concept is spawned.
Fingerprint rescue note: feature-adapter recurrence ablations now distinguish `model_embed`, `pre_adapter_embed`, `hybrid_raw_pre_adapter`, `weighted_hybrid_raw_pre_adapter`, `centered_hybrid_raw_pre_adapter`, `attenuated_hybrid_raw_pre_adapter`, `double_raw_hybrid_pre_adapter`, and `bootstrap_raw_hybrid_pre_adapter`, and recurrence runs also emit per-round Phase A `fp_loss` diagnostics.
Bootstrap fingerprint note: `bootstrap_raw_hybrid_pre_adapter` keeps Phase A on raw-only padded fingerprints while the memory bank has at most one concept, then switches to `raw + pre_adapter` features.
Autonomy note: `run_adapter_research_loop.py` provides a resumable scaffold-mutation loop that screens candidates, compares them against the current champion, and keeps/reverts variants without blindly editing core model code.

## 项目身份

**FedProTrack** — Federated Proactive Concept Identity Tracking

面向 NeurIPS 2026 投稿的研究项目，核心问题是：在联邦概念漂移下，服务器如何在通信受限条件下推断潜在概念身份（latent concept identity），以及记忆何时有益、何时会因误归属而有害。

**当前状态**：合成漂移管线、Gibbs 后验 / 两阶段协议、动态记忆库、Phase 4 记忆修复、扩展 baseline 套件、Rotating-MNIST 与 CIFAR-100 真实数据 benchmark 都已落地。当前重点是维持 matched-budget 对比、真实数据实验和项目元数据的一致性。当前测试树用 `pytest --collect-only` 可收集 381 个测试。

**技术栈**：
- Python 3.10+（Windows 环境优先用 `conda run -n base python`）
- 核心依赖：numpy >= 1.24、river >= 0.23、scikit-learn >= 1.3、scipy >= 1.10、matplotlib >= 3.7、pyyaml >= 6.0、torch >= 2.1
- 真实数据依赖：torchvision >= 0.16（Rotating-MNIST / CIFAR-100），缓存目录为 `.mnist_cache/`、`.cifar100_cache/`、`.feature_cache/`
- CUDA：由当前 PyTorch 安装决定；设备路由统一走 `fedprotrack/device.py`，可用 `FEDPROTRACK_FORCE_CPU=1` 强制 CPU，用 `FEDPROTRACK_GPU_THRESHOLD=0` 总是优先 GPU
- 测试：pytest >= 7.0，`slow` marker 用于可能下载或构建真实数据缓存的测试

**架构概览**：`fedprotrack/` 现在包含 10 个核心子模块——`drift_generator/` 生成可控的 K×T 概念矩阵与流数据；`drift_detector/` 封装 River 的 ADWIN / PageHinkley / KSWIN；`concept_tracker/` 负责概念身份追踪；`federation/` 提供基础聚合逻辑；`metrics/` 与 `evaluation/` 负责论文指标和可视化；`posterior/` 实现 Gibbs 后验、动态记忆库、retrieval-key / memory-slot 基础和两阶段 / soft aggregation 的 FedProTrack 主流程；`baselines/` 维护 17 个 matched-budget baseline（FedAvg-Full、FedProto、TrackedSummary、FedCCFA、Flash、FedDrift、IFCA、CFL、FeSEM、FedRC、FedEM、pFedMe、APFL、ATP、FLUX、FLUX-prior、CompressedFedAvg）；`models/` + `device.py` 提供 Torch 线性模型、feature-adapter 模型与 CPU/CUDA 设备管理；`real_data/` 提供 Rotating-MNIST 和 CIFAR-100 recurrence 数据集；`experiment/` / `experiments/` 输出 Phase 3/4 分析、表格、label-overlap failure diagnosis 辅助构件和可视化。主要入口脚本是 `run_experiments.py`、`run_phase3_experiments.py`、`run_cifar100_comparison.py`、`run_cifar100_budget_matched.py`、`run_cifar100_label_overlap.py`，CLI 入口是 `fedprotrack/cli.py`。

---

## 开发规范

### 编码约定

**文件头**：每个 `.py` 文件的第一行必须是 `from __future__ import annotations`。

**文档字符串**：NumPy 风格，包含 Parameters / Returns / Raises 节。

```python
def budget_normalized_score(
    accuracy_curve: np.ndarray,
    total_bytes: float,
) -> float:
    """Compute communication-budget-normalised accuracy AUC.

    Parameters
    ----------
    accuracy_curve : np.ndarray
        Array of shape (K, T) with per-client per-step classification accuracy.
    total_bytes : float
        Total communication budget in bytes consumed by the algorithm.

    Returns
    -------
    float
        AUC of mean accuracy divided by ``total_bytes``.

    Raises
    ------
    ValueError
        If ``total_bytes`` <= 0.
    """
```

**类型注解**：使用 Python 3.10+ 的 `|` 联合语法（`float | None`），不用 `Optional`。

**数据容器**：优先用 `@dataclass`，并在 numpy 数组边界显式约定 dtype。

```python
@dataclass
class DriftResult:
    is_drift: bool
    is_warning: bool = False
    detector_name: str = ""
```

**随机种子**：数据采样和概念轨迹遵循确定性递增规则，优先使用 `seed + k * T + t + 10000` 这类可推导公式，禁止依赖全局随机状态。

**抽象基类模式**：共享接口优先定义抽象基类并提供 `clone()` / `_init_kwargs()` 之类的最小复用点。

### 错误处理模式

- `@dataclass` 配置类在 `__post_init__` 中尽早抛出 `ValueError` 拒绝非法参数。
- 统计量不足时返回安全默认值，而不是在内部路径上无意义地抛异常。
- 业务层输入错误要显式抛出 `ValueError`，并附带具体值。
- 不使用裸 `except`；不吞异常。

### 测试要求

- 测试文件位于 `tests/`，命名为 `test_<module>.py`。
- 完整测试命令：`conda run -n base python -m pytest tests/ -v`
- 快速校验命令：`conda run -n base python -m pytest tests/ -m "not slow" -v`
- 研究摘要管理命令：`conda run -n base python manage_findings.py save/list/show/promote ...`
- 当前测试树用 `pytest --collect-only` 可收集 381 个测试；`slow` 测试可能下载 MNIST / CIFAR-100 或构建特征缓存。任何改动后都必须保持测试为绿。
- 新增功能必须附带测试，至少覆盖正常路径、边界条件和异常输入。
- River 0.23 中 `drift.DDM` 已被移除，**不要使用**，用 `drift.PageHinkley` 替代。

---

## 当前优先事项

### 当前迭代目标（Phase 4 稳定化 + 真实数据实验）

1. **稳定 FedProTrack 主流程**：保持 Gibbs 后验、模型记忆、两阶段通信和 soft aggregation 在合成数据与真实数据上行为一致。
2. **维护 17 方法 matched-budget 对比**：新增 baseline 时必须同步更新 `baselines/`、`experiments/method_registry.py`、预算脚本和指标解释。
3. **校准真实数据 benchmark**：持续维护 Rotating-MNIST 与 CIFAR-100 recurrence 的缓存、预算设置、入口脚本和结果可复现性。
4. **保持元数据同步**：当依赖、数据集、CUDA 路径或脚本入口变化时，同步更新 `AGENTS.md`、`CLAUDE.md`、`pyproject.toml` 和测试配置。

### 已知要规避的问题

- **River 兼容性**：River 0.23 移除了 `drift.DDM`，只能用 ADWIN / PageHinkley / KSWIN。
- **Windows 环境**：`python` / `python3` 可能指向 Windows Store，优先用 `conda run -n base python`。
- **PyTorch CUDA 安装**：`pyproject.toml` 只声明 `torch` 依赖，是否启用 CUDA 取决于环境里安装的 PyTorch build，变更环境后必须重新确认 `torch.cuda.is_available()`。
- **Windows + MKL / KMeans warning**：`sklearn` 在 Windows 上可能提示 KMeans memory leak warning；如需压制该环境问题，可在运行前设置 `OMP_NUM_THREADS=1`。
- **实验可比性**：所有 baseline 必须在 matched-budget 条件下公平比较（total bytes，而不是 per-round）。
- **真实数据缓存**：`.mnist_cache/`、`.cifar100_cache/`、`.feature_cache/` 会显著影响首次运行耗时，改动数据入口时必须考虑缓存命名和复用。
- **结论摘要沉淀**：agent 产出的 root-cause / finding / summary 不要只留在线程或 worktree 里，统一先落到共享 ledger，再把稳定结论 promote 到 `docs/findings/`。
- **代码膨胀**：任何改动必须服务于论文主线，不做泛化重构；优先复用现有 pipeline、config 和日志系统。
- **种子管理**：不要用全局 `np.random.seed()`；改动随机流程时必须保留可推导、可复现实验路径。

## Sync Notes (2026-03)

- `Oracle` baseline now means true-concept federated aggregation across clients, not per-client memorization.
- `FeSEM` remains an alias of `IFCA` in this repo; smoke comparison entrypoints should de-duplicate it unless alias behaviour is under study.
- Fast CIFAR-100 all-baselines smoke entrypoint: `python run_cifar100_all_baselines_smoke.py`.
- The smoke entrypoint also supports `--fpt-mode {base,auto,calibrated,hybrid,update-ot,labelwise,hybrid-labelwise,hybrid-proto,hybrid-proto-early,hybrid-proto-firstagg,hybrid-proto-subagg}`; `calibrated` re-scales novelty/merge thresholds from fingerprint quantiles, `hybrid` mixes a lightweight projected model-signature similarity into Phase A routing, `update-ot` mixes a lightweight OT similarity over projected classifier rows, `labelwise` mixes label-wise batch-prototype-to-classifier-row OT routing, `hybrid-labelwise` combines global model-signature routing with label-wise prototype routing, `hybrid-proto` applies prototype-aware classifier alignment after aggregation, `hybrid-proto-early` uses a stronger prototype-alignment mix only in the earliest federation round before falling back to the steady-state mix, `hybrid-proto-firstagg` pre-aligns client models to concept prototypes before the earliest aggregation round, and `hybrid-proto-subagg` tries subgroup-aware prototype pre-alignment before collapsing back to one concept-level aggregate.
- The smoke entrypoint now also emits `FedAvg-FPTTrain`, a fairness control that uses the same `--fpt-lr` and `--fpt-epochs` local training strength as FedProTrack.

### 最近完成的工作（避免重复）

- `drift_generator/`：可控合成漂移生成器（SINE / SEA / CIRCLE）和 K×T 概念矩阵生成器。
- `posterior/`：Gibbs 后验、动态记忆库、两阶段协议、soft aggregation、Phase 4 概念-模型记忆修复。
- `baselines/`：从早期的 3 方法扩展到 17 方法 matched-budget baseline 套件，并补齐 `runners.py`、`budget_sweep.py` 和 capability registry。
- `models/` + `device.py`：引入 `TorchLinearClassifier`、`TorchFeatureAdapterClassifier` 和统一 CPU / CUDA 设备管理。
- `real_data/`：新增 `rotating_mnist.py` 与 `cifar100_recurrence.py`，支持真实数据实验和特征缓存。
- 根目录脚本：新增 / 维护 `run_phase3_experiments.py`、`run_phase4_analysis.py`、`run_cifar100_comparison.py`、`run_cifar100_budget_matched.py`、`run_cifar100_recurrence_gap.py`、`run_cifar100_label_overlap.py`。
- 测试树：当前仓库 `pytest --collect-only` 收集 381 个测试项。
