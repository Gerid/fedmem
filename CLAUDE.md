# CLAUDE.md

## 项目身份

**FedProTrack** — Federated Proactive Concept Identity Tracking

面向 NeurIPS 2026 投稿的研究项目，核心问题：在联邦概念漂移下，服务器如何在通信受限条件下推断潜在概念身份（latent concept identity）——记忆何时有益、何时因误归属而有害？

**当前状态**：Phase 1 已完成（合成漂移生成器 + 指标管线 + baseline 族 + matched-budget 对比框架）。141 个测试全部通过。正处于 Phase 0（文献精读）与 Phase 2（核心方法实现：Gibbs 后验 + 动态记忆库）之间。

**技术栈**：
- Python 3.10+（Windows 环境，**必须用** `conda run -n base python`，`python` / `python3` 是 Windows Store 重定向）
- numpy >= 1.24, river 0.23（仅用 `river.drift` 检测器和 `river.datasets.synth` 生成器）, scikit-learn >= 1.3, matplotlib >= 3.7, scipy >= 1.10, pyyaml >= 6.0
- 测试：pytest >= 7.0

**架构概览**：`fedprotrack/` 下 7 个子模块沿数据管线排列——`drift_generator/` 生成可控的 K×T 概念矩阵与流数据 → `drift_detector/` 封装 River 的 ADWIN/PageHinkley/KSWIN 做在线漂移检测 → `concept_tracker/` 用 ConceptFingerprint（增量均值/协方差/标签分布）+ ConceptTracker 做概念身份追踪 → `federation/` 实现 FedAvg 与 ConceptAwareFedAvg 聚合 → `metrics/` 计算五类论文指标（re-ID accuracy, assignment entropy, wrong-memory reuse, dip/recovery, budget-normalized score）并输出 phase diagram + budget frontier → `baselines/` 实现三种 matched-budget baseline（FedAvg-Full / FedProto / TrackedSummary）+ 通信字节计数 + budget sweep → `experiment/` 编排 runner、baselines（LocalOnly / FedAvg / Oracle）和可视化。入口脚本是 `run_experiments.py`，CLI 入口是 `fedprotrack/cli.py`（含 `budget-sweep` 子命令）。

---

## 开发规范

### 编码约定

**文件头**：每个 `.py` 文件的第一行必须是 `from __future__ import annotations`。

**文档字符串**：NumPy 风格，包含 Parameters / Returns / Raises 节：

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

**数据容器**：用 `@dataclass`，指定显式 numpy dtype：

```python
@dataclass
class DriftResult:
    is_drift: bool
    is_warning: bool = False
    detector_name: str = ""
```

**随机种子**：确定性递增，公式为 `seed + k * T + t + 10000`，禁止使用全局随机状态。

**抽象基类模式**（以 drift detector 为例）：

```python
class BaseDriftDetector(ABC):
    @abstractmethod
    def update(self, value: float) -> DriftResult: ...
    @abstractmethod
    def reset(self) -> None: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def _init_kwargs(self) -> dict: ...

    def clone(self) -> BaseDriftDetector:
        return self.__class__(**self._init_kwargs())
```

### 错误处理模式

- **`__post_init__` 校验**：`@dataclass` 的配置类在构造时立即用 `ValueError` 拒绝非法参数（见 `GeneratorConfig`）。
- **防御性返回**：统计量不足时返回安全默认值而非抛异常（如 `ConceptFingerprint.covariance` 在 count < 2 时返回单位矩阵，`similarity()` 返回 0.0）。
- **业务层 ValueError**：指标函数对不合理输入显式抛出（如 `total_bytes <= 0`、`T < 2`），附带具体值的 f-string 消息。
- 不使用裸 `except`；不吞异常。

### 测试要求

- 测试文件位于 `tests/`，命名 `test_<module>.py`。
- 运行命令：`conda run -n base python -m pytest tests/ -v`
- 当前 141 个测试全部通过，**任何改动后必须确保测试仍全部通过**。
- 新增功能必须附带测试。测试要覆盖：正常路径、边界条件、异常输入。
- River 0.23 中 `drift.DDM` 已被移除，**不要使用**，用 `drift.PageHinkley` 替代。

---

## 当前优先事项

### 当前迭代目标（Phase 2，目标 4 月 26 日前完成）

1. **实现 Gibbs 后验概念分配模块**——基于 Factorial HMM 的后验推断：$p(z_t^{(i)}=k) \propto \exp\{-\omega \cdot \ell(o, m_k)\} \cdot p(z_t^{(i)}=k \mid z_{t-1}^{(i)})$
2. **实现动态记忆库**——在线 EM 风格收缩更新 + spawn/merge 策略
3. **实现两阶段通信协议**——Phase A 轻量 prototype 交换做概念识别，Phase B 在已识别集群内做 full-model 聚合
4. 在 SINE 合成数据上端到端跑通 FedProTrack，与现有 baselines 产出可比结果

### 已知要规避的问题

- **River 兼容性**：River 0.23 移除了 `drift.DDM`，只能用 ADWIN / PageHinkley / KSWIN
- **Windows 环境**：`python` / `python3` 命令指向 Windows Store，必须用 `conda run -n base python`
- **实验可比性**：所有 baseline 必须在 matched-budget 条件下公平比较（total bytes 而非 per-round）
- **代码膨胀**：任何改动必须服务于论文主线，不做泛化重构；优先复用现有 pipeline、config 和日志系统
- **种子管理**：不要用全局 `np.random.seed()`，按 `seed + k*T + t + 10000` 确定性递增

### 最近完成的工作（避免重复）

- ✅ `drift_generator/`：可控合成漂移生成器（SINE/SEA/CIRCLE），支持 (rho, alpha, delta) 三轴 sweep，输出 K×T 概念矩阵 + ground truth concept ID
- ✅ `drift_detector/`：ADWIN、PageHinkley、KSWIN、NoDrift 四种检测器，统一 `BaseDriftDetector` 接口
- ✅ `concept_tracker/`：ConceptFingerprint（增量均值/协方差/标签分布 + Hellinger 相似度）+ ConceptTracker（新概念检测 / 复现检测）
- ✅ `federation/`：FedAvg + ConceptAwareFedAvg 聚合器
- ✅ `metrics/`：完整五类指标（concept_re_id_accuracy, assignment_entropy, wrong_memory_reuse_rate, worst_window_dip_recovery, budget_normalized_score）+ Hungarian 对齐 + phase diagram 构建与可视化
- ✅ `experiment/`：ExperimentRunner + 三族 baseline（LocalOnly / FedAvg / Oracle）+ 可视化管线
- ✅ `run_experiments.py`：完整实验入口，支持 `--quick` 模式
- ✅ `baselines/`：matched-budget baseline skeleton
  - `comm_tracker.py`：`model_bytes` / `prototype_bytes` / `fingerprint_bytes`（纯解析字节计数）
  - `fedproto.py`：FedProto 原文实现（nearest prototype 分类 + per-class 加权聚合）
  - `tracked_summary.py`：ConceptFingerprint cosine 聚类 → 组内 FedAvg
  - `budget_sweep.py`：`BudgetPoint`、`run_budget_sweep`（3 方法 × federation_every）、`find_crossover_points`
- ✅ `metrics/visualization.py`：新增 `plot_budget_frontier()`
- ✅ `cli.py`：新增 `budget-sweep` 子命令（`--dataset-dir / --federation-every-list / --output-dir`）
- ✅ 141 个测试全部通过
