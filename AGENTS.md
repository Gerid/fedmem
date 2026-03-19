# AGENTS.md

Compatibility note: public Phase 3 / E6 / CIFAR comparison entrypoints now expose explicit `FedProTrack` variants (`legacy`, `plan_c_linear`, `plan_c_feature_adapter`) and default to `legacy` so `main` preserves the original FedProTrack semantics.
Plan C note: recurrence / overlap research entrypoints may still opt into the Plan C or adapter presets explicitly; these are no longer the silent default for the public benchmark scripts.
Recurrence benchmark note: recurrence-style CIFAR subset ablations may explicitly set `global_shared_aggregation=False` for feature-adapter runs; this is a benchmark-specific knob, not a global preset default.
Feature-adapter note: routed local training is available as an experimental ablation knob and is off by default; do not treat it as a validated default path.
Stage 2 model note: a prototype `factorized_adapter` model type now exists in the shared model stack and runner, and recurrence / overlap variant registries expose it as the opt-in `factorized_local_shared`, `factorized_weighted_freeze`, `factorized_anchor_freeze`, and `factorized_top2_freeze` research paths.
Stage 1 screening note: `run_cifar100_recurrence_gap.py` now implements the adapter rescue screening workflow (`seed=42` gate, 3-seed decision table, root-cause summary) and writes `results.json`, `recurrence_selection_table.json/csv`, and `root_cause_summary.txt`.
Overlap veto note: `run_cifar100_label_overlap.py` now compares selected adapter variants on overlap points and writes `raw_results.csv`, `overlap_veto_table.json/csv`, and `root_cause_summary.txt`.
Runner policy note: adapter rescue experiments now use explicit `fingerprint_source`, `expert_update_policy`, and `shared_update_policy` knobs plus `shared_drift_norm`, `expert_update_coverage`, and `multi_route_rate` diagnostics; factorized Stage 2 runs additionally support `factorized_slot_preserving` with primary/secondary expert anchor alphas.
Factorized consolidation note: Stage 2 factorized ablations may also set `factorized_primary_consolidation_steps` to run a short primary-slot-only local tail after multiroute anchor/freeze updates.
Head-only consolidation note: factorized consolidation ablations may set `factorized_primary_consolidation_mode="head_only"` to restrict that tail to the primary slot head while keeping shared/private frozen.
Sharpened-read note: Stage 2 factorized ablations may also expose `routed_read_top_k` / `routed_read_temperature` to sharpen routed read-time mixtures without changing the write path.
Conditional sharpened-read note: Stage 2 factorized ablations may further gate read sharpening with `routed_read_only_on_ambiguity`, `routed_read_min_entropy`, `routed_read_min_secondary_weight`, and `routed_read_max_primary_gap` so only ambiguous posterior mixtures are sharpened.
Phase A hysteresis note: Stage 2 factorized ablations may also set `novelty_hysteresis_rounds` to require repeated novel evidence before a new concept is spawned.
Fingerprint rescue note: feature-adapter recurrence ablations now distinguish `model_embed`, `pre_adapter_embed`, `hybrid_raw_pre_adapter`, `weighted_hybrid_raw_pre_adapter`, `centered_hybrid_raw_pre_adapter`, `attenuated_hybrid_raw_pre_adapter`, `double_raw_hybrid_pre_adapter`, and `bootstrap_raw_hybrid_pre_adapter`, and recurrence runs also emit per-round Phase A `fp_loss` diagnostics.
Bootstrap fingerprint note: recurrence and overlap scaffolds expose `bootstrap_raw_hybrid_pre_adapter`, which uses raw features padded with zeros while the memory bank has at most one concept, then switches to `raw + pre_adapter` features.
Autonomy note: `run_adapter_research_loop.py` provides a resumable scaffold-mutation loop that screens candidates, compares them against the current champion, and keeps/reverts variants without blindly editing core model code.

## 项目快照

FedProTrack 是一个面向联邦概念漂移与概念身份追踪的研究代码库。当前仓库已经包含合成漂移生成器、FedProTrack 的 Gibbs 后验 / 两阶段协议、17 方法 matched-budget baseline 套件，以及 Rotating-MNIST 与 CIFAR-100 recurrence 两个真实数据 benchmark。

## 当前技术栈

- Python 3.10+
- 核心依赖：`numpy`、`river>=0.23`、`scikit-learn`、`scipy`、`matplotlib`、`pyyaml`、`torch>=2.1`
- 真实数据依赖：`torchvision>=0.16`
- GPU 路由统一在 `fedprotrack/device.py`；`FEDPROTRACK_FORCE_CPU=1` 可禁用 CUDA，`FEDPROTRACK_GPU_THRESHOLD=0` 可总是优先 GPU
- `pyproject.toml` 声明了 `torch` 依赖，但是否真正启用 CUDA 取决于当前环境安装的 PyTorch build
- Windows + MKL 环境下，`sklearn` 的 KMeans 可能提示 memory leak warning；需要时可在运行前设置 `OMP_NUM_THREADS=1`

## 关键模块

- `fedprotrack/posterior/`：Gibbs posterior、dynamic memory bank、two-phase protocol、soft aggregation
- `fedprotrack/baselines/`：17 个 matched-budget baseline、通信计数、统一 runner
- `fedprotrack/real_data/`：`rotating_mnist.py`、`cifar100_recurrence.py`
- `fedprotrack/models/`：`TorchLinearClassifier`、`TorchFeatureAdapterClassifier`
- `fedprotrack/experiments/` 与根目录 `run_*.py`：Phase 3/4 分析、预算比较、CIFAR-100 实验，以及 label-overlap / failure-diagnosis 辅助入口

## 常用命令

- 完整测试：`conda run -n base python -m pytest tests/ -v`
- 快速测试：`conda run -n base python -m pytest tests/ -m "not slow" -v`
- 当前测试树：`pytest --collect-only` 可收集 381 个测试
- 安装真实数据依赖：`conda run -n base python -m pip install -e .[real-data,dev]`
- 保存摘要到账本：`conda run -n base python manage_findings.py save --title "<title>" --body-file summary.md`
- 检索摘要：`conda run -n base python manage_findings.py list` / `show <finding-id>` / `promote <finding-id>`

## 更新规则

当 baseline、数据集、设备路由、依赖或实验入口发生变化时，必须同时更新本文件、`CLAUDE.md` 和 `pyproject.toml`。
结论性摘要不要直接散落在当前 worktree；先写入共享 ledger，再把稳定结论 promote 到 `docs/findings/`。

## Sync Notes (2026-03)

- `fedprotrack/experiment/baselines.py` 中的 `Oracle` baseline 现在表示“按真实 concept 进行跨客户端聚合”的上界，而不是每客户端私有记忆。
- `FeSEM` 在当前仓库里仍是 `IFCA` 的实现别名；做 smoke / ranking 表时默认应去重，除非明确要检查 alias 行为。
- 小规模 CIFAR-100 全基线入口是 `python run_cifar100_all_baselines_smoke.py`。
- 该 smoke 入口支持 `--fpt-mode {base,auto,calibrated,hybrid,update-ot,labelwise,hybrid-labelwise,hybrid-proto,hybrid-proto-early,hybrid-proto-firstagg,hybrid-proto-subagg}`；`calibrated` 会按数据集的 fingerprint 相似度分位数重标定阈值，`hybrid` 混入轻量模型签名相似度，`update-ot` 混入投影 classifier-row 的轻量 OT 相似度，`labelwise` 混入 label-wise batch prototype 到 classifier-row 的轻量 OT 路由，`hybrid-labelwise` 组合全局模型签名与 label-wise prototype 路由，`hybrid-proto` 在聚合后做 prototype-aware classifier 对齐，`hybrid-proto-early` 则只在最早一轮 federation 使用更强的 prototype mix 后回落到稳态 mix，`hybrid-proto-firstagg` 则只在最早一轮 federation 先对客户端参数做 prototype-aware pre-alignment 再聚合，`hybrid-proto-subagg` 则先做 predictive subgroup 的 prototype-aware pre-alignment，再回落到单一 concept 聚合。
- 该 smoke 入口额外输出 `FedAvg-FPTTrain`，用于和 FedProTrack 使用相同本地训练强度（`--fpt-lr` / `--fpt-epochs`）的公平对照。
