# AGENTS.md

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
- `fedprotrack/models/`：`TorchLinearClassifier`
- `fedprotrack/experiments/` 与根目录 `run_*.py`：Phase 3/4 分析、预算比较、CIFAR-100 实验

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
