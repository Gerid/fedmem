# Findings Ledger

这套机制用于保存 agent 产出的“结论性摘要”，而不是原始 `.npy` / `.json` / 图表结果。

## 为什么不用直接写在当前 worktree

worktree 适合临时代码修改，不适合作为长期研究记忆。摘要如果直接散落在各个 worktree 里，会出现三个问题：

1. 后续 agent 在别的 worktree 看不到。
2. 很难追溯“这个结论来自哪个 commit / 哪组实验 / 哪个脚本”。
3. 真正沉淀下来的结论和临时推断混在一起。

## 现在的落盘方式

- **共享草稿账本**：`git common dir/fedprotrack/findings/`
  所有 worktree 共用，适合保存刚生成的 finding / root-cause / 摘要。
- **版本化结论**：`docs/findings/`
  当一条 finding 已经足够稳定、值得随仓库提交时，再 promote 到这里。

## 推荐工作流

1. agent 完成实验和摘要后，先执行：

```bash
conda run -n base python manage_findings.py save \
  --title "CIFAR-100 fingerprint collapse causes repeated spawn" \
  --kind root-cause \
  --status validated \
  --dataset cifar100 \
  --tag cifar100 \
  --tag fingerprint \
  --artifact results_cifar100_budget/summary.json \
  --command "conda run -n base python run_cifar100_budget_matched.py --results-dir results_cifar100_budget" \
  --body-file summary.md
```

2. 后续在任何 worktree 里检索：

```bash
conda run -n base python manage_findings.py list --dataset cifar100
conda run -n base python manage_findings.py show <finding-id>
```

3. 当结论稳定后，promote 到版本控制目录：

```bash
conda run -n base python manage_findings.py promote <finding-id>
```

## 每条 finding 至少要包含什么

- 标题：一句话说明核心结论
- 结论正文：不是实验过程，而是“结论 + 证据 + 含义”
- 数据集 / baseline / 主题标签
- 对应脚本命令
- 对应 artifact 路径
- 当前 branch / commit（脚本会自动记录）

## 建议状态

- `draft`：刚生成，尚未复核
- `validated`：已经用实验或代码检查核过
- `final`：准备写进论文、报告或长期文档
