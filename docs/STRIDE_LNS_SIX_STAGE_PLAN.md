# STRIDE-LNS 六阶段计划

`STRIDE-LNS` 是 V2/V3 之后的一条独立研究线，目标是在不改变初始化、候选池和 PP 修复器的前提下，选择当前一步更稳健、修复后结构更容易继续求解的邻域。

## 命名边界

- 研究线：`stride-lns`
- 同数据旧标签控制模型：`stride-control-v1`
- 新标签模型：`stride-quality-v1`
- 新标签模式：`lns2.stride.quality_label.v1`
- 产物目录：统一使用 `stride-` 前缀

这些名称不得作为 `v2-full`、`mixed-full-v2` 或 `v3-s3` 的别名。新模型通过正式门槛前不得替换默认控制器。

## 六个阶段

1. 冻结 V2/Mixed、124维特征、候选池和 PP，并审计历史数据能否支持新标签。
2. 收集240个独立状态，使用完整候选池和四个配对 PP 种子验证单步质量与修复后结构标签。
3. Pilot 通过后扩展到不少于600个状态、24张地图和约43,200次修复试验。
4. 在相同数据上对比冻结基线、旧标签重训控制和 STRIDE 新标签模型，再进行分组特征消融。
5. 先做 action-preserving Shadow，再做300秒完整预算 Quick 对照。
6. 在12张全新地图、36个配对实例和600秒预算下完成 MovingAI/OOD 正式评估。

Cost-to-Go、Receding-Q 和剩余修复轮数只允许作为诊断指标，不得作为 STRIDE 的主训练标签。完整晋级门槛以预注册实验配置为准。

## Stage 1 运行

```powershell
python scripts/run_stride_pipeline.py audit `
  --config configs/stride_stage1_audit.json `
  --output build/stride-stage1-audit-v1
```

报告写入 `stage1_audit_report.json`。`ready_for_stride_pilot` 表示冻结基线和历史数据审计有效；`requires_fresh_stride_collection=true` 表示旧数据缺少完整的逐候选配对 PP outcome 或修复后结构指标，Stage 2 必须重新采集。

Stage 2 的逐次修复数据完成后，使用独立标签入口：

Stage 2 原生采集必须在已通过 `runtime-wsl` 环境检查的 WSL 中运行：

```bash
PYTHONPATH=build/linux/project /usr/bin/python3 scripts/run_stride_pipeline.py collect \
  --selection build/stride-stage2-pilot-selection-v1.jsonl \
  --output build/stride-stage2-pilot-v1 \
  --workers 4
```

采集器对每个状态生成冻结的18候选完整池，对每个候选执行四个配对 PP 种子。每个状态原子落盘；中断后使用相同命令加 `--resume`，完成状态不会重跑。`collection_report.json` 必须满足 `complete=true`、`error_state_count=0` 且实际 trial 数等于候选数乘4。

```powershell
python scripts/run_stride_pipeline.py label `
  --trials build/stride-stage2-pilot-v1/repair_trials.jsonl `
  --output build/stride-stage2-labels-v1
```

该入口严格要求每个候选四个不同 PP 种子、124维冻结特征和完整修复后结构指标。输出正反训练对，每个状态的总样本权重均为1；运行时间不参与标签。

## Git 审查规则

在当前阶段分支实施。每个通过测试的阶段，以及标签、特征、数据分割、运行时接口或实验结论发生重大变化时，建立独立可审查提交并推送 GitHub。合并到 `main` 需要在正式评估完成后单独决定。
