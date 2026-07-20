# 项目文件指南

当前分支已经收缩为“可运行求解器 + 冻结控制器 + 当前评测 + 正式证据”。历史模型训练、失败审计和旧 Stage 代码不再混在日常目录中；完整版本可从 Git 标签 `pre-minimal-runtime-2026-07-20` 恢复。

## 日常只看这些目录

| 目录 | 内容 | 什么时候需要看 |
| --- | --- | --- |
| `configs/` | 当前数据、采集、评测和维护配置。 | 修改实验参数时。 |
| `docs/` | 当前操作说明、研究结论和环境记录。 | 查协议与结论时。 |
| `experiments/` | 运行时、特征、候选生成、采集和分析实现。 | 修改算法或排查运行时。 |
| `generators/` | 合成地图与静态 OD 任务生成。 | 修改数据设计时。 |
| `include/`、`src/` | C++ 接口、官方内核包装、pybind 和原生特征。 | 修改求解器或性能代码时。 |
| `scripts/` | 稳定 CLI。 | 日常执行实验时。 |
| `tests/` | runtime、data、evaluation、maintenance 测试。 | 验证修改时。 |

Explorer 默认隐藏但必须保留：

| 目录 | 为什么保留 |
| --- | --- |
| `artifacts/` | 冻结 v1/v2 模型、schema 和紧凑证据。 |
| `third_party/` | 固定版本的官方 MAPF-LNS2 与 GPBS。 |
| `build/` | 本机数据、trace、虚拟环境和构建结果；不进入 Git。 |

## 活动控制器

- `v1-full`：冻结的 pairwise HistGradientBoosting 模型，输入当前动态冲突状态和实际 agent 邻域的 124 维特征。
- `v2-full`：与 v1 选择完全一致，使用紧凑树和加速特征实现。
- `v2-stall-safe`：在 v2 上增加冻结的停滞保护规则。
- 官方对照：Adaptive、Target、Collision、Random。

`v2-balanced`、`v2-cascade` 和 proposal pruner 没有通过推广门槛，已经退出活动代码。旧 bundle 的 pruner 字段仍可只读解析，以免历史 manifest 无法打开，但不会在当前控制器中执行。

## experiments 目录

| 文件 | 作用 |
| --- | --- |
| `closed_loop_confirmation.py` | 闭环候选生成、模型选择、修复执行和 trace。 |
| `online_feature_engine.py` | Python/native 在线特征统一接口。 |
| `state_analysis.py` | 状态、路径、冲突图和栅格分析。 |
| `neighborhood_features.py` | proposal 与 realized neighborhood 特征。 |
| `neighborhood_candidates.py` | 候选去重、代表邻域和确定性 seed。 |
| `context_audit.py` | 仅保留旧 sklearn pickle 所需的 `PairwiseModel` 兼容类。 |
| `compact_controller_model.py` | 紧凑 portable tree bundle 导出和读取。 |
| `stall_guard.py`、`stalled_state_probe.py` | stall-safe 判定与诊断。 |
| `closed_loop_trace_storage.py`、`closed_loop_trace_conversion.py` | 紧凑 trace 写入与迁移。 |
| `closed_loop_confirmation_analysis.py` | 同族闭环分析。 |
| `movingai_ood_confirmation.py` | 标准 MovingAI OOD 分析。 |
| `lns2_bottleneck.py`、`tradeoff_evaluation.py` | 当前双轨评测与 manifest 兼容接口。 |
| `repair_collection.py`、`repair_quality.py` | 基础修复经验采集与质量报告。 |
| `result_consolidation.py` | 24 项冻结证据的确定性汇总。 |

## scripts 目录

常用入口：

- `generate_dataset.py`、`generate_instance.py`、`inspect_dataset.py`
- `collect_closed_loop_confirmation.py`
- `analyze_closed_loop_confirmation.py`
- `analyze_movingai_ood_confirmation.py`
- `run_lns2_tradeoff_evaluation.py`
- `probe_stalled_state.py`
- `consolidate_research_results.py`
- `audit_repository_hygiene.py`
- `check_environment.py`

其余脚本用于 trace 转换、等价性验证、性能 benchmark、模型导出或旧采集 manifest 恢复，仍有测试或正式证据依赖。

## configs 与 artifacts

`configs/` 只保留当前可执行协议：闭环、MovingAI OOD、repair collection、stall guard、数据生成、结果收束和仓库维护。配置中的历史文档链接指向安全标签，不要求当前分支保留历史源码。

`artifacts/initlns-closed-loop-policy-v1` 是 canonical v1；`artifacts/initlns-closed-loop-controller-v2` 是 canonical v2。不得用确认集结果重新训练或替换它们。

## build 目录

`build/` 不是源码。它包含约数 GiB 的正式 trace、MovingAI 数据、`venv-graph`、Linux/Windows 构建和验证日志。路径被正式结果 manifest 引用，因此本轮不移动或改名。需要空间清理时先运行：

```powershell
python scripts/audit_repository_hygiene.py --check
python scripts/audit_repository_hygiene.py --emit-build-plan build/repository-hygiene
```

审计工具只给清单，不自动删除。

## 构建与验证

普通构建使用根 `CMakeLists.txt`。仅构建 native feature 模块时使用：

```bash
cmake -S . -B build/native-features -DLNS2_FEATURES_ONLY=ON
cmake --build build/native-features -j4
```

完整验证包括：

```powershell
python -m unittest discover -s tests -p "test_*.py"
python scripts/consolidate_research_results.py --config configs/result_consolidation.json --verify-build
python scripts/audit_repository_hygiene.py --check
```

WSL 中另运行 CTest 10/10 和官方 parity。环境锁文件现在位于根目录 `requirements-policy-training-wsl.lock`。

## 已移除内容如何恢复

历史 `research/`、`tests/research/`、旧随机简化 LNS2、Stage 3-5、失败模型训练器和未推广控制器均不在当前分支。查看或恢复时使用：

```bash
git show pre-minimal-runtime-2026-07-20:research/README.md
git switch --detach pre-minimal-runtime-2026-07-20
```

不要把历史实现重新复制回活动目录来复现结论；优先在标签上建立临时分支。
