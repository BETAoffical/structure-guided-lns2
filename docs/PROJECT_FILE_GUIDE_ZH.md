# 项目文件导航

这份文档回答三个问题：现在应该看哪些文件、每组文件负责什么、哪些内容只是为了历史复现而保留。
仓库包含约 480 个受控文件，但其中约 200 个来自 `third_party/` 和 `archive/`；日常开发不需要逐个阅读。

## 先看这 10 个入口

| 目的 | 文件或目录 | 说明 |
|---|---|---|
| 了解项目结论 | `docs/INITLNS_RESEARCH_REPORT_ZH.md` | 冻结的中文研究报告、正负结果和证据边界。 |
| 了解研究演变 | `docs/RESEARCH_ROADMAP.md` | 从静态上下文到显式邻域排序的完整路线。 |
| 了解当前求解流程 | `README.md` | 构建、CLI、Python 环境和主要运行命令。 |
| 官方 LNS2 内核 | `third_party/mapf_lns2/` | 固定版本的完整 MAPF-LNS2；核心算法来源。 |
| 项目 C++ 扩展 | `src/`、`include/` | repair-only CLI、Python binding、trace 和在线特征。 |
| Python 闭环控制器 | `experiments/closed_loop_confirmation.py` | 候选生成、特征、模型选择、repair 循环和 trace 主实现。 |
| 当前冻结模型 | `artifacts/initlns-closed-loop-policy-v1/` | 原始 portable pairwise GBDT。 |
| 当前优化模型 | `artifacts/initlns-closed-loop-controller-v2/` | 与 v1 决策语义相同的紧凑/原生加速版本。 |
| 最新性能诊断 | `experiments/lns2_bottleneck.py` | 分解候选生成、特征、推理、PP、trace 和墙钟时间。 |
| 最新 stall-safe 试验 | `docs/V2_STALL_SAFE_EVALUATION.md` | 针对长时间停滞状态的在研保护策略，不属于冻结主结论。 |

## 当前状态边界

仓库现在同时保留三条不同性质的内容，阅读时不要混在一起。

1. **冻结研究主线**：`realized_dynamic v1` 在同族未见地图和多 solver seed 上通过闭环确认；MovingAI 跨布局结果有正信号但未达到 5% 门槛。正式边界见 `docs/INITLNS_RESEARCH_REPORT_ZH.md`。
2. **v2 工程加速**：不重新训练模型，只压缩特征、树和推理路径，目标是保持候选、分数和选择完全一致并降低控制开销。
3. **stall-safe 在研试验**：当 frozen v2 连续选择导致 PP 失败且状态不变时，逐级限制邻域大小、黑名单重复失败候选，最后回退 Adaptive。它尚不能改写冻结研究结论。

`v1-full`、`v2-full` 和 `v2-balanced` 容易产生误解：

- `v1-full`：旧的 Python/字典特征部署路径。
- `v2-full`：同一冻结 GBDT 的紧凑、稠密、原生加速路径，不是重新训练的新模型。
- `v2-balanced`：在部分状态跳过模型的混合路由，不是 BALANCE 论文实现，也不是独立模型。
- `v2-stall-safe`：在 v2 排名之后施加运行期安全保护的实验控制器。

## 一次运行经过哪些文件

```text
MovingAI .map/.scen 或生成数据
        |
        v
generators/ 或 experiments/movingai_devset.py
        |
        v
src/python_bindings.cpp -> lns2_env
        |
        v
experiments/closed_loop_confirmation.py
        |-- 官方 Adaptive/Target/Collision/Random
        |-- v1/v2 显式邻域候选
        |-- online_feature_engine.py / src/online_features.cpp
        |-- compact_controller_model.py
        |-- PP + SIPPS repair（third_party/mapf_lns2）
        v
build/.../episodes + manifest + report
```

`scripts/*.py` 是命令行入口；主要实现通常在同名或相近的 `experiments/*.py` 中；参数在 `configs/*.json` 中；验证在 `tests/test_*.py` 中。

## 顶层目录

| 路径 | 用途 | 是否日常修改 |
|---|---|---|
| `src/` | 项目 C++ 实现和 pybind11 binding。 | 是，修改底层接口或性能时。 |
| `include/` | 项目 C++ 公共头文件。 | 偶尔。 |
| `experiments/` | 数据采集、特征、模型、分析和闭环实验的 Python 实现。 | 是，研究代码主体。 |
| `scripts/` | 面向用户的 CLI 薄入口和少量编排代码。 | 是，但算法逻辑优先放在 `experiments/`。 |
| `configs/` | 数据、采集、模型和验收门槛的版本化配置。 | 新实验时。 |
| `generators/` | 合成仓储地图和静态 OD 任务生成。 | 修改数据语义时。 |
| `artifacts/` | 提交到 Git 的冻结模型、紧凑证据和图表。 | 只在正式冻结/导出时。 |
| `docs/` | 每一阶段的设计、协议和结论。 | 实验完成后。 |
| `tests/` | Python/C++ 测试和小型地图 fixture。 | 随源码同步。 |
| `third_party/` | 固定提交的官方 MAPF-LNS2 和 GPBS。 | 原则上不改；项目 hook 除外。 |
| `archive/legacy_stage5/` | 接入官方 LNS2 前的简化求解器和 Stage 3-5 负面实验。 | 不用于当前构建。 |
| `native_features/` | 独立构建在线特征扩展的 CMake 入口。 | 性能开发时。 |
| `requirements/` | 可选训练环境的锁定依赖。 | 环境升级时。 |
| `build/` | 本地二进制、虚拟环境、数据、trace、模型中间件和报告。被 Git 忽略。 | 运行产生，不手工编辑。 |

## C++ 与原生接口

| 文件 | 职责 |
|---|---|
| `CMakeLists.txt` | 构建 `mapf_lns2_core`、`lns_official`、`lns2_repair`、`lns2_env`、GPBS 和 CTest。 |
| `src/repair_driver.cpp` | repair-only 命令行入口，停在首次无冲突解。 |
| `src/python_bindings.cpp` | `LNS2RepairEnv.reset/step/propose` 的 pybind11 接口和计时字段。 |
| `src/online_features.cpp` | 稠密在线状态/候选特征、冲突扫描和路径聚合。 |
| `src/online_features.h` | 在线特征 C++ 接口。 |
| `src/online_features_module.cpp` | 独立原生特征模块入口。 |
| `src/jsonl_observer.cpp` | C++ JSONL trace observer。 |
| `include/structure_guided/jsonl_observer.hpp` | observer 声明。 |
| `third_party/mapf_lns2/inc/RepairPolicy.h` | 项目加入官方内核的 repair action/policy 接口。 |
| `third_party/mapf_lns2/inc/InitLNS.h`、`src/InitLNS.cpp` | 官方 InitLNS 及项目的 step/propose/显式邻域扩展。 |

`third_party/mapf_lns2/inc/CBS/`、`inc/PIBT/` 和对应 `src/` 是官方低层/初始求解器实现，不需要为了理解当前高层策略逐个阅读。

## 数据生成

| 文件 | 职责 |
|---|---|
| `generators/models.py` | 地图和任务数据类。 |
| `generators/config.py` | 生成配置读取和校验。 |
| `generators/warehouse.py` | `regular_beltway`、`compartmentalized`、`dead_end_aisles` 等仓储布局生成。 |
| `generators/task_flows.py` | balanced、bottleneck、cross-zone、intersection 等静态 OD 任务。 |
| `generators/dataset.py` | split、seed 和整套数据集生成。 |
| `generators/io.py` | MovingAI `.map/.scen` 与 JSON sidecar I/O。 |
| `generators/validation.py` | 连通性、起终点、距离和语义约束。 |
| `generators/visualization.py` | 地图/任务可视化。 |

常用入口是 `scripts/generate_instance.py`、`scripts/generate_dataset.py`、`scripts/inspect_dataset.py` 和 `scripts/generate_gallery.py`。

## 当前控制器与性能代码

| 模块 | 作用 | 当前地位 |
|---|---|---|
| `closed_loop_confirmation.py` | 完整闭环执行器、候选生成、模型选择和 trace。 | 主线核心。 |
| `online_feature_engine.py` | Python/原生在线特征后端和一致性检查。 | v2 核心。 |
| `feature_schema_v2.py` | v2 稠密特征顺序、ID 和 schema hash。 | v2 核心。 |
| `compact_controller_model.py` | 将 portable GBDT 压缩并加载为 Python/C++ 可推理 bundle。 | v2 核心。 |
| `controller_performance_benchmark.py` | 对比旧/新特征和推理开销。 | 工程验证。 |
| `candidate_pruning.py`、`proposal_pruner_training.py` | outcome-blind proposal 剪枝训练与执行。 | 已审计，当前 pruner 未获推广。 |
| `balanced_controller.py` | v2-balanced 混合路由。 | 未获推广，保留复现。 |
| `lns2_speed_quality_calibration.py` | 速度/质量校准。 | v2 前置工程实验。 |
| `lns2_tradeoff.py` | v1/v2/balanced/Adaptive 的综合比较和报告。 | 工程评估。 |
| `lns2_bottleneck.py` | historical、墙钟和 timeout 轨道的精确耗时分解。 | 最新性能诊断。 |
| `stall_guard.py` | 连续失败时的 cap、blacklist 和 Adaptive fallback 状态机。 | 在研。 |
| `stalled_state_probe.py` | 重放停滞状态并比较替代邻域/repair order。 | 在研机制探针。 |

对应 CLI：

- `scripts/benchmark_controller_v2.py`
- `scripts/benchmark_exact_runtime.py`
- `scripts/train_proposal_pruner_v2.py`
- `scripts/calibrate_lns2_speed_quality.py`
- `scripts/run_lns2_tradeoff_evaluation.py`
- `scripts/probe_stalled_state.py`
- `scripts/verify_closed_loop_equivalence.py`

最新本地输出主要在：

- `build/initlns-exact-runtime-benchmark/`
- `build/initlns-lns2-bottleneck-quick-v2-exact/`
- `build/initlns-stalled-state-probe-v1/`
- `build/initlns-v2-stall-safe-targeted-v1/`
- `build/initlns-v2-stall-safe-quick-wall-v1/`

## 历史实验模块

这些文件不是废弃垃圾。它们保存预注册假设、负面结果和停止规则，但通常不应作为新代码入口。

`experiments/_common.py` 只保存 JSON/JSONL、SHA、统计量和稳定 ID 等无实验语义的公共工具；`experiments/__init__.py` 只导出当前包级 repair collection 接口。

### 数据采集和上下文

| 模块 | 对应文档 | 结论/用途 |
|---|---|---|
| `repair_collection.py`、`repair_quality.py` | `REPAIR_COLLECTION.md` | qualification、baseline、反事实轨迹和标签质量。 |
| `context_audit.py` | `CONTEXT_AUDIT.md` | 首轮静态上下文增量审计，未通过。 |
| `context_confirmation.py` | `CONTEXT_SECONDARY_AUDIT.md` | 二次置换/分组验证，未通过。 |
| `local_representation_audit.py` | `LOCAL_REPRESENTATION_AUDIT.md` | 局部预生成/实际邻域表示恢复审计，未通过。 |

### 邻域机制和排序

| 模块 | 对应文档 | 结论/用途 |
|---|---|---|
| `movingai_mechanism_probe.py`、`movingai_probe_quality.py` | `MOVINGAI_MECHANISM_PROBE.md`、`MOVINGAI_PROBE_QUALITY.md` | 名义动作、trial 稳定性和标签质量。 |
| `independent_layout_probe.py` | `INDEPENDENT_LAYOUT_PROBE.md` | 独立布局动作机制，名义动作不稳定。 |
| `realized_neighborhood_probe.py` | `REALIZED_NEIGHBORHOOD_PROBE.md` | 显式 agent 集合比名义动作稳定，通过。 |
| `realized_neighborhood_ranking_audit.py` | `REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md` | 显式邻域监督排序，通过。 |
| `realized_ranking_confirmation.py` | `REALIZED_RANKING_CONFIRMATION.md` | 新地图排序确认。 |
| `natural_distribution_confirmation.py` | `NATURAL_DISTRIBUTION_CONFIRMATION.md` | 不筛选冲突难度的独立确认，通过。 |

`natural_distribution_confirmation_analysis.py` 和 `realized_ranking_confirmation_analysis.py` 是上述 collector 的纯分析层，不负责运行求解器。

### 闭环、聚合和模型诊断

| 模块 | 对应文档 | 结论/用途 |
|---|---|---|
| `closed_loop_confirmation_analysis.py` | `CLOSED_LOOP_CONFIRMATION.md` | 首次闭环确认。 |
| `policy_visited_aggregation.py` | `POLICY_VISITED_AGGREGATION.md`、`POLICY_VISITED_NATURAL_DISTRIBUTION.md` | 策略访问状态聚合 v2，未改善。 |
| `policy_visited_aggregation_analysis.py` | 同上 | policy-visited 标签、分布偏移和闭环汇总。 |
| `policy_visited_independent_confirmation.py` | 同上 | 聚合模型通过开发门槛后才允许的独立确认门控。 |
| `ranking_objective_audit.py` | `RANKING_OBJECTIVE_AUDIT.md` | pairwise/双头目标对齐，未通过。 |
| `ranking_objective_confirmation.py` | 同上 | 排序目标胜者的独立确认数据接口；门槛失败后未成为主线。 |
| `model_capacity_audit.py` | `MODEL_CAPACITY_AUDIT.md` | 更大 GBDT 过拟合。 |
| `graph_representation_audit.py` | `GRAPH_REPRESENTATION_AUDIT.md` | MLP/DeepSets/GNN 未超过 GBDT。 |
| `graph_feature_gbdt_audit.py` | `GRAPH_FEATURE_GBDT_AUDIT.md` | 图统计特征 GBDT 未通过。 |
| `sequential_credit_audit.py` | `SEQUENTIAL_CREDIT_AUDIT.md` | Horizon-4 标签不稳定。 |
| `repair_order_probe.py`、`contextual_repair_order_audit.py` | `REPAIR_ORDER_PROBE.md`、`CONTEXTUAL_REPAIR_ORDER_AUDIT.md` | repair order 重要，但 selector 未通过。 |

### 标准地图、基线和结果收束

| 模块 | 作用 |
|---|---|
| `movingai_devset.py` | 下载/验证标准 MovingAI 地图和 scenario。 |
| `feasibility_benchmark.py` | LNS2 与 GPBS 统一 feasibility runner。 |
| `movingai_ood_confirmation.py` | MovingAI 跨布局闭环分析。 |
| `final_model_evaluation.py` | 最终冻结模型评估和门槛。 |
| `result_consolidation.py` | 读取正式 JSON，生成中文报告、CSV 和图表。 |
| `closed_loop_trace_storage.py`、`closed_loop_trace_conversion.py` | trace 压缩、迁移和完整性验证。 |
| `route_counterfactual.py` | 对混合路由跳过的状态执行单步反事实。 |
| `run_output_guard.py` | 输出目录锁、fingerprint 和 resume 防护。 |

## scripts 目录怎么找入口

脚本名称已经按动作分组：

- `generate_*`：生成地图、任务、数据集、gallery 或确认集。
- `collect_*`：运行求解器并收集 episode/transition/counterfactual。
- `analyze_*`：只读取已有结果生成报告。
- `run_*_audit.py`：执行一次冻结的模型/表示/标签审计。
- `benchmark_*`：计时和等价性能微基准。
- `verify_*`、`audit_*`、`check_*`：只读验证或一致性检查。
- `export_closed_loop_models.py`：从 sklearn 模型导出无 sklearn 依赖的 portable bundle。
- `run_final_model_evaluation.py`：正式模型评估编排。
- `run_lns2_tradeoff_evaluation.py`：当前最大的一体化工程评估入口；包含 quick/formal、双控制器、墙钟和 stall-safe 路由。
- `consolidate_research_results.py`：重新生成冻结研究报告，不训练模型。

如果脚本和实现同名，先读脚本的参数，再跳到它 import 的 `experiments` 模块。不要从 6 万行的编排脚本顺序通读整个项目。

### 完整 scripts 索引

| 脚本 | 作用 |
|---|---|
| `analyze_closed_loop_confirmation.py` | 分析已完成的闭环 collection。 |
| `analyze_independent_layout_probe.py` | 分析独立布局探针。 |
| `analyze_movingai_ood_confirmation.py` | 汇总 MovingAI OOD 闭环结果。 |
| `analyze_movingai_probe.py` | 分析 MovingAI 机制探针。 |
| `analyze_realized_neighborhood_probe.py` | 分析固定显式邻域 trials。 |
| `analyze_repair_experience.py` | 生成 calibration 标签质量报告。 |
| `audit_movingai_probe_quality.py` | 审计 trial 稳定性、重复状态和标签质量。 |
| `audit_repository_hygiene.py` | 只读仓库/build 保护和冗余审计。 |
| `benchmark_controller_v2.py` | v2 feature/controller 性能基准。 |
| `benchmark_exact_runtime.py` | 候选、特征、分数、排名完全等价的精确计时。 |
| `benchmark_portable_tree_inference.py` | portable tree Python/C++ 推理微基准。 |
| `calibrate_lns2_speed_quality.py` | 速度、质量和 pruner 校准编排。 |
| `check_environment.py` | 只读检查 WSL runtime 或 Windows training 环境。 |
| `collect_closed_loop_confirmation.py` | 通用闭环 episode collector。 |
| `collect_natural_distribution_confirmation.py` | 自然冲突分布 collection。 |
| `collect_policy_visited_experience.py` | 冻结策略访问状态及候选 trial collection。 |
| `collect_realized_neighborhood_probe.py` | 固定显式邻域多 trial collection。 |
| `collect_realized_ranking_confirmation.py` | 显式邻域排序独立确认 collection。 |
| `collect_repair_experience.py` | qualification、baseline 和 counterfactual 总入口。 |
| `consolidate_research_results.py` | 从冻结 JSON 生成证据清单、CSV、图和中文报告。 |
| `convert_closed_loop_traces.py` | full-v1 trace 转 delta-gzip-v2。 |
| `export_closed_loop_models.py` | sklearn 模型导出 portable JSON。 |
| `fetch_movingai_devset.py` | 下载并校验 MovingAI 开发地图。 |
| `generate_dataset.py` | 从配置生成合成数据集。 |
| `generate_gallery.py` | 生成地图/任务可视化 gallery。 |
| `generate_instance.py` | 生成单个地图和任务实例。 |
| `generate_ranking_objective_confirmation.py` | 生成排序目标独立确认数据。 |
| `inspect_dataset.py` | 检查数据集 split、统计和语义。 |
| `manage_repair_collection.py` | collection 状态、resume 和 manifest 管理。 |
| `prepare_movingai_probe.py` | 准备 MovingAI 机制探针数据。 |
| `probe_stalled_state.py` | 对一个停滞状态运行替代候选/repair-order 探针。 |
| `recover_counterfactual_manifest.py` | 从已有分支文件恢复中断的反事实 manifest。 |
| `run_context_audit.py` | 首轮静态上下文审计。 |
| `run_context_confirmation.py` | 上下文二次诊断和置换验证。 |
| `run_contextual_repair_order_audit.py` | repair-order 上下文 selector 审计。 |
| `run_feasibility_benchmark.py` | LNS2/GPBS 统一 feasibility benchmark。 |
| `run_final_model_evaluation.py` | 冻结模型最终评估和正式门槛。 |
| `run_graph_feature_gbdt_audit.py` | 图统计特征 GBDT 审计。 |
| `run_graph_representation_audit.py` | MLP/DeepSets/GNN 表示审计。 |
| `run_lns2_tradeoff_evaluation.py` | v1/v2/Adaptive、墙钟、timeout 和 stall-safe 总编排。 |
| `run_local_representation_audit.py` | 局部预生成/实际邻域表示审计。 |
| `run_model_capacity_audit.py` | GBDT 容量曲线。 |
| `run_natural_distribution_confirmation.py` | 自然分布结果分析和门控。 |
| `run_policy_visited_aggregation.py` | policy-visited 索引、训练和离线评估。 |
| `run_policy_visited_independent_confirmation.py` | policy-visited v2 独立确认。 |
| `run_ranking_objective_audit.py` | pairwise 加权/双头目标比较。 |
| `run_realized_neighborhood_ranking_audit.py` | 显式邻域 leave-one-map-out 排序审计。 |
| `run_realized_ranking_confirmation.py` | 显式邻域排序确认分析。 |
| `run_repair_order_probe.py` | PP repair order 机制探针。 |
| `run_sequential_credit_audit.py` | policy-visited Horizon-4 长期信用审计。 |
| `train_proposal_pruner_v2.py` | outcome-blind proposal pruner 训练。 |
| `update_controller_v2_promotion.py` | 汇总等价/性能/pruner 门槛并更新 v2 manifest。 |
| `verify_closed_loop_equivalence.py` | v1/v2 transition、候选、分数和选择等价检查。 |

## configs 目录怎么对应

配置文件遵循以下命名：

- `*_dataset.json`：地图、任务、split 和 seed。
- `*_collection.json`：策略、solver seed、时间/修复预算、workers 和输出。
- `*_analysis.json`：输入结果、指标、bootstrap 和门槛。
- `*_pilot.json`：smoke/pilot 小规模配置，不是正式结论。
- `*_audit.json`：冻结模型参数、特征组和预注册判定。

主要配置族：

| 前缀 | 对应阶段 |
|---|---|
| `repair_*` | Pilot v1/v2、calibration 和反事实采集。 |
| `context_*` | 静态上下文首轮/二次审计。 |
| `movingai_mechanism_*`、`independent_layout_*` | MovingAI 动作机制与稳定性。 |
| `realized_neighborhood_*`、`realized_ranking_*` | 显式邻域探针、排序和确认。 |
| `natural_distribution_*` | 自然冲突分布确认。 |
| `closed_loop_*` | 冻结模型闭环和多 seed 确认。 |
| `policy_visited_*` | 策略访问状态采集和聚合。 |
| `ranking_objective_*`、`model_capacity_*`、`graph_*` | 模型失败原因诊断。 |
| `sequential_credit_*`、`repair_order_*` | 长期信用和 PP order。 |
| `movingai_ood_*` | 标准地图跨布局确认。 |
| `proposal_pruner_v2.json` | v2 proposal pruner；当前未启用。 |
| `v2_stall_guard_v1.json` | 当前 stall-safe cap/blacklist/fallback 规则。 |
| `result_consolidation.json` | 24 项冻结正式证据及 SHA。 |
| `repository_hygiene.json` | build 保护、审计和清理分类。 |

### 完整 config 索引

下面按实验阶段列出全部配置；同一行中的 `dataset/collection/analysis/pilot` 分别负责数据、执行、汇总和小规模验证。

| 阶段 | 配置文件 |
|---|---|
| 初始生成 | `stage1_example.json`、`repair_transfer_pilot.json` |
| 修复经验 | `repair_collection_pilot.json`、`repair_collection_calibration.json`、`repair_collection_hardening_smoke.json` |
| 上下文确认 | `context_confirmation_dataset.json`、`context_confirmation_collection.json` |
| MovingAI 开发集 | `movingai_devset.json`、`movingai_ood_devset.json` |
| MovingAI 机制探针 v1 | `movingai_mechanism_probe_dataset.json`、`movingai_mechanism_probe_collection.json` |
| MovingAI 机制探针 v2 | `movingai_mechanism_probe_v2_dataset.json`、`movingai_mechanism_probe_v2_collection.json`、`movingai_mechanism_probe_v2_bounded_collection.json`、`movingai_probe_quality_audit.json` |
| 独立布局探针 | `independent_layout_probe_dataset.json`、`independent_layout_probe_collection.json`、`independent_layout_probe_quality.json` |
| 显式邻域 | `realized_neighborhood_stability_probe.json`、`realized_neighborhood_ranking_audit.json` |
| 显式排序确认 | `realized_ranking_confirmation_dataset.json`、`realized_ranking_confirmation_collection.json`、`realized_ranking_confirmation_analysis.json` |
| 自然分布确认 | `natural_distribution_confirmation_dataset.json`、`natural_distribution_confirmation_collection.json`、`natural_distribution_confirmation_analysis.json`、`natural_distribution_confirmation_pilot.json` |
| 闭环确认 | `closed_loop_confirmation_dataset.json`、`closed_loop_confirmation_collection.json`、`closed_loop_confirmation_analysis.json`、`closed_loop_confirmation_pilot.json` |
| 多 seed 闭环 | `closed_loop_multiseed_dataset.json`、`closed_loop_multiseed_collection.json`、`closed_loop_multiseed_analysis.json` |
| Policy-visited | `policy_visited_dataset.json`、`policy_visited_collection.json`、`policy_visited_analysis.json`、`policy_visited_natural_collection.json`、`policy_visited_natural_analysis.json`、`policy_visited_confirmation_dataset.json`、`policy_visited_independent_confirmation.json` |
| 排序目标 | `ranking_objective_audit.json`、`ranking_objective_confirmation_dataset.json` |
| 模型容量/图表示 | `model_capacity_audit.json`、`graph_representation_audit.json`、`graph_feature_gbdt_audit.json` |
| 长期信用/repair order | `sequential_credit_audit.json`、`repair_order_probe.json`、`contextual_repair_order_audit.json` |
| MovingAI OOD | `movingai_ood_dataset.json`、`movingai_ood_collection.json`、`movingai_ood_analysis.json` |
| v2 工程 | `proposal_pruner_v2.json`、`v2_stall_guard_v1.json` |
| 结果和仓库 | `result_consolidation.json`、`repository_hygiene.json` |

## artifacts 目录

| 路径 | 内容 |
|---|---|
| `initlns-closed-loop-policy-v1/` | 冻结 v1 的 proposal/realized pairwise GBDT JSON、manifest 和等价报告。 |
| `initlns-closed-loop-controller-v2/` | 紧凑模型、训练来源报告、性能基准和 promotion report。 |
| `initlns-research-evidence-v1/` | 24 项实验的 evidence manifest、指标 CSV、验证和 SVG 图。 |
| `initlns-movingai-ood-compact-migration-v2/` | 720 episode trace 压缩迁移证据；不是删除授权。 |

只有通过冻结门槛的模型和小型证据进入 `artifacts/`。大模型中间文件、原始 trace 和 rollout 留在 `build/`。

## build 目录

`build/` 是最大的目录，也是最容易迷路的地方。它不是源码树。

### 环境和二进制

- `build/linux/project/`：WSL CMake 构建、`lns2_env`、`lns2_repair` 和 CTest。
- `build/windows/`、`build/windows-v2/`：Windows 构建。
- `build/native-features-windows/`：Windows 原生特征扩展。
- `build/venv-graph/`：图模型审计的独立 Python 环境。
- `build/gpbs-upstream/`：GPBS 构建/运行环境。

### 命名规律

- `*-dataset*`：地图、scenario、sidecar 和 manifest。
- `*-collection*`：episode trace、collection manifest 和 run config。
- `*-report*`、`*-audit*`：汇总 JSON、Markdown、CSV、图表和模型诊断。
- `*-frozen-models*`、`*-training*`：训练中间件和冻结模型。
- 名称含 `smoke`、`verify`、`preregister`、`dry-run`：通常是可再生临时结果，但删除前仍须运行卫生审计。

### 当前最值得看的结果

| 路径 | 内容 |
|---|---|
| `initlns-lns2-bottleneck-quick-v2-exact/report/` | 最新 v2 与 Adaptive 逐阶段耗时。 |
| `initlns-exact-runtime-benchmark/` | 候选、特征、分数和选择完全一致的微基准。 |
| `initlns-movingai-ood-report-v1/` | 冻结 v1 的正式 MovingAI OOD 结果。 |
| `initlns-movingai-ood-collection-v2-compact/` | 720 个正式 episode 的紧凑 `delta-gzip-v2` 轨迹，是当前保留的跨布局原始证据。旧 full-v1 `episodes/` 已在 720/720 等价验证后删除。 |
| `initlns-closed-loop-multiseed-v1-report/` | 同族新地图多 seed 闭环确认。 |
| `initlns-stalled-state-probe-v1/` | stall 原因和替代候选探针。 |
| `initlns-v2-stall-safe-targeted-v1/` | stall-safe 单目标实验。 |
| `initlns-v2-stall-safe-quick-wall-v1/` | stall-safe quick 墙钟队列。 |

不要凭目录名直接删除 `build/` 内容。只读检查命令：

```powershell
python scripts/audit_repository_hygiene.py --check
python scripts/audit_repository_hygiene.py --emit-build-plan build/repository-hygiene-review
```

2026-07-20 的空间优先清理记录位于
`build/repository-hygiene-space-cleanup-20260720/`。该轮释放约 15.18 GiB，
没有删除 `build/venv-graph`、构建环境、正式紧凑轨迹、冻结模型或 24 项证据来源。

## tests 目录

- `test_<experiment>.py` 通常与 `experiments/<experiment>.py` 一一对应。
- `test_closed_loop_confirmation.py`、`test_controller_v2.py`、`test_lns2_bottleneck.py`、`test_stall_guard.py` 和 `test_stalled_state_probe.py` 是当前控制器/性能主线测试。
- `test_repair_interfaces.cpp` 验证 C++ repair/propose/随机流和原生接口。
- `test_python_env.py` 验证 pybind 环境、计时 schema 和在线特征。
- `check_path_hash.py` 验证官方 LNS2 parity hash。
- `check_trace.py` 验证 JSONL trace schema。
- `tests/data/` 是小型确定性地图 fixture，不是实验数据集。

历史实验测试仍保留，是为了保证旧结论和 schema 可以复现，并不表示这些模型仍在当前运行路径中。

## docs 目录

文档基本与实验阶段一一对应。除前文已列出的审计文档外，还有几个横向入口：

| 文档 | 用途 |
|---|---|
| `CONFIGURATION.md` | 配置 schema、字段和实例。 |
| `ENVIRONMENT_AUDIT.md` | WSL 可见性、真实 Ubuntu 用户和依赖盘点。 |
| `TRACE_AND_POLICY_API.md` | observation、action、trace 和 Python policy API。 |
| `MOVINGAI_BASELINES.md` | 标准 MovingAI runner 和 GPBS/LNS2 基线。 |
| `CLOSED_LOOP_MULTISEED_CONFIRMATION.md` | 同族新地图多 solver seed 正式闭环结果。 |
| `STAGE1.md` | 仍保留的仓储数据生成第一阶段说明。 |
| `REPOSITORY_HYGIENE.md` | 证据保护和安全清理规则。 |
| `INITLNS_RESEARCH_REPORT_ZH.md` | 最终研究结论；阅读历史实验时以它的 decision 为准。 |

## archive 与 third_party

### `archive/legacy_stage5/`

这里保存接入官方内核前的简化 C++ solver、KNN/检索/监督排序、Stage 5 v1-v4 和 rollout。它已经退出根 CMake、当前 Python package 和默认实验。只有追溯早期负面实验时才需要阅读。

### `third_party/mapf_lns2/`

完整官方 MAPF-LNS2 固定提交。重要入口：

- `UPSTREAM.md`：来源和 commit。
- `license.txt`：许可证。
- `src/driver.cpp`：官方 CLI。
- `inc/InitLNS.h`、`src/InitLNS.cpp`：首次可行解修复。
- `inc/BasicLNS.h`、`src/BasicLNS.cpp`：可行解后的 anytime LNS。
- `inc/SIPP.h`、`src/SIPP.cpp`：低层安全区间规划。

### `third_party/gpbs/`

独立 GPBS feasibility 基线。它不是 LNS2 destroy heuristic；只在端到端求解器对照中使用。

## 按问题找文件

| 你想做什么 | 从这里开始 |
|---|---|
| 运行官方 LNS2 | `README.md` -> `CMakeLists.txt` -> `third_party/mapf_lns2/src/driver.cpp` |
| 运行 repair-only | `src/repair_driver.cpp` |
| 理解 Python 环境 | `src/python_bindings.cpp` -> `docs/TRACE_AND_POLICY_API.md` |
| 理解当前 GBDT | `docs/REALIZED_NEIGHBORHOOD_RANKING_AUDIT.md` -> `artifacts/initlns-closed-loop-policy-v1/` |
| 理解 v2 为什么更快 | `experiments/compact_controller_model.py` -> `experiments/online_feature_engine.py` -> 最新 bottleneck report |
| 比较 v2 和 Adaptive | `scripts/run_lns2_tradeoff_evaluation.py` -> `experiments/lns2_bottleneck.py` |
| 看 MovingAI 逐地图结果 | `docs/MOVINGAI_OOD_CLOSED_LOOP.md` -> `build/initlns-movingai-ood-report-v1/` |
| 看当前 stall-safe | `docs/V2_STALL_SAFE_EVALUATION.md` -> `experiments/stall_guard.py` -> `experiments/stalled_state_probe.py` |
| 重新生成中文总结 | `scripts/consolidate_research_results.py` -> `configs/result_consolidation.json` |
| 判断文件能否删除 | `docs/REPOSITORY_HYGIENE.md` -> `scripts/audit_repository_hygiene.py` |

## 维护规则

新增实验时保持四件套：

1. `experiments/<name>.py`：实现。
2. `scripts/run_<name>.py` 或 `collect_<name>.py`：CLI。
3. `configs/<name>.json`：数据、参数和门槛。
4. `tests/test_<name>.py` 与 `docs/<NAME>.md`：验证和结论。

不要把新的实验逻辑继续塞入 `run_lns2_tradeoff_evaluation.py`；它已经承担较多编排职责。新的长期研究应建立独立模块，冻结结果后再进入 evidence ledger。
