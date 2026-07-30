# V2、Mixed Full 与官方 LNS2 的分层墙钟确认

## 研究问题

本实验冻结候选生成、PP+SIPPS 修复、特征定义和模型参数，只比较三种 InitLNS 高层控制器：

- `official_adaptive`：官方 Adaptive InitLNS。
- `v2-full`：冻结的原 V2 模型，运行于批量 proposal、原生特征和 fingerprint 快路径。
- `mixed-full-v2`：使用 311 个旧状态和 144 个高负载状态重训的 Mixed Full 模型，接入与 V2 完全相同的优化运行时。

Mixed Full 的模型输入为 124 个 `realized_dynamic` 基础特征构成的 147 维 pairwise 输入。训练参数固定为 100 棵树、15 个叶节点、最小叶节点 20、学习率 0.05、L2 0.1 和随机种子 20260714。本轮不改变标签、候选池或修复器。

## 数据与盲选

数据包含 12 张从未参与模型训练的地图：6 张固定 SHA256 的 MovingAI 地图，以及 master seed `20270831` 生成的三类结构化地图各两张。72 个任务配合 solver seeds `[1,2,3]` 形成 216 个 qualification reset。

正式 cohort 只根据初始 PP 的唯一冲突 agent 对数选择，不读取任何控制器结果：

- 低冲突：1 至 10 对。
- 中冲突：11 至 100 对。
- 高冲突：101 至 500 对。
- 每层按稳定哈希选 12 个，共 36 个配对实例。

每层至少包含 4 个 MovingAI、4 个生成实例、两个 agent 数量区间，单张地图每层最多贡献两个实例。任一层不足时实验以“数据门槛失败”结束，不更改边界或重抽任务。

## 墙钟边界

正式运行是单 worker。每个实例的三种控制器按六种排列均衡轮换，避免固定顺序偏差。主要 TTF 从 `env.reset()` 前计时到首次可行解，包含初始 PP、在线候选生成、特征提取、模型推理和 PP+SIPPS 修复。失败按 600 秒计，外部进程硬超时为 660 秒。

报告同时保留 environment 构造、reset、proposal、特征、推理、fingerprint、PP 重规划、完整 repair、SIPPS nodes、修复次数和固定 100 步冲突 AUC。组件微基准是工程证据，不能替代正式端到端 TTF。

## 预注册判定

Mixed Full 只有同时满足以下条件才替代 V2：成功数不降低；总体 capped TTF 改善至少 5%；AUC 退化不超过 2%；至少 8/12 张地图和 2/3 个冲突层级不劣；5,000 次地图级配对 bootstrap 不显示显著退化；并且没有非法动作、fingerprint mismatch 或未解释错误。

若只在中高冲突改善，本轮仅记录条件性信号，不事后添加负载门控。若门槛失败，默认控制器仍为 `v2-full`。

## 可追溯入口

- 数据配置：`configs/balanced_wall_clock_generated_dataset.json`
- MovingAI 来源：`configs/balanced_wall_clock_movingai_source.json`
- 采集配置：`configs/balanced_wall_clock_collection.json`
- Mixed Full artifact：`artifacts/initlns-mixed-full-controller-v2`
- 主入口：`scripts/run_balanced_wall_clock.py`
- 固定动作组件微基准：`scripts/benchmark_exact_runtime.py`
- 旧七图报告提交：`bf9318ae967d6bdcda46c6b6d7535c29072dcacb`

正式 episode 开始前必须提交并推送以上配置、模型 SHA、指标和门槛。

## Qualification 结果

预注册提交 `9d12420c1fe27c475750538c9c23ad50369d618e` 后运行了全部 216 次初始 PP reset。216 次均有效，无错误和超时；98 次初始即无冲突，118 次存在冲突。唯一冲突 agent 对分布为：低 62、中 45、高 11、超过 500 为 0。

正式 cohort 门槛未通过。高冲突层不足 12 个；同时 MovingAI 的非零冲突状态主要集中在 `maze-128-128-2`，在“单图每层最多两个实例”的限制下，三个层级都无法达到每层至少 4 个 MovingAI 实例。因此决策为 `data_gate_failed_no_formal_episode_run`，未运行 108 个正式控制器 episode，也没有事后调整阈值、scenario、agent 数量或地图。

紧凑结果登记位于 `artifacts/initlns-v2-mixed-balanced-wall-clock-v1/qualification_gate.json`；完整 qualification 数据继续保存在忽略的 `build/` 下。

## 局部替换数据集 V2

V1 门槛失败后没有整体重做地图。V2 保留原数据中 7 张能够提供有效冲突层级的地图，只移除 5 张在固定 scenario、agent 数量和 solver seed 下始终为零冲突的 MovingAI 地图。候选补充过程仍只运行初始 PP qualification，不读取 Adaptive、V2 或 Mixed Full 的任何控制器结果。

最终补入 5 张地图：MovingAI 的 `lt_gallowstemplar_n`，以及 4 张独立 master seed 生成的 compartmentalized/dead-end 地图。候选 `ht_mansion_n` 能提供低、中冲突，但不能满足高冲突层的跨地图上限，因此没有进入最终数据集。MovingAI 文件来自官方 [MAPF benchmark](https://movingai.com/benchmarks/mapf/index.html)，下载 URL、archive SHA256 和成员路径固定在 `configs/balanced_wall_clock_movingai_candidates_v2.json`。

最终 V2 数据集含 12 张地图、72 个任务和 solver seeds `[1,2,3]`。216/216 次 reset 有效，错误和超时均为 0；初始唯一冲突 agent 对分布为：零冲突 14、低冲突 78、中冲突 90、高冲突 31、超过 500 的极端实例 3。零冲突与极端实例保留在 qualification 报告中，但不进入正式速度 cohort。

盲选器最终在低、中、高三层各冻结 12 条，共 36 条配对实例。三层的来源下限、agent 数量区间、单图每层最多 2 条和地图多样性检查全部通过，decision 为 `eligible_for_formal`。紧凑登记位于 `artifacts/initlns-v2-mixed-balanced-wall-clock-v2/qualification_gate.json`；正式 Adaptive/V2/Mixed Full 控制器 episode 尚未运行，因此本节只证明数据门槛通过，不构成模型速度或质量结论。

V2 的活动入口为：

- 替换清单：`configs/balanced_wall_clock_replacement_v2.json`
- 最终采集配置：`configs/balanced_wall_clock_collection_v2.json`
- 本地数据集：`build/initlns-v2-mixed-balanced-dataset-v2c`
- 本地 qualification：`build/initlns-v2-mixed-balanced-qualification-v2c`
- 本地冻结 cohort：`build/initlns-v2-mixed-balanced-cohort-v2c`

## 正式墙钟结果

预注册提交 `9d7bf5524c09adc170c38576f06813d4931d84e4` 后，按冻结 schedule 完成了 36 个实例乘 3 个控制器，共 108 个正式 episode。运行采用单 worker、600 秒 solver budget、660 秒外部超时和 100 次修复上限。108 条结果全部为 `ok`，非法动作、fingerprint mismatch、采集错误和外部超时均为 0。

| 控制器 | 成功数 | 平均 capped wall TTF | 固定 100 步冲突 AUC | 平均修复次数 |
| --- | ---: | ---: | ---: | ---: |
| 官方 Adaptive | 34/36 | 58.093 秒 | 1114.486 | 28.278 |
| V2 Full | 34/36 | 60.564 秒 | 499.472 | 20.139 |
| Mixed Full V2 | 35/36 | 43.763 秒 | 507.944 | 19.861 |

Mixed Full V2 相对 V2 Full 的 capped wall TTF 改善 `27.741%`，成功数增加 1；固定冲突 AUC 退化 `1.696%`，仍在预注册的 2% 容许范围内。它在 9/12 张地图、2/3 个冲突层级上不劣于 V2；地图级 paired bootstrap 的 TTF 改善 95% 区间为 `[-2.299%, 98.867%]`，未显示显著退化，但区间跨过 0，因此不能声称墙钟改善已经达到传统统计显著性。因此八项晋级门槛全部通过，正式 decision 为 `mixed_full_candidate`，本 cohort 上应将 Mixed Full V2 晋级为优先控制器。

相对官方 Adaptive，Mixed Full V2 的 capped wall TTF 改善 `24.667%`，固定冲突 AUC 改善 `54.423%`。但该结果仅支持冻结的 balanced cohort，不能自动外推为所有 MovingAI 地图都更快，也不能恢复静态地图/OD/密度上下文的迁移主张。

完整正式数据保存在 `build/initlns-v2-mixed-balanced-formal-v2c`，确定性报告位于 `build/initlns-v2-mixed-balanced-report-v2c/balanced_wall_clock_report.json`，SHA256 为 `953af2f961650062263511fac3c4cd722d2d8d78fbf389836c71a8e63f95ace0`。紧凑 Git 证据位于 `artifacts/initlns-v2-mixed-balanced-wall-clock-v2/formal_result.json`。
