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
