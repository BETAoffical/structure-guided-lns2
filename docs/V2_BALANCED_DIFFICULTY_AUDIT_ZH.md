# V2/Mixed Full 分层墙钟难度审计

## 审计目的

旧正式实验按初始 PP 的唯一冲突 agent 对数选择低、中、高各 12 个实例。本审计不重跑控制器，而是从冻结的 36 个初始 state blob 重建冲突与路径，并检查“冲突数量分层”是否同时代表实际计算负载。

原正式报告及 SHA256 保持不变。审计只修正可支持的解释范围，不能事后改写预注册门槛。

## 完整性

- 36/36 个初始状态可解析。
- 108/108 个控制器 episode 仍属于相同冻结 schedule。
- state fingerprint、唯一冲突对数、重建冲突边和 agent 数量 mismatch 均为 0。
- 求解器仍只在唯一冲突对数降到 0 时判断可行。

因此没有证据表明“数百冲突在一秒内修完”来自冲突计算或可行性判断 Bug。

## 两种分层

原冲突分层正确执行：

| 唯一冲突 agent 对 | 数量 |
| --- | ---: |
| 低，1-10 | 12 |
| 中，11-100 | 12 |
| 高，101-500 | 12 |

新增硬件无关的初始 PP 负载诊断，使用 low-level generated nodes：

| 初始 PP generated nodes | 数量 | 来源 |
| --- | ---: | --- |
| 低，至多 100,000 | 19 | 全部为生成地图 |
| 中，100,001-1,000,000 | 5 | 全部为生成地图 |
| 高，超过 1,000,000 | 12 | 全部为 MovingAI |

这说明旧 cohort 在冲突对数上均衡，但在计算负载上不均衡，而且地图来源与负载完全混杂。它仍可比较同一实例上的三个控制器，却不能回答“在相同难度下，生成地图和 MovingAI 谁更慢”。

## 为什么冲突数不是难度

官方 Adaptive 的 Spearman 相关性如下：

| 初始指标 | 与初始 PP 时间的相关性 |
| --- | ---: |
| 唯一冲突 agent 对数 | 0.436 |
| agent 数 | 0.851 |
| 总初始路径 cost | 0.981 |
| 初始 PP generated nodes | 0.994 |

冲突对数描述需要修复的关系数量；PP/SIPPS 的计算量还取决于 agent 数、路径长度、约束表大小、地图尺度和低层搜索节点。相同的数百个冲突可以对应完全不同的搜索工作量。

## 时间与失败记账

| 来源 | 控制器 | 成功 | 平均实际 episode 时间 | 平均 capped TTF |
| --- | --- | ---: | ---: | ---: |
| 生成 | Adaptive | 23/24 | 0.440 s | 25.386 s |
| 生成 | V2 | 23/24 | 0.591 s | 25.472 s |
| 生成 | Mixed Full | 24/24 | 0.446 s | 0.444 s |
| MovingAI | Adaptive | 11/12 | 84.842 s | 123.507 s |
| MovingAI | V2 | 11/12 | 93.923 s | 130.749 s |
| MovingAI | Mixed Full | 11/12 | 92.302 s | 130.401 s |

`capped TTF` 对每个失败记 600 秒。Mixed Full 的总体 capped TTF 优势主要来自它多解决了一个生成地图实例，而不是在 MovingAI 高负载实例上运行得更快。MovingAI 子集上三个控制器均成功 11/12，Adaptive 的实际时间反而最低。

## 修正后的结论

旧正式结果降级解释为：

> 冲突对数均衡、但计算负载和来源混杂的固定 cohort 上的配对 Pilot。

以下结论仍成立：

- 三种控制器的初始状态一致，动作和可行性统计没有实验错误。
- V2 和 Mixed Full 在该固定 cohort 上显著降低冲突 AUC。
- Mixed Full 多成功一个实例，因此 capped TTF 优于 V2 和 Adaptive。

以下结论不能由该实验支持：

- Mixed Full 在等难度 MovingAI 上比官方 LNS2 更快。
- 数百个初始冲突天然代表高计算负载。
- 地图来源或地图结构本身造成了全部时间差异。

## 下一轮数据门槛

下一轮正式控制器运行前，qualification 必须记录冲突事件、活跃冲突 agent、最大冲突分量、总路径 cost 和初始 PP generated/expanded nodes。数据选择至少满足：

1. 冲突对数低、中、高均有覆盖。
2. 初始 PP 负载低、中、高数量近似均衡。
3. 生成地图和 MovingAI 在每个 PP 负载层均有实例，不能继续完全分离。
4. 分别报告端到端 TTF、reset/initial PP、repair-only、实际执行时间和失败 capped TTF。
5. Adaptive/V2/Mixed Full 都按冲突层、PP 负载层和来源输出配对结果。

若现有地图无法产生来源与负载重叠，应先扩大生成地图尺度和 agent 数，并降低部分 MovingAI agent 数；数据门槛通过后才值得重新运行正式108个或更多 episode。

新选择器使用 `3 个冲突层 × 3 个 PP 负载层 × 每格 2 个生成实例和 2 个 MovingAI 实例`，共36条；任一格缺少来源重叠时不会生成正式 schedule：

```powershell
python scripts/run_balanced_wall_clock.py select-load-balanced `
  --dataset <新数据集> `
  --qualification <带 initial_complexity 的新 qualification> `
  --cohort <新 cohort 目录>
```

## 入口

```powershell
python scripts/run_balanced_wall_clock.py audit-difficulty `
  --collection build/initlns-v2-mixed-balanced-formal-v2c `
  --cohort build/initlns-v2-mixed-balanced-cohort-v2c `
  --report build/initlns-v2-mixed-balanced-difficulty-audit-v2c
```

完整本地输出包括 `difficulty_audit.json`、`difficulty_episodes.csv` 和 `difficulty_audit_zh.md`；紧凑证据位于 `artifacts/initlns-v2-mixed-balanced-wall-clock-v2/difficulty_audit.json`。
