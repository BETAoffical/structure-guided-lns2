# STRIDE Repairability Stability V1 报告

## 结论

16-seed 的 PP repairability 目标明显比历史 MultiValue/Cost-to-Go 稳定，但尚未满足逐地图稳定性门槛，因此不训练模型。

它同时暴露出一个更根本的边界：在 adverse 案例的首个结构动作上，结构选择本身的即时 PP 可修复率已经达到 0.919，V2 为 0.971；这些动作仍可能在取得一次进展后形成长尾。由此可知，单纯预测“当前动作的 PP 是否能成功改变路径”不能从入口预防循环。下一目标应是 **候选动作造成的后状态 repairability**。

## 数据与边界

- 78 个冻结状态、90 个逻辑检查点、2,502 个候选、40,032 个 PP 试验；
- 固定半组为 trial 0–7 与 8–15；
- 候选集合先受 V2-relative 一步质量约束，V2 锚点始终保留；
- 没有新求解、训练、未来轨迹、TTF 或运行时接入。

初版将两侧全并列成功率向量的 Spearman 记为 0。全并列时没有可区分的 repairability 排序，属于决策等价而不是反向不稳定；因此透明修正为：两半都为常量记 1，只有一半为常量记 0，其余使用普通 tie-aware Spearman。cohort、门槛和 claim boundary 均未改变。

## 总体稳定性

| 指标 | 结果 | 门槛 | 总体 |
|---|---:|---:|---|
| mean Top-3 overlap | 0.8037 | >=0.80 | 通过 |
| median rank correlation | 0.9113 | >=0.60 | 通过 |
| V2-relative direction agreement | 0.8140 | >=0.70 | 通过 |
| mean cross-half regret | 0.0167 | <=0.02 | 通过 |

90 个检查点中，43 个在两个半组均为全并列 repairability，39 个具有可计算排序，8 个只有一半为常量。

## 逐地图门槛

总体通过不能掩盖地图集中失败：

- `maze-128-128-1`：两个检查点组 Top-3 overlap 均为 0.7917，略低于 0.80；
- `maze-128-128-2`：首动作与重复入口两组全部通过；
- `maze-32-32-4` 重复入口：方向一致率 0.6147、regret 0.0331，失败；
- `maze-32-32-4` 首动作：Top-3 overlap 0.6863、regret 0.0257，失败。

因此 `stability_passed=false`，不能用总体平均启动训练。

## 与历史失败的区别

- 历史 MultiValue 初始 8-seed Top-3 overlap 只有 0.4028、rank correlation 0.1883；扩种子后仍受 teacher 和长时域策略影响。
- 当前目标总体 Top-3 0.8037、方向一致 0.814、regret 0.0167，说明 PP repairability 是更接近机制、噪声更低的量。
- 但它只描述当前 PP。首动作即时 repairability 很高仍会形成低 repairability 后状态，所以直接训练它会解决“进入平台后如何选”，不能保证“从入口不形成平台”。

## 下一步

### 1. 不直接加种子训练当前标签

额外种子可能改善 `maze-32-32-4` 的估计方差，但不能修复标签时序边界。先增加到 32 seeds 后训练，会再次把资源投入一个不能从入口解释长尾的目标。

### 2. 定义 Successor-Repairability

对每个入口候选 `c`：

1. 严格配对地执行一次 `c`，得到后状态分布 `s'`；
2. 不运行长时域 teacher，只在每个 `s'` 上构造冻结 V2 锚点和同一候选池；
3. 用配对 PP seeds 测量后状态中最佳可接受动作的路径改变率、严格冲突下降率和 no-progress；
4. 标签是相对 V2 的后状态 repairability 分布，同时保留当前一步质量硬约束。

这回答的是“当前动作是否制造了一个随后几乎修不动的状态”，而不是“未来 128 轮谁最终赢”。

### 3. 先复用已有强制 continuation

在启动嵌套 PP 采集前，先检查已有 PreTail、TailSwitch、CycleTransition 和 MultiValue 强制首动作产物是否保存了可恢复的第一步后状态。能恢复则直接对这些冻结后状态做 PP repairability；不能恢复才预注册新的最小采集。

### 4. 晋级条件

Successor-Repairability 必须在固定半组和逐地图上通过稳定性，再允许训练 anchor-relative gate。最终仍需 bounded continuation 证明平台从入口减少，并用完整 paired raw TTF 验证；本阶段没有长尾避免或性能声明。
