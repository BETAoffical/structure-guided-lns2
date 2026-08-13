# STRIDE Successor-State Inventory V1 报告

## 结论

全部 270 条 PreTail 强制首动作轨迹都能无损重建第一步后状态，但“首步是否成功”或“后状态下一轮是否原地 no-op”都不能作为长尾入口标签。

最强证据是：90.7% 的强制首动作立即严格降低冲突，总体仍有 30.7% episode 在 200 轮/300 秒边界内右删失；在 `maze-128-128-2` 上三个动作臂的首步严格下降率都是 100%，最终成功率却只有 16.7%–29.2%。这说明动作可以取得明显的一步进展，同时把求解带入后续难以持续修复的状态。

下一步应测量候选动作造成的后继状态中，完整 V2 基础候选池的 PP 可修复性分布。不能把 observed no-op、单一下一动作或未来 continuation 结果直接当在线特征。

## 数据与完整性

- 45 个冻结入口状态；
- 2 个严格配对首动作 PP seed；
- `actual_selected`、`one_step_oracle`、`coverage_diverse` 三个固定臂；
- 合计 270 条 bounded continuation；
- 234 个不同完整后状态，223 个不同 repair-structure 后状态；
- 270/270 首动作、后状态、候选 ID、邻域、PP seed 和 PP order 完整匹配；
- 0 非法动作、0 trace SHA 错误、0 前后状态指纹错误；
- 没有新求解、训练或运行时修改。

## 指纹语义修正

初版错误地用完整状态指纹判断自循环。完整指纹包含 iteration 和 low-level 计数，因此路径与冲突完全不变时也会随每轮改变，初版把所有 no-op 都错误报告为 0。

该初版已保存在 `build/stride-successor-state-inventory-v1-rejected-full-fingerprint`，不作为科学结论。修正版规定：

- 完整状态指纹只用于验证 delta trace 无损重建；
- 路径/冲突是否原地不动使用 `repair_structure_fingerprint`；
- cohort、轨迹、结果、完整性门槛和 claim boundary 均未改变。

这避免重复历史上“按错误状态身份处理时间序列，导致重复历史全部归零”的失败。

## 三个动作臂

| 动作臂 | 首步 repair-state 改变 | 首步严格降冲突 | 后继 exact no-op 率 | no-op P95 / max | bounded 成功 | 右删失 | normalized AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| actual selected | 91.1% | 91.1% | 22.2% | 9 / 18 | 62.2% | 37.8% | 0.1314 |
| coverage diverse | 82.2% | 81.1% | 15.6% | 3 / 8 | 67.8% | 32.2% | 0.1102 |
| one-step oracle | 100.0% | 100.0% | 26.7% | 4 / 23 | 77.8% | 22.2% | 0.0814 |

Oracle 的后继 exact no-op 率最高，但完整成功率和 AUC 最好。因此“避免任何下一轮重复”会系统性排斥一部分优质动作，重现 V3 为规避风险而偏向保守邻域的问题。

进一步按 no-op 是否出现分组：

- actual selected 中，有 no-op 的 20 条成功率 75.0%，无 no-op 的 70 条成功率 58.6%；
- coverage diverse 中，有 no-op 的 14 条全部成功；
- one-step oracle 中，有 no-op 的 24 条成功率 95.8%，无 no-op 的 66 条成功率 71.2%。

这是 observed continuation 的描述性结果，受后续控制器和可执行轮数影响，不能解释为 no-op 有益；但足以否定“出现一次后继 no-op 就是危险入口”的规则。

## Adverse 与地图证据

在 17 个 adverse 案例的两个配对 seed（每臂 34 条）中：

| 动作臂 | 首步严格降冲突 | bounded 成功 | 右删失 | normalized AUC |
|---|---:|---:|---:|---:|
| actual selected | 91.2% | 32.4% | 67.6% | 0.2267 |
| coverage diverse | 91.2% | 38.2% | 61.8% | 0.1667 |
| one-step oracle | 100.0% | 47.1% | 52.9% | 0.1403 |

三个臂的一步质量都很高，后续结果仍大幅不同。差异不是“PP 第一次能否改动”，而是改动后形成的冲突结构是否仍有一组可靠可修动作。

`maze-128-128-2` 更明确：所有动作臂首步严格下降率都是 100%，但 actual/coverage/oracle 的成功率分别只有 16.7%、29.2%、29.2%。单步下降在这个地图上几乎没有入口判别力。

## 被否定与仍未证明的内容

本阶段否定：

- 用第一次 PP 是否成功判断会不会长尾；
- 用下一轮是否 exact no-op 作为二元风险标签；
- 只要禁止重复邻域/重复状态就能从入口消除循环；
- 一步 Oracle 已经证明能避免长尾。

本阶段尚未证明：

- 后继状态 PP repairability 可以稳定预测长尾；
- 哪种入口候选在未知状态上最好；
- 新候选池或排序器会降低 TTF。

## 下一步：bounded Successor-Repairability probe

1. 保留 270 个 occurrence 身份，不按结果筛选；相同 repair-state 可以共享计算缓存，但不能在数据表中合并历史位置。
2. 对 223 个不同 repair-state 无损恢复后，仅生成冻结 V2 原始基础候选池。这里 V2 候选池是统一测量仪器，不是新候选池或部署方案，也不引入 StructPool 固定 `8/16/24/32`。
3. 对每个后状态候选先运行 8 个严格配对 PP seeds，保存路径改变、严格冲突下降、no-progress 和 after repair fingerprint；候选 × trial 为原子作业，16 workers，每作业 300 秒硬上限，首错/超时停止并可恢复。
4. 状态级输出完整分布：V2 anchor、最佳候选、Top-k、零可修候选比例、严格下降率和 seed-half 稳定性。不能先压成“会/不会长尾”。
5. 只把 bounded continuation success/AUC 当离线关联验证；不得进入在线特征。若后继 repairability 与右删失、AUC及配对动作差异没有稳定方向，停止该目标，不训练。
6. 通过后再用 MultiValue 已保存的 16-seed 强制首动作后状态做独立标签稳定性确认；仍不直接进入 TTF。

最终候选池和排序器仍需另行解决。本探针只回答一个更基础的问题：某个入口动作是否制造了一个连冻结 V2 完整候选池都难以继续修复的后状态。
