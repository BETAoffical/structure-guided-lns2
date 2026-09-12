# Native shadow 去冗余独立验证

## 唯一改动

在独立 `NativeShadowTopology` 中，native 模式的 shadow 不再先构造未被读取的 Python TemporalConflictIndex；保留空路径检查、重新执行全量 native 冲突提取和从状态重新统计 heat。Python fallback 保留原路径。

不更改增量变化集合、shadow 每20次的频率、异常处理、模型、候选、特征、评分、PP、接受条件和最终 fingerprint。原失败版本及报告不动。

## 执行

复用上一轮配对 worker 和 collector 的同一 code object，仅用私有 globals 绑定新运行时、输出目录与登记校验。禁止全局 monkeypatch。

- 同样8个案例、792次逐状态配对与完整历史前缀；预期35次shadow，全部计入组件时间。
- 参考仍是冻结 `SingleFullCheckPool`，不是只和慢的增量v1相比。
- 单worker，逐状态交替调用顺序；相同的600秒案例fuse、300秒重放PP安全上限和原子保存/暂停/resume。
- 无端到端TTF试验，无自由rollout，无新模型和native。
- 与历史全量对齐heat、拓扑、特征、候选、分数、动作、路径与低层计数；任意错误停止。
- 单测要求native模式不构造Python索引，Python fallback仍构造并读取索引；全量shadow输出与旧版相同，损坏heat和空路径继续被拒绝。

沿用原门槛：总选择时间至少减少5%，至少6/8案例更快，全部等价性检查通过，才作为独立TTF确认候选。不降低标准，不调整shadow频率。若失败，按实际成本解释，不自动扩大底层重构。

输出 `build/sa-native-shadow-heat-audit-v2`，本地登记提交后才执行。新旧跨轮时间仅作背景，不把机器状态差异归因于代码；结论以本轮内部新旧配对为准。
