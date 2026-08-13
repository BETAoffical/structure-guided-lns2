# STRIDE Successor-Repairability V1 预注册协议

## 要回答的问题

首动作通常已经能降低冲突，但仍可能进入长尾。本实验不再判断“一步是否成功”或“下一轮是否 no-op”，而是测量首动作生成的精确后状态中，冻结 V2 基础候选池是否仍存在一组可靠可修动作。

这是一项机制探针，不是新候选池、排序器、控制器或 TTF 实验。

## 冻结数据和状态身份

- 使用 Successor-State Inventory 已无损重建的全部 270 个 occurrence；
- 保留 `case + trial + arm + successor full fingerprint + successor repair fingerprint`，不同历史位置不得合并为一条训练记录；
- 计算可以按 `task + repair fingerprint + exact agent set + PP trial + PP/SIPPS semantics` 共享；
- 准备和采集阶段禁止读取 bounded success、AUC、右删失和未来轨迹来选择状态、候选或作业。

## 测量候选池

每个精确后状态只生成冻结 V2 原始基础候选：

- family：`target`、`collision`、`random`；
- size：`4`、`8`、`16`；
- 每个 family/size 最多 2 个候选；
- 完全关闭 StructPool、SlotPool、TopologyBoundary 和 Causal/Closure 扩展；
- 用冻结 `v2-full` 124 维 realized-dynamic 模型标记 V2 anchor。

这里的 `4/8/16` 只是统一诊断仪器，不能解释为最终候选池尺寸规则，更不能复用 StructPool 的人工 `8/16/24/32`。

## 原子采集与上限

- 先准备全部 occurrence 和候选，但 preflight 只选每张地图一个 hash-only 后状态；
- preflight 使用 trial 0–1；通过后初始正式采集使用 trial 0–7；
- 候选 × trial 是独立原子检查点，16 workers，单作业 300 秒硬上限；
- 首个错误或超时即安全停止，可从同一 run fingerprint 恢复；
- 若 8-seed 标签不稳定，只能统一扩展全部状态和候选到 trial 8–15，不得只补测表现好的案例。

## 输出与门槛

保留每个候选的完整分布：repair-state change、严格降冲突、no-progress、replan success、normalized reduction 和 after-repair fingerprint。状态级报告 V2 anchor、观察到的最佳候选、Top-3、零可修候选比例及全池均值。

固定半组 0–3 与 4–7 的初始稳定性必须同时满足：

- mean Top-3 overlap 至少 0.80；
- repair-state change 和 strict-drop 的 median rank correlation 均至少 0.60；
- V2-relative direction agreement 至少 0.70；
- mean cross-half regret 不超过 0.02；
- 每张地图分别通过，且 0 错误、0 超时。

只有完整采集结束后，才把后状态 repairability burden 与已有 bounded success/AUC 做离线关联。主 burden 固定为 `1 - pool mean strict-drop rate`。它必须与 AUC 有预注册方向，并在同一 case/trial 的动作对中表现出一致方向；否则停止该目标，不训练。

## 声明边界

即使全部门槛通过，也只允许预注册独立的已保存后状态确认。不得在本阶段训练模型、接入运行时、运行 TTF，或宣称已经避免长尾。
