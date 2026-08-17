# Seed25 同状态短程回放诊断结果

## 结论

当前主要问题不是“Component16 和 Hotspot16 各只有一个候选”，而是冻结排序器及其 Copeland 聚合没有可靠表达两件事：

1. 当前候选在 native PP 下是否真的可修；
2. 该动作把后续 V2 带入了更容易还是更难修的状态。

Random 和 Room 的候选池里已经存在有效的 Hotspot16，但排序没有选中它；Random 还出现了结构候选加入后，纯 V2 winner 从 Target16 翻成较差 Target8 的 choice-set 效应。Warehouse 的 Component16 则确实是高价值动作。Maze 的三个暴露动作在这一配对 PP seed 下全部回滚，属于候选整体难修，不能归因于一次局部错排。

因此，直接增加更多结构候选会扩大当前排序/集合效应，不能视为修复。固定 size16 也不是统一根因：同样的 size16 在不同 family 和状态上分别出现成功、失败和轨迹反转。

## 执行边界

- 来源：`stride-structshell-maze32-n300-fourmap-quick-v2` 的 seed25 Dual16/Plateau traces。
- 状态选择只读取动作前 fingerprint、候选池和已选择 agent set，不读取该步结果或终局 TTF。
- checkpoint：Maze d31、Random d48、Room d30、Warehouse d0。
- 主动作：V2 anchor、Component16、Hotspot16；Random 额外回放 choice-set 诱导的 observed Dual V2@8 winner。
- 每个分支：恢复完全相同的 repair paths，使用同一派生 PP seed 强制首步，再执行最多六步 fresh V2。
- 共 13 个分支、91 次 PP；每次 PP 原子限时 5 秒，总进程上限 600 秒。
- 实际执行约 56.25 秒；91/91 完成，54 次失败全部原子回滚，0 TIME_LIMIT，0 执行错误。
- 这不是完整 TTF、不是原轨迹的精确继续，也不允许控制器晋级声明。

## 结果

`首步下降` 和 `7步下降` 均以冲突对减少量表示，越大越好。PP 是七次动作累计 native replan 时间；RB 是回滚次数。

| 地图 | 动作 | Dual池分数/排名 | 首步下降 | 7步下降 | 最终冲突 | 进展步/RB | PP (s) |
|---|---|---:|---:|---:|---:|---:|---:|
| Maze N300 | Component16（原Dual选择） | 15.029 / 1 | 0 | 0 | 496 | 0/7 | 0.981 |
| Maze N300 | V2 anchor | 13.999 / 2 | 0 | 0 | 496 | 0/7 | 0.908 |
| Maze N300 | Hotspot16 | 12.527 / 3 | 0 | 16 | 480 | 2/5 | 0.836 |
| Random high | observed V2@8（原Dual选择） | 12.635 / 1 | 0 | 2 | 262 | 1/6 | 0.226 |
| Random high | V2 anchor@16 | 12.576 / 2 | 0 | 9 | 255 | 2/4 | 0.174 |
| Random high | Hotspot16 | 11.860 / 3 | 11 | 18 | 246 | 3/3 | 0.346 |
| Random high | Component16 | 8.533 / 4 | 0 | 0 | 264 | 0/7 | 0.195 |
| Room | Component16（原Dual选择） | 15.594 / 1 | 0 | 30 | 42 | 5/2 | 4.460 |
| Room | Hotspot16 | 15.334 / 2 | 8 | 32 | 40 | 4/3 | 5.380 |
| Room | V2 anchor | 12.942 / 3 | 1 | 5 | 67 | 2/5 | 7.049 |
| Warehouse | Component16（原Dual选择） | 15.947 / 1 | 59 | 123 | 73 | 6/1 | 3.333 |
| Warehouse | Hotspot16 | 15.391 / 2 | 22 | 84 | 112 | 3/4 | 4.599 |
| Warehouse | V2 anchor | 12.676 / 3 | 0 | 128 | 68 | 6/0 | 3.287 |

## 逐图解释

### Maze N300

三个首步都因 `conflict_bound_exceeded` 回滚。原 trace 中 Component16 曾在另一未显式配对的 repair order 下把 496 降到 476，而本次配对 seed 下失败，说明 repairability 对 PP 随机顺序敏感。Hotspot 分支在后续 V2 第六、七步才下降到 480；这只能解释为“动作尝试加后续控制器历史”的轨迹效应，不能倒推 Hotspot 首步本身有效。

### Random high

这是最明确的局部排序和 choice-set 失败。纯 V2 池的 Target16 anchor 在合并结构候选后，被分数仅高 0.059 的 Target8 取代。Target8 首步回滚、七步只减少 2；未被选择的 Hotspot16 首步减少 11、七步减少 18。当前池中已经有更有效的结构动作，缺口首先在选择，不在数量。

### Room

Component16 比 Hotspot16 仅高 0.260 分，却首步回滚；Hotspot16 首步减少 8，七步最终为 40，Component 为 42，V2 为 67。这里同样是 family repairability 判别不足。不过 Component 首步失败后，后续 V2 仍走出较好轨迹，说明单步成败也不能代替短程价值。

### Warehouse

Component16 的首步排序是正确的：立即减少 59，明显优于 Hotspot 的 22 和 V2 的 0。这解释了结构候选在 Warehouse 中为什么能成功。但六步 V2 续跑后，V2 anchor 路径以 68 对 73 小幅反超 Component，而且 PP 累计几乎相同。即便首步非常成功，也不能直接推出完整 TTF 更优。

## 对问题的回答

- **排序器有问题吗？** 有明确的状态条件校准问题，不是实现选错最大分。Random 和 Room 中，分数最高动作并非同 seed 下最可修或短程最优动作；Random 还存在 Copeland 的候选集合依赖。
- **一个 Component16 加一个 Hotspot16 是否太少？** Maze 可能存在覆盖不足，但另外三图已存在有效候选，主要问题是没有可靠选中。用当前排序器直接增加候选更可能扩大过度暴露和 choice-set 扰动。
- **size16 是否错了？** 证据不足。相同 size16 在 Random/Room/Warehouse 已产生有效动作，Maze 又全部失败；没有同状态 8/16/24/32 反事实，不能把问题归咎于尺寸。
- **为什么结果不稳定？** PP repair order、初始 repair state 和早期动作都会改变后续可修性；冻结分数没有吸收这种 repairability/trajectory 差异。Maze 原动作跨 PP 顺序翻转、Room/Random family 翻转和 Warehouse 的短程反转都是同一问题的不同表现。

## 下一步边界

在改 ranker、加候选或搜索尺寸前，先用相同冻结状态对现有四个动作做少量多 PP seed 的单步重复，只估计 `成功率、回滚率、冲突下降分布、PP时间`。这是必要的，因为本轮每状态只有一个新配对 seed，Maze 已直接证明单次结果会随 repair order 翻转。

只有在某个 family 的 size16 跨 PP seed 仍系统性失败时，才对该 family 单独做 8/16/24/32 的同状态、ranker-free 回放。不要把多个尺寸重新一起放入 Copeland，也不要在修正 repairability 校准前扩大 Dual 候选池。

## 完整性

- Config SHA-256: `2a5531b709005d4053cd2e1420d59cec261b51131a1bb0074e4c625096ae675b`
- Plan SHA-256: `53d357f81a929b645e134bc9c21b9cf1f62b4f5dcd2eadf08a8fc6754eb1cfb6`
- Report SHA-256: `bf90557e7d37b8cd9063b55b89ba50cacf46f17eab4d888dbf6487ab32984b16`
- 13 branch files and all 91 transitions were independently rechecked against the source manifest, run-config and trace hashes.
