# SA历史选择器：采样修正协议

2026-09-16，用户授权“继续修正”。只修正历史状态和标注候选采样，保持原GBDT参数、146/127维特征、三头标签及评分规则。
旧v1b结果不覆盖、不改判定；新输出为`build/sa-history-selector-sampling-v2`。
不改Official、Dual16或SA运行时，不运行正式TTF或RL。

## 先只读准入，再采集

来源仍是96条既有高压力Dual16+SA开发轨迹。每条只检查决策0至512，逐步重建路径及历史。
所有判断在当前动作发生之前：不读当前动作结果、未来是否成功、未来TTF来选根。

每条episode最多登记三类的第一个符合点：

- `history_exact`：至少32步，至少16步未刷新最佳，池中集合最近32步尝试至少2次，且至少一次具有相同外部路径与相关冲突；同时存在未尝试集合。
- `history_changed`：同样重复/停滞条件，但上述重复集合均没有相同外部条件经验；同时存在未尝试集合。
- `progress`：至少16步，距离最近刷新最佳不超过4步，完整候选池中没有最近32步尝试过的集合。

每图最多2个历史根和2个普通根。历史根优先各取一种，类型缺失时只在本图历史根中补齐。
按固定seed与episode ID哈希排序；同episode只能提供一个状态，不用其他地图填补缺额。
“progress”是过去指标定义，不保证后续动作成功。

候选最多6个，依次保留：原赢家、优先同条件的至少两次历史集合、最高分未尝试集合、其他高分候选，最后按4/8/16与哈希补足。
这仅改变训练数据抽样，不是新在线候选生成器。模型不接收stratum或地图ID。

只读重建时提前保存根指纹、完整池SHA、候选ID和19维历史特征及SA温度/阶段值。
采集时以原native重放逐项完全核对，禁止只有采集完成后才发现历史列全零。

采集硬门槛：8张图、24至32个不同episode状态；至少10个历史状态来自至少5张图；其中至少4个同条件状态来自至少2张图；
至少12个普通对照；每个历史根的已选候选必须同时含至少两次尝试集合与未尝试集合。
未通过则只输出覆盖不足，不生成标签、不事后放宽规则。

## 执行及统计

- 最多32×6×4=768个新trial，另最多32个原动作对照；20个只读扫描或分支进程，重放根串行。
- 原native SHA、模型artifact SHA、路径和pool重放、进程死亡清理及完整receipt沿用v1b。
- 每次PP5秒，branch fuse90秒，根fuse600秒。超时是未知，不是不可行；错误停止后审查。
- 旧原生运行及原动作结果不可更改。完整采集后只训练本轮新根，不与v1b标签混合或以v1b选模型。
- 同样八折留图、每状态等权、四trial先聚合、5000次地图bootstrap；输出总组和各stratum结果。
- 精确条件/改变条件样本仍不是独立新地图证据；采样刻意富集历史，不能当自然发生频率或整体成功率。
- 允许结果有好有坏，不要求逐案例全胜；无论结果如何本轮不自动上线或开始计时。

## 复现与保存

入口沿用`train_sa_history_selector.py`，新增显式配置参数：

```text
python scripts/train_sa_history_selector.py prepare --config configs/sa_history_selector_corrected.json
python scripts/train_sa_history_selector.py collect --config configs/sa_history_selector_corrected.json
python scripts/train_sa_history_selector.py train --config configs/sa_history_selector_corrected.json
```

prepare与collect均在指定冻结native的WSL运行，避免跨平台浮点库导致预检历史值差异；train使用Windows已有sklearn。
修改前标签`pre-sa-history-sampling-correction-20260916`保留旧实现；旧报告、模型和完整结果ZIP SHA不变。
旧计划严格绑定实现SHA，复现旧实验应使用旧tag/源码备份，不篡改旧登记使新源码冒充旧实现。
