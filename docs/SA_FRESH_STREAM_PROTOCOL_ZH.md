# 冻结无决策上限模型：新随机流重复验证

## 问题与边界

上一轮新模型成功 11/16，Dual16 为 10/16、旧 condition 为 10/16、未训练探索为 9/16。
净增量小，需要在不训练、不改模型的前提下检验随机流稳定性。本轮不是新地图验证，也不是正式 TTF。
两张已查看开发地图、八个原始任务/solver 条件保持不变。

## 固定范围

- phase 保持 `crossfit-comparison`，仅将 replica 从已查看的 `[0,1]` 换为未使用的 `[2,3]`。
- 五个方法：Official+SA、Dual16+SA、未训练探索、旧 bounded_condition、新 uncapped_condition。
- 每方法 16 个 episode，共 80 个全新 episode；不借用旧随机流的基线。
- 模型、特征、候选、PP、SA 温度和原生二进制均冻结，依赖递归 SHA 校验。
- 决策上限为 null，native repair 上限为 0；256 只保留作历史输入归一化常数。
- 每 episode 最多 25,000,000 repair generated nodes；PP 20 秒、episode 900 秒、进程 fuse 960 秒保护不变。
- 默认 20 个独立进程，四批运行；安全暂停在当前批结束后生效。非正式计时，耗时不得解释为 TTF。
- 准入扫描历史 SA registration 的 phase/pair/replica 身份，拒绝已经注册的随机流；训练地图和随机流也须隔离。

## 判定预先固定

主要看本轮 16 对，不把新旧批次合并来挽救结论。新模型相对旧 condition 净成功增量 >0，
相对 Dual16 >=0，相对未训练探索 >0，记为“新随机流上净收益重复”。允许个别失败和地图回退，
不要求完全支配，不自动晋级正式控制器。新模型同时低于旧 condition 和 Dual16，记为重复回退。
其他情况记为“净收益未重复”，保留各比较的成功数、胜负位置和地图差异，不等于证明永远无效。

历史批次并列展示；32 对合并数只能作为次要描述。两张地图不足以作确认性地图 bootstrap，
不宣称跨布局泛化。成功数量按固定节点预算比较；安全超时为未知，需要检查，不冒充节点预算失败。
不会依据中间表现追加 replica、改预算或重新训练。

## 输出与复核

入口 `scripts/compare_sa_fresh_stream.py prepare|verify|collect|audit|report|stop`；
显式 `collect --resume` 只复用完整且校验通过的 episode，部分输出或失败需先检查。
全量复核 80 条轨迹的状态、候选、概率、随机流、PP 动作、SA 接受、合法路径及最终冲突数。
同一任务/replica 的五种方法初始状态与 RNG 身份必须完全一致。
共同成功样本内单独报告节点、决策数、SOC、makespan、等待步数；失败不填零、不混用分母。
保留登记、原子结果、完成清单、审计与报告于 `build/sa-uncapped-fresh-stream-v1`。
执行前提交协议并保存 Git bundle；结束后保存结果说明及证据备份，不修改历史结果。
