# 困难训练条件的 Official+SA 同起点参考

## 问题与边界

前一轮增加探索后，M04/d25仍没有成功终局，不能从全失败组假造成功方向。
本轮仅补齐严格同起点的封存 Official+SA 参考，判断它是否能提供可用成功见证。
不是新模型、正式TTF、泛化确认或单独选择器消融；不改官方PP、SA、模型、候选或native。
在95份顶层SA登记/报告中未找到以下四个完整条件的Official参考。历史其他M04不能按简称混用。

## 固定设计

- 困难图 `sa_linear_v1_m04_station_centric_0000`，对照图 `sa_linear_v1_m01_station_centric_0000`。
- 各取 `__task_0001` 的 `bottleneck_d25`，solver seeds 233/239，各四个replica，共16条新增轨迹。
- 原actor-1、actor-0探索、半均匀采样各16条只读复用；不读取M06/M07或Test/OOD。
- 同一初始fingerprint、25M修复generated nodes、冻结native和SA温度/接受随机流。
- 不设决策次数上限；256仅为旧模型输入尺度。一次完整修复可越过节点界限，计入真实节点数。
- PP保护20秒、episode保护900秒、外部保护960秒；触发保护记未知，不称无解，不自动补跑。
- workers=20，本轮只有16个作业，最多16个实际进程。非计时诊断不输出TTF优劣。

## 随机性与归因

Official动作必须仍为 `{"mode":"official"}`，不能为制造不同轨迹加入random_seed。
四replica改变SA接受uniform，不单独重置官方邻域/PP随机流；若实际轨迹重复，报告去重数，
不把16条写成16个独立实例。两张Train地图、四个完整初始化条件仍是本轮范围。
学习策略的显式修复会设置PP随机seed，因此相同SA流不代表两种策略PP顺序相同。
Official邻域大小和启发式权重继承封存模板。任何性能差异都包含候选、选择及搜索路径的差异。

## 执行与结果解释

入口 `scripts/probe_sa_hard_official_reference.py`：prepare / dry-run / collect / audit / report / all / stop。
独立输出 `build/sa-hard-official-reference-v1`，原子文件、锁、整批安全停止和显式resume复用旧实现。
使用 `reference_registration.json` 标明复用起点，而非伪装成新条件登记。
正式执行前固定代码/配置/测试/协议提交，记录所有输入SHA；旧运行器一字不改。

1. M04有成功：获得完整成功见证；下一步只读/独立重放核对当前候选池覆盖，不能直接归因模型评分。
2. M04仍全失败且M01全成功：困难状态在同预算下也难住参考；停止此处重复探索/重训，不称不可解。
3. 对照也损伤：先检查参考配置及轨迹，不依据两图差异作普遍因果结论。
4. 任一未知/错误/身份不一致：完整参考不成立，保留记录并检查，不补成失败零分。

主报告含每条件成功数、每地图配对新增/损失、冲突序列、最小/最终冲突、节点/步数、
共同成功的SOC/makespan/wait及实际轨迹去重。无候选池调整、自动训练、晋级或正式计时。
