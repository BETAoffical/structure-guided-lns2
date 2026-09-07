# 高压力任务扩充：独立起终点重复

## 目的与边界

在不改变地图、密度、OD生成规则、控制器或求解器的条件下，增加一组独立起终点，减少结论对单次任务抽样的依赖。
本轮只生成并静态验证数据，不调用原生求解器，不训练模型，不启动计时。

原先每个“地图 × 密度 × OD模式”仅有一个任务生成seed。新增批次使用固定master seed `2026090802`，原批次为 `2026090801`。
两批都保留，不因冲突数量、是否容易求解或后续控制器结果替换任务。生成失败必须记录，不重抽seed。

| 项目 | 原批次 | 新增批次 | 合计 |
|---|---:|---:|---:|
| 仓库地图 | 8 | 同样8张 | 8张不同地图 |
| 每图密度 | 15%、20%、25% | 相同 | 3档 |
| 每图OD模式 | balanced、bottleneck_eligible | 相同 | 2类 |
| 每个组合的起终点抽样 | 1 | 1 | 2 |
| 静态任务 | 48 | 48 | 96 |
| 后续计划solver seeds | 61、62 | 61、62 | 每任务2个 |
| 后续任务-seed条件 | 96 | 96 | 192 |

瓶颈最短路存在性约束仍为40%，不代表PP必须经过指定格点。密度不是冲突数量，任务名称也不是求解难度标签。

## 三类随机性的区别

- 新任务生成seed：改变起终点抽样，检查结论是否依赖某一组OD实例。
- Solver seed：在同一任务上重复求解，评估算法随机流的影响。
- 新地图：增加独立布局样本，本轮没有增加。因此地图级bootstrap仍只有8个地图组，不能把192个条件当作192张独立地图。

增加任务不能消除计时中的CPU竞争、温度及电源模式波动。后续仍须串行配对、均衡方法顺序、保存环境记录。
本轮仍是复用地图的开发实验，不构成独立跨地图泛化证明，也不保证置信区间必然缩小。

## 保存与验证

- 原批次：`configs/path_quality_pressure_pilot_v2.json`、`build/path-quality-pressure-pilot-v2/`，原样保留。
- 新批次：`configs/path_quality_pressure_tasks_replica2.json`、`build/path-quality-pressure-tasks-replica2/`。
- 使用既有静态生成器，不修改已登记实现或原864项排程。

```powershell
python scripts/prepare_path_quality_pressure.py --config configs/path_quality_pressure_tasks_replica2.json
python scripts/prepare_path_quality_pressure.py --config configs/path_quality_pressure_tasks_replica2.json --verify
```

静态检查包括唯一起点、唯一终点、可达性、距离限制、受约束agent配额及最短路存在性。
跨批次还须核对地图和因素组合相同、任务seed与ID不重叠、实际起终点不同，保存输出SHA。
静态通过不等于联合无冲突可解，也不等于规定时间内可解。

## 后续工作与时间

新增48任务尚需96次Pilot reset检查，之后还需60/120秒预算专属准入。不要直接复用不同任务的旧anchor。
合并正式排程必须新建版本，登记两批输入SHA；旧864项排程不包含新增任务。计时仍需单独授权。

若两批全部运行原三个协议，每组为96任务 × 2seeds × 3方法 = 576个episode，共1728个。
60/120秒两组的名义规划预算合计为576 × (60+120) = 28.8小时。
三组都用满规划预算时合计48小时；按外部fuse累计为105.6小时，另加调度开销。这些是预算，不是耗时预测。
因此应在episode边界可安全暂停的前提下分批计时，不能把本次扩充准备理解为授权连续运行整个扩充排程。

## 本轮准备结果

- 新增48/48任务静态有效，0生成错误；两批合计96个有效任务。
- 48个对应因素组合使用相同地图、相同agent数量，不同起点和终点；任务ID和seed不重叠。
- 新seed与Pilot v1、Pilot v2及原始来源任务均无重复；没有生成失败后重抽seed。
- 新批次严格SHA复核通过；跨批次检查保存在 `build/path-quality-pressure-tasks-replica2/replication_audit.json`。
- 原864项排程、192项预算准入仍通过身份验证，计时授权文件不存在。
- Windows相关单元测试62项通过。只新增配置、测试和本文档；未修改生成器、控制器、native或冻结模型，未重跑Linux CTest。
- 新增求解器调用0次，正式计时0次；新增任务的PP冲突分布仍未测量。

新增准备报告SHA256：`7b002713c165552049b400a9f7918018c8cb01b8536674de1567efd70d70c236`。
跨批次检查SHA256：`9b9415bf861a05cc9d327e68a146aec3136185310ed401ca9b1af30ad799079d`。
