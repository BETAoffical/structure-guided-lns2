# 第二结构核心邻域：固定动作续跑诊断

## 目的与边界

前一轮 `SA_STRUCTURAL_FOCUS_RESULT_ZH.md` 冻结了15个根状态各一个新邻域。
本轮只回答：在这些根状态，把原 Dual16+SA 的第一次选择替换为已经登记的新成员集合，
后续仍用冻结 Dual16+SA，32次修复内的完成情况是否改善。

不是一步最优标签训练，不是在线新策略，也不是正式 TTF。8张地图此前已查看，
且15个根状态是能够产生不同成员的条件子集，不能外推到全部地图或全部长尾。
11个根位于第4次决策，1个位于第16次，3个位于第32次；主要检验早期分流机会。
替换2至16名成员，不把全部干预称作微小调整。

## 固定比较

- 全部15个根，不依据任何续跑结果取舍。
- 原始 anchor 与固定 second-core 候选，均为实际16车集合。
- 每臂8条独立 reset/prefix replay 分支，共240条；同根同 trial 的 PP seed 和退火 draw 相同，种子不含候选ID。
- 第一次强制动作之后最多31次冻结控制器动作，总 H32；成功即止。
- 新候选仅加入该分支第一次强制动作的诊断池，不评分、不加入后续在线候选池。
- 保持原 PP+SIPPS、SA 温度和接受逻辑、proposal/评分器、模型与 native SHA。
- 原生 SHA：`5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。
- 完整重放原初始化、动作、接受随机数、历史信息和根候选池，重新从实际恢复状态生成已登记的新集合。
- 不读取成功见证，不扫描第三个核心，不增加候选、trials或参数。

## 预算与可靠性

沿用已有资源上限：每次PP 5秒、每分支180秒（含初始化、prefix、根检查和续跑），20个进程。
最多7680次H32修复，另最多2496次prefix修复。每分支独立文件、原子提交、SHA回执与运行锁。
无解/超时不能混淆：H32完整但仍有冲突是“此H32未完成”；PP预算或进程 fuse 是“未知”。
不自动补跑未知，不把并行运行耗时当成 TTF。

正式诊断前，选择最早与最晚的根各两臂、trial0，共4条，全部独立重复一次：
8条预检不计入240条结果；必须科学字段完全复现、零动作/重放错误、无预算删失。
预检只控制完整性准入，不以新候选是否更好决定是否采集。

## 评价预先约定

主结果为H32完成率；报告每根、每图、总120对trial的新增完成与受损完成。
同时给出按根等权与按地图等权的差值。5000次地图级paired bootstrap，固定seed。
保留全部根和未知结果，用最坏/最好情形给出完成率及差值区间，不删除未知根。
报告首步冲突变化、完整冲突序列、最终冲突、节点数和实际修复步数，作为原因诊断。

正向/负向区间仅标记开发性信号；跨零标记混合或不确定，不能据此说没有任何局部价值。
不要求每个根都赢，不据全局均值选择有利子集；同时列出收益与伤害。
本轮不自动晋级、训练、修改正式控制器或启动TTF。
如果有信号，下一步首先判断预动作特征能否区分收益与伤害，或做独立小样本确认；
如果无净收益，停止重复同一第二核心规则，不继续扫第三核心/温度/大小。

## 入口

`scripts/collect_sa_structural_focus.py` 提供 `prepare / verify / dry-run / preflight / collect / analyze / stop`。
WSL显式选择冻结模块：

```bash
env PYTHONPATH=build/linux/sa-wall-clock-v1:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 scripts/collect_sa_structural_focus.py preflight
env PYTHONPATH=build/linux/sa-wall-clock-v1:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /usr/bin/python3 scripts/collect_sa_structural_focus.py collect
```

`stop` 在本批已启动作业完成后暂停；`collect --resume` 逐项校验旧回执，不重封已变化的结果。
完整性错误先排查再续跑，已完成分支不自动重算。正式输出独立保存于
`build/sa-structural-focus-continuation-v1`，历史目录不覆盖。
代码协议先本地提交和增量备份，结果后再次提交/备份；当前不尝试远端推送。
