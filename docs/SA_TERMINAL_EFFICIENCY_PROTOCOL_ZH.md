# 完整任务成本训练对照

## 目的与历史边界

当前 raw-1 相对 Dual16+SA 有高负载和总体点估计正信号，但较低负载更慢。
96条原训练轨迹只有20条提供非零终局更新；输出头零初始化后仅更新一次，隐藏层参数未变。
本轮区分继续训练本身与完整成功轨迹成本的增量，不把它们归因于更大网络。

历史的一步成本、预测剩余轮数、H4/H32、成本到达模型、状态基线、课程、tabu、温度扫描和简单扩池已做过。
本轮不重复这些机制；不会让失败中的局部冲突下降成为成功标签，不预测剩余修复次数。
二元续训曾有负面结果，本轮保留它作为同数据对照，而不是预先认定续训有效。

## 冻结与数据

- 当前分支起点 d234691，完整Git备份保存在 build/pre-sa-terminal-efficiency-20260927.bundle。
- parent为raw-1，文件SHA为fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84。
- 官方、Dual16、parent、129维特征、归一化、候选、PP/SIPPS、SA及冻结native不修改。
- Train沿用原六张Train仓库图、24个任务/solver条件；当前parent新phase采集4次独立重复，共96条完整轨迹。
- 比较沿用另外六张同族图、24条件，新phase两次重复，parent/A/B共144条。
- 比较地图与Train地图、任务隔离，但已在前轮查看，因此本轮明确属于开发对照，不重新声称独立泛化。
- 比较结果不得进入训练。不同方法共享初始fingerprint以及选择、PP、SA随机流定义。

## 唯一两个更新

A：R = success，作为相同当前策略数据上的二元续训对照。

B：R = success * [1 + 0.1 * (1 - min(G / 25000000, 1))]。
G是从初始化完成到实际首次可行的全部修复generated nodes，不是单步成本；初始化不受选择器影响。
成功得[1,1.1]，正常预算失败得0；原子修复越过节点边界后成功仍得1。
保护超时是unknown，不补0、不丢弃后训练。全失败条件仍可能零信用，这条路线不声称解决这一盲区。

0.1为本次固定系数，不扫权重。它是有界次目标，不是数学上保证成功率不下降的词典序约束。
整体期望奖励可能仍有成功率/成本取舍，所以必须报告实际新增和丢失成功。
节点数是并行条件下的硬件无关代理，不是TTF，更不包含控制开销；实际加速必须以后单worker验证。

两臂从同一个raw-1分叉、同一批96条数据、相同map-equal权重与leave-one-replica-out基线。
对整条轨迹log概率求和，不按轨迹长度重加权。固定原KL限制和回溯步幅，每臂仅一次梯度更新。
固定父策略和非零raw头允许后续隐藏层获得梯度；不增加层数或更换特征。
零梯度或没有符合偏移限制的步长时停止，不追加梯度、数据或系数扫描。

## 执行与失败处理

默认20个独立进程，不做时间比较。无决策次数上限，25M节点预算。
沿用PP保护20秒、episode保护900秒、进程fuse960秒；触发保护时unknown，整阶段停止检查，不充作失败。
逐episode原子结果，逐步trace，全量路径/概率/动作/工作量审计后才训练或汇总。
整组锁、批次进度、显式resume；STOP_AFTER_BATCH等当前最多20个作业完成后安全停止。
中断不自动删除partial、不自动重跑；训练部分写出后也必须检查，不允许覆盖。
原工程等价验证不重复，复用已有运行器与native；只验证新增目标与新模型跨平台概率和选择。

## 分析

报告三组配对：A-parent、B-parent、B-A。
主要是全部48位置的成功数、新增/丢失成功、共同成功完整节点数，另列修复次数、SOC和makespan。
报告逐图、逐密度结果，5000次地图级bootstrap。不要求每例胜出，保留实际取舍。
开发正信号：成功数不降低，并且净成功增加或共同成功节点数至少少5%。
这仅决定是否值得后续串行TTF，不是自动晋级；置信区间、地图数和丢失成功同时公开。
无信号则停止这一更新配方；不得用单步下降、较少剩余冲突或挑选子集改判通过。
无论结果如何，本脚本不启动正式TTF、不替换默认模型。

## 入口

```text
python scripts/run_sa_terminal_efficiency.py prepare
python scripts/run_sa_terminal_efficiency.py collect [--resume]
python scripts/run_sa_terminal_efficiency.py audit --lane train
python scripts/run_sa_terminal_efficiency.py train
python scripts/run_sa_terminal_efficiency.py parity --platform windows
python scripts/run_sa_terminal_efficiency.py parity --platform wsl
python scripts/run_sa_terminal_efficiency.py compare [--resume]
python scripts/run_sa_terminal_efficiency.py audit --lane comparison
python scripts/run_sa_terminal_efficiency.py report
python scripts/run_sa_terminal_efficiency.py stop
```

采集、审计及WSL parity用冻结native的WSL环境；训练和Windows parity用已有build/venv-graph。
输出build/sa-terminal-efficiency-v1；所有源码、配置、协议、输入SHA在采集前固定。
