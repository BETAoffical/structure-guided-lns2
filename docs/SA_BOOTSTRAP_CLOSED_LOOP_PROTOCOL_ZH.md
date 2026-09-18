# SA 自助采样选择器：有界闭环开发诊断

## 问题与边界

上一轮 47 状态的 H32 离线分析显示，自助采样投票的期望完成率优于均匀随机，但不如单 GBDT。离线结果不说明连续使用后的闭环表现。本轮只验证概率选择是否能改善连续修复，不增加置信门控，不重做一步标签，不训练 RL，不修改正式控制器。

代码内 `posterior` 仅是实验臂名称：这里是地图 bootstrap 成员抽样，不是经校准的贝叶斯后验，也不是模型预测的真实成功概率。

## 冻结设计

- 使用原 `sa-linear-closed-loop-v1a` 的 8 张已查看开发地图，每图按固定哈希选 1 个任务，不读取性能后更换。与 47 状态训练地图隔离，但不是新的独立确认集。
- 两个 solver seeds 为 227、229，共 16 个配对实例，四臂 64 episodes。保留零冲突；reset 必须与历史初始 fingerprint 相同，不以活跃比例重抽任务。
- `frozen`：原 Dual16+SA；不是 Official+SA。
- `gbdt`：已有 47 状态完整训练的单 GBDT，模型字节不变。
- `posterior`：相同参数，按地图 bootstrap 训练 20 个成员；每步独立均匀抽一个成员，再按其排名选择。这等价于当步按成员赢家票数分布抽候选，不是对分数取平均。
- `uniform`：从相同候选子集中均匀抽取。
- 各臂从相同的官方生成规则与 Dual16 候选池中取得最多 4 个候选，包含冻结 anchor。路径分化后，候选池可以自然分化。PP/SIPPS、SA 温度、PP/接受随机流和候选 seed 不变。选择随机流使用独立固定哈希，臂名称只区分选择随机流。
- 原训练为 47 状态、8 地图；地图均衡状态权重乘 bootstrap 次数。未抽到的地图不以零权重混入树的分箱。20 个成员与全部特征 schema、训练输入 SHA 一起冻结。

## 资源和执行

每 episode 最多 128 次修复，步间 generated nodes 上限 2500 万，PP 安全界 20 秒，episode 安全界 300 秒，进程 fuse 360 秒。最后一步可能越过节点界，必须记录；这不是生产算法的修复次数限制。最多 20 个独立进程，模型各线程数 1。

总计最多 8192 次修复，不含 reset。并行 wall-time 不作为 TTF。episode 安全时钟从 reset 和模型加载后开始，进程 fuse 覆盖整个作业，两者不混用。wall/PP 超时标为 censored unknown，而非证明无解。

复用旧执行器和逐步审计器，只在子进程内用可恢复的局部适配替换模型加载与选择；不改历史源码。记录完整模型分数、被抽成员、候选、特征、动作和状态 delta。先完整验证一个配对实例的四臂，再在同一冻结协议下补齐，不据此修改参数。

支持逐 episode 原子完成凭据、输出 SHA、`--resume`、批次安全停止。非完整输出或身份不一致必须先审计，不能覆盖重试。`request-stop` 让当前批次收尾；继续前需确认并移除本轮停止标记。异常退出不自动换 seed。

## 分析与解释

报告全部 16 个实例的完成数、工作预算终止与安全界未知数；报告按地图配对的完成差与 5000 次地图 bootstrap，以及未知结果上下界。共同成功实例单独比较 nodes、修复次数、SOC、makespan。额外报告 anchor 偏离、状态重复及特征越界，不能以较少重复或较少单步开销替代完成优势。

不要求每图必胜，不设置新的 5% 晋级门槛。对冻结、单 GBDT、均匀随机三者分别比较；只有相对强对照有一致的闭环正信号，才值得独立扩大验证。失败时先判别标签策略偏移、候选覆盖和资源截断，不追加成员数、seed 或重抽任务。本轮无论结果如何都不自动晋级正式控制器。

## 命令

Windows 现有 sklearn 1.5.0 只用于 `freeze`；WSL 冻结 native 用于正确性与闭环，禁止安装新依赖。

```text
python scripts/run_sa_bootstrap_closed_loop.py prepare
python scripts/run_sa_bootstrap_closed_loop.py freeze
python scripts/run_sa_bootstrap_closed_loop.py verify-model
python scripts/run_sa_bootstrap_closed_loop.py qualify
python scripts/run_sa_bootstrap_closed_loop.py collect --limit 4
python scripts/run_sa_bootstrap_closed_loop.py analyze --limit 4
python scripts/run_sa_bootstrap_closed_loop.py collect --resume
python scripts/run_sa_bootstrap_closed_loop.py analyze
python scripts/run_sa_bootstrap_closed_loop.py verify
python scripts/run_sa_bootstrap_closed_loop.py request-stop
```

运行前冻结代码、配置和输入登记。本地提交和 bundle 保留回滚点；远端状态单独报告。正式 TTF、Official+SA 四路性能结论和新地图确认不在本轮范围。
