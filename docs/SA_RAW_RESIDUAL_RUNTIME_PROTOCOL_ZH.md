# Raw actor连续采样与终局更新：固定开发协议

2026-09-25。只补齐已验证raw表达合同的隔离运行和一次完整终局更新，不切换正式Dual16/V2，不改native、PP、SA、候选池、特征或归一化，不做TTF。

## 1. 不再重复的工作

旧有界残差的表达限制已确认，但未被证明是长尾的原因。零初始化和构造表达见证不再重复作为研究结果。
旧温度/课程/探索及一步排序的负面结果保留。本次只检验新策略族能否从新完整轨迹中获得有用更新，不能重复拟合旧批次。

## 2. 原生准入

- 冻结`sa-raw-residual-contract-v1/prototype.initial.json`，base为actor-1。
- 冻结`sa-wall-clock-v1` native SHA；用已有六张Train图中两个不同地图的最短非零完整actor-1轨迹作精确复现控制。
- 相同reset、proposal、选择/PP/SA随机流：初末fingerprint、路径delta、候选、特征、概率、选择、节点、SOC/makespan必须一致。逐步native指标由原审计器完整校验；墙钟字段不要求相等。
- 新schema仅在独立进程显式适配，退出/异常恢复旧加载器；不得在线程间切换。旧加载器仍拒绝新schema。

## 3. 固定预算

- 已见Train六张地图、12个任务、solver seeds233/239，共24条件。不使用Validation/Test/OOD，不声称新地图。
- 新随机流`raw-terminal-train-20260925`，每条件4条完整轨迹，共96。初始化与父策略相同，不重复采一遍相同父策略。
- 不设决策次数上限；每episode修复generated nodes预算25,000,000。单PP安全20秒、episode安全900秒、外部fuse960秒，均非TTF。
- 20进程、每进程单BLAS线程；记录原子结果和批次锁，`stop`等待当前批完成。异常/安全中断记未知，阻止整批训练，不静默丢弃、补零或重试。
- 96条全部经过原生轨迹审计才训练；源代码、native、输入、模型与配置SHA登记。输出独立`build/sa-raw-residual-runtime-v1`。

## 4. 一次终局更新

完整可行=1，节点预算耗尽=0，安全中断=未知。leave-one-replica-out终局baseline；地图、条件、完整episode等权。
每条信用乘整段`sum(log pi)`，不除轨迹长度、不按决策截断、不使用剩余冲突或单步结果作为标签。
129→32→1 raw修正，冻结base不再训练；固定同一梯度，最大L2步0.25、12次二分回溯。
更新相对直接行为策略检查平均KL≤0.002、轨迹平均KL≤0.1、最大状态KL≤0.02。零梯度或全部回溯不通过则停止，不改参数重试。
整批最多一次更新。不把通过KL写成成功率提升。Windows既有Torch CPU训练，WSL仅NumPy推理，不安装依赖。

## 5. 配对开发检查

只有更新模型通过Windows Torch/NumPy和WSL NumPy的概率及选择验证后才运行。
另一随机流`raw-terminal-compare-20260925`，24条件×2 replica×初始raw/更新raw，共96episode，相同初始状态及随机数。
初始raw已证明等于父策略；相同模型身份独立保存，比较不与训练轨迹混算。
报告新增/损失成功、各图成功、共同成功的generated/修复步数/SOC/makespan及地图bootstrap。wall safety不作为失败样本。
净成功增加是开发正信号，不要求每图全胜，也不自动晋级；净增加≤0不能声称成功率改善，条件性资源变化照实保留。
无论结果如何不自动追加训练轮数、改温度或启动TTF；新地图确认另立协议。

## 6. 入口

```text
python scripts/run_sa_raw_residual.py prepare
python scripts/run_sa_raw_residual.py controls
python scripts/run_sa_raw_residual.py collect
python scripts/run_sa_raw_residual.py audit --lane train
python scripts/run_sa_raw_residual.py train
python scripts/run_sa_raw_residual.py parity --platform windows
python scripts/run_sa_raw_residual.py parity --platform wsl
python scripts/run_sa_raw_residual.py compare
python scripts/run_sa_raw_residual.py audit --lane comparison
python scripts/run_sa_raw_residual.py report
```

controls/collect/audit/compare在冻结WSL环境，train/windows parity在现有Windows Torch环境。
`collect/compare/controls --resume`仅接受已完整校验的结果；`stop`不终止当前任务。
本阶段不重新声称静态迁移、独立泛化或长尾已修复。
