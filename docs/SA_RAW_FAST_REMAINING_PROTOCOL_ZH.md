# 冻结快路径模型：补齐剩余配对条件

## 目的与边界

上一轮工程结论已完成，不重跑36条原/快/Dual16对照，不重训，不再诊断同一未使用特征开销。
本轮仅补原48条件中未用快路径计时的36条件，判断快路径raw-1与Dual16+SA的成功/时间取舍是否依赖少数随机流。
这些地图已经被研究过，不能称为新独立确认地图或跨OOD验证。

## 冻结排程

- 六张原地图，每图bottleneck_d20/d25；补solver/replica组合(251,1)、(257,0)、(257,1)。
- 36条件×两方法=72条新计时，逐条件两种顺序轮换，各18次。上一轮(251,0)的12条件不重跑。
- 仅raw_fast和dual16_sa，无raw_reference、raw_parent或Official新增任务。
- raw-1文件SHA `fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84`，native SHA `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。
- 模型、候选、随机流标识、PP/SIPPS、SA温度和接受规则均不变。新输出目录不进入动作随机种子。
- 同一计时函数和串行collector代码体，只有登记加载器绑定新输出；单worker、120秒首解预算、240秒进程fuse，无节点/决策执行上限。

## 不做无效重复

36条Dual16前缀沿用旧192条实验中已验证且SHA固定的同条件证明，不声称重新执行了native。
36条新raw快路径各做3步真实native前缀校验，共108次非计时修复；最多20进程。
两类证明分别记录reused及native_steps_executed。正式运行仍逐episode检查初始fingerprint和模型身份，结果后逐条复算完整特征、动作和状态。

## 分析

- 先独立报告新增36对，不把前12对已知结果隐藏到总均值。
- 再报告两批合并48对；校验没有重复、缺失，模型和任务配对一致。按solver/replica、地图、密度分别报告。
- 成功率使用全部条件；TTF/路径质量只在共同成功任务上配对。失败不填0或把120秒写成原始TTF。
- 完成时间按交付 + makespan×0.5/1/2秒估算，不做机器人实测或第二阶段主张。
- 5,000次地图级bootstrap，不要求每图每例全胜，不修改主模型。成功数与TTF出现取舍时如实保留，不事后加入路由或重训。
- 不自动晋级；不根据中途结果删任务、改预算、追加replica。两批相隔不同时间，分别列出批次结果，不能把全部差异都归因于模型。

## 执行与恢复

入口：`python scripts/run_sa_raw_fast_remaining.py prepare|verify|preflight|collect|resume|audit|report|stop|all`。
prepare和preflight不启动TTF。collect/all必须在确认启动计时后单独调用。stop在当前episode完成后安全停止，resume只补缺项，不重跑正常失败；异常保留原文件并先检查。
输出为`build/sa-raw-fast-remaining-ttf-v1`，源码本地提交、注册及数据分阶段备份，不更改上一轮JSON/SHA。

预计串行采集45-90分钟，规划预算累计上限144分钟，进程fuse累计上限288分钟；准备、审计和调度另计，非性能预测。正式计时不得并行运行求解器。
