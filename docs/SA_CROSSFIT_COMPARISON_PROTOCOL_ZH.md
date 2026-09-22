# 有界原型的开发闭环比较

## 问题与边界

本轮只验证两个已经完成训练的原型能否改善完整 episode 的预算内完成率，
不再训练、挑选标签或调整超参数，也不是 TTF 计时。
上一轮同状态动作变化、KL、Brier 和梯度离散度不能替代闭环结果。

使用原登记的 8 个 development_holdout 条件，每个 2 次独立随机流，
共 16 个配对位置、5 个方法、80 条 episode。这两张地图此前已被查看，
只能构成开发证据，不是新的独立泛化验证。不得按冲突或成功筛选任务。

## 冻结方法

- Official+SA：保持官方邻域和原生 RNG 调用顺序。
- Dual16+SA：冻结候选及排序，确定性选择原 anchor。
- untrained_exploration：actor-0，90% anchor 先验加 10% 均匀探索。
- bounded_condition：从 actor-0 用条件基线训练的一次有界更新。
- bounded_state：从 actor-0 用交叉拟合状态基线训练的一次有界更新。

两个新原型不是顺次更新，也不是第 2 轮训练；状态基线只用于训练，
在线只加载 actor。完整模型文件及语义 SHA 固定在配置中。
模型按各自独立目录复制，原模型和已有结果不改写。

原执行引擎把学习策略统称 `trained_actor`。本轮复用该引擎，不改变算法，
独立 registration、目录和 comparison_receipt 将它映射回真实实验方法名。
原科学 binding、模型 iteration 和 policy SHA 保留；比较本身使用额外 binding。
不可只凭 raw result 的通用 engine arm 混合两个模型。

## 配对与资源

- 相同任务、solver seed、初始路径及状态 fingerprint。
- 独立 phase 的确定性随机流，不含方法名。显式方法共享每步 PP seed、
  选择随机数、SA 随机数；不同 replica、purpose、phase 分离。
- Official 保留原生 PP/邻域随机调用，不强制改成显式方法的 PP 随机流；
  其 SA 随机数配对。这不是保证不同方法每一步拥有相同随机过程。
- 科学工作预算：256 次决策或 25,000,000 个 repair generated nodes，
  在完整原子 PP 边界检查；初始化节点不计入 repair 工作预算。
- 单次 PP 安全阀 20 秒，episode 安全阀 900 秒，独立进程 fuse 960 秒。
- 最多 20 个进程；并行运行时间不用于速度优越性或 TTF 声明。
- 四批，每批最多 20 条。安全暂停等当前批完成，恢复只运行未开始条目。
  安全中断、异常、部分文件均不得自动重试或增加预算。
- 原子结果、完整 trace、模型回执、全局及局部锁；已有锁不自动删除。

## 分析规则

主要报告全 16 个位置的成功数、失败类别、成对新增完成和丢失完成，
以及逐地图结果。不要求每张图或每个 seed 都胜出，不用代理指标否决实测收益。
成功后的决策数、节点、SOC、makespan、等待步数只在共同成功位置成对比较。

安全阀触发为未知，不是普通预算失败；报告成功差的最好/最坏界限。
发生未知先停止新批次并审计，不用删去这些位置得到更好成功率。
未运行完或未通过完整 trace 审计时，不生成“完整比较”报告。

若存在净完成收益，下一步才讨论独立数据确认；不能由 16 个已查看开发位置
宣布泛化或真实加速。没有净收益时先区分行为几乎未改变、轨迹分叉得失相抵、
工作预算不足和运行错误，不自动进行第二轮训练或放大更新。

## 执行

```text
python scripts/compare_sa_crossfit.py prepare
python scripts/compare_sa_crossfit.py verify
python scripts/compare_sa_crossfit.py collect
python scripts/compare_sa_crossfit.py stop
python scripts/compare_sa_crossfit.py collect --resume
python scripts/compare_sa_crossfit.py audit
python scripts/compare_sa_crossfit.py report
```

collect/audit 使用已固定 native 的 WSL `/usr/bin/python3`。
prepare 前保存代码检查点；正式执行前登记配置、输入、模型、代码 SHA。
输出 `build/sa-crossfit-comparison-v2`，不进入 Git。保留所有旧数据。

## v2 运行器修正登记

首个 v1 批次 20 条：16 条完整，4 条 Official+SA 被旧 Python 运行器
错误标成 incomplete_pp。原生返回 not_run、尝试 agent 数 0、冲突及路径不变，
不是 20 秒 PP 超时。源码 `InitLNS::step` 在生成邻域没有任何冲突边时，
更新 Adaptive 权重、计一次 iteration，然后合法返回；不需要执行 PP 或接受判断。
旧运行器的 `not acceptance_evaluated` 判定漏掉了这个合法分支。

v1 登记、20 条记录和提交 607fd3a 保留，不作为性能证据。v2 完整重跑 80 条，
不只重跑某个方法的失败；使用相同 phase 和 RNG，不更换模型、任务或资源预算。
新运行器仅改变中断分类：not_run 必须满足合法动作、没有 PP 尝试、没有回滚，
路径、冲突及低层计数完全不变；已生成邻域必须不触及冲突 agent。
这样的空操作仍计一次决策，仍保留 native 的 Adaptive 更新，不人为强制成功。
真正 time_limit 或 PP 未完整结束仍是未知，不能用该修复掩盖。

原 hash-frozen 训练代码不修改；新引擎副本用 AST 测试证明只增加这一分类调用，
其余循环语法树与旧版完全一致。最终要求 20 条旧记录的动作/路径前缀完全一致，
16 条原本完整记录最终结果也完全一致。旧正式 TTF 使用另一运行器，不能据此
宣称旧 TTF 失败均来自此问题；本轮另记录影响范围。
