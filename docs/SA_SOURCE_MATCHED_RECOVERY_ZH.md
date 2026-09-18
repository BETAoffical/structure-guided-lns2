# 同来源 H32 标签的资源恢复协议

本轮不是新模型实验，也不是正式 TTF。上一轮 376 个作业中，360 个得到完整标签，
16 个在同一个 d128 根状态撞到 180 秒进程上限。本协议单独登记资源恢复，不覆盖原结论。

## 固定范围

- 输入为 `build/sa-source-matched-collection-v1` 已封存的 plan、report 和 analysis_records。
- 只补原先 16 个 `hard_fuse` 作业，保留相同状态、候选、trial、随机流、H32、温度和冻结继续策略。
- 另重放该根状态三个候选的 trial 0 作语义兼容控制，不能用控制结果替换已有标签。
- 进程数从 20 降为 4。每作业仍独立 reset、完整重放前缀；不使用 fork 或共享求解状态。
- PP 上限仍为 5 秒，单作业父进程 fuse 仍为 180 秒。计入初始化、重放、继续搜索和结果处理。
- 不额外增加 trials、不延长 horizon 或预算、不训练、不调整排序器、不启动计时实验。

## 运行与完整性

新增阶段日志，记录加载、reset、前缀步数、根状态校验、候选池、模型构造、proposal、
repair、转换校验、标签校验和封存。日志不进入特征、随机数或标签。
继续搜索使用同一个绝对剩余预算，避免模型构造遗漏于软时钟；父进程预算不改变。

三个控制必须在剔除明确列举的耗时元数据后，全部候选、分数、动作、路径增量、
搜索计数和最终 fingerprint 完全相同，才准许补采 16 个未知分支。
每个已完成作业均以文件 SHA 封存。中断的不完整作业不自动重试，要求先检查。
支持 `stop` 在当前批次完成后暂停，以及显式 `--resume`。
进度分别统计 sealed、valid、feasible、horizon_nonfeasible、censored，避免把写完结果等同于有效标签。

## 分析规则

原有 360 个已知标签不变。新报告只在原来的 None 位置填入对应补采结果，逐项记录来源 SHA。
重复控制不增加统计样本量；相同随机流的补采也不算新增独立 trial。
继续沿用原 35 个 4/4 split 的双向交叉拟合与 5,000 次地图级 bootstrap。
若还有未知，不自动增预算或改方法，相关全组统计继续保持缺失。
这是事后资源恢复的开发证据，不是新的独立模型确认。

## 命令与预算

```text
python scripts/recover_sa_source_matched.py prepare
python scripts/recover_sa_source_matched.py dry-run
python scripts/recover_sa_source_matched.py collect
python scripts/recover_sa_source_matched.py stop
python scripts/recover_sa_source_matched.py collect --resume
python scripts/recover_sa_source_matched.py analyze
```

共 19 个作业，最多 2,432 个前缀修复、608 个继续修复；实际继续可能提前可行。
先一批三个控制，再四批四个补采。按 180 秒上限，五批理想并行上限约 15 分钟，
另加校验和调度。并行资源诊断时间不得当成单 worker TTF。
旧 collector、控制器、模型、native、正式报告和证据文件全部保持原 SHA。
