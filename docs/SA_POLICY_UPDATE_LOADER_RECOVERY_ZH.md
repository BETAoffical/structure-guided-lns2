# 单轮更新模型：闭环加载修复

v1已收齐并全量审计768个标签分支、12个重放control，拟合一次129维候选输入模型。
训练阶段和WSL原生fixture验证均正确，但闭环启动触发`paired schema mismatch`。
原因是worker及对应审计入口将bundle的258维delta/mean列名当成129维原始候选列名传给portable_model。
训练和verify-model使用training manifest中的原始列名，因此未发现这个入口差异。

这是本轮新原型的适配bug，不是旧Dual16、官方PP/SIPPS、SA或训练标签错误。
首次闭环0个episode完成，错误导致整个批次停止；残留初始/部分trace不视为实验结果，也不复用。

修复只有模型加载：增加共用load_updated_model，读取并验证训练manifest、receipt、bundle SHA，
使用登记的原始列名。新增测试直接覆盖真实加载函数，而不是只测试模型对象的rank。
修复前script按原SHA保存在v1/frozen-source；v1的注册计划、标签、模型和失败记录均不改。

恢复入口为scripts/recover_sa_policy_aligned_update_evaluation.py。
单独v1b输出仅重跑24条开发闭环；配置除输出目录外必须逐项等同v1。
恢复计划登记原版源文件快照、旧计划、标签审计、模型及当前修复代码SHA。
模型和fixture逐字节复制，不重新训练、不重新采集标签、不改变候选、随机数、预算或验收规则。
原12个control以明确的reused_source引用，不冒充新运行；重新进行模型native parity。
先用完整的一个任务三方法闭环验证加载通路，再resume剩余任务。

原v1严格验证需使用预注册787ba55的源码或保存快照；不能修改旧计划的期望SHA掩盖代码变更。
本轮仍是开发性工作量实验，不能解读成正式TTF或独立泛化确认。
