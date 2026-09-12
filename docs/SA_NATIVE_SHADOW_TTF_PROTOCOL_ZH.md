# Native shadow 去冗余：串行 TTF 确认协议

## 固定范围

只比较冻结 `SingleFullCheckPool` 与 `NativeShadowHeatPool`，不改变模型、候选、PP、退火、每20次的完整shadow及默认控制器。前置组件报告SHA为 `2b20f8ef8f529a4fbf5e35b08e601918e816f6eaf65881e81e49ab6b19a95958`；792步完整重放已通过，本轮直接验证该证据，不重复做一轮相同重放。

复用相同4张仓库地图、8个案例、solver seed103。每案例新旧运行时各执行2次，共32个episode；顺序按案例及重复轮次交替。单worker，60秒规划预算，无修复次数上限。180秒solver fuse与1800秒独立审计fuse沿用旧协议；审计不能和下一次求解重叠。

## 时钟与保存

复用冻结求解循环的同一code object，只私有绑定选择器和持久化回调。主TTF从reset前开始至首次可行，包含reset、选择与PP；模型加载/环境构造在外，完整事后审计和写出也在外。这是规划TTF，不是路径交付或机器人执行时间。两侧边界完全相同，另报告selection与PP累计时间。无解按60秒cap，保留共同成功原始时间，不把失败记零。

沿用逐episode原子capture、独立完整audit、运行锁、attempt日志与resume。`stop`在当前episode和审计完成后暂停；异常不自动重跑，无capture的旧attempt拒绝再次求解。已有结果须通过身份、摘要、完整路径与计数校验才可复用。

执行前登记源码、前置证据、native和模型身份，提交并本地备份。运行器通过私有依赖绑定复用旧调度、监督和统计，不修改旧代码/结果。

## 判定

报告全部16组配对、每案例两次均值、4张地图均值及5000次地图级paired bootstrap，重复运行不算新地图。

只有无成功损失、完整配对动作/路径等价、平均capped TTF改善为正且“新减旧秒数”的95%区间上界不大于0，才保留为开发性工程优化。区间跨0时标为TTF不确定，只保留组件证据。不要求端到端达到组件5%门槛，也不事后新增门槛或筛选案例。

未通过时分析selection、PP和配对波动，不自动增加重复次数、不调参、不改shadow频率。任何无解释不一致停止。即使通过也只代表这些复用仓库案例的工程证据，不是新布局泛化、Official对照或成功率提升，不自动替换默认控制器。

## 命令

`scripts/confirm_sa_native_shadow_ttf.py register|verify|collect|resume|stop|analyze`

输出 `build/sa-native-shadow-ttf-v1`。计时在单worker的WSL冻结native环境执行；预计求解加完整审计15-25分钟，非硬完成承诺。
