# CPLNS 完整 PP 与退火接受拆分

## 固定边界

分支 `codex/cplns-complete-greedy-ablation`；恢复标签 `post-cplns-real-task-mechanism-20260911`。
只新增20个真实任务作业，复用前轮无SA/无重启和SA/无重启的40条记录，不重跑旧对照。
不改正式V2/Dual16/native，不训练，不跑TTF，不扫描温度，不导入旧停滞路径。

## 唯一实现变更

在已通过计数回归的作者副本中，仅移除 `InitLNS::runPP` 内“非SA且局部冲突达到旧值则提前break”的块。
CLI继续保持 `sa=false`、重启阈值2，完整PP后仅接受冲突不增加的结果。
时间限制、空路径处理、回滚、SIPPS、邻域规则、随机打乱、Target比较器等全部保留。
“完整”表示不因当前局部冲突数提前结束，不是取消资源截止时间。
源码副本全部固定并逐字节验证，不将实验修复器冒充作者原版或官方LNS2。

## 执行与校验

- 完全相同的5任务、seeds 0/2/3/5、30秒求解预算、60秒外部fuse和原任务路径。
- 20进程，仅一批新求解作业。20个新作业及40个旧记录统一重新独立验证路径、计数、接受和回滚。
- 新旧初始完整路径、SOC、冲突以及第一次PP顺序必须一致；不一致则停止解释收益。
- 新版本的前128次观测不得接受变差；重启关闭，不得有额外初始化。
- 显式验证源码只删除登记代码块，原接受条件、SA调用保护和计数修正均未变化。
- 复用既有运行锁、原子作业记录、输出SHA、resume和完成当前批后安全停止。
- 注册和源码身份改变时拒绝续跑；失败或孤立记录不得自动覆盖。

## 如何解释

三路对照：原提前中止/贪心、完整PP/贪心、完整PP/SA。输出逐任务完整PP调用数、最终可行、
前128次可行、接受变差，以及PP次数的配对胜负；失败不把较少调用数当成优势。
只做机制描述，无控制器晋级门槛，平均轮数减少不等于计算量或TTF减少。

如果完整PP/贪心已接近或优于SA，说明不必先引入接受变差；若SA仍有额外信号，才讨论后续独立验证。
作者全局rand使SA额外随机抽样改变后续随机流；初始及第一顺序一致仍不表示未来所有候选和PP顺序相同。
因此不能把差值归因于单次接受概率的纯因果作用。

旧Dual16上complete-greedy已有负面/弱信号；本轮只解释为什么作者参考有不同表现。
禁止由本轮直接替换Dual16、重训或进入大规模TTF。后续先根据三路结果决定是否值得在旧停滞状态上做桥接。

## 入口

```bash
python3 scripts/diagnose_cplns_complete_greedy.py prepare
cmake -S build/cplns-complete-greedy-ablation-v1/source -B build/cplns-complete-greedy-ablation-v1/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/cplns-complete-greedy-ablation-v1/build -j 12
python3 scripts/diagnose_cplns_complete_greedy.py register
python3 scripts/diagnose_cplns_complete_greedy.py run
```

新数据和CSV只留在 `build/cplns-complete-greedy-ablation-v1`。源码与协议先本地提交；不绕过远端历史artifact推送权限。
