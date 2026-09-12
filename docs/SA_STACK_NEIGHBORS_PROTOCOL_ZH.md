# SIPPS四邻接栈枚举：独立机制协议

只替换`SIPP::findPath(ConstraintTable,double)`内一个getNeighbors循环：按原`+1,-1,+cols,-cols`顺序枚举栈上四元素数组，用原validMove过滤。不改其它搜索器、优先队列、随机平局、截止检查、PP、候选、模型或SA。单列地图产生的重复邻居也照原样保留，不擅自去重。

1. 冻结输入、归档源码。使用现有Release静态库和绑定对象；只在build生成一处机械替换的SIPP.cpp。编译并显式优先链接新SIPP.o，输出独立lns2_env.so，绝不覆盖原native。不安装依赖。
2. 先将原Instance.getNeighbors与栈枚举在4张实际地图全部自由格逐项比较（含顺序），再各跑200000次查询，预热后4轮交替计时。累计查询减少至少10%才继续；这不是PP/TTF加速，也不代表真实展开位置分布。Python单测补充单行、单列、边界和障碍语义。
3. 独立native完整重放原8案例、792个修复，验证reset、候选评分选择、PP顺序、路径和低层计数。使用冻结SingleFullCheckPool；新native身份显式登记，不冒充冻结原库。8个独立进程加快非计时等价验证，每案例900秒fuse，每步300秒安全PP预算。时间不用于性能结论。
4. 原子逐案例保存并设运行锁；有旧attempt则拒绝自动重跑。任何不一致停止，不重抽状态、不改阈值。全部通过仅允许下一轮有限的配对PP组件计时，不自动开始完整TTF。

输出build/sa-stack-neighbors-probe-v1。新构建和对象记录SHA，原native、模型与历史报告必须保持不变。保留微基准和完整重放结果；不删除旧失败结论。
