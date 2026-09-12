# PP工程优化机会：先验证每节点容器开销

## 结论

仍有值得验证的工程空间，但还没有新的加速结论。本轮只读审查C++调用链及最近32份已完成记录，新增可复现统计入口，未修改求解器、native、模型、退火、默认策略或启动新求解/计时。

最优先的具体候选是：**SIPPS节点展开时，用保持原顺序的四元素栈上枚举替代临时邻接链表**。其次才是安全区间临时结果和每次搜索的全地图容器初始化。不能仅凭这些代码存在就承诺节省百分比。

## 实际工作量

重新校验最近TTF清单、32个raw及审计收据的身份与SHA。下面是每个运行时各16个episode的累计，两侧计数逐案例、逐重复完全一致；已扣除reset初始规划。

|指标|每侧累计|
|---|---:|
|邻域修复调用|1,584|
|单车findPath搜索|23,392|
|expanded计数|77,806,100|
|generated计数|111,282,934|
|PP完成并插入路径的agent次数|23,392|
|按源码推导的SIT槽位初始化数|80,842,752|

23,392不是修复轮数，一次修复会依次为多辆车搜索。generated不是精确的malloc次数，支配逻辑会调整生成计数。expanded包含在展开邻接前就终止的目标节点，因此不能直接叫作getNeighbors精确调用数。

四张地图均为48×72。每次findPath构造`ReservationTable`，其`vector<list<Interval>>`按3456格初始化，所以槽位数为23392×3456。**这是累计容器槽位，不是8084万次堆分配，也不是同时占用8084万个槽位。**真实触达格子数、构造/析构耗时尚未采样。

## 候选一：取消每节点邻接链表，优先

[Instance.cpp](../third_party/mapf_lns2/src/Instance.cpp)的`getNeighbors`（约473行）每次创建新的`list<int>`，按`curr+1, curr-1, curr+cols, curr-cols`顺序检查`validMove`，为合法邻居emplace节点。

[SIPP.cpp](../third_party/mapf_lns2/src/SIPP.cpp)的`findPath`（约96行）在非终止节点展开时调用它。静态四邻接不会随PP路径表改变，但当前查询反复进行小链表分配/释放。

最小实验不需要改公共Instance接口，也不需要共享动态缓存：仅在独立SIPPS变体中按原四方向顺序遍历栈上候选，用同一个`validMove`过滤，内部生成后继的循环保持原样。先核对所有格子的结果和顺序，包括边界、障碍、单行/单列，再验证真实前缀的路径、计数及后续随机流。

这比先改优先队列、节点内存池或动态路径缓存边界更小。静态邻接缓存可以后续比较，但当前不同时加入第二个改动。

## 候选二：安全区间返回容器

[ReservationTable.cpp](../third_party/mapf_lns2/src/ReservationTable.cpp)的`get_safe_intervals`（约313行）为每次查询返回新的`list<tuple<...>>`。查询发生在每个合法方向上，可能带来大量临时节点分配。

备选方案是每个搜索器独享、可复用容量的顺序容器。必须保持返回顺序、区间边界和迭代有效性，不能让下一次查询覆盖仍在使用的结果。本轮没有测到查询次数或分配占比，不与邻接修改一起实施。

## 候选三：SIT容器生命周期

[ReservationTable.h](../third_party/mapf_lns2/inc/ReservationTable.h)约12行构造整个地图大小的SIT，但`updateSIT`是按访问位置惰性计算。这意味着存在整图容器初始化与局部实际访问之间的潜在差额。

可研究的是复用容器容量、只清空触达位置；不是无条件复用上一次区间答案。约束依赖目标位置、时间限制以及路径表；PP插入下一辆车或回滚都会改变约束。若没有明确版本和失效设计，跨agent缓存会用旧约束产生错误路径。优先级低于无状态的四邻接枚举。

## 排除重复路线和危险改动

- Release实际flags含`-O3 -DNDEBUG`，不是误跑Debug；源码assert中的额外冲突扫描不在此Release执行。
- 所有步骤已关闭详细PP diagnostics，不能再次把关闭它列为新增收益。
- batch proposal、native特征、静态拓扑缓存、prepared复用和单次完整fingerprint已有实现；不重复开发。
- [既有成本审计](SA_RUNTIME_COST_AUDIT_ZH.md)显示native PP外部与绑定层空间小；拒绝尝试整个PP也只占旧样本约0.75%，不能通过删除回滚解决主体开销。
- `LLNode`比较器仍在平局时调用`rand()`。本轮不将其改成严格确定性排序，不换heap，不改变后继顺序。该改动会改变搜索语义，不能冒充工程等价优化。
- 节点`new/delete`确实很多，但generated不是分配计数；没有分配热点数据前不启动内存池重构。
- 安全区间SIT已经是单次搜索内惰性生成，不能说当前完全没有缓存，也不能把新增缓存的理论收益当实测。

## 下一步的最小验证

1. 使用独立诊断构建目录，不覆盖冻结so；先分清findPath、冲突扫描、路径插入等PP子部分，或针对邻接查询做范围受限的剖析。现有环境有g++/nm，未发现perf入口，本轮未安装任何工具。
2. 只对首选邻接改动做保持方向顺序的等价测试。正式baseline与模型不改，任何路径、节点计数或随机流不一致先停止。
3. 在相同冻结状态/动作上记录剖析的计时扰动，不把instrumented wall当真实TTF；微基准只证明该查询可加速，不代表它占PP多少。
4. 只有确认热点和行为等价后，才做独立、有限的配对修复计时；收益不足就保留负面证据，不直接扩成32个完整episode。

本轮决定是`profile_order_preserving_neighbor_enumeration_before_native_change`。它与以前重复训练、回退阈值、扩大邻域或改优先顺序不同，目标是相同搜索决策下少付出容器维护成本；不会直接提高每步修复成功率。减少真实耗时是否能提高限时成功数仍需后续验证。

## 保存

入口：`python scripts/audit_sa_pp_workload.py`。读取冻结报告，不调用solver。输出`build/sa-pp-workload-audit-v1/report.json`；15项定向测试通过。再次读取结果应得到相同SHA。

新分支`codex/sa-pp-workload-audit`，恢复标签`pre-sa-pp-workload-audit-20260912`。保留[紧凑证据](../artifacts/sa-pp-workload-audit-v1/evidence.json)，本地提交备份，未推送。没有重编译native或执行CTest/parity。
