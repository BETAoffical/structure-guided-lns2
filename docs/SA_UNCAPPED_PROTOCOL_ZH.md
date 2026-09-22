# 取消决策次数上限的冻结比较

本轮按用户要求取消执行决策上限，不使用大整数代替无上限。旧的训练与 256 次决策比较原样保留。

## 固定范围

- 同一组开发留出案例：两张已查看地图、八个任务与 solver seed 条件、两个 replica、五种方法，共 80 条完整重跑。
- 方法保持 Official+SA、Dual16+SA、未训练探索、bounded_condition、bounded_state；不训练、不改模型或 native。
- 保留每条轨迹 25,000,000 generated nodes 的修复预算，原子 PP 可以越过节点边界；可行优先于预算耗尽。
- PP 安全限制 20 秒、episode 安全限制 900 秒、进程保护 960 秒；超出安全限制属于删失未知，不当作普通失败。
- 20 个进程，非 TTF 实验。支持完成当前批次后安全暂停和 episode 级显式续跑；部分结果、异常和删失禁止自动重试。

## 冻结输入与公平性

执行配置 `max_decisions=null`。冻结模型原来的 `budget.remaining_decision_fraction` 特征仍按训练参考 256 归一化并在零处截断。这只是模型输入，不限制执行；256 步以后的部署超出原训练 horizon，单独统计。

所有方法沿用原比较的随机流 phase、任务、replica、种子。每条旧轨迹的动作、候选、特征、概率、状态 delta 和路径 fingerprint 必须逐步一致。排除机器相关 metrics 时间后比较完整事件；metrics 另由完整状态审计校验。

原来非决策上限停止的 61 条轨迹应完全复现；原来 19 条 decision_budget 轨迹必须复现前 256 步并继续。若安全限制引起删失或前缀不一致，停止并检查，不混算或静默重试。

## 执行与输出

`scripts/compare_sa_uncapped.py prepare|verify|collect|audit|report|stop`，`collect --resume` 只复用通过完整校验的结果。配置位于 `configs/sa_crossfit_uncapped.json`，输出位于 `build/sa-crossfit-no-decision-cap-v1`。配置、源码与测试在采集前提交，输入和模型 SHA 登记到独立 registration。

汇总成功数、节点预算耗尽、删失、超过训练 horizon 的数量、逐任务和逐地图配对、共同成功上的修复次数、节点、SOC、makespan、等待步数。比较全部五种方法，不能把多 worker 诊断时间解释为真实 TTF，也不自动晋级模型或声称独立跨地图泛化。
