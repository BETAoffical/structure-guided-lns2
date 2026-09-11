# CPLNS 单求解器只读观测桥

## 目的和保护边界

恢复点为 `pre-cplns-observation-20260911`。原始作者源码、参考二进制、官方 LNS2、
V2/Dual16 均不修改。单独复制固定作者源码，新增一个本项目的只读头文件，
仅在 `LNS.cpp`、`InitLNS.cpp` 插入事件记录调用，不替换任何原语句。
作者源码没有明确的顶层再分发许可证，因此原文及修改副本均只保留在忽略目录。

桥接源码必须逐文件等于固定原文加预先登记的插入；其他文件不得改变。
配置、runner、header、完整桥接源码、两个二进制及 fixture SHA 一并登记。
已登记运行拒绝源码变化、结果篡改及自动重试失败作业。

## 固定诊断

- 四个微型输入：原开放并行、不可换序单通道、8x8 高密度开放图、8x8 双门图。
- 后两个输入的起终点由 seed 917 固定抽取，不按运行结果换例。
- 每图四种作者参数：无 SA 无重启、仅 SA、仅重启、SA 加重启；solver seeds 0/1。
- 三个执行版本：未修改 CLI、桥接 CLI 关闭观测、桥接 CLI 开启观测。
- 共 96 个工程作业；每作业 solver 3 秒、外部 fuse 25 秒、20 个独立 worker。
- 每个进程仍只有一个求解器。并发耗时、观测耗时均不作为正式 TTF 或加速证据。

输出初始 PP、InitLNS 初始化、PP 顺序、接受后的状态以及结束路径。
记录前 128 次修复的完整信息；结束路径始终输出。超过 128 步的中间状态明确标记未观测，
不宣称检查了全部修复。初始外层 PP 可能尚未规划全部车辆，该事件不冒充完整路径集。

## 验证与判定

1. 根据 `.map/.scen` 独立检查 agent 身份、起终点、合法移动和障碍。
2. 重建顶点、交换和终点持续占用冲突，按唯一 agent 对计数，并重算 SOC。
3. 检查 PP 顺序与邻域一致、外部路径不变、拒绝时完整回滚、接受后的全局冲突差。
4. 与未修改 CLI 的原有 screen=3 输出比较邻域、PP 顺序、初始化路径、低层展开和接受日志。
5. 双方都求解完成时比较完整科学日志序列和最终 SOC；限时作业仅比较前至多 248 个共同记录，
   避开最终低层调用受墙钟截断影响的部分。有限前缀相同不能证明后续所有路径完全相同。
6. 所有比较一致且在合法状态上观测到正 delta 接受，才记 `bounded_observation_pass`；
   否则停在 `investigate_before_real_cases`，不直接接入 Dual16。

观测时钟从作者 `LNS::run()` 入口开始，记录的是对应事件边界的单调时间，包含之前观测开销。
它排除构造/加载，不等同于本项目常驻规划器正式计时协议，更不能把观测开销事后减去后称为 TTF。
本轮不评价成功率提升，不修改温度、邻域大小或重启阈值，不训练模型。

## 入口

在仓库根目录的真实用户 WSL 环境中执行：

```bash
python3 scripts/verify_cplns_observation.py prepare
cmake -S build/cplns-observation-bridge-v1/source -B build/cplns-observation-bridge-v1/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/cplns-observation-bridge-v1/build -j 12
python3 scripts/verify_cplns_observation.py register
python3 scripts/verify_cplns_observation.py dry-run
python3 scripts/verify_cplns_observation.py collect
python3 scripts/verify_cplns_observation.py report
```

`collect` 重入只复用 SHA 和身份完全一致的已完成作业。任何异常保留原结果并停止准入。
原始 CLI 的非 SA 配置并不等于本项目官方 LNS2，不能改名为 Official baseline。

## 首轮观测与后续定位登记

首轮实现提交 `923dc24`：96 个作业完整，3128 个完整状态/结束快照、3072 次修复检查通过，
观测到 480 次正 delta 接受。但 64 组原版对桥接比较仅 32 组一致，复杂两图的全部 32 组首次分歧
都在 `Generate 8 neighbors by target`。保留失败结论 `investigate_before_real_cases`。

作者 `SingleAgentSolver.cpp` 的 Target 辅助搜索使用 `pairing_heap<Node*> open_list`，没有将所声明的
节点比较器传入堆。相同 seed 不能控制分配器给出的指针顺序；观测本身可能改变分配历史。
此外目标计数表达式缺少括号的问题也仍在作者固定代码中，本轮不修改。
项目历史提交 `ab833fd` 曾检查/修改过同类目标搜索的比较器和计数语义；本轮不能再把修正后的版本
冒充未经修改的作者算法，也不重训模型来补偿复现差异。

下一项严格限于原因定位：沿用两个复杂 fixture、seed 0、SA 开启且重启关闭，分别固定
Adaptive、Target、Collision、Random。每组运行原版两次、桥接关闭和桥接开启，共 32 个 3 秒作业。
比较前 128 个原始科学记录。入口为 `scripts/diagnose_cplns_observation_divergence.py prepare|collect|report`，
独立输出 `build/cplns-observer-divergence-v1`，计划及输入 SHA 在运行前冻结。
该对照只判断差异是否集中于 Target，不覆盖或放宽首轮门槛，也不评价速度或求解成功率。

## 定位结果

定位协议提交 `61f52b5`。32/32 作业正常返回，没有外部进程超时。
每种规则在两张图上比较原版重复、原版对关闭观测、原版对开启观测、关闭对开启观测：

| InitLNS 规则 | 相同前缀 / 比较数 |
|---|---:|
| Adaptive | 2/8 |
| Target | 2/8 |
| Collision | 8/8 |
| Random | 8/8 |

两张图的四种规则中，原二进制与自身重复运行均相同；Target/Adaptive 的跨版本或观测开关比较不同。
这支持 Target 地址相关搜索受分配历史影响的定位，但没有通过修改分配器证明唯一因果。
观测头文件使用 `std::ofstream` 等对象，可能影响分配历史。下一步若继续桥接，应优先尝试无堆分配的
输出实现，而不是改变 Target 比较器后宣称原作者动作等价。

覆盖审查还发现原 main 将 CLI `screen` 减 1 后传入 LNS，因此本次 `screen=3` 的 InitLNS 实际为 2。
原始日志没有预期的全部 `Agent` 初始化路径或逐 agent 低层展开记录；实际比较覆盖邻域、PP 顺序、
旧/新碰撞数和退火接受日志。观测版自身有独立完整路径检查，但不能据此宣称原版复杂图的初始路径
逐项相同。旧门槛失败原样保留；后续观测版本须修复覆盖合同，不能在旧数据上补写“已验证”。

更重要的是独立状态校验阻止了定位汇总：8 个有路径输出的作业中，7 个通过，一个发生冲突漏计。
原定位脚本按协议抛错，`run_status.json` 保存 `interrupted_or_failed`；这不是 worker 中断或求解进程崩溃。
没有重新运行或覆盖这些作业。另用只读 `scripts/audit_cplns_observation_outputs.py` 生成失败审计。

具体见证为 `two_door-Collision-observer_on`：

- InitLNS restart 0、内部 iteration 60（原 stdout 的统计向量记作 Iteration 59）。
- 所选车辆为 7、9、22、32、38、39、41、42。
- 接受前相关冲突 23 对；作者记录修复后相关冲突 19 对，实际为 22 对。
- 全局作者记录 52 对，独立路径重算为 55 对；SOC 均为 684。
- agent 22 的 start=goal=55，本次输出路径只有一个位置。
- 原日志碰撞图漏掉 `(20,22)`、`(22,37)`、`(22,45)`，对应其他车辆经过其持续占用的终点。
- 作者 `InitLNS::updateCollidingPairs` 在 `path.size()<2` 时直接返回，跳过了这类路径的终点持续占用检查。

以上是合法任务语义：车辆已经在目标上不意味着可以被其他车辆穿过。不能删除该 agent/任务来让校验通过。
本轮只记录作者分支的问题，未修改作者接受规则、碰撞计数或本项目正式修复器。

## 结论与下一步

当前 decision 为 `blocked_on_target_parity_and_collision_accounting`：

- 退火接受分支已动态覆盖，首轮路径和接受后冲突差校验通过，但不代表方法有效。
- 复杂图上的原版动作等价尚未建立，且固定规则对照暴露了作者的单位置路径漏计。
- 不开始真实案例比较或 TTF，不接入 Dual16，不训练模型。
- 下一工程轮分开验证无分配观测与终点占用回归；若需要修作者计数，必须另命名为 corrected reference，
  保留 untouched reference，不能静默替换原版或让历史数据混用。
- 重启次数与精确首次可行时钟的正式验证仍未完成，现有事件时间只作诊断。

## 证据身份

- 首轮 report SHA256：`161f5e8a8f4890be6d18adab665f92ba5f861957cdd0c1c0d34c0ea487eddcfe`。
- 原因定位 plan SHA256：`c8d9ca38981650b55fb531a4c123fff25fd2ca296fd74d1f93964f0d0ceb83ae`。
- 独立失败审计 SHA256：`02dd4cf8f3e8e01f981e332b9d7081d0edd0783202bf5de871143fcc0b05691d`。
- 观测 binary SHA256：`a53c10917b688add8e8a6ecfe4d73c056ae071aedb154b2169490ab86a77e5fa`。
- 未修改作者 binary SHA256 保持 `c57de074f49065a39f1f1ad6e87b698964df3b9ea3b7ba5cb51746fc1e867445`。
- 本项目旧冻结 native SHA256 保持 `7e84f535cc992d7424959cbafe63fce122a5b97c952b92c68d8978093f190e08`。

所有原始轨迹保留在 `build/cplns-observation-bridge-v1` 和 `build/cplns-observer-divergence-v1`；
失败解释独立输出在 `build/cplns-observer-failure-audit-v1`，不覆盖原失败结果。

## 回归与备份

- 20 worker Python 分区回归：1573 passed、68 skipped，95.84 秒。
- 原登记 native 的时间敏感及历史身份分区串行回归：9 passed，12.62 秒。
- 本项目 LNS2 CTest 12/12，15.57 秒，两组官方路径 parity 通过；GPBS 未纳入本轮。
- 检查结束时无残留 CPLNS 求解器或诊断 collector。
- 源码、配置、测试及结果说明均本地提交；本轮未推送远端，未绕过此前涉及历史证据 ancestry 的推送权限限制。
- 不把这些验证时间或并发工程 fixture 的耗时写成算法 TTF。
