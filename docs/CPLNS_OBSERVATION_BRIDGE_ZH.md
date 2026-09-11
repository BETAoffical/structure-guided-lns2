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
