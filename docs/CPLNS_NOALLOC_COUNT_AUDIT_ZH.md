# CPLNS 无分配观测与计数函数回归

## 固定边界

恢复标签 `pre-cplns-noalloc-20260911`，独立分支 `codex/cplns-noalloc-and-count-audit`。
本轮不改变正式 native、V2/Dual16、作者原版或上一轮结果，不训练模型，不跑正式 TTF。
两个验证相互独立，不通过改变 Target 或退火接受规则来修饰观测一致性。

## A：无堆分配观测

- 用 POD 线程局部字段、固定栈缓冲区、`to_chars` 和 `open/write/close` 替代 iostream 输出。
- 不消费随机数；输出失败退出独立诊断进程，不静默丢事件。存在的输出文件不能覆盖。
- 编译专用 C++ harness：预先准备路径后禁止 `operator new/new[]`，检查开启/关闭、长路径跨缓冲区、
  128 步截断、结束事件必写、重复文件及 I/O 错误。此检测覆盖 C++ 分配，不宣称系统库的所有内部分配均已审计。
- 仍复制固定的作者源码，只加原有观测调用，换观测头文件，不修作者算法。
- CLI `screen=4`，使 InitLNS 实际 screen=3，必须获得完整初始化路径日志。
- 原开放、单通道、密集开放、双门四个 fixture；四种 SA/重启组合；seed 0/2；
  原版、关闭观测、开启观测三版本，共 96 个短作业。0/2 避免 glibc 的 0/1 初始随机流重合。
- 每作业 solver 3 秒、外部 25 秒，20 worker。不是单 worker 性能实验。
- 逐项比较完整初始化路径，以及最多 504 个共同科学日志记录；完整成功时比较全序列。
  开放图 PP 直接可行，不伪造 InitLNS 初始化记录。
- 记录原版对关闭、原版对开启及关闭对开启三种比较。所有匹配且所有导出状态合法才通过。
- 作者已有计数问题若再次出现，保留事件并明确记失败，不修改或跳过不利样例。

## B：单位置路径计数回归

- 固定上轮失败审计 SHA，验证原 job 与事件 SHA，读取 restart 0、iteration 60 的完整路径见证。
- 从作者固定源码原样提取 `InitLNS::updateCollidingPairs` 到独立 C++ 函数 harness。
- 使用仅提供该函数所读字段的 path-table fixture；这不是完整原生 PP 重放，更不是新修复器。
- 同时编译原函数与仅一处 guard 改动的版本：`path.size()<2` 改为 `path.empty()`。
  不改 Target、优先级顺序、SA、SIPPS、碰撞目标或默认基线。
- 8 个手工边界条件，加保存状态的全部 48 个 agent，共 56 个函数案例。
- Python 从路径时间序列独立求冲突 agent 对，比较完整 pair 集，而不只比较数量。
- 报告原版差异、guard 修正后差异；不把函数回归通过当作端到端修复成功。

## 下一步门槛

无分配观测若仍不一致，先记录差异是否仍位于 Target；不自动修 Target 比较器。
计数函数回归通过后，还需在独立 corrected reference 验证实际调用入口、接受/回滚、全局计数及路径合法性，
不能只因函数正确就认为完整 PP 已经修好。两项通过也不立即声称 SA 加速或启动真实 TTF。

## 运行入口

```bash
python3 scripts/verify_cplns_noalloc.py prepare
cmake -S build/cplns-noalloc-audit-v1/source -B build/cplns-noalloc-audit-v1/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/cplns-noalloc-audit-v1/build -j 12
g++ -std=c++17 -O2 -Iexperiments tests/evaluation/cplns_noalloc_harness.cpp -o build/cplns-noalloc-audit-v1/noalloc_harness
python3 scripts/verify_cplns_noalloc.py register
python3 scripts/verify_cplns_noalloc.py harness
python3 scripts/verify_cplns_noalloc.py collect
python3 scripts/verify_cplns_noalloc.py counter-prepare
g++ -std=c++17 -O2 build/cplns-count-function-audit-v1/counter.cpp -o build/cplns-count-function-audit-v1/counter
python3 scripts/verify_cplns_noalloc.py counter-run
```

每项输出独立保存在忽略目录；不覆盖历史原始数据或正式结论。

## A/B 结果与原生计数验证登记

实现提交 `b891487`。C++ 分配陷阱 harness 通过：258 个事件，包括超出单个缓冲区的长路径；
开启/关闭成功，重复文件和无效路径按预期拒绝。

A 的 96/96 作业完整；96/96 组观测对照通过，72 组非 PP-direct 初始化路径逐项相同。
但两条双门 seed 2 的 SA 轨迹仍发生计数漏报，decision 保持 `blocked_before_real_cases`。
两者均在内部 iteration 12，实际 62 对、记录 59 对，agent 22 只有位置 55 的单位置路径。
这轮解决了观测扰动问题，未解决作者计数问题，也不构成 TTF 证据。

B 的 56/56 函数案例在 guard 修正后全部匹配独立 pair 集。原函数有三项不匹配：
两个手工单位置案例，以及保存状态中的 agent 22；后者精确补回 `(20,22)`、`(22,37)`、`(22,45)`。

现在单独登记原生回归，不改 A/B 或作者原版：

- 独立 `build/cplns-stationary-count-native-v1`，名称为 `author_stationary_count_corrected_not_official`。
- 在无分配观测副本上只改该一处 guard，不改调用入口、Target、PP 顺序、SIPPS、SA 或重启。
- 原 32 个 observer-on task/profile/seed 全部保留、相同参数、每作业 3 秒、20 worker。
- 原开放、单通道和密集开放 24 条中性对照保持科学前缀一致。
- 双门 8 条检查初始化相同、修复后路径/冲突/回滚正确；修正计数后的后续接受和路径可以改变，不伪称动作等价。
- 必须 32 条完整、0 路径/计数不一致，中性对照通过；否则停止，不继续扩大 patch 范围。
- 通过也仅为原生计数回归，不宣称成功率/速度提升，不接入本项目正式控制器。

原生回归入口：`scripts/verify_cplns_stationary_count.py prepare|register|collect|report`。
prepare 后在输出目录的 `source` 与 `build` 上使用相同 CMake Release 命令构建，再 register 和 collect。
