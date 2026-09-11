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

## C：独立原生计数回归结果

预注册提交 `81bc2aa`；32/32 作业完整且全部通过，decision 为 `bounded_native_counter_pass`。
24 条中性对照保持科学前缀一致。双门 8 条初始化路径相同，其中原来漏计的两条 SA seed 2
轨迹在修正后的科学前缀发生变化，其余六条前缀一致。这种变化是计数修正的预期影响，不是动作等价证据。

共检查 3,128 个导出状态/结束记录、3,072 次有完整前后路径的修复，以及 448 次接受更差候选的事件。
所有导出完整状态均通过路径合法性、终点持续占用、唯一冲突对数和 SOC 重建检查；
观测到的修复通过外部路径不变、接受增量和拒绝回滚检查。
每条轨迹只导出前 128 次决策及最终状态，因此不声称检查了截断后全部中间修复。

这里的“路径合法”指栅格移动、起终点及记录一致，不代表所有诊断任务都找到了无冲突解。
作者代码的 Target 指针排序、其他未涉及的实现和编译警告仍保留。修正版是独立参考，
不能再称为未经修改的作者代码，也不替换本项目的官方 LNS2、V2 或 Dual16。

### 固定证据

| 文件 | SHA256 |
|---|---|
| `build/cplns-noalloc-audit-v1/report.json` | `3e65ab812c0cc188bd3df64a497ba5a5f433b0167ac80a40f429d1fc6a4dd2eb` |
| `build/cplns-count-function-audit-v1/report.json` | `14db21d044b9af36ca6c4c12af7005959a7683afd3407b70ec93b3c98869910b` |
| `build/cplns-stationary-count-native-v1/report.json` | `488d1b9eddc751de207dbd30597075385172990528d9493302f6e5b8f55912c2` |
| `build/cplns-stationary-count-native-v1/build/plns` | `1aa54faf31eda5c6c17258bc623eb7af43a5b54f4ec500e0ffc66249798377ef` |

### 对后续路线的影响

1. 可以使用这个明确标记的参考版本开展下一轮有预算的真实困难案例机制验证。
2. 比较无退火、仅退火、仅重启、退火加重启，保持参考求解器、初始化和其他参数一致；
   单独登记案例、停止条件与观测范围，不将原版与修正版混合统计。
3. 重点判断接受更差解后能否持续突破停滞、是否只靠重启获益，以及搜索开销是否可控。
   仅接受了更差解或某条轨迹恢复，均不等于总体成功率和 TTF 改善。
4. 本轮到正确性验收为止，不自动启动新的真实案例或正式串行 TTF，不重训任何模型。

## 项目回归与保留边界

- Python 主回归：`1584 passed, 68 skipped`，20 worker，104.67 秒。
- 单独使用冻结正式 native 的进程/路径交付测试：`9 passed`，串行，14.56 秒。
- 跳过项为 Windows 专用 1 项、其他隔离原生扩展 30 项、Windows sklearn 训练环境 4 项、
  由 Linux CTest 覆盖的原生模块/采集器 33 项；不把跳过写成通过。
- LNS2 CTest：`12/12`，15.73 秒；本轮未运行无关 GPBS 测试。
- 两组官方路径 parity 继续为
  `915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf` 和
  `031d1bf843ada89f03be6880809bcf7632fdaeba509fd035a15657fcc93aa83a`。
- 冻结正式 native SHA 仍为 `7e84f535cc992d7424959cbafe63fce122a5b97c952b92c68d8978093f190e08`；
  作者未改动 binary SHA 仍为 `c57de074f49065a39f1f1ad6e87b698964df3b9ea3b7ba5cb51746fc1e867445`。
- 结束时只读进程检查无残留 CPLNS 求解器或本轮 collector；没有安装依赖、删除旧结果或改动正式模型。
- 分支与恢复标签保存在本地。此前远端推送权限未覆盖待推送历史中的 evidence artifacts，
  本轮不绕过该限制，不将本地提交描述为已上传 GitHub。
