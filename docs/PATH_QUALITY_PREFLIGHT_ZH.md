# 路径质量与任务完成时间：测试前准备

## 当前边界

本轮准备案例、冻结输入、定义指标，并补齐第一阶段到第二阶段的直接路径交接。
只允许微型人工地图的功能测试和已有官方 parity 回归测试，不运行候选案例的 reset、
控制器或计时实验，不训练、不重新抽取任务，不修改冻结模型或默认控制器。
新增的 C++ 接口是显式调用的扩展，原有 InitLNS 和 LNS 入口仍保留。
`scripts/prepare_path_quality_cases.py` 只有准备和只读验证功能，没有 collect/run 参数。

这些地图和任务来自已查看过的历史数据，因此本轮只能作为开发性诊断，不能称为独立
确认，也不能用于重新训练或选择冻结模型。本轮不根据控制器成功数、TTF、SOC 或
makespan 选择任务，不排除共同失败或某个控制器较差的任务。

## 案例清单

| 来源 | 地图 | 任务与规模 | 当前角色 |
|---|---|---|---|
| 合成仓库 | `very_high_confirm_v2_station_centric_0000` 至 `0007` | 每图一个原始任务；259、274、290、290、303、292、292、223 agents | 完整任务主组，重新从原始起终点开始，不恢复扰动检查点 |
| MovingAI Room | `room-64-64-8`、`room-64-64-16` | 每图 random scenario 4、5，600 agents | 完整任务结构扩展组 |
| MovingAI Maze32 | `maze-32-32-4` | random scenario 4、5，200 agents | 完整任务边界组，不因历史负面结果删除 |
| MovingAI Maze128 | `maze-128-128-1` | random scenario 4、5，400 agents | 隔离；历史 reset 空路径问题未重新验证，不进入计时排程 |

合计 12 张地图、16 个任务。其中 14 个任务可继续准备，2 个 Maze128 任务隔离。
计划 solver seeds 为 `[51, 52]`，对应 28 个可候选 instance-seed。
计划三组是 `official_adaptive`、`v2-full`、`dual16`：保留 V2 对照以区分完整方法与
结构候选增强。Component16、Mixed、V3 和新训练模型本轮不加入。
若未来全部获准执行，第一阶段为 84 个 episode；这只是计划数量，当前启动数为零。
加上固定总预算 60 秒与 120 秒各 84 个独立 episode，完整预留调度共 252 项。
三组协议独立运行，不能用一次长预算轨迹倒推出短预算的真实运行结果。

另登记历史 16 个仓库扰动检查点，仅供独立的离线全路径修复分析，不混入主组。
它们不等同于固定已执行前缀后的在线恢复，不可直接计算真实剩余任务完成时间。

## 静态检查与冻结

- 严格匹配历史 manifest SHA、地图尺寸、场景地图名称、agent 前缀和任务 sidecar。
- 检查起点唯一、终点唯一、坐标合法、障碍与单 agent 起终点连通性。
- 每个 OD 重新计算四邻域 BFS 最短距离；可达不表示整个多 agent 实例已经证明可解。
- MovingAI 场景第九列的参考距离不一律等于四邻域距离。保留原字段，将差异数和示例
  写入报告；SOC 绕行比的基准使用重算距离，不改地图和场景文件。
- 冻结每个地图、场景、任务、检查点 blob、模型 manifest、模型成员和相关源码的 SHA。
- 输出只使用仓库相对路径。相同输入重复验证，任何 SHA、配置或清单差异均拒绝续用。
- 静态准备脚本不加载 WSL native。功能测试使用独立的
  `build/linux/anytime-handoff-validation`；没有覆盖原有实验构建。
  正式执行时仍须单独登记并验证实际加载的 native SHA。

## 指标与两种交付方式

主目标是从开始规划到全部机器人完成任务的总时间，而不是单独最小化修复轮数。
仍保留规划 TTF、SOC、makespan、等待步数和最短路基准，解释性能变化来自哪一部分。

定义采用上下左右移动或等待、同步离散时间、到达终点后永久占据终点。指标检查器
会排除纯粹用于数组对齐的末端重复终点，但保留中途等待；agent 提前经过终点后又
离开，不算提前完成。验证顶点冲突、对向边交换冲突和到达后的终点占用冲突。
当前只用小型人工数据测试这些函数，没有对候选案例或历史解质量作新评估。

1. **首次可行即出发**：规划最多 120 秒，无 100 次修复限制；得到首个有效解就交付。
2. **固定总预算后出发**：分别预设 60、120 秒的总计算预算；首次可行后接同一个官方
   anytime LNS，用剩余预算优化。不能额外给双方相同的第二阶段时间而忽略第一阶段
   不同开销；预算内没有可行解则报告失败。

第二阶段统一 Adaptive、PP+SIPPS、邻域大小 8，保持无冲突并优化官方 SOC；同时报告
makespan，因为降低总到达时间不保证最后一台车更早完成。第二阶段随机种子要独立
于第一阶段的随机数消耗次数，并在对应任务、重复实验间配对。

对于先规划、全部等待、随后执行的理想模型：

`modeled_batch_completion = dispatch_wall_seconds + step_seconds * makespan_steps`

固定预算协议的出发时刻是预设总预算，不得事后选择曲线上最有利时刻。若求解器或
交接超过截止时间，必须记录超时并使用实际交付时刻，不允许按更早时刻记账。
步长秒数先报告 `0.5 / 1.0 / 2.0` 的敏感性结果，没有物理标定前不声称真实机器人提升。
等待期间的服务时间、加速、转向和通信尚不建模；不能称为完整仓库仿真。

所有方法的成功率先按全组报告；SOC/makespan 等路径质量在共同成功任务上配对，
另外列出不共同成功的全部任务。失败不能当作零执行时间，也不能悄悄移除。
报告按地图族和任务分层；地图级 bootstrap 不把两个 solver seeds 当独立地图样本。
该小型开发集不设置新的全局晋级结论。

## 已补齐的交接接口

`LNS2RepairEnv` 仍只负责 InitLNS。新增
`lns2_env.optimize_feasible_paths(...)` 接收完全相同的第一阶段可行路径，
重建官方 `PathTable`，跳过 `getInitialSolution()`，复用官方 anytime 优化循环。
它不把我们的 GBDT 搬到第二阶段，也不重新运行初始 PP。

Python 入口是 `lns2_selector.evaluation.anytime_handoff.continue_with_official_anytime`：

- 要求第一阶段明确可行，agent 顺序与场景一致，路径端点、SOC 和实际任务相符。
- 检查 native schema 与调用方登记的二进制 SHA；旧模块不能静默充当新接口。
- 输入不允许末端对齐 padding，避免导入时悄悄改变路径或 SOC。
- 总预算从调用方记录的第一阶段起点开始扣除，校验、导入、导出都计入开销。
- 第二阶段显式播种，不受第一阶段 RNG 消耗次数影响；如果再次使用旧环境，
  必须显式重置随机流，不能偷偷延续已被其他接口改变的全局 `rand()`。
- 输出包含原始路径、最终路径、SOC、makespan、等待步数和单独的阶段信息。
- `max_iterations=0` 表示不设轮数限制；微型功能测试才使用两轮限制。

官方 PP 的截止时间不是可抢占的硬中断。一次 SIPPS 搜索可能越过截止时间。
扩展仅保留预算内完成复制的更优可行路径，超时搜索得到的更优解不能冒充按时输出。
实际交付时刻仍包含搜索超出、验证和导出开销，必须报告 `budget_overshoot_seconds`。
预算内已有初始可行解时，第二阶段超出与“没有可行解”分开记账；初始解本身迟到时
`success_by_deadline=false`，不能作为按时成功。此接口不承诺实时系统的硬截止。

官方 Intersection 要求地图存在分支节点；无分支走廊在该扩展中明确拒绝 Adaptive/
Intersection，不擅自替换官方启发式。当前案例是否可被整体控制器接受，仍需独立检查。

## 调度、路径保存与中断处理

`lns2_selector/evaluation/path_quality_execution.py` 已提供静态排程、worker 输入适配、
路径 journal 和单 episode 进程监管。没有新增可以启动地图队列的 CLI。

- 三种方法复用 `_closed_loop_episode_worker`：Official 保留官方分支；V2 保留原排序器；
  Dual16 只加入已存在的 Component16 + Hotspot16 增强配置，不重写候选或排序逻辑。
- 相同任务和 solver seed 使用同一个第二阶段 seed；三种方法的执行顺序按六种排列轮换。
- 原 worker 的可选 `path_observer` 默认关闭。启用时分别保存初始状态、终止状态和首个
  可行解，所有路径在写出时重新检查。第一阶段尚未完成 trace 收尾时可行解就已单独保存。
- 每个文件包含输入绑定和内容摘要，原子写入；已有不同内容不能覆盖，resume 校验身份和 SHA。
- 第二阶段使用保存的首次可行时间，而不是收尾后的接口调用时间判断初始解是否按时出现。
- 权威出发时间为结果中 `dispatch_wall_seconds`，包含最终路径保存前的全部规划和收尾开销；
  不把旧 TTF 指标直接当作新的“规划加交付”结果。该时刻仍是建模时刻，没有实际驱动车辆。
- 该时钟沿用现有 worker 的 reset 前起点：进程启动、模型加载和环境构造不在这个起点内。
  环境构造已有单独字段；若报告要求从冷启动开始，仍须另外补齐进程级总时间，不能把当前
  reset-inclusive 指标称为包含冷启动的总耗时。
- 独立子进程超出 fuse 时终止并回收。超时保留 `first_feasible_preserved` 与
  `first_feasible_within_budget`，但 `success_by_deadline=false`，单独标记中断，不能冒充正常完成。
- 没有可行解、worker 异常、外部超时和完整成功分别记录。已有完整结果可校验后复用；
  缺少完整监管结果的中断目录要求人工检查，不从 episode 中间继续或静默自动重跑。
- 进程启动 API 默认拒绝执行。准备表的 `execution_authorized` 全部为 false；本轮只有
  测试代码显式运行模拟进程和两-agent 人工地图，不对候选案例开放执行。

## 计时前剩余事项

1. 对真正使用的构建登记 native SHA、模型及代码清单；准备阶段未把验证构建自动晋级。
2. 完成整组结果的配对完整性检查与汇总接线；微型样例一致不代替真实案例的初始状态核验。
3. 正式确定每个预算对应的外部 fuse，并登记异常、保存成功和正常成功的分开统计规则。
4. Maze128 两项仍隔离；不因小地图交接通过而自动解除隔离。
5. 获得用户单独的计时授权后，才开放串行地图队列。

当前是“静态准备与交接功能已补齐，整组执行与计时门禁仍关闭”。
功能测试通过不表示两阶段性能更好，更不表示实际机器人执行已得到验证。

## 使用方式

在仓库根目录执行：

```powershell
python scripts/prepare_path_quality_cases.py
python scripts/prepare_path_quality_cases.py --verify
python -m unittest tests.evaluation.test_path_quality_preflight -v
python -m unittest tests.evaluation.test_anytime_handoff -v
python -m unittest tests.evaluation.test_path_quality_execution -v
```

当前 revision 4 输出保存在 `build/path-quality-preflight-v4/`：`cases.jsonl`、
`execution_schedule.jsonl`、`preflight_report.json` 和 `preflight_report.md`。配置仍为
`configs/path_quality_preflight_v1.json`（协议 schema 不变，准备修订号为 4）。
前面的准备修订输出原样保留；新代码不允许覆盖旧指纹结果。v4 补齐了 V2 部署模式
原有的抽样状态校验参数，避免适配层默认退回每轮完整校验而增加开销。
Windows 功能测试使用标准库 unittest，WSL 使用现有环境，不安装依赖。

仅用于微型 native 功能回归的命令（不是案例计时）：

```bash
PYTHONPATH=build/linux/anytime-handoff-validation:. /usr/bin/python3 -m unittest tests.solver.test_anytime_native -v
```

## 本轮验证记录

- 新增及相关 Python 功能测试：49 项通过，没有运行全仓库测试。
- 独立构建中 LNS2 CTest：12/12 通过；未运行无关的 GPBS 测试。
- 官方初始路径 SHA 保持
  `915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf`。
- 官方修复路径 SHA 保持
  `031d1bf843ada89f03be6880809bcf7632fdaeba509fd035a15657fcc93aa83a`。
- 机器可读功能测试结果：
  `build/linux/anytime-handoff-validation/path-execution-tests.xml`；
  CTest 日志：`build/linux/anytime-handoff-validation/Testing/Temporary/LastTest.log`。
- 上述是回归与接口验证，不是地图队列的性能实验，也不证明模型改善了 SOC/makespan。
