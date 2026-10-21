# 五方法独立比较：已授权执行

2026-10-04，用户在准备完成后回复“开始”，授权运行既有五方法清单。准备阶段文件、配置、冻结模型和全部任务不改，实际执行状态另外保存在实验目录的 `run_status.json`。

## 准入结果

72 个初始状态检查全部通过，未剔除任何条件。第一张地图 d20/d25、seed 359 上的五方法接口检查共 10 项，全部通过。最新单模型与 A 的候选集合及共同特征一致；最新单模型不调用 Dual 评分。原版 Official 的有时限接口与直接原版步骤一致。三个纯执行排程测试也已通过。

## 执行与文件

正式清单仍为 12 张地图、24 个任务、三个求解种子、72 个条件、360 个作业。比较最新单模型＋SA、A＋SA、原版 MAPF-LNS2、MAPF-LNS2＋SA、Dual16＋SA。计时采用一个 worker，每次规划预算 120 秒，进程保险 240 秒，不继续训练或切换默认模型。保留失败；故障或保险中断停止后续任务，不自动重试或将未知改写为普通失败。

准备时关闭的执行开关未在原文件中改写。另存的 `execution-registration.json` 给每个作业补入授权和已核查的初始状态 fingerprint，固定新的执行来源身份。运行入口位于本批实验目录，而不修改冻结的生产代码或其保留清单。

目录为 `build/sa-direct-five-arm-confirmation-v1/`，主要文件如下。

- `execution-authorization.json`：本轮用户授权及初始化检查代码身份。
- `qualification.complete.json`：72 个初始状态及证明文件哈希。
- `preflight.complete.json`：10 项接口检查。
- `execution-registration.json`：固定的实际执行清单与运行入口身份。
- `execution-backup/`：启动前登记、清单和运行代码副本。
- `launch.json`、`execution.log`：独立后台进程身份及运行日志。
- `run_status.json`：当前状态、完成数及活动方法。
- `episodes/`、`collect/`：逐次原始状态、轨迹、结果及执行回执。
- `audit/`：计时结束后的轨迹／概率／修复规则核对和终态碰撞检查。
- `report.json`、`results.csv`：全部作业完成且核对通过后生成的结果。

轨迹核对在整个计时批次结束后以 12 个 worker 并行进行，不与正式计时争抢算力。最新单模型核对保存轨迹的输入特征、概率、动作、SA 随机流和状态转移；其他方法复用既有核对实现。终态另查顶点冲突、边交换冲突及到达目标后的占用。核对不重新执行求解器。

原 `status.json` 是准备阶段的不可变记录，仍写着 `prepared_not_started`，不表示正式比较没有运行。当前状态以 `run_status.json` 为准。

## 操作入口

在现有 Ubuntu-22.04 环境、仓库根目录使用 `/usr/bin/python3 build/sa-direct-five-arm-confirmation-v1/run_frozen_comparison.py status` 查看状态。`stop` 要求完成当前一次作业后停止；`resume` 只续接完整的已有回执，遇到故障或残缺作业会要求检查，不覆盖现场。不要重复执行 `launch`。

汇总以每方法完整 72 个条件的成功率为准。TTF 在各对共同成功条件上比较，不把失败填成观测 TTF；使用固定的 5,000 次地图级配对 bootstrap。主比较为最新单模型相对 Dual16＋SA，其他比较属于次要比较。不拼接旧批次数据，也不从这批任务选择或训练新模型。
