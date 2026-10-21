# 最新单模型与四项对照：准备完成，未启动计时

2026-10-04。本批按用户确认准备五种方法，未执行求解、初始路径资格检查、候选选择、修复或计时测试，也未继续训练任何模型。

## 固定方法

| 清单标识 | 方法 | 固定版本 |
|---|---|---|
| direct | 最新单模型＋SA | 完成两轮实际反馈训练的 round-2 |
| a | A＋SA | completion-only，iteration 2；不是 A2 或 Raw-1 |
| official | 原版 MAPF-LNS2 | Official Adaptive、标准 PP＋SIPPS |
| official_sa | MAPF-LNS2＋SA | Official Adaptive＋既有 SA |
| dual16_sa | Dual16＋SA | 既有冻结 Dual16 选择器和 SA |

最新单模型和 A 已逐字节复制到本批 `models/direct.json`、`models/a.json`，原文件未改。两者文件 SHA 分别为 `1ff2ad3767ad307f83576aa26b19443aeeefaeb0e74185097ef5eda547c582db` 和 `81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062`。Dual 的模型依赖与五项方法的运行代码也已记录文件哈希；原生模块沿用 `sa-wall-clock-v1`，未重新编译。

## 已准备的数据与排程

12 张新生成的 48×72 station-centric 仓库地图，每图各一个 d20、d25 静态任务，共 24 个任务。d20 的机器人数量为 347–443，d25 为 434–554；密度按可通行格子数计算。每个任务使用 359、367、373 三个求解种子，共 72 个配对条件。每种方法各有 72 个作业，总计 360 个。

生成采用 12 个进程并行进行，均使用登记种子。本批 24 个任务全部由原随机采样器生成，预先规定的 d25 端点匹配补齐未被启用。地图、起终点、场景文件一致性和逐机器人可达性检查通过。静态可达性不证明多机器人联合问题一定有解。

新地图几何哈希与登记历史没有重叠，map/task seeds 也没有重叠。隔离范围为保留的历史快照、当前顶层 cases 登记，以及最新单模型的训练／测试来源；历史中仍有 384 个缺失地图文件，本批没有补造这些文件。这是同族新仓库任务，不是跨布局泛化实验。

五种方法在每个条件上配对使用相同任务和求解种子。执行顺序采用五方法 Williams 轮换，每种方法在每个顺序位置出现 14 或 15 次。主比较为最新单模型相对 Dual16＋SA；相对 A、原版 LNS2、LNS2＋SA 是次要比较。

## 运行状态与下一步边界

当前状态为 `prepared_not_started`。全部作业保持 `execution_authorized=false`，初始路径 fingerprint 留空。没有生成 episodes、qualification、preflight、成功率或 TTF 结果。四个纯准备测试通过；生成结束后也已核对输入、模型副本、任务文件和完整排程。

未来计时计划为单 worker 串行运行，每次 120 秒规划预算，240 秒进程保险，不设决策次数或生成节点停止上限。只测首个无碰撞方案，不执行后续路径代价优化。正式运行前，另经用户授权完成新种子的初始化资格检查和必要的接口准入，再启动计时；本轮不执行这些步骤。

将来报告全 72 个条件的成功率，以及各对共同成功条件的 TTF、胜负和地图级配对 bootstrap 区间，不以删掉某个方法独立成功的实例提高结果。

## 文件位置

本批输出目录为 `build/sa-direct-five-arm-confirmation-v1/`。

- `design.json`：固定配置、模型身份、依赖文件哈希、历史隔离范围和地图主种子。
- `cases.json`：24 个任务、静态检查及地图／任务文件哈希。
- `schedule.json`：360 个待测试作业及配对顺序。
- `schedule.csv`：便于查看的调度清单。
- `status.json`：准备完成、未启动计时的状态。
- `dataset/shards/00` 至 `11`：每张地图及两档任务。

准备程序是 `scripts/prepare_sa_direct_confirmation.py`。它只接受 `prepare`、`generate`、`schedule`、`verify`、`all`；`all` 也仅完成准备，不会开始测试。现有准备可用 `verify` 只读复核。保留清单修改前的副本位于 `../backups/20261004-before-direct-confirmation/retention_manifest.json`。
