# Official+SA 高压力补充实验：计时准入

2026-09-16完成准备，**停在正式计时前**。未执行collect/resume；正式episodes目录和manifest均不存在。

## 冻结范围

- 仅新增Official+SA，8张仓库地图、48任务、seeds 61/62，共96次运行。
- 单worker、120秒预算、240秒进程fuse、无修复次数上限。没有重启、第二阶段或模型训练。
- 使用历史冻结native，不重编译后续修改过的C++源码。
- 与旧Dual16+SA比较属于跨批次探索性证据，不自动晋级、不声称同期因果加速。
- 运行协议见 [OFFICIAL_SA_PRESSURE_SUPPLEMENT_PROTOCOL_ZH.md](OFFICIAL_SA_PRESSURE_SUPPLEMENT_PROTOCOL_ZH.md)。

## 本轮检查

| 项目 | 结果 |
|---|---|
| 48任务静态数据及原登记 | 通过 |
| 历史950个证据文件，约425 MiB | SHA全部匹配 |
| 96次初始化，20个独立进程 | 96/96路径与fingerprint完全匹配 |
| 15%和25%任务的两个2秒功能smoke | 2/2流程通过 |
| smoke实际修复调用分支 | 两条轨迹分别14与33个决策，均经完整轨迹审计 |
| smoke退出 | 一条正常可行交付，一条正常预算内未解；均不是正式性能数据 |
| 相关Python测试 | 82 passed，无跳过；不是全仓库测试 |
| 冻结LNS2 CTest | 12/12；未运行无关GPBS测试 |
| 官方初始路径parity | `915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf` |
| 官方修复路径parity | `031d1bf843ada89f03be6880809bcf7632fdaeba509fd035a15657fcc93aa83a` |
| 安全暂停/续跑/篡改检测/外部超时 | 定向回归通过 |
| 最终ready检查 | `ready=true, timing_authorized=false, formal_started=false` |

程序对旧三个控制器仅保留原行为，新增Official+SA的三处分支匹配：官方job映射、退火step调用、退火轨迹审计。官方邻域选择和native PP不改。

## 已修复的问题与备份

第一版smoke把执行预算缩为2秒后，误用该预算查找120秒登记的初始化锚点，出现KeyError；发生在求解器启动前。
现在锚点始终绑定120秒登记，smoke执行预算仍为2秒，并增加专门回归测试。
旧准备目录完整保存为 `build/official-sa-pressure-supplement-v1-pre-smoke-fix`，没有删除失败记录。
修复后重新冻结登记并重做96次初始化准入；正式实验始终没有启动。

- 修改前恢复标签：`pre-official-sa-pressure-preparation-20260916`。
- 初始实现提交：`9550574`；修复并用于最终准入的实现提交：`c19393b`。
- 源码ZIP：`build/official-sa-pressure-supplement-v1/source-c19393b.zip`。
- 源码ZIP SHA：`31d73987bef382d55da839488bf9c1b1cee1992639f810a2e2790679bd87ad93`。
- 以上是本地备份，本轮未推送远端。

## 当前登记

输出目录：`build/official-sa-pressure-supplement-v1`。

| 文件/身份 | SHA256或fingerprint |
|---|---|
| 登记fingerprint | `08967418051a2ddb1f47c14ec698b8db0fd03dd43991bda57e2e49caa55df615` |
| registration.json | `c56008aa9fac9d3a54ee4441c277f72b38dfb77dcac9e95127bc90c411f66279` |
| admission.json | `046258cde693af7a3b1d234e2ea991c9e6f9cfb309eda4a92f002114b052a593` |
| smoke.json | `7f85b13ff36c87fdfbf304c93fa97c09a6bc8e3b8b23b817aa552d10effc9026` |
| 实际加载native | `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d` |

获得单独计时授权后才执行协议中的collect命令。保持接电、固定电源模式、避免重负载程序。
可随时请求完成当前episode后暂停。正常求解超时记录后继续；外部fuse或完整性错误暂停并先审计，禁止静默重跑。
