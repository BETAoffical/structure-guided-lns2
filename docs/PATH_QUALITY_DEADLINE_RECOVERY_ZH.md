# 路径质量计时：截止边界修复版恢复协议

## 范围与证据边界

2026-09-08审计确认：旧第一批345条记录中284条正常完成、60条正常无解超时、1条已知程序错误。
119,726次修复转移和3,327个episode文件通过核验。旧结果不覆盖、不删除、不事后改写错误分类。

本轮采用完整协议作为版本边界：

| 数据组 | 协议 | 修复版运行数 | 旧结果处理 |
|---|---|---:|---|
| 第一批 | 首次可行，120秒 | 0 | 完整288条独立保留 |
| 第一批 | 第一阶段加官方第二阶段，60秒 | 288 | 旧57条仅保留在旧版记录中 |
| 第一批 | 第一阶段加官方第二阶段，120秒 | 288 | 此前未执行 |
| 第二批 | 首次可行、60秒、120秒三个协议 | 864 | 此前未执行 |

后续共1,440条，包括57条重新执行和1,383条原排程未执行项。
57条中56条原数据有效，重新执行是为了统一未完成的60秒协议运行时。
最小配对复测只需要最后一组三方法，共3条；本方案额外执行54条60秒任务，按旧记录预计增加约0.95小时。
以上是排程条数，不是新增1,440个不同任务。

首次可行协议的两批结果来自不同运行时版本，必须分别报告。
不能将旧版与修复版静默合并成同一版本的墙钟实验，也不能事后扣除日志验证/路径交付开销。
这仍是复用仓库布局的开发性证据，不是独立地图泛化或真实机器人实测。

## 配置与身份

- 第一批：`configs/path_quality_pressure_deadline_recovery_v1.json`。
- 第二批：`configs/path_quality_pressure_deadline_recovery_replica2.json`。
- 输出：`build/path-quality-pressure-deadline-recovery-v1/` 和 `build/path-quality-pressure-deadline-recovery-replica2/`。
- Native：`build/linux/proposal-deadline-fix-validation/lns2_env.cpython-310-x86_64-linux-gnu.so`。
- Native SHA256：`7e84f535cc992d7424959cbafe63fce122a5b97c952b92c68d8978093f190e08`。
- 审计SHA256：`15404542cf58bf6eae5f131b9fb0d60550e8d877690f9ce73913efa88b006bd3`。
- 推荐排程SHA256：`3310bf6f4ab313444b2680de072f4f84ee8c5c0a689abe17d4cefe7c544dfba0`。

两个配置均绑定原registration、只读审计和推荐排程。准备时逐项核对原任务、seed、方法顺序和预算。
新目录不得与旧目录重叠。第一批保留原schedule_index 288至863，不重新编号或更改job ID。
未指定`protocol_scope`的旧设计仍保持864条排程语义。

## 本轮只做准备与reset准入

在WSL项目目录运行以下命令，第二批替换配置文件名即可：

```bash
PYTHONPATH=build/linux/proposal-deadline-fix-validation /usr/bin/python3 scripts/run_path_quality_pressure_evaluation.py prepare --config configs/path_quality_pressure_deadline_recovery_v1.json
PYTHONPATH=build/linux/proposal-deadline-fix-validation /usr/bin/python3 scripts/run_path_quality_pressure_evaluation.py validate --config configs/path_quality_pressure_deadline_recovery_v1.json --resume
```

每批重新执行192次reset，共384次：48个任务乘2个solver seed乘2个预算。
60秒和120秒均需实际初始化；不把旧anchor换名或直接绑定到新native。
新旧对应状态fingerprint及路径必须一致，0修复、0正式计时episode。
若不一致，停止准入并调查，不重抽seed、不删除难例。

Windows与WSL共享文件时，若WSL Git仅因CRLF显示工作区脏，可在进程内使用
`GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=core.autocrlf GIT_CONFIG_VALUE_0=true`；不能重写源码来掩盖身份变化。

## 后续启动边界

计时仍须单独授权，显式`collect --authorize-timing --quiet-machine --on-ac`，并验证干净分支已推送。
准备输出中的`pause.request`是额外安全闸，授权后才能移除。
运行中重新建立该文件，当前episode结束后安全暂停；异常不自动重试，先保留记录并检查。
三方法配对组可以跨会话完成，但不允许只重跑表现较差的方法。

续跑先验证已有manifest为完整排程前缀及结果artifact身份，保留原机器环境，不重跑已完成任务。
暂停后的进度以已完成数量继续累计。若某episode已有binding却尚未写入manifest，必须先审计中断现场，
不能自动重试或把恢复时的机器环境冒充原运行环境。

正式运行仍使用单worker，无修复次数上限，原规划预算及预算加120秒的进程保护上限不变。
本轮不训练模型、不修改候选生成/排序/PP+SIPPS、不自动恢复半小时巡检，不执行正式计时。

## 2026-09-08 准入完成

实现提交：`16fd6529163b8952a5ec56dae94ea6fc8b07aabb`，分支`codex/path-quality-pressure-deadline-recovery`。

- 两批各192次reset，总计384/384通过，0错误。
- 384项新旧state fingerprint、完整路径和初始质量指标逐项相等。
- 旧3,327个episode文件SHA全部保持不变；旧错误仍保留为错误。
- Python最终完整测试：`1288 passed, 47 skipped`；LNS2 CTest：`12/12`通过。
- 两项GPBS测试不在本次比较范围，未执行。原生Python接口测试由CTest专用环境执行。
- 官方路径SHA保持`915ee104...13ecf`和`031d1bf8...aa83a`。
- Windows全局Python缺少pytest，使用WSL现有pytest验证，没有安装新依赖。
- 两个新输出目录均保留`pause.request`，无计时授权文件、无正式episode。半小时巡检未恢复。

| 身份 | 第一批 | 第二批 |
|---|---|---|
| 排程条数 | 576 | 864 |
| Registration fingerprint | `a098e5ef1b9f92f1d00b8a6afabe516250b0361325c2478d6b2d2d37114b682a` | `50a78b94c8dab66ab4f7861907401d5c4b10897d6b4afa4ad204b23ba8bd0f97` |
| Admission SHA256 | `1646fbeaa0111aabf55e34a8b035a81ed2ba0b8e4250c1b479790978510cb2c9` | `075175132e5f6cec1e455005db85c38238cf623c0a5b1ce3aa40d015bc1a25dd` |

详细准入报告和测试日志保存在`build/path-quality-deadline-recovery-preparation-v1/`，
入口报告为`readiness.json`。当前结论仅为“计时前准备与初始化一致性通过”，不是新的性能结论。
