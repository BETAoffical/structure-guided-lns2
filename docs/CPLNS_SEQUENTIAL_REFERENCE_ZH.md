# CPLNS 作者单求解器参考：实现边界与准入

## 本轮目标

建立作者公开代码的未修改参考，不替换官方 MAPF-LNS2、V2 或 Dual16。
固定提交 `e3fbcc82bf6a1f2cbab0ceb166d547865ca8210a`，来源 <https://github.com/KaenChan/cplns>。
源码及二进制仅留在 `build/cplns-sequential-reference-v1`。未见顶层 LICENSE，不将作者源码重新发布到项目 Git。
恢复标签 `pre-cplns-reference-20260911`；分支 `codex/cplns-sequential-reference`。

上次意外中断时已完成下载与 CMake 配置，未产生 plns，也无残留编译/求解进程；本轮复用目录继续构建。
已有 GCC 11.4、CMake、Boost 1.74 program_options/system/filesystem、Eigen 3.4，未安装任何依赖。
原始 CMake Release 构建，12 路编译；保持作者编译警告原样，不顺手修改其实现。

## 固定运行条件

- 1 个求解器、1 个 PP 求解器、1 个分组；程序仍有调度线程，不等于整个 OS 进程只有一个线程。
- PP 初始化、PP 修复、SIPP 系列低层、Adaptive 邻域，邻域大小 8，启发式权重为 1。
- 保留作者单组内部记账/同步开关；只有一个求解器，不存在多个求解器之间交换不同解。
- `maxIterations=0` 在该代码用于关闭后续 CostLNS 迭代，不限制 InitLNS 修复次数。
- 不加载本项目模型，不使用 Dual16 候选池，不引入 fingerprint 派生的 proposal seed。
- 无解前始终仅允许合法中间路径，不能把碰撞路径当作最终可行解。

四个参数对照仅用于确认开关边界，不做性能排名：

| 名称 | sa | sa_max_con_fail | 意义 |
|---|---|---|---|
| author_no_sa_no_restart | false | 2 | 作者非 SA、关闭阈值触发的重启 |
| author_sa_no_restart | true | 2 | 作者 SA、关闭阈值触发的重启 |
| author_no_sa_restart | false | 0.4 | 作者非 SA、保留重启 |
| author_sa_restart | true | 0.4 | 作者默认 SA 与重启 |

作者的失败比例不超过 1，阈值 2 使正常单求解器的比例重启条件不成立；不是改变源码。
**不把 author_no_sa_no_restart 命名为官方 LNS2。** 作者关闭 SA 后提前中止条件为 `>=`，本项目官方兼容版本为 `>`。
作者的 `--sa false` 不会关闭重启。论文算法 5 与公开代码的失败计数、重启条件也有差异；本参考以固定源码为准。
SA 开启时完整 PP 后接受非劣解；劣解按 exp(-delta/(T+epsilon)) 接受，C rand 同时服务其他随机动作。
默认 T=1000、每轮乘 0.99；重启恢复温度并重新生成路径。公开代码还要求迭代数超过 300 才触发比例重启。

## 微型正确性检查，不是研究数据

两个固定 8-agent 工程 fixture：8x8 开放平行通行，以及 1x10 单通道顺序反转。
后者不能在单通道中交换车辆顺序，预期本预算不提供无冲突解；它不是用来评价算法在仓库上的成功率。
每种 fixture、4 种配置、种子 0/1，共 16 个作业。单个 solver budget 2 秒，外部 fuse 20 秒。
最多 20 个独立进程；每个 plns 内仅一个求解器。总预算小于一分钟量级，但不作为性能结论。
逐作业日志、返回码、命令、SHA 和原子 JSON 单独保存。超时不自动重试，resume 验证已有结果及日志。

准入检查：

- ZIP SHA 固定；解压后的每个文件逐字节匹配 ZIP，禁止未登记文件、修改和符号链接。
- `--help` 在作者程序中正常返回 1，不能套用零返回码规则；其余执行必须正常退出并有唯一 final 行。
- 开放 fixture 必须报告零冲突，并经过作者 `validateSolution()` 调用；单通道不得错误报告零冲突。
- 汇总重新解析日志，不信任可单独修改的 JSON 摘要。源码、模型/策略无关路径和二进制身份固定。
- **作者 CLI 未导出最终路径供独立重算。** 内部校验不等于项目独立路径校验，必须明确保留此未完成项。
- `t1/t2` 是作者强转整数后的值，禁止把 0 当作零耗时，禁止本轮生成 TTF 优势百分比。
- 原始 restart 计数含进入初始化轮次，不能直接当作实际重启次数；`restart_log_events` 只计匹配日志行，包含退出摘要，也不能当作实际重启次数。

## 复现命令

在 WSL 仓库根目录，以已有系统 Python 运行，不安装 sklearn：

```bash
cmake -S build/cplns-sequential-reference-v1/cplns-e3fbcc82bf6a1f2cbab0ceb166d547865ca8210a -B build/cplns-sequential-reference-v1/build -DCMAKE_BUILD_TYPE=Release
cmake --build build/cplns-sequential-reference-v1/build -j 12
python3 scripts/verify_cplns_reference.py prepare
python3 scripts/verify_cplns_reference.py verify
python3 scripts/verify_cplns_reference.py smoke
python3 scripts/verify_cplns_reference.py report
```

prepare 不能覆盖已有 registry；源码或二进制改变后拒绝续跑。所有数据留在忽略目录。

## 后续门槛

本轮最多确认“作者 CLI、参数和小例子运行可用”，不是复现论文性能，也不解释此前 Dual16 的全部失败。
下一步须先补只读观测桥：导出初始/最终路径、准确首次可行事件、逐次接受及重启记录，
并与未修改 CLI 在相同种子上对照，证明观测不改变动作，再进行有限真实案例验证。
只有可信参考在相同预算下给出可复核收益后，才考虑将其中机制接入 Dual16；不继续对混合版本盲调温度。

## 本轮实际结果

实现登记提交 `0644633`。148 个作者文件与 ZIP 逐字节一致，参考二进制 SHA256：
`c57de074f49065a39f1f1ad6e87b698964df3b9ea3b7ba5cb51746fc1e867445`。
registry SHA256：`b18271370b62f2f84299fa7b35353c9a196c3940b444bded9763ecb95fe0572a`。
smoke report SHA256：`15a53e1f2666d2226d939583b80db65c033620d8765fe1194b2970934a3518ae`。

- 新增适配器及维护检查 21/21 通过。
- 项目分区回归：并行 1557 passed、68 skipped；时间敏感/历史 native 身份检查另行串行 9/9 通过，未放宽断言。
- 项目 LNS2 CTest 12/12 通过，两组官方 parity 保持不变；GPBS 未纳入本次 CTest。旧冻结 native SHA 仍为 `7e84f535cc992d7424959cbafe63fce122a5b97c952b92c68d8978093f190e08`。
- 16/16 工程作业符合预期，均正常退出，无外部 fuse；开放案例 8/8 报告可行且经过作者内部校验，SOC=56。
- 单通道 8/8 保持非零冲突，未误报可行；4 种配置的最终冲突均为 28。这里的不可换序输入不是性能评测数据。
- 关闭重启的两个配置，InitLNS 摘要均为 restart 0。
- 非 SA + 开启重启，两个种子均实际重启 55 次；主程序共享计数打印 56，证明不能直接拿该字段当实际重启次数。
- SA + 开启重启，两个种子均实际重启 0 次；平台上完整重规划被接受但冲突不减少，并未持续累计拒绝失败。
  这与公开代码计数规则一致，不是参考移植漏掉了重启；也说明不能假定原版重启必然解决所有平台停滞。
- 这组 fixture 没有触发正 delta 的退火接受，尚未动态覆盖这一分支；不得将 CLI smoke 写成完整机制验证。
- main.cpp 采用约 1 秒的调度轮询。开放案例即使很快找到解，进程总耗时也约 1 秒；加上截断的 t1，现有输出不适合亚秒级 TTF 比较。
- 未解决输入的 SOC 字段可能是 `MAX_COST=1073741823` 哨兵，不是有效路径成本；本轮不汇总其 SOC 或计算提升率。

仍待完成：独立路径导出/重算、精确首次可行事件、接受及重启观测桥的等价验证，以及能触发正 delta 的非计时机制案例。
本轮没有运行正式 TTF、没有接入 Dual16、没有训练模型，也没有改写前一轮负面结论。
