# 仓库扰动路径集修复：报告版本与独立确认准备

日期：2026-09-10。状态：协议和方法冻结，尚未生成新地图、构造检查点或运行计时。当前产物不是新性能结果，也不是可以直接执行的计时排程。

## 1. 保留的成果与准确题目

建议报告题目：**基于动态冲突结构与显式邻域排序的仓库扰动路径集修复**。

保留当前Official、冻结V2、Dual16，不训练新模型、不加入救援重试。Dual16保留V2候选，增加冲突分量与时空热点的固定结构候选，再用同一冻结模型评价实际agent集合，交给官方PP+SIPPS修复。当前静态数组复用只属于工程实现改进。

关键边界：旧代码在既有可行路径的热点时刻插入等待，然后通过 `reset_paths()` 恢复完整扰动路径。没有将任务起点更新为扰动时刻的车辆位置，也没有在修复中锁定已执行前缀。因此本报告研究完整路径集的规划修复，不是在线执行中的机器人故障恢复；不能声称已经验证真实暂停、交通安全、lifelong MAPF或仓库吞吐率。

这一点不使旧比较无效：各方法解决同一个被认证的扰动路径输入，计时边界一致。但现实应用需要增加执行前缀、当前状态和安全约束，那是独立的后续研究，不在本次偷偷改变问题定义。

## 2. 已有结果如何写

| 已有实验 | Official / Dual16平均TTF | 支持的结论 |
|---|---|---|
| 六张新地图负载扩展，极高负载12个检查点 | 1.213 / 0.758秒 | 该层改善约37.5%，旧分层门槛通过 |
| 同次扩展，中高负载12个检查点 | 0.428 / 0.397秒 | 小幅正差，门槛未通过 |
| 同次扩展，高负载12个检查点 | 0.564 / 0.647秒 | 均值退化，门槛未通过 |
| 八图16个检查点的后验开发比较 | 0.897 / 0.497秒 | Dual16共15/16次更快，但案例已经使用，不能冒充新独立确认 |

上述各组双方均全部成功。TTF包含对应协议的reset和检查点恢复，不含此前生成可行路径及构造扰动的成本；这些成本属于离线输入准备，不能拿本表比较从零开始解决原始任务的总成本。

当前自然从头规划的高压力仓库120秒实验仍保留Official 89/96、V2 79/96、Dual16 65/96的负面边界。两类任务不得混算，不能只保留有利的恢复场景而从报告中删除从头规划失败。

来源：[负载扩展](STRIDE_WAREHOUSE_DISRUPTION_RECOVERY_LOAD_EXTENSION_V1_RESULT.md)、[16检查点开发比较](STRIDE_WAREHOUSE_COMPONENT16_DUAL16_INDEPENDENT_TTF_V1_RESULT.md)、[研究复盘](RESEARCH_DECISION_REVIEW_ZH.md)。正式JSON身份列入本次紧凑登记。

## 3. 新确认的固定输入规则

配置：[warehouse_repair_confirmation_v1.json](../configs/warehouse_repair_confirmation_v1.json)。

- 8张station-centric新地图，每张2个相同OD分布、独立task seed的任务。复用旧48×72地图生成参数，不改道路、货架或任务规则以增加有利案例。
- agent数量按可通行单元的0.15密度生成。OD权重保留storage→station 0.35、station→storage 0.35、left→right 0.15、right→left 0.15。
- master seed为2026102101。这个值仅完成了受控配置中的初步检索，尚未完成所有历史map/task seed及实际栅格SHA排重；不能提前宣称地图独立。
- 每任务先由Official生成一份无冲突路径，最多120秒、不限修复轮数。冻结这份路径供两个扰动replica共享，不让两个replica各自重新求一份可能不同的基准解。
- 从最早的最大热点占用时刻选择约15%的未完成车辆，插入4步等待，热点半径1；具体人数取整沿用原函数。没有足够热点车辆、无法生成incumbent等正常构造失败均保留，不重抽seed或换图。
- 延迟车辆位置、时间、数量和冲突图均保存。Replica共享地图、任务和incumbent，不是独立地图样本。
- 目标子集沿用冲突对至少16、活动冲突agent至少32、最大冲突分量至少16；这是输入可见的目标范围，不使用三控制器结果筛选。
- 32个构造位置必须全部有记录；至少24个合格检查点，8张图均有合格检查点，否则报告证据不足，不修改门槛。所有构造有效但不合格、零冲突及失败位置单列，报告从32到最终子集的保留率。

地图生成自身只允许原有几何合法性重试；不得根据冲突、求解时间或任何控制器输赢重试。历史隔离清单必须在生成前确定；发生种子或地图内容重复应停止审计，不能自动换seed继续。

## 4. 方法、计时与判定

主要比较Dual16对Official；V2对Official、Dual16对V2为预先声明的解释性消融，不用确认结果再挑一个新赢家。密度0.15源于旧开发结果，本次目的正是前瞻性确认该条件，不宣称整个仓库地图族有效。

最多96个episode，合格检查点全部进入三方法配对。单worker，六种方法顺序轮换，无修复轮数上限。每episode墙钟和solver预算60秒，外部进程保护90秒；完成当前episode后可安全暂停，原子落盘并支持身份严格校验的resume。

保留checkpoint-restore-inclusive口径：冷启动和模型加载单列；reset、检查点恢复、候选、特征、推理、PP及协议要求的路径合法性确认均不能从实际交付时间中事后扣除。初始状态、路径、任务及随机种子身份必须三方法完全配对。旧执行器在此处的具体时钟实现尚须复用并验证，新计时器未完成前禁止计时。

主判定不降低旧门槛：成功数不降低、无新增超时/截断、配对胜率至少70%、平均capped TTF改善至少15%、按实际占用计算的成功任务/小时改善至少20%。补充要求8图至少6图均值不劣，5000次地图级paired bootstrap的改善区间下界不低于0。成功任务/小时仅为求解器计算吞吐，不是物理仓库吞吐。

超时无解在capped统计中按60秒，原始成功TTF及路径质量另在共同成功交集报告；不将失败当零时间，不删除失败。报告中位数、最大值、慢例、逐图分布及实际分母；小样本的高分位只作描述。零分母或无法计算的指标记为不足，不自动通过。

路径质量补充SOC、makespan和等待步数；不将规划TTF直接称为车辆完成时间。负面自然规划实验和旧低负载失败必须进入局限。

## 5. 本次完成与剩余工作

已完成：恢复备份、冻结方法/配置、派生数据配置、96个未绑定checkpoint的排程位置、源码/native/model哈希快照及7项准备测试。20个只读线程用于文件哈希；没有native调用和计时进程。

入口：[prepare_warehouse_repair_confirmation.py](../scripts/prepare_warehouse_repair_confirmation.py)。它故意没有collect命令：

```powershell
python scripts/prepare_warehouse_repair_confirmation.py --output build/warehouse-repair-confirmation-preparation-copy
```

数据准备入口只提供以下三个阶段，没有solver或计时入口：

```powershell
python scripts/generate_warehouse_repair_confirmation.py inventory
python scripts/generate_warehouse_repair_confirmation.py generate
python scripts/generate_warehouse_repair_confirmation.py verify
```

`inventory`在生成前登记当前可访问的历史输入文件、种子与地图几何哈希；排除新确认目录、构建环境及依赖环境，不读取episode轨迹或outcome标签。此盘点不等价于检查已删除文件或全部Git历史，也不能证明新地图在统计意义上与历史布局独立。重复种子或相同地图内容会阻止继续，不自动换seed。

`generate`复用原生成器，拒绝覆盖已有数据；几何合法性重试沿用原协议，任务失败不重抽seed。`verify`重新读取地图、任务sidecar及MovingAI场景，检查唯一起终点、可通行性、最短距离、密度、身份和实际种子顺序。最多20个独立进程做任务校验；8图生成沿用原串行入口，避免为小批输入另改生成器。生成失败或中断的目录保留，必须审查原因，不能自动重跑。

完成8图16任务验证后，才构造32个检查点。构造和非计时校验最多20个进程，准备求解成本与正式TTF分开。旧三档负载执行器写死6图/18任务及两种控制器，不能通过改常量覆盖历史协议；新三方法执行入口尚未完成。

### 2026-09-10 数据准备结果

生成前登记提交为`0222db00504b253ff57b668e4f3f4e4e821edc66`。历史清单包含8,091份保留输入，其中1,462份MovingAI格式地图文件（含可能的历史副本，并非1,462张独立地图）。固定主种子、8个地图种子及16个任务种子未与盘点范围内的历史种子重复。

8张48×72地图、16个任务均已生成；没有更换种子或按求解表现筛选。每图两个任务有独立任务种子，但共享地图，不应作为独立地图样本。

| 地图后缀 | 可通行格 | 每任务agent数 | 任务数 |
|---|---:|---:|---:|
| 0000 | 1995 | 299 | 2 |
| 0001 | 1591 | 239 | 2 |
| 0002 | 1680 | 252 | 2 |
| 0003 | 1988 | 298 | 2 |
| 0004 | 1882 | 282 | 2 |
| 0005 | 1867 | 280 | 2 |
| 0006 | 1678 | 252 | 2 |
| 0007 | 1793 | 269 | 2 |

完整map ID以前缀`repair_confirm_v1_station_centric_`开头。原始最短距离范围为8至100步，不是多车无冲突路径长度。全部任务通过地图连通、唯一起终点、可通行性、距离、密度、manifest/sidecar/scenario一致性校验。8张新地图几何互不重复，也未重复历史盘点中的地图内容；这不证明跨地图统计独立性或性能泛化。

新准备工具及原生成器的针对性回归测试为28项通过。Windows全局Python缺少pytest，测试使用既有WSL环境，未安装依赖。独立重复运行`verify`通过且正式校验报告SHA不变。没有修改运行时、模型、native或生成器；本次没有重新跑全量Python测试、CTest或官方求解parity，也没有任何solver调用和计时episode。

数据在`build/warehouse-repair-confirmation-v1/dataset`，完整校验报告在`build/warehouse-repair-confirmation-v1/dataset-audit/dataset_validation.json`；Git中的[紧凑证据](../artifacts/warehouse-repair-confirmation-v1/dataset_validation.json)登记其SHA。旧准备快照保留为历史阶段记录，不覆盖；当前状态是数据已验证、检查点尚未构造、不能开始计时。

计时前还必须完成：三方法恢复/配对/时间边界微型测试、暂停/续跑和失败统计测试、冻结有效checkpoint清单及最终排程、登记新增执行器身份、提交推送并取得单独计时授权。此时方法与数据参数保持不变；新增适配器身份不能混入已完成历史运行。

确认通过则保留为有条件的扰动路径集修复方法；未通过则保留开发证据并停止该确认路线，不扩图、增扰动、挑更有利的案例直到通过。本轮不会训练、改初始化器、在线路由、PP顺序或救援机制。

### 检查点与执行适配

检查点准备代码先以`f84981c`提交并推送，再执行原生准备。16个任务均由Official在120秒预算内得到一份参考路径，每份路径由两个扰动副本共享；共32个检查点全部满足16/32/16门槛，8张地图均有供应。唯一冲突agent对范围222至371，活动冲突agent范围162至230，最大冲突分量范围157至228。没有换图、换seed、删除非理想结果，也没有在这些检查点上执行控制器速度比较。

准备采用最多20个独立进程，每任务外部保护300秒；该保护不是额外求解预算。`checkpoints/result-XX.json`逐任务原子保存，保存的同任务incumbent只规划一次。缺少最终结果的中断任务不能自动重跑；旧准备脚本身份保存在`preparation_identity.json`。原模型、候选生成、PP+SIPPS、native及随机流逻辑不变。

执行复用`path_quality_execution`的三方法映射、路径日志和单进程监督器，通过`initial_restore`直接接收认证的完整路径检查点，不从零重算PP。正式预算为60秒，外部进程保护90秒，取消修复轮数上限；每个episode包括独立启动、加载与完整校验记录。

**计时实现澄清，必须在正式结果产生前冻结：**旧报告的算法TTF截至native最后一步返回，最终路径校验、写出和收尾不包含在该字段中。本轮同时保存两种时间，不改写历史报告：

- `algorithm_ttf_seconds`：从`reset_paths()`前到native首次返回可行路径，包含过程中已经发生的候选生成、特征、排序、PP及日志开销。
- `validated_delivery_seconds`：从同一起点到可行路径校验、保存及算法收尾完成，复用路径执行器的`dispatch_wall_seconds`。按本轮“交付前校验不能扣除”的协议，该字段的60秒capped版本用于主门槛。它不能直接当作旧纯算法TTF来计算工程加速比。
- 正常结束且预算内找到可行解记为求解成功；超过预算才交付另外记录，不能隐去。没有成功或进程被杀死时主时间按60秒记，不把未交付的临时可行路径当正常完成。
- `process_wall_seconds`：包含冷启动、模型/输入加载及子进程收尾，用于实际计算占用下的成功任务/小时；冷启动和交付时间仍分开列出。它不是实体仓库吞吐。

固定5000次地图级bootstrap，随机种子`2026102500`。主比较是Dual16对Official；另外两项只解释结果，不选择替代模型。零分母和未完成配对不能判通过。源码及输入SHA绑定每个episode，缺失或改变的文件、路径或trace必须拒绝续跑。

准备与执行入口相互分离：

```text
python scripts/run_warehouse_repair_confirmation.py prepare
python scripts/warehouse_repair_confirmation_runtime.py validate
python scripts/warehouse_repair_confirmation_runtime.py verify
```

只有最终readiness登记、完整验证和单独授权后才能使用：

```text
python scripts/warehouse_repair_confirmation_runtime.py collect --authorize-timing
python scripts/warehouse_repair_confirmation_runtime.py stop
python scripts/warehouse_repair_confirmation_runtime.py clear-stop
python scripts/warehouse_repair_confirmation_runtime.py collect --authorize-timing --resume
python scripts/warehouse_repair_confirmation_runtime.py analyze
```

上述Python命令应在指定`Ubuntu-22.04`的项目目录中用`/usr/bin/python3`执行；脚本严格加载登记的native。`collect`没有授权参数时必须在启动前失败。`stop`只写安全停止请求，当前episode完成并校验、原子落盘后不再启动下一个；可在另一个终端执行。不要用关闭终端或强制结束WSL代替安全停止。正常超时有明确记录且不自动重试；未解释错误立即停止整组，保留现场。

最多96个串行episode，规划预算累计上限96分钟；按90秒进程保护计算最多144分钟，另加调度和外层校验。通常耗时目前未知，不将旧样本的秒级TTF当作新案例的完成保证。本次仅完成准备与功能测试，不运行这96个计时位置。

## 6. 报告先写哪些内容

建议正文先按8至10页组织，篇幅可按实际报告模板调整：

1. 问题定义与应用边界，1页：完整路径集扰动修复与自然初始规划、在线执行恢复的区别。
2. 官方InitLNS基线与方法，2页：候选生成、动态结构、实际邻域排序、官方修复。
3. 实现与可复现性，1页：原生特征、静态表示复用、冻结身份及动作一致性边界。
4. 实验设计，1至2页：旧开发证据与新确认分开、分层负载、扰动构造、共同计时口径。
5. 结果，2页：先写旧正式结果；新确认表保留待测，不预填正结论。
6. 局限与后续，1至2页：自然规划成功率缺口、旧负面方法、人工扰动/完整路径语义、缺少执行前缀保护。

不能为了报告临时降低门槛或隐藏失败；也不必等所有长尾解决才写方法、已有结果和边界。新算法研究与本版本收口分开。
