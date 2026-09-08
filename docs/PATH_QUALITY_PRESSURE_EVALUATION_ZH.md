# 高压力路径质量：冻结排程与计时前准备

## 研究定位

本轮完整复用压力Pilot v2的48个任务，不按冲突数、控制器表现或是否容易求解筛选。
它是**开发性高压力比较**，不是独立新地图确认，不用于恢复静态上下文迁移主张。
Official Adaptive、V2、Dual16保持冻结；候选、排序、PP+SIPPS、第二阶段及计时边界均沿用原路径质量实验。
不训练模型、不加入RL，不覆盖旧252个episode的结果。

数据因素：8张仓库地图 × 3档自由格点密度（15%/20%/25%）× 2种OD模式（balanced/bottleneck_eligible）× seeds 61、62。
共48个静态任务、96个任务-seed条件，不把模型运行次数当成独立地图样本量。
瓶颈模式是40%的最短路存在性约束，不保证所有受约束agent实际经过指定格点。

## 固定排程

| 协议 | 每个episode规划预算 | 方法 | episode数量 |
|---|---:|---|---:|
| 首次可行即交付 | 最多120秒 | Official、V2、Dual16 | 288 |
| 第一阶段 + 官方第二阶段 | 总计60秒 | Official、V2、Dual16 | 288 |
| 第一阶段 + 官方第二阶段 | 总计120秒 | Official、V2、Dual16 | 288 |
| 合计 | | | 864 |

依次运行三组协议；每组96个配对条件使用六种方法排列均衡轮换，各排列16次。
不设修复轮数上限，单worker。外部进程保护上限为规划预算加120秒，不是额外优化预算。
第二阶段均为相同官方Adaptive anytime LNS，PP+SIPPS、邻域8，直接接收各自首次可行路径，使用独立且方法间配对的第二阶段seed。

主时钟仍为常驻规划器reset前至有效路径交付，另列TTF、冷启动、路径质量及交付收尾开销。
模型化整批完成时间使用交付时间 + makespan × 步长，主步长1秒，0.5/2秒敏感性，不等同真实机器人测量。
成功率用完整分母；原始TTF与路径质量在共同成功条件上配对，不给失败填零，也不隐藏非共同成功任务。
按地图、密度、OD及协议分层，保持5000次地图级paired bootstrap。多个对比是开发分析，不自动晋级控制器。

## 工程实现

新增 `pressure_evaluation.py` 薄封装，不修改旧入口的固定14/252约束。
复用旧episode执行、路径审计、身份验证、预算准入和汇总函数；新封装只适配完整高压力案例、动态总数、密度/OD报告和单独授权。
输入适配目录保存字节相同的8份地图和48组task/scenario副本。源文件与副本均登记SHA，以满足旧预检器禁止路径越出manifest目录的约束。
不存在受扰动checkpoint输入；无checkpoint清单为空。
全部源报告、reset文件、模型、代码、native及配置身份进入登记，身份变化拒绝混算。

## 命令与安全停止

```powershell
python scripts/run_path_quality_pressure_evaluation.py prepare
```

在具有已登记native的WSL环境中完成预算准入：

```bash
PYTHONPATH=build/linux/anytime-handoff-validation /usr/bin/python3 scripts/run_path_quality_pressure_evaluation.py validate
```

需要分别检查60秒、120秒的初始PP，共192次reset。三个方法共用相同初始化配置；不能只改预算名称就复用不同预算的anchor。
这不是864次正式运行，也不执行修复或模型推理。

**只有用户另行授权计时、确认接电且没有重负载后，才可执行：**

```bash
PYTHONPATH=build/linux/anytime-handoff-validation /usr/bin/python3 scripts/run_path_quality_pressure_evaluation.py collect --authorize-timing --quiet-machine --on-ac
```

在 `build/path-quality-pressure-evaluation-v1` 建立 `pause.request`，当前episode结束后暂停，不中断该任务。
移除此请求并检查环境后，原命令增加 `--resume` 可续跑。模型、代码、配置身份必须相同，异常不自动重试。
三方法的一个配对条件可能分多次会话完成，环境记录保留，不挑选性重跑较差方法。
运行正常结束后，单独执行 `analyze`；中途查阅使用 `analyze --partial`，避免把尚未完成的数据当最终结论。

## 时间规模

两个固定预算组的名义规划预算合计为288×(60+120)=14.4小时。
首解组取决于实际TTF，不能根据初始冲突数量准确预测；若所有episode用满规划预算，三组名义累计24小时。
按外部进程保护上限累计为288×(240+180+240)=52.8小时，另有调度开销；这不是正常耗时预测。
因此本轮不能承诺短时间跑完全部864次。建议计时授权时先开始首解组，可随时在episode边界暂停。
当前仅冻结完整排程，**不视为已经获得运行864次计时的授权**。

## 准备阶段交付位置

- Git配置：`configs/path_quality_pressure_evaluation_v1.json`。
- 构建输入、登记、排程、后续结果：`build/path-quality-pressure-evaluation-v1/`。
- 后续压力分层分析：`pressure_analysis/`；中途分析为`pressure_partial_analysis/`。
- 原Pilot及容量/冲突报告完整保留。

## 本轮准入结果

- 192/192预算专属reset通过，0错误，0修复，0正式计时episode。
- 60秒、120秒及上一轮Pilot的对应初始状态fingerprint逐项一致，192项比对0不匹配。
- 192份准入结果文件SHA均核对通过；没有遗留collection锁或计时授权文件。
- Windows相关Python测试60项通过，WSL新增封装测试6项通过，覆盖未授权拒绝、六种方法顺序、864/192计数、文件篡改拒绝、暂停和动态完成数。
- 原控制器、C++、native、旧计时模块均未修改。本轮未重跑CTest；沿用上一轮已通过两组parity的同一native SHA。
- 原252次正式报告SHA仍为 `67134c59382d953bb9ed9c94d1a85d4b070f2e83b6acf713c0b12834a1e90538`。

冻结登记fingerprint：`1eb70352da48187b1d0d474a8d6bf3eb50c850ebb672210e46cdf19afbf55e3b`。
排程SHA256：`7db15ebc3df546f50a3369f0149106d79fafadfaac627b87731899847766ac9f`。
准入报告SHA256：`6168029633589522008a88c4ea4a7111eb520b2ec07cf7e1da31b969e1a0bdd2`。

本轮没有新的速度、成功率或路径质量比较结论；只表明数据及运行入口已通过计时前准入。

## 2026-09-08 截止时间边界修复

上文是原准备版本的历史记录。获得授权后，第一批计时运行到第345条发生错误：
Dual16、457 agents、seed 61、60秒总预算。已记录284条正常完成、60条预算内无解和1条错误；原数据全部保留。

独立诊断重建440次transition并逐项校验fingerprint。最后一次修复墙钟为59.993611秒，仍有2对冲突。
相同路径和冲突图下，288个请求在预算充足时全部有效；批量调用跨过截止时间时，后续请求被native拒绝，
原接口却统一记为invalid，导致上层抛出`valid online proposal was rejected`并停止整组。
这是共用proposal接口的终止分类缺陷，不是模型训练或第二阶段优化错误。

修复分支：`codex/path-quality-proposal-deadline-fix`。原冻结提交仍为`6b20d653504128206b551f097e27bf01ecb4e190`。

- Native仅给合法且因时间耗尽停止的请求标记`deadline_exhausted`。非法seed、未初始化、已可行和修复轮数截止不得冒充时间耗尽。
- 正常proposal输出不变；只有到期返回才增加标记。Grouped输出增加`deadline_indices`；dict输出增加`deadline_exhausted=true`；compact到期行增加第四项`true`，正常行仍为三项。
- Python在明确标记、终止状态及路径/冲突图/计数未变的条件下，抛出专门的`ProposalDeadlineExceeded`。真正非法请求或状态变化仍是硬错误。
- Worker丢弃不完整候选池，不排序、不消耗下一次repair动作、不执行额外修复，按`wall_timeout`正常收尾。若始终未可行，外层结果为`no_feasible_solution`。
- 混合了非法请求的批次不得被转为超时；参考、compact、grouped和shadow路径均有边界测试。
- 没有改变模型、PP+SIPPS、邻域规则、预算或失败惩罚，没有直接修改原错误记录。

新native只编译到`build/linux/proposal-deadline-fix-validation/`，不覆盖旧构建：

| 身份 | SHA256 |
|---|---|
| 旧正式native | `1f3e41f9853a88acc640d666429508567f469fd3d722c48727eff24923c154aa` |
| 修复版native | `7e84f535cc992d7424959cbafe63fce122a5b97c952b92c68d8978093f190e08` |

未到期等价验证使用同一任务、三个控制器、seeds 61/62、每条5次修复，共30次transition、380个候选与特征向量。
比较的是旧Git源码+旧native和修复源码+新native，而不是仅替换Python后比较两个二进制。
候选、特征、分数、显式动作、repair order、前后状态fingerprint及低层节点计数逐项一致；科学字段签名SHA均为
`56f40ed64ab83c0138fc810af147d450072bde905fd073cec7636ae479d18875`。
这属于有限样本的功能等价证据，不是全部实例或墙钟性能不变的证明。

最终验收：Python `1278 passed, 47 skipped`；LNS2 CTest `12/12`通过（不包含未构建且不参与本次比较的两项GPBS测试）。
Python中的原生环境测试由CTest单独设置环境后执行，已包含真实native到期联通测试。
两组官方路径SHA保持`915ee104...13ecf`与`031d1bf8...aa83a`。旧native及事故四份结果文件SHA保持不变。

验证文件在`build/path-quality-pressure-batches-v1/`下：原事故诊断、deadline复现和`equivalence-reference/fixed`短轨迹。
这次修复验证不授权恢复原目录的计时，不修改旧registration，不将新旧结果静默混算。
正式恢复前还需登记新源码/native，决定旧结果复用与必要的配对补测协议。

后续已按协议边界完成修复版登记及384次初始化核验，见
[截止边界修复版恢复协议](PATH_QUALITY_DEADLINE_RECOVERY_ZH.md)。上文旧登记与结果继续保留，不用于直接续跑修复版。
