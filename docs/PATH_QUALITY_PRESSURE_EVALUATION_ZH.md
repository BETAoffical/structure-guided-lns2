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
