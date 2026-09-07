# 路径质量高压力 Pilot：准备协议

后续容量诊断、独立v2修订及96次初始PP检查见 [高压力Pilot v2报告](PATH_QUALITY_PRESSURE_RESET_ZH.md)。本文保留v1失败记录。

## 边界

本轮是复用地图的实验设计 Pilot，不是新的正式确认，不训练模型，不运行求解器或计时。
旧的 14 任务、252 episode 协议、结果和冻结控制器均不修改。不能将本轮输出续跑到旧计时目录。

## 数据设计

- 复用原确认集的全部 8 张 station-centric 仓库地图，地图文件不复制、不改动。
- 可通行格点密度为 0.15、0.20、0.25；这是设计因素，不预先称为低、中、高冲突。
- 每个密度有 balanced 与 bottleneck_eligible 两种任务，共 48 个独立静态 OD 任务。
- 两种任务使用相同的加权 OD 配置：storage->station 35%、station->storage 35%、left->right 15%、right->left 15%。实际数量由既有生成器采样，并非精确配额。
- bottleneck_eligible 要求约 60% agents 存在经过结构先验高值格点的最短路。它不强迫初始 PP 使用该路，也不证明真实交通一定拥堵。
- 地图、密度、模式与 master seed 共同决定 task seed，不使用低密度任务的 agent 前缀。保留生成失败，不自动换 seed、降低约束或退化到普通任务。
- 后续 reset-only 使用 seeds 61、62，完整设计对应 96 个 reset。当前尚未执行，不能宣称已有高冲突实例或证明联合可解。
- 只检查新 task seed 与本次源清单及彼此之间的隔离，不宣称与全部历史训练数据完全隔离。

## 命令

```powershell
python scripts/prepare_path_quality_pressure.py
python scripts/prepare_path_quality_pressure.py --verify
```

入口不提供 collect、reset、训练或计时参数。输出到 `build/path-quality-pressure-pilot-v1`。
registration 绑定配置、生成器、静态审计实现和源地图 SHA。每任务单独保存完成标记；中断后仅复用身份和输出 SHA 一致的任务。
失败标记也会被复用，不隐式重抽样。更改设计或实现需要新输出版本。

## 171 步长尾案例复核

源报告：`build/path-quality-evaluation-v1/analysis/report.json`。
案例：`very_high_confirm_v2_station_centric_0001__task_0000`，solver seed 52。
读取保存的 first_feasible artifact，并核对报告登记的文件 SHA。没有重新执行修复。

agent 253 的最短路为 36 步：

| 方法 | 该 agent 路径步数 | 移动步数 | 等待步数 | 全组首次 makespan |
|---|---:|---:|---:|---:|
| Official Adaptive | 67 | 62 | 5 | 114 |
| V2 | 36 | 36 | 0 | 114 |
| Dual16 | 171 | 168 | 3 | 171 |

Dual16 的额外 135 步由 132 步额外移动和 3 步等待组成，主要是绕行，不是等待。
native 的 delay=135 表示路径成本减最短路成本，不能解释成等待 135 步。
不同方法选中的修复集合可能改变重规划约束，但只凭最终路径不能定位首次引入长绕行的具体动作，亦不能将其归因为代码 bug。

相同案例的第二阶段独立预算实验，最终整组 makespan 为：

| 总规划预算 | Official Adaptive | V2 | Dual16 |
|---|---:|---:|---:|
| 60 秒 | 119 | 99 | 101 |
| 120 秒 | 102 | 99 | 101 |

这说明该长尾在后续优化后明显缩短，但不能推广为所有长尾都会消除。官方第二阶段优化 SOC，并不保证 makespan 单调下降；此例 Official 在 60 秒协议下从 114 变为 119 就体现了二者的区别。

## 下一步门控

1. 完成静态生成和检查，列出失败任务及约束原因。生成失败是任务设计/采样问题，不能当作算法求解超时。
2. 失败时不删除对应地图或偷偷换 seed；先审计候选端点容量和有限采样失败，再决定是否建立明确的新 Pilot 版本。
3. 静态设计可用后另建 reset-only 准入，登记当前 native 身份，统计初始冲突对数、归一化冲突密度、冲突分量、实际 PP 瓶颈利用与初始等待。
4. 从初始 PP 而非扰动 checkpoint 出发；零冲突保留，不用控制器性能筛选正式案例。
5. 根据独立设计 Pilot 冻结正式任务、预算和统计协议，另行取得计时授权。Official、V2、Dual16 保持冻结；同一任务和 seed 配对、串行运行、episode 结束后可安全停止。
6. 分别报告成功率、TTF、交付时间、SOC、makespan、等待和理想执行完成时间。扰动修复作为另外一组，不与从头规划 TTF 混算。

当前旧 cohort 执行器有固定 14/252 准入约束；本轮没有绕过或改写它。新的 Pilot manifest 也不直接冒充可计时 cohort。

## 本次准备结果

48 个排程位置全部有结果：46 个通过静态校验，2 个生成失败。

| 自由格点密度 | 排程任务 | 静态有效任务 | 有效任务 agent 数范围 |
|---|---:|---:|---:|
| 15% | 16 | 16 | 223–303 |
| 20% | 16 | 15 | 298–405 |
| 25% | 16 | 15 | 372–506 |

两个失败均来自地图 0000 的 bottleneck_eligible：20% 密度在 agent 182、25% 密度在 agent 181 处用完有限端点采样尝试。
这只证明当前采样器在该配置/seed 下未完成，尚不能证明端点组合在数学上不存在，也不是 PP/LNS2 超时。
下一步先分析候选端点容量和唯一性约束，不直接提高采样次数或去掉失败地图。

准备报告身份：`5e5c66946e3cf7ecf6d7cbef8c8e56a7d4a55e1138918a908b39ecf35b2d551d`。
`--verify` 的输入/输出哈希检查通过，但按 fail-closed 约定仍返回 exit 1，表示存在生成错误、不能进入准入；不是验证工具崩溃。
原正式报告 SHA 保持 `67134c59382d953bb9ed9c94d1a85d4b070f2e83b6acf713c0b12834a1e90538`。

Windows 全局 Python 未提供 pytest，本轮未安装依赖。使用标准库 unittest 运行 `test_path_quality*.py`，44 项通过，其中 7 项新增。
未重跑 Linux CTest、官方 parity 或任何计时实验；本轮没有修改原控制器、native 或旧计时协议。
