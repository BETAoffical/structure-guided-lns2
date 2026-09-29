# 冻结 A/A2 独立仓库 TTF 确认

## 问题与边界

主问题：冻结 A＋SA 相对同机制 Official＋SA 的成功率与真实 TTF 优势，能否在另一批全新仓库地图上重复？另报告相对 Official 的结果。A2 仅为附加探索对照，检验开发诊断中的省节点能否转化为真实加速，不自动替换 A。

A2 的开发结果为 40/48 对 A 的 41/48，共同成功任务节点下降 6.30%；这是有条件的成本/可靠性取舍，不是已证实的全面提升。本轮不再训练、调参或改温度，不加入密度路由，不更改官方 PP/SIPPS、候选生成和计时实现。

## 冻结身份

| 内部键 | 本轮身份 | iteration | 模型 SHA256 |
|---|---|---:|---|
| official | Official Adaptive，标准 PP | 0 | 无模型 |
| official_sa | Official Adaptive＋现有同一 SA，complete PP | 0 | 无模型 |
| parent | 冻结 A＋SA | 2 | `81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062` |
| completion | 冻结 A2＋SA，探索对照 | 3 | `9ba2192c53f02563c02b8f09637d69a5448ccfb7d847f1f88de4f642784d15a9` |

保留历史内部键仅为复用已验证的调度和统计函数。本轮 parent 不是 raw-1，也不是 Dual16。Frozen native SHA 为 `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。

## 数据与排程

- 12 张全新 station_centric 仓库地图；master seed `2026093007`，48×72 栅格参数沿用冻结生成器。
- 每图独立 d20/d25 静态 OD 任务，40% 最短路径瓶颈约束；solver seeds `[307,311,313]`。共 24 个任务、72 个配对条件、288 个 episode。
- 地图/任务 seed、地图几何 SHA 与本地历史清单和最近 cases 清单校验隔离。历史缺失文件明确报告，不能声称超出可核对范围的绝对隔离。
- 沿用既有的原采样器加 d25 端点容量匹配补救，仅处理指定采样容量错误，不看求解结果，不重抽 seed。补救任务单独标记，排除它们的敏感性分析也报告。
- 准入只检查合法性与初始路径完整，不按冲突数量筛选。零冲突和困难任务均保留，错误暂停而非换例。
- 四臂 Williams 顺序均衡，按地图、密度、seed 分层轮换。准备和事后审计最多 20 进程；计时始终 1 worker、数值库 1 线程。

## 时间与执行

- 常驻规划器 TTF 从 reset 前至首次可行返回，含初始化、proposal、特征、推理、记账及 native repair；冷启动/模型加载不混入。沿用 `deferred_scientific_audit_v1`，完整科学审计、路径验证和证据序列化不在 TTF 内。
- 每 episode 120 秒规划预算、240 秒进程安全 fuse；无决策次数和执行节点上限。25M/256 仅为原特征归一化参考，不是终止条件。
- 交付时间及 SOC、makespan、等待步数另报。理想完成时间为交付时间加 makespan×0.5/1/2 秒，不称真实机器人实测。
- 先推送代码协议，生成准入登记，再推送紧凑 ready 证据，最后单独执行 collect/all。支持 stop 在当前 episode 后暂停；resume 严格校验哈希。部分结果或错误不得自动重跑。

## 统计与解释（结果产生前固定）

所有 72 个条件报告成功数、配对新增成功/损失、10/30/60/120 秒已解数。TTF、节点、SOC 和 makespan 在各对照共同成功集合配对；完整列出非共同成功，不把超时填为零或伪装成观测 TTF。报告均值、中位数、尾部分位数、胜负数，以及 5,000 次地图级 paired bootstrap。

主对照 A 对 Official＋SA：成功数不低且共同成功平均 TTF 改善至少 5%、地图 bootstrap 未显示显著退化时，记为同族地图重复正信号。未达到则如实报告局部收益、取舍或不确定，不要求每个任务/地图都胜出，也不据此改阈值。A 对 Official 同时报告，不能用不同 SA 机制对照单独归因选择器。

A2 对 A 为探索性成本/可靠性检验，不做自动晋级；即使有正信号也不在本批数据上再更新模型。路径执行质量单列，TTF 改善不能替代整批机器人完成时间改善。本轮不是跨布局 OOD 或静态上下文迁移证明。

## 命令与交付

入口 `scripts/run_sa_frozen_confirmation_ttf.py` 支持 prepare/generate/qualify/register/verify/preflight/collect/resume/audit/report/stop/all。WSL 使用已登记 native；不安装依赖。

输出 `build/sa-frozen-confirmation-ttf-v1`：设计、地图、准入、登记、原子 episode、全量审计、JSON/CSV、生成器敏感性报告。正式数据不进 Git；提交协议、代码、测试、ready/complete 紧凑证据和中文结论。

288×120 秒规划预算上界 9.6 小时；安全 fuse 累计上界 19.2 小时，均非预测运行时间。旧批次仅可用于粗略估算，不能保证新地图难度相同。计时期间不并行运行重负载任务。
