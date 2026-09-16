# SA 历史条件化选择器：独立训练 Pilot

## 授权与边界

2026-09-16。用户授权重新设计模型、开始训练，并允许小规模新标签采集。
冻结 Official、Dual16、Dual16+SA、native、原模型和旧结果；不做正式 TTF、不训练 RL。
新名称 `sa-history-transition-v1`。本轮模型只保存为诊断 artifact，不注册为运行时控制器。
分支 `codex/sa-history-selector-pilot`；开始前恢复标签 `pre-sa-history-selector-pilot-20260916`。

## 历史审计与实质区别

本轮对全部保留 docs 文档建立路径、标题、SHA 清单；这是目录级证据盘点，不冒充每份全文均已人工精读。
以下直接相关报告进行了内容核对；旧清理标签中的源码不恢复到活动目录。

| 历史方向 | 实际训练情况及失败 | 新设计必须避免的问题 |
|---|---|---|
| 旧 policy-visited、目标对齐、容量、图模型 | 训练过；未稳定改善选择或闭环 | 不把加树、换网络或仅扩大旧标签当新设计 |
| STRIDE Stage4 / 4R | 600状态36图训练；离线改善，小规模TTF反而变慢 | 一步预测不是长期完成能力，旧新质量标签约98%同最优 |
| SlotPool | 真正训练并通过候选质量确认，后续TTF慢4.59% | 不把候选保留率当端到端成果 |
| Temporal-anchor incremental | 已拟合离线GBDT；历史增量AUC约0.0136未过门槛；没有精确不变transition | 不能宣称“历史从未训练过”；不同SA采样才是本轮差别之一 |
| 早期pre-action trigger | 拟合过；按状态去重导致全部历史计数为0 | 保留 episode+decision，不按物理状态去重 |
| HistoryRank | 规则和13个单特征筛查；未训练完整多特征历史模型 | 不要求每个特征独立单调有效，也不预设新颖度越高越好 |
| CycleTransition | 5432次标签，仅18个soft-cycle；未训练 | 不以“下一步再次选同一集合”的稀疏联合事件作标签 |
| ResidualHazard | 6种结构指标的尾部方向跨图不一致；未训练 | 不把边保留直接称为长尾概率，不事后翻转标签符号 |
| MultiValue / H4 | 长期排序与跨teacher稳定性未过 | 不拟合剩余修复轮数，不把旧后缀拼到新动作 |
| SA readiness | 只有3个同图反事实状态；没训练SA新模型 | 收集未选动作真值，执行轨迹不能凭空补标签 |
| Feedback exploration | 随机与反馈均有收益及损失，反馈无明确增量 | 不包装固定换候选、永久禁忌或精确成功键复用 |

重点依据：`REJECTED_METHODS_LEDGER.md`、`EXPERIMENT_LESSONS.md`、
`INITLNS_RESEARCH_REPORT_ZH.md`、`STRIDE_STAGE4_RESULT.md`、
`STRIDE_HISTORYRANK_FEATURE_V1_REPORT.md`、`STRIDE_CYCLETRANSITION_PILOT_V1_RESULT.md`、
`STRIDE_RESIDUALHAZARD_V1_REPORT.md`、`STRIDE_MULTIVALUE_PILOT_V1_REPORT.md`、
`SA_TRAINING_READINESS_RESULTS_ZH.md`、`SA_RANKER_ADMISSION_ZH.md`、
`FEEDBACK_EXPLORATION_RESULT_ZH.md`、`RESEARCH_DECISION_REVIEW_ZH.md`。
Temporal-anchor 原始报告保留在 `build/stride-temporal-anchor-incremental-readiness-v1/temporal_anchor_incremental_readiness_report.json`。

## 可证伪假设

在相同的SA源状态、候选、四trial标签、学习器和地图划分下，增加真实前缀的
持续冲突边年龄、最近32步集合尝试、外部约束有效性以及接受/回滚历史，
可能改善接受后一步结果的预测与候选选择。其增量不能由“换了一批数据”解释。
这不是首次方法主张；也不要求证明特征单独有效后才准训练组合模型。

## 数据合同

- 只用既有120秒高压力Dual16+SA的96条开发轨迹，不混入重叠300秒续跑。
- 8张地图，每图在决策0、16、64各选最多2个存在的状态；同组按固定seed+episode哈希排序。
- 最多48状态；不读取任务最终成功、未来TTF或候选修复结果来选源。短轨迹没有后期状态时如实记缺失，不增加门槛或重抽seed。
- 每状态最多6个去重候选：原赢家、最高分其他候选、大小4/8/16的确定性哈希候选，再哈希补齐。它是采样协议，不是新在线生成器。
- 每候选四个PP随机顺序trial，最多1152新标签；另有最多48个原动作精确重放对照。
- 完整重放原prefix，验证每步fingerprint及PP字段；在同一native内存上fork。重新生成完整候选池、原分数和原赢家必须一致。
- 每trial的PP seed和SA接受随机流独立派生，候选之间共享trial方案，不能声称不同集合的实际PP顺序相同。
- 使用来源决策的原温度，不重新升温。单PP5秒，进程fuse90秒，单根作业fuse600秒。
- 最多20个原生分支进程；根重放串行，避免20根各自再开20分支。分支期间父根不求解。
- 每根原子结果及SHA receipt；已完成根可验证resume，未完成根不自动重试。硬超时/错误停止，不能当不可行标签。
- Linux父进程死亡信号同时保护根进程和trial；根被硬终止时，trial也被内核终止。receipt必须包含完整的原动作对照、根状态和全部trial，不能仅验证清单中碰巧存在的文件。
- 如任一候选trial被预算截断，该状态只进入删失覆盖报告，不进入完整标签比较；结果因此是“候选全部完成的状态”的条件估计，不能代表全部预选状态。点估计与地图bootstrap均使用等状态权重，另列等地图差值。
- 有任一trial截断则该状态不参与主训练和排序比较，完整报告排除情况，不填零、不补采。

## 输入、标签与评分

输入复用原 `realized_dynamic` 特征，加冻结模型分数与SA的log温度/决策轮数。
历史组另加边年龄、集合重试、最近窗口的改进/边变化/回滚比例、旧外部条件已失效的次数、未知条件标记。
集合相同不等于修复条件相同；外部路径及涉及集合的当前冲突不同，则先前结果不作为精确条件经验。
历史特征必须在当前trial执行前形成，runtime和当前trial的generated/outcome不能进输入。

三个连续结果头分别预测四trial均值：

1. 接受后剩余冲突数 / 当前冲突数；
2. 当前冲突边的加权保留比例，权重 `1 + min(连续年龄,32)/32`；
3. 一步可行率。

第2头是具体转移量，不称为长尾标签，不宣称低保留必然更好。
新评分先按四trial分辨率量化预测可行率，再保留预测剩余冲突不超过同组最优值1对的候选，
在其中选预测加权保留最低者，最后按剩余冲突与candidate ID确定平局。
这个1对容差是预先固定的研究取舍，不是在看到结果后选的最佳参数；通过也不能自动上线。
额外报告同一历史模型不用保留头的选择，以识别评分取舍本身的作用。

## 训练和判断

- 同数据无历史模型 vs 历史模型，均为三个固定HistGradientBoostingRegressor结果头，不调参。
- 固定参数在JSON配置中；每状态总训练权重相同，trial先聚合而不是当独立样本。
- 八折leave-one-map-out；同图的所有任务、来源和trial保持同折。
- 对照冻结赢家、uniform精确期望、无历史模型、历史模型和历史冲突头选择。
- 报告一步Pareto命中、冲突regret、可行率、边保留、大小分布、预测误差、半组标签稳定性及5000次地图bootstrap。
- 这些是旧开发图上的探索性结果，不能作为独立跨地图确认；不根据某一特征恒定就静默删除列。
- 改善、不确定或退化均训练收尾、保存，不要求零案例损失，不事后调参数寻找通过。
- 无论结果如何，本轮都不替换控制器。若有增量信号，下一项另立未见地图确认与有预算闭环；若无信号，先报告究竟是标签/预测/评分哪个环节，而非自动再训更大模型。

## 运行与恢复

`python scripts/train_sa_history_selector.py prepare` 创建冻结清单。
WSL指定原native后运行 `collect --limit 1` 做第一根完整性检查，再 `collect --resume`。
Windows现有scikit-learn 1.5.0运行 `train`；最多8个单线程折进程，不安装依赖。
输出均在 `build/sa-history-selector-pilot-v1`。无研究结果进入旧collection，也不改旧报告。
