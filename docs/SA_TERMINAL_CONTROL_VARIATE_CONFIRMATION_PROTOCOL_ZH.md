# 冻结终局基线的新随机流确认协议

## 问题与边界

上一轮64条轨迹的读档诊断发现，score-squared基线相对LORO二阶矩下降18.60%，但M04条件退化5.28%，不能据此更新模型。本轮只问：在上一轮拟合并冻结的基线，在同四个Train条件的新独立随机流上，是否仍比同样冻结的普通均值基线降低梯度贡献二阶矩。

不训练、不改A、Dual16、候选池、PP+SIPPS或SA，不产生正式TTF，不扩地图，不恢复静态迁移主张。四个条件已被查看，不是总体随机样本。本轮是新随机流确认，不是跨地图确认。

## 预注册输入与执行

- 来源：`build/sa-completion-credit-replicability-v1`，正式报告SHA为`18466f108adfbf051bed14059c94074458510643a6ed6c83bff0fa8627fff923`。
- 上轮基线审计：`build/sa-terminal-control-variate-v1`，正式报告SHA为`6669c6eb8941f5b88011525c316449d9a62eda1ea647d3e90a2a4956c54c2bb3`。
- A模型SHA为`81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062`。冻结native SHA为`5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。
- 固定M01、M00、M02、M04四个已有条件，每条件16次，计64个episode；重新初始化相同输入，phase固定为`terminal-control-variate-confirmation-20261001`。
- 原始轨迹、模型、初始状态、输入源码和协议SHA全部登记。新流与前80条诊断流不得重合；新初始状态和模型身份必须相同。
- 20个独立进程。每episode为2500万generated节点边界，原子修复允许跨界；无决策次数上限。PP安全20秒、episode安全900秒、进程fuse960秒。超时未知或错误不能作为失败训练标签，出现后先检查，不自动重跑。
- 沿用逐episode原子输出、批次锁、完整native审计和停止当前批次后安全退出。没有必要另跑两种控制器：两种基线只对同一批64条轨迹离线计算。

## 冻结基线与指标

令完整终局回报R为预算内成功0/1，完整轨迹score为所有决策的`grad log pi`之和S。每条件仅从上一轮16条轨迹拟合：

```text
b_mean = mean(R)
b_score_squared = sum(R * ||S||^2) / sum(||S||^2)
g_i = -(R_i - b_frozen) * S_i
second_moment = mean(||g_i||^2)
```

能量全零时，score-squared退回同批普通均值。保留原浮点拟合值，不根据新数据校准、截断或改标签。每条件的两个标量在registration中冻结，在新结果出现之前登记到Git。

主比较为四条件等权二阶矩。另报每条件二阶矩、样本内贡献离散度、梯度方向对齐、最大单episode贡献占比、新成功数和预算终止原因。二阶矩不是真实梯度方差，不是规划时间或成功率改善。

冻结描述性确认门槛：总体二阶矩下降至少10%，且任何条件的二阶矩增加不得超过10%。包括新回报全相同的条件，不因其变得困难而排除。若普通均值二阶矩为零，只有另一方法也为零才通过条件保护；总体分母为零则无信号。

2000次条件内配对replica bootstrap仅报告区间，不作为晋级门槛，不能解释为地图总体推断。相同抽样索引用于两种冻结基线，不对每次bootstrap重新拟合。

## 验证与决策

- 完整64/64轨迹、64/64 native审计、完整score；缺失、未知、错误或身份不一致均停止分析。
- 测试只用旧数据拟合、全轨迹梯度、常数回报、零能量、流隔离、门槛边界、旧实现不被全局补丁影响和CLI无输入help。
- SHA检验精确；浮点重算允许`rtol=1e-10, atol=1e-12`的跨平台数值差异，但决策、身份、标签和数量不允许变化。
- 本轮通过仅支持冻结基线降噪信号重现，仍不自动更新A；需要下一轮单独批准的有界更新及端到端验证。
- 本轮失败则保留A，不追加replica、不重新拟合到本批数据或降低门槛；解释哪些条件和贡献导致失败。
- 安全标签、预注册提交、ready/collected/complete备份先后保存。旧正式报告和TTF结论不变。

## 命令与预算

```powershell
F:/install/python/python.exe scripts/confirm_sa_terminal_control_variate.py prepare
F:/install/python/python.exe scripts/confirm_sa_terminal_control_variate.py dry-run
# Ubuntu-22.04 real-user native context: collect, then audit
F:/install/python/python.exe scripts/confirm_sa_terminal_control_variate.py analyze
F:/install/python/python.exe scripts/confirm_sa_terminal_control_variate.py verify
```

入口支持`collect --resume`与`stop`。已完成结果不覆盖、不跨实现身份续跑。

上一批collection约1043秒，native审计约654秒。本次采集、审计和分析预计约30至45分钟，加上实现与测试时间；四批进程fuse预算上界为3840秒，审计另计。20-worker耗时绝不能当成单worker TTF。
