# Raw终局选择器：冻结模型的独立同族地图确认

2026-09-25，分支`codex/sa-raw-independent-confirmation`。本协议在生成新任务和读取新结果前冻结。

## 问题与边界

上一轮父策略37/48、新策略39/48，但区间跨零，且只在已见Train地图上换随机流。
本轮只回答：冻结更新的收益能否延续到新地图，以及相对Dual16+SA和Official+SA如何。
不训练、不调参、不改变候选/PP/SIPPS/SA，不执行正式TTF，不读取旧Validation/Test/OOD标签。
新数据为本轮独立确认用途，禁止用于拟合、挑模型或重抽任务。

## 固定设计

- 新master seed为2026092601，确定性生成六张station_centric仓库地图。
- 复用原数据生成器及48×72模板；每图bottleneck_d20/d25，agent数按可通行格确定，不能笼统写成400/500辆。
- 两个solver seed 251/257，每条件两个replica，24条件、48配对。
- 四臂共192个episode：冻结raw-0父策略、冻结raw-1、新协议下的Dual16+SA锚点选择和Official+SA。
- 新地图/任务seed及几何SHA对本地保留的历史manifest检查；缺失历史地图明确列出，不声称验证了不可用文件。
- 所有任务保留，零冲突直接计成功；不删除困难任务或按结果补图。
- 准入只要求24/24 reset有效，至少12个非零条件、四张有效地图。不足时报告证据不足，不改seed或阈值。

冻结模型来自`build/sa-raw-residual-runtime-v1/models/raw-0.json`和`raw-1.json`。
raw-1文件SHA为`fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84`。
模型归一化、输入列和原生SHA全部随registration锁定。历史实现不修改，只新增独立入口。

## 执行与比较语义

- 复用已审计的`sa_uncapped_runtime`：每episode修复generated nodes预算2500万，无决策次数上限。
- 节点预算在完整PP动作边界检查；原子超出预算与历史协议一致，完整报告实际节点。
- 单PP20秒、episode900秒、进程960秒仅为安全界限，触发时记未知并停止整批解释，不改写成失败。
- 最多20个独立进程、BLAS单线程。采集器分批调度；`stop`等待当前批完成，不终止正在执行的PP。
- 正常节点耗尽是有定义的失败，异常/部分输出不能自动重试；修复实现时必须另立版本。
- 三个显式候选策略共享按phase/pair/replica/decision生成的选择、PP和SA随机流。
- Official保持`mode=official`原始邻域/PP随机调用，配对相同reset seed及外部SA接受uniform，不强行重置其PP RNG。
- Dual16使用原SingleFullCheckPool和原锚点GBDT，不添加raw概率修正。其PP seed采用本诊断公共随机流，不声称逐轨迹等同历史正式TTF调度器。
- 四臂每个配对的初始完整fingerprint必须相同；每条轨迹重新审计路径、特征、概率、抽样、SA接受、节点、终止和质量。
- 同一任务四臂不被当作独立样本；地图级bootstrap 5000次，固定seed2026092602。

## 预先固定的分析

首要指标是全48配对分母的成功数、新增成功/丢失成功及成功率差区间。
分别比较raw-1对raw-0、Dual16+SA、Official+SA；六图和两个密度均报告，不能只展示赢家。
共同成功子集另外报告generated nodes、修复次数、SOC、makespan、等待步数，失败不能填零。
墙钟仅作安全诊断，不能从并行数据宣称真实TTF收益。

不要求逐任务或逐图全胜，也不在事后更改晋级阈值：

- 区间下界严格大于0：该单项比较有正向统计支持；三项比较不作联合显著性主张。
- 点估计为正但区间跨零：正向但不确定，不自动晋级或扩大结论。
- 区间上界小于0：负向统计信号；其余情况如实报告无正向成功率信号。
- 任一结果都不自动替换正式控制器。只有与现有同机制基线的优势得到支持，才另行考虑串行TTF。
- 若只胜父策略不胜Dual16+SA，说明学会了改善父模型，不等于改进了项目现有方法。
- 若独立数据无收益，冻结本轮负面结论，不继续用本批确认数据重训。

## 接口与验证

`python scripts/confirm_sa_raw_independent.py prepare|verify|generate|qualify|dry-run|collect|audit|report|stop`

`qualify`与`collect`支持显式`--resume`；输出独立保存在忽略的`build/sa-raw-independent-confirmation-v1`。
先测试四臂身份、随机流、无步数上限、数据配对、失败/缺失拒绝和汇总；再提交本地预登记及Git bundle备份。
复用上一轮跨平台模型一致性证明，模型字节不变。使用冻结WSL native，不装软件。
准入后dry-run打印192个作业及预算。十批960秒fuse理想累计上限160分钟，不包含准备/审计或异常处理；不是预计TTF。
结束保留全部轨迹、增益和损失案例、SHA及中文结论，并核对无残留工作进程。
