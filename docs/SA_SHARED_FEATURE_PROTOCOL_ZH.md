# Raw 选择器共享特征：动作等价工程核验

日期：2026-09-26。冻结起点为 `f2217a6`；恢复分支为
`codex/backup-sa-shared-features-20260926`，完整 Git bundle 位于忽略的 build。

## 唯一问题

已完成的历史字段剪除不重复。当前 Dual16 的原生状态分析和候选特征已经计算，
但 raw actor 再用独立引擎扫描相同状态并提取相同特征。
本轮仅核验能否共享本次决策的原生特征；不改变模型、候选、排序、PP、SA 或随机流。

## 实现边界

- 新增显式 SharedFeaturePolicy，旧 FastPolicy、默认控制器和计时 CLI 均不修改。
- 一次提取完整124列 realized_dynamic；原 GBDT 接收原有列投影，actor读取同批完整列。
- 保留候选顺序、GBDT分数、全部129维actor输入、概率、固定draw选择和PP随机seed。
- 缓存限定为当前state对象及fingerprint，核对候选ID、成员、来源。拒绝过期、缺失或修改的候选。
- 保留当前proposal状态检查和全部外层完整性校验，不修改native。

## 固定核验

来源为已冻结 `build/sa-raw-serial-ttf-v1`，report SHA
`3995070a223e787817ef4cc9cf72f95ab60f5c2a392b9794e6857474e9f26f69`。
不续跑旧源码绑定登记；新审计登记当前源码及只读历史artifact身份。

1. 先测原生微型特征的dict/dense、全列/投影严格一致；过期和错误输入必须拒绝。
2. 重放96条父/新actor全部轨迹，重新提取GBDT特征和分数，逐项比较候选、129维输入、概率与动作。
3. 固定新actor的solver251/replica0共12案例，取首/中/末状态，共36状态。
4. 在上述状态做cProfile定位，再交替顺序各3次串行组件基准。包含状态分析、GBDT评分及actor选择，
   使用已保存proposal，故不含真实候选生成、PP、完整iteration或TTF。不得冒充端到端提升。
5. 12案例各重放最多前三步真实proposal/PP，共36次修复，核对路径与低层计数。

非计时重放最多20独立进程，组件计时单进程，无训练、无新标签和正式计时。
入口 `python scripts/audit_sa_shared_features.py prepare|profile|verify|benchmark|native|report`。
输出 `build/sa-shared-features-v1`；逐作业原子结果、输入fingerprint、独立运行锁及失败保留复用现有设施。

## 解释与停止

任何科学字段不一致即停止，不扩大浮点容差、不回退或改模型。
通过等价核验且组件时间下降才保留可选实现；没有收益则记录负面，不继续微调。
本轮结束不自动进入TTF、训练或替换默认模型。历史成功数和报告SHA不改变。
