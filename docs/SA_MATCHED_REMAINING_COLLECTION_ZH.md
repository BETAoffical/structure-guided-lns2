# 同来源配对补采：双候选入口

## 冻结范围

使用 `sa-matched-remaining-preparation-v1` 的31根、62候选、496作业清单。
不重选根或候选，不增加anchor，不拟合模型，不运行正式TTF。
短prefix的432个作业最多20进程，长prefix的64个作业4进程；按既有38批顺序执行。
每作业180秒包含reset、prefix及H32 continuation，PP安全限5秒。超时是未知，不填false。

直接调用冻结 `collect_sa_source_matched.trial_worker`、失败封存、receipt及标签校验。
因此reset、每步prefix指纹、历史指纹、完整pool、成员、PP/SA随机流、温度、最终路径校验均不变。
只改采集外围排程与双候选汇总，不改变PP、SA、模型、候选生成或在线控制器。

## 原生准入

按决策步和根ID盲选最短、最长prefix两根，每根两个候选的trial0，共4个独立作业。
再在独立repeat目录重复每根的第一个候选，共2个作业。所有动作、pool、路径、搜索计数、
完成标签必须一致；只排除既有计时字段和存储binding。不读取结果挑选更容易的检查根。
6个准入作业不加入496个正式诊断标签。任何未知、错误或不一致均停止，不自动增加预算。

## 完整性和暂停

- 所有文件SHA在执行前后严格复核，批次之间复核代码、模型和native。
- 使用同一个全局采集锁，原子作业目录、结果与receipt。子进程逐作业隔离全局RNG。
- `stop` 写入标记，当前已启动批次排空后停止，不再发下一批。
- `--resume` 仅保留已封存结果并续未启动作业；部分输出拒绝自动重跑。
- 未解释错误先排空当前批次并暂停整组；被时间限制删失的已封存结果保留，不自动恢复。
- 批次缓存只避免反复解析全部旧结果，重新启动及结束仍逐结果校验，不把缓存当receipt。

## 分析

每根只有真实两个候选的8次H32完成标签。保留真实anchor身份，但若它不在pair内，
不创建其标签，不把未采集误写为8次失败或资源未知。
报告配对差异、不同结果trial数量、35个4/4划分、ID平局规则和已登记的平局中性敏感性。
后两项只表示标签中可重复机会，不是模型预测能力或TTF收益。
地图等权、图内状态等权，5,000次地图bootstrap；有删失则相应全组均值/区间为null，
不得过滤未知根后声称全组提升。没有新的晋级门槛或自动训练。
原16根与新31根应分开报告，后续联合分析也不能将采集组或trial当成独立地图。

## 命令

```bash
python scripts/collect_sa_matched_remaining.py prepare
python scripts/collect_sa_matched_remaining.py dry-run
python scripts/collect_sa_matched_remaining.py preflight
python scripts/collect_sa_matched_remaining.py collect
python scripts/collect_sa_matched_remaining.py stop
python scripts/collect_sa_matched_remaining.py collect --resume
python scripts/collect_sa_matched_remaining.py analyze
```

可用 `collect --limit-batches 1` 验证真实暂停和resume；不改变批次或随机流。
运行需WSL的兼容native路径优先，单线程BLAS，已有环境无需安装。
数据在忽略目录 `build/sa-matched-remaining-collection-v1`，旧结果原样保留。
