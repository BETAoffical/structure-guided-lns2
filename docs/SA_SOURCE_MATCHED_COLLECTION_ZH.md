# 同来源同大小候选：独立分支采集

本阶段接续已冻结的准备清单，不更改其16个根状态、47个候选和376条新trial。不训练、调参、接入控制器或比较TTF。

## 固定合同

- 准备plan SHA：`84e0250e278171cb36ad58d9b4dd5fde2831a4f439c58fe43ed03a6641b46f4e`。
- 每个trial独立进程、独立reset、完整显式prefix重放，根状态、历史与整个候选池必须和原轨迹一致。
- 同根同trial共用PP/SA随机数计划；新seed为202609182，不混入旧trial。成员集合不同，并不意味着最终规划顺序也相同。
- 首步强制指定候选，此后复用冻结Dual16+SA，H32包含首步，温度按绝对decision继续。
- 每个job从进程启动起最多180秒，包含reset、prefix和rollout；每次PP最多5秒。prefix触发PP安全限记为未知而非replay错误；非安全限下的指纹不一致必须停止审查。
- 不新增低层节点门槛。旧通用branch接口的节点上限设为64位最大正整数，仅为兼容，实际预算由32步和时间限制决定。
- 最多20个并行独立进程；每批结束检查安全停止请求。异常时排空当前批后停止，不换种子、不自动补跑超时、不覆盖部分写出的trial。

## 记录与分析

每trial保存started、result和SHA receipt。原子写入，续跑须验证已完成字节；中断但未封口的目录拒绝自动重跑，须先审计。

若已确认worker退出的硬超时打断原子写入，将残留临时字节保留在`interrupted_writes/`，再验证已写出的完整结果或封口为未知，不删除原记录。停止请求不依赖全部运行输入仍然匹配；每批排空后重新校验输入，漂移时不派发下一批。

实际模型、Python运行时及任务文件的SHA从旧registration继承并核验，不能用当前值覆盖历史值。`run_sa_path_quality.py`在旧registration后仅因提交`9550574`增加Official+SA分支；已检查该差异不改变Dual16+SA路径，新采集登记当前文件SHA。当前C++源码并非冻结二进制的身份，本轮不重新编译。

主分析为同来源/同实际size16的两个候选，不把旧anchor混入选优。报告8-trial胜负平、未知、全部35种4/4分区的双向交叉选优及相对两候选均匀期望的收益。分区不是独立样本；按地图等权、图内状态等权，5000次地图级bootstrap。anchor单列描述性结果，不把事后选优当作可部署策略。

数据仍来自已查看过的开发状态，不是独立新地图证据。出现信号也只允许下一步验证动作前可预测性，不能据此宣称连续控制、更高成功率或更低TTF。平局或不稳定结果不触发自动扩采/训练。

## 命令

```text
python scripts/collect_sa_source_matched.py prepare
python scripts/collect_sa_source_matched.py verify
python scripts/collect_sa_source_matched.py dry-run
python scripts/collect_sa_source_matched.py collect --smoke --limit-jobs 3
python scripts/collect_sa_source_matched.py collect
python scripts/collect_sa_source_matched.py stop
python scripts/collect_sa_source_matched.py collect --resume
python scripts/collect_sa_source_matched.py analyze
```

使用已登记的WSL冻结native，不安装依赖。`--smoke`独立目录，不作为正式证据。正式trial结果位于忽略的`build/sa-source-matched-collection-v1`。prepare后源码变更会导致verify拒绝运行；正式采集前须完成测试、提交并本地备份。

安全停止最多等待当前20个已启动job结束，不再派发下一批。resume清除明确的停止请求，但不会重试已经记录的未知结果。
