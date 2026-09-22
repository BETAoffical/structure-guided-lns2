# 无决策次数上限的训练终局补齐协议

## 目的与历史边界

上一轮取消评价中的256次决策上限后，Official+SA新增5个成功，两个学习分支均未新增成功。不能继续把被该上限截断的训练轨迹当作完整失败，也不能据此调整留出集标签或参数。本轮只补齐原训练集终局，不重新设计模型、不做正式TTF。

原训练集96条轨迹来自6张Train地图、24个条件、每条件4个独立随机流，均由同一冻结actor-0生成。其中80条已经可行或耗尽节点预算，直接引用原文件及审计；仅16条旧decision_budget轨迹重新执行。新旧完整前缀必须逐步一致，任何不一致停止训练。

## 固定边界

- `max_decisions=null`，无替代性大整数上限。原256仅保留为冻结特征的归一化参考，超过后该特征归零，不触发停止。
- 修复节点预算2500万；单次PP安全限制20秒；episode安全限制900秒；外部fuse960秒。安全中断为未知，不赋失败标签，不丢弃后继续训练。
- workers=20，只有16个新采集作业，不为填满worker重复采样。并行机制运行不作为TTF比较。
- 保留同一地图、任务、初始路径、solver seed、actor-0、候选、SA、PP及随机流；不读取已查看的80条留出评价进行训练。
- 新输出 `build/sa-uncapped-training-recovery-v1`，历史文件不覆盖。配置、实现、模型、源轨迹及缓存SHA在采集前登记。

## 是否更新的预设判断

完整审计后重算每条件leave-one-replica-out终局成功基线，各地图等权，每个episode的系数乘整条轨迹的log概率之和，不以长度再除一次。新函数与历史梯度系数实现相同，只替换终局截断语义。

仅当系数改变或新增非零系数的轨迹后缀时，允许一次固定的condition-baseline更新。若两者均未变化，记录无新梯度信息，跳过训练。若合计梯度仍为零，也跳过训练。

只用原actor-0、原129维特征和32单元tanh残差网络，沿用原固定KL更新配置：initial L2=0.25，最多12次仅缩小步幅的预设回溯，mean KL<=0.002、mean trajectory KL<=0.1、max state KL<=0.02。此回溯是优化器安全检查，不是环境决策上限，不改变采集预算。无可行步幅则停止，不追加模型或调参。不再重复已无稳定收益的state-critic分支。

80条完整轨迹复用旧NPZ，但全部换用补齐后重新计算的系数；只为16条新轨迹提取缓存。Torch、NumPy及加载冻结native的WSL运行环境验证候选概率和相同随机数下的选择一致，并包含256次之后的状态。

## 命令与产物

`scripts/recover_sa_uncapped_training.py` 提供 `prepare / verify / collect / audit / analyze / extract / train / parity / stop`。

先提交本协议与实现，再prepare登记；collect与audit使用WSL，train和`parity --torch`使用现有Windows Torch环境，普通parity使用WSL。collect必须显式`--resume`续跑，未知、部分文件或错误不自动重试。stop等待当前作业批次完成。

输出源引用、新轨迹、collection清单、前缀及native审计、terminal_analysis、缓存、一次更新记录和模型校验。补齐标签、模型导出均不代表成功率或长尾已经改善。本轮不进入留出评价，不更换正式控制器；新模型如产生，下一轮另行冻结配对验证。
