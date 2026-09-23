# 原任务评价前的课程模型登记

本文件在72条原任务比较产生结果之前登记。训练协议提交为`053ef94`，没有改动原actor-1或原actor-2。
新模型是actor-1的另一条iteration=2分支，不是从旧actor-2继续更新，也不替换正式控制器。

## 身份

| 项目 | 值 |
|---|---|
| 输出 | build/sa-curriculum-update-v1/models/actor-2.json |
| prototype_arm | curriculum_condition |
| 训练登记绑定 | d2ed155d9107e116d8fefe8a50d25097bea4bb1e9b71d12bea7f329ac1b63bc5 |
| 模型文件SHA256 | c5684bc2d93bd431668bff6b299e8f1ea766d1bcb1720cb99fa4a7bab218e0f0 |
| policy SHA256 | eccf5a94cec79e706b81164d0d1e0d3975fe37a71152408797e31f376c609458 |
| 父策略policy SHA256 | 129d30ed61e68db939068cd65e2a57b257a054d2e0db81539236004f2f8b51f1 |
| update.json SHA256 | 3e7f7b84f487b22d3ff346b7fca3cc46eae476e81df1644edaa585e5bb347a97 |
| training_registration.json SHA256 | 14a20c55ddab59b13fc451e376caafbe4bdaa42999d4ce6bf5c82715ba55e1f9 |

## 训练与数值检查

完整使用144条、17,771步，其中20条提供非零终局系数；所有父策略概率重放最大误差4.44e-16。
原数据梯度范数0.2857805939，等于历史actor-2梯度按2/3重加权；新增过渡数据梯度范数0.2108693294。
两部分梯度余弦0.6184691；只作解释，没有按该数值挑选配方。
四个参数张量的最大梯度分量分别做完整目标有限差分，最大绝对误差1.43e-8。

只更新一次。步幅0.25因平均轨迹KL=0.10314486超过0.1而二分至0.125。
最终平均状态KL=0.00037131、平均轨迹KL=0.03649661、最大状态KL=0.00272943，均在原限制内。
原状态相同随机数下644/17,771次选择改变，约3.624%；不是新增成功数。

Windows Torch、Windows NumPy、WSL NumPy在505个状态、8,080个抽样选择上完全一致，
概率最大误差4.44e-16。旧PP、SA、冻结native、候选池和特征归一化未改。

## 接下来允许什么

只允许[已登记协议](SA_CURRICULUM_UPDATE_PROTOCOL_ZH.md)中的72条未抽减原任务三模型比较。
这仍是Train地图和已有初始化的新随机流开发诊断，无独立泛化或TTF主张。
新数据只有一个新增混合条件，模型可能无法将子任务经验迁回完整任务；数值检查通过不代表科学假设通过。
禁止根据评价结果追加更新、挑选更好副本、删除失败或使用评价标签训练。
