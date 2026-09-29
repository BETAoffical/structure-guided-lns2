# 冻结A的一次on-policy终局成功续训

## 问题

接受器消融支持保留A+SA。A由raw-1的一批轨迹进行一次终局成功更新得到，
尚未检查用A自身的新轨迹做同目标下一次更新是否仍有增益。
本轮只检验这一点，不重新扫描奖励、温度、容量或候选，不使用一步下降、剩余轮数、H4标签。
不是为了反复更新直到显著，也不要求每张图全胜。

## 冻结范围

- A文件SHA为`81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062`，iteration=2。
- 使用原terminal-efficiency登记的六张训练图、12任务、两solver seed、四replica，共96条新A轨迹。
- 比较使用该登记的另六张旧开发图、12任务、两seed、两replica；冻结A和A2共96条。
- 训练和比较的map/task ID、地图/场景内容SHA必须隔离。旧开发图不是独立确认。
- 新phase `completion-continuation-train-20260930`及`completion-continuation-eval-20260930`形成独立流；
  比较两模型共享选择draw、PP seed和接受draw的派生键。
- 不读取扩样72条件或接受器24条件的计时结果进行训练、筛选任务或选择超参数。
- 两臂保持129维特征、32单元隐藏层、归一化、候选、PP+SIPPS、SA日程和native不变。

## 更新与资源

- 复用已验证的完整轨迹信用：R=预算内最终成功，按地图等权，leave-one-replica-out基线。
- 仅一次更新，原KL/回溯和步幅保持不变；全失败或全成功条件不凭空生成优劣标签。
- 没有非零梯度或可接受步长时停止，不增加trials，不将失败末态冲突下降转成奖励。
- 20个独立进程；每episode2500万低层节点，无决策次数上限。
- 单PP20秒、episode900秒、process960秒仅为保护；保护中断为未知，停止训练准入，不填失败。
- 并行秒数不是TTF。采集和完整轨迹审计均原子保存，支持resume和完成当前批次后安全停止。
- 理论最多192条采集；正式比较前验证Torch/Windows NumPy/WSL NumPy概率和选择一致。

## 比较与停止规则

完整审计后报告全部48个配对位置：成功、新增/损失、共同成功节点、修复次数、SOC、makespan，
两负载、逐地图和5000次地图级bootstrap。失败不填零成本。完整保留负面结果。

沿用旧开发信号条件：成功数不降低，且净成功增加，或共同成功节点减少至少5%。
这只是有界下一步的信号，不是统计显著、真正TTF加速或自动晋级标准。
若通过，再单独冻结模型设计TTF验证；若未通过，保留A，不自动追加第三次更新或换奖励。
即使出现收益，也不能宣称解决之前扩样批次的三个失败或获得跨布局泛化。

实现只新增独立入口，通过私有函数绑定复用旧采集、审计、梯度及parity实现，不修改其全局变量。
数据写入`build/sa-completion-continuation-v1`；训练前提交代码、配置、测试和协议，先保存备份标签。
入口：`scripts/run_sa_completion_continuation.py prepare|verify|collect|audit|train|parity|compare|report|stop`。
