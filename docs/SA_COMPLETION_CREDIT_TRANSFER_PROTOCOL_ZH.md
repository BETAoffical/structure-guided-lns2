# 同一A父策略的跨批次终局信用审计

2026-09-30。只读A2的96条与A-wide的192条Train轨迹，不新增求解、训练、更新或TTF。
两批均由同一冻结A采样；它们是既有开发证据，不是未查看的确认实验。

## 历史核对与本次新增问题

- [首次梯度审计](SA_ONPOLICY_GRADIENT_RESULT_ZH.md)已检查零输出头、三条件抵消、replica两半及阶段饱和。本次不重做这些实验。
- [状态基线](SA_ONPOLICY_CROSSFIT_RESULT_ZH.md)已尝试交叉拟合Ridge降方差及真实闭环。本次不再训练基线。
- [bootstrap选择器](SA_BOOTSTRAP_CLOSED_LOOP_RESULT_ZH.md)是在H32标签上拟合GBDT成员，不是当前终局梯度的跨批次方向转移。
- A2与A-wide从同一个非零输出A出发；扩大地图后方向余弦约-0.875，但尚未分离逐地图贡献。

新增问题仅是：两批从同一点估计的完整终局梯度能否支持对方的参数方向，反对是否只由单图主导。
不同地图、任务和随机流同时变化，不把它称为单独“地图数量”的因果实验。

## 冻结分析

1. 登记两批Train审计、result/trace SHA、A/A2/A-wide及旧native和正式TTF报告SHA；只读保留。
2. 重放所有288条轨迹的行为概率、draw与终局信用，保持地图等权、leave-one-replica-out及整条log概率求和。
3. 用通用非零两层tanh导数计算各episode参数score；每个有信用地图取首个非零episode与Torch autograd核对。
4. 合成梯度必须复现原梯度范数及实际保存的参数更新，不能把零输出头专用公式用于当前A。
5. 报告map/condition梯度范数份额、抵消、留单图方向、两批更新交叉方向导数及既有参数的经验log概率代理变化。
6. 5000次地图重采样描述方向敏感性；零梯度抽样单独记未知，不补为反向；不重做replica两半实验。
7. 不读取Development/Validation/TTF标签选参数；旧正式结果只核SHA，不用其数字训练。

方向导数为 `-loss_gradient(batch)·unit_update(other)`；正值表示另一更新在该批经验终局目标的局部方向有支持。
经验代理为终局信用乘已选动作log概率差；不是离策略完成率估计，不是候选真实价值，也不预言实际TTF。
梯度范数份额是参数空间贡献描述，不是独立样本量；bootstrap负比例不是训练失败概率。

## 执行与解释

`scripts/audit_sa_completion_credit_transfer.py prepare|analyze|verify`；分析可`--resume`，输出独立`build/sa-completion-credit-transfer-v1`。
最多20个NumPy读轨迹进程，Torch核验单进程单线程；逐episode原子保存与整组运行锁，不调用求解器。

- 核验不符：先修诊断，不据错误结果改正式模型。
- 跨批次反向且遍及多图：限制是经验训练方向不具有稳定批次支持，不仅是某张图偶然支配；仍不能因果锁定表示或方差。
- 少数图主导：公开样本/贡献集中，不通过删除这些图、梯度裁剪或调步长事后造正结果。
- 方向支持但闭环失败：不能只凭本分析宣布训练可用；现有失败结论不变。

本轮没有晋级门槛，不产生新模型。A仍默认保留；进一步训练须独立设计并解决本次具体信息缺口，不能盲目续训、扩采或重试旧冷却日程。
