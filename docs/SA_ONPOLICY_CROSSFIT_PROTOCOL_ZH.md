# 状态基线与有界策略更新原型

2026-09-22。承接[梯度诊断](SA_ONPOLICY_GRADIENT_RESULT_ZH.md)。本轮实现并拟合两个actor-0分叉，
只用已审计96条Train轨迹；不新求解、不读留出标签、不测TTF、不替换Dual16/SA正式控制器。

## 数据与训练合同

- 旧actor-1及其结果保留。两个新arm都从actor-0独立计算一次梯度，actor-0正是96条轨迹的采集策略。
  这是对同一on-policy批次的受控训练消融，不是拿旧轨迹对actor-1继续训练，也不是独立效果确认。
- 完整轨迹终局可行得1，固定工作预算未完成得0；不加一步下降、便宜动作或剩余轮数目标。
- actor的129维输入、归一化、32隐藏单元、先验探索、PP/SIPPS、SA和候选池全部不改。
- 只增加用于训练的状态回报基线，输入限`state.*`、`sa.*`、`budget.*`。
  每个候选对应的这些值必须完全一致。禁止选中动作、proposal来源、候选局部特征、修复结果进入基线。
- 四折按replica ID留一；每折72条训练、24条预测，整条episode及其回报必须隔离。
  标准化只拟合该折训练部分。它不是留地图评价，更不能称独立跨地图泛化。
- 固定sklearn 1.5.0 StandardScaler和Ridge(alpha=1, solver=svd)，预测裁剪到[0,1]，不调参。
  基线拟合时每episode总权重相同且保留地图/条件等权，状态权重除以该episode状态数，折内再归一到1。
  actor梯度仍对整条轨迹求和，不按轨迹长度除权；两个权重用途不得混淆。

## 两个固定分叉

1. `bounded_condition`：原同条件其余replica回报基线，只改变一步更新尺度。
2. `bounded_state`：交叉拟合状态基线，同样的更新尺度约束。

两者都只用PyTorch autograd计算一次完整批梯度，不追加actor epochs。
原基线梯度范数必须复现0.34801956564081743，否则停止。
沿固定梯度方向从参数L2距离0.25开始，最多12次减半；每次从actor-0重新构造候选步，不连续累积更新。
仅按以下预先固定的动作分布限制选第一个合格步，不按回报、loss、成功数挑步长：

- `KL(old || new)`的episode等权状态均值不超过0.002；
- episode等权的轨迹KL总和均值不超过0.1；
- 已见Train状态的最大KL不超过0.02。

这些是经验输入上的变化限制，不是未见状态安全保证，也不是成功率单调提高保证。
不设置必须改变多少动作的最低门槛。若无合格步，保存诊断并停止，不加预算或放宽限制。

## 分析边界

报告四折回报预测Brier、同条件episode梯度离散度代理、原梯度复现、新旧概率/相同draw选择变化、
实际步长、KL限制和Torch/NumPy跨平台一致性。离散度代理受交叉拟合共享训练数据影响，不叫无偏方差估计。
Brier更好也不保证梯度方差更低；动作改变更多不等于修复效果更好。
两个arm无论局部数字高低都保留，不依据当前Train诊断宣布晋级；闭环效果必须另立有界配对实验。

概念上借鉴价值函数降低策略梯度方差及限制策略变化的思路：
[GAE原论文摘要](https://arxiv.org/abs/1506.02438)、[TRPO原论文摘要](https://arxiv.org/abs/1502.05477)。
本轮复核的是摘要与既有合同，不是重新完整复现两篇论文。这里不使用GAE、TD bootstrap、自然梯度或共轭梯度，
所以不能称TRPO/PPO/GAE复现，也不能继承它们的理论保证。

## 执行与安全

`scripts/run_sa_crossfit_update.py prepare|extract|baseline|train|parity`。
准备前本地commit/bundle；`extract`最多20个轻量NumPy进程，训练阶段按依赖串行执行。
基线用既有全局Windows sklearn；actor用既有`build/venv-graph` PyTorch；WSL只验证portable推理及native身份。
不安装依赖、不修改原结果，输出在`build/sa-onpolicy-crossfit-v1`，禁止覆盖或静默重试部分产物。
运行锁沿用恢复入口的遇锁拒绝，不使用Windows进程存活信号探测。
完成后检查原模型/数据SHA，专项测试、工作区及备份；不重跑正式计时、全量CTest或历史审计。
