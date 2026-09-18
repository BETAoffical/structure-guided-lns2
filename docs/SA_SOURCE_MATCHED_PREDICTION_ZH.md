# 同来源候选的冻结预测审计

本轮仅做现有模型的动作前推理和开发性分析，不训练、不调参、不新增标签、不调用 reset/repair、不进行 TTF。
沿用补采后的 16 个状态、8 张地图、47 个候选、376 个 H32 trial；不移除平局或负面状态。

## 先区分模型身份

最近的 47 状态模型只封存了原候选上的留图分数，没有封存可加载权重。
新增配对中仅 6/32 个候选有历史分数，0/16 对完整；不能伪造缺失分数或临时重训。
其状态必须报告为 `not_evaluable_missing_pair_scores_and_weights`，不是验证失败或通过。

另有 `build/sa-unbalanced-coverage-v1/old_fold_models` 的八份旧权重。
它们基于更早 16 状态训练，每折排除一张完整地图，实际训练 14 个状态。
本轮以独立身份 `historical_16_state_lomo_gbdt` 进行辅助诊断，不替代最近模型。
保留原 Dual16 完整 pool 的已冻结分数为另一个描述性对照。

## 预测规则

- 主比较限制为已登记的同来源、同实际 size16 两候选。
- 历史 H32 GBDT 沿用原始对称化 pair margin，选择分数较高者；平局保留原 anchor（若在 pair 内），否则按 candidate ID。
- 原 Dual16 pool 分数沿用 12 位取整及 candidate ID 平局规则，仅限制选择范围到 pair，不重新算全池分数。
- 另报告历史 GBDT 在 pair 与 anchor 的去重并集上的排序，明确为次要描述性比较。
- 不比较多个新模型后挑赢家，不改变特征，不增加 coverage 规则，不根据标签改变选择阈值。

## 特征与身份校验

读取保存的根状态和修复前候选，使用同一个 SHA 的 native 特征提取器；不启动求解器。
127 个输入为原 `realized_dynamic`、`source.score`、SA 温度与当前 decision，不包含未来 trial 结果。
`history.*` 不在这个历史模型的 dynamic profile 中，不能称其为历史序列模型。

每个状态同时重算原四个已存候选的特征，必须与原值完全一致。
加载本地 pickle 前校验历史登记 SHA，再核验类对应源码、列顺序、参数、train IDs 和 train maps。
对存在历史旧模型预测的状态，原四候选分数及选择必须完全复现。
模型加载和推理使用现有 Windows sklearn 1.5.0；native 特征提取使用已有 WSL。
特征任务独立进程，最多 20 workers，实际最多 16；其余是小规模纯推理/统计。

## 分析与边界

先封存动作前特征与预测，再单独读新标签统计 H32 完成率。
报告相对 pair 均匀期望、相对 anchor 的差值、逐状态正负平和 5,000 次地图级 bootstrap。
地图等权、图内状态等权；八个 trial 不当成八个独立状态。
事后较优 pair 仅为经验上界，不当成可部署方法或真实总体上界。
主预测不完整或身份错误则拒绝分析，不默认补零或过滤难例。

即使历史模型表现为正，也只能说明旧模型的固定偏好与本批标签局部对齐，
不能证明最近模型有效、新地图泛化、连续控制有效或 TTF 改善。
本轮不设置晋级门槛，不由结果触发自动训练或采集。

```text
python scripts/audit_sa_source_matched_prediction.py prepare
python scripts/audit_sa_source_matched_prediction.py features
python scripts/audit_sa_source_matched_prediction.py predict
python scripts/audit_sa_source_matched_prediction.py analyze
python scripts/audit_sa_source_matched_prediction.py verify
```

代码和规则先本地提交及备份，再执行特征和预测。旧采集、模型和报告不覆盖。
