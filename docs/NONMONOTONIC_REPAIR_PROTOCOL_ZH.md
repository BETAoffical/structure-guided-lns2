# InitLNS 非单调接受：有界开发机制验证

本轮修改接受规则，不是官方 bug 修复，不是完整 CPLNS 复现，也不推广冻结控制器。
恢复标签 `pre-nonmonotonic-repair-20260911`；工作分支 `codex/nonmonotonic-repair-pilot`。
不覆盖冻结 native、模型、正式结果；不启动串行 TTF，不训练，不重新选图。

## 对照与隔离

- standard：当前官方兼容 PP，冲突超限提前中止。
- complete_greedy：取消超限提前中止，完整规划后只接受冲突不增加。
- annealed：与 complete_greedy 相同规划，完整规划后按 exp(-delta/T) 接受冲突增加。
- 超时/未完成必须回滚；T=0 时退化为完整规划的贪心接受。
- 接受随机数由独立 Python Random 实例产生，不消耗 PP 的 C RNG；三臂同状态同决策使用相同 PP seed。
- 原 `step`、`step_with_time_limit`、CLI、V2/Dual16 默认逻辑不变。实验走单独的 `step_experimental_pp`。
- 不照搬作者代码：其关闭 SA 的提前中止为 >=，与我们的 > 不同。此次保留我们的官方对照。

## 冻结范围

复用 `build/feedback-exploration-diagnostics-v1/plan.json` 中全部5个 discovery 状态，
不读取原 validation 状态进行选择，不补样本。它们是已查看开发案例，不是独立确认。
模型/候选沿用原来源 Dual16；每一步重新生成、评分，不修改候选池和冻结模型。
初始路径、完整前缀、首次候选 ID/成员/分数必须复现。新 native 另存并登记 SHA。

每状态4个trial、3臂，共60个作业。H=64；单PP上限5秒；来源重放后最多120秒；
单作业硬上限180秒（包含prefix重放）；整轮最多1800秒，20个独立进程。
安全停止在当前批次作业完成后执行；独立结果原子保存，resume验证配置/源码/模型/native身份。
理论新增修复上限3840次，另记prefix重放次数；并行耗时不是TTF。

温度只取论文设定1000、降温0.99；从来源状态iteration继续衰减，不在晚期状态重新升温。
这是来源轨迹切入的机制探针，既不复现从头运行的CPLNS，也不拟合最优温度。
不重启、不改SIPPS权重、不扩大数据或增加参数档位。

## 门槛与错误处理

- 所有路径合法、冲突图/成本与路径一致，外部路径不变、回滚完整；非法动作、前缀不一致立即停止。
- 资源删失单列为未知，不能作为无解证据；有删失则结果为资源不足，不进入下一阶段。
- annealed必须分别相对standard和complete_greedy新增至少2个成功配对、覆盖至少2个状态，
  无任何配对成功损失，总generated不超过各对照的1.25倍，才得到开发性信号。
- 同时报告最终冲突、固定H冲突AUC、接受增加的次数和恢复、搜索开销；不能把离开停滞当作成功。
- 未通过仅否定这组冻结控制器/温度/预算/来源状态组合，不证明所有非单调方法无效。
- 无论结果如何不自动追加温度、重启、trial、地图或正式计时。先检查失败来自接受后的轨迹、
  模型状态分布变化还是资源预算。不得用旧单调标签直接宣称排序器已适应新接受规则。

## 论文依据与复现

CPLNS, DOI 10.1109/TPDS.2024.3408030，算法5及表II。
作者代码审阅提交 e3fbcc82bf6a1f2cbab0ceb166d547865ca8210a；无直接复制作者实现。
论文LNS2-SA包含退火和重启，不能把其全部增益归因于本实验。
本地输出 `build/nonmonotonic-repair-pilot-v1`，正式开始前提交协议、实现和测试。

```bash
python scripts/diagnose_nonmonotonic_repair.py prepare
python scripts/diagnose_nonmonotonic_repair.py dry-run
python scripts/diagnose_nonmonotonic_repair.py smoke
python scripts/diagnose_nonmonotonic_repair.py collect
python scripts/diagnose_nonmonotonic_repair.py analyze
```
