# 后期目标入口证书与定向补车诊断

本轮只使用已有开发状态 `0decb088a8e0529a3c77c2bf`，不运行正式 TTF、不训练、不修改正式控制器。

## 依据

在原 16 车集合之外的路径固定、其余 14 辆选中车完全移除的宽松问题中，152 与 362 都必须在第 83 步位于 2739。证书通过精确前向可达集和后向目标可达集的交集验证，不使用成功见证或人为到达截止时间。最终静态目标连通分量为 2738、2739；边界终点占用者为 72、300、319、326。

这个证书说明固定外部路径下原集合不足以消除全部冲突，不说明四辆外部车全部必须移动，也不说明任意补入一辆就能解决完整任务。

## 固定比较

- 原 16 车；分别补入四个证书阻挡者；补入四者全集。
- 按状态哈希选择一个及四个非证书外部车辆，作为同样大小的对照。
- 共 8 个候选；每候选 1 次宽松两车 A*+CBS 参考搜索，以及 4 个配对原生 PP/SA trial。
- 参考搜索上限 45 秒、250000 次低层展开；PP 单次 5 秒；独立进程 fuse 90 秒；最多 20 个进程。
- 原生仍使用已冻结 `5b1b2af1...925d` 模块、原状态温度、同 trial 的共同随机种子和接受随机数。
- 参考可行性仅表示宽松问题存在路径；原生结果必须通过路径、集合外路径不变和退火接受规则校验。

## 结果边界

分别报告完整可行、冲突下降、目标 pair 是否消失、节点数及预算截断。任何超时为未知，不能记为不可行。全部加入不优于单车时保留负面结果，不事后增加候选或 trials。

这是与 t=1 强迫出口不同的第二种开发机制。只完成本轮有界比较；局部下降不自动晋级控制器，不自动启动计时或重训。

```text
python scripts/diagnose_sa_goal_slot.py
python scripts/probe_sa_certificate_augmentation.py prepare --output build/sa-certificate-augmentation-goal-v1 --certificate build/sa-goal-obstruction-v1/report.json
python scripts/probe_sa_certificate_augmentation.py reference --output build/sa-certificate-augmentation-goal-v1
python scripts/probe_sa_certificate_augmentation.py native --output build/sa-certificate-augmentation-goal-v1
```
