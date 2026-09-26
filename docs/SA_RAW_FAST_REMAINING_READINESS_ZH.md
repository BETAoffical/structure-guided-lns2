# 快路径剩余条件：计时前准入记录

日期：2026-09-26。状态：准备完成，尚未启动新的正式计时。

以上为计时前快照。后续72条计时及审计已完成，见[正式结果](SA_RAW_FAST_REMAINING_RESULT_ZH.md)。

## 已完成

- 冻结36个剩余条件，快路径raw-1与Dual16+SA各36条，共72条；不重复上一批12个条件。
- 新排程逐条件配对，先后顺序各18组；模型、任务、随机流、PP/SIPPS和SA不变。
- Windows新增测试10项通过；WSL八组定向测试71项通过、8项因缺少PyTorch跳过。
- 36个快路径条件各执行3步原生重放，共108次修复；候选、特征、模型选择、路径和SA流一致。
- 核验36条旧Dual16三步证明，明确标记为复用，而不是本次原生执行。
- 全部72条准入记录通过，1,670项冻结输入SHA核验通过；无失败、超时或残留运行锁。
- 安全暂停、显式续跑、拒绝不完整episode自动重试、失败不填零时间均有测试覆盖。
- 本轮未重跑整个仓库测试或CTest；没有修改native、正式模型或历史结果。

## 身份与备份

| 项目 | SHA / 标识 |
|---|---|
| 预注册提交 | `100092a5b4120761371405189b50a61a304b7f25` |
| 运行binding | `87440fd65bb6d319239b2e0ade1f5a49020982ba392c362c47cd28069e4bfea6` |
| registration.json | `bee2bc70ffd86d4bb46ef771b441f01db9906430f74232988398b80392231107` |
| preflight.complete.json | `d869d4250dac94e34c7b429d2f746677cabe4bbf2bae9d708aa41278e6fc87fc` |
| 准入ZIP | `019ee4d15c0dc56cf979f5efbd06f86c783beb8921353b9d7c58b32b0f5c1279` |

输出：`build/sa-raw-fast-remaining-ttf-v1`。机器可读核验记录为`readiness.json`。

Git完整历史备份：`build/sa-raw-fast-remaining-preregister-20260926.bundle`，已验证。
准入数据备份：`build/sa-raw-fast-remaining-ready-20260926.zip`，76个文件逐项SHA一致。
本轮仅本地提交与备份，未推送远端。

## 后续启动

确认计时范围后，使用现有WSL和冻结native调用：

```bash
env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=build/linux/sa-wall-clock-v1 \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /usr/bin/python3 -B scripts/run_sa_raw_fast_remaining.py all
```

在当前episode完成后暂停：同一入口的`stop`阶段。
恢复采集：`resume`；完成采集后依次执行`audit`、`report`。
不要对已有采集状态再次调用`all`，也不要删除异常输出来绕过恢复检查。

计时固定单worker、每条120秒首解预算，无决策/节点次数上限。
预计45-90分钟；规划预算累计上限144分钟，进程保护上限累计288分钟，另加校验和调度。
先分别报告新增36对与原12对，再汇总48对；这仍是已查看地图上的开发证据，不构成新独立泛化结论。
