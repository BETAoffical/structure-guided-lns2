# 冻结 A/A2 独立 TTF：准入记录

预注册提交：`f87485abd2bb0cc96ea6909f6366288a66afd6ed`，已推送。备份标签 `pre-completion-frozen-confirmation-20260930` 指向 `dea3dc05bd7b6c8295bb8bd9b7c6ee9385b6aa13`，远端已核对。

## 准备结果

- 12 张新地图、24 个任务、72 个配对条件；没有换 seed、换图或按冲突筛选。
- 24 个任务全部通过原采样器，端点容量补救次数为 0。
- 72/72 初始 PP 合法完整，均非零冲突。
- 288/288 四臂 preflight 通过；初始 fingerprint 配对，proposal 不改变状态，模型身份正确；此阶段没有执行 native repair。
- Windows unittest 与 WSL pytest 各 30 项通过、1 项跳过。跳过的是显式 opt-in 的旧工程等价性复测，未重复已有加速验证。本轮没有全库测试或重新编译 native。

| 任务密度 | 条件数 | agents | 初始唯一冲突对数 | 平均冲突对数 |
|---|---:|---:|---:|---:|
| d20 | 36 | 335–416 | 38–437 | 130.2222 |
| d25 | 36 | 418–520 | 85–752 | 292.7500 |

密度是 agent 数相对可通行格子的比例，不是冲突严重度分桶。高冲突和困难实例未删除。

历史登记覆盖 2,613 个 seed、562 个地图 SHA；本轮无已知重复。直接登记的 36 张来源地图与新 12 张地图亦无 SHA 交集。旧清单另有 384 个不可访问的历史地图路径，不能对这些已缺失文件再做几何核对；隔离结论限于可核对证据。

## 冻结与备份

- Registration SHA：`09d30b2fdd80d264664111189de1f1abdb29536a9f46043919e94d0db1029cca`
- Binding：`03ec91e1ee3851c4321d2b898c889dd3b138d69b23088d61afec149f4c4526d5`
- Frozen native SHA：`5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`
- 本轮 `parent` 明确对应 A，`completion` 对应 A2，不对应历史 raw-1。
- Ready 备份：`build/sa-frozen-confirmation-ready-20260930.zip`，548 文件、12,575,485 字节；逐成员 SHA 校验通过。
- 备份 SHA：`6660a812259649b0c78206c017d84e37d0a61b400971610760126ee734682c69`
- 紧凑证据：`artifacts/sa-frozen-confirmation-ttf-v1/ready.json`。
- 启动前只读环境检查：Windows 电源方案 Turbo，C 盘可用约 54.8 GiB。不会自动检测或消除用户其他活动带来的全部扰动。

## 执行

计时最多 288 个 episode，1 worker，120 秒规划预算，无决策/节点次数上限。准备和审计不进入 TTF。当前登记尚不构成性能结果。

在 WSL 仓库中执行：

```bash
/usr/bin/python3 scripts/run_sa_frozen_confirmation_ttf.py all
```

需要暂停时执行 `stop`，当前 episode 完成后停止；用 `resume` 继续。恢复后收集完成，再执行 `audit` 与 `report`。有错误或部分 episode 时先审查，不自动覆盖重跑。最终数据保留在 `build/sa-frozen-confirmation-ttf-v1`。
