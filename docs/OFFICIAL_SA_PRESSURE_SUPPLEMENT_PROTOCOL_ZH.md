# 高压力 Official+SA 补充对照协议

## 边界

仅新增 Official+SA 的96次首解运行，复用历史 Dual16+SA 的96次记录。不训练、不改邻域规则、不加重启、不运行第二阶段、不重编译 native。
当前用户只授权准备、reset准入和短功能smoke；正式计时需再次明确授权。`prepare/validate/smoke/ready` 均不自动调用 `collect`。

- 数据：8张仓库地图、48个任务、seeds 61/62，密度15%/20%/25%，balanced/bottleneck_eligible。
- 每次120秒，修复次数无限制，进程fuse 240秒；单worker串行。
- 冻结 native：`build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so`，SHA `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。
- 官方Adaptive邻域，大小8，PP+SIPPS；完整PP尝试、退火接受，温度1000、每决策乘0.99。
- `official_sa` 不构建模型、不生成Dual16候选。接受随机数规则复用旧SA runner；官方邻域与PP随机流保持官方模式，不冒充两种选择器具有相同规划顺序。
- 重启关闭。此组合不是完整CPLNS复现。

## 证据与准入

旧登记fingerprint为 `bece1ef3f35cef210efcf0d99407061cfe5343b82e3473fc1c97acd7dc6a7181`。
旧正式报告SHA为 `16a16a350e1ffa61e384ab5ba6a6f24f896c5e73d4382a34905c5226238e611a`。
核验旧登记、manifest、96个初始化锚点及96条Dual16+SA的原始结果/轨迹SHA。
后来7个C++源码文件有变动，不用于构建本轮二进制；当前runner仅扩展Official+SA分支，旧三个分支保留。所有已审阅源码差异逐项登记，拒绝其他未审阅旧输入变化。

准备时静态复核48任务；20进程运行96次reset，必须与旧路径和fingerprint完全一致。这里只检查正确性，不比较初始化时间。
随后在固定首个15%与25%任务各做一次2秒功能smoke，验证SA分支、路径合法性、初始身份、轨迹重建与正常预算退出。数据单独标记，不进入性能报告。
正式collect再次核验登记/输入/旧证据/准入/smoke。代码变化使旧准入失效，不能静默覆盖登记。

## 时间和分析

与旧实验一致：首次可行TTF从reset前到原生返回首解；交付时间包含路径检查、写盘和收尾，冷启动另列。
失败TTF保留空值；原始TTF只在双方共同成功条件比较，同时完整报告96条件的成功率和失败列表。报告120秒封顶指标时明确其失败记账语义。
报告密度/地图分层、修复次数、PP节点、makespan、SOC和理想离散执行完成时间；不混入第二阶段。
所有比较标记 `cross_batch_exploratory_not_concurrent_timing`，不自动晋级、不把跨批次时间差称为同期因果加速。

## 暂停、续跑与错误

整个批次使用运行锁；逐次原子保存manifest和状态。`stop`写停止请求，只在当前episode完整结束并审计后停止，不截断当前求解。
`resume --authorize-timing --clear-stop` 才显式清除停止请求。正常120秒无解保留并继续；外部fuse、身份不符、非法路径或未解释错误停止全组。
有目录但未提交manifest的episode拒绝自动重跑，必须先审计中断。完成项续跑前校验SHA及结果身份。

所有命令在仓库根目录的WSL运行，先设置 `PYTHONPATH=build/linux/sa-wall-clock-v1:.`，使用现有 `/usr/bin/python3 -B`。

```bash
python3 -B scripts/prepare_official_sa_pressure.py prepare
python3 -B scripts/prepare_official_sa_pressure.py validate --workers 20
python3 -B scripts/prepare_official_sa_pressure.py smoke
python3 -B scripts/prepare_official_sa_pressure.py ready
# 以下只能在另行获得计时授权后运行：
python3 -B scripts/prepare_official_sa_pressure.py collect --authorize-timing
python3 -B scripts/prepare_official_sa_pressure.py stop
python3 -B scripts/prepare_official_sa_pressure.py resume --authorize-timing --clear-stop
python3 -B scripts/prepare_official_sa_pressure.py analyze
```

输出 `build/official-sa-pressure-supplement-v1`，不覆盖历史数据。恢复标签 `pre-official-sa-pressure-preparation-20260916`。
