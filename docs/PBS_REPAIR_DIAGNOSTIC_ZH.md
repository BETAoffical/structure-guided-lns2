# PBS 修复器准入诊断

## 边界与冻结

只诊断现有 PBS 接口，不修改官方 PP、PBS、正式 native、模型、候选或退火接受逻辑；不训练、不运行 TTF。
本地恢复标签：`pre-pbs-diagnostic-20260914`，父提交 `642893c`。
冻结 native：`build/linux/sa-wall-clock-v1/lns2_env.cpython-310-x86_64-linux-gnu.so`，
SHA256：`5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。

## 检查对象与解释

- 历史四状态 PP/GCBS screen 的输入、原选中集合和原 action seed；不是重新筛选成功案例。
- 历史记录仅写了 PBS 的首次 restored-state smoke 崩溃，没有保存确切首次调用和堆栈；不能声称完全复现原崩溃。
- 两个微型案例：开放地图两车相向与一辆固定外部车；单行走廊两车对换。
- 每个状态分别恢复后执行 PP/PBS 一步，同 seed 重复两次。共 24 个独立进程作业，最多 20 并发。
- 每分支恢复后 solver budget 为 5 秒，外部 fuse 为 20 秒。PBS 现有预算并非单车搜索内的合作式 deadline；超时只记未知。
- 禁止 core 文件，保存标准输出、错误、退出信号、结构指纹及返回路径。
- 校验来源 SHA、恢复路径结构、显式集合、路径合法性、冲突图、外部路径不变及失败路径回滚。
- PBS 严格下降、PP 非增接受不同，不能据本次结果估计两者的同接受规则效果或因果增益。
- 任一崩溃、非法路径、回滚错误或非确定性均停止机制效果晋级。若有崩溃，可单独用隔离检查构建定位，不替换正式二进制。

入口：`python scripts/diagnose_pbs_repair.py prepare`，然后 `collect`。
重复 collect 只续用同 binding 作业，包括失败记录，不自动重试失败；输出保留在忽略的 `build/pbs-repair-admission-v1`。
微型失败检查不构成完整预算中断回滚准入，全部返回正常也不代表可部署。
