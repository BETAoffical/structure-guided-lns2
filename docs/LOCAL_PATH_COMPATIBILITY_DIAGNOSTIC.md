# 局部路径兼容性诊断

本入口只用于开发性机制诊断，不改变 Official Adaptive、V2、Dual16、模型、候选生成或 PP+SIPPS。
输入是六个已查看的状态，不是新地图确认集。十对车辆也不是十个独立地图样本。

## 搜索与预算

- 从每个状态按冲突事件数选最多两对车辆，平局按 agent ID。
- 每对运行 ordinary、random_paths、directed_paths；每个作业包含两种顺序，共享180秒和100万次低层展开。
- 后两者先运行普通顺序。只有失败时才重选，最多四次；首车成本不得超过对应普通路径。
- 随机重选只改变同成本首车路径的 BFS 平局顺序，不修改原生随机比较器。
- 定向约束来自后车忽略首车后的参考路径与首车路径的最早四个冲突。每个分支使用一个顶点或交换边约束，不累积不相关约束。
- 三种前置方法均未解决的车辆对才进入独立两车 CBS，同样共享180秒/100万次展开。
- 底层使用固定外部路径的硬约束，以及外部路径结束后的完整静态尾部。目标到达意味着之后一直占据目标，不是第一次经过目标。
- CBS 的高层失败可证明固定外部条件下无解；顺序方法的 not_found 不可以。任一资源上限导致 unknown。
- 每个子进程还有15秒启动收尾 fuse；它不是额外搜索预算。默认四个进程，没有线程级求解。

外部车辆互相已有冲突不会导致两车见证被误判；成功要求所有涉及所选车辆的冲突消失，外部路径及外部冲突集合不变。

## 操作

在仓库根目录运行：

```powershell
python scripts/diagnose_local_path_compatibility.py prepare
python scripts/diagnose_local_path_compatibility.py diagnose --dry-run
```

首次运行前，必须在兼容的 WSL Python 中用冻结 native 复核三组历史见证：

```bash
python3 scripts/diagnose_local_path_compatibility.py verify --native --historical-only
```

然后使用 Windows Python 运行参考搜索，无 sklearn 或新增依赖：

```powershell
python scripts/diagnose_local_path_compatibility.py diagnose --workers 4
python scripts/diagnose_local_path_compatibility.py diagnose --stop
python scripts/diagnose_local_path_compatibility.py diagnose --resume --clear-stop
```

`--stop` 只写停止请求：停止调度新作业，等待正在运行的最多四个作业结束。
`--max-jobs 2` 可做安全暂停 smoke；下一次使用 `--resume`。
每个结果原子写入，manifest 增量更新。已有 unknown 不自动重试；错误结果拒绝续跑，需先检查原因。
运行锁存在时禁止第二个 collector。异常退出留下的锁不得未经进程检查直接删除。

完成后：

```powershell
python scripts/diagnose_local_path_compatibility.py verify --repeat
```

```bash
python3 scripts/diagnose_local_path_compatibility.py verify --native
python3 scripts/diagnose_local_path_compatibility.py report
```

路径校验只调用 `reset_paths`，绝不称为 native PP 修复成功。`--repeat` 重复未耗尽预算的三个顺序方法科学输出；CBS 有界结果不混同于无限搜索定理。

## 证据与输出

配置：`configs/local_path_compatibility_v1.json`。最终输出目录由配置指定。
保留输入、实现、正式报告、冻结模型和 native 的 SHA；身份改变必须另立输出版本，不能续用旧结果。
目录包含 `manifest.json`、`source_diagnostics.json`、逐作业 `results/`、`results_manifest.jsonl`、`progress.jsonl`、`run_status.json`、两类验证报告与中文报告。
最初预检目录仅保留两个 smoke 结果；跨平台路径检查修正后使用独立 final 目录，不混合结果。

## 研究边界

PCS/PaPS 启发路径选择与顺序选择的分离；PUSH 启发具体失败时空资源反馈；CBS 参考搜索用于区分联合协调与固定外部占位障碍。本实现不声称复现上述完整算法，也不复制其代码或继承其保证。
没有训练、RL、正式 TTF 或控制器晋级。本轮发现必须经过独立原生实现和新的串行确认，才能作为求解器改进。

- https://www.jair.org/index.php/jair/article/download/19358/27244
- https://arxiv.org/html/2608.06702v1
- https://arxiv.org/abs/2103.07116
