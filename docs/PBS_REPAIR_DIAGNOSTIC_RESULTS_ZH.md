# PBS 准入诊断结果

## 结论

2026-09-14，诊断准入未通过，原因是可复现的 native 内存访问错误，不是 PBS 修复效果被否定。
正式 PP、Dual16、退火、模型、候选规则和正式 native 全部未改；本轮没有训练或 TTF 实验。

## 本轮证据

按 `0731107` 预登记：六状态、PP/PBS、同 seed 两次，共 24 个隔离作业，最多 20 并发。
完整历史来源和恢复结构指纹验证通过。原历史 PBS 首次调用没有保存确切参数及堆栈，本轮是使用已保存的四状态重现崩溃类型，不声称重放了原来那一次调用。

| 状态 | 修复前冲突对 | PP 修复后 | PBS |
|---|---:|---:|---|
| Maze32 / 300 agents | 496 | 496 | 两次均 SIGSEGV |
| Random32 高负载 | 264 | 264 | 两次均 SIGSEGV |
| Room64 | 72 | 72 | 两次均 SIGSEGV |
| Warehouse opposite-exchange | 196 | 170 | 两次均 SIGSEGV |
| 微型开放地图，含固定外部车 | 1 | 0 | 两次均降至 0 |
| 微型单行走廊相向交换 | 1 | 1 | 两次均正常失败并回滚 |

PP 12/12 正常返回；PBS 4/12 正常返回、8/12 段错误；外部硬超时为 0。
正常返回的八个 `(状态, 算法)` 组合，重复运行后的结构指纹一致；崩溃分支没有输出路径，不能称为确定性修复通过。
正常返回分支完成显式集合、外部路径不变、路径合法性、冲突图及失败路径回滚检查。
重复运行不是独立随机 trial，以上不是成功率或速度比较。PP 与 PBS 接受条件也不相同。

## 定位与最小复现

用未修改源码另建 `build/linux/pbs-admission-asan-v1`，AddressSanitizer 对六个相同输入重新诊断。
四个历史状态均定位到相同调用链：

```text
InitLNS::runPBS -> PBS::generateRoot -> PBS::planPath
-> SIPP::findPath -> ConstraintTable::getLastCollisionTimestep
-> ConstraintTable.cpp:25 的 cat[location][t]
```

源头位于 `third_party/mapf_lns2/src/ConstraintTable.cpp:23`：

```cpp
for (auto t = cat[location].size() - 1; t > rst; t--)
```

`size()` 是无符号数，因此 `auto t` 也是无符号数。发现两个相关错误：

1. 外部路径在目标格存在冲突时刻，而本地 CAT 的该格时间序列为空：`0-1` 下溢为极大整数，进入循环并越界读。
2. 外部路径在该格没有冲突，即 `rst=-1`，但本地 CAT 存在冲突：比较时 `-1` 转成无符号极大值，循环被跳过，可能漏掉本地最后冲突时刻。

最小 C++ 复现仅构造 ConstraintTable 和两条路径，没有模型、Python 状态恢复或候选选择：

| 条件 | 正确结果 | 实际结果 |
|---|---:|---|
| 外部命中、本地该格为空 | 2 | ASan 越界段错误 |
| 仅本地命中 | 1 | 错误返回 -1 |
| 外部、本地均命中 | 3 | 3 |
| 两者均不命中 | -1 | -1 |
| 仅使用外部路径表，不建立本地 CAT | 2 | 2 |

该函数在本仓库从首次官方代码导入 `13f7cc8` 起没有后续修改；这里确认的是仓库 Git 历史，不据此概括所有上游版本。

## 与当前正式算法的关系

- 当前 PP 修复路径通过共享 PathTable 提供 CAT，没有 PBS 中逐条 `insert2CAT` 所建立的本地 CAT，因此不会以本轮复现方式触发该分支。
- 不能用此次 PBS 崩溃解释现有 Dual16/SA 的长尾，也没有证据要求重跑既有 PP 正式实验。
- 这是共享约束表函数，修复若直接进入正式构建，仍需独立审计所有调用方，不能只验证 PBS 后静默替换 baseline。
- 此次没有修改共享函数；正式 native SHA 仍为 `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。
- 冻结模型 SHA 仍为 `b6afff3bf436c8a2acb5571094817ce94aef71b5aa0cb1c2369de1c0e1d3abe2`。

## 下一步边界

只在独立 PBS 诊断构建中修复有符号反向索引，保持正式构建不变；先验证五个 CAT 边界、六个修复输入、失败回滚、外部路径恢复及超时，再判断是否存在其他 PBS 错误。
预算仍存在单车搜索不能合作式及时中断的问题，微型失败回滚也不能替代搜索中途超时回滚检查。
只有这些准入通过，才能对同一 agent 集合比较 PP/PBS 的局部修复机会。任何一步都不自动升级为生产控制器或 TTF 试验。

## 验证与证据

- 新增 Python 定向测试 7/7 通过。
- 24 个冻结 native 诊断、6 个 ASan 诊断、5 个最小查询诊断全部完成；错误按上述分类保留，不伪装为通过测试。
- 正式 C++ 源码和 native 未改，本轮不重跑完整 CTest 或正式 parity 实验；仅核对冻结 SHA 与已有来源输入。
- 原始记录保留在忽略目录 `build/pbs-repair-admission-v1`；日志含 ASan 平台绝对路径，不进入 Git。
- `report.json` SHA256：`525e6685eba514590a9373e0e4edc39094180e69267b0d3fed5f0095c8bdf1d6`。
- `asan/report.json` SHA256：`c1530b68419b5e3af496dcac6ffcb44778d0b0facd17c29cfb96bc6a0a94b5e3`。
- `minimal/report.json` SHA256：`1df866f58881915c60bb7e6ec6d59cd03f2bc67f4c8f732f5aff9ca9af529770`。
