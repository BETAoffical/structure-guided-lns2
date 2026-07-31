# 项目文件指南

当前分支只保留官方 LNS2 边界、四个现行控制器、V3-S3 数据与训练链、
墙钟评估以及必要的历史只读兼容。旧 stall/rescue/value/receding-Q 执行代码
已从活跃分支删除，完整源码可从远端 cleanup 标签恢复。

## 目录

| 目录 | 用途 |
| --- | --- |
| `lns2_selector/solver` | 官方 native 求解器边界与适配器。 |
| `lns2_selector/runtime` | 选择契约、状态指纹、时间指标、portable 模型与修复结果语义。 |
| `lns2_selector/controllers` | `official_adaptive`、`v2-full`、`mixed-full-v2`、`v3-s3`。 |
| `lns2_selector/training` | 多个保留训练链共享的训练工具。 |
| `lns2_selector/evaluation` | 墙钟评估入口。 |
| `lns2_selector/compatibility` | 历史 100 轮 AUC 等只读兼容。 |
| `experiments/` | 当前 collection、training、trace、评估的具体实现。 |
| `scripts/` | 薄 CLI；算法逻辑不应复制到脚本中。 |
| `tests/` | solver/runtime/controllers/evaluation/integration/maintenance 测试。 |
| `third_party/mapf_lns2` | 固定的官方 native 源码边界，保持原样。 |
| `artifacts/` | 冻结模型、schema 与研究证据。 |
| `build/` | 本机生成结果，不进入 Git，也不在本次重构中移动或删除。 |

## 控制器关系

LNS2 是求解器内核，V2/V3-S3 是邻域选择器，不建立三个平级“版本目录”。
新实验只接受四个控制器标识：

- `official_adaptive`
- `v2-full`
- `mixed-full-v2`
- `v3-s3`

历史别名只允许 trace/artifact compatibility reader 读取，不再由 CLI 执行。

## 评估指标

新实验使用完整墙钟 deadline，不再把 100 次修复作为执行上限。报告至少包含：

- 成功率和 TTF；
- 最终冲突数与 repair rounds；
- 完整 wall-clock conflict AUC；
- PP 时间、邻域选择时间和选择开销；
- timing closure、native identity 与配对指纹一致性。

100 轮 AUC 只用于读取已登记的历史报告，不参与新的晋级判断。

## 历史代码与恢复

value/receding-Q、旧 V3/H3、stall-safe/shadow/oracle、rescue 与专属 gate
的结论和哈希见 `RETIRED_RESEARCH_EVIDENCE.md`。源代码可从以下远端恢复点
创建临时分支查看：

```bash
git switch --detach backup/selector-cleanup-00-original
```

不要为了查看历史结论把旧执行链复制回当前活跃目录。

## 验证

在既有 Ubuntu-22.04 WSL checkout 中运行：

```bash
cmake --build build/linux/project -j
ctest --test-dir build/linux/project --output-on-failure
/usr/bin/python3 -m pytest -q
/usr/bin/python3 scripts/audit_repository_hygiene.py \
  --config configs/repository_hygiene.json
```

受限 sandbox 看不到 WSL 注册是已知假阴性；不要因此重装或重复注册 WSL。
