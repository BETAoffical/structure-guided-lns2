# 项目文件指南

当前分支只保留官方行为的 LNS2 native 边界、四个现行控制器、V3-S3
数据与训练链、墙钟评估，以及读取历史正式产物所需的最小兼容代码。
旧 stall/rescue/value/receding-Q 执行代码已经删除，完整源码可从远端
cleanup 标签恢复。

## 目录结构

| 目录 | 用途 |
| --- | --- |
| `lns2_selector/solver` | 官方 native 求解器边界与身份检查 |
| `lns2_selector/runtime` | 选择契约、指纹、墙钟指标、portable 推理和 repair 结果语义 |
| `lns2_selector/controllers` | `official_adaptive`、`v2-full`、`mixed-full-v2`、`v3-s3` 适配器 |
| `lns2_selector/training` | 当前 bundle 重建所需的共享训练工具 |
| `lns2_selector/evaluation` | 墙钟评估公共边界 |
| `lns2_selector/compatibility` | 历史指标和 artifact 的只读兼容支持 |
| `experiments` | 当前 collection、training、trace 和评估实现 |
| `scripts` | 薄 CLI；算法逻辑不应复制进脚本 |
| `configs` | 版本化实验注册；详见 `configs/README.md` |
| `tests` | 当前行为、已知 bug、artifact 完整性和复现契约 |
| `third_party/mapf_lns2` | 固定的官方 native 源码边界，保持原样 |
| `artifacts` | 冻结模型、schema 和研究证据 |
| `build` | 本机生成结果，不进入 Git，也不在本次重构中删除 |

LNS2 是求解器内核；V2 和 V3-S3 是求解器内部的邻域选择器，因此不建立
三个平级的“版本目录”。新控制器接口只接受以下标识：

- `official_adaptive`
- `v2-full`
- `mixed-full-v2`
- `v3-s3`

历史策略标签只允许 artifact/trace compatibility reader 读取，不能通过新的
controller loader 执行。

## 地图与配置

`configs/` 中数量较多的 JSON 是不同正式 cohort 的冻结注册，并不是重复地图。
MovingAI 地图和生成的 warehouse 地图位于被 Git 忽略的 `build/` 中；上游测试
fixtures 位于 `third_party/mapf_lns2`。配置文件只有在不再服务当前复现、兼容或
活动链路时才能删除。

每个生产 Python 模块和 pytest 文件都必须出现在
`configs/retention_manifest.json` 中，且分类为 `active`、`shared`、
`reproducibility` 或 `compatibility`。`evidence-only` 和 `obsolete` 只说明删除
决策，不得继续作为可执行代码留在活动分支。

## 评估指标

新实验使用完整墙钟 deadline，不再把 100 次 repair 作为执行上限，也不使用
前 100 次 AUC 做晋级判断。报告至少包含：

- 成功率和 TTF；
- 最终冲突数与 repair rounds；
- 完整 wall-clock conflict AUC；
- PP 时间、邻域选择时间和选择开销；
- timing closure、native identity 和配对指纹一致性。

100 次 AUC 仅用于读取已注册的历史报告。

## 历史代码恢复

value/receding-Q、旧 V3/H3、stall-safe/shadow/oracle 和 rescue 的结论、配置与
哈希记录在 `docs/RETIRED_RESEARCH_EVIDENCE.md`。如需查看源码，应从远端标签
创建临时分支，不要把它们复制回活动目录：

```bash
git switch --detach backup/selector-cleanup-00-original
```

## WSL 验证

`/home/beta/LNS2-RL` 是另一份带用户修改的 checkout，不用于验证当前 Windows
工作树。从当前仓库的 PowerShell 目录取得 WSL 挂载路径：

```powershell
$repo = (Get-Location).Path
$wslRepo = (wsl.exe -d Ubuntu-22.04 -- wslpath -a $repo).Trim()
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- sh -lc 'cmake --build build/linux/project -j && ctest --test-dir build/linux/project --output-on-failure && /usr/bin/python3 -m pytest -q'
wsl.exe -d Ubuntu-22.04 --cd $wslRepo -- /usr/bin/python3 scripts/audit_repository_hygiene.py --check
```

受限 sandbox 看不到 WSL 注册是已知假阴性；不要因此重装、重新注册或复制 WSL。
