# 独立原生路径兼容性：先核验，再诊断

## 范围

正式 Official、V2、Dual16、模型和冻结 `lns2_env` 均不修改。
`src/path_probe` 独立链接已存在的 PIC 核心 archive，只新增另一个扩展模块。
禁止把硬约束两车诊断结果称作正式 InitLNS PP 成功率或端到端 TTF 提升。

本次是下一阶段的准入子步骤，不是完整原生修复器部署：

1. 六个已有开发状态，各两个固定 PP seed，核验完整历史邻域普通 PP。
   比较最终全部路径、回滚、尝试冲突对数、尝试车辆数和逐车成本/冲突数。
2. 全部 12 项通过后，在自动选择的 10 对冲突车辆、20 个顺序上比较普通、
   无定向重选和定向重选，共 60 个作业。外部路径固定为硬约束。
3. 定向约束仍由首车与忽略首车后的后车路径冲突产生，最多四条单独尝试；
   首车不增加成本。无定向方法最多四个 tie seed，集合及顺序不变。
4. 所有成功路径同时经 Python 冲突重建和冻结 native `reset_paths` 校验。

两车不是历史完整邻域的替代品。有些两车对并不同时属于历史选中邻域，
即使这里存在机会，也不能声称正式控制器能够直接利用。

## 冻结预算和失效处理

- 四个独立进程，不使用线程共享 `rand()`。
- 单作业搜索预算 30 秒，硬约束单次 native 搜索最多 5 秒；外部进程 fuse 60 秒。
- 不增加新节点硬上限：现有原生接口没有这一功能，只记录展开/生成数量。
- `empty` 只表示本次单车受约束搜索返回空路径；方法整体用 `not_found`，不作联合无解证明。
- 超时为 `unknown`，不自动增加预算或重试。任何 parity mismatch/error 停止准入。
- `STOP` 文件触发完成当前活动作业后安全停止；原子结果通过身份和内容哈希验证后续跑。
- 最多 72 个作业，搜索预算串行上限 36 分钟；四进程理论约 9 分钟，
  另加载入、验证和 Python 冲突重建开销。不是性能预测或单 worker TTF。

## 命令

先用已有 WSL 工具链单独构建，不覆盖冻结目录：

```bash
cmake -S src/path_probe -B build/linux/native-path-probe-v1 \
  -DCMAKE_BUILD_TYPE=Release \
  -DLNS2_FROZEN_CORE_FILE="$PWD/build/linux/proposal-deadline-fix-validation/libmapf_lns2_core.a"
cmake --build build/linux/native-path-probe-v1 --parallel 4
python3 scripts/diagnose_native_path_compatibility.py prepare
python3 scripts/diagnose_native_path_compatibility.py dry-run
python3 scripts/diagnose_native_path_compatibility.py parity --workers 4
python3 scripts/diagnose_native_path_compatibility.py diagnose --workers 4
python3 scripts/diagnose_native_path_compatibility.py report
```

恢复时使用对应命令的 `--resume`。`--max-jobs` 用于小规模流程检查。
配置、实现、冻结库及输入哈希变化必须使用新输出目录，禁止混合结果。
未通过普通 PP 等价性前，不开发完整邻域的替代/回溯动作。
