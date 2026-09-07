# 扩充高压力任务：计时前准入

## 两批设计

保持原批次不变，将新增独立起终点作为第二批。
每批均为8张地图 × 3种密度 × 2种OD × 2个solver seed，共48任务、96个任务-seed条件。
两批合计96任务、192个条件，独立地图仍为8张。各批均运行Official Adaptive、V2、Dual16三个冻结控制器。

| 批次 | 配置 | 输出目录 |
|---|---|---|
| 原批次 | `configs/path_quality_pressure_evaluation_v1.json` | `build/path-quality-pressure-evaluation-v1/` |
| 新起终点批次 | `configs/path_quality_pressure_evaluation_replica2.json` | `build/path-quality-pressure-evaluation-replica2/` |

各批独立登记、独立暂停和续跑，每批864个episode，总计1728个。
每批内部依次执行首解交付、60秒总预算、120秒总预算；不改写原批次排程。
这不是新增独立地图确认，也不是根据冲突筛选的均匀冲突组。

## 初始化与身份边界

1. 第二批48任务通过静态检查。
2. 第二批96次Pilot PP reset，禁止修复与计时比较。
3. 第二批60/120秒预算分别准入，共192次reset；第一批已有192次预算准入。
4. 核对预算专属fingerprint与对应Pilot reset一致，三方法使用同一初始状态anchor。
5. 两批的模型/runtime模板、native SHA、协议、计时边界和官方第二阶段相同。

所有任务保留，不根据冲突、控制器表现或是否容易求解更换任务。
零冲突也应保留；本轮实际是否出现零冲突由初始化报告记录。
合法初始路径允许碰撞。准入通过不能证明联合可行，也不能保证在规划预算内求解成功。

## 只读汇总核验

```powershell
python scripts/audit_path_quality_pressure_batches.py
```

该脚本不加载native执行求解，不调用collect。读取已登记的两批排程、模型/runtime身份和384份预算准入。
检查不同任务ID/seed/实际OD、相同地图SHA、相同因素组合、1728个唯一排程位置及384项fingerprint一致性。
结果保存到 `build/path-quality-pressure-batches-v1/admission_audit.json`。
它仅用于计时前检查：发现已存在计时授权或episode目录会拒绝，以免把运行后的状态报告成尚未计时。

## 启动与安全停止

**计时必须单独授权，并确认接电、固定电源模式和无重负载。完成初始化不自动启动计时。**

在WSL中分别选择上述批次配置，使用既有入口：

```bash
PYTHONPATH=build/linux/anytime-handoff-validation /usr/bin/python3 scripts/run_path_quality_pressure_evaluation.py collect --config configs/path_quality_pressure_evaluation_v1.json --authorize-timing --quiet-machine --on-ac
```

第二批仅将配置改为 `configs/path_quality_pressure_evaluation_replica2.json`。两批不得同时运行。
在当前批次输出目录创建 `pause.request` 后，当前episode结束时安全暂停，不杀死正在求解的任务。
检查环境、移除暂停请求后，相同命令增加 `--resume`；已完成结果先校验后复用，异常不自动重试。
暂停发生在单个方法的episode边界，不必等三个方法都完成。初始PP检查的暂停文件另为 `STOP_AFTER_RESET`。

合计两组固定预算的名义规划时长为28.8小时；所有规划预算用满合计48小时，全部外部fuse累计105.6小时。
这些是预算上限计算，不是实际耗时预测。默认不承诺一次连续跑完两批。

## 统计边界

最终须同时报告两批结果和全部任务结果，不从两批中挑较好者。
合并时按地图做paired bootstrap，同一地图的两组OD抽样和solver重复都放在同一地图组。
报告自然初始可行率、条件TTF和路径质量，区分失败及共同成功；不要把192个条件当成192张地图。
本轮只有初始化数据，没有新的TTF、makespan优劣或控制器晋级结论。

## 初始压力结果

新增批次96/96次Pilot PP reset有效，0错误，0零冲突，0修复。
两批合并后的统计如下；冲突指唯一碰撞agent对数，不是重复时空事件数。

| 自由格点密度 | 任务-seed条件 | 最少冲突 | 平均冲突 | 最多冲突 |
|---|---:|---:|---:|---:|
| 15% | 64 | 5 | 21.28125 | 62 |
| 20% | 64 | 31 | 79.59375 | 181 |
| 25% | 64 | 89 | 201.5 | 377 |

新增批次各档均值为21.3125、79.28125、198.65625，与原批次21.25、79.90625、204.34375接近。
这表明在这两组起终点抽样中压力梯度一致；它不是修复速度、路径质量或泛化优越性的证据。
高密度任务出现89个冲突也完整保留，不为达到某个冲突数量阈值替换任务。

新增Pilot reset报告SHA256：`4078708aaff118000276233ceb4b1164ae75c5d1f9d24279011be29c014dd655`。
新增批次排程登记fingerprint：`787660b9e699a384f07a9fb02141a914ad1b685ada7debb5f4267a3f2c96e493`。

## 最终准入验收

- 新增192/192项60/120秒预算准入通过，结合原批次共384/384项；与各自Pilot初始状态fingerprint交叉核对，0不匹配。
- 两批共96个不同起终点任务、192个任务-seed条件、1728个唯一排程位置；每种方法和协议均衡，未自动授权任何episode。
- 模型/runtime模板、native SHA和地图文件身份跨批次一致；原批次配置、排程和旧252次正式报告保持不变。
- 汇总核验重复执行结果一致，状态为 `ready_pending_separate_timing_authorization`。
- Windows相关测试66项通过；维护修正后WSL专项20项通过；WSL完整复测最终为 **1273 passed, 46 skipped**。
- 跳过项为41项独立Linux CTest覆盖的原生接口检查、4项Windows sklearn训练环境检查、1项Windows专用路径检查。
- 当前注册构建的LNS2 CTest **12/12通过**。另两项GPBS测试未运行，因为该构建没有GPBS二进制；GPBS不属于本轮比较方法。
- 官方原始路径SHA保持 `915ee104f0168c463f05925541fef1c22ec1eb37e9bf8df7ab09807753013ecf`；单次修复parity SHA保持 `031d1bf843ada89f03be6880809bcf7632fdaeba509fd035a15657fcc93aa83a`。

首轮完整测试有2项维护检查失败：15个近期压力实验文件未登记用途，冻结cohort模块存在两个预先留下的无用导入。
已补齐用途登记；对 `time` 和 `read_artifact` 添加了精确到文件与名称的说明性保留项，没有全面关闭无用导入检查。
这两个导入本轮仍保留，后续另立源码版本清理，避免更改已冻结的执行模块SHA；没有修改控制器、计时边界或求解器。
首轮日志保留为 `build/path-quality-pressure-batches-v1/python-tests.xml`，最终完整复测为同目录 `python-tests-final.xml`。

新增预算准入报告SHA256：`a4f7945e95b83f5b6d92bb06a207f6329dae676b5f17940fd4f0774e89576114`。
新增排程SHA256：`902c1f6f3f0db78c6466fb730b50a1360b1ff265a5b8142891bb7e6ca1a114e8`。
两批准入汇总SHA256：`cda76c3bb2b3362269c2b3b98784cbc55eee31fef6e71999adb1a8759aab2ea9`。

本轮正式计时episode仍为0。计时启动前还需用户单独授权，确认接电及无重负载，并保证当前注册分支的源码已推送、工作区干净。
