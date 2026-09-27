# 冻结 A 与 parent 的轻量串行计时

## 目的与边界

用户在完整任务成本训练对照后要求继续。本轮只验证纯成功率续训 A 的有限信号能否转化为真实首次可行时间收益。
parent/raw-1 与 A 已冻结；B 不参与。本轮不训练、调参、重测工程加速或修改 native、候选、PP、SA。
六张比较地图已经被查看，因此仍是开发性计时，不是新的独立泛化证明。
官方及 Dual16 的历史结果保留，但本轮不把异期结果混成同期对照。

## 固定数据与模型

- 来源：`build/sa-terminal-efficiency-v1` 的全部 parent/A 比较位置，不筛选成功、高负载或有利地图。
- 六张仓库图、d20/d25、solver seeds 251/257、replicas 0/1，共48条件、96 episode。
- 复用来源随机流，初始 fingerprint、候选及 PP/SA 抽样定义配对；允许两模型动作不同。
- 依据完整排序的条件交替 AB/BA，两个模型各24次先运行，每张图均衡。
- parent 文件 SHA：`fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84`。
- A 文件 SHA：`81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062`。
- 来源报告 SHA：`32ae46e4fefa4dfc1cbe5860684f0f6db1a389a1f2e7352d7bafd32847d06a7a`。

## 时间与预算

直接复用已验证的 SharedFeaturePolicy 与 deferred_scientific_audit_v1 计时循环。
主 TTF 从 reset 前到首次可行 env 返回，包括初始 PP、候选生成、特征、推理、修复及此前发生的轻量记账。
终态校验和路径写出计入 delivery，不计入 TTF；逐步科学审计、指纹重建及轨迹压缩全部在计时后。
仍保留模型本身需要的特征、指纹、历史更新及动作合法性检查，不从时间中事后减开销。

单 worker、每次120秒规划预算，外部进程 fuse240秒。没有修复次数或节点上限。
归一化仍使用模型冻结的参考尺度，不改变特征定义。
启动、模型加载及环境构造另报；并行诊断秒数不作为本轮 TTF。
正常120秒未可行为预算失败；fuse、非法路径和身份错误为实验异常，停止检查，不能填作普通失败或自动重试。

## 准入、暂停与审计

- 在计时外对96个模型/条件各重放最多三步，与该模型对应的冻结来源轨迹核对动作、特征、路径和随机流。
- 这是新模型接入检查，不重做旧工程组件速度测试；非计时准入与事后审计最多20进程。
- 沿用运行锁、独立目录、逐episode原子结果和manifest；`stop`在当前episode完成后暂停，`resume`显式续跑。
- 不覆盖残留partial或失败记录；异常应先查明原因，改变实现时另立版本，不混合结果。
- 计时前提交代码、配置、测试及协议，并保存本地Git bundle。没有新增远端授权时不绕过推送限制。
- 计时期间不同时运行审计、训练、其他求解器或重负载程序；外部干扰并不能由脚本自动完全识别。

## 汇总与决定

先报告全部48条件的成功数及新增/丢失成功，再报告双方共同成功的原始TTF。
列出逐地图、逐负载、均值、中位数、p95、快慢数、修复次数、节点、控制开销、SOC和makespan。
失败不填0秒，不将120秒惩罚均值当成原始TTF。所有共同成功指标写出分母。
交付后执行按 `delivery + makespan * step_seconds` 建模，step_seconds为0.5/1/2；仅用双方按时交付的有效路径。
该模型不是机器人实测，不能用TTF单项代替整批完成时间。
使用5,000次地图级配对bootstrap，保留地图聚类，不把48位置当48张独立地图。

固定解释：成功数不下降且共同成功TTF至少降低5%，才称为本批速度正信号；
净成功增加但TTF未达到5%只称为可靠性取舍；负载分层有效只称条件性信号。
区间跨零必须注明不确定；不要求每例获胜，也不据一个均值自动晋级。
若不改善，保留结果而非追加奖励权重或训练更新。默认模型不自动替换。

预计计时45--90分钟，但不是性能承诺；规划预算累计上限3.2小时，进程fuse累计上限6.4小时，另加准入、审计和调度。

```text
python scripts/run_sa_terminal_efficiency_ttf.py prepare
python scripts/run_sa_terminal_efficiency_ttf.py preflight
python scripts/run_sa_terminal_efficiency_ttf.py all
python scripts/run_sa_terminal_efficiency_ttf.py stop
python scripts/run_sa_terminal_efficiency_ttf.py resume
python scripts/run_sa_terminal_efficiency_ttf.py audit
python scripts/run_sa_terminal_efficiency_ttf.py report
```

数据输出 `build/sa-terminal-efficiency-ttf-v2`，不覆盖历史节点预算和TTF结果。

## 准入前接线修正

初版登记保留在 `build/sa-terminal-efficiency-ttf-v1`。准入入口复用了含同模型pair阶段的调度函数，
其字典构造提前访问本轮不存在的pair_worker，导致AttributeError。
错误发生在调度任何准入作业和计时之前，初版没有episode结果。
修正为只选择当前preflight/audit worker，并补充实际调用phase的回归测试。
新目录v2另行登记；模型、任务、随机流、统计规则及计时循环不变。
