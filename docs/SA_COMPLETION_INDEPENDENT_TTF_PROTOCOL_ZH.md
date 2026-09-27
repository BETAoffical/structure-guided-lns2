# A、原模型与 Official+SA 的新仓库地图计时确认

## 冻结问题与范围

在任何新任务生成和计时之前固定本协议。上一轮六图属于已查看的开发证据，
A高负载增加成功但较低负载变慢。本轮只检验这种取舍能否在新同族地图重现。
不训练、不调参、不改候选、PP、SA、冻结native或默认控制器，不加入事后密度路由。

- 原模型：raw-1，文件SHA `fad8c8c64d68aefbd19bcfa620eb36f479fbd95d835d819ca9fd30ae2021fe84`。
- A：completion-only续训，文件SHA `81727429f423734aef544c1e12ad61fae965a2118898fadafeab699dd9307062`。
- Official+SA：原生官方Adaptive邻域与同一实验SA接受机制，无模型、无候选池排序。
- native继续使用 `build/linux/sa-wall-clock-v1`，SHA `5b1b2af1a8a388e76567c3f1e874674ef8c2701775c3706d0a1e4d9678d1925d`。

这里的Official+SA不是未加SA的官方LNS2，不能混用其名称或历史计时。
父模型/A继续采用已经验证的SharedFeaturePolicy工程路径，正式求解循环复用lean_worker，
旧工程加速不重复验收。现存完整身份链纳入新登记，不绕过旧结果保护。

## 数据

12张新48x72 station_centric仓库图，复用上一轮生成器与几何参数。
master seed为2026092801，确定性生成12个map master，不因结果重抽。
每图bottleneck_d20/d25，密度为agent数/可通行格数，瓶颈约束比例0.4；任务独立生成而非agent前缀。
solver seeds固定271/277，replica仅0。24任务、48条件、144三方法episode。
map/task种子及可读取地图SHA与现存历史manifest隔离；历史缺失文件明确报告，不能声称检查了已删除原始数据。
solver seed不是独立地图样本，不以数字不同替代几何隔离。

资格检查只检查全部48个reset合法完整、地图/任务身份匹配。零冲突保留、难例保留。
不要求75%产生冲突，不按冲突数换图，不按算法是否成功选择测试条件。
任何生成或reset错误停止准入，先检查原因，不悄悄更换种子。

## 计时与准入

配置、协议、源码先提交；新数据和reset结果再形成独立execution registration。
每个任务/seed三方法reset fingerprint完全相同；同条件选择/PP/SA流由固定stream2026092802、
phase和条件身份派生。Official保留自身native邻域/PP RNG，不强迫与学习器采用同一邻域。

准入采用真实reset、冻结模型读取、第一步选择与proposal状态不变检查，不执行额外PP前缀。
这是新输入准入，不是重复工程等价或提前计时。正式完整轨迹事后校验。

- 每episode120秒规划预算；外部fuse240秒，不是额外求解预算。
- 无决策次数、节点数上限。25M节点和256步仅保留为冻结模型特征归一化参考。
- 正式单worker，数值库单线程；三方法六种排列均衡轮换，各密度/seed分层均衡。
- TTF从reset前到首次可行env返回，含初始化、候选、特征、推理、修复及此前必要记账。
- 最终路径校验/保存计入delivery；全步科学审计、trace压缩在交付后，不计入TTF。
- 进程启动及模型构造单列，不计入常驻规划器TTF。
- 非计时最多20进程，生成只有12个独立作业。不能把并行准备耗时当成TTF。
- 每episode完成后可安全暂停，显式resume才继续；部分输出、fuse或身份错误不得自动重试。
- 原始输出原子保存，输入/模型/数据变化拒绝续跑。正常预算未解保留并继续。

求解预算总上限4.8小时；进程fuse累计上限9.6小时，均不含准备、导出和审计，不是性能预测。

## 预固定分析

主对照A对parent；同期A对Official+SA、parent对Official+SA完整报告。
主问题是成功率和真实TTF，不要求每张地图或每项指标都胜出。
报告净增/丢失成功、各自成功集和共同成功集的不同分母；失败原始TTF为空，不填0或罚值。
平均、中位数、p95、胜负数、逐图与d20/d25分层；5000次地图级paired bootstrap，seed2026092803。
路径SOC、makespan、等待步数与每车平均步数仅在共同成功集配对，失败路径不作为可执行解。
整批完成时间为delivery + makespan * 每步秒数，主1秒、敏感性0.5/2秒；不是实车实验，不运行第二阶段。

可靠性信号与速度信号分别解释，不设置两者必须同时全胜的晋级门槛：
- 净成功增加且跨图结果支持，保留可靠性改善信号，同时列出全部成功损失与速度代价。
- 共同成功平均TTF降低，完整报告其CI、成功率和各图取舍；5%仅为既有实质速度参考，不改为唯一价值标准。
- CI跨零表示方向不确定，不称为已证明提升；没有显著差异也不等于方法完全无效。
- 高负载收益/轻载退化若重复出现，作为未来独立负载保护研究的依据，不在本轮加路由。
- 默认不自动替换模型，不据结果追加训练或挑图重测。

新同族地图检验不等于跨布局、跨OD迁移。三臂配对及完整排程校验后再发布报告。

## 命令

`python scripts/run_sa_completion_independent_ttf.py prepare`
`generate`、`qualify`、`register`、`preflight`依次执行；计时和审计在WSL真实用户环境中运行。
`all`执行串行collect、完整audit和report；`stop`完成当前episode后暂停，`resume`仅恢复collect。
恢复收集完成后显式执行`audit`和`report`，不重复已有完成阶段。
所有生成数据仅在忽略目录 `build/sa-completion-independent-ttf-v1`，旧证据不覆盖。
