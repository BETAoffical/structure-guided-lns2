# Raw 动态选择器的无效历史计算审计

## 边界与历史

从已完成串行 TTF 的本地提交 `140a617` 开始。仅工程诊断，不训练、不改变默认控制器、不执行整组 TTF。
既有 grouped proposal、native 特征、prepared 复用、单次完整 fingerprint 均保留，不重复开发。
PP、SIPPS、native、候选池、模型、归一化、SA、排序和随机数流不得修改。

`SA_ONPOLICY_DESIGN_ZH.md` 已冻结 dynamic 输入：127 维动态/阶段特征加 2 个预算参考特征，共 129 维。
当前模型没有 `history.*` 列。旧 `features_for` 仍逐候选计算 History.features，包括外部路径 JSON SHA，
然后由 `profile_features(..., 'dynamic')` 删除全部 history.*。这是未使用计算，不是本轮增加或删除模型输入。
温度、decision、预算参考及 History.observe 保持；不把这一轮称为历史信息学习有效性的证明。

## 实现与验证

- 新增显式选择的 FastPolicy，不修改历史 Policy、selection、features_for。
- 私有函数绑定复用冻结 code object，不修改全局函数或默认入口。
- 仅将被过滤掉的历史计算替换为相同的阶段数值；保留候选合法性和历史连续性检查。
- 原 frozen 模型/特征 schema 不匹配时拒绝，不自动回退或修改模型。
- 所有 source SHA、登记报告、逐 episode 文件 SHA 均验证；输出独立放在 build/sa-raw-selection-engineering-v1。

1. 串行剖析 12 条 raw_updated 轨迹（全部六图、两密度、seed251、replica0），每条早中晚三个状态。
   使用已审计的原候选池，零新 PP；cProfile 只定位候选生成之后的热点，不当成实际 TTF 占比。
2. 20 进程重算全部 96 条父/新模型轨迹。每步特征、概率、越界率、抽样动作、PP seed 完全等于原记录。
   只重建已保存状态，不重跑求解；禁止选择性跳过失败轨迹或最后步骤。
3. 等价通过后，在同一 36 状态串行测量原/快路径，三次重复、交替顺序。
   每次用独立 engine 和状态副本，在计时外准备初始静态分析及前一状态；不跨测量复用当前状态热缓存。
   数据读取、模型装载、复制和相等性检查在计时外。测的是候选生成后的选择成本，不含 proposal/PP，不能换算成 TTF。
4. 用 12 条相同来源的前三步做原生完整选择/修复前缀验证，共 36 步；特征、动作、路径、冲突和低层计数必须一致。
   单 PP 20 秒安全限额仅用于前缀验证；不是实际 episode 新预算。进程 fuse 900 秒，出现不一致停止。
5. 报告组件差异、完整性、保留输入 SHA 和局限，默认不晋级、不自动开启新 TTF。

CLI：`python scripts/audit_sa_raw_selection_cost.py prepare|profile|verify|benchmark|native|report`。
profile/benchmark 单进程；verify/native 最多20进程。逐作业原子保存，已成功记录可以续用，失败记录必须检查后另行处理。
本轮保留全部历史证据，备份、本地提交，不在未获新授权时推送。
