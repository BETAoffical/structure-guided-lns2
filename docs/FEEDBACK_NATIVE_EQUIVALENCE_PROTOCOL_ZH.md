# 反馈记忆的native条件等价性诊断

## 运行前固定范围

不改变Official、V2、Dual16、模型、候选或PP逻辑。本轮只判断可见修复条件键是否遗漏
影响结果的native历史信息，不训练反馈策略，不测TTF，不追加地图。

来源：`build/feedback-memory-audit-v2/report.json`，SHA
`3457c9b43ebeb5f045ef3101e5f146674767c7c5cb506aeacb1a28267e1d38a1`。
这是已查看结果的开发诊断，不是独立效果确认。

- 三个慢案例09/13/22：各取最早出现的“同修复问题键、不同选中旧路径”快照对。
- 两个恢复对照：排除上述三例后，按checkpoint/controller排序选前两组
  “同键先不下降、后来严格下降”的快照对。
- 每组两个快照，各采用直接reset_paths和原初始状态完整prefix重放。
- 每种恢复使用ID升序/降序两个固定repair_order、两个配对PP随机seed。
- 合计5组 x 2快照 x 2恢复 x 2顺序 x 2seed = 80个独立作业。
- PP预算固定5秒，环境600秒仅用于容纳历史prefix；单作业硬超时120秒，20worker。
  prefix仍使用原请求预算并验证路径、状态指纹及原repair_order，不把新诊断预算写回历史。

## 验证与门槛

每个group/order/seed形成四个对照，共20个cell。固定要求：

1. 80/80作业完整，源SHA和native身份不变，0非法动作、0prefix mismatch。
2. 同cell的可见修复条件键相同；接受/回滚、失败类型、尝试冲突数、失败agent、
   已尝试agent数、最终冲突数与可行性一致。
3. 接受时，四个条件的选中agent新路径一致；回滚时验证各自输入完整恢复，
   不能要求本来不同的旧路径变得相同。
4. 任何截断的cell均为未知，不能计为通过或正常负反馈。
5. generated/expanded/runs差异独立报告，不把计数相同等同TTF相同。

20/20无截断且结果一致，只支持这批条件下的native等价性；不证明普遍等价，
不晋级控制器。任何不一致则停止反馈实现，先分析遗漏变量，不自动改native排序、
扩大预算或追加随机种子。

外部路径、选中起终点和接受上限变化必须使键失效，由前一轮单元测试验证；
负对照不要求搜索结果一定不同。

## 执行与恢复

```bash
python scripts/diagnose_feedback_equivalence.py prepare
python scripts/diagnose_feedback_equivalence.py collect
python scripts/diagnose_feedback_equivalence.py analyze
```

prepare只保存快照与冻结计划，报告准确prefix修复次数。collect逐作业原子写出，
进程级隔离，使用整组锁，已有失败文件拒绝直接续跑。每20作业一批，若输出目录
存在`STOP`文件则当前批结束后停止，不再调度下一批。异常中断保留已有结果。
实现SHA在prepare登记，变更后拒绝继续混跑。

输出：`build/feedback-native-equivalence-v1`，不进入Git。
本地恢复标签：`pre-feedback-native-equivalence-20260911`。
不自动推送包含此前尚未获准上传结果证据的分支历史。

## 准备登记

- 计划SHA：`6b351af9a8b459b40979ee5587dfa21838017c82e01ed648f3db9ff256653d7a`。
- 80次新修复，另有372次历史prefix修复用于恢复条件；这些都不是TTF样本。
- 快照：Dual16 09为d6/d13，13为d8/d9，22为d5/d6；
  恢复对照Dual16 08为d9/d10、V2 11为d12/d15。
- 22选择的是规则命中的最早快照，不是后来最长平台的入口，不能据此代表其整个末期长尾。
- 准备阶段修复了新脚本的state_blob相对根目录错误，未开始修复作业，没有旧新结果混算。
  新增路径回归测试后，共31项新增及关联测试通过。
