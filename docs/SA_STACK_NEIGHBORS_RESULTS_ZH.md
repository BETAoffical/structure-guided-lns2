# SIPPS栈上四邻接枚举：局部收益与完整重放通过

## 结论

在不修改冻结baseline的情况下，生成并编译了一个独立native：仅将SIPPS `findPath`中的一次邻接链表查询改为栈上四方向枚举，保留同一个validMove过滤、原方向顺序、重复邻居及其余搜索逻辑。

两层验证均通过：

- 四张实际地图的全部自由格，邻居内容和顺序逐项一致。查询微基准累计时间减少 **70.5197%**。
- 独立native在8案例、792步完整前缀中，reset、候选、评分、选择、路径、低层计数、PP顺序及修复结果全部与冻结历史记录一致。

决定是 `eligible_for_bounded_pp_timing`。**这不是PP减少70.5%，更不是TTF减少70.5%。** 当前只证明查询本身可减少开销，以及这些轨迹上的行为一致；还没有实际PP或端到端加速结论。不替换默认控制器、不改SA、不训练。

## 唯一生产代码差异

原来的循环调用 `instance.getNeighbors(curr->location)`，返回临时`list<int>`。

独立生成的SIPP.cpp改为：

```cpp
const int neighbor_candidates[4] = {curr->location + 1, curr->location - 1,
    curr->location + instance.num_of_cols, curr->location - instance.num_of_cols};
for (int next_location : neighbor_candidates)
{
    if (!instance.validMove(curr->location, next_location)) continue;
    // Original successor expansion body follows unchanged.
}
```

并非静态地图缓存，没有缓存失效状态。仅替换带time_limit参数的findPath函数中的这个循环；其它SIPP方法、Instance公共接口、heap比较器、rand调用、deadline检查和路径表维护全部不改。单列地图原本可能有重复枚举，本轮照原样保留。

构建使用原Release的编译flags、静态库和绑定对象，重新编译独立SIPP.o后优先链接到独立so。源树中的third_party/mapf_lns2没有编辑，原build/linux/sa-wall-clock-v1未覆盖。生成源码反向替换可逐字恢复原文，测试保护这一边界。

## 查询微基准

直接链接原Instance::getNeighbors作为参考，栈枚举使用同一个Instance::validMove。先逐格比较，再预热两侧；每轮每侧200000次查询，4轮交替顺序。每张地图每侧800000次，下面为累计秒。

|地图|自由格数|原链表秒|栈枚举秒|减少|
|---|---:|---:|---:|---:|
|m01|1796|0.0569405|0.02136613|62.48%|
|m03|1825|0.0628879|0.01875702|70.17%|
|m05|1971|0.0720952|0.01767850|75.48%|
|m07|1527|0.0596690|0.01636867|72.57%|
|合计|四图|0.2515926|0.07417032|70.52%|

保留校验和防止结果被优化掉，未调用随机数。微基准按自由格轮转，不代表SIPPS实际展开位置分布；只有查询，没有区间搜索、heap、冲突扫描、路径表或Python控制器。其绝对时间很短，不能外推总求解节省。预注册的10%局部门槛通过后才构建native。

## 完整native重放

使用既有4图8案例，solver seed103。每步重新生成并评分候选，对比冻结selected_index和完整pool，再按保存的显式动作、温度和uniform执行PP，逐步重建历史状态并比较完整fingerprint及PP字段。

|地图|15%密度步数|25%密度步数|结果|
|---|---:|---:|---|
|m01|48|195|全部一致|
|m03|6|249|全部一致|
|m05|11|76|全部一致|
|m07|19|188|全部一致|
|合计|84|708|792/792|

8个独立进程用于加快非计时验证，不能拿运行用时评价PP速度。每案例900秒fuse、每步300秒安全上限；没有错误、路径分歧或超时。仍是已查看的开发轨迹，不能声称覆盖所有状态、所有地图或所有随机数种子。

## 测试和证据

- 构建前17项测试通过，包含源码变换、边界/障碍/单行单列语义、工作量审计及保留清单。
- 新native加载路径下43项测试通过，包含现有原生Python接口、deadline、状态和相关检查。两批有重叠，不相加称为60项独立测试。
- 8份重放结果的SHA校验通过；新native `f214c49b...c04e3fd`，原native仍为 `5b1b2af1...1925d`。
- 原模型、baseline和历史报告身份通过原验证链。未运行新版本完整CTest/官方CLI parity，因此不宣称所有C++入口已通过。
- 注册提交 `6761ef8`；恢复标签 `pre-sa-stack-neighbors-20260913`。本地源码ZIP、生成对象和native留在build，不提交二进制。
- [紧凑证据](../artifacts/sa-stack-neighbors-probe-v1/evidence.json)；原数据 `build/sa-stack-neighbors-probe-v1`。仅本地提交备份，未推送，无残留诊断进程。

## 下一步

只做范围受限的原native与新native配对PP组件计时，使用相同前状态、动作和随机设置，单worker、交替顺序，并继续核对路径和节点计数。计入实际修复，不只测getNeighbors；新旧二进制身份必须明确区分。

先看PP真实净收益再决定是否值得完整TTF，不能因为微基准快70.5%就直接扩展大规模正式试验。本轮到机制验证为止，未自动开始PP计时或TTF。
