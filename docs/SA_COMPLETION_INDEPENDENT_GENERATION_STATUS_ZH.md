# 新地图确认：计时前任务生成阻塞

## 当前状态

预登记提交 `7242335`；尚未执行qualification、正式计时或模型训练。
计划144次计时，实际0次。不能将本文件称作计时结果或模型失败结论。

Windows与WSL定向测试分别50通过、1跳过；共享运行时、native和冻结模型均未修改。
源分支 `codex/sa-completion-independent-ttf`。修改前完整Git bundle及预登记增量bundle已校验。

## 准备阶段

最初WSL prepare在Windows挂载盘历史目录扫描较慢；只读扫描未写出登记时主动中断，
改由Windows运行相同prepare成功。未改变输入、种子、算法或正式时钟。
历史清单186份、2541个map/task种子；384个历史地图文件已不可读取，隔离证明必须注明此限制。

12个固定map master生成了12张地图：10张两档任务完整，M07和M10仅d20成功，d25采样失败。
全部原始文件、失败地图和成功任务保留在 `build/sa-completion-independent-ttf-v1`。
没有重抽seed、替换地图、降低密度/约束，也没有只对成功生成的10张启动计时。

## 有界诊断

两进程重放同一生成器、同一task seed，捕获失败前缀；枚举该前缀剩余合法起终点对。
另外在原始地图上构造全部符合OD、8--100步最短距离、瓶颈最短路径条件的二分图，
使用已安装SciPy的maximum_bipartite_matching求最大端点匹配。没有调用MAPF求解器。

| 地图 | d25车辆 | 瓶颈约束配额 | 贪心已分配 | 前缀剩余合法起点 | 全图最大瓶颈端点匹配 |
|---|---:|---:|---:|---:|---:|
| M07 | 514 | 206 | 203 | 2 | 206 |
| M10 | 488 | 195 | 193 | 2 | 196 |

M07固定前缀尚需3个瓶颈起点，实际仅余2个，不能仅增加尝试次数补齐。
M10固定前缀仅余3种合法OD组合，随机拒绝采样100,000次仍失败。
原始全图匹配分别206和196，说明瓶颈配额单独并未超过最大端点容量，
随机贪心前缀/极低采样概率是实际问题，不能直接说地图容量不足。

该匹配只验证瓶颈端点的静态容量，不证明还能补齐全部非瓶颈车辆，
更不证明联合MAPF可解。成功见证、计时、控制器性能仍未验证。

## 下一步待确认

推荐保留全部12张图、原seed、密度和40%瓶颈要求，对失败任务采用有记录的确定性端点分配修复，
另立生成修订版本并在计时前冻结；不能覆盖本次失败记录。
它会改变任务生成实现，因此已请求用户确认，没有自动实施。
另一选择是仅更换两张生成失败地图，也需先修订“不换图”协议，不能悄悄执行。

## 证据

- `build/sa-completion-independent-ttf-v1/design.json`
- `build/sa-completion-generation-diagnostic-20260928.json`
- `build/sa-completion-generation-capacity-20260928.json`
- `scripts/diagnose_sa_completion_generation.py`
- `build/pre-sa-completion-independent-20260928.bundle`
- `build/sa-completion-independent-preregister-20260928.bundle`

计时尚未开始；因此尚无新TTF、成功率、SOC或makespan结果可分析。
