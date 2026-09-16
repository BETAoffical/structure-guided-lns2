# 分段历史信号的匹配负对照

2026-09-17；这是第一轮结果后的探索性跟进，不称为独立确认。
第一轮JSON SHA：`1ffdca0ced13d71ce575cccfa692906afc97854c5041b4e27e2b217ce58cbd69`。
ordered相对aggregate持续改进Brier降低9.759%，7/8图不劣；resource_ordered没有顺序增量。
原来的五组乱序为223维组合特征，不能直接隔离191维ordered的顺序贡献。

本轮只补齐同维度负对照，不改变已有标签、地图、状态权重或学习器：

- 五组191维temporal_shuffled：仅将ordered使用的同一组记录乱序，保留同样分段与原aggregate。
- 一组154维temporal_bag：原aggregate加8种时间记录量的无序均值，不加入外部路径资源字段。
- 复用原ordered/dynamic/aggregate的折外预测，不重新选择参数。仍只使用204观察点、69episode、8图。
- 两个目标均报告；持续改进为主，32步完成为次。每种对照八折，共96个小模型拟合。
- 至多20个单线程进程；0求解器运行，0新反事实标签，0控制器变更。

如果ordered不能在配对地图区间上优于temporal_bag和全部五组matched shuffle，就不把收益归因于顺序。
即使通过，仍只是观察性表示信号。不能把这套未来事件预测器直接对未选邻域打分。
不根据结果增加第六次乱序或改32步窗口，不把次目标较好的对照替换主方案。
输出另存`build/sa-history-information-matched-v1`，绝不覆盖第一轮报告。
