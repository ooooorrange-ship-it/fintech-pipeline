# 最终项目一致性审计

总体状态：pass

| 检查项 | 状态 | 详情 |
|---|---|---|
| valid全量账户预测覆盖 | pass | 预测行=11087，账户数=11087，重复ID=0 |
| valid全量指标存在 | pass | 融合模型报告文件存在 |
| test全量账户预测覆盖 | pass | 预测行=11087，账户数=11087，重复ID=0 |
| test全量指标存在 | pass | 融合模型报告文件存在 |
| 主模型标签覆盖 | pass | 标签账户数=11087，节点账户数=11087 |
| 未来交易未进入分片图样本 | pass | {"train": 0, "valid": 0, "test": 0} |
| 标签时间限制已显式记录 | pass | 风险标签表没有标签时间，无法验证“未来新增标签”口径；当前审计保证标签字段不进入特征，动态特征只使用 split end 之前的滚动历史交易。 |
| 特征文件不含标签字段 | pass | 命中特征文件=[] |
| train动态图观察截止时间 | pass | observation_end=2025-10-31 23:59:59，split_end=2025-10-31 23:59:59 |
| valid动态图观察截止时间 | pass | observation_end=2025-11-30 23:59:59，split_end=2025-11-30 23:59:59 |
| test动态图观察截止时间 | pass | observation_end=2025-12-31 23:59:59，split_end=2025-12-31 23:59:59 |
| 59个嫌疑账户审计完整 | pass | 审计=59，确认嫌疑人=59 |
| 缺边恢复队列完整 | pass | 恢复队列=56，审计缺边账户=56 |
| 缺边恢复队列账户可追溯 | pass | 恢复队列账户=56，审计缺边账户=56 |
| Top30巡检队列完整 | pass | 巡检账户数=30 |
| 分层覆盖统计与CSV一致 | pass | coverage JSON 与 CSV 行数一致 |
| 关联账户节点ID可追溯 | pass | 记录数=70，未知账户=[] |
| 可疑路径节点ID可追溯 | pass | 记录数=144，未知账户=[] |
| 资金结构节点ID可追溯 | pass | 记录数=2，未知账户=[] |
| 任务2使用全量账户评估 | pass | evaluation_scope=all_accounts |
| 任务2 AUC达标 | pass | AUC=0.9879674541844181 |
| 任务2 Top5%召回达标 | pass | Top5%召回=0.9661016949152542 |
| 任务2强基线PR-AUC提升20% | pass | 相对强XGB提升=0.24444788811646034 |
| 任务3人工抽检状态真实 | warning | 人工抽检通过率需要业务同学在 task4_consistency_audit.csv 基础上复核后填写；当前脚本只生成可抽检证据。 |
| 任务4典型案例不少于5个 | pass | 案例数=5 |
| 交付物关联行数一致 | pass | layered=70，deliverable=70，audit=70 |
| 交付物路径行数一致 | pass | layered=144，deliverable=144，audit=144 |
| 交付物资金结构行数一致 | pass | layered=2，deliverable=2，audit=2 |
| 交付物缺边恢复队列行数一致 | pass | layered=56，deliverable=56，audit=56 |

说明：warning 不代表代码不可运行，表示赛题口径或人工环节仍需在答辩中如实说明。