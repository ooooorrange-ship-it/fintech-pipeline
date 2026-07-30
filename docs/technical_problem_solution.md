# 技术攻关问题解决说明

## 问题1：动态资金图谱构建与时间切分

- 状态：solved_with_data_limitation
- 解决方案：
  - 账户作为节点、转账作为有向边，按 train/valid/test cutoff 构建滚动历史动态图谱样本。
  - 边样例包含资金流向、时间桶、金额分箱、自环/非正金额异常标记。
  - 金额分箱边界只由 train 窗口计算，再应用到 valid/test，动态特征只使用截至 split end 的交易，避免未来交易泄露。
- 证据文件：
  - `task1_dynamic_graph_window_definition.json`
  - `task1_time_split_leakage_audit.json`
  - `task1_graph_statistics_by_split.csv`
  - `task1_graph_samples/*.csv`
- 未来交易泄露率：0.0000
- 未来标签泄露审计：原始标签表没有标签时间，无法严格验证未来新增标签；当前保证标签字段不进入模型特征。

## 问题2：涉诈账户风险识别

- 状态：solved
- 解决方案：
  - 采用 Strategy A 过滤受害人进行训练，同时保留全量账户排名，避免候选池指标偏乐观。
  - XGBoost 使用类别权重处理 59 个嫌疑人导致的极端不平衡。
  - 构造滚动动态图快照、时间桶、金额分箱、时序资金流模体和时间衰减节点记忆，并补充轻量异配图传播和融合消融。
- 证据文件：
  - `task2_model_metrics_summary.csv`
  - `task2_requirement_audit.json`
  - `outputs/metrics/*feature_importance.csv`
- 主模型：`model8_final_dynamic_fusion_v7_strategy_A`
- Test AUC：0.9866
- Test PR-AUC：0.6938
- Test Top5% 召回：96.61%

## 问题3：可疑关联账户和资金链路解释

- 状态：solved_with_observed_edge_limitation_and_recovery_queue
- 解决方案：
  - 围绕高风险账户输出 Top20 关联账户。
  - 挖掘 24 小时内 root-mid-out、in-root-out 等多跳可疑路径。
  - 支持输出多入一出、一入多出、闭环回流等资金结构；当前数据实际命中闭环回流结构。
  - 对没有任何入边或出边的确认嫌疑账户生成缺边恢复队列，记录模型分、节点画像、补数查询和补数后应重建的链路类型。
- 证据文件：
  - `task3_top20_associations.csv`
  - `task3_suspicious_paths.csv`
  - `task3_fund_flow_structures.csv`
  - `task3_link_recovery_queue.csv`
  - `docs/task3_link_visualization_samples.md`
  - `docs/task3_typical_cases.md`
- 局限：当前脱敏交易边表中 56 个确认嫌疑账户没有任何入边或出边，因此无法在现有数据范围内恢复真实资金链路；项目已将其转为可执行的缺边恢复队列，而不是伪造路径。
- Top20 关联账户行数：71
- 多跳路径行数：186
- 汇聚/分散结构行数：2
- 缺边恢复队列行数：56
- 当前真实链路覆盖率：3/59（5.08%）
- 处理方式：对缺边账户输出模型分数、节点画像、缺边状态和补数查询；补数后重新运行解释脚本即可自动恢复链路分层。

## 问题4：辅助研判证据可信可用

- 状态：solved
- 解决方案：
  - 研判报告样例至少引用模型分数和账户特征两类证据。
  - 存在交易边时追加关联账户或多跳路径证据。
  - 对无可追溯交易边账户明确写出局限，避免无依据结论。
- 证据文件：
  - `docs/task4_judgement_report_template.md`
  - `docs/task4_judgement_report_samples.md`
  - `task4_evidence_field_dictionary.csv`
  - `task4_consistency_audit.csv`
- 典型案例数：5
- 样例证据完整率：100.00%
