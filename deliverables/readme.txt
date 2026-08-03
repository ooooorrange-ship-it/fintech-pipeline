# deliverables 目录说明

本目录是比赛最终交付目录，与 outputs/ 分离：outputs/ 存放可复现建模的中间产物，本目录只放四项攻关任务的结构化交付物与审计结果。提交作品时上传本目录即可。

任务1（动态资金图谱样本构建）：
- task1_dynamic_graph_window_definition.json：窗口定义
- task1_field_dictionary.csv：字段字典
- task1_graph_statistics_by_split.csv：分切分图统计
- task1_time_split_leakage_audit.json：未来交易泄露审计
- task1_graph_samples/：train/valid/test 节点、边、标签样例

任务2（涉诈账户风险识别）：
- task2_requirement_audit.json：赛题指标达成审计
- task2_model_metrics_summary.csv：模型指标汇总

任务3（关联账户与资金链路解释）：
- task3_top20_associations.csv：Top20 关联账户
- task3_suspicious_paths.csv：多跳可疑路径
- task3_fund_flow_structures.csv：资金汇聚/分散/闭环结构
- task3_link_recovery_queue.csv：缺边账户补数恢复队列
- task3_typical_cases.csv：典型案例结构化表
- task3_manual_review_form.csv / task3_manual_review_form.xlsx：人工抽检表
- task3_task4_explanation_audit.json：任务3/4 解释审计

任务4（辅助研判报告生成）：
- task4_evidence_field_dictionary.csv：证据字段说明
- task4_consistency_audit.csv：报告一致性审计

全局审计：
- data_edge_coverage_audit.json：原始交易边覆盖审计
- deliverable_manifest.json：交付物清单
- final_project_audit.json：最终一致性审计
- submission_readiness_audit.json：提交就绪审计
- task_completion_checklist.csv：任务完成清单
- technical_problem_solution_audit.json：技术问题解决方案审计
