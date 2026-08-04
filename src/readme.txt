# src 目录说明

本目录是全部参赛源代码，按流水线编号运行。完整顺序见根目录 DEPLOYMENT.md。

- 01_data_cleaning.py：读取三张原始表，清洗、打标脏数据并生成清洗表
- 02_label_builder.py：生成三种标签策略（A/B/C）标签宽表
- 03_features_stat.py：账户交易统计与异常时序特征
- 04_features_graph.py：图拓扑特征与 Node2Vec
- 05_model_xgb.py：静态 XGBoost 基线（Model0/1/2）与消融
- 06_model_gnn.py：轻量图传播分支（Model3）与堆叠配置
- 07_explain_links.py：早期 Top-N 链路解释脚本
- 08_model_stack.py：v6 动态特征堆叠融合（Model4）
- 09_dynamic_graph_viz.py：滚动动态资金图谱页面数据生成
- 09_build_deliverables.py：生成四项任务交付物与审计 JSON
- 10_features_dynamic_graph.py：120 天滚动动态资金图谱特征
- 11_model_dynamic_graph_xgb.py：动态图 XGBoost（Model5）
- 12_layered_explainability.py：分层解释，含缺边审计、巡检队列、路径解释
- 13_final_project_audit.py：交付前最终一致性审计
- 14_submission_audit.py：提交就绪审计（源码、模型、依赖、可加载性）
- 15_model_traditional_baselines.py：逻辑回归与 RandomForest 基线
- 16_model_graphsage.py：真实 GraphSAGE 消融
- 17_compare_models.py：模型对比汇总
- 18_model_final_fusion.py：动态图树融合模型（Dynamic Graph Tree Ensemble）
- 19_model_catboost.py：CatBoost 动态特征模型（Model9）
- 20_model_tgn.py：轻量 TGN 时间事件流模型（Model10）
- 21_select_best_model.py：仅用验证集选择图-树-时序融合模型权重
- 22_cross_validation_overfit_audit.py：账户级五折过拟合审计
- 23_model_cv_bagging.py：五折 Bagging 对照模型（Model12）
- 24_model_overfit_guardrails.py：过拟合护栏（Model13）
- 25_model_rule_aware_calibrator.py：规则感知校准（Model14）
- 26_data_edge_coverage_audit.py：原始数据 56/59 缺边事实核验
