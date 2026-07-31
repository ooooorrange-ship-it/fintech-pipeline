# 比赛提交就绪审计

总体状态：pass

| 检查项 | 状态 | 详情 |
|---|---|---|
| 完整源代码 | pass | 脚本数=25，缺失=[] |
| Conda 环境文件 | pass | environment.yml |
| 部署文档 | pass | DEPLOYMENT.md |
| 项目说明 | pass | readme.md |
| 可复现的清洗数据输入 | pass | 缺失=[] |
| 确认嫌疑账户交易边覆盖审计 | pass | 嫌疑账户=59，有边=3，无边=56 |
| 原始/清洗交易行数一致 | pass | raw=904395，clean=904395 |
| 关键比赛输出 | pass | 缺失=[] |
| 运行依赖可导入 | pass | 缺失=[] |
| 模型文件 dynamic_xgboost | pass | models/model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json |
| 模型文件 heterophily_propagation | pass | models/model3_hetero_prop_v3_no_customer_type_strategy_A.joblib |
| 模型文件 stack_config | pass | models/model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json |
| 模型文件 logistic_regression_baseline | pass | models/baseline_logistic_regression_v1_no_customer_type_strategy_A.joblib |
| 模型文件 random_forest_baseline | pass | models/baseline_random_forest_v1_no_customer_type_strategy_A.joblib |
| 模型文件 dynamic_graph_random_forest | pass | models/model7_dynamic_graph_random_forest_v1_no_customer_type_strategy_A.joblib |
| 模型文件 graphsage | pass | models/model6_graphsage_v1_no_customer_type_strategy_A.pt |
| 模型文件 catboost_dynamic | pass | models/model9_catboost_dynamic_v1_no_customer_type_strategy_A.joblib |
| 模型文件 tgn_temporal_memory | pass | models/model10_tgn_v1_no_customer_type_strategy_A.pt |
| 模型文件 cv_bagging_experiment | pass | models/model12_cv_bagged_dynamic_v1_no_customer_type_strategy_A.joblib |
| 模型文件 previous_final_dynamic_fusion | pass | models/model8_final_dynamic_fusion_v7_strategy_A.json |
| 模型文件 final_selected_model | pass | models/model11_validation_selected_best_strategy_A.json |
| 动态 XGBoost 权重可加载 | pass | 特征数=383 |
| 图传播模型包可加载 | pass | 缺失键=[] |
| 传统基线 logistic_regression_baseline 可加载 | pass | 缺失键=[] |
| 传统基线 random_forest_baseline 可加载 | pass | 缺失键=[] |
| 传统基线 dynamic_graph_random_forest 可加载 | pass | 缺失键=[] |
| 传统基线 catboost_dynamic 可加载 | pass | 缺失键=[] |
| 传统基线 cv_bagging_experiment 可加载 | pass | 缺失键=[] |
| GraphSAGE 权重可加载 | pass | 缺失键=[] |
| 融合权重可加载 | pass | weights={'xgb': 0.0, 'graph_propagation': 0.1, 'rule': 0.0, 'dynamic': 0.9} |
| 上一版最终动态融合权重可加载 | pass | weights={'dynamic_random_forest': 0.7, 'dynamic_stack': 0.3, 'graphsage': 0.0} |
| 最终验证集选择权重可加载 | pass | weights={'model8_current_best': 0.75, 'model9_catboost': 0.2, 'model10_tgn': 0.05} |
| 模型校验清单 | pass | models/model_manifest.json |

## 数据边界

- 风险标签表没有标签确认时间，不能声称严格预测未来新增标签。
- 59 个确认嫌疑账户中只有 3 个在交易边表中有可追溯边，其余 56 个只能做缺边审计。
- 以上是数据集边界，不能通过算法伪造成真实资金链路。