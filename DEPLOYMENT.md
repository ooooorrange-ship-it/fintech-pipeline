# 部署与复现说明

## 1. 交付内容

| 类别 | 路径 | 说明 |
|---|---|---|
| 源代码 | `config.py`、`src/` | 清洗、特征、建模、融合、解释、展示和审计代码 |
| 环境 | `environment.yml` | Conda 依赖和版本约束 |
| 模型 | `models/` | 动态树模型、CatBoost、TGN、融合权重及 SHA-256 清单 |
| 中间数据 | `outputs/clean/`、`outputs/features/` | 可直接用于复现建模的清洗数据和特征 |
| 评估结果 | `outputs/metrics/`、`outputs/predictions/` | 指标、特征重要性和账户风险分 |
| 比赛交付物 | `deliverables/` | 四项任务对应的结构化证据 |
| 动态页面 | `outputs/dynamic_graph/index.html` | 滚动动态资金图谱与研判展示 |

## 2. 环境安装

推荐使用 Miniforge/Conda：

```bash
conda env create -f environment.yml
conda activate fintech09
```

已测试的核心环境为 Python 3.12、pandas 2.2、scikit-learn 1.9、XGBoost 3.3、CatBoost 1.2.10、PyTorch 2.13 和 PyTorch Geometric 2.8。

## 3. 数据放置

完整从原始表复现时，项目根目录需包含：

```text
09-智能风控与量化建模赛道-江苏银行-基于资金图谱的涉诈账户发现与可疑链路解释/
├── 账户节点表.xlsx
├── 交易边表.xlsx
└── 风险标签表.xlsx
```

原始银行数据默认不上传公开 Git 仓库。比赛离线提交包应按主办方规则携带数据；如评委使用主办方同版本数据，保持上述文件名即可。

若只复现特征、模型和指标，可直接使用已交付的 `outputs/clean/` 三张清洗表，从下一节第 2 步开始。

如需从赛题服务器重新下载原始数据，可运行 `python download_data.py`，并提前通过环境变量 `SCOW_DOWNLOAD_PASSWORD` 提供下载凭据；公开仓库不保存明文密码。

## 4. 完整复现顺序

```bash
# 1. 原始数据清洗
python src/01_data_cleaning.py

# 2. 标签、统计特征、静态图特征
python src/02_label_builder.py
python src/03_features_stat.py
python src/04_features_graph.py

# 3. 规则和静态 XGBoost 对照分支
python src/05_model_xgb.py --drop-customer-type --experiment-suffix v2_no_customer_type

# 4. 轻量图传播分支及其模型包
python src/06_model_gnn.py --drop-customer-type --experiment-suffix v3_no_customer_type

# 5. 滚动动态图特征及 XGBoost 权重
python src/10_features_dynamic_graph.py
python src/11_model_dynamic_graph_xgb.py --drop-customer-type --experiment-suffix v6_rolling_memory_dynamic_no_customer_type

# 6. 统一传统基线和真实 GraphSAGE 消融
python src/15_model_traditional_baselines.py
python src/16_model_graphsage.py

# 7. model8_final_dynamic_fusion_v7_strategy_A 旧版动态融合权重和预测
python src/08_model_stack.py \
  --experiment-suffix v6_rolling_memory_dynamic_no_customer_type \
  --xgb-pred outputs/predictions/model2_xgb_stat_graph_v2_no_customer_type_strategy_A.csv \
  --graph-propagation-pred outputs/predictions/model3_hetero_prop_v3_no_customer_type_strategy_A.csv \
  --rule-pred outputs/predictions/model0_rule_v2_no_customer_type_strategy_A.csv \
  --dynamic-pred outputs/predictions/model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv
python src/18_model_final_fusion.py

# 8. 新模型对照：CatBoost 和轻量 TGN
python src/19_model_catboost.py
python src/20_model_tgn.py

# 9. 只用验证集选择最终冠军模型
python src/21_select_best_model.py

# 10. 五折账户级过拟合审计和正则化 Bagging 对照
python src/22_cross_validation_overfit_audit.py
python src/23_model_cv_bagging.py
python src/24_model_overfit_guardrails.py
python src/25_model_rule_aware_calibrator.py

# 11. 统一模型对比、链路解释、页面和比赛交付物
python src/17_compare_models.py
python src/12_layered_explainability.py --split test --tx-scope history --top-risk-active 30 --top-k-counterparties 20
python src/09_dynamic_graph_viz.py --split test --top-n 30 --top-k-counterparties 20 --window monthly
python src/09_build_deliverables.py

# 12. 最终审计
python src/14_submission_audit.py
python src/13_final_project_audit.py
```

注意：当前最终主模型（model11_validation_selected_best_strategy_A）由验证集选择 `model8_final_dynamic_fusion_v7_strategy_A=0.75、model9_catboost_dynamic_v1_no_customer_type_strategy_A=0.20、model10_tgn_v1_no_customer_type_strategy_A=0.05`。model8_final_dynamic_fusion_v7_strategy_A 内部为动态图 RandomForest 0.7、v6 动态融合 0.3、GraphSAGE 0。TGN 和 GraphSAGE 都保留为可复现的动态图模型对照分支。Model12 是五折正则化 Bagging 对照模型，用于检查过拟合和账户级泛化风险，不替代当前最终主模型。Model13 是过拟合护栏配置，验证集结果选择 `最终主模型=1.0、Model12=0.0`，因此主模型保持最终主模型。Model14 是规则感知校准层，验证集结果选择 `最终主模型=1.0、rule_score=0.0`，因此规则层只作为解释锚点，不改变最终风险分。

## 5. 快速验证

不重新训练时，可直接检查已交付制品：

```bash
python src/14_submission_audit.py
python src/13_final_project_audit.py
```

两个命令均应输出 `status: pass` 且 `error_count: 0`。数据边界和人工抽检事项会以 warning/已知局限记录，不会被伪装成已解决。

## 6. 模型制品

| 文件 | 作用 |
|---|---|
| `models/model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json` | 动态资金图谱 XGBoost 权重 |
| `models/model3_hetero_prop_v3_no_customer_type_strategy_A.joblib` | 图传播分类器、标准化器、缺失值处理器和特征列 |
| `models/model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json` | v6 动态融合组件权重与归一化规则 |
| `models/baseline_logistic_regression_v1_no_customer_type_strategy_A.joblib` | 逻辑回归基线及特征定义 |
| `models/baseline_random_forest_v1_no_customer_type_strategy_A.joblib` | 随机森林基线及特征定义 |
| `models/model6_graphsage_v1_no_customer_type_strategy_A.pt` | PyTorch Geometric GraphSAGE 权重、结构参数和标准化参数 |
| `models/model7_dynamic_graph_random_forest_v1_no_customer_type_strategy_A.joblib` | 动态图 RandomForest 模型和特征定义 |
| `models/model8_final_dynamic_fusion_v7_strategy_A.json` | 最终验证集选权融合配置 |
| `models/model9_catboost_dynamic_v1_no_customer_type_strategy_A.joblib` | CatBoost 动态特征模型 |
| `models/model10_tgn_v1_no_customer_type_strategy_A.pt` | 轻量 TGN 时间事件流模型 |
| `models/model11_validation_selected_best_strategy_A.json` | 最终主模型（含 model8_final_dynamic_fusion_v7_strategy_A/CatBoost/TGN 权重）验证集选择配置 |
| `models/model12_cv_bagged_dynamic_v1_no_customer_type_strategy_A.joblib` | 五折正则化 Bagging 对照模型 |
| `models/model13_guardrailed_final_strategy_A.json` | 过拟合护栏最终配置 |
| `models/model14_rule_aware_guardrailed_strategy_A.json` | 规则感知校准配置 |
| `models/model_manifest.json` | 文件大小、SHA-256 和运行库版本 |

## 7. 结果查看

- 模型指标：`deliverables/task2_requirement_audit.json`
- 链路解释：`deliverables/task3_top20_associations.csv` 和 `task3_suspicious_paths.csv`
- 人工抽检：`deliverables/task3_manual_review_form.csv`
- 研判一致性：`deliverables/task4_consistency_audit.csv`
- 交互页面：直接打开 `outputs/dynamic_graph/index.html`

## 8. 必须如实说明的限制

1. 标签表没有标签确认时间，所以不能严格声称“过去交易预测未来新增风险标签”。
2. 59 个确认嫌疑账户中只有 3 个在原始交易表中有边，其余 56 个不能生成真实资金链路。
3. train/valid/test 使用不同交易时间窗口，但包含相同的账户节点和最终标签；当前评估是时点风险识别，不是未见账户冷启动评估。
4. 任务 3 的确认风险关联命中率当前为 0%，需由业务成员在 `task3_manual_review_form.csv` 中完成 5 个真实有边案例的抽检，人工通过率达到 70% 后才能声称该项指标达标。
