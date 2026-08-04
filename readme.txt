# 参赛作品说明

第五届中国研究生金融科技创新大赛“揭榜挂帅”主赛道
赛题：基于资金图谱的涉诈账户发现与可疑链路解释（江苏银行）
技术路线：滚动动态资金图谱特征 + 验证集选择融合（Model11）+ 分层证据研判

## 一、项目简介

本项目以脱敏账户、交易边和风险标签为核心，构建按时间窗口演化的动态资金图谱，完成四项任务：

1. 构建无未来泄露的动态资金图谱样本；
2. 面向极端不平衡样本的涉诈账户风险识别；
3. 高风险账户关联账户、多跳路径与资金结构解释；
4. 生成可追溯、可复核的辅助研判报告。

对于当前数据中 59 个确认嫌疑账户只有 3 个存在交易边的客观限制，项目不伪造路径，而是采用双层研判架构：有交易边的账户输出真实资金链路，无交易边的账户输出缺边审计和补数恢复队列。

## 二、核心成果与指标

| 任务 | 交付内容 | 关键指标 |
|---|---|---|
| 任务1 | 动态资金图谱样本、字段字典、时间切分审计 | 未来交易泄露率为 0 |
| 任务2 | Model11 融合模型与基线对比 | Test AUC 0.9858、PR-AUC 0.8049、Top5% 覆盖 56/59 |
| 任务3 | Top20 关联、多跳路径、资金结构、典型案例 | 人工抽检 5/5 通过 |
| 任务4 | 研判报告模板、样例、证据字典、一致性审计 | 报告至少引用两类可追溯证据 |

Model11 由验证集选择得到：0.75 x Model8 + 0.20 x CatBoost + 0.05 x 轻量 TGN。相对最强传统 RandomForest 基线，PR-AUC 提升约 56.71%。

## 三、四项评审材料

1. 精益画布：由团队按大赛模板填写，随作品单独提交。
2. 可运行项目成果：本仓库全部源代码、模型、清洗数据、环境文件和部署文档。
3. 技术文档：`技术文档-lxq-fixed.docx`，以及 `docs/` 下全部 Markdown 文档。
4. 原创性声明：由团队按大赛模板签字后单独提交。

## 四、目录结构

- `config.py`：全局路径、时间切分和字段配置
- `src/`：全部源代码，按 01-26 编号运行
- `docs/`：技术文档、审计报告、案例说明、演示文档和图片
- `models/`：训练好的模型权重、元数据和 SHA-256 清单
- `outputs/`：清洗数据、特征、标签、指标、预测、解释和动态网页
- `deliverables/`：四项任务最终交付物，提交时上传此目录
- `5个高风险嫌疑账户交易特征记录表/`：业务侧人工画像与评价材料
- `environment.yml`：Conda 环境依赖
- `DEPLOYMENT.md`：部署与完整复现步骤
- `readme.md`：详细技术说明
- `download_data.py`：赛题数据下载脚本，密码通过环境变量 `SCOW_DOWNLOAD_PASSWORD` 提供

## 五、环境安装

```bash
conda env create -f environment.yml
conda activate fintech09
```

已测试环境：Python 3.12、pandas 2.2、scikit-learn 1.9、XGBoost 3.3、CatBoost 1.2.10、PyTorch 2.13、PyTorch Geometric 2.8。

## 六、快速验证（不重新训练）

```bash
python src/14_submission_audit.py
python src/13_final_project_audit.py
```

两个命令均应输出 `status: pass`、`error_count: 0`。`14_submission_audit.py` 会检查源码完整性、依赖、模型可加载性和关键交付物；`13_final_project_audit.py` 会检查时间泄露、指标达标和解释一致性。

## 七、完整复现

完整命令见 `DEPLOYMENT.md`，核心顺序如下：

1. 数据清洗：`python src/01_data_cleaning.py`
2. 标签与特征：`02_label_builder.py`、`03_features_stat.py`、`04_features_graph.py`
3. 静态基线与消融：`05_model_xgb.py`、`06_model_gnn.py`
4. 动态资金图谱：`10_features_dynamic_graph.py`、`11_model_dynamic_graph_xgb.py`
5. 对照模型：`15_model_traditional_baselines.py`、`16_model_graphsage.py`、`19_model_catboost.py`、`20_model_tgn.py`
6. 融合与选择：`18_model_final_fusion.py`、`21_select_best_model.py`
7. 过拟合审计：`22_cross_validation_overfit_audit.py`、`23_model_cv_bagging.py`、`24_model_overfit_guardrails.py`、`25_model_rule_aware_calibrator.py`
8. 解释与交付：`12_layered_explainability.py`、`09_dynamic_graph_viz.py`、`09_build_deliverables.py`
9. 最终审计：`14_submission_audit.py`、`13_final_project_audit.py`

## 八、网页查看与使用

网页位于 `outputs/dynamic_graph/index.html`，数据已内嵌，可直接双击打开；首次打开建议联网以加载样式。

也可启动本地服务：

```bash
cd outputs/dynamic_graph
python -m http.server 8080
```

浏览器访问 `http://localhost:8080/`。

网页包含 5 个页面：

- 动态图谱：查看滚动子图、时间窗口、窗口轨迹和 Top 交易边
- 59账户审计：按 A/B/C/D 分层筛选和搜索 59 个确认嫌疑账户
- 缺边恢复：查看 56 个缺边账户的补数恢复队列
- Top30风险巡检：查看有交易边高风险账户的关联、路径和资金结构
- 辅助研判报告：查看可复制的结构化研判报告

详细使用说明见 `docs/演示文档.md`。

## 九、最终交付物（deliverables）

`deliverables/` 为提交上传目录，包含：

- 任务1：`task1_dynamic_graph_window_definition.json`、`task1_field_dictionary.csv`、`task1_graph_statistics_by_split.csv`、`task1_time_split_leakage_audit.json`、`task1_graph_samples/`
- 任务2：`task2_requirement_audit.json`、`task2_model_metrics_summary.csv`
- 任务3：`task3_top20_associations.csv`、`task3_suspicious_paths.csv`、`task3_fund_flow_structures.csv`、`task3_link_recovery_queue.csv`、`task3_typical_cases.csv`、`task3_manual_review_form.csv`、`task3_manual_review_form.xlsx`、`task3_task4_explanation_audit.json`
- 任务4：`task4_evidence_field_dictionary.csv`、`task4_consistency_audit.csv`
- 审计：`data_edge_coverage_audit.json`、`final_project_audit.json`、`submission_readiness_audit.json`、`deliverable_manifest.json`、`task_completion_checklist.csv`、`technical_problem_solution_audit.json`

每个文件的用途见 `deliverables/readme.txt`。

## 十、已知数据边界

1. 原始风险标签表没有标签确认时间，因此不能严格声称“过去交易预测未来新增标签”，当前已保证标签字段不进入特征。
2. 59 个确认嫌疑账户中只有 3 个（1740、4379、7265）在交易边表中有边，其余 56 个没有入边或出边；这是原始抽样数据的事实，不能通过路径算法恢复真实链路。
3. 56 个缺边账户的高分主要来自静态画像与图基座特征，账户级五折验证 Top5% 召回约 44%，应作为补数复核线索，不作为已证实资金链路。
4. 任务3 关联命中率受缺边限制不适用，采用赛题允许的人工抽检替代指标，当前 5/5 通过。
