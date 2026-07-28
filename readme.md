# 基于资金图谱的涉诈账户发现模型说明

## 1. 项目任务

本项目对应第五届中国研究生金融科技创新大赛“揭榜挂帅”主赛道赛题：

> 基于资金图谱的涉诈账户发现与可疑链路解释

核心目标是：

1. 基于账户交易流水构建资金图谱。
2. 识别高风险涉诈账户。
3. 输出高风险账户的关联账户和可疑资金链路。
4. 为业务同学提供可写入研判报告的结构化证据。

本项目目前完成的是一条完整的建模管线：

```text
原始账户表 + 原始交易表 + 原始标签表
        ↓
数据清洗
        ↓
标签策略构建
        ↓
按交易时间窗口划分 train / valid / test
        ↓
交易统计特征 + 时序资金流特征 + 图结构特征
        ↓
滚动动态资金图谱特征：时间分桶 + 金额分箱 + 时序模体 + 节点记忆
        ↓
动态资金图谱 XGBoost 识别模型
        ↓
轻量图传播模型和融合模型做消融验证
        ↓
高风险账户、关联账户、可疑链路解释
```

## 2. 当前模型是什么

当前最终主模型是：

> 动态资金图谱识别模型 = 滚动交易事件流图状态 + XGBoost 风险分类器

它不是纯规则模型，也不是生产级智能体，而是围绕“账户节点、转账边、时间分桶、金额分箱、风险标签”构建的动态图谱识别模型。模型按 split end 形成当前时点的资金图状态，使用 120 天滚动历史窗口，不使用 cutoff 之后的未来交易。

原因是当前确认嫌疑人只有 59 个，正样本非常少。如果直接训练重型端到端 GNN，容易因为样本太少而过拟合。相比之下，动态图事件流特征 + XGBoost 的解耦式方案更稳，也更容易解释。

融合评估器在 v6 实验中自动选择：

```text
xgb = 0.0
gnn = 0.1
rule = 0.0
dynamic = 0.9
```

也就是说，最终主模型以动态资金图谱分支为主，并用轻量图传播分支提供 0.1 的补充信息；权重是在全量账户验证集上选择的。

当前项目里还额外实现了一个轻量图传播模型：

> 轻量异配图传播特征 + 类均衡分类器

这个模型不是用来替代动态资金图谱主模型，而是用来证明资金图谱结构在弱画像场景下有增量价值。

## 3. 数据和标签

原始数据包括三张表：

| 表 | 作用 |
|---|---|
| 账户节点表 | 每个账户的基础属性和最终风险标签 |
| 交易边表 | 付款账户、收款账户、交易时间、金额等交易关系 |
| 风险标签表 | 账户对应的风险类型 |

清洗后的数据规模：

| 数据 | 数量 |
|---|---:|
| 账户数 | 11087 |
| 交易数 | 904395 |
| 标签数 | 11087 |

标签分布：

| 标签 | 含义 | 数量 |
|---|---|---:|
| 0 | 其它 | 9875 |
| 1 | 嫌疑人 | 59 |
| 2 | 受害人 | 1153 |

当前主实验使用 Strategy A：

```text
嫌疑人 = 1
其它 = 0
受害人 = -1，训练和评估时剔除
```

这样做的原因是：受害人的交易模式和正常账户不同，如果把受害人强行并入负样本，会污染“正常账户”的定义。

## 4. 训练集和测试集怎么划分

当前项目按交易时间窗口划分，不是随机划分。

| 数据集 | 交易时间范围 |
|---|---|
| train | 2025-07-01 到 2025-10-31 23:59:59 |
| valid | 2025-11-01 到 2025-11-30 23:59:59 |
| test | 2025-12-01 到 2025-12-31 23:59:59 |

注意：

> 账户没有被切开，交易是按时间切开的。

也就是说，每个账户在 train、valid、test 都会各生成一行特征。基础审计仍按时间顺序切分，动态模型实际使用的是“截至当前 split end 的 120 天滚动历史交易图”。

例如账户 4379：

```text
train 动态特征：看 2025-07-03 到 2025-10-31 的滚动历史交易
valid 动态特征：看 2025-08-02 到 2025-11-30 的滚动历史交易
test 动态特征 ：看 2025-09-02 到 2025-12-31 的滚动历史交易
```

一个重要限制是：原始标签表没有标签时间。

标签时间指的是：

> 账户在什么时候被确认为嫌疑人、受害人或其它。

由于数据里没有这个字段，所以当前实验不能严格说成“用过去交易预测未来新增风险标签”。更严谨的说法是：

> 本项目按交易时间窗口构建账户风险特征，避免交易特征的未来信息泄露；但由于风险标签表未提供标签确认时间，当前实验使用最终风险标签进行监督学习。

## 5. 特征工程

### 5.1 数据清洗特征

清洗阶段没有简单删除异常交易，而是把它们转成风险信号：

| 特征 | 含义 |
|---|---|
| self_loop | 自环交易，付款账户等于收款账户 |
| non_positive_amount | 金额小于等于 0 |
| negative_amount | 金额小于 0 |
| zero_amount | 金额等于 0 |

这样做的原因是：在反欺诈场景里，异常交易本身可能是风险信号，不能直接当脏数据删除。

### 5.2 交易统计特征

账户级交易统计包括：

| 特征类别 | 示例 |
|---|---|
| 交易次数 | 总交易数、入账次数、出账次数 |
| 金额统计 | 总金额、平均金额、最大金额、中位数金额 |
| 活跃度 | 活跃天数、活跃月份、日均交易数 |
| 方向特征 | 入账金额、出账金额、入出金额比例 |
| 异常金额 | 小额交易比例、大额交易比例、整数金额比例 |

### 5.3 时序资金流特征

这是当前模型里最贴合赛题的部分。

| 特征类别 | 业务含义 |
|---|---|
| 快进快出 | 入账后 1/6/24 小时内快速转出 |
| 多入一出 | 多个入账来源短时间汇入后集中转出 |
| 一入多出 | 一笔入账后短时间拆分转给多个账户 |
| 交易突发性 | 某几天交易量突然升高 |
| 交易间隔 | 交易之间的平均间隔、最小间隔、波动程度 |
| 小额试探 | 小额交易次数和比例 |

这些特征对应真实反诈业务里的资金流模式：

```text
受害人 / 正常账户
        ↓
    涉诈账户
        ↓
短时间转出、拆分、归集
```

### 5.4 图结构特征

资金图谱中：

```text
账户 = 节点
交易 = 有向边
```

当前提取的图特征包括：

| 特征类别 | 示例 |
|---|---|
| 度数特征 | 入度、出度、总度数 |
| 二跳邻居 | 二跳可达账户数 |
| 互惠关系 | A 转 B 且 B 转 A |
| 连通分量 | 所在资金子图大小、密度 |
| PageRank | 账户在资金网络中的重要性 |
| 邻居行为聚合 | 邻居的交易金额、交易次数、快进快出强度 |
| Node2Vec | 无监督图结构向量 |

图特征的意义是：欺诈账户不是孤立出现的，它们往往存在异常的资金连接关系。

### 5.5 动态资金图谱特征

v6 主模型新增的动态特征包括：

| 特征类别 | 含义 |
|---|---|
| 时间分桶 | 周、日、小时、夜间/上午/下午/晚间分布 |
| 金额分箱 | 使用 train 金额分位数生成金额箱，再应用到 valid/test |
| 动态图快照 | 每个时间桶里的入度、出度、活跃周数、度数波动 |
| 交易对手 churn | 前后半窗口交易对手新增、流失、稳定比例 |
| 时序资金流模体 | dyn_motif_* 快进快出、多入一出、一入多出 |
| 节点记忆 | dyn_mem_* 时间衰减入账、出账、金额、夜间交易记忆 |

节点记忆使用 1h、6h、24h、7d 四个半衰期，作用是让模型知道账户的近期交易状态，而不是只看静态全局统计。

## 6. 模型结构

当前模型分成 5 个层次：

| 模型 | 作用 |
|---|---|
| Model 0：规则模型 | 作为最低基线 |
| Model 1：XGBoost + 交易统计特征 | 验证交易行为是否有效 |
| Model 2：XGBoost + 交易统计 + 图结构特征 | 强基线和 TopK 参考模型 |
| Model 3：轻量异配图传播模型 | 图学习消融分支 |
| Model 4：融合模型 | XGB、图传播、规则分支加权融合 |
| Model 5：动态资金图谱模型 | 显式使用滚动时间窗口、金额分箱、动态边、时序模体和节点记忆 |

最终主模型选择动态资金图谱融合模型：

```text
model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A
```

该模型由全量账户验证集自动选择权重。最终权重为 `dynamic = 0.9、gnn = 0.1`，删除了 `customer_type`，并显式引入滚动时间窗口、金额分箱、动态交易关系、时序资金流模体和时间衰减节点记忆。

## 7. 当前结果

### 7.1 主模型结果

主模型：

```text
滚动动态资金图谱模型
删除 customer_type
标签策略 Strategy A
```

测试集结果：

| 指标 | 结果 |
|---|---:|
| AUC | 0.9880 |
| PR-AUC | 0.3616 |
| Top1% 命中 | 33/59 |
| Top1% 召回率 | 55.93% |
| Top5% 命中 | 57/59 |
| Top5% 召回率 | 96.61% |

以上是全量 11087 个账户的正式评估口径。训练阶段使用 Strategy A 排除受害人，但比赛排名和测试指标保留全量账户；候选池指标不能替代正式全量指标。

这个结果说明：

> 如果银行只人工复核风险分最高的前 5% 账户，动态资金图谱模型可以覆盖 59 个嫌疑人中的 57 个。

在正样本只有 59 个的极端不平衡场景下，Top5% 召回率是最重要的业务指标之一。

### 7.2 消融实验结果

消融实验 1：去掉 `customer_type`

| 模型（全量账户口径） | Test AUC | Test PR-AUC | Top5% 命中 |
|---|---:|---:|---:|
| XGBoost + 交易统计 | 0.9833 | 0.2756 | 58/59 |
| XGBoost + 交易统计 + 图结构（强基线） | 0.9855 | 0.2905 | 57/59 |
| v6 滚动动态资金图谱融合模型 | 0.9880 | 0.3616 | 57/59 |

结论：

> 全量账户口径下，动态资金图谱融合模型较强 XGBoost 基线的 PR-AUC 提升约 24.44%，达到赛题“提升不低于 20%”的要求；Top5% 召回保持 57/59。

消融实验 2：去掉 `customer_type + region_code + account_age_months`

| 模型（全量账户口径） | Test AUC | Test PR-AUC | Top5% 命中 |
|---|---:|---:|---:|
| XGBoost + 交易统计 + 图结构 | 0.7198 | 0.0212 | 9/59 |
| 纯 dyn_* 滚动动态特征 | 0.7903 | 0.0186 | 9/59 |

结论：

> 剥离 customer_type、region_code、account_age_months 后，模型明显下降，说明当前数据中的静态画像仍有较强区分力。弱画像分支仍保留交易和动态图结构，但暂未达到主模型的识别能力，因此不能把它表述成已经解决静态画像依赖。

### 7.3 是否过拟合

从验证集和测试集表现看，当前主模型没有明显过拟合。

| 数据集 | AUC | PR-AUC | Top5% 命中 |
|---|---:|---:|---:|
| valid | 0.9757 | 0.3331 | 56/59 |
| test | 0.9880 | 0.3616 | 57/59 |

验证集和测试集没有断崖式差距，因此不像典型训练集过拟合。

但需要注意：

> 模型存在一定静态画像依赖风险。

这不是普通意义上的训练集过拟合，而是模型可能学到部分静态画像规律，比如地区、开户时长等。这个问题已经通过消融实验进行了说明。

## 8. 可疑链路解释

当前解释脚本可以输出：

1. 高风险账户证据。
2. Top20 关联账户。
3. 多跳可疑路径。

正式解释与交付物使用分层解释脚本：

```bash
python src/12_layered_explainability.py --split test --tx-scope history --top-risk-active 30 --top-k-counterparties 20
```

当前测试集中 59 个确认嫌疑人里，只有 3 个在历史交易中能追到明确可解释链路。

这不是脚本错误，而是数据本身的限制：其余 56 个确认嫌疑人在当前交易边表中没有可追溯历史交易边。

因此报告中应该诚实表述：

> 动态资金图谱识别层在全量账户 Top5% 中覆盖 57/59 个嫌疑人；统计+图 XGBoost 强基线同样覆盖 57/59 个嫌疑人；路径解释层只能对存在历史交易边的账户生成真实链路证据。

已经整理出的 3 个典型案例：

| 账户 | 案例类型 | 说明 |
|---|---|---|
| 4379 | 强时序闭环 | 短时间内出现近似回流结构 |
| 1740 | 稳定星状网络 | 与少量固定对手重复交互 |
| 7265 | 边缘微小试探 | 交易规模小、连接弱，更像试探或养号行为 |

## 9. 怎么给其他成员讲

### 9.1 30 秒版本

可以这样讲：

> 我这边搭了一条完整的动态图谱风控建模管线。先清洗账户、交易和标签三张表，然后按时间 cutoff 构建 120 天滚动历史资金图，再提取交易统计、时序资金流、图结构、时间分桶、金额分箱、时序模体和节点记忆特征。最终主模型在全量账户测试口径下 AUC 为 0.9880、PR-AUC 为 0.3616，Top5% 命中 59 个嫌疑人中的 57 个；相对强 XGBoost 基线的 PR-AUC 提升为 24.44%，达到赛题的 20% 提升要求。

### 9.2 会议展示版本

建议按 5 页展示：

第 1 页：任务和数据

```text
任务：基于资金图谱发现涉诈账户并解释可疑链路
数据：账户节点表、交易边表、风险标签表
难点：正样本只有 59 个，类别极度不平衡
```

第 2 页：建模流程

```text
数据清洗
→ 标签策略
→ 时间窗口切分
→ 交易特征
→ 图结构特征
→ 120天滚动动态资金图谱特征
→ 动态资金图谱识别模型
→ 链路解释
```

第 3 页：核心特征

```text
交易统计：交易次数、金额、入出账比例
时序模体：快进快出、多入一出、一入多出、突发交易
图结构：入度、出度、二跳邻居、PageRank、邻居行为聚合
动态边特征：周级交易突发、金额分箱分布、时间段分布、交易对手变化、节点记忆
```

第 4 页：模型结果

```text
Test AUC：0.9880
Test PR-AUC：0.3616
Top1% 命中：33/59
Top5% 命中：57/59
```

第 5 页：消融和解释

```text
去掉 customer_type 后仍然很强：
说明模型不完全依赖客户类型。

去掉 customer_type + region_code + account_age_months 后下降：
说明静态画像仍有贡献。

弱画像分支的全量账户 Top5% 为 9/59：
说明仅依赖交易和图结构时仍存在明显困难，需要补充更完整的历史流水或非交易关系。
```

## 10. 推荐汇报口径

不要这样说：

```text
我们的 GNN 超过了 XGBoost。
```

因为事实不是这样。

应该这样说：

```text
在当前正样本极少的真实金融反诈数据中，重型端到端 GNN 容易过拟合；为了严格贴合赛题要求，最终采用滚动动态资金图谱识别模型。
该模型显式使用账户节点、转账边、时间分桶、金额分箱、时序资金流模体和时间衰减节点记忆；验证集融合器选择 dynamic=0.9、gnn=0.1，说明动态资金图谱是主分支，轻量图传播提供补充。
```

这个表述更严谨，也更适合答辩。

## 11. 主要运行命令

```bash
# 1. 数据清洗
python src/01_data_cleaning.py

# 2. 标签构建
python src/02_label_builder.py

# 3. 交易统计和时序特征
python src/03_features_stat.py

# 4. 图结构特征和 Node2Vec
python src/04_features_graph.py

# 5. 主模型：删除 customer_type
python src/05_model_xgb.py --drop-customer-type --experiment-suffix v2_no_customer_type

# 6. 弱画像消融：删除 customer_type、region_code、account_age_months
python src/05_model_xgb.py --drop-static-profile --experiment-suffix v2_txn_graph_only

# 7. 轻量图传播模型
python src/06_model_gnn.py --drop-customer-type --experiment-suffix v3_no_customer_type
python src/06_model_gnn.py --drop-static-profile --experiment-suffix v3_txn_graph_only

# 8. 融合模型
python src/08_model_stack.py --experiment-suffix v3_no_customer_type
python src/08_model_stack.py --experiment-suffix v3_txn_graph_only --xgb-pred outputs/predictions/model2_xgb_stat_graph_v2_txn_graph_only_strategy_A.csv --gnn-pred outputs/predictions/model3_hetero_prop_v3_txn_graph_only_strategy_A.csv --rule-pred outputs/predictions/model0_rule_v2_txn_graph_only_strategy_A.csv

# 9. 动态资金图谱特征和模型
python src/10_features_dynamic_graph.py
python src/11_model_dynamic_graph_xgb.py --drop-customer-type --experiment-suffix v6_rolling_memory_dynamic_no_customer_type
python src/11_model_dynamic_graph_xgb.py --dynamic-only --drop-static-profile --experiment-suffix v6_rolling_memory_dynamic_only
python src/08_model_stack.py --experiment-suffix v6_rolling_memory_dynamic_no_customer_type --xgb-pred outputs/predictions/model2_xgb_stat_graph_v2_no_customer_type_strategy_A.csv --gnn-pred outputs/predictions/model3_hetero_prop_v3_no_customer_type_strategy_A.csv --rule-pred outputs/predictions/model0_rule_v2_no_customer_type_strategy_A.csv --dynamic-pred outputs/predictions/model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv

# 10. 可疑链路解释
python src/07_explain_links.py --split test --top-n 5 --top-k-counterparties 20 --tx-scope history --confirmed-only

# 11. 分层解释优化：确认嫌疑人缺边审计 + 有边高风险账户研判报告
python src/12_layered_explainability.py --split test --tx-scope history --top-risk-active 30 --top-k-counterparties 20

# 12. 综合网页：Top30 动态图谱 + 59账户审计 + Top30巡检 + 研判报告
python src/09_dynamic_graph_viz.py --split test --top-n 30 --top-k-counterparties 20 --window monthly

# 13. 比赛交付物
python src/09_build_deliverables.py
python src/13_final_project_audit.py
```

## 12. 文件对应关系

| 文件 | 作用 |
|---|---|
| `config.py` | 全局路径、字段名、时间窗口配置 |
| `src/01_data_cleaning.py` | 原始数据清洗 |
| `src/02_label_builder.py` | 三套标签策略 |
| `src/03_features_stat.py` | 交易统计和时序资金流特征 |
| `src/04_features_graph.py` | 图结构特征和 Node2Vec |
| `src/05_model_xgb.py` | XGBoost 主模型 |
| `src/06_model_gnn.py` | 轻量异配图传播模型 |
| `src/07_explain_links.py` | 高风险账户和可疑链路解释 |
| `src/08_model_stack.py` | 多模型融合 |
| `src/09_dynamic_graph_viz.py` | 滚动动态资金图谱 HTML 展示和窗口证据报告 |
| `src/09_build_deliverables.py` | 比赛任务交付物生成 |
| `src/10_features_dynamic_graph.py` | 滚动动态资金图谱特征，含时间分桶、金额分箱、时序模体和节点记忆 |
| `src/11_model_dynamic_graph_xgb.py` | 动态资金图谱 XGBoost 模型 |
| `src/12_layered_explainability.py` | 分层解释优化：缺边审计、风险巡检队列、链路解释和研判报告 |
| `src/13_final_project_audit.py` | 交付前全量覆盖、泄露、解释结果和任务指标一致性审计 |
| `outputs/features/` | 特征宽表 |
| `outputs/metrics/` | 模型评估指标 |
| `outputs/predictions/` | 账户风险分数 |
| `outputs/explanations/` | 关联账户和链路解释结果 |

## 13. 最终结论

当前最贴合赛题要求的最终模型是：

```text
model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A
```

核心结论：

1. 滚动动态资金图谱模型显式使用账户节点、转账关系、时间分桶、金额分箱、风险标签、时序资金流模体和节点记忆。
2. 动态资金图谱融合模型全量账户 Test AUC 为 0.9880，PR-AUC 为 0.3616，Top5% 命中 57/59 个嫌疑人。
3. 弱画像消融显示模型确实存在静态画像依赖。
4. 统计+图 XGBoost 强基线全量账户 Top5% 命中 57/59；主模型 PR-AUC 相对其提升 24.44%，达到严格的 20% 提升要求。
5. 由于标签表没有标签时间，当前实验应表述为“按交易时间窗口构建风险识别模型”，不要严格说成“预测未来新增风险标签”。

## 14. 比赛任务目标交付物

已按赛题 4 个任务目标生成集中交付目录：

```text
outputs/deliverables/
```

最重要的检查入口：

| 文件 | 作用 |
|---|---|
| `task_completion_checklist.md` | 对照赛题 4 个任务逐项检查完成情况 |
| `technical_problem_solution.md` | 对照 4 个技术攻关问题逐项说明解决方案 |
| `task1_dynamic_graph_window_definition.json` | 动态资金图谱窗口、节点、边、标签定义 |
| `task1_field_dictionary.csv` | 字段字典 |
| `task1_time_split_leakage_audit.json` | 时间切分和泄露审计 |
| `task1_graph_statistics_by_split.csv` | 图统计报告 |
| `task2_model_metrics_summary.csv` | 模型指标和基线对比 |
| `task2_requirement_audit.json` | AUC、PR-AUC、Top5% 是否达标 |
| `task3_top20_associations.csv` | 高风险账户 Top20 关联账户 |
| `task3_suspicious_paths.csv` | 多跳可疑路径 |
| `task3_fund_flow_structures.csv` | 资金闭环、汇聚、分散结构 |
| `docs/task3_typical_cases.md` | 5 个典型案例分析 |
| `docs/task3_link_visualization_samples.md` | 链路可视化样例 |
| `docs/task4_judgement_report_template.md` | 辅助研判报告模板 |
| `docs/task4_judgement_report_samples.md` | 辅助研判报告样例 |
| `docs/final_project_audit.md` | 交付前自动一致性审计，硬错误为 0 才算通过 |

生成命令：

```bash
python src/09_build_deliverables.py
python src/13_final_project_audit.py
```

注意：任务1中的未来交易泄露率已审计为 0；但原始风险标签表没有标签时间，因此不能严格验证“未来新增标签”口径，只能保证标签字段不进入模型特征。

## 15. 滚动动态资金图谱如何展示

新增脚本：

```bash
python src/09_dynamic_graph_viz.py --split test --top-n 30 --top-k-counterparties 20 --window monthly
```

输出目录：

```text
outputs/dynamic_graph/
```

说明性 Markdown 统一输出到项目同级的 `docs/`，`outputs/` 只保留可供程序读取的 CSV/JSON 和网页文件。

核心入口：

| 文件 | 作用 |
|---|---|
| `index.html` | 可直接打开的滚动动态资金图谱页面 |
| `docs/top_accounts_dynamic_report.md` | 可写进报告/答辩稿的图谱展示说明 |
| `rolling_window_stats.csv` | 每个高风险账户在每个窗口里的交易统计和风险信号 |
| `top_accounts_dynamic_edges.csv` | 每个窗口中实际展示的聚合资金边 |
| `top_accounts_dynamic_nodes.csv` | 每个窗口中实际展示的账户节点、标签和模型分数 |
| `dynamic_graph_data.json` | HTML 页面使用的结构化图谱数据 |

展示逻辑：

```text
账户 = 节点
转账 = 有向边
月份 = 滚动快照窗口
金额分箱 = 边的金额等级
模型风险分 = 根账户风险强度
短时闭环、快进快出、多入一出 = 右侧研判证据
```

当前正式页面展示 Top30 高风险巡检账户，并同时包含 59 个确认嫌疑账户审计、分层解释结果和研判报告。若只做三个真实链路案例复盘，可额外使用 `--confirmed-only`；其中账户 4379 在 2025-11 窗口出现 24 小时内短时闭环，最短回流间隔为 21 秒，是最适合作为答辩主案例的动态图谱证据。

如果要模拟业务上线后的“未知账户风险巡检”，去掉 `--confirmed-only` 即可：

```bash
python src/09_dynamic_graph_viz.py --split test --top-n 30 --top-k-counterparties 20 --window monthly
```

两种口径的区别：

| 口径 | 用途 |
|---|---|
| 带 `--confirmed-only` | 典型案例复盘，方便展示模型是否抓住真实嫌疑账户的资金链路 |
| 不带 `--confirmed-only` | 业务巡检口径，展示模型当前认为最需要人工复核的 Top 风险账户 |

答辩推荐表述：

> 我们不是展示全量 90 万条交易边，而是围绕模型输出的高风险账户构建滚动 ego 子图。每个窗口只使用 cutoff 之前的历史交易，展示账户节点、资金流向、金额分箱、Top 关联账户，以及短时闭环、快进快出、多入一出等时序资金流模体，从而把黑盒风险分数转化为可追溯的辅助研判证据。

## 16. 59 个嫌疑人只有 3 个有真实链路怎么办

这个问题不能靠“换一个路径搜索算法”硬解决。脚本审计结果显示：

```text
确认嫌疑账户总数：59
有历史交易边的确认嫌疑账户：3
无历史交易边的确认嫌疑账户：56
```

也就是说，其余 56 个账户在当前交易边表中没有可追溯资金边。这个问题不能通过换路径算法解决；如果强行生成链路，就是伪造解释，反而会破坏报告可信度。

因此项目新增了分层解释和缺边恢复脚本：

```bash
python src/12_layered_explainability.py --split test --tx-scope history --top-risk-active 30 --top-k-counterparties 20
```

结构化结果目录：

```text
outputs/explanations/layered/
```

说明性审计和研判 Markdown 位于同级 `docs/`，对应 CSV/JSON 仍保留在 `outputs/explanations/layered/`。

核心文件：

| 文件 | 作用 |
|---|---|
| `docs/confirmed_suspect_explainability_audit.md` | 59 个确认嫌疑账户的解释覆盖审计 |
| `confirmed_suspect_explainability_audit.csv` | 每个确认嫌疑账户的解释等级、历史交易数、缺边原因 |
| `risk_review_queue_active_accounts.csv` | 模型发现的有交易边高风险账户巡检队列 |
| `risk_review_queue_top20_associations.csv` | 有边高风险账户 Top20 关联账户 |
| `risk_review_queue_suspicious_paths.csv` | 有边高风险账户可疑多跳路径 |
| `risk_review_queue_fund_flow_structures.csv` | 资金闭环、汇聚、分散结构 |
| `outputs/explanations/layered/suspect_link_recovery_queue.csv` | 56 个缺边账户的补数查询、恢复优先级和预期链路类型 |
| `docs/layered_judgement_report_samples.md` | 分层辅助研判报告样例 |
| `layered_explainability_coverage.json` | 分层解释覆盖统计 |

分层解释结果：

| 解释对象 | A 链路证据型 | B 直接关联型 | D 数据缺边型 |
|---|---:|---:|---:|
| 59 个确认嫌疑账户 | 1 | 2 | 56 |
| 模型 Top 有边高风险巡检账户 30 个 | 9 | 21 | 0 |

这套优化后的答辩口径是：

> 对已确认的 59 个嫌疑账户，我们首先进行可解释性覆盖审计。审计发现只有 3 个账户在当前脱敏交易边表中存在历史交易边，因此只有这 3 个账户能生成真实链路解释；其余 56 个账户进入缺边恢复队列。对这 56 个账户，系统完整保留模型风险分、节点画像、边覆盖状态、恢复优先级、补数查询和补数后应重建的链路类型；不会把不存在的账户关系写成路径。与此同时，系统从模型输出中筛选有历史交易边的高风险账户，生成 Top20 关联账户、多跳可疑路径、资金闭环/汇聚/分散结构和辅助研判报告。这样把“确认嫌疑人复盘”和“业务主动巡检”分成两个真实可核验的证据层。

向业务同学提出的补数建议：

```text
1. 按 `suspect_link_recovery_queue.csv` 中的 `required_query` 补充 56 个缺边嫌疑账户的完整历史流水。
2. 补数后重新运行 `src/12_layered_explainability.py`，系统会自动把账户从 D 级恢复为 A/B/C 级并重新生成链路。
3. 补充标签确认时间，才能严格评估“过去交易预测未来新增风险标签”。
4. 补充设备/IP/开户批次/案件号等非交易关系，可构建异构图解释隐性协同网络。
5. 对模型 Top 有边高风险账户进行人工抽检，记录解释通过率。
```
