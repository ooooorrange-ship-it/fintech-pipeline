# outputs/features 目录说明

本目录存放四类特征宽表及对应报告，train/valid/test 按时间切分。

- stat_features_train.csv、stat_features_valid.csv、stat_features_test.csv：交易统计与异常时序特征
- graph_features_train.csv、graph_features_valid.csv、graph_features_test.csv：图拓扑特征
- node2vec_features_train.csv、node2vec_features_valid.csv、node2vec_features_test.csv：Node2Vec 表征
- dynamic_graph_features_train.csv、dynamic_graph_features_valid.csv、dynamic_graph_features_test.csv：120 天滚动动态图特征
- stat_feature_report.json、graph_feature_report.json、dynamic_graph_feature_report.json：特征口径报告
