# 参赛作品说明

第五届中国研究生金融科技创新大赛“揭榜挂帅”主赛道
赛题：基于资金图谱的涉诈账户发现与可疑链路解释（江苏银行）
技术路线：滚动动态资金图谱特征 + 验证集选择融合（Model11）+ 分层证据研判

## 一、四项评审材料

1. 精益画布：由团队按大赛模板填写（本仓库不包含模板正文）。
2. 可运行项目成果：本仓库的 config.py、src/、models/、outputs/、environment.yml、DEPLOYMENT.md。
3. 技术文档：技术文档-lxq-fixed.docx（已随本仓库提交）。
4. 原创性声明：由团队按大赛模板签字后单独提交。

## 二、目录结构

- config.py：全局路径与特征配置
- src/：全部源代码（清洗、特征、建模、融合、解释、审计）
- docs/：技术说明、审计报告、案例文档
- models/：训练好的模型权重与清单
- outputs/：清洗数据、特征、指标、预测、解释与比赛交付物
- 5个高风险嫌疑账户交易特征记录表/：业务侧人工画像与评价材料
- environment.yml：Conda 环境依赖
- DEPLOYMENT.md：部署与复现步骤
- readme.md：详细技术说明
- download_data.py：赛题数据下载脚本，密码通过环境变量 SCOW_DOWNLOAD_PASSWORD 提供

## 三、快速运行

```bash
conda env create -f environment.yml
conda activate fintech09
python src/14_submission_audit.py
python src/13_final_project_audit.py
```

完整复现顺序见 DEPLOYMENT.md。四个任务的交付物位于 outputs/deliverables/。
