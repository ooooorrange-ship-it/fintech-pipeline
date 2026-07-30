"""交付前最终一致性审计。

只检查已有输出，不重新训练模型。硬错误会返回非零状态；数据边界限制和人工复核事项作为 warning 留在审计结果中。
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, DOCS_DIR, EXPLANATION_DIR, ID_COL, LABEL_DIR, MODEL_DIR, OUTPUT_DIR, PREDICTION_DIR, SPLITS, SRC_COL, DST_COL, TIME_COL  # noqa: E402


DELIVERABLE_DIR = OUTPUT_DIR / "deliverables"
LAYERED_DIR = EXPLANATION_DIR / "layered"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def add_check(checks: list[dict], name: str, passed: bool, detail: str, severity: str = "error") -> None:
    checks.append({"name": name, "status": "pass" if passed else severity, "detail": detail})


def audit_prediction_coverage(checks: list[dict], accounts: pd.DataFrame, labels: pd.DataFrame) -> None:
    path = PREDICTION_DIR / "model8_final_dynamic_fusion_v7_strategy_A.csv"
    if not path.exists():
        add_check(checks, "主模型预测文件", False, str(path))
        return
    pred = pd.read_csv(path)
    expected_ids = set(accounts[ID_COL].astype(int))
    for split in ["valid", "test"]:
        part = pred[pred["split"].eq(split)].copy()
        ids = set(part[ID_COL].astype(int))
        add_check(
            checks,
            f"{split}全量账户预测覆盖",
            len(part) == len(expected_ids) and ids == expected_ids and not part[ID_COL].duplicated().any(),
            f"预测行={len(part)}，账户数={len(expected_ids)}，重复ID={int(part[ID_COL].duplicated().sum())}",
        )
        add_check(
            checks,
            f"{split}全量指标存在",
            (OUTPUT_DIR / "metrics" / "final_dynamic_fusion_metrics_v7.json").exists(),
            "融合模型报告文件存在",
        )
    add_check(
        checks,
        "主模型标签覆盖",
        set(labels[ID_COL].astype(int)) == expected_ids,
        f"标签账户数={len(labels)}，节点账户数={len(accounts)}",
    )


def audit_leakage(checks: list[dict]) -> None:
    leakage = read_json(DELIVERABLE_DIR / "task1_time_split_leakage_audit.json")
    splits = leakage.get("splits", {})
    all_zero = bool(splits) and all(int(v.get("future_transaction_rows_used", 1)) == 0 for v in splits.values())
    add_check(checks, "未来交易未进入分片图样本", all_zero, json.dumps({k: v.get("future_transaction_rows_used") for k, v in splits.items()}, ensure_ascii=False))
    add_check(
        checks,
        "标签时间限制已显式记录",
        leakage.get("label_time_available") is False and bool(leakage.get("label_time_note")),
        leakage.get("label_time_note", "缺少标签时间说明"),
        severity="warning",
    )
    dynamic_report = read_json(OUTPUT_DIR / "features" / "dynamic_graph_feature_report.json")
    feature_hits = [name for name, cols in leakage.get("feature_label_column_hits", {}).items() if cols]
    add_check(checks, "特征文件不含标签字段", not feature_hits, f"命中特征文件={feature_hits}")
    for split, (_, end) in SPLITS.items():
        observed_end = dynamic_report.get("splits", {}).get(split, {}).get("observation_end", "")
        add_check(checks, f"{split}动态图观察截止时间", observed_end <= str(end), f"observation_end={observed_end}，split_end={end}")


def audit_explanations(checks: list[dict], accounts: pd.DataFrame) -> None:
    coverage = read_json(LAYERED_DIR / "layered_explainability_coverage.json")
    audit = pd.read_csv(LAYERED_DIR / "confirmed_suspect_explainability_audit.csv") if (LAYERED_DIR / "confirmed_suspect_explainability_audit.csv").exists() else pd.DataFrame()
    recovery = pd.read_csv(LAYERED_DIR / "suspect_link_recovery_queue.csv") if (LAYERED_DIR / "suspect_link_recovery_queue.csv").exists() else pd.DataFrame()
    queue = pd.read_csv(LAYERED_DIR / "risk_review_queue_active_accounts.csv") if (LAYERED_DIR / "risk_review_queue_active_accounts.csv").exists() else pd.DataFrame()
    expected_suspects = int((pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")["label_code"] == 1).sum())
    add_check(checks, "59个嫌疑账户审计完整", len(audit) == expected_suspects, f"审计={len(audit)}，确认嫌疑人={expected_suspects}")
    expected_missing_edges = int((audit["history_txn_count"] == 0).sum()) if "history_txn_count" in audit.columns else -1
    add_check(checks, "缺边恢复队列完整", len(recovery) == expected_missing_edges, f"恢复队列={len(recovery)}，审计缺边账户={expected_missing_edges}")
    if not recovery.empty and ID_COL in recovery.columns and ID_COL in audit.columns:
        expected_ids = set(audit.loc[audit["history_txn_count"].eq(0), ID_COL].astype(int))
        recovery_ids = set(recovery[ID_COL].astype(int))
        add_check(checks, "缺边恢复队列账户可追溯", recovery_ids == expected_ids, f"恢复队列账户={len(recovery_ids)}，审计缺边账户={len(expected_ids)}")
    add_check(checks, "Top30巡检队列完整", len(queue) == 30, f"巡检账户数={len(queue)}")
    add_check(checks, "分层覆盖统计与CSV一致", int(coverage.get("confirmed_suspect_total", -1)) == len(audit) and int(coverage.get("active_risk_review_account_count", -1)) == len(queue), "coverage JSON 与 CSV 行数一致")

    account_ids = set(accounts[ID_COL].astype(int))
    for name, key in [("关联账户", "counterparty_id"), ("可疑路径", "account_1"), ("资金结构", "root_account_id")]:
        path = {
            "关联账户": LAYERED_DIR / "risk_review_queue_top20_associations.csv",
            "可疑路径": LAYERED_DIR / "risk_review_queue_suspicious_paths.csv",
            "资金结构": LAYERED_DIR / "risk_review_queue_fund_flow_structures.csv",
        }[name]
        frame = pd.read_csv(path) if path.exists() else pd.DataFrame()
        bad = [] if frame.empty or key not in frame.columns else sorted(set(frame[key].dropna().astype(int)) - account_ids)
        add_check(checks, f"{name}节点ID可追溯", not bad, f"记录数={len(frame)}，未知账户={bad[:5]}")


def audit_deliverables(checks: list[dict]) -> None:
    task2 = read_json(DELIVERABLE_DIR / "task2_requirement_audit.json")
    task3 = read_json(DELIVERABLE_DIR / "task3_task4_explanation_audit.json")
    add_check(checks, "任务2使用全量账户评估", task2.get("evaluation_scope") == "all_accounts", f"evaluation_scope={task2.get('evaluation_scope')}")
    add_check(checks, "任务2 AUC达标", bool(task2.get("auc_requirement_auc_ge_0_85")), f"AUC={task2.get('test_auc')}")
    add_check(checks, "任务2 Top5%召回达标", bool(task2.get("top5pct_recall_requirement_ge_50pct")), f"Top5%召回={task2.get('test_top5pct_recall')}")
    add_check(
        checks,
        "任务2最强传统基线PR-AUC提升20%",
        bool(task2.get("pr_auc_improvement_ge_20pct")),
        f"相对传统RF提升={task2.get('pr_auc_improvement_vs_strongest_traditional_ratio')}",
        severity="warning",
    )
    review_path = DELIVERABLE_DIR / "task3_manual_review_form.csv"
    review = pd.read_csv(review_path, keep_default_na=False) if review_path.exists() else pd.DataFrame()
    reviewed = review[review.get("manual_pass", pd.Series(dtype=str)).astype(str).str.strip().ne("")] if not review.empty else pd.DataFrame()
    passed = reviewed["manual_pass"].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "是", "通过"}).sum() if not reviewed.empty else 0
    manual_pass_rate = float(passed / len(reviewed)) if len(reviewed) else 0.0
    association_hit_rate = float(task3.get("confirmed_risk_association_hit_rate", 0.0))
    task3_metric_met = association_hit_rate >= 0.5 or (len(reviewed) >= 5 and manual_pass_rate >= 0.7)
    add_check(
        checks,
        "任务3关联命中或人工抽检指标",
        task3_metric_met,
        f"关联命中率={association_hit_rate:.2%}，已抽检={len(reviewed)}/5，人工通过率={manual_pass_rate:.2%}",
        severity="warning",
    )
    add_check(checks, "任务4典型案例不少于5个", int(task3.get("typical_case_count", 0)) >= 5, f"案例数={task3.get('typical_case_count')}")
    for name, key, layered_name, deliverable_name in [
        ("交付物关联行数一致", "top20_association_rows", "risk_review_queue_top20_associations.csv", "task3_top20_associations.csv"),
        ("交付物路径行数一致", "suspicious_path_rows", "risk_review_queue_suspicious_paths.csv", "task3_suspicious_paths.csv"),
        ("交付物资金结构行数一致", "fund_flow_structure_rows", "risk_review_queue_fund_flow_structures.csv", "task3_fund_flow_structures.csv"),
        ("交付物缺边恢复队列行数一致", "missing_edge_recovery_queue_rows", "suspect_link_recovery_queue.csv", "task3_link_recovery_queue.csv"),
    ]:
        layered_path = LAYERED_DIR / layered_name
        deliverable_path = DELIVERABLE_DIR / deliverable_name
        layered_count = len(pd.read_csv(layered_path)) if layered_path.exists() else -1
        deliverable_count = len(pd.read_csv(deliverable_path)) if deliverable_path.exists() else -1
        add_check(
            checks,
            name,
            layered_count == deliverable_count == int(task3.get(key, -2)),
            f"layered={layered_count}，deliverable={deliverable_count}，audit={task3.get(key)}",
        )


def audit_submission_assets(checks: list[dict]) -> None:
    required_models = [
        MODEL_DIR / "model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json",
        MODEL_DIR / "model3_hetero_prop_v3_no_customer_type_strategy_A.joblib",
        MODEL_DIR / "model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.json",
        MODEL_DIR / "model8_final_dynamic_fusion_v7_strategy_A.json",
        MODEL_DIR / "baseline_logistic_regression_v1_no_customer_type_strategy_A.joblib",
        MODEL_DIR / "baseline_random_forest_v1_no_customer_type_strategy_A.joblib",
        MODEL_DIR / "model7_dynamic_graph_random_forest_v1_no_customer_type_strategy_A.joblib",
        MODEL_DIR / "model6_graphsage_v1_no_customer_type_strategy_A.pt",
        MODEL_DIR / "model_manifest.json",
    ]
    missing_models = [path.name for path in required_models if not path.exists()]
    add_check(checks, "最终模型权重与清单", not missing_models, f"缺失={missing_models}")
    add_check(checks, "Conda环境文件", (OUTPUT_DIR.parent / "environment.yml").exists(), "environment.yml")
    add_check(checks, "部署文档", (OUTPUT_DIR.parent / "DEPLOYMENT.md").exists(), "DEPLOYMENT.md")
    readiness = read_json(DELIVERABLE_DIR / "submission_readiness_audit.json")
    add_check(
        checks,
        "提交就绪审计",
        readiness.get("status") == "pass" and int(readiness.get("error_count", 1)) == 0,
        f"status={readiness.get('status')}，error_count={readiness.get('error_count')}",
    )


def render_markdown(result: dict) -> str:
    lines = ["# 最终项目一致性审计", "", f"总体状态：{result['status']}", "", "| 检查项 | 状态 | 详情 |", "|---|---|---|"]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    lines.extend(["", "说明：warning 不代表代码不可运行，表示赛题口径或人工环节仍需在答辩中如实说明。"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="运行交付前最终一致性审计。")
    parser.parse_args()
    checks: list[dict] = []
    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    audit_prediction_coverage(checks, accounts, labels)
    audit_leakage(checks)
    audit_explanations(checks, accounts)
    audit_deliverables(checks)
    audit_submission_assets(checks)
    errors = [x for x in checks if x["status"] == "error"]
    warnings = [x for x in checks if x["status"] == "warning"]
    result = {
        "status": "pass" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
        "known_limitations": [
            "原始风险标签表没有标签确认时间，不能严格宣称预测未来新增风险标签。",
            "59个确认嫌疑账户中56个在交易边表内无入边或出边，不能生成真实资金链路。",
            "任务3人工抽检通过率需要业务成员对生成证据进行人工确认。",
        ],
    }
    out_json = DELIVERABLE_DIR / "final_project_audit.json"
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out_md = DOCS_DIR / "final_project_audit.md"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
