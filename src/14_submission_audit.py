"""比赛提交就绪审计。

检查完整源代码、Conda 环境、部署文档、最终模型制品和关键输出，
并为模型文件生成 SHA-256 清单。
"""

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, DELIVERABLE_DIR, DOCS_DIR, DST_COL, ID_COL, MODEL_DIR, OUTPUT_DIR, PROJECT_ROOT, SRC_COL  # noqa: E402


MAIN_SUFFIX = "v6_rolling_memory_dynamic_no_customer_type"
MODEL_FILES = {
    "dynamic_xgboost": MODEL_DIR / f"model5_xgb_dynamic_graph_{MAIN_SUFFIX}_strategy_A.json",
    "heterophily_propagation": MODEL_DIR / "model3_hetero_prop_v3_no_customer_type_strategy_A.joblib",
    "stack_config": MODEL_DIR / f"model4_stack_{MAIN_SUFFIX}_strategy_A.json",
    "logistic_regression_baseline": MODEL_DIR / "baseline_logistic_regression_v1_no_customer_type_strategy_A.joblib",
    "random_forest_baseline": MODEL_DIR / "baseline_random_forest_v1_no_customer_type_strategy_A.joblib",
    "dynamic_graph_random_forest": MODEL_DIR / "model7_dynamic_graph_random_forest_v1_no_customer_type_strategy_A.joblib",
    "graphsage": MODEL_DIR / "model6_graphsage_v1_no_customer_type_strategy_A.pt",
    "catboost_dynamic": MODEL_DIR / "model9_catboost_dynamic_v1_no_customer_type_strategy_A.joblib",
    "tgn_temporal_memory": MODEL_DIR / "model10_tgn_v1_no_customer_type_strategy_A.pt",
    "cv_bagging_experiment": MODEL_DIR / "model12_cv_bagged_dynamic_v1_no_customer_type_strategy_A.joblib",
    "overfit_guardrailed_final": MODEL_DIR / "model13_guardrailed_final_strategy_A.json",
    "rule_aware_calibration": MODEL_DIR / "model14_rule_aware_guardrailed_strategy_A.json",
    "previous_final_dynamic_fusion": MODEL_DIR / "model8_final_dynamic_fusion_v7_strategy_A.json",
    "final_selected_model": MODEL_DIR / "model11_validation_selected_best_strategy_A.json",
}

REQUIRED_SOURCE_FILES = [
    "config.py",
    "download_data.py",
    "src/01_data_cleaning.py",
    "src/02_label_builder.py",
    "src/03_features_stat.py",
    "src/04_features_graph.py",
    "src/05_model_xgb.py",
    "src/06_model_gnn.py",
    "src/07_explain_links.py",
    "src/08_model_stack.py",
    "src/09_build_deliverables.py",
    "src/09_dynamic_graph_viz.py",
    "src/10_features_dynamic_graph.py",
    "src/11_model_dynamic_graph_xgb.py",
    "src/12_layered_explainability.py",
    "src/13_final_project_audit.py",
    "src/14_submission_audit.py",
    "src/15_model_traditional_baselines.py",
    "src/16_model_graphsage.py",
    "src/17_compare_models.py",
    "src/18_model_final_fusion.py",
    "src/19_model_catboost.py",
    "src/20_model_tgn.py",
    "src/21_select_best_model.py",
    "src/22_cross_validation_overfit_audit.py",
    "src/23_model_cv_bagging.py",
    "src/24_model_overfit_guardrails.py",
    "src/25_model_rule_aware_calibrator.py",
    "src/26_data_edge_coverage_audit.py",
]

REQUIRED_OUTPUTS = [
    "outputs/predictions/model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
    "outputs/metrics/stack_experiment_metrics_v6_rolling_memory_dynamic_no_customer_type.json",
    "deliverables/task1_time_split_leakage_audit.json",
    "deliverables/task2_requirement_audit.json",
    "deliverables/task3_top20_associations.csv",
    "deliverables/task3_suspicious_paths.csv",
    "deliverables/task3_manual_review_form.csv",
    "deliverables/task4_consistency_audit.csv",
    "deliverables/data_edge_coverage_audit.json",
    "docs/data_edge_coverage_audit.md",
    "outputs/dynamic_graph/index.html",
    "outputs/metrics/adjusted_model_comparison.csv",
    "outputs/metrics/graphsage_metrics_v1_no_customer_type.json",
    "outputs/metrics/final_dynamic_fusion_metrics_v7.json",
    "outputs/predictions/model8_final_dynamic_fusion_v7_strategy_A.csv",
    "outputs/metrics/catboost_metrics_v1_no_customer_type.json",
    "outputs/metrics/tgn_metrics_v1_no_customer_type.json",
    "outputs/metrics/final_model_selection_metrics_v8.json",
    "outputs/predictions/model11_validation_selected_best_strategy_A.csv",
    "outputs/metrics/five_fold_overfit_audit_v1.json",
    "outputs/metrics/cv_bagging_metrics_v1_no_customer_type.json",
    "outputs/predictions/model12_cv_bagged_dynamic_v1_no_customer_type_strategy_A.csv",
    "outputs/metrics/model_overfit_guardrail_audit_v1.json",
    "outputs/predictions/model13_guardrailed_final_strategy_A.csv",
    "outputs/metrics/rule_aware_calibration_metrics_v1.json",
    "outputs/predictions/model14_rule_aware_guardrailed_strategy_A.csv",
    "outputs/explanations/rule_aware_evidence_v1.csv",
]


def add_check(checks: list[dict], name: str, passed: bool, detail: str, severity: str = "error") -> None:
    checks.append({"name": name, "status": "pass" if passed else severity, "detail": detail})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_artifacts(checks: list[dict]) -> list[dict]:
    artifacts = []
    for role, path in MODEL_FILES.items():
        add_check(checks, f"模型文件 {role}", path.exists(), str(path.relative_to(PROJECT_ROOT)))
        if not path.exists():
            continue
        artifacts.append(
            {
                "role": role,
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )

    dynamic_path = MODEL_FILES["dynamic_xgboost"]
    if dynamic_path.exists():
        try:
            import xgboost as xgb

            booster = xgb.Booster()
            booster.load_model(dynamic_path)
            add_check(checks, "动态 XGBoost 权重可加载", True, f"特征数={booster.num_features()}")
        except Exception as exc:
            add_check(checks, "动态 XGBoost 权重可加载", False, f"{type(exc).__name__}: {exc}")

    gnn_path = MODEL_FILES["heterophily_propagation"]
    if gnn_path.exists():
        try:
            bundle = joblib.load(gnn_path)
            required = {"model", "scaler", "imputer", "base_feature_columns", "propagation_prefixes"}
            missing = sorted(required - set(bundle))
            add_check(checks, "图传播模型包可加载", not missing, f"缺失键={missing}")
        except Exception as exc:
            add_check(checks, "图传播模型包可加载", False, f"{type(exc).__name__}: {exc}")

    for role in ["logistic_regression_baseline", "random_forest_baseline", "dynamic_graph_random_forest", "catboost_dynamic", "cv_bagging_experiment"]:
        path = MODEL_FILES[role]
        if path.exists():
            try:
                bundle = joblib.load(path)
                required = {"models", "feature_columns", "metadata"} if role == "cv_bagging_experiment" else {"model", "feature_columns", "metadata"}
                missing = sorted(required - set(bundle))
                add_check(checks, f"传统基线 {role} 可加载", not missing, f"缺失键={missing}")
            except Exception as exc:
                add_check(checks, f"传统基线 {role} 可加载", False, f"{type(exc).__name__}: {exc}")

    graphsage_path = MODEL_FILES["graphsage"]
    if graphsage_path.exists():
        try:
            import torch

            bundle = torch.load(graphsage_path, map_location="cpu", weights_only=True)
            required = {"state_dict", "input_dim", "hidden_dim", "feature_columns", "scaler_mean", "scaler_scale"}
            missing = sorted(required - set(bundle))
            add_check(checks, "GraphSAGE 权重可加载", not missing, f"缺失键={missing}")
        except Exception as exc:
            add_check(checks, "GraphSAGE 权重可加载", False, f"{type(exc).__name__}: {exc}")

    stack_path = MODEL_FILES["stack_config"]
    if stack_path.exists():
        try:
            stack = json.loads(stack_path.read_text(encoding="utf-8"))
            weights = stack.get("selected_weights", {})
            weight_sum = sum(float(value) for value in weights.values())
            add_check(checks, "融合权重可加载", abs(weight_sum - 1.0) < 1e-9, f"weights={weights}")
        except Exception as exc:
            add_check(checks, "融合权重可加载", False, f"{type(exc).__name__}: {exc}")

    for role, label in [
        ("previous_final_dynamic_fusion", "上一版最终动态融合权重可加载"),
        ("final_selected_model", "最终验证集选择权重可加载"),
        ("overfit_guardrailed_final", "过拟合护栏最终权重可加载"),
        ("rule_aware_calibration", "规则感知校准权重可加载"),
    ]:
        final_fusion_path = MODEL_FILES[role]
        if not final_fusion_path.exists():
            continue
        try:
            fusion = json.loads(final_fusion_path.read_text(encoding="utf-8"))
            weights = fusion.get("selected_weights", {})
            weight_sum = sum(float(value) for value in weights.values())
            add_check(checks, label, abs(weight_sum - 1.0) < 1e-9, f"weights={weights}")
        except Exception as exc:
            add_check(checks, label, False, f"{type(exc).__name__}: {exc}")
    return artifacts


def render_markdown(result: dict) -> str:
    lines = [
        "# 比赛提交就绪审计",
        "",
        f"总体状态：{result['status']}",
        "",
        "| 检查项 | 状态 | 详情 |",
        "|---|---|---|",
    ]
    for item in result["checks"]:
        lines.append(f"| {item['name']} | {item['status']} | {item['detail']} |")
    lines.extend(
        [
            "",
            "## 数据边界",
            "",
            "- 风险标签表没有标签确认时间，不能声称严格预测未来新增标签。",
            "- 59 个确认嫌疑账户中只有 3 个在交易边表中有可追溯边，其余 56 个只能做缺边审计。",
            "- 以上是数据集边界，不能通过算法伪造成真实资金链路。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    checks: list[dict] = []
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    missing_source = [path for path in REQUIRED_SOURCE_FILES if not (PROJECT_ROOT / path).exists()]
    add_check(checks, "完整源代码", not missing_source, f"脚本数={len(REQUIRED_SOURCE_FILES)}，缺失={missing_source}")
    readme_dirs = [
        "",
        "5个高风险嫌疑账户交易特征记录表",
        "docs",
        "models",
        "outputs",
        "outputs/clean",
        "deliverables",
        "deliverables/task1_graph_samples",
        "outputs/dynamic_graph",
        "outputs/explanations",
        "outputs/explanations/layered",
        "outputs/features",
        "outputs/labels",
        "outputs/metrics",
        "outputs/predictions",
        "src",
    ]
    missing_readmes = [d for d in readme_dirs if not (PROJECT_ROOT / d / "readme.txt").exists()]
    add_check(checks, "文件夹readme.txt覆盖", not missing_readmes, f"目录数={len(readme_dirs)}，缺失={missing_readmes}")

    add_check(checks, "Conda 环境文件", (PROJECT_ROOT / "environment.yml").exists(), "environment.yml")
    add_check(checks, "部署文档", (PROJECT_ROOT / "DEPLOYMENT.md").exists(), "DEPLOYMENT.md")
    add_check(checks, "项目说明", (PROJECT_ROOT / "readme.md").exists(), "readme.md")

    edge_audit_path = DELIVERABLE_DIR / "data_edge_coverage_audit.json"
    edge_audit = json.loads(edge_audit_path.read_text(encoding="utf-8")) if edge_audit_path.exists() else {}
    add_check(
        checks,
        "数据边覆盖审计",
        bool(edge_audit.get("conclusion", {}).get("missing_edge_is_raw_data_fact")),
        "56/59 缺边为原始数据事实，未伪造资金链路",
    )

    clean_files = [CLEAN_DIR / name for name in ["clean_accounts.csv", "clean_transactions.csv", "clean_labels.csv"]]
    missing_clean = [str(path.relative_to(PROJECT_ROOT)) for path in clean_files if not path.exists()]
    add_check(checks, "可复现的清洗数据输入", not missing_clean, f"缺失={missing_clean}")
    suspect_with_edge = None
    suspect_without_edge = None
    if not missing_clean:
        labels = pd.read_csv(CLEAN_DIR / "clean_labels.csv")
        transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", usecols=[SRC_COL, DST_COL])
        suspect_ids = set(labels.loc[labels["label_code"].eq(1), ID_COL].astype(int))
        active_ids = set(transactions[SRC_COL].astype(int)).union(set(transactions[DST_COL].astype(int)))
        suspect_with_edge = len(suspect_ids & active_ids)
        suspect_without_edge = len(suspect_ids - active_ids)
        add_check(
            checks,
            "确认嫌疑账户交易边覆盖审计",
            len(suspect_ids) == 59 and suspect_with_edge + suspect_without_edge == 59,
            f"嫌疑账户={len(suspect_ids)}，有边={suspect_with_edge}，无边={suspect_without_edge}",
        )

        cleaning_report_path = CLEAN_DIR / "cleaning_report.json"
        cleaning_report = json.loads(cleaning_report_path.read_text(encoding="utf-8"))
        raw_rows = int(cleaning_report.get("raw_rows", {}).get("transactions", -1))
        clean_rows = int(cleaning_report.get("clean_rows", {}).get("transactions", -2))
        add_check(checks, "原始/清洗交易行数一致", raw_rows == clean_rows == len(transactions), f"raw={raw_rows}，clean={clean_rows}")

    missing_outputs = [path for path in REQUIRED_OUTPUTS if not (PROJECT_ROOT / path).exists()]
    add_check(checks, "关键比赛输出", not missing_outputs, f"缺失={missing_outputs}")

    package_versions = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "xgboost", "catboost", "networkx", "gensim", "joblib", "node2vec", "torch", "torch-geometric"]:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "missing"
    missing_packages = [name for name, version in package_versions.items() if version == "missing"]
    add_check(checks, "运行依赖可导入", not missing_packages, f"缺失={missing_packages}")

    artifacts = load_model_artifacts(checks)
    model_manifest = {
        "final_model": "model11_validation_selected_best_strategy_A",
        "artifacts": artifacts,
        "package_versions": package_versions,
        "reproduction_document": "DEPLOYMENT.md",
    }
    (MODEL_DIR / "model_manifest.json").write_text(
        json.dumps(model_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    add_check(checks, "模型校验清单", bool(artifacts), "models/model_manifest.json")

    errors = [item for item in checks if item["status"] == "error"]
    warnings = [item for item in checks if item["status"] == "warning"]
    result = {
        "status": "pass" if not errors else "failed",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "checks": checks,
        "model_artifacts": artifacts,
        "known_data_limitations": [
            "风险标签表无标签确认时间。",
            f"59个确认嫌疑账户中{suspect_without_edge}个未出现在交易边表。",
            "56个缺边账户的高分主要来自静态画像与图基座特征；账户级五折留出验证Top5%召回约44%，应作为补数复核线索而非已证实链路。",
        ],
    }
    (DELIVERABLE_DIR / "submission_readiness_audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (DOCS_DIR / "submission_readiness_audit.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
