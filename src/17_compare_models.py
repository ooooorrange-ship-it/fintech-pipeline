"""汇总传统基线、GraphSAGE、XGBoost 和最终融合模型的统一指标。"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import METRIC_DIR  # noqa: E402


def read_json(name: str) -> dict:
    path = METRIC_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"缺少指标文件：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def add_rows(rows: list[dict], model: str, family: str, profile_policy: str, metrics: dict) -> None:
    for split in ["train_all_accounts", "valid_all_accounts", "test_all_accounts"]:
        if split not in metrics:
            continue
        item = metrics[split]
        rows.append(
            {
                "model": model,
                "model_family": family,
                "profile_policy": profile_policy,
                "split": split.removesuffix("_all_accounts"),
                "evaluation_scope": "all_accounts",
                "auc": item["auc"],
                "pr_auc": item["pr_auc_average_precision"],
                "top1pct_hits": item["top1pct_hits"],
                "top1pct_recall": item["top1pct_recall"],
                "top5pct_hits": item["top5pct_hits"],
                "top5pct_recall": item["top5pct_recall"],
            }
        )


def main() -> None:
    traditional = read_json("traditional_baseline_metrics_v1_no_customer_type.json")
    traditional_weak = read_json("traditional_baseline_metrics_v1_txn_graph_dynamic_only.json")
    graphsage = read_json("graphsage_metrics_v1_no_customer_type.json")
    graphsage_weak = read_json("graphsage_metrics_v1_txn_graph_dynamic_only.json")
    xgb = read_json("xgb_experiment_metrics_v2_no_customer_type.json")
    dynamic = read_json("dynamic_graph_experiment_metrics_v6_rolling_memory_dynamic_no_customer_type.json")
    stack = read_json("stack_experiment_metrics_v6_rolling_memory_dynamic_no_customer_type.json")
    xgb_weak = read_json("xgb_experiment_metrics_v2_txn_graph_only.json")
    dynamic_weak = read_json("dynamic_graph_experiment_metrics_v6_rolling_memory_dynamic_txn_graph_only.json")
    final_fusion = read_json("final_dynamic_fusion_metrics_v7.json")

    rows: list[dict] = []
    main_profile = "drop_customer_type"
    weak_profile = "transaction_graph_dynamic_only"
    add_rows(rows, "LogisticRegression", "traditional_baseline", main_profile, traditional["logistic_regression"])
    add_rows(rows, "RandomForest", "traditional_baseline", main_profile, traditional["random_forest"])
    add_rows(rows, "动态资金图谱RandomForest", "dynamic_graph_random_forest", main_profile, traditional["dynamic_graph_random_forest"])
    add_rows(rows, "GraphSAGE", "pytorch_geometric_gnn", main_profile, graphsage)
    add_rows(rows, "XGBoost统计特征", "xgboost_baseline", main_profile, xgb["model1_xgb_stat_v2_no_customer_type_strategy_A"])
    add_rows(rows, "XGBoost统计+图特征", "xgboost_strong_baseline", main_profile, xgb["model2_xgb_stat_graph_v2_no_customer_type_strategy_A"])
    add_rows(rows, "滚动动态资金图谱XGBoost", "dynamic_graph_xgboost", main_profile, dynamic["model5_xgb_dynamic_graph_strategy_A"])
    add_rows(rows, "最终融合模型", "rank_weighted_ensemble", main_profile, stack)
    add_rows(rows, "最终动态融合模型v7", "validation_selected_dynamic_ensemble", main_profile, final_fusion)
    add_rows(rows, "LogisticRegression弱画像", "traditional_baseline", weak_profile, traditional_weak["logistic_regression"])
    add_rows(rows, "RandomForest弱画像", "traditional_baseline", weak_profile, traditional_weak["random_forest"])
    add_rows(rows, "动态资金图谱RandomForest弱画像", "dynamic_graph_random_forest", weak_profile, traditional_weak["dynamic_graph_random_forest"])
    add_rows(rows, "GraphSAGE弱画像", "pytorch_geometric_gnn", weak_profile, graphsage_weak)
    add_rows(rows, "XGBoost统计+图特征弱画像", "xgboost_strong_baseline", weak_profile, xgb_weak["model2_xgb_stat_graph_v2_txn_graph_only_strategy_A"])
    add_rows(rows, "滚动动态资金图谱XGBoost弱画像", "dynamic_graph_xgboost", weak_profile, dynamic_weak["model5_xgb_dynamic_graph_strategy_A"])

    frame = pd.DataFrame(rows)
    frame.to_csv(METRIC_DIR / "adjusted_model_comparison.csv", index=False)
    test = frame[frame["split"].eq("test") & frame["profile_policy"].eq(main_profile)].sort_values(
        ["pr_auc", "top5pct_recall", "auc"], ascending=False
    )
    result = {
        "evaluation_scope": "all 11087 accounts",
        "selection_policy": "validation metrics select/tune models; test metrics are reported once",
        "label_time_limitation": "Risk labels have no confirmation time; no model is described as predicting future-new labels.",
        "best_test_by_pr_auc": test.iloc[0]["model"] if len(test) else "",
        "test_ranking": test.to_dict(orient="records"),
        "weak_profile_test": frame[
            frame["split"].eq("test") & frame["profile_policy"].eq(weak_profile)
        ].sort_values("pr_auc", ascending=False).to_dict(orient="records"),
    }
    (METRIC_DIR / "adjusted_model_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(test.to_string(index=False))


if __name__ == "__main__":
    main()
