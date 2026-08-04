"""最终模型过拟合护栏审计。

这个脚本不使用测试集选择模型，只做三件事：
1. 对 model8_final_dynamic_fusion_v7_strategy_A、model11_validation_selected_best_strategy_A、Model12 做 valid/test 时间留出稳定性审计。
2. 读取五折账户级审计，记录底层模型的账户级过拟合风险。
3. 只用 valid 在最终主模型与正则化 Model12 之间做护栏式融合检查。

若 Model12 不能提升 valid PR-AUC，则最终护栏模型保持最终主模型不变。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ID_COL, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, PROJECT_ROOT, RANDOM_SEED  # noqa: E402


MODEL_METRIC_FILES = {
    "model8_final_dynamic_fusion_v7_strategy_A": METRIC_DIR / "final_dynamic_fusion_metrics_v7.json",
    "model11_validation_selected_best_strategy_A": METRIC_DIR / "final_model_selection_metrics_v8.json",
    "Model12": METRIC_DIR / "cv_bagging_metrics_v1_no_customer_type.json",
}
PREDICTION_FILES = {
    "model11_validation_selected_best_strategy_A": PREDICTION_DIR / "model11_validation_selected_best_strategy_A.csv",
    "Model12": PREDICTION_DIR / "model12_cv_bagged_dynamic_v1_no_customer_type_strategy_A.csv",
}
STEM = "model13_guardrailed_final_strategy_A"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def normalize(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        idx = np.argsort(-score)[:k]
        hits = int(y_true[idx].sum())
        prefix = f"top{int(rate * 100)}pct"
        result[f"{prefix}_k"] = k
        result[f"{prefix}_hits"] = hits
        result[f"{prefix}_precision"] = float(hits / k)
        result[f"{prefix}_recall"] = float(hits / positives) if positives else 0.0
    return result


def temporal_overfit_rows() -> list[dict]:
    rows = []
    for model_name, path in MODEL_METRIC_FILES.items():
        report = read_json(path)
        valid = report.get("valid_all_accounts", {})
        test = report.get("test_all_accounts", {})
        if not valid or not test:
            continue
        pr_gap = float(valid["pr_auc_average_precision"] - test["pr_auc_average_precision"])
        top5_gap = float(valid["top5pct_recall"] - test["top5pct_recall"])
        rows.append(
            {
                "model": model_name,
                "valid_auc": valid["auc"],
                "test_auc": test["auc"],
                "valid_pr_auc": valid["pr_auc_average_precision"],
                "test_pr_auc": test["pr_auc_average_precision"],
                "valid_top5_recall": valid["top5pct_recall"],
                "test_top5_recall": test["top5pct_recall"],
                "valid_minus_test_pr_auc": pr_gap,
                "valid_minus_test_top5_recall": top5_gap,
                "temporal_overfit_flag": bool(pr_gap > 0.05 or top5_gap > 0.05),
            }
        )
    return rows


def load_component_split(split: str) -> pd.DataFrame:
    merged = None
    for model_name, path in PREDICTION_FILES.items():
        frame = pd.read_csv(path)
        frame = frame[frame["split"].eq(split)][[ID_COL, "target", "score"]]
        frame = frame.rename(columns={"score": f"{model_name}_score"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns="target"), on=ID_COL, how="inner")
    if merged is None or len(merged) != 11087:
        raise RuntimeError(f"{split} 护栏候选模型没有共同覆盖全量 11087 个账户。")
    return merged


def select_guardrailed_model() -> tuple[dict, pd.DataFrame]:
    valid = load_component_split("valid")
    test = load_component_split("test")
    candidates = []
    valid_m11 = normalize(valid["model11_validation_selected_best_strategy_A_score"].to_numpy())
    valid_m12 = normalize(valid["Model12_score"].to_numpy())
    test_m11 = normalize(test["model11_validation_selected_best_strategy_A_score"].to_numpy())
    test_m12 = normalize(test["Model12_score"].to_numpy())
    y_valid = valid["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)

    for step in range(21):
        w11 = step / 20
        w12 = 1.0 - w11
        valid_score = w11 * valid_m11 + w12 * valid_m12
        metrics = evaluate(y_valid, valid_score)
        candidates.append(
            {
                "weight_model11": w11,
                "weight_model12": w12,
                "valid_auc": metrics["auc"],
                "valid_pr_auc": metrics["pr_auc_average_precision"],
                "valid_top5_recall": metrics["top5pct_recall"],
                "valid_top5_hits": metrics["top5pct_hits"],
            }
        )

    candidates = sorted(candidates, key=lambda x: (x["valid_pr_auc"], x["valid_top5_recall"], x["valid_auc"]), reverse=True)
    selected = candidates[0]
    valid_score = selected["weight_model11"] * valid_m11 + selected["weight_model12"] * valid_m12
    test_score = selected["weight_model11"] * test_m11 + selected["weight_model12"] * test_m12
    selected_report = {
        "selected_weights": {
            "model11_validation_selected_best_strategy_A": selected["weight_model11"],
            "Model12": selected["weight_model12"],
        },
        "selection_data": "valid_all_accounts",
        "selection_metric_order": ["valid PR-AUC", "valid Top5% recall", "valid AUC"],
        "valid_all_accounts": evaluate(y_valid, valid_score),
        "test_all_accounts": evaluate(y_test, test_score),
        "candidate_count": len(candidates),
        "candidate_top5": candidates[:5],
        "decision": (
            "keep_model11"
            if selected["weight_model11"] == 1.0
            else "blend_model11_and_model12"
        ),
        "decision_note": (
            "正则化 Model12 未提升验证集 PR-AUC，因此不强行加入最终主模型。"
            if selected["weight_model11"] == 1.0
            else "正则化 Model12 在验证集上提供增量，因此纳入护栏融合。"
        ),
    }
    predictions = pd.concat(
        [
            pd.DataFrame({ID_COL: valid[ID_COL], "split": "valid", "target": y_valid, "score": valid_score}),
            pd.DataFrame({ID_COL: test[ID_COL], "split": "test", "target": y_test, "score": test_score}),
        ],
        ignore_index=True,
    )
    return selected_report, predictions


def main() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    selected_report, predictions = select_guardrailed_model()
    cv_audit = read_json(METRIC_DIR / "five_fold_overfit_audit_v1.json")
    temporal_rows = temporal_overfit_rows()
    report = {
        "status": "ok",
        "artifact_type": "overfit_guardrailed_final_model",
        "random_seed": RANDOM_SEED,
        "final_recommendation": "model11_validation_selected_best_strategy_A remains final model; Model13 is a guardrail alias with audited selection.",
        "label_time_limitation": "风险标签表没有标签确认时间，不能严格声称预测未来新增风险标签。",
        "temporal_holdout_overfit_audit": temporal_rows,
        "account_level_cv_audit": cv_audit.get("models", []),
        "account_level_cv_note": "五折账户级审计暴露底层模型存在账户级泛化风险；该结果用于风险披露，不替代比赛时间切分。",
        "guardrailed_selection": selected_report,
    }
    artifact = {
        "artifact_type": "guardrailed_final_selection",
        "selected_weights": selected_report["selected_weights"],
        "component_predictions": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in PREDICTION_FILES.items()},
        "normalization": "within_split_average_percentile_rank",
        "selection_data": selected_report["selection_data"],
        "selection_metric_order": selected_report["selection_metric_order"],
        "decision": selected_report["decision"],
        "decision_note": selected_report["decision_note"],
    }
    predictions.to_csv(PREDICTION_DIR / f"{STEM}.csv", index=False)
    (METRIC_DIR / "model_overfit_guardrail_audit_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / f"{STEM}.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
