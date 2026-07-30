"""在全量验证集上选择动态图谱分类器融合权重。"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ID_COL, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, PROJECT_ROOT, RANDOM_SEED  # noqa: E402


COMPONENTS = {
    "dynamic_random_forest": PREDICTION_DIR / "model7_dynamic_graph_random_forest_v1_no_customer_type_strategy_A.csv",
    "dynamic_stack": PREDICTION_DIR / "model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
    "graphsage": PREDICTION_DIR / "model6_graphsage_v1_no_customer_type_strategy_A.csv",
}
STEM = "model8_final_dynamic_fusion_v7_strategy_A"


def normalize(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def evaluate(y: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y, score)),
        "pr_auc_average_precision": float(average_precision_score(y, score)),
    }
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y[order].sum())
        prefix = f"top{int(rate * 100)}pct"
        result.update(
            {
                f"{prefix}_k": k,
                f"{prefix}_hits": hits,
                f"{prefix}_precision": float(hits / k),
                f"{prefix}_recall": float(hits / y.sum()) if y.sum() else 0.0,
            }
        )
    return result


def load_split(split: str) -> pd.DataFrame:
    merged = None
    for name, path in COMPONENTS.items():
        frame = pd.read_csv(path)
        frame = frame[frame["split"].eq(split)][[ID_COL, "target", "score"]]
        frame = frame.rename(columns={"score": f"{name}_score"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns="target"), on=ID_COL, how="inner")
    if merged is None or len(merged) != 11087:
        raise RuntimeError(f"{split} 分支没有共同覆盖全量 11087 个账户。")
    return merged


def main() -> None:
    valid = load_split("valid")
    test = load_split("test")
    names = list(COMPONENTS)
    valid_scores = {name: normalize(valid[f"{name}_score"].to_numpy()) for name in names}
    test_scores = {name: normalize(test[f"{name}_score"].to_numpy()) for name in names}
    y_valid = valid["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)

    best = None
    for rf_step in range(21):
        for stack_step in range(21 - rf_step):
            weights = {
                "dynamic_random_forest": rf_step / 20,
                "dynamic_stack": stack_step / 20,
                "graphsage": (20 - rf_step - stack_step) / 20,
            }
            score = sum(weights[name] * valid_scores[name] for name in names)
            metrics = evaluate(y_valid, score)
            key = (metrics["pr_auc_average_precision"], metrics["top5pct_recall"], metrics["auc"])
            if best is None or key > best[0]:
                best = (key, weights, metrics, score)
    if best is None:
        raise RuntimeError("未找到有效融合权重。")

    _, weights, valid_metrics, valid_score = best
    test_score = sum(weights[name] * test_scores[name] for name in names)
    test_metrics = evaluate(y_test, test_score)
    predictions = pd.concat(
        [
            pd.DataFrame({ID_COL: valid[ID_COL], "split": "valid", "target": y_valid, "score": valid_score}),
            pd.DataFrame({ID_COL: test[ID_COL], "split": "test", "target": y_test, "score": test_score}),
        ],
        ignore_index=True,
    )
    predictions.to_csv(PREDICTION_DIR / f"{STEM}.csv", index=False)

    artifact = {
        "artifact_type": "validation_selected_rank_ensemble",
        "selected_weights": weights,
        "component_predictions": {name: str(path.relative_to(PROJECT_ROOT)) for name, path in COMPONENTS.items()},
        "normalization": "within_split_average_percentile_rank",
        "weight_grid_step": 0.05,
        "selection_metric_order": ["valid PR-AUC", "valid Top5% recall", "valid AUC"],
        "random_seed": RANDOM_SEED,
    }
    (MODEL_DIR / f"{STEM}.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {
        "_metadata": artifact,
        "status": "ok",
        "valid_all_accounts": valid_metrics,
        "test_all_accounts": test_metrics,
    }
    (METRIC_DIR / "final_dynamic_fusion_metrics_v7.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
