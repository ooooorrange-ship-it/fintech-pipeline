"""在验证集上从 model8_final_dynamic_fusion_v7_strategy_A、CatBoost 和 TGN 中选择最终冠军模型。"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import ID_COL, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, PROJECT_ROOT, RANDOM_SEED  # noqa: E402


COMPONENTS = {
    "model8_current_best": PREDICTION_DIR / "model8_final_dynamic_fusion_v7_strategy_A.csv",
    "model9_catboost": PREDICTION_DIR / "model9_catboost_dynamic_v1_no_customer_type_strategy_A.csv",
    "model10_tgn": PREDICTION_DIR / "model10_tgn_v1_no_customer_type_strategy_A.csv",
}
STEM = "model11_validation_selected_best_strategy_A"


def normalize(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def evaluate(y: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y, score)),
        "pr_auc_average_precision": float(average_precision_score(y, score)),
    }
    positives = int(y.sum())
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
                f"{prefix}_recall": float(hits / positives) if positives else 0.0,
            }
        )
    return result


def load_split(split: str) -> pd.DataFrame:
    merged = None
    for name, path in COMPONENTS.items():
        if not path.exists():
            raise FileNotFoundError(f"缺少候选模型预测文件：{path}")
        frame = pd.read_csv(path)
        frame = frame[frame["split"].eq(split)][[ID_COL, "target", "score"]]
        frame = frame.rename(columns={"score": f"{name}_score"})
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame.drop(columns="target"), on=ID_COL, how="inner")
    if merged is None or len(merged) != 11087:
        raise RuntimeError(f"{split} 候选模型没有共同覆盖全量 11087 个账户。")
    return merged


def candidate_weights(names: list[str]) -> list[dict[str, float]]:
    weights = []
    for name in names:
        weights.append({n: 1.0 if n == name else 0.0 for n in names})
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            for step in range(1, 20):
                w = step / 20
                weights.append({n: (w if n == left else 1 - w if n == right else 0.0) for n in names})
    for a in range(21):
        for b in range(21 - a):
            c = 20 - a - b
            weights.append({names[0]: a / 20, names[1]: b / 20, names[2]: c / 20})
    dedup = {}
    for weight in weights:
        key = tuple(round(weight[name], 4) for name in names)
        dedup[key] = weight
    return list(dedup.values())


def main() -> None:
    valid = load_split("valid")
    test = load_split("test")
    names = list(COMPONENTS)
    y_valid = valid["target"].to_numpy(dtype=int)
    y_test = test["target"].to_numpy(dtype=int)
    valid_scores = {name: normalize(valid[f"{name}_score"].to_numpy()) for name in names}
    test_scores = {name: normalize(test[f"{name}_score"].to_numpy()) for name in names}

    rows = []
    best = None
    for weights in candidate_weights(names):
        valid_score = sum(weights[name] * valid_scores[name] for name in names)
        metrics = evaluate(y_valid, valid_score)
        key = (metrics["pr_auc_average_precision"], metrics["top5pct_recall"], metrics["auc"])
        rows.append(
            {
                **{f"weight_{name}": weights[name] for name in names},
                "valid_auc": metrics["auc"],
                "valid_pr_auc": metrics["pr_auc_average_precision"],
                "valid_top5pct_recall": metrics["top5pct_recall"],
                "valid_top5pct_hits": metrics["top5pct_hits"],
            }
        )
        if best is None or key > best[0]:
            best = (key, weights, metrics, valid_score)
    if best is None:
        raise RuntimeError("没有找到有效最终模型。")

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
    pd.DataFrame(rows).sort_values(
        ["valid_pr_auc", "valid_top5pct_recall", "valid_auc"],
        ascending=False,
    ).to_csv(METRIC_DIR / "final_model_selection_candidates_v8.csv", index=False)

    artifact = {
        "artifact_type": "validation_selected_best_or_rank_ensemble",
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
    (METRIC_DIR / "final_model_selection_metrics_v8.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
