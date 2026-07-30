"""统一特征口径下的逻辑回归和随机森林基线。"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import FEATURE_DIR, ID_COL, LABEL_DIR, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, RANDOM_SEED, SPLITS  # noqa: E402


EXPERIMENT = "v1_no_customer_type"


def ensure_dirs() -> None:
    for path in [METRIC_DIR, MODEL_DIR, PREDICTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
        positives = int(y_true.sum())
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


def load_split(split: str, drop_static_profile: bool) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"graph_features_{split}.csv"), on=ID_COL, how="left")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"dynamic_graph_features_{split}.csv"), on=ID_COL, how="left")
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    frame = frame.merge(
        labels[[ID_COL, "label_A_suspect_vs_other", "label_code", "label_text"]],
        on=ID_COL,
        how="inner",
    ).rename(columns={"label_A_suspect_vs_other": "target"})
    frame["target_all_accounts"] = frame["label_code"].eq(1).astype("int8")
    frame = frame[[c for c in frame.columns if not c.startswith("customer_type_")]]
    if drop_static_profile:
        frame = frame.drop(columns=[c for c in ["region_code", "account_age_months"] if c in frame])
    return frame.replace([np.inf, -np.inf], np.nan)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=0.2,
                        class_weight="balanced",
                        max_iter=3000,
                        solver="lbfgs",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
        "dynamic_graph_random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=3,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="逻辑回归和随机森林统一传统基线。")
    parser.add_argument("--drop-static-profile", action="store_true", help="额外删除地区和开户时长。")
    parser.add_argument("--experiment-suffix", default=EXPERIMENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    frames = {split: load_split(split, args.drop_static_profile) for split in SPLITS}
    cols = feature_columns(frames["train"])
    train = frames["train"]
    train_mask = train["target"].ge(0).to_numpy()
    x_train = train.loc[train_mask, cols]
    y_train = train.loc[train_mask, "target"].to_numpy(dtype=int)

    report = {
        "_metadata": {
            "experiment_suffix": args.experiment_suffix,
            "feature_policy": "LR/RF use statistical features; dynamic_graph_random_forest uses all graph and dynamic features.",
            "drop_static_profile": bool(args.drop_static_profile),
            "feature_count": len(cols),
            "transaction_time_windows": SPLITS,
            "evaluation_policy": "Strategy A training excludes victims; *_all_accounts keeps all 11087 accounts.",
            "label_time_limitation": "Risk labels have no confirmation time; this is snapshot evaluation, not future-new-label validation.",
        }
    }

    stat_source_cols = set(pd.read_csv(FEATURE_DIR / "stat_features_train.csv", nrows=0).columns)
    stat_cols = [c for c in cols if c in stat_source_cols]
    for model_name, model in build_models().items():
        model_cols = cols if model_name == "dynamic_graph_random_forest" else stat_cols
        model.fit(x_train[model_cols], y_train)
        model_report = {
            "status": "ok",
            "feature_count": len(model_cols),
            "feature_policy": "stat_graph_dynamic" if model_name == "dynamic_graph_random_forest" else "stat_only",
        }
        predictions = []
        for split, frame in frames.items():
            score = model.predict_proba(frame[model_cols])[:, 1]
            candidate_mask = frame["target"].ge(0).to_numpy()
            candidate_y = frame.loc[candidate_mask, "target"].to_numpy(dtype=int)
            all_y = frame["target_all_accounts"].to_numpy(dtype=int)
            model_report[split] = evaluate(candidate_y, score[candidate_mask])
            model_report[f"{split}_all_accounts"] = evaluate(all_y, score)
            predictions.append(
                pd.DataFrame({ID_COL: frame[ID_COL], "split": split, "target": all_y, "score": score})
            )

        prefix = "model7" if model_name == "dynamic_graph_random_forest" else "baseline"
        stem = f"{prefix}_{model_name}_{args.experiment_suffix}_strategy_A"
        joblib.dump(
            {
                "model": model,
                "feature_columns": model_cols,
                "metadata": report["_metadata"],
            },
            MODEL_DIR / f"{stem}.joblib",
        )
        pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{stem}.csv", index=False)
        report[model_name] = model_report

    output = METRIC_DIR / f"traditional_baseline_metrics_{args.experiment_suffix}.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
