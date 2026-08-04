"""五折账户级 Bagging 模型。

每一折只使用 80% 的 Strategy A 账户训练 RandomForest 和 CatBoost，
再对 5 折模型的 rank score 做平均。该模型用于降低账户级过拟合，
并与动态图树融合模型 / 图-树-时序融合模型的时间窗口结果对照。
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import FEATURE_DIR, ID_COL, LABEL_DIR, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, RANDOM_SEED, SPLITS  # noqa: E402


N_SPLITS = 5
EXPERIMENT = "v1_no_customer_type"


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
        prefix = f"top{int(rate * 100)}pct"
        result[f"{prefix}_hits"] = hits
        result[f"{prefix}_recall"] = float(hits / positives) if positives else 0.0
    return result


def load_split(split: str) -> pd.DataFrame:
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
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    return [
        col
        for col in frame.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(frame[col])
    ]


def normalize(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def make_rf(seed_offset: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=350,
        max_depth=7,
        min_samples_leaf=8,
        max_features=0.5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED + seed_offset,
    )


def make_catboost(seed_offset: int, class_weight: float):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=900,
        depth=4,
        learning_rate=0.025,
        l2_leaf_reg=30.0,
        random_strength=1.0,
        class_weights=[1.0, class_weight],
        random_seed=RANDOM_SEED + seed_offset,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def prepare_cat(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame[columns].copy()
    categorical = []
    if "region_code" in out.columns:
        out["region_code"] = out["region_code"].fillna(-1).astype(str)
        categorical.append("region_code")
    for col in out.columns:
        if col not in categorical:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out, categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="五折账户级 Bagging 模型。")
    parser.add_argument("--experiment-suffix", default=EXPERIMENT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [METRIC_DIR, MODEL_DIR, PREDICTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    frames = {split: load_split(split) for split in SPLITS}
    columns = feature_columns(frames["train"])
    train = frames["train"]
    eligible_mask = train["target"].ge(0).to_numpy()
    eligible = train.loc[eligible_mask].reset_index(drop=True)
    x_train = eligible[columns]
    y_train = eligible["target"].to_numpy(dtype=int)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    valid_rf_scores = []
    valid_cat_scores = []
    test_rf_scores = []
    test_cat_scores = []
    fold_reports = []
    saved_models = []

    for fold, (fit_idx, holdout_idx) in enumerate(splitter.split(x_train, y_train), start=1):
        y_fit = y_train[fit_idx]
        positives = max(1, int(y_fit.sum()))
        negatives = max(1, int((y_fit == 0).sum()))
        class_weight = negatives / positives

        rf = make_rf(fold)
        rf.fit(x_train.iloc[fit_idx], y_fit)
        valid_rf = rf.predict_proba(frames["valid"][columns])[:, 1]
        test_rf = rf.predict_proba(frames["test"][columns])[:, 1]
        valid_rf_scores.append(valid_rf)
        test_rf_scores.append(test_rf)

        x_fit_cat, categorical = prepare_cat(x_train.iloc[fit_idx], columns)
        valid_cat_frame, _ = prepare_cat(frames["valid"], columns)
        test_cat_frame, _ = prepare_cat(frames["test"], columns)
        cat = make_catboost(fold, class_weight)
        cat.fit(
            x_fit_cat,
            y_fit,
            cat_features=categorical,
            eval_set=(prepare_cat(x_train.iloc[holdout_idx], columns)[0], y_train[holdout_idx]),
            use_best_model=True,
            early_stopping_rounds=100,
            verbose=False,
        )
        valid_cat_scores.append(cat.predict_proba(valid_cat_frame)[:, 1])
        test_cat_scores.append(cat.predict_proba(test_cat_frame)[:, 1])
        fold_reports.append(
            {
                "fold": fold,
                "fit_account_count": int(len(fit_idx)),
                "holdout_account_count": int(len(holdout_idx)),
                "holdout_positive_count": int(y_train[holdout_idx].sum()),
                "catboost_best_iteration": int(cat.get_best_iteration()),
            }
        )
        saved_models.append({"fold": fold, "random_forest": rf, "catboost": cat})

    valid_rf = np.mean(valid_rf_scores, axis=0)
    valid_cat = np.mean(valid_cat_scores, axis=0)
    test_rf = np.mean(test_rf_scores, axis=0)
    test_cat = np.mean(test_cat_scores, axis=0)
    valid_score = 0.5 * normalize(valid_rf) + 0.5 * normalize(valid_cat)
    test_score = 0.5 * normalize(test_rf) + 0.5 * normalize(test_cat)
    valid_y = frames["valid"]["target_all_accounts"].to_numpy(dtype=int)
    test_y = frames["test"]["target_all_accounts"].to_numpy(dtype=int)

    report = {
        "_metadata": {
            "model_family": "5-fold account-level bagging: regularized RandomForest + CatBoost",
            "experiment_suffix": args.experiment_suffix,
            "feature_policy": "stat + static graph + rolling dynamic graph; customer_type removed",
            "fold_count": N_SPLITS,
            "random_seed": RANDOM_SEED,
            "selection_policy": "equal rank average of five RF predictions and five CatBoost predictions",
            "label_time_limitation": "Risk labels have no confirmation time; this is snapshot evaluation, not future-new-label validation.",
        },
        "status": "ok",
        "folds": fold_reports,
        "valid_all_accounts": evaluate(valid_y, valid_score),
        "test_all_accounts": evaluate(test_y, test_score),
        "valid_random_forest": evaluate(valid_y, normalize(valid_rf)),
        "valid_catboost": evaluate(valid_y, normalize(valid_cat)),
        "test_random_forest": evaluate(test_y, normalize(test_rf)),
        "test_catboost": evaluate(test_y, normalize(test_cat)),
    }

    stem = f"model12_cv_bagged_dynamic_{args.experiment_suffix}_strategy_A"
    joblib.dump(
        {
            "models": saved_models,
            "feature_columns": columns,
            "metadata": report["_metadata"],
        },
        MODEL_DIR / f"{stem}.joblib",
    )
    pd.DataFrame(
        [
            {ID_COL: row[ID_COL], "split": "valid", "target": int(valid_y[idx]), "score": valid_score[idx]}
            for idx, row in frames["valid"].reset_index(drop=True).iterrows()
        ]
        + [
            {ID_COL: row[ID_COL], "split": "test", "target": int(test_y[idx]), "score": test_score[idx]}
            for idx, row in frames["test"].reset_index(drop=True).iterrows()
        ]
    ).to_csv(PREDICTION_DIR / f"{stem}.csv", index=False)
    (METRIC_DIR / f"cv_bagging_metrics_{args.experiment_suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
