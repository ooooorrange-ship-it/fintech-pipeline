"""五折交叉验证与时序留出过拟合审计。

审计口径：
1. 账户级五折 StratifiedKFold：检查模型对未参与训练账户的泛化能力。
2. 时间窗口 valid/test：检查同一账户在不同交易观察窗口上的稳定性。
3. 随机五折不替代比赛要求的时间切分，只作为补充的过拟合诊断。

由于风险标签没有确认时间，所有结果都不能解释成严格的未来新增标签预测。
"""

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
from config import FEATURE_DIR, ID_COL, LABEL_DIR, METRIC_DIR, PREDICTION_DIR, RANDOM_SEED, SPLITS  # noqa: E402


N_SPLITS = 5
EXPERIMENT = "v1_no_customer_type"


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc": float(average_precision_score(y_true, score)),
    }
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
        key = f"top{int(rate * 100)}pct"
        result[f"{key}_hits"] = hits
        result[f"{key}_recall"] = float(hits / positives) if positives else 0.0
    return result


def load_feature_frame(split: str) -> pd.DataFrame:
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


def make_rf() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def make_catboost():
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=700,
        depth=6,
        learning_rate=0.03,
        l2_leaf_reg=10.0,
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def prepare_catboost(frame: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = frame[columns].copy()
    categorical = []
    if "region_code" in out.columns:
        out["region_code"] = out["region_code"].fillna(-1).astype(str)
        categorical.append("region_code")
    for col in out.columns:
        if col not in categorical:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out, categorical


def summarize_fold_metrics(fold_rows: list[dict], model_name: str) -> dict:
    frame = pd.DataFrame([row for row in fold_rows if row["model"] == model_name])
    result = {
        "model": model_name,
        "fold_count": int(len(frame)),
        "positive_count_per_fold": frame["validation_positive_count"].astype(int).tolist(),
    }
    for metric in ["train_pr_auc", "validation_pr_auc", "train_auc", "validation_auc", "validation_top5pct_recall"]:
        result[f"{metric}_mean"] = float(frame[metric].mean())
        result[f"{metric}_std"] = float(frame[metric].std(ddof=0))
    result["pr_auc_train_validation_gap_mean"] = float(
        (frame["train_pr_auc"] - frame["validation_pr_auc"]).mean()
    )
    result["auc_train_validation_gap_mean"] = float(
        (frame["train_auc"] - frame["validation_auc"]).mean()
    )
    result["overfit_flag"] = bool(
        result["pr_auc_train_validation_gap_mean"] > 0.20
        or result["validation_pr_auc_std"] > 0.10
    )
    result["stability_conclusion"] = (
        "需要关注训练-验证差距或折间波动"
        if result["overfit_flag"]
        else "五折结果相对稳定，未发现明显账户级过拟合"
    )
    return result


def temporal_holdout_summary() -> list[dict]:
    files = {
        "model8_final_dynamic_fusion_v7_strategy_A": METRIC_DIR / "final_dynamic_fusion_metrics_v7.json",
        "model11_validation_selected_best_strategy_A": METRIC_DIR / "final_model_selection_metrics_v8.json",
        "CatBoost": METRIC_DIR / "catboost_metrics_v1_no_customer_type.json",
        "TGN": METRIC_DIR / "tgn_metrics_v1_no_customer_type.json",
        "Model12_cv_bagging": METRIC_DIR / "cv_bagging_metrics_v1_no_customer_type.json",
    }
    rows = []
    for model_name, path in files.items():
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        row = {"model": model_name}
        for split in ["train_all_accounts", "valid_all_accounts", "test_all_accounts"]:
            metrics = report.get(split)
            if not metrics:
                continue
            row[f"{split}_auc"] = metrics["auc"]
            row[f"{split}_pr_auc"] = metrics["pr_auc_average_precision"]
            row[f"{split}_top5pct_recall"] = metrics["top5pct_recall"]
        if "valid_all_accounts_pr_auc" in row and "test_all_accounts_pr_auc" in row:
            row["valid_test_pr_auc_gap"] = row["valid_all_accounts_pr_auc"] - row["test_all_accounts_pr_auc"]
        rows.append(row)
    return rows


def main() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    frames = {split: load_feature_frame(split) for split in SPLITS}
    train = frames["train"]
    columns = feature_columns(train)
    eligible = train["target"].ge(0).to_numpy()
    data = train.loc[eligible].reset_index(drop=True)
    x_numeric = data[columns]
    y = data["target"].to_numpy(dtype=int)

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_rows = []
    oof_scores = {
        "dynamic_random_forest": np.zeros(len(data), dtype=float),
        "catboost": np.zeros(len(data), dtype=float),
    }

    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(x_numeric, y), start=1):
        x_fit = x_numeric.iloc[fit_idx]
        y_fit = y[fit_idx]
        x_valid = x_numeric.iloc[valid_idx]
        y_valid = y[valid_idx]
        positives = max(1, int(y_fit.sum()))
        negatives = max(1, int((y_fit == 0).sum()))

        rf = make_rf()
        rf.fit(x_fit, y_fit)
        rf_train_score = rf.predict_proba(x_fit)[:, 1]
        rf_valid_score = rf.predict_proba(x_valid)[:, 1]
        oof_scores["dynamic_random_forest"][valid_idx] = rf_valid_score
        train_metrics = evaluate(y_fit, rf_train_score)
        valid_metrics = evaluate(y_valid, rf_valid_score)
        fold_rows.append(
            {
                "fold": fold,
                "model": "dynamic_random_forest",
                "fit_account_count": len(fit_idx),
                "validation_account_count": len(valid_idx),
                "validation_positive_count": int(y_valid.sum()),
                "train_auc": train_metrics["auc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "validation_auc": valid_metrics["auc"],
                "validation_pr_auc": valid_metrics["pr_auc"],
                "validation_top5pct_recall": valid_metrics["top5pct_recall"],
                "class_weight_positive": negatives / positives,
            }
        )

        x_fit_cat, categorical = prepare_catboost(x_fit, columns)
        x_valid_cat, _ = prepare_catboost(x_valid, columns)
        catboost = make_catboost()
        catboost.set_params(class_weights=[1.0, negatives / positives])
        catboost.fit(
            x_fit_cat,
            y_fit,
            cat_features=categorical,
            eval_set=(x_valid_cat, y_valid),
            use_best_model=True,
            early_stopping_rounds=80,
            verbose=False,
        )
        cat_train_score = catboost.predict_proba(x_fit_cat)[:, 1]
        cat_valid_score = catboost.predict_proba(x_valid_cat)[:, 1]
        oof_scores["catboost"][valid_idx] = cat_valid_score
        train_metrics = evaluate(y_fit, cat_train_score)
        valid_metrics = evaluate(y_valid, cat_valid_score)
        fold_rows.append(
            {
                "fold": fold,
                "model": "catboost",
                "fit_account_count": len(fit_idx),
                "validation_account_count": len(valid_idx),
                "validation_positive_count": int(y_valid.sum()),
                "train_auc": train_metrics["auc"],
                "train_pr_auc": train_metrics["pr_auc"],
                "validation_auc": valid_metrics["auc"],
                "validation_pr_auc": valid_metrics["pr_auc"],
                "validation_top5pct_recall": valid_metrics["top5pct_recall"],
                "class_weight_positive": negatives / positives,
            }
        )

    fold_frame = pd.DataFrame(fold_rows)
    summary = [summarize_fold_metrics(fold_rows, name) for name in ["dynamic_random_forest", "catboost"]]
    oof_report = {}
    for name, score in oof_scores.items():
        oof_report[name] = evaluate(y, score)
    oof_predictions = pd.DataFrame({ID_COL: data[ID_COL], "target": y})
    for name, score in oof_scores.items():
        oof_predictions[f"{name}_oof_score"] = score
    oof_predictions.to_csv(PREDICTION_DIR / "five_fold_oof_predictions_v1.csv", index=False)
    fold_frame.to_csv(METRIC_DIR / "five_fold_cv_fold_metrics_v1.csv", index=False)
    report = {
        "status": "ok",
        "cross_validation_policy": {
            "method": "5-fold StratifiedKFold on Strategy A eligible accounts",
            "n_splits": N_SPLITS,
            "shuffle": True,
            "random_seed": RANDOM_SEED,
            "purpose": "account-level generalization and overfit diagnosis",
            "not_a_replacement_for_temporal_split": True,
        },
        "data": {
            "train_account_count_all": int(len(train)),
            "train_account_count_eligible": int(len(data)),
            "positive_count": int(y.sum()),
            "feature_count": int(len(columns)),
        },
        "models": summary,
        "oof_metrics": oof_report,
        "temporal_holdout_models": temporal_holdout_summary(),
        "interpretation": [
            "五折验证折间指标用于判断账户级泛化和稳定性。",
            "train/valid/test 时间窗口仍是比赛正式评估口径。",
            "标签表没有确认时间，不能把任何结果表述为严格未来新增标签预测。",
        ],
    }
    (METRIC_DIR / "five_fold_overfit_audit_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
