"""CatBoost 动态资金图谱特征模型。

说明：
1. 输入仍然是统计、静态图和滚动动态图特征，保持与动态图树融合模型可比。
2. 默认删除 customer_type，region_code 作为类别特征保留。
3. 训练只使用 Strategy A 的嫌疑人/其它账户，受害人不进入损失。
4. 验证集只用于早停和后续模型选择，测试集只做一次正式报告。
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

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
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
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


def prepare_features(frame: pd.DataFrame, drop_static_profile: bool) -> tuple[pd.DataFrame, list[str]]:
    excluded = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    features = frame[[c for c in frame.columns if c not in excluded]].copy()
    categorical = []
    if not drop_static_profile and "region_code" in features:
        features["region_code"] = features["region_code"].fillna(-1).astype(str)
        categorical.append("region_code")
    for col in features.columns:
        if col not in categorical:
            features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)
    return features, categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CatBoost 动态资金图谱特征模型。")
    parser.add_argument("--drop-static-profile", action="store_true")
    parser.add_argument("--experiment-suffix", default=EXPERIMENT)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--l2-leaf-reg", type=float, default=10.0)
    parser.add_argument("--patience", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise SystemExit("缺少 catboost，请先安装：pip install catboost") from exc

    frames = {split: load_split(split, args.drop_static_profile) for split in SPLITS}
    feature_frames = {}
    categorical = []
    for split, frame in frames.items():
        feature_frames[split], categorical = prepare_features(frame, args.drop_static_profile)

    train = frames["train"]
    train_mask = train["target"].ge(0).to_numpy()
    x_train = feature_frames["train"].loc[train_mask]
    y_train = train.loc[train_mask, "target"].to_numpy(dtype=int)
    valid_mask = frames["valid"]["target"].ge(0).to_numpy()
    x_valid = feature_frames["valid"].loc[valid_mask]
    y_valid = frames["valid"].loc[valid_mask, "target"].to_numpy(dtype=int)
    positives = max(1, int(y_train.sum()))
    negatives = max(1, int((y_train == 0).sum()))

    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=args.l2_leaf_reg,
        class_weights=[1.0, negatives / positives],
        random_seed=RANDOM_SEED,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(
        x_train,
        y_train,
        cat_features=categorical,
        eval_set=(x_valid, y_valid),
        use_best_model=True,
        early_stopping_rounds=args.patience,
        verbose=False,
    )

    report = {
        "_metadata": {
            "model_family": "CatBoost dynamic graph feature model",
            "experiment_suffix": args.experiment_suffix,
            "feature_policy": (
                "stat + static graph + rolling dynamic graph; customer_type removed"
                if not args.drop_static_profile
                else "transaction + graph + rolling dynamic features; static profile removed"
            ),
            "categorical_features": categorical,
            "feature_count": int(feature_frames["train"].shape[1]),
            "class_weights": [1.0, negatives / positives],
            "evaluation_policy": "Strategy A training excludes victims; formal metrics use all accounts.",
            "label_time_limitation": "Risk labels have no confirmation time; this is snapshot evaluation, not future-new-label validation.",
            "random_seed": RANDOM_SEED,
            "best_iteration": int(model.get_best_iteration()),
        },
        "status": "ok",
    }
    predictions = []
    for split, frame in frames.items():
        score = model.predict_proba(feature_frames[split])[:, 1]
        candidate_mask = frame["target"].ge(0).to_numpy()
        all_y = frame["target_all_accounts"].to_numpy(dtype=int)
        report[split] = evaluate(
            frame.loc[candidate_mask, "target"].to_numpy(dtype=int),
            score[candidate_mask],
        )
        report[f"{split}_all_accounts"] = evaluate(all_y, score)
        predictions.append(pd.DataFrame({ID_COL: frame[ID_COL], "split": split, "target": all_y, "score": score}))

    stem = f"model9_catboost_dynamic_{args.experiment_suffix}_strategy_A"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_frames["train"].columns.tolist(),
            "categorical_features": categorical,
            "metadata": report["_metadata"],
        },
        MODEL_DIR / f"{stem}.joblib",
    )
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{stem}.csv", index=False)
    (METRIC_DIR / f"catboost_metrics_{args.experiment_suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
