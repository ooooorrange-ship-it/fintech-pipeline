import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import FEATURE_DIR, ID_COL, LABEL_DIR, METRIC_DIR, PREDICTION_DIR, RANDOM_SEED, SPLITS  # noqa: E402


def ensure_dirs() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y = y_true[order]
    positives = y.sum()
    if positives == 0:
        return 0.0
    precision_at_k = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision_at_k * y).sum() / positives)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def topk_metrics(y_true: np.ndarray, score: np.ndarray, rate: float) -> dict:
    k = max(1, int(np.ceil(len(y_true) * rate)))
    order = np.argsort(-score)[:k]
    hit = int(y_true[order].sum())
    total_pos = int(y_true.sum())
    return {
        f"top{int(rate * 100)}pct_k": int(k),
        f"top{int(rate * 100)}pct_hits": hit,
        f"top{int(rate * 100)}pct_precision": float(hit / k),
        f"top{int(rate * 100)}pct_recall": float(hit / total_pos) if total_pos else 0.0,
    }


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    out = {
        "auc": roc_auc(y_true, score),
        "pr_auc_average_precision": average_precision(y_true, score),
    }
    out.update(topk_metrics(y_true, score, 0.01))
    out.update(topk_metrics(y_true, score, 0.05))
    return out


def attach_labels(df: pd.DataFrame, filter_strategy_a: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    out = df.merge(labels[[ID_COL, "label_A_suspect_vs_other", "label_code", "label_text"]], on=ID_COL, how="inner")
    out = out.rename(columns={"label_A_suspect_vs_other": "target"})
    out["target_all_accounts"] = out["label_code"].eq(1).astype("int8")
    if filter_strategy_a:
        out = out[out["target"] >= 0].copy()
    return out


def load_split(split: str, include_static_graph: bool, include_dynamic: bool) -> pd.DataFrame:
    stat = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    out = stat
    if include_static_graph:
        out = out.merge(pd.read_csv(FEATURE_DIR / f"graph_features_{split}.csv"), on=ID_COL, how="left")
    if include_dynamic:
        out = out.merge(pd.read_csv(FEATURE_DIR / f"dynamic_graph_features_{split}.csv"), on=ID_COL, how="left")
    return out.fillna(0)


def feature_columns(df: pd.DataFrame, drop_customer_type: bool, drop_static_profile: bool) -> list[str]:
    drop_cols = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    cols = [c for c in df.columns if c not in drop_cols]
    if drop_customer_type or drop_static_profile:
        cols = [c for c in cols if not c.startswith("customer_type_")]
    if drop_static_profile:
        cols = [c for c in cols if c not in {"region_code", "account_age_months"}]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def output_stem(name: str, suffix: str) -> str:
    return f"{name}_{suffix}" if suffix else name


def fit_xgb(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    suffix: str,
    train_all: pd.DataFrame,
    valid_all: pd.DataFrame,
    test_all: pd.DataFrame,
) -> dict:
    try:
        import xgboost as xgb
    except Exception as exc:
        return {"status": "skipped", "reason": f"缺少 xgboost 依赖: {type(exc).__name__}: {exc}"}

    x_train = train[cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    x_valid = valid[cols].to_numpy(dtype=float)
    y_valid = valid["target"].to_numpy(dtype=int)
    x_test = test[cols].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=int)

    pos = max(1, int(y_train.sum()))
    neg = max(1, int((y_train == 0).sum()))
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 3,
        "eta": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "lambda": 2.0,
        "alpha": 0.5,
        "scale_pos_weight": neg / pos,
        "seed": RANDOM_SEED,
    }

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=cols)
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=cols)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=600,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=40,
        verbose_eval=False,
    )

    predictions = []
    metrics = {
        "status": "ok",
        "feature_count": len(cols),
        "best_iteration": int(model.best_iteration),
        "candidate_pool_policy": "训练和候选池指标剔除受害人；all_accounts 指标保留全量账户排名。",
    }
    for split, df, x_arr in [("valid", valid, x_valid), ("test", test, x_test)]:
        score = model.predict(xgb.DMatrix(x_arr, feature_names=cols))
        y = df["target"].to_numpy(dtype=int)
        metrics[split] = evaluate(y, score)
    for split, all_df in [("train", train_all), ("valid", valid_all), ("test", test_all)]:
        all_x = all_df[cols].to_numpy(dtype=float)
        score = model.predict(xgb.DMatrix(all_x, feature_names=cols))
        all_y = all_df["target_all_accounts"].to_numpy(dtype=int)
        metrics[f"{split}_all_accounts"] = evaluate(all_y, score)
        predictions.append(pd.DataFrame({ID_COL: all_df[ID_COL], "split": split, "target": all_y, "score": score}))

    stem = output_stem("model5_xgb_dynamic_graph", suffix)
    model.save_model(str(PREDICTION_DIR / f"{stem}_strategy_A.json"))
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{stem}_strategy_A.csv", index=False)

    importance = model.get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame([{"feature": k, "gain": float(v)} for k, v in importance.items()])
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(METRIC_DIR / f"{stem}_strategy_A_feature_importance.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="动态资金图谱 XGBoost 识别模型。")
    parser.add_argument("--drop-customer-type", action="store_true", help="删除 customer_type one-hot 特征。")
    parser.add_argument("--drop-static-profile", action="store_true", help="删除 customer_type、region_code、account_age_months。")
    parser.add_argument("--dynamic-only", action="store_true", help="只使用动态资金图谱特征，不拼接传统统计和静态图特征。")
    parser.add_argument("--experiment-suffix", default="v6_rolling_memory_dynamic_no_customer_type", help="输出后缀。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if args.drop_static_profile:
        args.drop_customer_type = True
    suffix = args.experiment_suffix.strip()

    include_static_graph = not args.dynamic_only
    train_all = attach_labels(load_split("train", include_static_graph, include_dynamic=True), filter_strategy_a=False)
    valid_all = attach_labels(load_split("valid", include_static_graph, include_dynamic=True), filter_strategy_a=False)
    test_all = attach_labels(load_split("test", include_static_graph, include_dynamic=True), filter_strategy_a=False)
    train = train_all[train_all["target"] >= 0].copy()
    valid = valid_all[valid_all["target"] >= 0].copy()
    test = test_all[test_all["target"] >= 0].copy()

    if args.dynamic_only:
        keep = [ID_COL, "target", "label_text"] + [c for c in train.columns if c.startswith("dyn_")]
        train = train[[c for c in keep if c in train.columns]]
        valid = valid[[c for c in keep if c in valid.columns]]
        test = test[[c for c in keep if c in test.columns]]

    cols = feature_columns(train, args.drop_customer_type, args.drop_static_profile)
    metrics = fit_xgb(train, valid, test, cols, suffix, train_all=train_all, valid_all=valid_all, test_all=test_all)
    report = {
        "_metadata": {
            "model_family": "dynamic_graph_xgboost",
            "experiment_suffix": suffix,
            "dynamic_only": bool(args.dynamic_only),
            "drop_customer_type": bool(args.drop_customer_type),
            "drop_static_profile": bool(args.drop_static_profile),
            "transaction_time_windows": SPLITS,
            "core_requirement_mapping": {
                "anonymized_account_nodes": "account_id",
                "anonymized_transfer_edges": "src_account_id -> dst_account_id",
                "time_bucket": "dyn_week_bucket / dyn_day_bucket / dyn_hour_bucket / dyn_part_of_day",
                "amount_bin": "dyn_amount_bin from train quantiles",
                "temporal_motif": "dyn_motif_* 快进快出、多入一出、一入多出事件流模体",
                "temporal_memory": "dyn_mem_* 时间衰减节点记忆，半衰期覆盖 1h/6h/24h/7d",
                "risk_label": "Strategy A suspect vs other, victim filtered",
            },
            "leakage_policy": "动态特征按滚动历史窗口构建，只使用截至 split end 的交易；金额分箱边界只由 train 计算；标签字段不进入特征。",
        },
        "model5_xgb_dynamic_graph_strategy_A": metrics,
    }
    out_path = METRIC_DIR / f"{output_stem('dynamic_graph_experiment_metrics', suffix)}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
