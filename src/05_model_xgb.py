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


def topk_metrics(y_true: np.ndarray, score: np.ndarray, rate: float = 0.05) -> dict:
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


def minmax_score(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros(len(df))
    arr = df[cols].fillna(0).to_numpy(dtype=float)
    lo = arr.min(axis=0)
    hi = arr.max(axis=0)
    scaled = (arr - lo) / np.where(hi > lo, hi - lo, 1.0)
    return scaled.mean(axis=1)


def load_split(split: str, include_graph: bool, include_node2vec: bool) -> pd.DataFrame:
    stat = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    out = stat
    if include_graph:
        graph_path = FEATURE_DIR / f"graph_features_{split}.csv"
        if graph_path.exists():
            out = out.merge(pd.read_csv(graph_path), on=ID_COL, how="left")
    if include_node2vec:
        n2v_path = FEATURE_DIR / f"node2vec_features_{split}.csv"
        if n2v_path.exists():
            out = out.merge(pd.read_csv(n2v_path), on=ID_COL, how="left")
    return out.fillna(0)


def attach_labels(df: pd.DataFrame, strategy: str, filter_strategy_a: bool = True) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    label_col = {
        "A": "label_A_suspect_vs_other",
        "B": "label_B_fraud_related_vs_other",
        "C": "label_C_three_class",
    }[strategy]
    out = df.merge(labels[[ID_COL, label_col, "label_code", "label_text"]], on=ID_COL, how="inner")
    out = out.rename(columns={label_col: "target"})
    out["target_all_accounts"] = out["label_code"].eq(1).astype("int8")
    if strategy == "A" and filter_strategy_a:
        out = out[out["target"] >= 0].copy()
    return out


def output_stem(name: str, suffix: str) -> str:
    return f"{name}_{suffix}" if suffix else name


def feature_columns(df: pd.DataFrame, drop_customer_type: bool, drop_static_profile: bool) -> list[str]:
    drop_cols = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    cols = [c for c in df.columns if c not in drop_cols]
    if drop_customer_type or drop_static_profile:
        cols = [c for c in cols if not c.startswith("customer_type_")]
    if drop_static_profile:
        cols = [c for c in cols if c not in {"region_code", "account_age_months"}]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def run_rule_baseline(strategy: str, suffix: str) -> dict:
    risky_cols = [
        "out_txn_count",
        "out_amount_sum",
        "out_counterparty_nunique",
        "out_large_amount_ratio",
        "counterparty_amount_top_ratio",
        "counterparty_txn_top_ratio",
        "burst_day_txn_ratio",
        "fast_in_out_balance_ratio_24h",
        "prior_in_before_out_out_amount_ratio_24h",
        "multi_in_one_out_count_24h",
        "total_non_positive_amount_count",
        "total_self_loop_count",
        "graph_reciprocal_neighbor_count",
        "graph_two_hop_neighbor_count",
        "graph_path_through_proxy",
        "graph_nb_fast_in_out_balance_ratio_24h_mean",
        "graph_nb_multi_in_one_out_count_24h_mean",
        "pagerank",
    ]
    metrics = {}
    predictions = []
    for split in ["valid", "test"]:
        df = attach_labels(load_split(split, include_graph=True, include_node2vec=False), strategy, filter_strategy_a=False)
        cols = [c for c in risky_cols if c in df.columns]
        score = minmax_score(df, cols)
        candidate_mask = df["target"].ge(0).to_numpy()
        candidate_y = (df["target"].to_numpy(dtype=int) == 1).astype(int) if strategy == "C" else df["target"].to_numpy(dtype=int)
        all_y = df["target_all_accounts"].to_numpy(dtype=int) if strategy == "A" else candidate_y
        metrics[split] = evaluate(candidate_y[candidate_mask], score[candidate_mask])
        metrics[f"{split}_all_accounts"] = evaluate(all_y, score)
        predictions.append(pd.DataFrame({ID_COL: df[ID_COL], "split": split, "target": all_y, "score": score}))
    pd.concat(predictions).to_csv(
        PREDICTION_DIR / f"{output_stem('model0_rule', suffix)}_strategy_{strategy}.csv",
        index=False,
    )
    return metrics


def run_xgb_model(
    strategy: str,
    model_name: str,
    include_graph: bool,
    include_node2vec: bool,
    drop_customer_type: bool,
    drop_static_profile: bool,
    suffix: str,
) -> dict:
    try:
        import xgboost as xgb
    except Exception as exc:
        return {"status": "skipped", "reason": f"缺少 xgboost 依赖: {type(exc).__name__}: {exc}"}

    train_all = attach_labels(load_split("train", include_graph, include_node2vec), strategy, filter_strategy_a=False)
    valid_all = attach_labels(load_split("valid", include_graph, include_node2vec), strategy, filter_strategy_a=False)
    test_all = attach_labels(load_split("test", include_graph, include_node2vec), strategy, filter_strategy_a=False)
    train = train_all[train_all["target"] >= 0].copy() if strategy == "A" else train_all.copy()
    valid = valid_all[valid_all["target"] >= 0].copy() if strategy == "A" else valid_all.copy()
    test = test_all[test_all["target"] >= 0].copy() if strategy == "A" else test_all.copy()

    cols = feature_columns(train, drop_customer_type, drop_static_profile)
    x_train = train[cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=int)
    x_valid = valid[cols].to_numpy(dtype=float)
    y_valid = valid["target"].to_numpy(dtype=int)
    x_test = test[cols].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=int)

    if strategy == "C":
        objective = "multi:softprob"
        params = {
            "objective": objective,
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "seed": RANDOM_SEED,
        }
    else:
        pos = max(1, int(y_train.sum()))
        neg = max(1, int((y_train == 0).sum()))
        params = {
            "objective": "binary:logistic",
            "eval_metric": "aucpr",
            "max_depth": 3,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "scale_pos_weight": neg / pos,
            "seed": RANDOM_SEED,
        }

    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=cols)
    dvalid = xgb.DMatrix(x_valid, label=y_valid, feature_names=cols)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    metrics = {
        "status": "ok",
        "feature_count": len(cols),
        "best_iteration": int(model.best_iteration),
        "candidate_pool_policy": "Strategy A 训练/候选指标剔除受害人；all_accounts 指标保留全量账户排名。",
    }
    predictions = []
    for split, candidate_df, all_df in [("train", train, train_all), ("valid", valid, valid_all), ("test", test, test_all)]:
        all_x = all_df[cols].to_numpy(dtype=float)
        raw_pred = model.predict(xgb.DMatrix(all_x, feature_names=cols))
        if strategy == "C":
            score = raw_pred[:, 1]
            all_y = (all_df["target"].to_numpy(dtype=int) == 1).astype(int)
            candidate_y = (candidate_df["target"].to_numpy(dtype=int) == 1).astype(int)
        else:
            score = raw_pred
            all_y = all_df["target_all_accounts"].to_numpy(dtype=int) if strategy == "A" else all_df["target"].to_numpy(dtype=int)
            candidate_y = candidate_df["target"].to_numpy(dtype=int)
        candidate_ids = set(candidate_df[ID_COL].astype(int))
        candidate_mask = all_df[ID_COL].astype(int).isin(candidate_ids).to_numpy()
        metrics[split] = evaluate(candidate_y, score[candidate_mask])
        metrics[f"{split}_all_accounts"] = evaluate(all_y, score)
        predictions.append(pd.DataFrame({ID_COL: all_df[ID_COL], "split": split, "target": all_y, "score": score}))

    model_stem = output_stem(model_name, suffix)
    model.save_model(str(PREDICTION_DIR / f"{model_stem}_strategy_{strategy}.json"))
    importance = model.get_score(importance_type="gain")
    importance_df = (
        pd.DataFrame(
            [{"feature": feature, "gain": float(gain)} for feature, gain in importance.items()]
        )
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(METRIC_DIR / f"{model_stem}_strategy_{strategy}_feature_importance.csv", index=False)
    pd.concat(predictions).to_csv(PREDICTION_DIR / f"{model_stem}_strategy_{strategy}.csv", index=False)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGBoost baseline and ablation runner.")
    parser.add_argument(
        "--drop-customer-type",
        action="store_true",
        help="删除 customer_type one-hot 特征，用于检验客户类型是否支配模型。",
    )
    parser.add_argument(
        "--drop-static-profile",
        action="store_true",
        help="删除 customer_type、region_code、account_age_months，只保留交易和图相关特征。",
    )
    parser.add_argument(
        "--experiment-suffix",
        default="",
        help="输出文件后缀，避免覆盖原始实验。例如 no_customer_type。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    suffix = args.experiment_suffix.strip()
    if args.drop_static_profile:
        args.drop_customer_type = True
    all_metrics = {}
    metadata = {
        "drop_customer_type": bool(args.drop_customer_type),
        "drop_static_profile": bool(args.drop_static_profile),
        "experiment_suffix": suffix,
        "transaction_time_windows": SPLITS,
        "label_time_available": False,
        "label_time_note": "原始风险标签表未提供标签时间，当前实验只能保证特征按交易时间窗口构建，不能严格验证未来新增标签。",
    }
    all_metrics["_metadata"] = metadata
    for strategy in ["A", "B", "C"]:
        all_metrics[f"{output_stem('model0_rule', suffix)}_strategy_{strategy}"] = run_rule_baseline(strategy, suffix)
        all_metrics[f"{output_stem('model1_xgb_stat', suffix)}_strategy_{strategy}"] = run_xgb_model(
            strategy,
            "model1_xgb_stat",
            include_graph=False,
            include_node2vec=False,
            drop_customer_type=args.drop_customer_type,
            drop_static_profile=args.drop_static_profile,
            suffix=suffix,
        )
        all_metrics[f"{output_stem('model2_xgb_stat_graph', suffix)}_strategy_{strategy}"] = run_xgb_model(
            strategy,
            "model2_xgb_stat_graph",
            include_graph=True,
            include_node2vec=False,
            drop_customer_type=args.drop_customer_type,
            drop_static_profile=args.drop_static_profile,
            suffix=suffix,
        )
        all_metrics[f"{output_stem('model25_xgb_stat_graph_node2vec', suffix)}_strategy_{strategy}"] = run_xgb_model(
            strategy,
            "model25_xgb_stat_graph_node2vec",
            include_graph=True,
            include_node2vec=True,
            drop_customer_type=args.drop_customer_type,
            drop_static_profile=args.drop_static_profile,
            suffix=suffix,
        )

    metrics_name = output_stem("xgb_experiment_metrics", suffix)
    with (METRIC_DIR / f"{metrics_name}.json").open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(json.dumps(all_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
