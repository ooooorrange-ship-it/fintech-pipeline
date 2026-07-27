import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, FEATURE_DIR, ID_COL, LABEL_DIR, METRIC_DIR, PREDICTION_DIR, RANDOM_SEED, SPLITS, SRC_COL, DST_COL, TIME_COL  # noqa: E402


def ensure_dirs() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)


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
    return {
        "auc": float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else 0.0,
        "pr_auc_average_precision": float(average_precision_score(y_true, score)) if y_true.sum() else 0.0,
        **topk_metrics(y_true, score, 0.01),
        **topk_metrics(y_true, score, 0.05),
    }


def output_stem(name: str, suffix: str) -> str:
    return f"{name}_{suffix}" if suffix else name


def attach_labels(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    label_col = {
        "A": "label_A_suspect_vs_other",
        "B": "label_B_fraud_related_vs_other",
        "C": "label_C_three_class",
    }[strategy]
    out = df.merge(labels[[ID_COL, label_col, "label_text"]], on=ID_COL, how="inner")
    out = out.rename(columns={label_col: "target"})
    if strategy == "A":
        out = out[out["target"] >= 0].copy()
    return out


def load_stat_split(split: str) -> pd.DataFrame:
    return pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")


def feature_columns(df: pd.DataFrame, drop_customer_type: bool, drop_static_profile: bool) -> list[str]:
    drop_cols = {ID_COL, "target", "label_text"}
    cols = [c for c in df.columns if c not in drop_cols]
    if drop_customer_type or drop_static_profile:
        cols = [c for c in cols if not c.startswith("customer_type_")]
    if drop_static_profile:
        cols = [c for c in cols if c not in {"region_code", "account_age_months"}]
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]


def safe_nan_to_num(arr: np.ndarray) -> np.ndarray:
    return np.nan_to_num(arr.astype(float), nan=0.0, posinf=0.0, neginf=0.0)


def build_sparse_adjacency(transactions: pd.DataFrame, accounts: pd.DataFrame, start: str, end: str) -> tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix]:
    tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ][[SRC_COL, DST_COL]].drop_duplicates()
    ids = accounts[ID_COL].astype("int64").tolist()
    id_to_idx = {int(acc): idx for idx, acc in enumerate(ids)}
    if tx.empty:
        n = len(ids)
        empty = sp.csr_matrix((n, n), dtype=float)
        return empty, empty, empty

    src_idx = tx[SRC_COL].map(id_to_idx).to_numpy(dtype=int)
    dst_idx = tx[DST_COL].map(id_to_idx).to_numpy(dtype=int)
    n = len(ids)

    data = np.ones(len(tx), dtype=float)
    a_out = sp.csr_matrix((data, (src_idx, dst_idx)), shape=(n, n))
    a_in = sp.csr_matrix((data, (dst_idx, src_idx)), shape=(n, n))
    a_und = ((a_out + a_in) > 0).astype(float).tocsr()
    return row_normalize(a_out), row_normalize(a_in), row_normalize(a_und)


def row_normalize(mat: sp.csr_matrix) -> sp.csr_matrix:
    if mat.nnz == 0:
        return mat.tocsr()
    row_sum = np.asarray(mat.sum(axis=1)).ravel()
    inv = np.zeros_like(row_sum)
    mask = row_sum > 0
    inv[mask] = 1.0 / row_sum[mask]
    return sp.diags(inv).dot(mat).tocsr()


def build_node_matrix(split: str, drop_customer_type: bool, drop_static_profile: bool) -> tuple[pd.DataFrame, list[str]]:
    stat = load_stat_split(split)
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    label_col = "label_A_suspect_vs_other"
    df = stat.merge(labels[[ID_COL, label_col, "label_text"]], on=ID_COL, how="inner").rename(columns={label_col: "target"})
    if drop_customer_type or drop_static_profile:
        df = df[[c for c in df.columns if not c.startswith("customer_type_")]]
    if drop_static_profile:
        df = df[[c for c in df.columns if c not in {"region_code", "account_age_months"}]]
    cols = feature_columns(df, False, False)
    cols = [c for c in cols if c not in {"label_A_suspect_vs_other"}]
    return df, cols


def propagate_features(x: np.ndarray, a_out: sp.csr_matrix, a_in: sp.csr_matrix, a_und: sp.csr_matrix) -> np.ndarray:
    out = a_out.dot(x)
    inn = a_in.dot(x)
    und = a_und.dot(x)
    und2 = a_und.dot(und)
    delta = np.abs(x - und)
    mix = x * (und + 1e-6)
    return np.hstack([x, out, inn, und, und2, delta, mix])


def build_features_for_split(split: str, drop_customer_type: bool, drop_static_profile: bool) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df, cols = build_node_matrix(split, drop_customer_type, drop_static_profile)
    x = safe_nan_to_num(df[cols].to_numpy(dtype=float))
    return df, x, cols


def fit_graph_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_train: np.ndarray,
    x_valid: np.ndarray,
    x_test: np.ndarray,
    train_adj: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    valid_adj: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
    test_adj: tuple[sp.csr_matrix, sp.csr_matrix, sp.csr_matrix],
) -> tuple[LogisticRegression, dict]:
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)

    x_train_p = propagate_features(x_train, *train_adj)
    x_valid_p = propagate_features(x_valid, *valid_adj)
    x_test_p = propagate_features(x_test, *test_adj)

    imputer = SimpleImputer(strategy="constant", fill_value=0.0)
    x_train_p = imputer.fit_transform(x_train_p)
    x_valid_p = imputer.transform(x_valid_p)
    x_test_p = imputer.transform(x_test_p)

    train_mask = train_df["target"].ge(0).to_numpy()
    valid_mask = valid_df["target"].ge(0).to_numpy()
    test_mask = test_df["target"].ge(0).to_numpy()
    y_train = train_df.loc[train_mask, "target"].to_numpy(dtype=int)
    y_valid = valid_df.loc[valid_mask, "target"].to_numpy(dtype=int)
    y_test = test_df.loc[test_mask, "target"].to_numpy(dtype=int)
    x_train_fit = x_train_p[train_mask]

    pos = max(1, int(y_train.sum()))
    neg = max(1, int((y_train == 0).sum()))
    sample_weight = np.where(y_train == 1, neg / pos, 1.0)

    base = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        random_state=RANDOM_SEED,
    )
    base.fit(x_train_fit, y_train, sample_weight=sample_weight)
    train_prob = base.predict_proba(x_train_fit)[:, 1]
    pos_cut = np.quantile(train_prob[y_train == 1], 0.25) if y_train.sum() else 1.0
    hard_neg = (y_train == 0) & (train_prob >= pos_cut)
    sample_weight = sample_weight * np.where(hard_neg, 1.5, 1.0)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight=None,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train_fit, y_train, sample_weight=sample_weight)

    valid_score = model.predict_proba(x_valid_p)[:, 1]
    test_score = model.predict_proba(x_test_p)[:, 1]
    train_score = model.predict_proba(x_train_p)[:, 1]
    metrics = {
        "train": evaluate(y_train, train_score[train_mask]),
        "valid": evaluate(y_valid, valid_score[valid_mask]),
        "test": evaluate(y_test, test_score[test_mask]),
    }
    masks = {"train": train_mask, "valid": valid_mask, "test": test_mask}
    return model, metrics, x_train_p, x_valid_p, x_test_p, masks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轻量异配图传播模型。")
    parser.add_argument("--drop-customer-type", action="store_true", help="删除 customer_type one-hot 特征。")
    parser.add_argument("--drop-static-profile", action="store_true", help="删除 customer_type、region_code、account_age_months。")
    parser.add_argument("--experiment-suffix", default="v3_no_customer_type", help="输出后缀。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    suffix = args.experiment_suffix.strip()
    if args.drop_static_profile:
        args.drop_customer_type = True

    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])

    train_df, x_train, _ = build_features_for_split("train", args.drop_customer_type, args.drop_static_profile)
    valid_df, x_valid, _ = build_features_for_split("valid", args.drop_customer_type, args.drop_static_profile)
    test_df, x_test, _ = build_features_for_split("test", args.drop_customer_type, args.drop_static_profile)

    train_adj = build_sparse_adjacency(transactions, accounts, *SPLITS["train"])
    valid_adj = build_sparse_adjacency(transactions, accounts, *SPLITS["valid"])
    test_adj = build_sparse_adjacency(transactions, accounts, *SPLITS["test"])

    model, metrics, x_train_p, x_valid_p, x_test_p, masks = fit_graph_model(
        train_df,
        valid_df,
        test_df,
        x_train,
        x_valid,
        x_test,
        train_adj,
        valid_adj,
        test_adj,
    )

    valid_score = model.predict_proba(x_valid_p)[:, 1]
    test_score = model.predict_proba(x_test_p)[:, 1]
    train_score = model.predict_proba(x_train_p)[:, 1]

    predictions = pd.concat(
        [
            pd.DataFrame({ID_COL: train_df.loc[masks["train"], ID_COL], "split": "train", "target": train_df.loc[masks["train"], "target"], "score": train_score[masks["train"]]}),
            pd.DataFrame({ID_COL: valid_df.loc[masks["valid"], ID_COL], "split": "valid", "target": valid_df.loc[masks["valid"], "target"], "score": valid_score[masks["valid"]]}),
            pd.DataFrame({ID_COL: test_df.loc[masks["test"], ID_COL], "split": "test", "target": test_df.loc[masks["test"], "target"], "score": test_score[masks["test"]]}),
        ],
        ignore_index=True,
    )
    pred_path = PREDICTION_DIR / f"{output_stem('model3_hetero_prop', suffix)}_strategy_A.csv"
    predictions.to_csv(pred_path, index=False)

    coef = model.coef_.ravel()
    feature_names = []
    base_cols = feature_columns(train_df, args.drop_customer_type, args.drop_static_profile)
    prefixes = ["self", "out_nb", "in_nb", "und_nb", "und2_nb", "delta", "mix"]
    for prefix in prefixes:
        feature_names.extend([f"{prefix}::{col}" for col in base_cols])
    importance = (
        pd.DataFrame({"feature": feature_names, "coef": coef, "abs_coef": np.abs(coef)})
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(METRIC_DIR / f"{output_stem('model3_hetero_prop', suffix)}_strategy_A_feature_importance.csv", index=False)

    report = {
        "_metadata": {
            "drop_customer_type": bool(args.drop_customer_type),
            "drop_static_profile": bool(args.drop_static_profile),
            "experiment_suffix": suffix,
            "model_family": "lightweight_heterophily_propagation",
            "graph_window_policy": "each split uses only transactions inside its own time window",
            "papers": [
                "SplitGNN",
                "IGSL",
                "HGIF",
                "MultiFraud",
                "GNNExplainer",
                "GraphSMOTE",
            ],
        },
        "status": "ok",
        "feature_count": int(x_train_p.shape[1]),
        "train": metrics["train"],
        "valid": metrics["valid"],
        "test": metrics["test"],
    }
    with (METRIC_DIR / f"{output_stem('gnn_experiment_metrics', suffix)}.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
