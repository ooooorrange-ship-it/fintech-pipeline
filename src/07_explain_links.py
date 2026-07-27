import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    AMOUNT_COL,
    CLEAN_DIR,
    DST_COL,
    EXPLANATION_DIR,
    FEATURE_DIR,
    ID_COL,
    LABEL_DIR,
    PREDICTION_DIR,
    SPLITS,
    SRC_COL,
    TIME_COL,
)


def ensure_dirs() -> None:
    EXPLANATION_DIR.mkdir(parents=True, exist_ok=True)


def find_prediction_file() -> Path:
    candidates = sorted(
        PREDICTION_DIR.glob("model2_xgb_stat_graph_*no_customer_type_strategy_A.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = PREDICTION_DIR / "model2_xgb_stat_graph_strategy_A.csv"
    if fallback.exists():
        return fallback
    raise FileNotFoundError("未找到 Model2 strategy_A 预测文件，请先运行 src/05_model_xgb.py。")


def label_weight(label_code: int) -> float:
    if label_code == 1:
        return 5.0
    if label_code == 2:
        return 3.0
    return 0.0


def split_transactions(transactions: pd.DataFrame, split: str) -> pd.DataFrame:
    start, end = SPLITS[split]
    return transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ].copy()


def scoped_transactions(transactions: pd.DataFrame, split: str, scope: str) -> pd.DataFrame:
    if scope == "split":
        return split_transactions(transactions, split)
    if scope == "history":
        _, end = SPLITS[split]
        return transactions[transactions[TIME_COL].le(pd.Timestamp(end))].copy()
    return transactions.copy()


def build_counterparty_events(tx: pd.DataFrame, root_id: int) -> pd.DataFrame:
    out_tx = tx.loc[tx[SRC_COL].eq(root_id), [DST_COL, TIME_COL, "amount_abs"]].copy()
    out_tx = out_tx.rename(columns={DST_COL: "counterparty_id"})
    out_tx["direction"] = "out"
    in_tx = tx.loc[tx[DST_COL].eq(root_id), [SRC_COL, TIME_COL, "amount_abs"]].copy()
    in_tx = in_tx.rename(columns={SRC_COL: "counterparty_id"})
    in_tx["direction"] = "in"
    return pd.concat([out_tx, in_tx], ignore_index=True)


def counterparty_summary(
    tx: pd.DataFrame,
    root_id: int,
    labels: pd.DataFrame,
    score_map: dict[int, float],
    top_k: int,
) -> pd.DataFrame:
    events = build_counterparty_events(tx, root_id)
    if events.empty:
        return pd.DataFrame()

    grouped = events.groupby("counterparty_id", sort=False).agg(
        direct_txn_count=("amount_abs", "size"),
        direct_amount_sum=("amount_abs", "sum"),
        first_time=(TIME_COL, "min"),
        last_time=(TIME_COL, "max"),
        in_txn_count=("direction", lambda s: int((s == "in").sum())),
        out_txn_count=("direction", lambda s: int((s == "out").sum())),
    ).reset_index()
    grouped["is_reciprocal"] = grouped["in_txn_count"].gt(0) & grouped["out_txn_count"].gt(0)
    grouped = grouped.merge(
        labels[[ID_COL, "label_code", "label_text"]].rename(columns={ID_COL: "counterparty_id"}),
        on="counterparty_id",
        how="left",
    )
    grouped["counterparty_score"] = grouped["counterparty_id"].map(score_map).fillna(0.0)
    max_amount = max(float(grouped["direct_amount_sum"].max()), 1.0)
    grouped["association_score"] = (
        grouped["label_code"].fillna(0).astype(int).map(label_weight)
        + grouped["counterparty_score"] * 3.0
        + np.log1p(grouped["direct_amount_sum"]) / np.log1p(max_amount)
        + grouped["direct_txn_count"].rank(pct=True)
        + grouped["is_reciprocal"].astype(float) * 2.0
    )
    grouped.insert(0, "root_account_id", root_id)
    return grouped.sort_values("association_score", ascending=False).head(top_k)


def add_path_row(
    rows: list[dict],
    root_id: int,
    path_type: str,
    account_1: int,
    account_2: int,
    account_3: int,
    time_1,
    time_2,
    amount_1: float,
    amount_2: float,
    label_map: dict[int, str],
    code_map: dict[int, int],
) -> None:
    delay_hours = (pd.Timestamp(time_2) - pd.Timestamp(time_1)).total_seconds() / 3600
    amount_ratio = amount_2 / amount_1 if amount_1 else 0.0
    label_score = sum(label_weight(code_map.get(int(x), 0)) for x in [account_1, account_2, account_3])
    balance_score = 1.0 - min(abs(amount_ratio - 1.0), 1.0)
    rows.append(
        {
            "root_account_id": root_id,
            "path_type": path_type,
            "account_1": int(account_1),
            "account_2": int(account_2),
            "account_3": int(account_3),
            "label_1": label_map.get(int(account_1), "未知"),
            "label_2": label_map.get(int(account_2), "未知"),
            "label_3": label_map.get(int(account_3), "未知"),
            "time_1": str(time_1),
            "time_2": str(time_2),
            "delay_hours": delay_hours,
            "amount_1": float(amount_1),
            "amount_2": float(amount_2),
            "amount_ratio": amount_ratio,
            "path_evidence_score": label_score + balance_score * 2.0 + max(0.0, 1.0 - delay_hours / 24.0),
        }
    )


def suspicious_paths(
    tx: pd.DataFrame,
    root_id: int,
    labels: pd.DataFrame,
    max_paths: int,
) -> pd.DataFrame:
    label_map = dict(zip(labels[ID_COL].astype(int), labels["label_text"].astype(str)))
    code_map = dict(zip(labels[ID_COL].astype(int), labels["label_code"].astype(int)))
    rows: list[dict] = []
    horizon = np.timedelta64(24, "h")

    incoming = tx.loc[tx[DST_COL].eq(root_id), [SRC_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    outgoing = tx.loc[tx[SRC_COL].eq(root_id), [DST_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    out_times = outgoing[TIME_COL].to_numpy(dtype="datetime64[ns]")
    out_records = outgoing.to_dict("records")
    for rec in incoming.to_dict("records"):
        t1 = np.datetime64(rec[TIME_COL])
        left = np.searchsorted(out_times, t1, side="right")
        right = np.searchsorted(out_times, t1 + horizon, side="right")
        for out_rec in out_records[left : min(right, left + 8)]:
            add_path_row(
                rows,
                root_id,
                "in_root_out_24h",
                rec[SRC_COL],
                root_id,
                out_rec[DST_COL],
                rec[TIME_COL],
                out_rec[TIME_COL],
                rec["amount_abs"],
                out_rec["amount_abs"],
                label_map,
                code_map,
            )

    second_hop_groups = {
        int(k): g[[DST_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
        for k, g in tx.groupby(SRC_COL, sort=False)
    }
    for rec in outgoing.to_dict("records"):
        mid = int(rec[DST_COL])
        second = second_hop_groups.get(mid)
        if second is None or second.empty:
            continue
        second_times = second[TIME_COL].to_numpy(dtype="datetime64[ns]")
        t1 = np.datetime64(rec[TIME_COL])
        left = np.searchsorted(second_times, t1, side="right")
        right = np.searchsorted(second_times, t1 + horizon, side="right")
        for second_rec in second.to_dict("records")[left : min(right, left + 5)]:
            add_path_row(
                rows,
                root_id,
                "root_mid_out_24h",
                root_id,
                mid,
                second_rec[DST_COL],
                rec[TIME_COL],
                second_rec[TIME_COL],
                rec["amount_abs"],
                second_rec["amount_abs"],
                label_map,
                code_map,
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("path_evidence_score", ascending=False).head(max_paths)


def account_evidence(top_accounts: pd.DataFrame, split: str, tx: pd.DataFrame) -> pd.DataFrame:
    stat = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    graph = pd.read_csv(FEATURE_DIR / f"graph_features_{split}.csv")
    cols = [
        ID_COL,
        "total_txn_count",
        "total_amount_sum",
        "counterparty_amount_top_ratio",
        "burst_day_txn_ratio",
        "fast_in_out_balance_ratio_24h",
        "prior_in_before_out_out_amount_ratio_24h",
        "multi_in_one_out_count_24h",
        "graph_total_degree",
        "graph_two_hop_neighbor_count",
        "graph_path_through_proxy",
        "pagerank",
    ]
    feature = stat.merge(graph, on=ID_COL, how="left")
    cols = [c for c in cols if c in feature.columns]
    evidence = top_accounts.merge(feature[cols], on=ID_COL, how="left")

    out_events = tx[[SRC_COL, DST_COL, "amount_abs"]].rename(columns={SRC_COL: ID_COL, DST_COL: "counterparty_id"})
    in_events = tx[[DST_COL, SRC_COL, "amount_abs"]].rename(columns={DST_COL: ID_COL, SRC_COL: "counterparty_id"})
    events = pd.concat([out_events, in_events], ignore_index=True)
    explain_stats = events.groupby(ID_COL, sort=False).agg(
        explain_txn_count=("amount_abs", "size"),
        explain_amount_sum=("amount_abs", "sum"),
        explain_counterparty_count=("counterparty_id", "nunique"),
    ).reset_index()
    return evidence.merge(explain_stats, on=ID_COL, how="left").fillna(0)


def active_prediction_candidates(predictions: pd.DataFrame, tx: pd.DataFrame, allow_inactive: bool) -> pd.DataFrame:
    if allow_inactive:
        return predictions
    active_ids = set(tx[SRC_COL].astype(int)).union(set(tx[DST_COL].astype(int)))
    out = predictions[predictions[ID_COL].astype(int).isin(active_ids)].copy()
    return out if not out.empty else predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="输出高风险账户关联账户和可疑链路解释。")
    parser.add_argument("--prediction-file", default="", help="默认自动选择最新的 no_customer_type Model2 预测文件。")
    parser.add_argument("--split", default="test", choices=sorted(SPLITS), help="解释哪个时间窗口。")
    parser.add_argument("--top-n", type=int, default=5, help="解释 Top N 高风险账户。")
    parser.add_argument("--top-k-counterparties", type=int, default=20, help="每个高风险账户输出 Top K 关联账户。")
    parser.add_argument("--max-paths", type=int, default=50, help="每个高风险账户最多输出多少条可疑链路。")
    parser.add_argument(
        "--tx-scope",
        default="history",
        choices=["split", "history", "full"],
        help="链路解释用当前窗口、截至当前窗口结束的历史交易，或全量交易。",
    )
    parser.add_argument("--confirmed-only", action="store_true", help="只解释已确认嫌疑人，用于典型案例复盘。")
    parser.add_argument("--allow-inactive", action="store_true", help="允许解释当前窗口无交易的高分账户。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    pred_path = Path(args.prediction_file) if args.prediction_file else find_prediction_file()
    predictions = pd.read_csv(pred_path)
    predictions = predictions[predictions["split"].eq(args.split)].copy()
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    tx = scoped_transactions(transactions, args.split, args.tx_scope)

    predictions = predictions.merge(labels[[ID_COL, "label_code", "label_text"]], on=ID_COL, how="left")
    if args.confirmed_only:
        predictions = predictions[predictions["label_code"].eq(1)].copy()
    score_map = dict(zip(predictions[ID_COL].astype(int), predictions["score"].astype(float)))
    candidates = active_prediction_candidates(predictions, tx, args.allow_inactive)
    top_accounts = candidates.sort_values("score", ascending=False).head(args.top_n)

    evidence = account_evidence(top_accounts, args.split, tx)
    associations = []
    paths = []
    for account_id in top_accounts[ID_COL].astype(int):
        associations.append(counterparty_summary(tx, account_id, labels, score_map, args.top_k_counterparties))
        paths.append(suspicious_paths(tx, account_id, labels, args.max_paths))

    non_empty_associations = [x for x in associations if not x.empty]
    non_empty_paths = [x for x in paths if not x.empty]
    association_df = pd.concat(non_empty_associations, ignore_index=True) if non_empty_associations else pd.DataFrame()
    path_df = pd.concat(non_empty_paths, ignore_index=True) if non_empty_paths else pd.DataFrame()

    stem = pred_path.stem.replace("_strategy_A", "")
    evidence_path = EXPLANATION_DIR / f"{stem}_{args.split}_top_accounts_evidence.csv"
    association_path = EXPLANATION_DIR / f"{stem}_{args.split}_top20_associations.csv"
    path_path = EXPLANATION_DIR / f"{stem}_{args.split}_suspicious_paths.csv"

    evidence.to_csv(evidence_path, index=False)
    association_df.to_csv(association_path, index=False)
    path_df.to_csv(path_path, index=False)

    if association_df.empty:
        related_hit_rate = 0.0
    else:
        related_hit_rate = float(association_df["label_code"].isin([1, 2]).mean())

    report = {
        "prediction_file": str(pred_path),
        "split": args.split,
        "tx_scope": args.tx_scope,
        "top_account_count": int(len(evidence)),
        "association_rows": int(len(association_df)),
        "suspicious_path_rows": int(len(path_df)),
        "top_association_risk_label_hit_rate": related_hit_rate,
        "outputs": {
            "top_accounts_evidence": str(evidence_path),
            "top20_associations": str(association_path),
            "suspicious_paths": str(path_path),
        },
    }
    report_path = EXPLANATION_DIR / f"{stem}_{args.split}_explanation_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
