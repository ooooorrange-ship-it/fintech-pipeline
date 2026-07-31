import argparse
import json
import math
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    AMOUNT_COL,
    CLEAN_DIR,
    DOCS_DIR,
    DST_COL,
    ID_COL,
    LABEL_DIR,
    OUTPUT_DIR,
    PREDICTION_DIR,
    SPLITS,
    SRC_COL,
    TIME_COL,
)


DYNAMIC_GRAPH_DIR = OUTPUT_DIR / "dynamic_graph"
LAYERED_DIR = OUTPUT_DIR / "explanations" / "layered"


def ensure_dirs() -> None:
    DYNAMIC_GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)


def json_records(path: Path) -> list[dict]:
    """读取分层解释 CSV，并把缺失值转换成可嵌入 HTML 的 JSON 值。"""
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    frame = frame.astype(object).where(pd.notna(frame), None)
    rows = []
    for row in frame.to_dict("records"):
        normalized = {}
        for key, value in row.items():
            if isinstance(value, (np.integer, np.floating)):
                normalized[key] = value.item()
            else:
                normalized[key] = value
        rows.append(normalized)
    return rows


def load_layered_data() -> dict:
    """加载 59 个审计账户、Top30 队列及其解释证据。"""
    coverage_path = LAYERED_DIR / "layered_explainability_coverage.json"
    coverage = json.loads(coverage_path.read_text(encoding="utf-8")) if coverage_path.exists() else {}
    report_path = DOCS_DIR / "layered_judgement_report_samples.md"
    return {
        "coverage": coverage,
        "audit": json_records(LAYERED_DIR / "confirmed_suspect_explainability_audit.csv"),
        "recovery": json_records(LAYERED_DIR / "suspect_link_recovery_queue.csv"),
        "queue": json_records(LAYERED_DIR / "risk_review_queue_active_accounts.csv"),
        "associations": json_records(LAYERED_DIR / "risk_review_queue_top20_associations.csv"),
        "paths": json_records(LAYERED_DIR / "risk_review_queue_suspicious_paths.csv"),
        "structures": json_records(LAYERED_DIR / "risk_review_queue_fund_flow_structures.csv"),
        "judgement_report_markdown": report_path.read_text(encoding="utf-8") if report_path.exists() else "",
    }


def find_prediction_file() -> Path:
    patterns = [
        "model11_validation_selected_best_strategy_A.csv",
        "model8_final_dynamic_fusion_v7_strategy_A.csv",
        "model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
        "model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
        "model2_xgb_stat_graph_v2_no_customer_type_strategy_A.csv",
    ]
    for name in patterns:
        path = PREDICTION_DIR / name
        if path.exists():
            return path
    candidates = sorted(
        PREDICTION_DIR.glob("*no_customer_type_strategy_A.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError("未找到 no_customer_type strategy_A 预测文件，请先运行模型脚本。")


def parse_account_ids(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def observation_window(transactions: pd.DataFrame, split: str, history_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, end = SPLITS[split]
    end_ts = pd.Timestamp(end)
    start_ts = max(transactions[TIME_COL].min(), end_ts - pd.Timedelta(days=history_days))
    return start_ts, end_ts


def monthly_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(start=start_ts.normalize().replace(day=1), end=end_ts, freq="MS")
    windows = []
    for month_start in starts:
        win_start = max(start_ts, month_start)
        win_end = min(end_ts, month_start + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23, minutes=59, seconds=59))
        if win_end >= start_ts and win_start <= end_ts:
            windows.append((month_start.strftime("%Y-%m"), pd.Timestamp(win_start), pd.Timestamp(win_end)))
    return windows


def weekly_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    starts = pd.date_range(start=start_ts.normalize(), end=end_ts, freq="7D")
    windows = []
    for idx, win_start in enumerate(starts, start=1):
        win_end = min(end_ts, win_start + pd.Timedelta(days=7) - pd.Timedelta(seconds=1))
        windows.append((f"W{idx:02d}", pd.Timestamp(win_start), pd.Timestamp(win_end)))
    return windows


def build_windows(start_ts: pd.Timestamp, end_ts: pd.Timestamp, mode: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if mode == "weekly":
        return weekly_windows(start_ts, end_ts)
    return monthly_windows(start_ts, end_ts)


def amount_bin_name(amount_sum: float, edges: list[float]) -> str:
    for idx in range(len(edges) - 1):
        if edges[idx] <= amount_sum <= edges[idx + 1]:
            return f"b{idx}"
    return f"b{max(0, len(edges) - 2)}"


def train_amount_edges(transactions: pd.DataFrame, bin_count: int = 5) -> list[float]:
    start, end = SPLITS["train"]
    tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ]
    edges = tx["amount_abs"].quantile(np.linspace(0, 1, bin_count + 1)).to_numpy(dtype=float)
    edges = sorted(set(float(x) for x in edges if np.isfinite(x)))
    if len(edges) < 2:
        max_amount = float(tx["amount_abs"].max()) if len(tx) else 1.0
        edges = [0.0, max_amount + 1.0]
    edges[0] = min(0.0, edges[0])
    edges[-1] = edges[-1] + 1e-9
    return edges


def load_inputs(prediction_file: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = pd.read_csv(prediction_file)
    predictions = predictions[predictions["split"].eq(split)].copy()
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    transactions["amount_abs"] = transactions.get("amount_abs", transactions[AMOUNT_COL].abs())
    predictions = predictions.merge(labels[[ID_COL, "label_code", "label_text"]], on=ID_COL, how="left")
    return predictions, labels, transactions


def choose_accounts(predictions: pd.DataFrame, transactions: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.account_ids:
        ids = parse_account_ids(args.account_ids)
        chosen = predictions[predictions[ID_COL].astype(int).isin(ids)].copy()
        missing = [x for x in ids if x not in set(chosen[ID_COL].astype(int))]
        if missing:
            chosen = pd.concat(
                [
                    chosen,
                    pd.DataFrame(
                        {
                            ID_COL: missing,
                            "split": args.split,
                            "target": np.nan,
                            "score": 0.0,
                            "label_code": np.nan,
                            "label_text": "未知",
                        }
                    ),
                ],
                ignore_index=True,
            )
        return chosen

    candidates = predictions.copy()
    if args.confirmed_only:
        candidates = candidates[candidates["label_code"].eq(1)].copy()
    if not args.allow_inactive:
        _, split_end = SPLITS[args.split]
        history_tx = transactions[transactions[TIME_COL].le(pd.Timestamp(split_end))]
        active_ids = set(history_tx[SRC_COL].astype(int)).union(set(history_tx[DST_COL].astype(int)))
        active = candidates[candidates[ID_COL].astype(int).isin(active_ids)].copy()
        if not active.empty:
            candidates = active
    return candidates.sort_values("score", ascending=False).head(args.top_n).copy()


def top_counterparties(tx: pd.DataFrame, root_id: int, top_k: int) -> list[int]:
    out_tx = tx.loc[tx[SRC_COL].eq(root_id), [DST_COL, "amount_abs"]].rename(columns={DST_COL: "counterparty_id"})
    in_tx = tx.loc[tx[DST_COL].eq(root_id), [SRC_COL, "amount_abs"]].rename(columns={SRC_COL: "counterparty_id"})
    events = pd.concat([out_tx, in_tx], ignore_index=True)
    if events.empty:
        return []
    grouped = events.groupby("counterparty_id", sort=False).agg(
        txn_count=("amount_abs", "size"),
        amount_sum=("amount_abs", "sum"),
    )
    grouped["rank_score"] = grouped["txn_count"].rank(pct=True) + np.log1p(grouped["amount_sum"])
    return grouped.sort_values("rank_score", ascending=False).head(top_k).index.astype(int).tolist()


def motif_stats_for_root(tx: pd.DataFrame, root_id: int) -> dict:
    incoming = tx.loc[tx[DST_COL].eq(root_id), [SRC_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    outgoing = tx.loc[tx[SRC_COL].eq(root_id), [DST_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    if incoming.empty and outgoing.empty:
        return {
            "fast_in_out_24h_count": 0,
            "fast_in_out_amount_ratio": 0.0,
            "multi_in_one_out_24h_count": 0,
            "one_in_multi_out_24h_count": 0,
            "temporal_closed_loop_24h_count": 0,
            "temporal_closed_loop_min_delay_sec": 0.0,
            "reciprocal_counterparty_count": 0,
            "self_loop_count": 0,
        }

    out_times = outgoing[TIME_COL].to_numpy(dtype="datetime64[ns]")
    out_amounts = outgoing["amount_abs"].to_numpy(dtype=float)
    out_prefix = np.concatenate([[0.0], np.cumsum(out_amounts)])
    fast_count = 0
    fast_balance = 0.0
    total_in = float(incoming["amount_abs"].sum())
    for rec in incoming.itertuples(index=False):
        t1 = np.datetime64(rec[1])
        left = np.searchsorted(out_times, t1, side="right")
        right = np.searchsorted(out_times, t1 + np.timedelta64(24, "h"), side="right")
        if right > left:
            fast_count += 1
            out_amount = float(out_prefix[right] - out_prefix[left])
            fast_balance += min(float(rec[2]), out_amount)

    in_times = incoming[TIME_COL].to_numpy(dtype="datetime64[ns]")
    multi_in_one = 0
    if len(in_times) and len(out_times):
        left = np.searchsorted(in_times, out_times - np.timedelta64(24, "h"), side="left")
        right = np.searchsorted(in_times, out_times, side="left")
        multi_in_one = int(((right - left) >= 3).sum())
        next_left = np.searchsorted(out_times, in_times, side="right")
        next_right = np.searchsorted(out_times, in_times + np.timedelta64(24, "h"), side="right")
        one_in_multi = int(((next_right - next_left) >= 3).sum())
    else:
        one_in_multi = 0

    out_cp = set(outgoing[DST_COL].astype(int))
    in_cp = set(incoming[SRC_COL].astype(int))
    closed_loop_count = 0
    closed_loop_delays = []
    if not incoming.empty and not outgoing.empty:
        incoming_by_src = {
            int(k): g[[TIME_COL, "amount_abs"]].sort_values(TIME_COL)
            for k, g in incoming.groupby(SRC_COL, sort=False)
        }
        outgoing_by_dst = {
            int(k): g[[TIME_COL, "amount_abs"]].sort_values(TIME_COL)
            for k, g in outgoing.groupby(DST_COL, sort=False)
        }
        for cp_id, out_group in outgoing_by_dst.items():
            in_group = incoming_by_src.get(cp_id)
            if in_group is None or in_group.empty:
                continue
            in_times_same = in_group[TIME_COL].to_numpy(dtype="datetime64[ns]")
            for out_rec in out_group.itertuples(index=False):
                t1 = np.datetime64(out_rec[0])
                left = np.searchsorted(in_times_same, t1, side="right")
                right = np.searchsorted(in_times_same, t1 + np.timedelta64(24, "h"), side="right")
                if right > left:
                    closed_loop_count += 1
                    delay = (in_times_same[left] - t1).astype("timedelta64[s]").astype(float)
                    closed_loop_delays.append(float(delay))
    return {
        "fast_in_out_24h_count": int(fast_count),
        "fast_in_out_amount_ratio": float(fast_balance / total_in) if total_in else 0.0,
        "multi_in_one_out_24h_count": int(multi_in_one),
        "one_in_multi_out_24h_count": int(one_in_multi),
        "temporal_closed_loop_24h_count": int(closed_loop_count),
        "temporal_closed_loop_min_delay_sec": float(min(closed_loop_delays)) if closed_loop_delays else 0.0,
        "reciprocal_counterparty_count": int(len(out_cp & in_cp)),
        "self_loop_count": int(tx[SRC_COL].eq(tx[DST_COL]).sum()),
    }


def window_summary(tx: pd.DataFrame, root_id: int) -> dict:
    in_tx = tx[tx[DST_COL].eq(root_id)]
    out_tx = tx[tx[SRC_COL].eq(root_id)]
    counterparties = set(in_tx[SRC_COL].astype(int)).union(set(out_tx[DST_COL].astype(int)))
    motifs = motif_stats_for_root(tx, root_id)
    active_days = tx[TIME_COL].dt.date.nunique() if len(tx) else 0
    summary = {
        "txn_count": int(len(tx)),
        "in_txn_count": int(len(in_tx)),
        "out_txn_count": int(len(out_tx)),
        "amount_sum": float(tx["amount_abs"].sum()) if len(tx) else 0.0,
        "in_amount_sum": float(in_tx["amount_abs"].sum()) if len(in_tx) else 0.0,
        "out_amount_sum": float(out_tx["amount_abs"].sum()) if len(out_tx) else 0.0,
        "counterparty_count": int(len(counterparties)),
        "active_day_count": int(active_days),
    }
    summary.update(motifs)
    summary["risk_signal_score"] = float(
        math.log1p(summary["txn_count"])
        + 1.5 * summary["reciprocal_counterparty_count"]
        + 2.0 * summary["fast_in_out_24h_count"]
        + 2.0 * summary["multi_in_one_out_24h_count"]
        + 1.2 * summary["one_in_multi_out_24h_count"]
        + 2.5 * summary["temporal_closed_loop_24h_count"]
        + 3.0 * summary["self_loop_count"]
    )
    return summary


def aggregate_edges(tx: pd.DataFrame, amount_edges: list[float], max_edges: int) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame()
    grouped = tx.groupby([SRC_COL, DST_COL], sort=False).agg(
        txn_count=("amount_abs", "size"),
        amount_sum=("amount_abs", "sum"),
        first_time=(TIME_COL, "min"),
        last_time=(TIME_COL, "max"),
        self_loop_count=("self_loop", "sum") if "self_loop" in tx.columns else ("amount_abs", "size"),
        non_positive_count=("non_positive_amount", "sum") if "non_positive_amount" in tx.columns else ("amount_abs", "size"),
    ).reset_index()
    grouped["edge_score"] = grouped["txn_count"].rank(pct=True) + np.log1p(grouped["amount_sum"])
    grouped = grouped.sort_values("edge_score", ascending=False).head(max_edges).copy()
    grouped["amount_bin"] = grouped["amount_sum"].apply(lambda x: amount_bin_name(float(x), amount_edges))
    return grouped


def node_records(
    node_ids: list[int],
    root_id: int,
    labels: pd.DataFrame,
    score_map: dict[int, float],
    window_tx: pd.DataFrame,
) -> list[dict]:
    label_map = dict(zip(labels[ID_COL].astype(int), labels["label_text"].astype(str)))
    code_map = dict(zip(labels[ID_COL].astype(int), labels["label_code"].astype(int)))
    rows = []
    for node_id in node_ids:
        direct = window_tx[window_tx[SRC_COL].eq(node_id) | window_tx[DST_COL].eq(node_id)]
        rows.append(
            {
                "id": int(node_id),
                "label": str(node_id),
                "is_root": int(node_id == root_id),
                "label_text": label_map.get(int(node_id), "未知"),
                "label_code": int(code_map.get(int(node_id), -1)),
                "score": float(score_map.get(int(node_id), 0.0)),
                "txn_count": int(len(direct)),
                "amount_sum": float(direct["amount_abs"].sum()) if len(direct) else 0.0,
            }
        )
    return rows


def build_visual_data(
    top_accounts: pd.DataFrame,
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    transactions: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start_ts, end_ts = observation_window(transactions, args.split, args.history_days)
    history_tx = transactions[(transactions[TIME_COL] >= start_ts) & (transactions[TIME_COL] <= end_ts)].copy()
    windows = build_windows(start_ts, end_ts, args.window)
    amount_edges = train_amount_edges(transactions)
    score_map = dict(zip(predictions[ID_COL].astype(int), predictions["score"].astype(float)))

    label_map = dict(zip(labels[ID_COL].astype(int), labels["label_text"].astype(str)))
    code_map = dict(zip(labels[ID_COL].astype(int), labels["label_code"].astype(int)))

    snapshots = {}
    summary_rows = []
    edge_rows = []
    node_rows = []
    accounts_json = []

    for rec in top_accounts.sort_values("score", ascending=False).itertuples(index=False):
        root_id = int(getattr(rec, ID_COL))
        root_score = float(getattr(rec, "score", 0.0))
        root_label = label_map.get(root_id, "未知")
        counterparties = top_counterparties(history_tx, root_id, args.top_k_counterparties)
        selected_nodes = [root_id] + [x for x in counterparties if x != root_id]
        accounts_json.append(
            {
                "id": root_id,
                "score": root_score,
                "label_text": root_label,
                "counterparty_count": len(counterparties),
            }
        )

        for window_name, win_start, win_end in windows:
            window_tx_all = history_tx[(history_tx[TIME_COL] >= win_start) & (history_tx[TIME_COL] <= win_end)].copy()
            ego_tx = window_tx_all[
                (
                    window_tx_all[SRC_COL].eq(root_id)
                    | window_tx_all[DST_COL].eq(root_id)
                    | (window_tx_all[SRC_COL].isin(selected_nodes) & window_tx_all[DST_COL].isin(selected_nodes))
                )
            ].copy()
            ego_tx = ego_tx[ego_tx[SRC_COL].isin(selected_nodes) | ego_tx[DST_COL].isin(selected_nodes)].copy()
            edges = aggregate_edges(ego_tx, amount_edges, args.max_edges_per_window)
            if edges.empty:
                present_nodes = [root_id]
            else:
                present_nodes = sorted(set(edges[SRC_COL].astype(int)).union(set(edges[DST_COL].astype(int))).union({root_id}))
            nodes = node_records(present_nodes, root_id, labels, score_map, ego_tx)
            summary = window_summary(ego_tx, root_id)
            summary.update(
                {
                    "root_account_id": root_id,
                    "window": window_name,
                    "window_start": str(win_start),
                    "window_end": str(win_end),
                    "model_score": root_score,
                    "label_text": root_label,
                }
            )
            summary_rows.append(summary)
            for node in nodes:
                row = dict(node)
                row["root_account_id"] = root_id
                row["window"] = window_name
                node_rows.append(row)
            for edge in edges.to_dict("records"):
                row = {
                    "root_account_id": root_id,
                    "window": window_name,
                    "src": int(edge[SRC_COL]),
                    "dst": int(edge[DST_COL]),
                    "txn_count": int(edge["txn_count"]),
                    "amount_sum": float(edge["amount_sum"]),
                    "amount_bin": edge["amount_bin"],
                    "first_time": str(edge["first_time"]),
                    "last_time": str(edge["last_time"]),
                    "self_loop_count": int(edge["self_loop_count"]),
                    "non_positive_count": int(edge["non_positive_count"]),
                }
                edge_rows.append(row)
            snapshots[f"{root_id}|{window_name}"] = {
                "nodes": nodes,
                "edges": [
                    {
                        "src": int(edge[SRC_COL]),
                        "dst": int(edge[DST_COL]),
                        "txn_count": int(edge["txn_count"]),
                        "amount_sum": float(edge["amount_sum"]),
                        "amount_bin": str(edge["amount_bin"]),
                        "first_time": str(edge["first_time"]),
                        "last_time": str(edge["last_time"]),
                    }
                    for edge in edges.to_dict("records")
                ],
                "summary": summary,
            }

    windows_json = [
        {"name": name, "start": str(start), "end": str(end)}
        for name, start, end in windows
    ]
    prediction_trajectory = predictions[predictions[ID_COL].isin(top_accounts[ID_COL])][
        [ID_COL, "split", "score", "label_text"]
    ].to_dict("records")
    used_label_ids = set()
    for row in node_rows:
        used_label_ids.add(int(row["id"]))
    visual_data = {
        "meta": {
            "prediction_file": str(args.prediction_file_resolved),
            "split": args.split,
            "history_start": str(start_ts),
            "history_end": str(end_ts),
            "history_days": args.history_days,
            "window": args.window,
            "top_n": len(top_accounts),
            "top_k_counterparties": args.top_k_counterparties,
            "amount_bin_edges_from_train": amount_edges,
            "leakage_policy": "可视化只读取截至 split end 的滚动历史交易；金额分箱边界来自 train；不读取 split end 之后的交易。",
        },
        "accounts": accounts_json,
        "windows": windows_json,
        "snapshots": snapshots,
        "prediction_trajectory": prediction_trajectory,
        "labels": {
            str(k): {"label_text": label_map.get(k, "未知"), "label_code": int(code_map.get(k, -1))}
            for k in sorted(used_label_ids)
        },
    }
    return visual_data, pd.DataFrame(summary_rows), pd.DataFrame(edge_rows), pd.DataFrame(node_rows)


def css() -> str:
    return """
    :root {
      color-scheme: light;
      --ink:#202733; --muted:#667085; --line:#d7dde6; --risk:#c9352b; --victim:#c97818;
      --normal:#2563a6; --bg:#f4f6f8; --panel:#ffffff; --soft:#eef2f6; --green:#17724c;
      --shadow:0 1px 2px rgba(16, 24, 40, .06), 0 8px 24px rgba(16, 24, 40, .05);
    }
    * { box-sizing: border-box; }
    body {
      margin:0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
      color:var(--ink); background:var(--bg); letter-spacing:0;
    }
    header {
      position:sticky; top:0; z-index:5; padding:18px 28px 14px;
      background:rgba(255,255,255,.96); border-bottom:1px solid var(--line);
      backdrop-filter:saturate(120%) blur(8px);
    }
    h1 { margin:0; font-size:24px; line-height:1.2; letter-spacing:0; }
    h2 { margin:0 0 6px; font-size:20px; line-height:1.25; letter-spacing:0; }
    h3 { letter-spacing:0; }
    p { margin:6px 0; color:var(--muted); line-height:1.55; }
    .topline { display:flex; justify-content:space-between; gap:18px; align-items:flex-start; }
    .eyebrow { margin-bottom:6px; color:var(--green); font-size:12px; font-weight:700; letter-spacing:0; }
    .header-meta { display:flex; gap:8px; flex-wrap:wrap; justify-content:flex-end; min-width:250px; }
    .meta-pill {
      border:1px solid var(--line); background:#f8fafc; color:#344054;
      border-radius:6px; padding:6px 9px; font-size:12px; font-weight:650;
    }
    #subtitle { margin-top:8px; font-weight:650; color:#344054; }
    .nav { display:flex; gap:8px; flex-wrap:wrap; margin-top:14px; }
    .nav-btn {
      border:1px solid var(--line); background:#fff; color:#344054; padding:8px 12px;
      border-radius:6px; cursor:pointer; font-size:13px; font-weight:650;
    }
    .nav-btn:hover { border-color:#8da2bd; background:#f8fafc; }
    .nav-btn.active { color:#fff; border-color:#2f5f8f; background:#2f5f8f; }
    .overview {
      display:grid; grid-template-columns:repeat(7, minmax(128px, 1fr)); gap:10px;
      padding:14px 28px; background:#fff; border-bottom:1px solid var(--line);
    }
    .overview-metric {
      padding:10px 12px; background:#fbfcfd; border:1px solid #e5eaf0; border-left:3px solid #8293a8;
      border-radius:8px; min-height:66px;
    }
    .overview-metric strong { display:block; font-size:21px; line-height:1.1; margin-top:7px; font-variant-numeric:tabular-nums; }
    .overview-metric span { color:var(--muted); font-size:12px; font-weight:650; }
    .view { display:none; }
    .view.active { display:block; }
    .graph-layout { display:grid; grid-template-columns: 282px minmax(520px, 1fr) 360px; gap:14px; padding:14px 28px 26px; min-height: calc(100vh - 210px); }
    aside, section { min-width:0; }
    .side-panel, .graph-panel, .right {
      background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:var(--shadow);
    }
    .side-panel { padding:14px; max-height:calc(100vh - 228px); overflow:auto; }
    .graph-panel { padding:16px; }
    .right { padding:16px; overflow:auto; max-height:calc(100vh - 228px); }
    .btn {
      width:100%; border:1px solid var(--line); background:#fff; color:var(--ink); padding:10px 12px;
      margin:6px 0; text-align:left; border-radius:7px; cursor:pointer; font-size:13px; line-height:1.35;
    }
    .btn:hover { border-color:#9aacbf; background:#f8fafc; }
    .btn.active { border-color:#2f5f8f; background:#edf5ff; box-shadow:inset 3px 0 0 #2f5f8f; }
    .btn b { display:block; margin-bottom:2px; }
    .btn small { color:var(--muted); }
    .window-row { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
    .window-row .btn { width:auto; min-width:84px; text-align:center; margin:0; }
    .panel-title { font-weight:750; margin:0 0 10px; color:#2a3441; }
    .section-head { display:flex; justify-content:space-between; gap:12px; align-items:flex-end; margin-bottom:12px; }
    .section-head p { margin:4px 0 0; font-size:13px; }
    .wide { padding:22px 28px 38px; }
    .toolbar { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:12px 0; }
    .toolbar input { border:1px solid var(--line); background:#fff; padding:8px 10px; border-radius:6px; color:var(--ink); min-width:240px; }
    .filter-btn { border:1px solid var(--line); background:#fff; padding:7px 10px; border-radius:6px; cursor:pointer; font-weight:650; }
    .filter-btn:hover { border-color:#9aacbf; background:#f8fafc; }
    .filter-btn.active { color:#fff; background:#2f5f8f; border-color:#2f5f8f; }
    .metric { display:grid; grid-template-columns: 1fr auto; gap:10px; padding:9px 0; border-bottom:1px solid #edf1f6; font-size:14px; }
    .metric span:first-child { color:var(--muted); }
    .metric b { font-variant-numeric:tabular-nums; }
    #graph { width:100%; height:590px; background:#fbfcfd; border:1px solid var(--line); border-radius:8px; }
    .legend { display:flex; gap:14px; flex-wrap:wrap; font-size:13px; color:var(--muted); margin:8px 0 14px; }
    .dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
    .table-wrap { overflow:auto; border:1px solid var(--line); background:#fff; border-radius:8px; }
    table { border-collapse:collapse; width:100%; background:#fff; font-size:13px; }
    th, td { padding:9px 10px; border-bottom:1px solid #edf1f6; text-align:left; vertical-align:top; }
    th { position:sticky; top:0; z-index:1; color:#5d6b7c; background:#f8fafc; font-weight:700; white-space:nowrap; }
    tr:last-child td { border-bottom:0; }
    tbody tr[data-account], tbody tr[data-queue-account], tbody tr[data-recovery-account] { cursor:pointer; }
    tbody tr[data-account]:hover, tbody tr[data-queue-account]:hover, tbody tr[data-recovery-account]:hover { background:#f3f7fb; }
    .evidence-cell { min-width:300px; max-width:520px; white-space:normal; line-height:1.5; }
    .reason-cell { min-width:230px; white-space:normal; line-height:1.45; }
    .badge { display:inline-block; padding:3px 7px; border-radius:4px; font-size:12px; font-weight:700; }
    .badge-a { background:#e5f4ec; color:#176442; }
    .badge-b { background:#eaf1ff; color:#345d9d; }
    .badge-c { background:#fff3d9; color:#8b5a00; }
    .badge-d { background:#fbe9e8; color:#a32924; }
    .detail-panel { margin-top:16px; padding:16px; border:1px solid var(--line); background:#fff; border-radius:8px; box-shadow:var(--shadow); }
    .detail-grid { display:grid; grid-template-columns:repeat(4, minmax(130px, 1fr)); gap:10px; margin:10px 0 16px; }
    .detail-item { background:#f8fafc; border:1px solid #e5eaf0; border-radius:8px; padding:10px; }
    .detail-item span { display:block; color:var(--muted); font-size:12px; }
    .detail-item b { display:block; margin-top:4px; font-variant-numeric:tabular-nums; }
    .subsection { margin-top:18px; }
    .subsection h3 { margin:0 0 8px; font-size:16px; }
    .report-grid { display:grid; grid-template-columns:repeat(2, minmax(320px, 1fr)); gap:14px; }
    .report-card { border:1px solid var(--line); background:#fff; border-radius:8px; padding:16px; box-shadow:var(--shadow); }
    .report-card h3 { margin:0 0 10px; font-size:16px; }
    .report-card p { line-height:1.5; }
    .raw-report { max-height:520px; white-space:pre-wrap; word-break:break-word; font:12px/1.6 ui-monospace, SFMono-Regular, Menlo, monospace; background:#f6f8fb; border:1px solid var(--line); border-radius:8px; padding:12px; overflow:auto; }
    .svg-label { font-size:12px; fill:#233142; pointer-events:none; }
    .edge-label { font-size:10px; fill:#52616f; pointer-events:none; }
    .empty { color:var(--muted); padding:18px; text-align:center; }
    .muted { color:var(--muted); }
    @media (max-width: 1180px) { .overview { grid-template-columns:repeat(4, 1fr); } .graph-layout { grid-template-columns:250px 1fr; } .graph-layout .right { grid-column:1 / -1; max-height:none; } }
    @media (max-width: 760px) { header, .wide { padding-left:14px; padding-right:14px; } .topline { display:block; } .header-meta { justify-content:flex-start; margin-top:12px; min-width:0; } .overview { padding:12px 14px; grid-template-columns:repeat(2, 1fr); } .graph-layout { display:block; padding:12px 14px 24px; } .side-panel, .graph-panel, .right { margin-bottom:12px; max-height:none; } #graph { height:430px; } .detail-grid { grid-template-columns:repeat(2, 1fr); } .report-grid { grid-template-columns:1fr; } }
    """


def js() -> str:
    return """
    const DATA = __DATA__;
    let currentAccount = DATA.accounts.length ? DATA.accounts[0].id : null;
    let currentWindow = DATA.windows.length ? DATA.windows[DATA.windows.length - 1].name : null;

    function colorNode(n) {
      if (n.is_root) return "#c9342d";
      if (n.label_code === 1) return "#c9342d";
      if (n.label_code === 2) return "#e8912d";
      if (n.score >= 0.5) return "#8e3b46";
      return "#2b6cb0";
    }
    function fmt(x) {
      const n = Number(x || 0);
      if (Math.abs(n) >= 100000000) return (n / 100000000).toFixed(2) + "亿";
      if (Math.abs(n) >= 10000) return (n / 10000).toFixed(2) + "万";
      return n.toFixed(2);
    }
    function setActiveButtons() {
      document.querySelectorAll("[data-account]").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.account) === Number(currentAccount)));
      document.querySelectorAll("[data-window]").forEach(btn => btn.classList.toggle("active", btn.dataset.window === currentWindow));
    }
    function renderSelectors() {
      const accounts = document.getElementById("accounts");
      accounts.innerHTML = DATA.accounts.map(a => `<button class="btn" data-account="${a.id}">账户 ${a.id}<br><small>score ${Number(a.score).toFixed(4)} · ${a.label_text}</small></button>`).join("");
      accounts.querySelectorAll("button").forEach(btn => btn.onclick = () => { currentAccount = Number(btn.dataset.account); render(); });
      const windows = document.getElementById("windows");
      windows.innerHTML = DATA.windows.map(w => `<button class="btn" data-window="${w.name}">${w.name}</button>`).join("");
      windows.querySelectorAll("button").forEach(btn => btn.onclick = () => { currentWindow = btn.dataset.window; render(); });
    }
    function layout(nodes) {
      const w = 980, h = 590, cx = w / 2, cy = h / 2;
      const out = {};
      const others = nodes.filter(n => !n.is_root);
      nodes.forEach(n => {
        if (n.is_root) out[n.id] = {x: cx, y: cy};
      });
      const radius = Math.min(245, 120 + others.length * 8);
      others.forEach((n, i) => {
        const angle = (2 * Math.PI * i / Math.max(1, others.length)) - Math.PI / 2;
        out[n.id] = {x: cx + radius * Math.cos(angle), y: cy + radius * Math.sin(angle)};
      });
      return out;
    }
    function renderGraph(snapshot) {
      const svg = document.getElementById("graph");
      const nodes = snapshot.nodes || [];
      const edges = snapshot.edges || [];
      if (!nodes.length) {
        svg.innerHTML = `<text x="490" y="295" class="svg-label" text-anchor="middle">当前窗口没有可展示交易边</text>`;
        return;
      }
      const pos = layout(nodes);
      const maxAmount = Math.max(1, ...edges.map(e => Number(e.amount_sum || 0)));
      const edgeSvg = edges.map((e, idx) => {
        const s = pos[e.src], t = pos[e.dst];
        if (!s || !t) return "";
        const dx = t.x - s.x, dy = t.y - s.y;
        const len = Math.sqrt(dx * dx + dy * dy) || 1;
        const sx = s.x + dx / len * 20, sy = s.y + dy / len * 20;
        const tx = t.x - dx / len * 24, ty = t.y - dy / len * 24;
        const width = 1.2 + 4.5 * Math.log1p(Number(e.amount_sum || 0)) / Math.log1p(maxAmount);
        const midx = (sx + tx) / 2, midy = (sy + ty) / 2;
        return `<g>
          <line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="#8293a8" stroke-width="${width.toFixed(2)}" marker-end="url(#arrow)" opacity="0.72"/>
          <text x="${midx}" y="${midy - 5}" class="edge-label" text-anchor="middle">${e.amount_bin} · ${e.txn_count}笔</text>
        </g>`;
      }).join("");
      const nodeSvg = nodes.map(n => {
        const p = pos[n.id];
        const r = n.is_root ? 20 : 12 + Math.min(10, Math.log1p(Number(n.txn_count || 0)) * 2);
        return `<g>
          <circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${colorNode(n)}" stroke="#fff" stroke-width="2"/>
          <text x="${p.x}" y="${p.y + r + 15}" class="svg-label" text-anchor="middle">${n.id}</text>
        </g>`;
      }).join("");
      svg.innerHTML = `<defs>
        <marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
          <path d="M0,0 L0,6 L8,3 z" fill="#8293a8"></path>
        </marker>
      </defs>${edgeSvg}${nodeSvg}`;
    }
    function renderEvidence(snapshot) {
      const s = snapshot.summary || {};
      const metrics = [
        ["模型风险分", Number(s.model_score || 0).toFixed(4)],
        ["窗口交易数", s.txn_count || 0],
        ["入账/出账", `${s.in_txn_count || 0} / ${s.out_txn_count || 0}`],
        ["窗口总金额", fmt(s.amount_sum)],
        ["交易对手数", s.counterparty_count || 0],
        ["快进快出 24h", s.fast_in_out_24h_count || 0],
        ["短时闭环 24h", s.temporal_closed_loop_24h_count || 0],
        ["闭环最短延迟", `${Number(s.temporal_closed_loop_min_delay_sec || 0).toFixed(0)} 秒`],
        ["多入一出 24h", s.multi_in_one_out_24h_count || 0],
        ["一入多出 24h", s.one_in_multi_out_24h_count || 0],
        ["互惠对手数", s.reciprocal_counterparty_count || 0],
        ["窗口风险信号", Number(s.risk_signal_score || 0).toFixed(3)]
      ];
      document.getElementById("evidence").innerHTML = metrics.map(m => `<div class="metric"><span>${m[0]}</span><b>${m[1]}</b></div>`).join("");
      const rows = (snapshot.edges || []).slice(0, 10).map(e => `<tr><td>${e.src} → ${e.dst}</td><td>${e.txn_count}</td><td>${fmt(e.amount_sum)}</td><td>${e.amount_bin}</td></tr>`).join("");
      document.getElementById("edgeTable").innerHTML = rows || `<tr><td colspan="4" class="empty">当前窗口无 Top 边</td></tr>`;
    }
    function renderTimeline() {
      const rows = DATA.windows.map(w => {
        const snap = DATA.snapshots[`${currentAccount}|${w.name}`] || {summary:{}};
        const s = snap.summary || {};
        return `<tr><td>${w.name}</td><td>${s.txn_count || 0}</td><td>${s.counterparty_count || 0}</td><td>${s.fast_in_out_24h_count || 0}</td><td>${s.temporal_closed_loop_24h_count || 0}</td><td>${Number(s.risk_signal_score || 0).toFixed(2)}</td></tr>`;
      }).join("");
      document.getElementById("timeline").innerHTML = rows;
    }
    function render() {
      setActiveButtons();
      const key = `${currentAccount}|${currentWindow}`;
      const snapshot = DATA.snapshots[key] || {nodes:[], edges:[], summary:{}};
      const account = DATA.accounts.find(a => Number(a.id) === Number(currentAccount)) || {};
      document.getElementById("subtitle").textContent = `账户 ${currentAccount} · ${account.label_text || "未知"} · ${currentWindow}`;
      renderGraph(snapshot);
      renderEvidence(snapshot);
      renderTimeline();
    }
    document.addEventListener("DOMContentLoaded", () => { renderSelectors(); render(); });
    """


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    script = js().replace("__DATA__", payload)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>滚动动态资金图谱展示</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>滚动动态资金图谱展示</h1>
    <p id="subtitle">账户风险子图</p>
    <p>观察窗口：{escape(data["meta"]["history_start"])} 至 {escape(data["meta"]["history_end"])}；边界只使用历史交易，金额分箱来自 train。</p>
  </header>
  <main>
    <aside>
      <div class="panel-title">高风险账户</div>
      <div id="accounts"></div>
    </aside>
    <section>
      <div class="panel-title">时间窗口</div>
      <div id="windows" class="window-row"></div>
      <div class="legend">
        <span><i class="dot" style="background:#c9342d"></i>嫌疑/根账户</span>
        <span><i class="dot" style="background:#e8912d"></i>受害人</span>
        <span><i class="dot" style="background:#2b6cb0"></i>普通账户</span>
        <span>边越粗表示窗口内金额越高</span>
      </div>
      <svg id="graph" viewBox="0 0 980 590" role="img" aria-label="滚动动态资金图谱"></svg>
      <h3>窗口轨迹</h3>
      <table>
        <thead><tr><th>窗口</th><th>交易数</th><th>对手数</th><th>快进快出</th><th>短时闭环</th><th>风险信号</th></tr></thead>
        <tbody id="timeline"></tbody>
      </table>
    </section>
    <section class="right">
      <div class="panel-title">研判证据</div>
      <div id="evidence"></div>
      <h3>Top 交易边</h3>
      <table>
        <thead><tr><th>边</th><th>笔数</th><th>金额</th><th>分箱</th></tr></thead>
        <tbody id="edgeTable"></tbody>
      </table>
    </section>
  </main>
  <script>{script}</script>
</body>
</html>
"""


def js_v2() -> str:
    return """
    const DATA = __DATA__;
    let currentAccount = DATA.accounts.length ? DATA.accounts[0].id : null;
    let currentWindow = DATA.windows.length ? DATA.windows[DATA.windows.length - 1].name : null;
    let auditGrade = "all";
    let auditQuery = "";
    let queueAccount = DATA.layered.queue.length ? DATA.layered.queue[0].account_id : null;
    let recoveryAccount = DATA.layered.recovery.length ? DATA.layered.recovery[0].account_id : null;

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => { if (ch === '"') return "&quot;"; return ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;"}[ch]); });
    }
    function num(value, digits=4) { const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits) : "-"; }
    function fmt(value) {
      const n = Number(value || 0);
      if (Math.abs(n) >= 100000000) return (n / 100000000).toFixed(2) + "亿";
      if (Math.abs(n) >= 10000) return (n / 10000).toFixed(2) + "万";
      return n.toFixed(2);
    }
    function badge(grade) { const g = String(grade || "").toLowerCase(); return `<span class="badge badge-${g}">${esc(grade || "-")}</span>`; }
    function labelColor(node) {
      if (node.is_root || node.label_code === 1) return "#c9342d";
      if (node.label_code === 2) return "#e8912d";
      if (node.score >= 0.5) return "#8e3b46";
      return "#2b6cb0";
    }
    function setView(view) {
      document.querySelectorAll(".view").forEach(el => el.classList.toggle("active", el.id === `${view}View`));
      document.querySelectorAll("[data-view]").forEach(btn => btn.classList.toggle("active", btn.dataset.view === view));
      if (view === "graph") renderGraphView();
      if (view === "audit") renderAudit();
      if (view === "recovery") renderRecovery();
      if (view === "queue") renderQueue();
      if (view === "report") renderReports();
    }
    function renderOverview() {
      const c = DATA.layered.coverage || {};
      const grades = c.confirmed_suspect_grade_counts || {};
      const qGrades = c.active_risk_review_grade_counts || {};
      const items = [
        ["确认嫌疑账户", c.confirmed_suspect_total || 0, "#c9342d"],
        ["可追溯历史交易", c.confirmed_suspect_with_history_transaction || 0, "#345d9d"],
        ["缺边审计账户", grades.D || 0, "#a32924"],
        ["缺边恢复队列", DATA.layered.recovery.length, "#8b5a00"],
        [`Top${DATA.meta.top_n}巡检账户`, c.active_risk_review_account_count || 0, "#1f7a55"],
        ["巡检A/B级", `${qGrades.A || 0} / ${qGrades.B || 0}`, "#345d9d"],
        ["路径/结构证据", `${c.suspicious_path_rows || 0} / ${c.fund_flow_structure_rows || 0}`, "#8b5a00"]
      ];
      document.getElementById("overviewMetrics").innerHTML = items.map(x => `<div class="overview-metric" style="border-left-color:${x[2]}"><span>${esc(x[0])}</span><strong>${esc(x[1])}</strong></div>`).join("");
    }
    function renderSelectors() {
      const accounts = document.getElementById("accounts");
      accounts.innerHTML = DATA.accounts.map(a => `<button class="btn" data-account="${a.id}"><b>账户 ${a.id}</b><small>风险分 ${num(a.score)} · ${esc(a.label_text)}</small></button>`).join("");
      accounts.querySelectorAll("button").forEach(btn => btn.onclick = () => { currentAccount = Number(btn.dataset.account); renderGraphView(); });
      const windows = document.getElementById("windows");
      windows.innerHTML = DATA.windows.map(w => `<button class="btn" data-window="${w.name}">${esc(w.name)}</button>`).join("");
      windows.querySelectorAll("button").forEach(btn => btn.onclick = () => { currentWindow = btn.dataset.window; renderGraphView(); });
    }
    function setGraphActive() {
      document.querySelectorAll("#accounts [data-account]").forEach(btn => btn.classList.toggle("active", Number(btn.dataset.account) === Number(currentAccount)));
      document.querySelectorAll("#windows [data-window]").forEach(btn => btn.classList.toggle("active", btn.dataset.window === currentWindow));
    }
    function layout(nodes) {
      const w = 980, h = 590, cx = w / 2, cy = h / 2, out = {};
      const others = nodes.filter(n => !n.is_root);
      nodes.forEach(n => { if (n.is_root) out[n.id] = {x:cx, y:cy}; });
      const radius = Math.min(245, 120 + others.length * 8);
      others.forEach((n, i) => { const angle = (2 * Math.PI * i / Math.max(1, others.length)) - Math.PI / 2; out[n.id] = {x:cx + radius * Math.cos(angle), y:cy + radius * Math.sin(angle)}; });
      return out;
    }
    function renderGraph(snapshot) {
      const svg = document.getElementById("graph"), nodes = snapshot.nodes || [], edges = snapshot.edges || [];
      if (!nodes.length) { svg.innerHTML = `<text x="490" y="295" class="svg-label" text-anchor="middle">当前窗口没有可展示交易边</text>`; return; }
      const pos = layout(nodes), maxAmount = Math.max(1, ...edges.map(e => Number(e.amount_sum || 0)));
      const edgeSvg = edges.map(e => {
        const s = pos[e.src], t = pos[e.dst]; if (!s || !t) return "";
        const dx = t.x-s.x, dy = t.y-s.y, len = Math.sqrt(dx*dx + dy*dy) || 1;
        const sx = s.x + dx / len * 20, sy = s.y + dy / len * 20, tx = t.x - dx / len * 24, ty = t.y - dy / len * 24;
        const width = 1.2 + 4.5 * Math.log1p(Number(e.amount_sum || 0)) / Math.log1p(maxAmount), midx=(sx+tx)/2, midy=(sy+ty)/2;
        return `<g><line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="#8293a8" stroke-width="${width.toFixed(2)}" marker-end="url(#arrow)" opacity="0.72"/><text x="${midx}" y="${midy-5}" class="edge-label" text-anchor="middle">${esc(e.amount_bin)} · ${e.txn_count}笔</text></g>`;
      }).join("");
      const nodeSvg = nodes.map(n => { const p=pos[n.id], r=n.is_root ? 20 : 12 + Math.min(10, Math.log1p(Number(n.txn_count || 0))*2); return `<g><circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${labelColor(n)}" stroke="#fff" stroke-width="2"/><text x="${p.x}" y="${p.y+r+15}" class="svg-label" text-anchor="middle">${n.id}</text></g>`; }).join("");
      svg.innerHTML = `<defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0,0 L0,6 L8,3 z" fill="#8293a8"></path></marker></defs>${edgeSvg}${nodeSvg}`;
    }
    function renderEvidence(snapshot) {
      const s = snapshot.summary || {}, metrics = [["模型风险分", num(s.model_score)], ["窗口交易数", s.txn_count || 0], ["入账/出账", `${s.in_txn_count || 0} / ${s.out_txn_count || 0}`], ["窗口总金额", fmt(s.amount_sum)], ["交易对手数", s.counterparty_count || 0], ["快进快出 24h", s.fast_in_out_24h_count || 0], ["短时闭环 24h", s.temporal_closed_loop_24h_count || 0], ["闭环最短延迟", `${num(s.temporal_closed_loop_min_delay_sec, 0)} 秒`], ["多入一出 24h", s.multi_in_one_out_24h_count || 0], ["一入多出 24h", s.one_in_multi_out_24h_count || 0], ["互惠对手数", s.reciprocal_counterparty_count || 0], ["窗口风险信号", num(s.risk_signal_score, 3)]];
      document.getElementById("evidence").innerHTML = metrics.map(m => `<div class="metric"><span>${esc(m[0])}</span><b>${esc(m[1])}</b></div>`).join("");
      document.getElementById("edgeTable").innerHTML = (snapshot.edges || []).slice(0, 10).map(e => `<tr><td>${e.src} → ${e.dst}</td><td>${e.txn_count}</td><td>${fmt(e.amount_sum)}</td><td>${esc(e.amount_bin)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">当前窗口无 Top 边</td></tr>`;
    }
    function renderTimeline() {
      document.getElementById("timeline").innerHTML = DATA.windows.map(w => { const s=(DATA.snapshots[`${currentAccount}|${w.name}`] || {summary:{}}).summary || {}; return `<tr><td>${esc(w.name)}</td><td>${s.txn_count || 0}</td><td>${s.counterparty_count || 0}</td><td>${s.fast_in_out_24h_count || 0}</td><td>${s.temporal_closed_loop_24h_count || 0}</td><td>${num(s.risk_signal_score,2)}</td></tr>`; }).join("");
    }
    function renderGraphView() {
      setGraphActive();
      const key = `${currentAccount}|${currentWindow}`, snapshot = DATA.snapshots[key] || {nodes:[],edges:[],summary:{}};
      const account = DATA.accounts.find(a => Number(a.id) === Number(currentAccount)) || {};
      document.getElementById("subtitle").textContent = `动态图谱 · 账户 ${currentAccount || "-"} · ${account.label_text || "未知"} · ${currentWindow || "-"}`;
      renderGraph(snapshot); renderEvidence(snapshot); renderTimeline();
    }
    function renderAudit() {
      const rows = DATA.layered.audit.filter(row => { const q=auditQuery.trim().toLowerCase(), text=`${row.account_id} ${row.label_text} ${row.explanation_reason} ${row.feature_evidence}`.toLowerCase(); return (auditGrade === "all" || row.explanation_grade === auditGrade) && (!q || text.includes(q)); });
      document.getElementById("auditCount").textContent = `显示 ${rows.length} / ${DATA.layered.audit.length} 条审计记录`;
      document.getElementById("auditTableBody").innerHTML = rows.map(row => `<tr data-account="${row.account_id}"><td>${row.risk_rank}</td><td><b>${row.account_id}</b><br><span class="muted">${esc(row.label_text)}</span></td><td>${num(row.score)}</td><td>${badge(row.explanation_grade)}</td><td class="reason-cell">${esc(row.explanation_reason)}</td><td>${row.history_txn_count || 0}</td><td>${fmt(row.history_amount_sum)}</td><td>${row.direct_counterparty_count || 0}</td><td class="evidence-cell">${esc(row.short_evidence || row.feature_evidence || "-")}</td><td class="reason-cell">${esc(row.recommended_next_step || "-")}</td></tr>`).join("") || `<tr><td colspan="10" class="empty">没有匹配的审计记录</td></tr>`;
      document.querySelectorAll("#auditTableBody tr[data-account]").forEach(row => row.onclick = () => showAuditDetail(Number(row.dataset.account)));
    }
    function showAuditDetail(id) {
      const row = DATA.layered.audit.find(x => Number(x.account_id) === Number(id)); if (!row) return;
      document.getElementById("auditDetail").innerHTML = `<h3>账户 ${row.account_id} 审计详情</h3><div class="detail-grid"><div class="detail-item"><span>模型风险分</span><b>${num(row.score)}</b></div><div class="detail-item"><span>解释等级</span><b>${badge(row.explanation_grade)}</b></div><div class="detail-item"><span>历史交易</span><b>${row.history_txn_count || 0} 笔</b></div><div class="detail-item"><span>直接对手</span><b>${row.direct_counterparty_count || 0} 个</b></div><div class="detail-item"><span>边覆盖</span><b>${esc(row.edge_coverage_status || "-")}</b></div></div><p><b>解释结论：</b>${esc(row.explanation_reason)}</p><p><b>节点画像证据：</b>${esc(row.node_profile_evidence || "-")}</p><p><b>特征证据：</b>${esc(row.feature_evidence || "-")}</p><p><b>短证据：</b>${esc(row.short_evidence || "-")}</p><p><b>下一步：</b>${esc(row.recommended_next_step || "-")}</p>`;
      document.getElementById("auditDetail").scrollIntoView({behavior:"smooth", block:"start"});
    }
    function renderRecovery() {
      const rows = DATA.layered.recovery || [];
      document.getElementById("recoveryCount").textContent = `显示 ${rows.length} 个缺边账户；当前不生成虚构链路`;
      document.getElementById("recoveryTableBody").innerHTML = rows.map(row => `<tr data-recovery-account="${row.account_id}"><td>${row.recovery_priority}</td><td><b>${row.account_id}</b><br><span class="muted">${esc(row.label_text)}</span></td><td>${num(row.score)}</td><td>${row.risk_rank}</td><td>${row.history_txn_count || 0}</td><td class="evidence-cell">${esc(row.node_profile_evidence || "-")}</td><td class="reason-cell">${esc(row.required_query || "-")}</td></tr>`).join("") || `<tr><td colspan="7" class="empty">没有缺边账户</td></tr>`;
      document.querySelectorAll("#recoveryTableBody tr[data-recovery-account]").forEach(row => row.onclick = () => showRecoveryDetail(Number(row.dataset.recoveryAccount)));
      if (rows.length && !rows.some(x => Number(x.account_id) === Number(recoveryAccount))) recoveryAccount = rows[0].account_id;
      showRecoveryDetail(recoveryAccount);
    }
    function showRecoveryDetail(id) {
      const row = (DATA.layered.recovery || []).find(x => Number(x.account_id) === Number(id)); if (!row) return;
      document.getElementById("recoveryDetail").innerHTML = `<h3>账户 ${row.account_id} 链路恢复任务</h3><div class="detail-grid"><div class="detail-item"><span>模型风险分</span><b>${num(row.score)}</b></div><div class="detail-item"><span>风险排名</span><b>${row.risk_rank}</b></div><div class="detail-item"><span>恢复优先级</span><b>${esc(row.recovery_priority)}</b></div><div class="detail-item"><span>当前历史交易</span><b>${row.history_txn_count || 0} 笔</b></div></div><p><b>当前边状态：</b>${esc(row.edge_coverage_status || "-")}</p><p><b>节点画像证据：</b>${esc(row.node_profile_evidence || "-")}</p><p><b>补数查询：</b>${esc(row.required_query || "-")}</p><p><b>补数后重建：</b>${esc(row.expected_link_evidence || "-")}</p><p><b>当前边界：</b>${esc(row.current_evidence_boundary || "-")}</p>`;
      document.getElementById("recoveryDetail").scrollIntoView({behavior:"smooth", block:"start"});
    }
    function renderQueue() {
      document.getElementById("queueTableBody").innerHTML = DATA.layered.queue.map(row => `<tr data-queue-account="${row.account_id}"><td>${row.risk_rank}</td><td><b>${row.account_id}</b><br><span class="muted">${esc(row.label_text)}</span></td><td>${num(row.score)}</td><td>${badge(row.explanation_grade)}</td><td class="reason-cell">${esc(row.explanation_reason)}</td><td>${row.history_txn_count || 0}</td><td>${fmt(row.history_amount_sum)}</td><td>${row.direct_counterparty_count || 0}</td><td class="evidence-cell">${esc(row.feature_evidence || "-")}</td></tr>`).join("") || `<tr><td colspan="9" class="empty">没有巡检记录</td></tr>`;
      document.querySelectorAll("#queueTableBody tr[data-queue-account]").forEach(row => row.onclick = () => { queueAccount=Number(row.dataset.queueAccount); renderQueueDetail(); });
      renderQueueDetail();
    }
    function renderQueueDetail() {
      const id=Number(queueAccount), row=DATA.layered.queue.find(x => Number(x.account_id) === id); if (!row) return;
      const assoc=DATA.layered.associations.filter(x => Number(x.root_account_id) === id).slice(0,20), paths=DATA.layered.paths.filter(x => Number(x.root_account_id) === id).slice(0,20), structures=DATA.layered.structures.filter(x => Number(x.root_account_id) === id), graphReady=DATA.accounts.some(x => Number(x.id) === id);
      document.getElementById("queueDetail").innerHTML = `<h3>账户 ${id} 详细巡检证据</h3><div class="detail-grid"><div class="detail-item"><span>风险分</span><b>${num(row.score)}</b></div><div class="detail-item"><span>解释等级</span><b>${badge(row.explanation_grade)}</b></div><div class="detail-item"><span>历史交易</span><b>${row.history_txn_count || 0} 笔</b></div><div class="detail-item"><span>历史金额</span><b>${fmt(row.history_amount_sum)}</b></div></div><p><b>解释结论：</b>${esc(row.explanation_reason)}</p><p><b>动态/交易特征：</b>${esc(row.feature_evidence || "-")}</p>${graphReady ? `<button class="filter-btn" onclick="openGraph(${id})">查看该账户滚动动态图谱</button>` : `<p class="muted">该账户没有进入当前动态图谱 Top${DATA.meta.top_n}，仍保留结构化证据。</p>`}
      <div class="subsection"><h3>Top20 关联账户（${assoc.length}）</h3><div class="table-wrap"><table><thead><tr><th>对手</th><th>标签</th><th>交易数</th><th>金额</th><th>方向</th><th>时间</th></tr></thead><tbody>${assoc.map(x => `<tr><td>${x.counterparty_id}</td><td>${esc(x.label_text)}</td><td>${x.direct_txn_count}</td><td>${fmt(x.direct_amount_sum)}</td><td>${x.in_txn_count || 0} 入 / ${x.out_txn_count || 0} 出</td><td>${esc(x.first_time)} 至 ${esc(x.last_time)}</td></tr>`).join("") || `<tr><td colspan="6" class="empty">没有直接关联账户</td></tr>`}</tbody></table></div></div>
      <div class="subsection"><h3>可疑多跳路径（${paths.length}）</h3><div class="table-wrap"><table><thead><tr><th>路径</th><th>时间间隔</th><th>金额比</th><th>证据分</th></tr></thead><tbody>${paths.map(x => `<tr><td>${x.account_1} → ${x.account_2} → ${x.account_3}</td><td>${num(Number(x.delay_hours) * 60, 2)} 分钟</td><td>${num(x.amount_ratio, 4)}</td><td>${num(x.path_evidence_score, 3)}</td></tr>`).join("") || `<tr><td colspan="4" class="empty">没有满足当前规则的多跳路径</td></tr>`}</tbody></table></div></div>
      <div class="subsection"><h3>资金流结构（${structures.length}）</h3><div class="table-wrap"><table><thead><tr><th>结构</th><th>锚定时间</th><th>入/出笔数</th><th>金额</th><th>业务含义</th></tr></thead><tbody>${structures.map(x => `<tr><td>${esc(x.structure_type)}</td><td>${esc(x.anchor_time)}</td><td>${x.in_txn_count || 0} / ${x.out_txn_count || 0}</td><td>${fmt(x.in_amount_sum)} / ${fmt(x.out_amount_sum)}</td><td class="evidence-cell">${esc(x.business_meaning)}</td></tr>`).join("") || `<tr><td colspan="5" class="empty">没有资金流结构证据</td></tr>`}</tbody></table></div></div>`;
    }
    function renderReports() {
      const md=DATA.layered.judgement_report_markdown || "", ids=[...md.matchAll(/## 案例\\s*\\d+：账户\\s*(\\d+)/g)].map(m => Number(m[1])), uniqueIds=[...new Set(ids)];
      document.getElementById("reportGrid").innerHTML = uniqueIds.map(id => { const row=DATA.layered.audit.find(x => Number(x.account_id) === id) || DATA.layered.queue.find(x => Number(x.account_id) === id); if (!row) return ""; const paths=DATA.layered.paths.filter(x => Number(x.root_account_id) === id).slice(0,2), structures=DATA.layered.structures.filter(x => Number(x.root_account_id) === id).slice(0,2), assoc=DATA.layered.associations.filter(x => Number(x.root_account_id) === id).slice(0,5); return `<article class="report-card"><h3>账户 ${id} · ${badge(row.explanation_grade)} · 风险分 ${num(row.score)}</h3><p><b>标签：</b>${esc(row.label_text)}　<b>风险排名：</b>${esc(row.risk_rank || "-")}</p><p><b>解释结论：</b>${esc(row.explanation_reason)}</p><p><b>交易证据：</b>历史交易 ${row.history_txn_count || 0} 笔，直接对手 ${row.direct_counterparty_count || 0} 个，历史金额 ${fmt(row.history_amount_sum)}</p><p><b>特征证据：</b>${esc(row.feature_evidence || "-")}</p>${paths.length ? `<p><b>可疑路径：</b>${paths.map(x => `${x.account_1} → ${x.account_2} → ${x.account_3}，间隔 ${num(Number(x.delay_hours) * 60, 2)} 分钟，金额比 ${num(x.amount_ratio, 4)}`).join("；")}</p>` : ""}${structures.length ? `<p><b>资金流结构：</b>${structures.map(x => `${esc(x.structure_type)}，${esc(x.business_meaning)}`).join("；")}</p>` : ""}${assoc.length ? `<p><b>关联账户：</b>${assoc.map(x => `${x.counterparty_id}（${esc(x.label_text)}，${x.direct_txn_count} 笔）`).join("；")}</p>` : ""}<p><b>处置建议：</b>进入人工复核队列，核验上下游账户、关键时间和金额比例。</p></article>`; }).join("") || `<p class="empty">暂无结构化研判报告</p>`;
      document.getElementById("rawReport").textContent = md || "暂无报告原文";
    }
    function openGraph(id) { currentAccount=Number(id); currentWindow=DATA.windows.length ? DATA.windows[DATA.windows.length-1].name : null; setView("graph"); }
    document.addEventListener("DOMContentLoaded", () => { renderOverview(); renderSelectors(); document.querySelectorAll("[data-view]").forEach(btn => btn.onclick=() => setView(btn.dataset.view)); document.querySelectorAll("[data-grade]").forEach(btn => btn.onclick=() => { auditGrade=btn.dataset.grade; document.querySelectorAll("[data-grade]").forEach(x => x.classList.toggle("active", x===btn)); renderAudit(); }); document.getElementById("auditSearch").oninput=e => { auditQuery=e.target.value; renderAudit(); }; renderGraphView(); renderAudit(); renderRecovery(); renderQueue(); renderReports(); });
    """


def render_html_v2(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    script = js_v2().replace("__DATA__", payload)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>动态资金图谱与辅助研判工作台</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <div class="topline">
      <div>
        <div class="eyebrow">江苏银行资金图谱风控 · 动态交易关系</div>
        <h1>动态资金图谱研判工作台</h1>
        <p id="subtitle">动态图谱 · 账户风险子图</p>
      </div>
      <div class="header-meta">
        <span class="meta-pill">split：{escape(str(data["meta"]["split"]))}</span>
        <span class="meta-pill">{escape(str(data["meta"]["window"]))} 窗口</span>
        <span class="meta-pill">Top{escape(str(data["meta"]["top_n"]))} 巡检</span>
        <span class="meta-pill">历史 {escape(str(data["meta"]["history_days"]))} 天</span>
      </div>
    </div>
    <p>观察窗口：{escape(data["meta"]["history_start"])} 至 {escape(data["meta"]["history_end"])}；仅使用历史交易边，金额分箱边界来自 train。</p>
    <nav class="nav" aria-label="页面视图"><button class="nav-btn active" data-view="graph">动态图谱</button><button class="nav-btn" data-view="audit">59账户审计</button><button class="nav-btn" data-view="recovery">缺边恢复</button><button class="nav-btn" data-view="queue">Top{escape(str(data["meta"]["top_n"]))}风险巡检</button><button class="nav-btn" data-view="report">辅助研判报告</button></nav>
  </header>
  <section class="overview" id="overviewMetrics"></section>
  <main id="graphView" class="view active"><div class="graph-layout">
    <aside class="side-panel"><div class="panel-title">高风险账户</div><div id="accounts"></div></aside>
    <section class="graph-panel"><div class="section-head"><div><div class="panel-title">滚动子图</div><p>按时间窗口查看账户交易边、金额分箱和时序资金流信号。</p></div><div id="windows" class="window-row"></div></div><div class="legend"><span><i class="dot" style="background:#c9342d"></i>嫌疑/根账户</span><span><i class="dot" style="background:#e8912d"></i>受害人</span><span><i class="dot" style="background:#2b6cb0"></i>普通账户</span><span>边越粗表示窗口内金额越高</span></div><svg id="graph" viewBox="0 0 980 590" role="img" aria-label="滚动动态资金图谱"></svg><h3>窗口轨迹</h3><table><thead><tr><th>窗口</th><th>交易数</th><th>对手数</th><th>快进快出</th><th>短时闭环</th><th>风险信号</th></tr></thead><tbody id="timeline"></tbody></table></section>
    <section class="right"><div class="panel-title">研判证据</div><div id="evidence"></div><h3>Top 交易边</h3><table><thead><tr><th>边</th><th>笔数</th><th>金额</th><th>分箱</th></tr></thead><tbody id="edgeTable"></tbody></table></section>
  </div></main>
  <main id="auditView" class="view wide"><div class="section-head"><div><h2>59 个确认嫌疑账户分层审计</h2><p>先审计数据覆盖，再判断解释等级；D 级表示当前交易边表未覆盖该账户，不生成虚构链路。</p></div></div><div class="toolbar"><button class="filter-btn active" data-grade="all">全部</button><button class="filter-btn" data-grade="A">A 链路证据</button><button class="filter-btn" data-grade="B">B 直接关联</button><button class="filter-btn" data-grade="C">C 特征证据</button><button class="filter-btn" data-grade="D">D 缺边审计</button><input id="auditSearch" placeholder="搜索账户、标签或证据"><span id="auditCount" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>风险排名</th><th>账户/标签</th><th>模型分</th><th>等级</th><th>解释原因</th><th>历史交易</th><th>历史金额</th><th>直接对手</th><th>证据摘要</th><th>下一步</th></tr></thead><tbody id="auditTableBody"></tbody></table></div><div id="auditDetail" class="detail-panel"><p class="empty">点击上方任一账户查看审计详情。</p></div></main>
  <main id="recoveryView" class="view wide"><div class="section-head"><div><h2>缺边嫌疑账户恢复队列</h2><p>仅展示真实模型/节点证据和补数任务；补充流水后再重建资金链路。</p></div><span id="recoveryCount" class="muted"></span></div><div class="table-wrap"><table><thead><tr><th>优先级</th><th>账户/标签</th><th>模型分</th><th>风险排名</th><th>当前历史交易</th><th>节点画像</th><th>补数查询</th></tr></thead><tbody id="recoveryTableBody"></tbody></table></div><div id="recoveryDetail" class="detail-panel"><p class="empty">点击任一账户查看链路恢复任务。</p></div></main>
  <main id="queueView" class="view wide"><div class="section-head"><div><h2>Top{escape(str(data["meta"]["top_n"]))} 高风险主动巡检队列</h2><p>筛选有历史交易边的高风险账户，输出可核验的关联、路径和资金结构证据。</p></div></div><div class="table-wrap"><table><thead><tr><th>风险排名</th><th>账户/标签</th><th>模型分</th><th>等级</th><th>解释原因</th><th>历史交易</th><th>历史金额</th><th>直接对手</th><th>动态/交易特征证据</th></tr></thead><tbody id="queueTableBody"></tbody></table></div><div id="queueDetail" class="detail-panel"></div></main>
  <main id="reportView" class="view wide"><div class="section-head"><div><h2>辅助研判报告</h2><p>报告同时引用模型风险分、交易统计、关键交易边/路径、图结构或动态特征中的至少两类证据。</p></div></div><div id="reportGrid" class="report-grid"></div><div class="subsection"><h3>报告原文</h3><pre id="rawReport" class="raw-report"></pre></div></main>
  <script>{script}</script>
</body>
</html>
"""


def write_report(data: dict, summaries: pd.DataFrame, edges: pd.DataFrame, html_path: Path) -> Path:
    report_path = DOCS_DIR / "top_accounts_dynamic_report.md"
    lines = [
        "# 滚动动态资金图谱展示报告",
        "",
        "## 1. 展示口径",
        "",
        f"- 预测文件：`{data['meta']['prediction_file']}`",
        f"- 解释 split：`{data['meta']['split']}`",
        f"- 观察窗口：`{data['meta']['history_start']}` 到 `{data['meta']['history_end']}`",
        f"- 窗口粒度：`{data['meta']['window']}`",
        f"- 每个根账户最多保留 Top{data['meta']['top_k_counterparties']} 关联账户。",
        "- 泄露控制：只使用 split end 之前的历史交易；金额分箱边界只由 train 交易分位数计算。",
        "",
        "## 2. 可视化入口",
        "",
        f"- HTML：`{html_path}`",
        "- 页面左侧选择高风险账户，中间切换滚动时间窗口，右侧查看模型分数、时序模体和 Top 交易边证据。",
        "",
        "## 3. Top 账户滚动轨迹摘要",
        "",
        "| 账户 | 标签 | 模型分 | 峰值窗口 | 峰值风险信号 | 总交易数 | 短时闭环合计 | 快进快出合计 | 多入一出合计 |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    if summaries.empty:
        lines.append("| 无 | - | 0 | - | 0 | 0 | 0 | 0 | 0 |")
    else:
        for account in data["accounts"]:
            df = summaries[summaries["root_account_id"].eq(account["id"])].copy()
            if df.empty:
                continue
            peak = df.sort_values("risk_signal_score", ascending=False).iloc[0]
            lines.append(
                "| {id} | {label} | {score:.4f} | {window} | {risk:.3f} | {txn} | {closed} | {fast} | {multi} |".format(
                    id=account["id"],
                    label=account["label_text"],
                    score=float(account["score"]),
                    window=peak["window"],
                    risk=float(peak["risk_signal_score"]),
                    txn=int(df["txn_count"].sum()),
                    closed=int(df["temporal_closed_loop_24h_count"].sum()),
                    fast=int(df["fast_in_out_24h_count"].sum()),
                    multi=int(df["multi_in_one_out_24h_count"].sum()),
                )
            )
    lines.extend(
        [
            "",
            "## 4. 怎么放进答辩",
            "",
            "这张图用来证明模型不是只输出一个黑盒分数，而是在每个滚动窗口里定位账户、交易边、金额分箱和时序资金流模体。",
            "",
            "推荐讲法：",
            "",
            "> 我们按滚动历史窗口构建动态资金图谱，每个窗口只使用 cutoff 之前的转账边。对模型判为高风险的账户，展示其 ego 子图、Top 关联账户、资金流向、金额分箱和快进快出/多入一出等时序模体，从而把风险分数转化为可追溯研判证据。",
            "",
            "## 5. 输出文件",
            "",
            "- `rolling_window_stats.csv`：每个高风险账户在每个滚动窗口里的交易统计和风险信号。",
            "- `top_accounts_dynamic_edges.csv`：HTML 中展示的窗口聚合交易边。",
            "- `top_accounts_dynamic_nodes.csv`：HTML 中展示的节点、标签和分数。",
            "- `dynamic_graph_data.json`：HTML 使用的完整嵌入数据。",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成滚动动态资金图谱可视化 HTML 和报告。")
    parser.add_argument("--prediction-file", default="", help="默认自动选择 v6 动态融合模型预测文件。")
    parser.add_argument("--split", default="test", choices=sorted(SPLITS), help="展示哪个时间窗口。")
    parser.add_argument("--top-n", type=int, default=30, help="展示 Top N 高风险账户，默认与主动巡检队列保持一致。")
    parser.add_argument("--top-k-counterparties", type=int, default=20, help="每个账户最多展示多少个关联账户。")
    parser.add_argument("--history-days", type=int, default=120, help="向前滚动观察多少天历史交易。")
    parser.add_argument("--window", default="monthly", choices=["monthly", "weekly"], help="滚动快照粒度。")
    parser.add_argument("--max-edges-per-window", type=int, default=80, help="每个窗口最多展示多少条聚合边。")
    parser.add_argument("--confirmed-only", action="store_true", help="只展示已确认嫌疑人，用于典型案例复盘。")
    parser.add_argument("--allow-inactive", action="store_true", help="允许展示历史窗口无交易的高分账户。")
    parser.add_argument("--account-ids", default="", help="手动指定账户 ID，逗号分隔。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    pred_path = Path(args.prediction_file) if args.prediction_file else find_prediction_file()
    args.prediction_file_resolved = pred_path
    predictions, labels, transactions = load_inputs(pred_path, args.split)
    top_accounts = choose_accounts(predictions, transactions, args)
    data, summaries, edges, nodes = build_visual_data(top_accounts, predictions, labels, transactions, args)
    data["layered"] = load_layered_data()

    summary_path = DYNAMIC_GRAPH_DIR / "rolling_window_stats.csv"
    edge_path = DYNAMIC_GRAPH_DIR / "top_accounts_dynamic_edges.csv"
    node_path = DYNAMIC_GRAPH_DIR / "top_accounts_dynamic_nodes.csv"
    data_path = DYNAMIC_GRAPH_DIR / "dynamic_graph_data.json"
    html_path = DYNAMIC_GRAPH_DIR / "index.html"

    summaries.to_csv(summary_path, index=False)
    edges.to_csv(edge_path, index=False)
    nodes.to_csv(node_path, index=False)
    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html_v2(data), encoding="utf-8")
    report_path = write_report(data, summaries, edges, html_path)

    result = {
        "prediction_file": str(pred_path),
        "split": args.split,
        "top_account_count": int(len(top_accounts)),
        "window_count": int(len(data["windows"])),
        "summary_rows": int(len(summaries)),
        "edge_rows": int(len(edges)),
        "node_rows": int(len(nodes)),
        "outputs": {
            "html": str(html_path),
            "markdown_report": str(report_path),
            "rolling_window_stats": str(summary_path),
            "dynamic_edges": str(edge_path),
            "dynamic_nodes": str(node_path),
            "data_json": str(data_path),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
