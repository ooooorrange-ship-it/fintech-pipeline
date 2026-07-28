import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import AMOUNT_COL, CLEAN_DIR, DST_COL, FEATURE_DIR, ID_COL, SPLITS, SRC_COL, TIME_COL  # noqa: E402


AMOUNT_BIN_COUNT = 5
MEMORY_HALFLIFE_HOURS = [1, 6, 24, 168]
MOTIF_HORIZON_HOURS = [1, 6, 24]
ROLLING_HISTORY_DAYS = 120


def ensure_dirs() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).fillna(0.0)


def safe_scalar_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def amount_edges_from_train(transactions: pd.DataFrame) -> list[float]:
    start, end = SPLITS["train"]
    train_tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ]
    quantiles = np.linspace(0.0, 1.0, AMOUNT_BIN_COUNT + 1)
    edges = train_tx["amount_abs"].quantile(quantiles).to_numpy(dtype=float)
    edges = sorted(set(float(x) for x in edges if np.isfinite(x)))
    if len(edges) < 2:
        edges = [0.0, float(train_tx["amount_abs"].max()) + 1.0]
    edges[0] = min(0.0, edges[0])
    edges[-1] = edges[-1] + 1e-9
    return edges


def observation_window(transactions: pd.DataFrame, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    _, end = SPLITS[split]
    end_ts = pd.Timestamp(end)
    min_ts = transactions[TIME_COL].min()
    start_ts = max(pd.Timestamp(min_ts), end_ts - pd.Timedelta(days=ROLLING_HISTORY_DAYS))
    return start_ts, end_ts


def add_dynamic_edge_fields(tx: pd.DataFrame, amount_edges: list[float]) -> pd.DataFrame:
    out = tx.copy()
    labels = [f"b{i}" for i in range(len(amount_edges) - 1)]
    out["dyn_amount_bin"] = pd.cut(
        out["amount_abs"],
        bins=amount_edges,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    out["dyn_week_bucket"] = out[TIME_COL].dt.to_period("W").astype(str)
    out["dyn_day_bucket"] = out[TIME_COL].dt.date.astype(str)
    out["dyn_hour_bucket"] = out[TIME_COL].dt.hour.astype(int)
    out["dyn_part_of_day"] = pd.cut(
        out["dyn_hour_bucket"],
        bins=[-1, 5, 11, 17, 23],
        labels=["night", "morning", "afternoon", "evening"],
    ).astype(str)
    return out


def bucket_aggregate(events: pd.DataFrame, prefix: str, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    if events.empty:
        return base

    by_bucket = events.groupby([ID_COL, "dyn_week_bucket"], sort=False).agg(
        bucket_txn_count=("amount_abs", "size"),
        bucket_amount_sum=("amount_abs", "sum"),
        bucket_counterparty_count=("counterparty_id", "nunique"),
    )
    stats = by_bucket.groupby(level=0).agg(
        **{
            f"{prefix}_active_week_count": ("bucket_txn_count", "size"),
            f"{prefix}_week_txn_mean": ("bucket_txn_count", "mean"),
            f"{prefix}_week_txn_std": ("bucket_txn_count", "std"),
            f"{prefix}_week_txn_max": ("bucket_txn_count", "max"),
            f"{prefix}_week_amount_mean": ("bucket_amount_sum", "mean"),
            f"{prefix}_week_amount_std": ("bucket_amount_sum", "std"),
            f"{prefix}_week_amount_max": ("bucket_amount_sum", "max"),
            f"{prefix}_week_counterparty_mean": ("bucket_counterparty_count", "mean"),
            f"{prefix}_week_counterparty_max": ("bucket_counterparty_count", "max"),
            f"{prefix}_total_bucket_txn": ("bucket_txn_count", "sum"),
            f"{prefix}_total_bucket_amount": ("bucket_amount_sum", "sum"),
        }
    ).fillna(0)
    stats[f"{prefix}_week_txn_cv"] = safe_ratio(stats[f"{prefix}_week_txn_std"], stats[f"{prefix}_week_txn_mean"])
    stats[f"{prefix}_week_amount_cv"] = safe_ratio(stats[f"{prefix}_week_amount_std"], stats[f"{prefix}_week_amount_mean"])
    stats[f"{prefix}_week_txn_burst_ratio"] = safe_ratio(stats[f"{prefix}_week_txn_max"], stats[f"{prefix}_total_bucket_txn"])
    stats[f"{prefix}_week_amount_burst_ratio"] = safe_ratio(
        stats[f"{prefix}_week_amount_max"], stats[f"{prefix}_total_bucket_amount"]
    )
    stats = stats.drop(columns=[f"{prefix}_total_bucket_txn", f"{prefix}_total_bucket_amount"])

    pivot_txn = by_bucket["bucket_txn_count"].unstack(fill_value=0).sort_index(axis=1)
    if pivot_txn.shape[1] >= 2:
        stats[f"{prefix}_last_week_txn"] = pivot_txn.iloc[:, -1]
        stats[f"{prefix}_prev_week_txn"] = pivot_txn.iloc[:, -2]
        stats[f"{prefix}_last_prev_week_txn_growth"] = safe_ratio(
            stats[f"{prefix}_last_week_txn"] - stats[f"{prefix}_prev_week_txn"],
            stats[f"{prefix}_prev_week_txn"] + 1,
        )
    else:
        stats[f"{prefix}_last_week_txn"] = pivot_txn.iloc[:, -1] if pivot_txn.shape[1] else 0
        stats[f"{prefix}_prev_week_txn"] = 0.0
        stats[f"{prefix}_last_prev_week_txn_growth"] = 0.0

    return base.merge(stats.reset_index(), on=ID_COL, how="left").fillna(0)


def categorical_distribution(events: pd.DataFrame, category_col: str, values: list[str], prefix: str, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    if events.empty:
        for value in values:
            base[f"{prefix}_{value}_count"] = 0.0
            base[f"{prefix}_{value}_ratio"] = 0.0
        base[f"{prefix}_entropy"] = 0.0
        return base

    counts = events.groupby([ID_COL, category_col], sort=False).size().unstack(fill_value=0)
    counts = counts.reindex(columns=values, fill_value=0)
    total = counts.sum(axis=1)
    out = counts.add_prefix(f"{prefix}_").add_suffix("_count")
    for value in values:
        out[f"{prefix}_{value}_ratio"] = safe_ratio(out[f"{prefix}_{value}_count"], total)
    prob = counts.div(total.replace(0, np.nan), axis=0).fillna(0)
    out[f"{prefix}_entropy"] = -(prob * np.log(prob.clip(lower=1e-12))).sum(axis=1)
    return base.merge(out.reset_index(), on=ID_COL, how="left").fillna(0)


def dynamic_graph_snapshot_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    if tx.empty:
        return base

    pair_bucket = tx[[SRC_COL, DST_COL, "dyn_week_bucket"]].drop_duplicates()
    out_deg = pair_bucket.groupby([SRC_COL, "dyn_week_bucket"])[DST_COL].nunique().rename("out_degree")
    in_deg = pair_bucket.groupby([DST_COL, "dyn_week_bucket"])[SRC_COL].nunique().rename("in_degree")
    out_stats = out_deg.groupby(level=0).agg(
        dyn_graph_out_degree_week_mean="mean",
        dyn_graph_out_degree_week_std="std",
        dyn_graph_out_degree_week_max="max",
        dyn_graph_out_active_week_count="size",
    ).fillna(0)
    in_stats = in_deg.groupby(level=0).agg(
        dyn_graph_in_degree_week_mean="mean",
        dyn_graph_in_degree_week_std="std",
        dyn_graph_in_degree_week_max="max",
        dyn_graph_in_active_week_count="size",
    ).fillna(0)
    out_stats["dyn_graph_out_degree_week_cv"] = safe_ratio(
        out_stats["dyn_graph_out_degree_week_std"], out_stats["dyn_graph_out_degree_week_mean"]
    )
    in_stats["dyn_graph_in_degree_week_cv"] = safe_ratio(
        in_stats["dyn_graph_in_degree_week_std"], in_stats["dyn_graph_in_degree_week_mean"]
    )

    result = base.merge(out_stats.reset_index().rename(columns={SRC_COL: ID_COL}), on=ID_COL, how="left")
    result = result.merge(in_stats.reset_index().rename(columns={DST_COL: ID_COL}), on=ID_COL, how="left")
    return result.fillna(0)


def counterparty_churn_features(events: pd.DataFrame, accounts: pd.DataFrame, split_start: str, split_end: str) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    if events.empty:
        return base.assign(
            dyn_cp_first_half_count=0.0,
            dyn_cp_second_half_count=0.0,
            dyn_cp_new_second_half_count=0.0,
            dyn_cp_lost_second_half_count=0.0,
            dyn_cp_new_second_half_ratio=0.0,
            dyn_cp_jaccard_first_second=0.0,
        )

    start_ts = pd.Timestamp(split_start)
    end_ts = pd.Timestamp(split_end)
    mid_ts = start_ts + (end_ts - start_ts) / 2
    rows = []
    for account_id, group in events.groupby(ID_COL, sort=False):
        first = set(group.loc[group[TIME_COL].lt(mid_ts), "counterparty_id"].astype(int))
        second = set(group.loc[group[TIME_COL].ge(mid_ts), "counterparty_id"].astype(int))
        union = first | second
        rows.append(
            {
                ID_COL: int(account_id),
                "dyn_cp_first_half_count": len(first),
                "dyn_cp_second_half_count": len(second),
                "dyn_cp_new_second_half_count": len(second - first),
                "dyn_cp_lost_second_half_count": len(first - second),
                "dyn_cp_new_second_half_ratio": safe_scalar_ratio(len(second - first), len(second)),
                "dyn_cp_jaccard_first_second": safe_scalar_ratio(len(first & second), len(union)),
            }
        )
    return base.merge(pd.DataFrame(rows), on=ID_COL, how="left").fillna(0)


def dynamic_temporal_motif_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    empty = {ID_COL: accounts[ID_COL].astype(int).to_numpy()}
    for h in MOTIF_HORIZON_HOURS:
        empty[f"dyn_motif_fast_in_out_count_{h}h"] = 0.0
        empty[f"dyn_motif_fast_in_out_in_amount_ratio_{h}h"] = 0.0
        empty[f"dyn_motif_fast_in_out_out_amount_ratio_{h}h"] = 0.0
        empty[f"dyn_motif_fast_in_out_balance_ratio_{h}h"] = 0.0
        empty[f"dyn_motif_fast_in_out_mean_delay_min_{h}h"] = 0.0
        empty[f"dyn_motif_prior_in_before_out_count_{h}h"] = 0.0
        empty[f"dyn_motif_prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
        empty[f"dyn_motif_prior_in_before_out_mean_delay_min_{h}h"] = 0.0
    empty.update(
        {
            "dyn_motif_one_in_multi_out_count_24h": 0.0,
            "dyn_motif_one_in_multi_out_in_amount_ratio_24h": 0.0,
            "dyn_motif_multi_in_one_out_count_24h": 0.0,
            "dyn_motif_multi_in_one_out_mean_amount_ratio_24h": 0.0,
            "dyn_motif_max_prior_in_count_before_out_24h": 0.0,
            "dyn_motif_max_next_out_count_after_in_24h": 0.0,
            "dyn_motif_low_bin_before_high_bin_out_count_24h": 0.0,
        }
    )
    if tx.empty:
        return pd.DataFrame(empty)

    in_groups = {
        int(k): g[[TIME_COL, "amount_abs", "dyn_amount_bin"]].sort_values(TIME_COL)
        for k, g in tx.groupby(DST_COL, sort=False)
    }
    out_groups = {
        int(k): g[[TIME_COL, "amount_abs", "dyn_amount_bin"]].sort_values(TIME_COL)
        for k, g in tx.groupby(SRC_COL, sort=False)
    }
    high_bin_name = f"b{AMOUNT_BIN_COUNT - 1}"

    rows: list[dict] = []
    for account_id in accounts[ID_COL].astype(int):
        account_id = int(account_id)
        ins = in_groups.get(account_id)
        outs = out_groups.get(account_id)
        row = {ID_COL: account_id}

        if ins is None:
            in_times = np.array([], dtype="datetime64[ns]")
            in_amounts = np.array([], dtype=float)
            in_bins = np.array([], dtype=str)
        else:
            in_times = ins[TIME_COL].to_numpy(dtype="datetime64[ns]")
            in_amounts = ins["amount_abs"].to_numpy(dtype=float)
            in_bins = ins["dyn_amount_bin"].astype(str).to_numpy()
        if outs is None:
            out_times = np.array([], dtype="datetime64[ns]")
            out_amounts = np.array([], dtype=float)
            out_bins = np.array([], dtype=str)
        else:
            out_times = outs[TIME_COL].to_numpy(dtype="datetime64[ns]")
            out_amounts = outs["amount_abs"].to_numpy(dtype=float)
            out_bins = outs["dyn_amount_bin"].astype(str).to_numpy()

        total_in_amount = float(in_amounts.sum())
        total_out_amount = float(out_amounts.sum())
        in_prefix = np.concatenate([[0.0], np.cumsum(in_amounts)])
        out_prefix = np.concatenate([[0.0], np.cumsum(out_amounts)])

        for h in MOTIF_HORIZON_HOURS:
            delta = np.timedelta64(h, "h")
            if len(in_times) and len(out_times):
                left = np.searchsorted(out_times, in_times, side="right")
                right = np.searchsorted(out_times, in_times + delta, side="right")
                matched = right > left
                matched_amount = float(in_amounts[matched].sum())
                matched_out_amounts = out_prefix[right] - out_prefix[left]
                matched_out_amount = float(matched_out_amounts[matched].sum())
                balance_amount = float(np.minimum(matched_out_amounts[matched], in_amounts[matched]).sum())
                delays = (
                    (out_times[left[matched]] - in_times[matched]).astype("timedelta64[s]").astype(float) / 60.0
                    if matched.any()
                    else np.array([], dtype=float)
                )
                row[f"dyn_motif_fast_in_out_count_{h}h"] = int(matched.sum())
                row[f"dyn_motif_fast_in_out_in_amount_ratio_{h}h"] = safe_scalar_ratio(matched_amount, total_in_amount)
                row[f"dyn_motif_fast_in_out_out_amount_ratio_{h}h"] = safe_scalar_ratio(matched_out_amount, total_out_amount)
                row[f"dyn_motif_fast_in_out_balance_ratio_{h}h"] = safe_scalar_ratio(balance_amount, total_in_amount)
                row[f"dyn_motif_fast_in_out_mean_delay_min_{h}h"] = float(np.mean(delays)) if len(delays) else 0.0

                prior_left = np.searchsorted(in_times, out_times - delta, side="left")
                prior_right = np.searchsorted(in_times, out_times, side="left")
                prior_matched = prior_right > prior_left
                if prior_matched.any():
                    last_in_times = in_times[prior_right[prior_matched] - 1]
                    prior_delays = (out_times[prior_matched] - last_in_times).astype("timedelta64[s]").astype(float) / 60.0
                    prior_out_amount = float(out_amounts[prior_matched].sum())
                    row[f"dyn_motif_prior_in_before_out_count_{h}h"] = int(prior_matched.sum())
                    row[f"dyn_motif_prior_in_before_out_out_amount_ratio_{h}h"] = safe_scalar_ratio(
                        prior_out_amount, total_out_amount
                    )
                    row[f"dyn_motif_prior_in_before_out_mean_delay_min_{h}h"] = float(np.mean(prior_delays))
                else:
                    row[f"dyn_motif_prior_in_before_out_count_{h}h"] = 0.0
                    row[f"dyn_motif_prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
                    row[f"dyn_motif_prior_in_before_out_mean_delay_min_{h}h"] = 0.0
            else:
                row[f"dyn_motif_fast_in_out_count_{h}h"] = 0.0
                row[f"dyn_motif_fast_in_out_in_amount_ratio_{h}h"] = 0.0
                row[f"dyn_motif_fast_in_out_out_amount_ratio_{h}h"] = 0.0
                row[f"dyn_motif_fast_in_out_balance_ratio_{h}h"] = 0.0
                row[f"dyn_motif_fast_in_out_mean_delay_min_{h}h"] = 0.0
                row[f"dyn_motif_prior_in_before_out_count_{h}h"] = 0.0
                row[f"dyn_motif_prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
                row[f"dyn_motif_prior_in_before_out_mean_delay_min_{h}h"] = 0.0

        one_in_multi_count = 0
        one_in_multi_amount = 0.0
        multi_in_one_count = 0
        multi_in_one_ratios: list[float] = []
        max_next_out_count = 0
        max_prior_in_count = 0
        low_before_high_out_count = 0

        if len(in_times) and len(out_times):
            delta = np.timedelta64(24, "h")
            next_left = np.searchsorted(out_times, in_times, side="right")
            next_right = np.searchsorted(out_times, in_times + delta, side="right")
            next_counts = next_right - next_left
            max_next_out_count = int(next_counts.max()) if len(next_counts) else 0
            one_mask = next_counts >= 3
            one_in_multi_count = int(one_mask.sum())
            one_in_multi_amount = float(in_amounts[one_mask].sum())

            prior_left = np.searchsorted(in_times, out_times - delta, side="left")
            prior_right = np.searchsorted(in_times, out_times, side="left")
            prior_counts = prior_right - prior_left
            max_prior_in_count = int(prior_counts.max()) if len(prior_counts) else 0
            for i, (left_idx, right_idx) in enumerate(zip(prior_left, prior_right)):
                if right_idx <= left_idx:
                    continue
                prior_amount = float(in_prefix[right_idx] - in_prefix[left_idx])
                ratio = safe_scalar_ratio(float(out_amounts[i]), prior_amount)
                if prior_counts[i] >= 3 and 0.5 <= ratio <= 1.5:
                    multi_in_one_count += 1
                    multi_in_one_ratios.append(ratio)
                if out_bins[i] == high_bin_name and np.any(in_bins[left_idx:right_idx] == "b0"):
                    low_before_high_out_count += 1

        row["dyn_motif_one_in_multi_out_count_24h"] = one_in_multi_count
        row["dyn_motif_one_in_multi_out_in_amount_ratio_24h"] = safe_scalar_ratio(one_in_multi_amount, total_in_amount)
        row["dyn_motif_multi_in_one_out_count_24h"] = multi_in_one_count
        row["dyn_motif_multi_in_one_out_mean_amount_ratio_24h"] = (
            float(np.mean(multi_in_one_ratios)) if multi_in_one_ratios else 0.0
        )
        row["dyn_motif_max_prior_in_count_before_out_24h"] = max_prior_in_count
        row["dyn_motif_max_next_out_count_after_in_24h"] = max_next_out_count
        row["dyn_motif_low_bin_before_high_bin_out_count_24h"] = low_before_high_out_count
        rows.append(row)

    return pd.DataFrame(rows).replace([np.inf, -np.inf], 0).fillna(0)


def temporal_memory_features(tx: pd.DataFrame, accounts: pd.DataFrame, split_end: str) -> pd.DataFrame:
    ids = accounts[ID_COL].astype(int).tolist()
    id_to_idx = {account_id: idx for idx, account_id in enumerate(ids)}
    n = len(ids)
    h = len(MEMORY_HALFLIFE_HOURS)

    in_count = np.zeros((n, h), dtype=float)
    out_count = np.zeros((n, h), dtype=float)
    in_amount = np.zeros((n, h), dtype=float)
    out_amount = np.zeros((n, h), dtype=float)
    last_update = np.full(n, np.nan, dtype=float)
    last_event = np.full(n, np.nan, dtype=float)
    last_in = np.full(n, np.nan, dtype=float)
    last_out = np.full(n, np.nan, dtype=float)
    high_bin_count = np.zeros((n, h), dtype=float)
    low_bin_count = np.zeros((n, h), dtype=float)
    night_count = np.zeros((n, h), dtype=float)

    half_life = np.array(MEMORY_HALFLIFE_HOURS, dtype=float)

    def decay_node(idx: int, now_hour: float) -> None:
        prev = last_update[idx]
        if not np.isfinite(prev):
            last_update[idx] = now_hour
            return
        elapsed = max(0.0, now_hour - prev)
        if elapsed:
            factor = np.power(0.5, elapsed / half_life)
            in_count[idx] *= factor
            out_count[idx] *= factor
            in_amount[idx] *= factor
            out_amount[idx] *= factor
            high_bin_count[idx] *= factor
            low_bin_count[idx] *= factor
            night_count[idx] *= factor
        last_update[idx] = now_hour

    if tx.empty:
        return pd.DataFrame({ID_COL: ids})

    start_ts = tx[TIME_COL].min()
    end_ts = pd.Timestamp(split_end)
    tx_sorted = tx.sort_values(TIME_COL)
    high_bin_name = f"b{AMOUNT_BIN_COUNT - 1}"

    for rec in tx_sorted[[SRC_COL, DST_COL, TIME_COL, "amount_abs", "dyn_amount_bin", "dyn_part_of_day"]].itertuples(index=False):
        src = id_to_idx.get(int(rec[0]))
        dst = id_to_idx.get(int(rec[1]))
        if src is None or dst is None:
            continue
        now_hour = (pd.Timestamp(rec[2]) - start_ts).total_seconds() / 3600.0
        amount = float(rec[3])
        amount_bin = str(rec[4])
        part_of_day = str(rec[5])

        decay_node(src, now_hour)
        decay_node(dst, now_hour)

        out_count[src] += 1.0
        out_amount[src] += amount
        in_count[dst] += 1.0
        in_amount[dst] += amount
        if amount_bin == high_bin_name:
            high_bin_count[src] += 1.0
            high_bin_count[dst] += 1.0
        if amount_bin == "b0":
            low_bin_count[src] += 1.0
            low_bin_count[dst] += 1.0
        if part_of_day == "night":
            night_count[src] += 1.0
            night_count[dst] += 1.0

        last_event[src] = now_hour
        last_event[dst] = now_hour
        last_out[src] = now_hour
        last_in[dst] = now_hour

    end_hour = (end_ts - start_ts).total_seconds() / 3600.0 if pd.notna(start_ts) else 0.0
    for idx in range(n):
        decay_node(idx, end_hour)

    rows = {ID_COL: ids}
    for j, half in enumerate(MEMORY_HALFLIFE_HOURS):
        suffix = f"{half}h" if half < 168 else "7d"
        rows[f"dyn_mem_in_count_{suffix}"] = in_count[:, j]
        rows[f"dyn_mem_out_count_{suffix}"] = out_count[:, j]
        rows[f"dyn_mem_total_count_{suffix}"] = in_count[:, j] + out_count[:, j]
        rows[f"dyn_mem_in_amount_{suffix}"] = in_amount[:, j]
        rows[f"dyn_mem_out_amount_{suffix}"] = out_amount[:, j]
        rows[f"dyn_mem_net_amount_{suffix}"] = in_amount[:, j] - out_amount[:, j]
        rows[f"dyn_mem_in_out_amount_ratio_{suffix}"] = np.divide(
            in_amount[:, j],
            np.where(out_amount[:, j] > 1e-9, out_amount[:, j], np.nan),
        )
        rows[f"dyn_mem_high_bin_ratio_{suffix}"] = np.divide(
            high_bin_count[:, j],
            np.where(in_count[:, j] + out_count[:, j] > 1e-9, in_count[:, j] + out_count[:, j], np.nan),
        )
        rows[f"dyn_mem_low_bin_ratio_{suffix}"] = np.divide(
            low_bin_count[:, j],
            np.where(in_count[:, j] + out_count[:, j] > 1e-9, in_count[:, j] + out_count[:, j], np.nan),
        )
        rows[f"dyn_mem_night_ratio_{suffix}"] = np.divide(
            night_count[:, j],
            np.where(in_count[:, j] + out_count[:, j] > 1e-9, in_count[:, j] + out_count[:, j], np.nan),
        )

    last_event_age = np.where(np.isfinite(last_event), end_hour - last_event, 0.0)
    last_in_age = np.where(np.isfinite(last_in), end_hour - last_in, 0.0)
    last_out_age = np.where(np.isfinite(last_out), end_hour - last_out, 0.0)
    rows["dyn_mem_last_event_age_hours"] = last_event_age
    rows["dyn_mem_last_in_age_hours"] = last_in_age
    rows["dyn_mem_last_out_age_hours"] = last_out_age
    out = pd.DataFrame(rows)
    return out.replace([np.inf, -np.inf], 0).fillna(0)


def build_dynamic_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    split: str,
    amount_edges: list[float],
) -> pd.DataFrame:
    start_ts, end_ts = observation_window(transactions, split)
    tx = transactions[
        (transactions[TIME_COL] >= start_ts)
        & (transactions[TIME_COL] <= end_ts)
    ].copy()
    tx = add_dynamic_edge_fields(tx, amount_edges)

    out_events = tx[[SRC_COL, DST_COL, TIME_COL, "amount_abs", "dyn_week_bucket", "dyn_amount_bin", "dyn_part_of_day"]].rename(
        columns={SRC_COL: ID_COL, DST_COL: "counterparty_id"}
    )
    in_events = tx[[DST_COL, SRC_COL, TIME_COL, "amount_abs", "dyn_week_bucket", "dyn_amount_bin", "dyn_part_of_day"]].rename(
        columns={DST_COL: ID_COL, SRC_COL: "counterparty_id"}
    )
    out_events["dyn_direction"] = "out"
    in_events["dyn_direction"] = "in"
    all_events = pd.concat([out_events, in_events], ignore_index=True)

    features = accounts[[ID_COL]].copy()
    features = features.merge(bucket_aggregate(all_events, "dyn_total", accounts), on=ID_COL, how="left")
    features = features.merge(bucket_aggregate(out_events, "dyn_out", accounts), on=ID_COL, how="left")
    features = features.merge(bucket_aggregate(in_events, "dyn_in", accounts), on=ID_COL, how="left")

    bin_values = [f"b{i}" for i in range(len(amount_edges) - 1)]
    features = features.merge(
        categorical_distribution(all_events, "dyn_amount_bin", bin_values, "dyn_total_amount_bin", accounts),
        on=ID_COL,
        how="left",
    )
    features = features.merge(
        categorical_distribution(out_events, "dyn_amount_bin", bin_values, "dyn_out_amount_bin", accounts),
        on=ID_COL,
        how="left",
    )
    features = features.merge(
        categorical_distribution(in_events, "dyn_amount_bin", bin_values, "dyn_in_amount_bin", accounts),
        on=ID_COL,
        how="left",
    )

    day_parts = ["night", "morning", "afternoon", "evening"]
    features = features.merge(
        categorical_distribution(all_events, "dyn_part_of_day", day_parts, "dyn_total_time_bucket", accounts),
        on=ID_COL,
        how="left",
    )
    features = features.merge(dynamic_graph_snapshot_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(counterparty_churn_features(all_events, accounts, str(start_ts), str(end_ts)), on=ID_COL, how="left")
    features = features.merge(dynamic_temporal_motif_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(temporal_memory_features(tx, accounts, str(end_ts)), on=ID_COL, how="left")

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    return features.fillna(0)


def main() -> None:
    ensure_dirs()
    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    amount_edges = amount_edges_from_train(transactions)

    report = {
        "feature_family": "dynamic_graph_time_amount_bucket_features",
        "amount_bin_count": len(amount_edges) - 1,
        "observation_policy": f"rolling_{ROLLING_HISTORY_DAYS}_days_until_split_end",
        "temporal_motif_horizon_hours": MOTIF_HORIZON_HOURS,
        "temporal_memory_halflife_hours": MEMORY_HALFLIFE_HOURS,
        "amount_bin_edges_from_train": amount_edges,
        "time_buckets": ["dyn_week_bucket", "dyn_day_bucket", "dyn_hour_bucket", "dyn_part_of_day"],
        "splits": {},
        "leakage_policy": "金额分箱边界仅由 train 窗口计算；每个 split 只使用截至 split end 的滚动历史交易边，不使用 split end 之后的未来交易。",
    }
    for split in SPLITS:
        feat = build_dynamic_features(accounts, transactions, split, amount_edges)
        path = FEATURE_DIR / f"dynamic_graph_features_{split}.csv"
        feat.to_csv(path, index=False)
        start_ts, end_ts = observation_window(transactions, split)
        report["splits"][split] = {
            "observation_start": str(start_ts),
            "observation_end": str(end_ts),
            "rows": int(len(feat)),
            "cols": int(feat.shape[1]),
            "path": str(path),
        }

    with (FEATURE_DIR / "dynamic_graph_feature_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
