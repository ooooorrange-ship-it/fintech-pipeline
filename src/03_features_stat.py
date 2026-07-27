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
    FEATURE_DIR,
    ID_COL,
    SPLITS,
    SRC_COL,
    TIME_COL,
)


def ensure_dirs() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).fillna(0.0)


def aggregate_side(
    tx: pd.DataFrame,
    account_col: str,
    prefix: str,
    counterparty_col: str,
) -> pd.DataFrame:
    if tx.empty:
        return pd.DataFrame(columns=[ID_COL])

    grouped = tx.groupby(account_col, sort=False)
    out = grouped.agg(
        **{
            f"{prefix}_txn_count": (AMOUNT_COL, "size"),
            f"{prefix}_amount_sum": ("amount_abs", "sum"),
            f"{prefix}_amount_mean": ("amount_abs", "mean"),
            f"{prefix}_amount_max": ("amount_abs", "max"),
            f"{prefix}_amount_median": ("amount_abs", "median"),
            f"{prefix}_counterparty_nunique": (counterparty_col, "nunique"),
            f"{prefix}_active_days": ("txn_date", "nunique"),
            f"{prefix}_active_months": ("txn_month", "nunique"),
            f"{prefix}_self_loop_count": ("self_loop", "sum"),
            f"{prefix}_non_positive_amount_count": ("non_positive_amount", "sum"),
            f"{prefix}_negative_amount_count": ("negative_amount", "sum"),
            f"{prefix}_zero_amount_count": ("zero_amount", "sum"),
            f"{prefix}_night_txn_count": ("is_night_txn", "sum"),
            f"{prefix}_round_1000_count": ("is_round_1000", "sum"),
            f"{prefix}_small_amount_count": ("is_small_amount", "sum"),
            f"{prefix}_large_amount_count": ("is_large_amount", "sum"),
        }
    )
    out = out.reset_index().rename(columns={account_col: ID_COL})
    out[f"{prefix}_txn_per_active_day"] = safe_ratio(out[f"{prefix}_txn_count"], out[f"{prefix}_active_days"])
    out[f"{prefix}_amount_per_txn"] = safe_ratio(out[f"{prefix}_amount_sum"], out[f"{prefix}_txn_count"])
    out[f"{prefix}_night_txn_ratio"] = safe_ratio(out[f"{prefix}_night_txn_count"], out[f"{prefix}_txn_count"])
    out[f"{prefix}_round_1000_ratio"] = safe_ratio(out[f"{prefix}_round_1000_count"], out[f"{prefix}_txn_count"])
    out[f"{prefix}_small_amount_ratio"] = safe_ratio(out[f"{prefix}_small_amount_count"], out[f"{prefix}_txn_count"])
    out[f"{prefix}_large_amount_ratio"] = safe_ratio(out[f"{prefix}_large_amount_count"], out[f"{prefix}_txn_count"])
    return out


def monthly_trend_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    if tx.empty:
        return accounts[[ID_COL]].assign(
            monthly_txn_count_std=0.0,
            monthly_amount_sum_std=0.0,
            first_month_txn_count=0.0,
            last_month_txn_count=0.0,
            txn_count_growth=0.0,
        )

    out_tx = tx.groupby([SRC_COL, "txn_month"]).agg(
        monthly_txn_count=(AMOUNT_COL, "size"),
        monthly_amount_sum=("amount_abs", "sum"),
    )
    pivot_count = out_tx["monthly_txn_count"].unstack(fill_value=0)
    pivot_amount = out_tx["monthly_amount_sum"].unstack(fill_value=0)

    trend = pd.DataFrame(index=pivot_count.index)
    trend["monthly_txn_count_std"] = pivot_count.std(axis=1).fillna(0)
    trend["monthly_amount_sum_std"] = pivot_amount.std(axis=1).fillna(0)
    trend["monthly_txn_count_mean"] = pivot_count.mean(axis=1).fillna(0)
    trend["monthly_amount_sum_mean"] = pivot_amount.mean(axis=1).fillna(0)
    trend["monthly_txn_count_cv"] = safe_ratio(trend["monthly_txn_count_std"], trend["monthly_txn_count_mean"])
    trend["monthly_amount_sum_cv"] = safe_ratio(trend["monthly_amount_sum_std"], trend["monthly_amount_sum_mean"])
    trend["first_month_txn_count"] = pivot_count.iloc[:, 0] if pivot_count.shape[1] else 0
    trend["last_month_txn_count"] = pivot_count.iloc[:, -1] if pivot_count.shape[1] else 0
    trend["txn_count_growth"] = safe_ratio(
        trend["last_month_txn_count"] - trend["first_month_txn_count"],
        trend["first_month_txn_count"] + 1,
    )
    return trend.reset_index().rename(columns={SRC_COL: ID_COL})


def daily_burst_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    if tx.empty:
        return base.assign(
            daily_txn_count_mean=0.0,
            daily_txn_count_std=0.0,
            daily_txn_count_max=0.0,
            daily_txn_count_cv=0.0,
            daily_amount_sum_mean=0.0,
            daily_amount_sum_std=0.0,
            daily_amount_sum_max=0.0,
            daily_amount_sum_cv=0.0,
            burst_day_txn_ratio=0.0,
            burst_day_amount_ratio=0.0,
        )

    payer = tx[[SRC_COL, "txn_date", "amount_abs"]].rename(columns={SRC_COL: ID_COL})
    receiver = tx[[DST_COL, "txn_date", "amount_abs"]].rename(columns={DST_COL: ID_COL})
    events = pd.concat([payer, receiver], ignore_index=True)
    daily = events.groupby([ID_COL, "txn_date"], sort=False).agg(
        daily_txn_count=("amount_abs", "size"),
        daily_amount_sum=("amount_abs", "sum"),
    )
    stats = daily.groupby(ID_COL).agg(
        daily_txn_count_mean=("daily_txn_count", "mean"),
        daily_txn_count_std=("daily_txn_count", "std"),
        daily_txn_count_max=("daily_txn_count", "max"),
        daily_amount_sum_mean=("daily_amount_sum", "mean"),
        daily_amount_sum_std=("daily_amount_sum", "std"),
        daily_amount_sum_max=("daily_amount_sum", "max"),
        total_active_day_txn_count=("daily_txn_count", "sum"),
        total_active_day_amount=("daily_amount_sum", "sum"),
    ).fillna(0)
    stats["daily_txn_count_cv"] = safe_ratio(stats["daily_txn_count_std"], stats["daily_txn_count_mean"])
    stats["daily_amount_sum_cv"] = safe_ratio(stats["daily_amount_sum_std"], stats["daily_amount_sum_mean"])
    stats["burst_day_txn_ratio"] = safe_ratio(stats["daily_txn_count_max"], stats["total_active_day_txn_count"])
    stats["burst_day_amount_ratio"] = safe_ratio(stats["daily_amount_sum_max"], stats["total_active_day_amount"])
    stats = stats.drop(columns=["total_active_day_txn_count", "total_active_day_amount"])
    return base.merge(stats.reset_index(), on=ID_COL, how="left").fillna(0)


def activity_timing_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    empty_cols = {
        "activity_hour_nunique": 0.0,
        "activity_hour_entropy": 0.0,
        "activity_hour_txn_max_ratio": 0.0,
        "txn_interarrival_min_mean": 0.0,
        "txn_interarrival_min_std": 0.0,
        "txn_interarrival_min_min": 0.0,
        "txn_interarrival_min_cv": 0.0,
    }
    if tx.empty:
        return base.assign(**empty_cols)

    payer = tx[[SRC_COL, TIME_COL, "txn_hour"]].rename(columns={SRC_COL: ID_COL})
    receiver = tx[[DST_COL, TIME_COL, "txn_hour"]].rename(columns={DST_COL: ID_COL})
    events = pd.concat([payer, receiver], ignore_index=True)

    hour_counts = events.groupby([ID_COL, "txn_hour"], sort=False).size().rename("hour_txn_count")
    hour_total = hour_counts.groupby(level=0).sum()
    hour_prob = hour_counts / hour_total.reindex(hour_counts.index.get_level_values(0)).to_numpy()
    hour_entropy = (-(hour_prob * np.log(hour_prob.clip(lower=1e-12)))).groupby(level=0).sum()
    hour_stats = hour_counts.groupby(level=0).agg(
        activity_hour_nunique="size",
        activity_hour_txn_max="max",
    )
    hour_stats["activity_hour_entropy"] = hour_entropy
    hour_stats["activity_hour_txn_max_ratio"] = safe_ratio(
        hour_stats["activity_hour_txn_max"],
        hour_total,
    )
    hour_stats = hour_stats.drop(columns=["activity_hour_txn_max"])

    events = events.sort_values([ID_COL, TIME_COL])
    delta_min = events.groupby(ID_COL)[TIME_COL].diff().dt.total_seconds().div(60)
    events = events.assign(delta_min=delta_min)
    inter_stats = events.groupby(ID_COL)["delta_min"].agg(
        txn_interarrival_min_mean="mean",
        txn_interarrival_min_std="std",
        txn_interarrival_min_min="min",
    )
    inter_stats["txn_interarrival_min_cv"] = safe_ratio(
        inter_stats["txn_interarrival_min_std"],
        inter_stats["txn_interarrival_min_mean"],
    )

    out = base.merge(hour_stats.reset_index(), on=ID_COL, how="left")
    out = out.merge(inter_stats.reset_index(), on=ID_COL, how="left")
    return out.fillna(0)


def counterparty_concentration_features(tx: pd.DataFrame, accounts: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL]].copy()
    empty_cols = {
        "counterparty_count": 0.0,
        "counterparty_amount_max": 0.0,
        "counterparty_amount_mean": 0.0,
        "counterparty_amount_top_ratio": 0.0,
        "counterparty_txn_max": 0.0,
        "counterparty_txn_mean": 0.0,
        "counterparty_txn_top_ratio": 0.0,
    }
    if tx.empty:
        return base.assign(**empty_cols)

    out_events = tx[[SRC_COL, DST_COL, "amount_abs"]].rename(
        columns={SRC_COL: ID_COL, DST_COL: "counterparty_id"}
    )
    in_events = tx[[DST_COL, SRC_COL, "amount_abs"]].rename(
        columns={DST_COL: ID_COL, SRC_COL: "counterparty_id"}
    )
    events = pd.concat([out_events, in_events], ignore_index=True)
    by_cp = events.groupby([ID_COL, "counterparty_id"], sort=False).agg(
        cp_txn_count=("amount_abs", "size"),
        cp_amount_sum=("amount_abs", "sum"),
    )
    total = by_cp.groupby(level=0).agg(
        total_cp_txn=("cp_txn_count", "sum"),
        total_cp_amount=("cp_amount_sum", "sum"),
    )
    stats = by_cp.groupby(level=0).agg(
        counterparty_count=("cp_txn_count", "size"),
        counterparty_amount_max=("cp_amount_sum", "max"),
        counterparty_amount_mean=("cp_amount_sum", "mean"),
        counterparty_txn_max=("cp_txn_count", "max"),
        counterparty_txn_mean=("cp_txn_count", "mean"),
    )
    stats = stats.merge(total, left_index=True, right_index=True, how="left")
    stats["counterparty_amount_top_ratio"] = safe_ratio(
        stats["counterparty_amount_max"],
        stats["total_cp_amount"],
    )
    stats["counterparty_txn_top_ratio"] = safe_ratio(
        stats["counterparty_txn_max"],
        stats["total_cp_txn"],
    )
    stats = stats.drop(columns=["total_cp_txn", "total_cp_amount"])
    return base.merge(stats.reset_index(), on=ID_COL, how="left").fillna(0)


def temporal_motif_features(
    tx: pd.DataFrame,
    accounts: pd.DataFrame,
    small_amount_threshold: float,
    large_amount_threshold: float,
) -> pd.DataFrame:
    horizons = [1, 6, 24]
    empty = {ID_COL: accounts[ID_COL].astype("int64").to_numpy()}
    for h in horizons:
        empty[f"fast_in_out_count_{h}h"] = 0.0
        empty[f"fast_in_out_in_amount_ratio_{h}h"] = 0.0
        empty[f"fast_in_out_out_amount_ratio_{h}h"] = 0.0
        empty[f"fast_in_out_balance_ratio_{h}h"] = 0.0
        empty[f"fast_in_out_mean_delay_min_{h}h"] = 0.0
        empty[f"prior_in_before_out_count_{h}h"] = 0.0
        empty[f"prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
        empty[f"prior_in_before_out_mean_delay_min_{h}h"] = 0.0
    empty.update(
        {
            "one_in_multi_out_count_24h": 0.0,
            "one_in_multi_out_in_amount_ratio_24h": 0.0,
            "multi_in_one_out_count_24h": 0.0,
            "multi_in_one_out_mean_amount_ratio_24h": 0.0,
            "max_prior_in_count_before_out_24h": 0.0,
            "max_next_out_count_after_in_24h": 0.0,
            "small_in_before_large_out_count_24h": 0.0,
        }
    )
    if tx.empty:
        return pd.DataFrame(empty)

    in_groups = {
        int(k): g[[TIME_COL, "amount_abs"]].sort_values(TIME_COL)
        for k, g in tx.groupby(DST_COL, sort=False)
    }
    out_groups = {
        int(k): g[[TIME_COL, "amount_abs"]].sort_values(TIME_COL)
        for k, g in tx.groupby(SRC_COL, sort=False)
    }

    rows = []
    for account_id in accounts[ID_COL].astype("int64"):
        account_id = int(account_id)
        ins = in_groups.get(account_id)
        outs = out_groups.get(account_id)
        row = {ID_COL: account_id}

        if ins is None:
            in_times = np.array([], dtype="datetime64[ns]")
            in_amounts = np.array([], dtype=float)
        else:
            in_times = ins[TIME_COL].to_numpy(dtype="datetime64[ns]")
            in_amounts = ins["amount_abs"].to_numpy(dtype=float)
        if outs is None:
            out_times = np.array([], dtype="datetime64[ns]")
            out_amounts = np.array([], dtype=float)
        else:
            out_times = outs[TIME_COL].to_numpy(dtype="datetime64[ns]")
            out_amounts = outs["amount_abs"].to_numpy(dtype=float)

        total_in_amount = float(in_amounts.sum())
        total_out_amount = float(out_amounts.sum())
        in_prefix = np.concatenate([[0.0], np.cumsum(in_amounts)])
        out_prefix = np.concatenate([[0.0], np.cumsum(out_amounts)])

        for h in horizons:
            if len(in_times) and len(out_times):
                delta = np.timedelta64(h, "h")
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
                row[f"fast_in_out_count_{h}h"] = int(matched.sum())
                row[f"fast_in_out_in_amount_ratio_{h}h"] = matched_amount / total_in_amount if total_in_amount else 0.0
                row[f"fast_in_out_out_amount_ratio_{h}h"] = (
                    matched_out_amount / total_out_amount if total_out_amount else 0.0
                )
                row[f"fast_in_out_balance_ratio_{h}h"] = balance_amount / total_in_amount if total_in_amount else 0.0
                row[f"fast_in_out_mean_delay_min_{h}h"] = float(np.mean(delays)) if len(delays) else 0.0
            else:
                row[f"fast_in_out_count_{h}h"] = 0
                row[f"fast_in_out_in_amount_ratio_{h}h"] = 0.0
                row[f"fast_in_out_out_amount_ratio_{h}h"] = 0.0
                row[f"fast_in_out_balance_ratio_{h}h"] = 0.0
                row[f"fast_in_out_mean_delay_min_{h}h"] = 0.0

            if len(in_times) and len(out_times):
                delta = np.timedelta64(h, "h")
                left = np.searchsorted(in_times, out_times - delta, side="left")
                right = np.searchsorted(in_times, out_times, side="left")
                matched = right > left
                if matched.any():
                    last_in_times = in_times[right[matched] - 1]
                    delays = (out_times[matched] - last_in_times).astype("timedelta64[s]").astype(float) / 60.0
                    matched_out_amount = float(out_amounts[matched].sum())
                    row[f"prior_in_before_out_count_{h}h"] = int(matched.sum())
                    row[f"prior_in_before_out_out_amount_ratio_{h}h"] = (
                        matched_out_amount / total_out_amount if total_out_amount else 0.0
                    )
                    row[f"prior_in_before_out_mean_delay_min_{h}h"] = float(np.mean(delays))
                else:
                    row[f"prior_in_before_out_count_{h}h"] = 0
                    row[f"prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
                    row[f"prior_in_before_out_mean_delay_min_{h}h"] = 0.0
            else:
                row[f"prior_in_before_out_count_{h}h"] = 0
                row[f"prior_in_before_out_out_amount_ratio_{h}h"] = 0.0
                row[f"prior_in_before_out_mean_delay_min_{h}h"] = 0.0

        one_in_multi_count = 0
        one_in_multi_amount = 0.0
        max_next_out_count = 0
        if len(in_times) and len(out_times):
            delta = np.timedelta64(24, "h")
            left = np.searchsorted(out_times, in_times, side="right")
            right = np.searchsorted(out_times, in_times + delta, side="right")
            next_counts = right - left
            max_next_out_count = int(next_counts.max()) if len(next_counts) else 0
            one_mask = next_counts >= 3
            one_in_multi_count = int(one_mask.sum())
            one_in_multi_amount = float(in_amounts[one_mask].sum())

        multi_in_one_count = 0
        multi_in_one_ratios = []
        max_prior_in_count = 0
        small_in_before_large_out_count = 0
        if len(in_times) and len(out_times):
            delta = np.timedelta64(24, "h")
            left = np.searchsorted(in_times, out_times - delta, side="left")
            right = np.searchsorted(in_times, out_times, side="left")
            prior_counts = right - left
            max_prior_in_count = int(prior_counts.max()) if len(prior_counts) else 0
            for i, (lft, rgt) in enumerate(zip(left, right)):
                if rgt <= lft:
                    continue
                prior_amount = float(in_prefix[rgt] - in_prefix[lft])
                ratio = out_amounts[i] / prior_amount if prior_amount else 0.0
                if prior_counts[i] >= 3 and 0.5 <= ratio <= 1.5:
                    multi_in_one_count += 1
                    multi_in_one_ratios.append(ratio)
                if out_amounts[i] >= large_amount_threshold and np.any(in_amounts[lft:rgt] <= small_amount_threshold):
                    small_in_before_large_out_count += 1

        row["one_in_multi_out_count_24h"] = one_in_multi_count
        row["one_in_multi_out_in_amount_ratio_24h"] = (
            one_in_multi_amount / total_in_amount if total_in_amount else 0.0
        )
        row["multi_in_one_out_count_24h"] = multi_in_one_count
        row["multi_in_one_out_mean_amount_ratio_24h"] = (
            float(np.mean(multi_in_one_ratios)) if multi_in_one_ratios else 0.0
        )
        row["max_prior_in_count_before_out_24h"] = max_prior_in_count
        row["max_next_out_count_after_in_24h"] = max_next_out_count
        row["small_in_before_large_out_count_24h"] = small_in_before_large_out_count
        rows.append(row)

    return pd.DataFrame(rows)


def build_stat_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    start: str,
    end: str,
    small_amount_threshold: float,
    large_amount_threshold: float,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    tx = transactions[(transactions[TIME_COL] >= start_ts) & (transactions[TIME_COL] <= end_ts)].copy()

    tx["is_night_txn"] = tx["txn_hour"].between(0, 5)
    tx["is_round_1000"] = tx["amount_abs"].mod(1000).eq(0)
    tx["is_small_amount"] = tx["amount_abs"].le(small_amount_threshold)
    tx["is_large_amount"] = tx["amount_abs"].ge(large_amount_threshold)

    features = accounts[[ID_COL, "account_age_months", "region_code", "customer_type"]].copy()
    features = features.merge(aggregate_side(tx, SRC_COL, "out", DST_COL), on=ID_COL, how="left")
    features = features.merge(aggregate_side(tx, DST_COL, "in", SRC_COL), on=ID_COL, how="left")
    features = features.merge(monthly_trend_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(daily_burst_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(activity_timing_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(counterparty_concentration_features(tx, accounts), on=ID_COL, how="left")
    features = features.merge(
        temporal_motif_features(tx, accounts, small_amount_threshold, large_amount_threshold),
        on=ID_COL,
        how="left",
    )

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0)
    features = features.copy()
    features["net_amount"] = features["in_amount_sum"] - features["out_amount_sum"]
    features["total_txn_count"] = features["in_txn_count"] + features["out_txn_count"]
    features["total_amount_sum"] = features["in_amount_sum"] + features["out_amount_sum"]
    features["in_out_amount_ratio"] = safe_ratio(features["in_amount_sum"], features["out_amount_sum"])
    features["in_out_count_ratio"] = safe_ratio(features["in_txn_count"], features["out_txn_count"])
    features["total_self_loop_count"] = features["out_self_loop_count"] + features["in_self_loop_count"]
    features["total_non_positive_amount_count"] = (
        features["out_non_positive_amount_count"] + features["in_non_positive_amount_count"]
    )
    return pd.get_dummies(features, columns=["customer_type"], dummy_na=False)


def main() -> None:
    ensure_dirs()
    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])

    train_start, train_end = SPLITS["train"]
    train_tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(train_start))
        & (transactions[TIME_COL] <= pd.Timestamp(train_end))
    ]
    small_amount_threshold = float(train_tx["amount_abs"].quantile(0.05))
    large_amount_threshold = float(train_tx["amount_abs"].quantile(0.95))

    report = {
        "small_amount_threshold_from_train_p05": small_amount_threshold,
        "large_amount_threshold_from_train_p95": large_amount_threshold,
        "splits": {},
    }
    for split, (start, end) in SPLITS.items():
        feat = build_stat_features(
            accounts,
            transactions,
            start,
            end,
            small_amount_threshold,
            large_amount_threshold,
        )
        path = FEATURE_DIR / f"stat_features_{split}.csv"
        feat.to_csv(path, index=False)
        report["splits"][split] = {
            "rows": int(len(feat)),
            "cols": int(feat.shape[1]),
            "path": str(path),
        }

    with (FEATURE_DIR / "stat_feature_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
