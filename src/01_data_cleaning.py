import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    AMOUNT_COL,
    CLEAN_DIR,
    DST_COL,
    ID_COL,
    LABEL_TEXT_TO_CODE,
    RAW_ACCOUNTS_PATH,
    RAW_LABELS_PATH,
    RAW_TRANSACTIONS_PATH,
    SRC_COL,
    TIME_COL,
)


def ensure_dirs() -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def read_raw_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    accounts = pd.read_excel(RAW_ACCOUNTS_PATH)
    transactions = pd.read_excel(RAW_TRANSACTIONS_PATH)
    labels = pd.read_excel(RAW_LABELS_PATH)
    return accounts, transactions, labels


def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    accounts = accounts.rename(
        columns={
            "账户脱敏id": ID_COL,
            "是否风险账户标签": "account_label_code",
            "开户时长": "account_age_months",
            "地区编码": "region_code",
            "客户类型": "customer_type",
        }
    )
    accounts[ID_COL] = accounts[ID_COL].astype("int64")
    accounts["account_label_code"] = accounts["account_label_code"].astype("int64")
    accounts["account_age_months"] = pd.to_numeric(accounts["account_age_months"], errors="coerce")
    accounts["region_code"] = accounts["region_code"].astype("int64")
    accounts["customer_type"] = accounts["customer_type"].astype("string").fillna("unknown")
    return accounts.drop_duplicates(subset=[ID_COL]).reset_index(drop=True)


def clean_labels(labels: pd.DataFrame) -> pd.DataFrame:
    labels = labels.rename(columns={"账户脱敏id": ID_COL, "标签类型": "label_type"})
    labels[ID_COL] = labels[ID_COL].astype("int64")
    labels["label_type"] = labels["label_type"].astype("string").str.strip()
    labels["label_code"] = labels["label_type"].map(LABEL_TEXT_TO_CODE).astype("int64")
    return labels.drop_duplicates(subset=[ID_COL]).reset_index(drop=True)


def clean_transactions(transactions: pd.DataFrame, valid_account_ids: set[int]) -> pd.DataFrame:
    transactions = transactions.rename(
        columns={
            "付款账户脱敏id": SRC_COL,
            "收款账户脱敏id": DST_COL,
            "交易时间": TIME_COL,
            "金额": AMOUNT_COL,
        }
    )
    transactions[SRC_COL] = transactions[SRC_COL].astype("int64")
    transactions[DST_COL] = transactions[DST_COL].astype("int64")
    transactions[AMOUNT_COL] = pd.to_numeric(transactions[AMOUNT_COL], errors="coerce")
    transactions[TIME_COL] = pd.to_datetime(
        transactions[TIME_COL].astype("string").str.strip(),
        format="%Y%m%d %H:%M:%S",
        errors="coerce",
    )

    transactions["missing_src_account"] = ~transactions[SRC_COL].isin(valid_account_ids)
    transactions["missing_dst_account"] = ~transactions[DST_COL].isin(valid_account_ids)
    transactions["invalid_time"] = transactions[TIME_COL].isna()
    transactions["invalid_amount"] = transactions[AMOUNT_COL].isna()
    transactions["self_loop"] = transactions[SRC_COL].eq(transactions[DST_COL])
    transactions["non_positive_amount"] = transactions[AMOUNT_COL].le(0)
    transactions["negative_amount"] = transactions[AMOUNT_COL].lt(0)
    transactions["zero_amount"] = transactions[AMOUNT_COL].eq(0)
    transactions["amount_abs"] = transactions[AMOUNT_COL].abs()

    # 这里不删除自环和非正金额。它们可能是反诈强信号，只做标记。
    hard_invalid = (
        transactions["missing_src_account"]
        | transactions["missing_dst_account"]
        | transactions["invalid_time"]
        | transactions["invalid_amount"]
    )
    transactions = transactions.loc[~hard_invalid].copy()
    transactions["txn_date"] = transactions[TIME_COL].dt.date.astype("string")
    transactions["txn_month"] = transactions[TIME_COL].dt.to_period("M").astype("string")
    transactions["txn_hour"] = transactions[TIME_COL].dt.hour.astype("int16")
    return transactions.reset_index(drop=True)


def build_report(
    raw_accounts: pd.DataFrame,
    raw_transactions: pd.DataFrame,
    raw_labels: pd.DataFrame,
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    labels: pd.DataFrame,
) -> dict:
    merged = accounts[[ID_COL, "account_label_code"]].merge(
        labels[[ID_COL, "label_code"]],
        on=ID_COL,
        how="outer",
        indicator=True,
    )
    report = {
        "raw_rows": {
            "accounts": int(len(raw_accounts)),
            "transactions": int(len(raw_transactions)),
            "labels": int(len(raw_labels)),
        },
        "clean_rows": {
            "accounts": int(len(accounts)),
            "transactions": int(len(transactions)),
            "labels": int(len(labels)),
        },
        "label_distribution_account_table": accounts["account_label_code"].value_counts(dropna=False).sort_index().to_dict(),
        "label_distribution_label_table": labels["label_code"].value_counts(dropna=False).sort_index().to_dict(),
        "account_label_merge": merged["_merge"].value_counts().to_dict(),
        "account_label_mismatch_count": int((merged["account_label_code"] != merged["label_code"]).sum()),
        "transaction_flags": {
            "self_loop": int(transactions["self_loop"].sum()),
            "non_positive_amount": int(transactions["non_positive_amount"].sum()),
            "negative_amount": int(transactions["negative_amount"].sum()),
            "zero_amount": int(transactions["zero_amount"].sum()),
        },
        "time_range": {
            "min": str(transactions[TIME_COL].min()),
            "max": str(transactions[TIME_COL].max()),
        },
    }
    return report


def main() -> None:
    ensure_dirs()
    raw_accounts, raw_transactions, raw_labels = read_raw_tables()
    accounts = clean_accounts(raw_accounts)
    labels = clean_labels(raw_labels)
    transactions = clean_transactions(raw_transactions, set(accounts[ID_COL]))

    report = build_report(raw_accounts, raw_transactions, raw_labels, accounts, transactions, labels)

    accounts.to_csv(CLEAN_DIR / "clean_accounts.csv", index=False)
    transactions.to_csv(CLEAN_DIR / "clean_transactions.csv", index=False)
    labels.to_csv(CLEAN_DIR / "clean_labels.csv", index=False)
    with (CLEAN_DIR / "cleaning_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

