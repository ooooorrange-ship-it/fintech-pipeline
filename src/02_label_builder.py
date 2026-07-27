import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, ID_COL, LABEL_CODE_TO_TEXT, LABEL_DIR  # noqa: E402


def ensure_dirs() -> None:
    LABEL_DIR.mkdir(parents=True, exist_ok=True)


def build_labels(accounts: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    base = accounts[[ID_COL, "account_label_code"]].merge(
        labels[[ID_COL, "label_code", "label_type"]],
        on=ID_COL,
        how="inner",
    )
    if not base["account_label_code"].eq(base["label_code"]).all():
        raise ValueError("账户节点表和风险标签表的标签不一致，请先核验原始数据。")

    base["label_text"] = base["label_code"].map(LABEL_CODE_TO_TEXT)

    # A: 主任务。嫌疑人=1，其它=0，受害人=-1 表示训练/评估时剔除。
    base["label_A_suspect_vs_other"] = base["label_code"].map({1: 1, 0: 0, 2: -1}).astype("int8")

    # B: 业务辅助。嫌疑人+受害人=1，其它=0，用于识别涉诈关联账户。
    base["label_B_fraud_related_vs_other"] = base["label_code"].isin([1, 2]).astype("int8")

    # C: 三分类。0=其它，1=嫌疑人，2=受害人。
    base["label_C_three_class"] = base["label_code"].astype("int8")
    return base


def main() -> None:
    ensure_dirs()
    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    labels = pd.read_csv(CLEAN_DIR / "clean_labels.csv")
    out = build_labels(accounts, labels)

    out.to_csv(LABEL_DIR / "labels_all_strategies.csv", index=False)
    report = {
        "rows": int(len(out)),
        "label_code_distribution": out["label_code"].value_counts().sort_index().to_dict(),
        "strategy_A_distribution": out["label_A_suspect_vs_other"].value_counts().sort_index().to_dict(),
        "strategy_B_distribution": out["label_B_fraud_related_vs_other"].value_counts().sort_index().to_dict(),
        "strategy_C_distribution": out["label_C_three_class"].value_counts().sort_index().to_dict(),
        "note": "strategy_A 中 -1 为受害人，主任务训练和评估默认剔除。",
    }
    with (LABEL_DIR / "label_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

