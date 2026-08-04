"""原始数据交易边覆盖审计。

核验 59 个确认嫌疑账户中 56 个缺边是原始数据事实，而不是清洗、
ID 类型或时间窗口过滤导致，并记录缺边账户的画像依据。
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    DELIVERABLE_DIR,
    DOCS_DIR,
    DST_COL,
    ID_COL,
    OUTPUT_DIR,
    RAW_ACCOUNTS_PATH,
    RAW_LABELS_PATH,
    RAW_TRANSACTIONS_PATH,
    SRC_COL,
)


def quantile_map(series: pd.Series) -> dict[str, float]:
    return {str(q): float(series.quantile(q)) for q in [0.25, 0.5, 0.75, 0.9]}


def main() -> None:
    DELIVERABLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    accounts = pd.read_excel(RAW_ACCOUNTS_PATH)
    transactions = pd.read_excel(RAW_TRANSACTIONS_PATH)
    labels = pd.read_excel(RAW_LABELS_PATH)

    accounts = accounts.rename(
        columns={
            "账户脱敏id": ID_COL,
            "是否风险账户标签": "label_code",
            "开户时长": "account_age_months",
            "地区编码": "region_code",
            "客户类型": "customer_type",
        }
    )
    transactions = transactions.rename(
        columns={
            "付款账户脱敏id": SRC_COL,
            "收款账户脱敏id": DST_COL,
            "交易时间": "time_raw",
            "金额": "amount_raw",
        }
    )
    labels = labels.rename(columns={"账户脱敏id": ID_COL, "标签类型": "label_type"})

    accounts[ID_COL] = accounts[ID_COL].astype(int)
    accounts["label_code"] = accounts["label_code"].astype(int)
    transactions[SRC_COL] = transactions[SRC_COL].astype(int)
    transactions[DST_COL] = transactions[DST_COL].astype(int)
    labels[ID_COL] = labels[ID_COL].astype(int)

    endpoints = set(transactions[SRC_COL]).union(set(transactions[DST_COL]))
    suspects = set(accounts.loc[accounts["label_code"].eq(1), ID_COL])
    victims = set(accounts.loc[accounts["label_code"].eq(2), ID_COL])
    with_edge_suspects = sorted(suspects & endpoints)
    without_edge_suspects = sorted(suspects - endpoints)

    missing_profile = accounts[accounts[ID_COL].isin(without_edge_suspects)]
    profile_summary = {
        "customer_type_counts": missing_profile["customer_type"].value_counts().to_dict(),
        "account_age_months_quantiles": quantile_map(missing_profile["account_age_months"]),
        "region_count": int(missing_profile["region_code"].nunique()),
        "region_top_counts": missing_profile["region_code"].value_counts().head(10).to_dict(),
    }

    suspect_tx = transactions[
        transactions[SRC_COL].isin(with_edge_suspects)
        | transactions[DST_COL].isin(with_edge_suspects)
    ].copy()
    suspect_tx["month"] = suspect_tx["time_raw"].astype(str).str[:6]
    with_edge_summary = {
        "transaction_count": int(len(suspect_tx)),
        "month_counts": suspect_tx["month"].value_counts().sort_index().to_dict(),
        "as_src_count": int(suspect_tx[SRC_COL].isin(with_edge_suspects).sum()),
        "as_dst_count": int(suspect_tx[DST_COL].isin(with_edge_suspects).sum()),
    }

    label_distribution = accounts["label_code"].value_counts().sort_index().to_dict()
    coverage = {}
    for code, name in [(0, "其它"), (1, "嫌疑人"), (2, "受害人")]:
        ids = set(accounts.loc[accounts["label_code"].eq(code), ID_COL])
        coverage[name] = {
            "total": int(len(ids)),
            "with_edge": int(len(ids & endpoints)),
            "without_edge": int(len(ids - endpoints)),
        }

    audit = {
        "audit_version": "v1",
        "source_files": {
            "accounts": str(RAW_ACCOUNTS_PATH),
            "transactions": str(RAW_TRANSACTIONS_PATH),
            "labels": str(RAW_LABELS_PATH),
        },
        "raw_rows": {
            "accounts": int(len(accounts)),
            "transactions": int(len(transactions)),
            "labels": int(len(labels)),
        },
        "transaction_time_range": {
            "min": str(transactions["time_raw"].min()),
            "max": str(transactions["time_raw"].max()),
        },
        "transaction_month_counts": transactions["time_raw"].astype(str).str[:6].value_counts().sort_index().to_dict(),
        "transaction_endpoint_account_count": int(len(endpoints)),
        "account_label_distribution": label_distribution,
        "endpoint_coverage_by_label": coverage,
        "confirmed_suspect": {
            "total": int(len(suspects)),
            "with_edge_count": int(len(with_edge_suspects)),
            "with_edge_ids": with_edge_suspects,
            "without_edge_count": int(len(without_edge_suspects)),
            "without_edge_ids": without_edge_suspects,
            "with_edge_transaction_summary": with_edge_summary,
            "without_edge_profile_summary": profile_summary,
        },
        "conclusion": {
            "missing_edge_is_raw_data_fact": True,
            "cannot_generate_real_links": True,
            "reason": (
                "交易边表去重端点为 7795 个账户，时间覆盖 2025-07 至 2025-12；"
                "59 个确认嫌疑账户中只有 3 个（1740、4379、7265）出现在端点集合，"
                "56 个确认嫌疑账户不是任何一条交易边的付款方或收款方。"
                "该缺边是原始抽样数据本身的事实，与清洗规则、ID 类型或时间窗口过滤无关。"
            ),
            "risk_source_note": (
                "56 个缺边账户在最终主模型测试平均风险分为 0.9847，但交易统计、图结构、"
                "动态资金图谱特征均为 0；其高分主要来自账户画像与图基座特征，属于画像层研判，"
                "不是资金行为链路证据。账户级五折留出验证 Top5% 召回约 44%，"
                "说明这部分识别应作为补数复核线索，不能宣称已生成真实资金链路。"
            ),
        },
    }

    json_path = DELIVERABLE_DIR / "data_edge_coverage_audit.json"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    md = [
        "# 原始数据交易边覆盖审计",
        "",
        "## 结论",
        "",
        "59 个确认嫌疑账户中 56 个缺边是原始数据事实：它们不是交易边表任何一条边的付款方或收款方。",
        "该缺边与清洗规则、ID 类型、时间窗口过滤无关，不能通过路径算法恢复真实资金链路；",
        "强行生成路径属于伪造解释。",
        "",
        "## 原始数据规模",
        "",
        f"- 账户节点表：{len(accounts)} 行",
        f"- 交易边表：{len(transactions)} 行，时间范围 {transactions['time_raw'].min()} 至 {transactions['time_raw'].max()}",
        f"- 风险标签表：{len(labels)} 行",
        f"- 交易边表去重端点账户：{len(endpoints)} / {len(accounts)}",
        "",
        "## 各标签账户的边覆盖",
        "",
        "| 标签 | 总账户数 | 有交易边 | 无交易边 |",
        "|---|---:|---:|---:|",
    ]
    for name in ["其它", "嫌疑人", "受害人"]:
        item = coverage[name]
        md.append(f"| {name} | {item['total']} | {item['with_edge']} | {item['without_edge']} |")
    md.extend(
        [
            "",
            "## 确认嫌疑账户",
            "",
            f"- 总数：{len(suspects)}",
            f"- 有边：{len(with_edge_suspects)}（{', '.join(map(str, with_edge_suspects))}）",
            f"- 缺边：{len(without_edge_suspects)}",
            f"- 有边账户交易：{with_edge_summary['transaction_count']} 笔，按月份 {with_edge_summary['month_counts']}",
            "",
            "## 缺边账户画像概览",
            "",
            f"- 客户类型：{profile_summary['customer_type_counts']}",
            f"- 开户时长分位：{profile_summary['account_age_months_quantiles']}",
            f"- 覆盖地区数：{profile_summary['region_count']}",
            f"- 地区分布 Top10：{profile_summary['region_top_counts']}",
            "",
            "## 没有交易边如何确认风险",
            "",
            "确认风险标签由账户节点表和风险标签表直接提供；模型对缺边账户的高分主要来自账户画像与图基座特征，",
            "而不是资金行为。账户级五折留出验证 Top5% 召回约 44%，说明这部分识别应作为补数复核线索，",
            "不能与真实资金链路证据混为一谈。",
            "",
            "## 可复现命令",
            "",
            "```bash",
            "python src/26_data_edge_coverage_audit.py",
            "```",
            "",
        ]
    )
    md_path = DOCS_DIR / "data_edge_coverage_audit.md"
    md_path.write_text("\n".join(md), encoding="utf-8")

    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
