import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
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


LAYERED_DIR = EXPLANATION_DIR / "layered"


def ensure_dirs() -> None:
    LAYERED_DIR.mkdir(parents=True, exist_ok=True)


def load_explain_module():
    spec = importlib.util.spec_from_file_location("explain_links", Path(__file__).resolve().parent / "07_explain_links.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def find_prediction_file() -> Path:
    preferred = [
        PREDICTION_DIR / "model4_stack_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
        PREDICTION_DIR / "model5_xgb_dynamic_graph_v6_rolling_memory_dynamic_no_customer_type_strategy_A.csv",
        PREDICTION_DIR / "model2_xgb_stat_graph_v2_no_customer_type_strategy_A.csv",
    ]
    for path in preferred:
        if path.exists():
            return path
    candidates = sorted(PREDICTION_DIR.glob("*no_customer_type_strategy_A.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if candidates:
        return candidates[0]
    raise FileNotFoundError("未找到 no_customer_type strategy_A 预测文件。")


def scoped_transactions(transactions: pd.DataFrame, split: str, scope: str) -> pd.DataFrame:
    if scope == "split":
        start, end = SPLITS[split]
        return transactions[
            (transactions[TIME_COL] >= pd.Timestamp(start))
            & (transactions[TIME_COL] <= pd.Timestamp(end))
        ].copy()
    if scope == "history":
        _, end = SPLITS[split]
        return transactions[transactions[TIME_COL].le(pd.Timestamp(end))].copy()
    return transactions.copy()


def load_feature_frame(split: str) -> pd.DataFrame:
    frames = []
    for name in ["stat", "graph", "dynamic_graph"]:
        path = FEATURE_DIR / f"{name}_features_{split}.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=ID_COL, how="left")
    return out.fillna(0)


def direct_event_stats(tx: pd.DataFrame, root_id: int) -> dict:
    direct = tx[tx[SRC_COL].eq(root_id) | tx[DST_COL].eq(root_id)].copy()
    if direct.empty:
        return {
            "history_txn_count": 0,
            "history_amount_sum": 0.0,
            "direct_counterparty_count": 0,
            "first_txn_time": "",
            "last_txn_time": "",
        }
    counterparties = set(direct.loc[direct[SRC_COL].eq(root_id), DST_COL].astype(int))
    counterparties |= set(direct.loc[direct[DST_COL].eq(root_id), SRC_COL].astype(int))
    return {
        "history_txn_count": int(len(direct)),
        "history_amount_sum": float(direct["amount_abs"].sum()),
        "direct_counterparty_count": int(len(counterparties)),
        "first_txn_time": str(direct[TIME_COL].min()),
        "last_txn_time": str(direct[TIME_COL].max()),
    }


def build_structure_rows(tx: pd.DataFrame, root_id: int) -> list[dict]:
    rows = []
    incoming = tx.loc[tx[DST_COL].eq(root_id), [SRC_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    outgoing = tx.loc[tx[SRC_COL].eq(root_id), [DST_COL, TIME_COL, "amount_abs"]].sort_values(TIME_COL)
    horizon = pd.Timedelta(hours=24)

    for out_rec in outgoing.to_dict("records"):
        t_out = pd.Timestamp(out_rec[TIME_COL])
        prior_in = incoming[incoming[TIME_COL].between(t_out - horizon, t_out, inclusive="left")]
        if len(prior_in) >= 2:
            rows.append(
                {
                    "root_account_id": root_id,
                    "structure_type": "multi_in_one_out_24h",
                    "anchor_time": str(t_out),
                    "source_count": int(prior_in[SRC_COL].nunique()),
                    "destination_count": 1,
                    "in_txn_count": int(len(prior_in)),
                    "out_txn_count": 1,
                    "in_amount_sum": float(prior_in["amount_abs"].sum()),
                    "out_amount_sum": float(out_rec["amount_abs"]),
                    "counterparty_examples": ",".join(map(str, prior_in[SRC_COL].astype(int).drop_duplicates().head(5))),
                    "business_meaning": "多个账户短时间汇入后由根账户集中转出。",
                }
            )
        mid = int(out_rec[DST_COL])
        loop_back = incoming[
            incoming[SRC_COL].eq(mid)
            & incoming[TIME_COL].between(t_out, t_out + horizon, inclusive="right")
        ]
        for in_back in loop_back.head(5).to_dict("records"):
            delay_sec = (pd.Timestamp(in_back[TIME_COL]) - t_out).total_seconds()
            rows.append(
                {
                    "root_account_id": root_id,
                    "structure_type": "closed_loop_return_24h",
                    "anchor_time": str(t_out),
                    "source_count": 1,
                    "destination_count": 1,
                    "in_txn_count": 1,
                    "out_txn_count": 1,
                    "in_amount_sum": float(in_back["amount_abs"]),
                    "out_amount_sum": float(out_rec["amount_abs"]),
                    "counterparty_examples": str(mid),
                    "delay_seconds": float(delay_sec),
                    "business_meaning": "根账户转出后短时间从同一对手回流，属于强时序闭环。",
                }
            )

    for in_rec in incoming.to_dict("records"):
        t_in = pd.Timestamp(in_rec[TIME_COL])
        next_out = outgoing[outgoing[TIME_COL].between(t_in, t_in + horizon, inclusive="right")]
        if len(next_out) >= 2:
            rows.append(
                {
                    "root_account_id": root_id,
                    "structure_type": "one_in_multi_out_24h",
                    "anchor_time": str(t_in),
                    "source_count": 1,
                    "destination_count": int(next_out[DST_COL].nunique()),
                    "in_txn_count": 1,
                    "out_txn_count": int(len(next_out)),
                    "in_amount_sum": float(in_rec["amount_abs"]),
                    "out_amount_sum": float(next_out["amount_abs"].sum()),
                    "counterparty_examples": ",".join(map(str, next_out[DST_COL].astype(int).drop_duplicates().head(5))),
                    "business_meaning": "根账户入账后短时间拆分转给多个账户。",
                }
            )
    return rows


def feature_evidence_for_account(features: pd.DataFrame, account_id: int, event_stats: dict | None = None) -> tuple[str, int]:
    row = features[features[ID_COL].eq(account_id)]
    event_stats = event_stats or {}
    event_pieces = []
    if event_stats.get("history_txn_count", 0):
        event_pieces.extend(
            [
                f"历史交易数={int(event_stats.get('history_txn_count', 0))}",
                f"历史交易金额={float(event_stats.get('history_amount_sum', 0.0)):.4g}",
                f"直接交易对手={int(event_stats.get('direct_counterparty_count', 0))}",
                f"交易时间跨度={event_stats.get('first_txn_time', '')}至{event_stats.get('last_txn_time', '')}",
            ]
        )
    if row.empty:
        if event_pieces:
            return "；".join(event_pieces), len(event_pieces)
        return "未找到特征宽表记录", 0
    row = row.iloc[0]
    priority_cols = [
        "total_txn_count",
        "total_amount_sum",
        "counterparty_amount_top_ratio",
        "burst_day_txn_ratio",
        "fast_in_out_balance_ratio_24h",
        "prior_in_before_out_out_amount_ratio_24h",
        "multi_in_one_out_count_24h",
        "one_in_multi_out_count_24h",
        "graph_total_degree",
        "graph_two_hop_neighbor_count",
        "graph_reciprocal_neighbor_count",
        "dyn_total_active_week_count",
        "dyn_total_week_txn_burst_ratio",
        "dyn_cp_new_second_half_ratio",
        "dyn_motif_fast_in_out_count_24h",
        "dyn_motif_multi_in_one_out_count_24h",
        "dyn_motif_one_in_multi_out_count_24h",
        "dyn_mem_total_count_7d",
        "dyn_mem_last_event_age_hours",
    ]
    pieces = []
    nonzero = 0
    for col in priority_cols:
        if col not in features.columns:
            continue
        value = row[col]
        if isinstance(value, (bool, np.bool_)):
            value = int(value)
        if pd.api.types.is_number(value) and float(value) != 0.0:
            nonzero += 1
            series = features[col]
            pct = float((series <= value).mean())
            pieces.append(f"{col}={float(value):.4g}(分位{pct:.1%})")
        if len(pieces) >= 6:
            break
    if event_pieces:
        pieces = event_pieces[:4] + pieces[:3]
    if pieces:
        return "；".join(pieces), nonzero
    return "交易统计、图结构、动态资金图谱特征均为0，说明当前交易边表没有覆盖该账户行为。", 0


def explanation_grade(has_path: bool, has_assoc: bool, feature_nonzero: int, history_txn_count: int) -> tuple[str, str]:
    if has_path:
        return "A", "链路证据型：存在多跳可疑路径或资金流结构，可直接画资金链路。"
    if has_assoc:
        return "B", "直接关联型：存在直接交易对手，可输出 Top20 关联账户。"
    if feature_nonzero > 0:
        return "C", "动态特征型：无显式路径，但有账户级交易/图谱异常特征。"
    if history_txn_count == 0:
        return "D", "数据缺边型：交易边表没有该账户历史交易，不能生成可信资金链路。"
    return "C", "账户行为型：有交易记录但未命中当前路径规则。"


def build_markdown_audit(audit_df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# 59 个确认嫌疑账户解释覆盖审计",
        "",
        "## 结论",
        "",
        "当前 59 个确认嫌疑账户中，只有存在历史交易边的账户才能生成真实资金链路。对没有交易边的账户，本项目不伪造路径，而是输出数据缺边原因、模型分数、特征证据和后续补数建议。",
        "",
        "## 分层解释口径",
        "",
        "| 等级 | 含义 | 可用于答辩的表述 |",
        "|---|---|---|",
        "| A | 链路证据型 | 可展示多跳路径、闭环、汇聚/分散结构 |",
        "| B | 直接关联型 | 可展示 Top20 交易对手和资金边 |",
        "| C | 动态特征型 | 可展示交易统计、图结构、动态时间窗口特征 |",
        "| D | 数据缺边型 | 当前交易边表未覆盖该账户，不生成链路，只做缺口审计 |",
        "",
        "## 覆盖统计",
        "",
    ]
    grade_counts = audit_df["explanation_grade"].value_counts().sort_index()
    lines.append("| 等级 | 账户数 |")
    lines.append("|---|---:|")
    for grade, count in grade_counts.items():
        lines.append(f"| {grade} | {int(count)} |")
    lines.extend(["", "## 账户明细", "", "| 账户 | 分数 | 排名 | 等级 | 历史交易数 | 对手数 | 证据摘要 |"])
    lines.append("|---|---:|---:|---|---:|---:|---|")
    for row in audit_df.sort_values(["explanation_grade", "score"], ascending=[True, False]).itertuples(index=False):
        lines.append(
            f"| {int(row.account_id)} | {float(row.score):.4f} | {int(row.risk_rank)} | {row.explanation_grade} | "
            f"{int(row.history_txn_count)} | {int(row.direct_counterparty_count)} | {row.short_evidence} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_judgement_reports(
    queue: pd.DataFrame,
    audit: pd.DataFrame,
    structures: pd.DataFrame,
    paths: pd.DataFrame,
    associations: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# 分层辅助研判报告样例",
        "",
        "说明：报告只引用数据中真实存在的交易边和特征证据。若账户属于数据缺边型，结论限定为“需补充流水后复核”，不生成虚构链路。",
        "",
    ]
    cases = pd.concat(
        [
            audit[audit["explanation_grade"].isin(["A", "B"])].sort_values("score", ascending=False).head(3),
            queue.sort_values("score", ascending=False).head(5),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=[ID_COL]).head(8)

    for idx, row in enumerate(cases.itertuples(index=False), start=1):
        account_structures = structures[structures["root_account_id"].eq(int(row.account_id))] if not structures.empty else pd.DataFrame()
        account_paths = paths[paths["root_account_id"].eq(int(row.account_id))] if not paths.empty else pd.DataFrame()
        account_assocs = associations[associations["root_account_id"].eq(int(row.account_id))] if not associations.empty else pd.DataFrame()
        lines.extend(
            [
                f"## 案例 {idx}：账户 {int(row.account_id)}",
                "",
                f"- 模型风险分：{float(row.score):.4f}",
                f"- 测试集排序：第 {int(row.risk_rank)} 名",
                f"- 标签：{row.label_text}",
                f"- 解释等级：{row.explanation_grade}，{row.explanation_reason}",
                f"- 交易证据：历史交易 {int(row.history_txn_count)} 笔，直接对手 {int(row.direct_counterparty_count)} 个。",
                f"- 特征证据：{row.feature_evidence}",
            ]
        )
        if not account_structures.empty:
            best = account_structures.iloc[0]
            lines.append(
                f"- 链路结构：{best['structure_type']}，锚定时间 {best['anchor_time']}，对手示例 {best['counterparty_examples']}。"
            )
        elif not account_paths.empty:
            best_path = account_paths.sort_values("path_evidence_score", ascending=False).iloc[0]
            lines.append(
                "- 可疑路径："
                f"{int(best_path['account_1'])}({best_path['label_1']}) -> "
                f"{int(best_path['account_2'])}({best_path['label_2']}) -> "
                f"{int(best_path['account_3'])}({best_path['label_3']})，"
                f"间隔 {float(best_path['delay_hours']):.4f} 小时，"
                f"金额比 {float(best_path['amount_ratio']):.4f}。"
            )
        elif not account_assocs.empty:
            top_assoc = account_assocs.sort_values("association_score", ascending=False).head(3)
            assoc_text = "；".join(
                f"{int(x.counterparty_id)}({x.label_text}, {int(x.direct_txn_count)}笔)"
                for x in top_assoc.itertuples(index=False)
            )
            lines.append(f"- 关联账户：{assoc_text}。")
        else:
            lines.append("- 链路结构：当前未观测到可追溯多跳路径。")
        if row.explanation_grade == "D":
            lines.append("- 处置建议：补充该账户完整历史流水、设备/IP、开户资料和处置时间后再复核。")
        else:
            lines.append("- 处置建议：进入人工复核队列，优先核验上下游账户、时间间隔和金额比例。")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成分层解释审计、风险巡检队列和辅助研判报告。")
    parser.add_argument("--prediction-file", default="", help="默认选择 v6 动态融合模型。")
    parser.add_argument("--split", default="test", choices=sorted(SPLITS), help="解释哪个 split。")
    parser.add_argument("--tx-scope", default="history", choices=["split", "history", "full"], help="交易解释范围。")
    parser.add_argument("--top-risk-active", type=int, default=30, help="输出多少个有交易边的模型高风险巡检账户。")
    parser.add_argument("--top-k-counterparties", type=int, default=20, help="每个账户输出 Top K 关联账户。")
    parser.add_argument("--max-paths", type=int, default=50, help="每个账户最多输出多少条可疑路径。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    explain = load_explain_module()

    pred_path = Path(args.prediction_file) if args.prediction_file else find_prediction_file()
    predictions = pd.read_csv(pred_path)
    predictions = predictions[predictions["split"].eq(args.split)].copy()
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    tx = scoped_transactions(transactions, args.split, args.tx_scope)
    features = load_feature_frame(args.split)

    predictions = predictions.merge(labels[[ID_COL, "label_code", "label_text"]], on=ID_COL, how="left")
    predictions = predictions.sort_values("score", ascending=False).reset_index(drop=True)
    predictions["risk_rank"] = np.arange(1, len(predictions) + 1)
    score_map = dict(zip(predictions[ID_COL].astype(int), predictions["score"].astype(float)))
    active_ids = set(tx[SRC_COL].astype(int)).union(set(tx[DST_COL].astype(int)))

    suspect_ids = labels[labels["label_code"].eq(1)][ID_COL].astype(int).tolist()
    all_associations = []
    all_paths = []
    all_structures = []
    audit_rows = []

    root_ids = set(suspect_ids)
    active_queue = predictions[predictions[ID_COL].astype(int).isin(active_ids)].head(args.top_risk_active)
    root_ids.update(active_queue[ID_COL].astype(int).tolist())

    for account_id in sorted(root_ids):
        assoc = explain.counterparty_summary(tx, account_id, labels, score_map, args.top_k_counterparties)
        paths = explain.suspicious_paths(tx, account_id, labels, args.max_paths)
        structures = pd.DataFrame(build_structure_rows(tx, account_id))
        if not assoc.empty:
            all_associations.append(assoc)
        if not paths.empty:
            all_paths.append(paths)
        if not structures.empty:
            all_structures.append(structures)

    associations = pd.concat(all_associations, ignore_index=True) if all_associations else pd.DataFrame()
    paths = pd.concat(all_paths, ignore_index=True) if all_paths else pd.DataFrame()
    structures = pd.concat(all_structures, ignore_index=True) if all_structures else pd.DataFrame()

    for account_id in suspect_ids:
        pred_row = predictions[predictions[ID_COL].eq(account_id)]
        if pred_row.empty:
            continue
        pred_row = pred_row.iloc[0]
        assoc = associations[associations["root_account_id"].eq(account_id)] if not associations.empty else pd.DataFrame()
        account_paths = paths[paths["root_account_id"].eq(account_id)] if not paths.empty else pd.DataFrame()
        account_structures = structures[structures["root_account_id"].eq(account_id)] if not structures.empty else pd.DataFrame()
        stats = direct_event_stats(tx, account_id)
        feature_text, feature_nonzero = feature_evidence_for_account(features, account_id, stats)
        has_path = (not account_paths.empty) or (not account_structures.empty)
        has_assoc = not assoc.empty
        grade, reason = explanation_grade(has_path, has_assoc, feature_nonzero, stats["history_txn_count"])
        if has_path:
            short = "存在可追溯资金链路或资金结构。"
        elif has_assoc:
            short = "存在直接交易对手，可进行上下游核验。"
        elif feature_nonzero:
            short = "存在账户级动态交易/图谱异常特征。"
        else:
            short = "交易边表未覆盖该账户，无法生成可信链路。"
        audit_rows.append(
            {
                ID_COL: int(account_id),
                "score": float(pred_row["score"]),
                "risk_rank": int(pred_row["risk_rank"]),
                "label_text": pred_row["label_text"],
                "explanation_grade": grade,
                "explanation_reason": reason,
                "has_direct_association": bool(has_assoc),
                "has_suspicious_path_or_structure": bool(has_path),
                "feature_nonzero_count": int(feature_nonzero),
                "feature_evidence": feature_text,
                "short_evidence": short,
                **stats,
                "recommended_next_step": (
                    "补充该账户完整交易流水或扩大子图抽样范围。"
                    if grade == "D"
                    else "进入人工复核队列，核验上下游账户和关键交易时间。"
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows).sort_values(["explanation_grade", "score"], ascending=[True, False])
    audit_path = LAYERED_DIR / "confirmed_suspect_explainability_audit.csv"
    audit_md_path = LAYERED_DIR / "confirmed_suspect_explainability_audit.md"
    audit_df.to_csv(audit_path, index=False)
    build_markdown_audit(audit_df, audit_md_path)

    queue_rows = []
    for row in active_queue.itertuples(index=False):
        account_id = int(row.account_id)
        assoc = associations[associations["root_account_id"].eq(account_id)] if not associations.empty else pd.DataFrame()
        account_paths = paths[paths["root_account_id"].eq(account_id)] if not paths.empty else pd.DataFrame()
        account_structures = structures[structures["root_account_id"].eq(account_id)] if not structures.empty else pd.DataFrame()
        stats = direct_event_stats(tx, account_id)
        feature_text, feature_nonzero = feature_evidence_for_account(features, account_id, stats)
        has_path = (not account_paths.empty) or (not account_structures.empty)
        has_assoc = not assoc.empty
        grade, reason = explanation_grade(has_path, has_assoc, feature_nonzero, stats["history_txn_count"])
        queue_rows.append(
            {
                ID_COL: account_id,
                "score": float(row.score),
                "risk_rank": int(row.risk_rank),
                "label_text": row.label_text,
                "explanation_grade": grade,
                "explanation_reason": reason,
                "feature_evidence": feature_text,
                **stats,
            }
        )
    queue_df = pd.DataFrame(queue_rows)
    queue_df.to_csv(LAYERED_DIR / "risk_review_queue_active_accounts.csv", index=False)
    associations.to_csv(LAYERED_DIR / "risk_review_queue_top20_associations.csv", index=False)
    paths.to_csv(LAYERED_DIR / "risk_review_queue_suspicious_paths.csv", index=False)
    structures.to_csv(LAYERED_DIR / "risk_review_queue_fund_flow_structures.csv", index=False)
    build_judgement_reports(
        queue_df,
        audit_df,
        structures,
        paths,
        associations,
        LAYERED_DIR / "layered_judgement_report_samples.md",
    )

    coverage = {
        "prediction_file": str(pred_path),
        "split": args.split,
        "tx_scope": args.tx_scope,
        "confirmed_suspect_total": int(len(audit_df)),
        "confirmed_suspect_with_history_transaction": int((audit_df["history_txn_count"] > 0).sum()),
        "confirmed_suspect_grade_counts": audit_df["explanation_grade"].value_counts().sort_index().to_dict(),
        "active_risk_review_account_count": int(len(queue_df)),
        "active_risk_review_grade_counts": queue_df["explanation_grade"].value_counts().sort_index().to_dict() if len(queue_df) else {},
        "association_rows": int(len(associations)),
        "suspicious_path_rows": int(len(paths)),
        "fund_flow_structure_rows": int(len(structures)),
        "core_note": "59个确认嫌疑账户中，缺少交易边的账户不能生成可信链路；本脚本将其纳入缺边审计，并对有边的高风险账户生成链路和研判报告。",
        "outputs": {
            "confirmed_suspect_audit_csv": str(audit_path),
            "confirmed_suspect_audit_md": str(audit_md_path),
            "risk_review_queue": str(LAYERED_DIR / "risk_review_queue_active_accounts.csv"),
            "associations": str(LAYERED_DIR / "risk_review_queue_top20_associations.csv"),
            "paths": str(LAYERED_DIR / "risk_review_queue_suspicious_paths.csv"),
            "structures": str(LAYERED_DIR / "risk_review_queue_fund_flow_structures.csv"),
            "judgement_reports": str(LAYERED_DIR / "layered_judgement_report_samples.md"),
        },
    }
    coverage_path = LAYERED_DIR / "layered_explainability_coverage.json"
    coverage_path.write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
