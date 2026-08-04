"""规则感知校准层。

灵感来自真实银行风控里的“模型 + 规则锚点”架构：
1. 用 train 窗口计算规则阈值，避免 valid/test 分布泄露。
2. 将快进快出、多入一出、闭环/自环、交易突发、对手方集中等规则信号汇总为 rule_score。
3. 只用 valid 选择最终主模型（model11_validation_selected_best_strategy_A）与 rule_score 的融合权重；若规则层无增益，则保持最终主模型。
4. 输出每个账户命中的规则证据，供研判报告引用。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import FEATURE_DIR, ID_COL, METRIC_DIR, MODEL_DIR, PREDICTION_DIR, PROJECT_ROOT, RANDOM_SEED  # noqa: E402


STEM = "model14_rule_aware_guardrailed_strategy_A"
BASE_PREDICTION = PREDICTION_DIR / "model11_validation_selected_best_strategy_A.csv"
EVIDENCE_PATH = PREDICTION_DIR.parent / "explanations" / "rule_aware_evidence_v1.csv"

RULE_GROUPS = {
    "fast_flow": {
        "weight": 1.4,
        "features": [
            "fast_in_out_count_24h",
            "fast_in_out_balance_ratio_24h",
            "prior_in_before_out_count_24h",
            "dyn_motif_fast_in_out_count_24h",
            "dyn_motif_fast_in_out_balance_ratio_24h",
            "dyn_motif_prior_in_before_out_count_24h",
        ],
        "description": "24小时快进快出或出账前存在短窗口入账来源",
    },
    "fan_motif": {
        "weight": 1.2,
        "features": [
            "one_in_multi_out_count_24h",
            "multi_in_one_out_count_24h",
            "max_prior_in_count_before_out_24h",
            "max_next_out_count_after_in_24h",
            "dyn_motif_one_in_multi_out_count_24h",
            "dyn_motif_multi_in_one_out_count_24h",
            "dyn_motif_max_prior_in_count_before_out_24h",
            "dyn_motif_max_next_out_count_after_in_24h",
        ],
        "description": "多入一出、一入多出或短时资金归集/分散",
    },
    "closed_loop_or_dirty_edge": {
        "weight": 1.3,
        "features": [
            "total_self_loop_count",
            "total_non_positive_amount_count",
            "graph_reciprocal_neighbor_count",
            "graph_reciprocal_ratio",
            "graph_cycle3_proxy_count",
            "graph_path_through_proxy",
        ],
        "description": "自环、非正金额、互惠边或闭环代理结构",
    },
    "burst_velocity": {
        "weight": 1.0,
        "features": [
            "burst_day_txn_ratio",
            "burst_day_amount_ratio",
            "txn_count_growth",
            "daily_txn_count_cv",
            "monthly_txn_count_cv",
            "dyn_total_week_txn_burst_ratio",
            "dyn_total_week_amount_burst_ratio",
            "dyn_total_last_prev_week_txn_growth",
        ],
        "description": "交易频次或金额在短时间内突增",
    },
    "counterparty_concentration": {
        "weight": 0.9,
        "features": [
            "counterparty_amount_top_ratio",
            "counterparty_txn_top_ratio",
            "out_counterparty_nunique",
            "in_counterparty_nunique",
            "dyn_cp_new_second_half_ratio",
            "dyn_cp_lost_second_half_count",
            "dyn_cp_jaccard_first_second",
        ],
        "description": "交易对手集中、新增/流失异常或关系稳定性异常",
    },
    "graph_neighbor_risk_proxy": {
        "weight": 1.1,
        "features": [
            "graph_two_hop_neighbor_count",
            "graph_low_degree_neighbor_ratio",
            "graph_nb_fast_in_out_count_24h_max",
            "graph_nb_multi_in_one_out_count_24h_max",
            "graph_nb_total_self_loop_count_max",
            "graph_nb_total_non_positive_amount_count_max",
            "graph_out_nb_fast_in_out_count_24h_mean",
            "graph_in_nb_fast_in_out_count_24h_mean",
        ],
        "description": "邻居节点存在快进快出、自环、非正金额或低度异常关系",
    },
}


def normalize(score: np.ndarray) -> np.ndarray:
    return pd.Series(score).rank(method="average", pct=True).to_numpy(dtype=float)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        idx = np.argsort(-score)[:k]
        hits = int(y_true[idx].sum())
        prefix = f"top{int(rate * 100)}pct"
        result[f"{prefix}_k"] = k
        result[f"{prefix}_hits"] = hits
        result[f"{prefix}_precision"] = float(hits / k)
        result[f"{prefix}_recall"] = float(hits / positives) if positives else 0.0
    return result


def load_features(split: str) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"graph_features_{split}.csv"), on=ID_COL, how="left")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"dynamic_graph_features_{split}.csv"), on=ID_COL, how="left")
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def build_thresholds(train: pd.DataFrame) -> dict:
    thresholds = {}
    all_features = sorted({feature for group in RULE_GROUPS.values() for feature in group["features"]})
    for feature in all_features:
        if feature not in train.columns:
            continue
        values = pd.to_numeric(train[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
        positive = values[values > 0]
        if len(positive) == 0:
            thresholds[feature] = {"mode": "positive", "threshold": 0.0, "scale": 1.0}
            continue
        threshold = float(np.quantile(positive, 0.75))
        scale = float(np.quantile(positive, 0.95))
        if not np.isfinite(scale) or scale <= threshold:
            scale = max(1.0, threshold)
        thresholds[feature] = {"mode": "quantile_from_train", "threshold": threshold, "scale": scale}
    return thresholds


def feature_signal(frame: pd.DataFrame, feature: str, threshold: dict) -> np.ndarray:
    if feature not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    values = pd.to_numeric(frame[feature], errors="coerce").fillna(0).to_numpy(dtype=float)
    thr = float(threshold.get("threshold", 0.0))
    scale = max(float(threshold.get("scale", 1.0)), 1e-9)
    signal = np.where(values > thr, np.clip(values / scale, 0.0, 1.0), 0.0)
    return signal.astype(float)


def score_rules(frame: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    out = frame[[ID_COL]].copy()
    raw_score = np.zeros(len(frame), dtype=float)
    max_score = 0.0
    evidence_lists = [[] for _ in range(len(frame))]

    for group_name, config in RULE_GROUPS.items():
        group_signal = np.zeros(len(frame), dtype=float)
        hit_features = []
        for feature in config["features"]:
            if feature not in thresholds:
                continue
            signal = feature_signal(frame, feature, thresholds[feature])
            if signal.max() > 0:
                hit_features.append(feature)
            group_signal = np.maximum(group_signal, signal)

        weight = float(config["weight"])
        raw_score += weight * group_signal
        max_score += weight
        out[f"rule_{group_name}_score"] = group_signal
        out[f"rule_{group_name}_hit"] = (group_signal > 0).astype("int8")
        for idx, value in enumerate(group_signal):
            if value > 0:
                evidence_lists[idx].append(
                    {
                        "rule": group_name,
                        "strength": round(float(value), 4),
                        "description": config["description"],
                        "features": hit_features[:5],
                    }
                )

    out["rule_score_raw"] = raw_score
    out["rule_score"] = raw_score / max(max_score, 1e-9)
    out["rule_hit_count"] = out[[c for c in out.columns if c.endswith("_hit")]].sum(axis=1)
    out["rule_evidence"] = [json.dumps(items, ensure_ascii=False) for items in evidence_lists]
    return out


def load_base_predictions(split: str) -> pd.DataFrame:
    pred = pd.read_csv(BASE_PREDICTION)
    return pred[pred["split"].eq(split)][[ID_COL, "target", "score"]].rename(columns={"score": "model11_score"})


def select_weight(valid: pd.DataFrame) -> tuple[dict, list[dict]]:
    y = valid["target"].to_numpy(dtype=int)
    model_score = normalize(valid["model11_score"].to_numpy(dtype=float))
    rule_score = normalize(valid["rule_score"].to_numpy(dtype=float))
    candidates = []
    best = None
    for step in range(21):
        model_weight = step / 20
        rule_weight = 1.0 - model_weight
        score = model_weight * model_score + rule_weight * rule_score
        metrics = evaluate(y, score)
        row = {
            "weight_model11": model_weight,
            "weight_rule_score": rule_weight,
            "valid_auc": metrics["auc"],
            "valid_pr_auc": metrics["pr_auc_average_precision"],
            "valid_top5_recall": metrics["top5pct_recall"],
            "valid_top5_hits": metrics["top5pct_hits"],
        }
        candidates.append(row)
        key = (row["valid_pr_auc"], row["valid_top5_recall"], row["valid_auc"])
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    candidates = sorted(candidates, key=lambda x: (x["valid_pr_auc"], x["valid_top5_recall"], x["valid_auc"]), reverse=True)
    return best[1], candidates[:5]


def apply_weight(frame: pd.DataFrame, selected: dict) -> np.ndarray:
    model_score = normalize(frame["model11_score"].to_numpy(dtype=float))
    rule_score = normalize(frame["rule_score"].to_numpy(dtype=float))
    return selected["weight_model11"] * model_score + selected["weight_rule_score"] * rule_score


def main() -> None:
    for path in [METRIC_DIR, MODEL_DIR, PREDICTION_DIR, EVIDENCE_PATH.parent]:
        path.mkdir(parents=True, exist_ok=True)

    train_features = load_features("train")
    thresholds = build_thresholds(train_features)
    frames = {}
    for split in ["valid", "test"]:
        features = load_features(split)
        rules = score_rules(features, thresholds)
        frames[split] = load_base_predictions(split).merge(rules, on=ID_COL, how="inner")

    selected, top_candidates = select_weight(frames["valid"])
    predictions = []
    evidence = []
    metrics = {"valid_all_accounts": None, "test_all_accounts": None}
    for split, frame in frames.items():
        final_score = apply_weight(frame, selected)
        y = frame["target"].to_numpy(dtype=int)
        metrics[f"{split}_all_accounts"] = evaluate(y, final_score)
        part = pd.DataFrame({ID_COL: frame[ID_COL], "split": split, "target": y, "score": final_score})
        predictions.append(part)
        evidence_part = frame[
            [
                ID_COL,
                "target",
                "model11_score",
                "rule_score",
                "rule_hit_count",
                "rule_evidence",
            ]
            + [c for c in frame.columns if c.startswith("rule_") and c.endswith("_hit")]
        ].copy()
        evidence_part.insert(1, "split", split)
        evidence_part["final_score"] = final_score
        evidence.append(evidence_part)

    selected_weights = {
        "model11_validation_selected_best_strategy_A": selected["weight_model11"],
        "rule_score": selected["weight_rule_score"],
    }
    report = {
        "status": "ok",
        "artifact_type": "rule_aware_guardrailed_calibration",
        "random_seed": RANDOM_SEED,
        "base_model": "model11_validation_selected_best_strategy_A",
        "selection_data": "valid_all_accounts",
        "selection_metric_order": ["valid PR-AUC", "valid Top5% recall", "valid AUC"],
        "selected_weights": selected_weights,
        "decision": "keep_final_model" if selected_weights["rule_score"] == 0 else "blend_final_model_and_rule_score",
        "decision_note": (
            "规则层未提升验证集 PR-AUC，因此仅作为解释锚点，不改变最终风险分。"
            if selected_weights["rule_score"] == 0
            else "规则层在验证集上提供增量，因此纳入最终校准分。"
        ),
        "rule_groups": RULE_GROUPS,
        "threshold_policy": "规则阈值仅由 train 窗口正值分布的75%/95%分位数生成。",
        "thresholds": thresholds,
        "candidate_top5": top_candidates,
        **metrics,
    }
    artifact = {
        "artifact_type": "rule_aware_guardrailed_selection",
        "selected_weights": selected_weights,
        "base_prediction": str(BASE_PREDICTION.relative_to(PROJECT_ROOT)),
        "normalization": "within_split_average_percentile_rank",
        "threshold_policy": report["threshold_policy"],
        "decision": report["decision"],
        "decision_note": report["decision_note"],
    }
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{STEM}.csv", index=False)
    pd.concat(evidence, ignore_index=True).to_csv(EVIDENCE_PATH, index=False)
    (METRIC_DIR / "rule_aware_calibration_metrics_v1.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (MODEL_DIR / f"{STEM}.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
