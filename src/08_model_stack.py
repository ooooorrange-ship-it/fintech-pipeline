import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys_path_added = False
import sys
if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys_path_added = True

from config import METRIC_DIR, PREDICTION_DIR, ID_COL, RANDOM_SEED  # noqa: E402


def ensure_dirs() -> None:
    METRIC_DIR.mkdir(parents=True, exist_ok=True)


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(-score)
    y = y_true[order]
    positives = y.sum()
    if positives == 0:
        return 0.0
    precision_at_k = np.cumsum(y) / (np.arange(len(y)) + 1)
    return float((precision_at_k * y).sum() / positives)


def roc_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = y_true.astype(int)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    pos_rank_sum = ranks[pos].sum()
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


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
    out = {
        "auc": roc_auc(y_true, score),
        "pr_auc_average_precision": average_precision(y_true, score),
    }
    out.update(topk_metrics(y_true, score, 0.01))
    out.update(topk_metrics(y_true, score, 0.05))
    return out


def load_pred(path: Path, split: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["split"].eq(split)].copy()


def normalize_score(score: np.ndarray) -> np.ndarray:
    score = score.astype(float)
    rank = score.argsort().argsort().astype(float)
    if len(score) <= 1:
        return score
    return rank / (len(score) - 1)


def grid_search_weights(y_true: np.ndarray, components: dict[str, np.ndarray]) -> tuple[dict[str, float], np.ndarray, dict]:
    names = list(components)
    grids = np.arange(0.0, 1.01, 0.1)
    best = None
    best_key = None
    best_metrics = None
    best_pred = None
    for wx in grids:
        for wg in grids:
            wr = round(1.0 - wx - wg, 10)
            if wr < -1e-9 or wr > 1.0:
                continue
            if abs(wx + wg + wr - 1.0) > 1e-9:
                continue
            weights = {"xgb": wx, "gnn": wg, "rule": wr}
            score = sum(weights[k] * components[k] for k in names)
            metrics = evaluate(y_true, score)
            key = (metrics["pr_auc_average_precision"], metrics["top5pct_recall"], metrics["auc"])
            if best_key is None or key > best_key:
                best = weights
                best_key = key
                best_metrics = metrics
                best_pred = score
    return best or {"xgb": 1.0, "gnn": 0.0, "rule": 0.0}, best_pred, best_metrics or {}


def output_stem(name: str, suffix: str) -> str:
    return f"{name}_{suffix}" if suffix else name


def find_latest(pattern: str) -> Path:
    candidates = sorted(PREDICTION_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"未找到匹配文件: {pattern}")
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XGB + 图分支 + 规则分支 stacking。")
    parser.add_argument("--xgb-pred", default="", help="XGB 预测文件，默认自动寻找最新文件。")
    parser.add_argument("--gnn-pred", default="", help="GNN 预测文件，默认自动寻找最新文件。")
    parser.add_argument("--rule-pred", default="", help="规则预测文件，默认自动寻找最新文件。")
    parser.add_argument("--experiment-suffix", default="v3_no_customer_type", help="输出后缀。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    suffix = args.experiment_suffix.strip()

    xgb_path = Path(args.xgb_pred) if args.xgb_pred else find_latest("model2_xgb_stat_graph_*no_customer_type_strategy_A.csv")
    gnn_path = Path(args.gnn_pred) if args.gnn_pred else find_latest("model3_hetero_prop_*no_customer_type_strategy_A.csv")
    rule_path = Path(args.rule_pred) if args.rule_pred else find_latest("model0_rule_*no_customer_type_strategy_A.csv")

    valid_xgb = load_pred(xgb_path, "valid")
    test_xgb = load_pred(xgb_path, "test")
    valid_gnn = load_pred(gnn_path, "valid")
    test_gnn = load_pred(gnn_path, "test")
    valid_rule = load_pred(rule_path, "valid")
    test_rule = load_pred(rule_path, "test")

    valid = valid_xgb[[ID_COL, "target", "score"]].rename(columns={"score": "xgb_score"})
    valid = valid.merge(valid_gnn[[ID_COL, "score"]].rename(columns={"score": "gnn_score"}), on=ID_COL, how="inner")
    valid = valid.merge(valid_rule[[ID_COL, "score"]].rename(columns={"score": "rule_score"}), on=ID_COL, how="inner")
    y_valid = valid["target"].to_numpy(dtype=int)

    components_valid = {
        "xgb": normalize_score(valid["xgb_score"].to_numpy()),
        "gnn": normalize_score(valid["gnn_score"].to_numpy()),
        "rule": normalize_score(valid["rule_score"].to_numpy()),
    }
    weights, valid_score, valid_metrics = grid_search_weights(y_valid, components_valid)

    test = test_xgb[[ID_COL, "target", "score"]].rename(columns={"score": "xgb_score"})
    test = test.merge(test_gnn[[ID_COL, "score"]].rename(columns={"score": "gnn_score"}), on=ID_COL, how="inner")
    test = test.merge(test_rule[[ID_COL, "score"]].rename(columns={"score": "rule_score"}), on=ID_COL, how="inner")
    y_test = test["target"].to_numpy(dtype=int)
    components_test = {
        "xgb": normalize_score(test["xgb_score"].to_numpy()),
        "gnn": normalize_score(test["gnn_score"].to_numpy()),
        "rule": normalize_score(test["rule_score"].to_numpy()),
    }
    test_score = sum(weights[k] * v for k, v in components_test.items())
    test_metrics = evaluate(y_test, test_score)

    out_pred = pd.concat(
        [
            pd.DataFrame({ID_COL: valid[ID_COL], "split": "valid", "target": y_valid, "score": valid_score}),
            pd.DataFrame({ID_COL: test[ID_COL], "split": "test", "target": y_test, "score": test_score}),
        ],
        ignore_index=True,
    )
    pred_path = PREDICTION_DIR / f"{output_stem('model4_stack', suffix)}_strategy_A.csv"
    out_pred.to_csv(pred_path, index=False)

    report = {
        "_metadata": {
            "experiment_suffix": suffix,
            "xgb_pred": str(xgb_path),
            "gnn_pred": str(gnn_path),
            "rule_pred": str(rule_path),
            "random_seed": RANDOM_SEED,
        },
        "status": "ok",
        "selected_weights": weights,
        "valid": valid_metrics,
        "test": test_metrics,
    }
    with (METRIC_DIR / f"{output_stem('stack_experiment_metrics', suffix)}.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
