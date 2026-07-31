"""轻量 TGN 风格时间事件流模型。

该实现保留 TGN 的核心思想：按时间顺序处理交易事件，并用节点记忆表示
账户近期交易状态。为保证本项目在 CPU 和有限样本下可复现，交易先聚合为
“日级 src-dst 事件”，每个事件包含交易次数、金额、小时、异常交易等边属性。

注意：由于风险标签没有确认时间，本模型仍然是按交易时间窗口进行快照评估，
不能表述为严格的“未来新增标签预测”。
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    CLEAN_DIR,
    DST_COL,
    FEATURE_DIR,
    ID_COL,
    LABEL_DIR,
    METRIC_DIR,
    MODEL_DIR,
    PREDICTION_DIR,
    RANDOM_SEED,
    SPLITS,
    SRC_COL,
    TIME_COL,
)


ROLLING_HISTORY_DAYS = 120
EXPERIMENT = "v1_no_customer_type"


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def ensure_dirs() -> None:
    for path in [METRIC_DIR, MODEL_DIR, PREDICTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    positives = int(y_true.sum())
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
        prefix = f"top{int(rate * 100)}pct"
        result.update(
            {
                f"{prefix}_k": k,
                f"{prefix}_hits": hits,
                f"{prefix}_precision": float(hits / k),
                f"{prefix}_recall": float(hits / positives) if positives else 0.0,
            }
        )
    return result


def load_side_features(split: str, drop_static_profile: bool) -> tuple[pd.DataFrame, list[str]]:
    frame = pd.read_csv(FEATURE_DIR / f"dynamic_graph_features_{split}.csv")
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    frame = frame.merge(
        labels[[ID_COL, "label_A_suspect_vs_other", "label_code"]],
        on=ID_COL,
        how="inner",
    ).rename(columns={"label_A_suspect_vs_other": "target"})
    frame["target_all_accounts"] = frame["label_code"].eq(1).astype("int8")
    columns = [c for c in frame.columns if c.startswith("dyn_")]
    if drop_static_profile:
        columns = [c for c in columns if c not in {"dyn_account_age_months", "dyn_region_code"}]
    return frame, columns


def build_daily_event_stream(
    transactions: pd.DataFrame,
    split: str,
    id_to_idx: dict[int, int],
) -> tuple[list[dict[str, torch.Tensor]], dict]:
    end = pd.Timestamp(SPLITS[split][1])
    start = max(pd.Timestamp(transactions[TIME_COL].min()), end - pd.Timedelta(days=ROLLING_HISTORY_DAYS))
    tx = transactions.loc[
        transactions[TIME_COL].between(start, end),
        [SRC_COL, DST_COL, TIME_COL, "amount_abs", "self_loop", "non_positive_amount"],
    ].copy()
    tx = tx[tx[SRC_COL].isin(id_to_idx) & tx[DST_COL].isin(id_to_idx)]
    if tx.empty:
        return [], {"observation_start": str(start), "observation_end": str(end), "event_count": 0}

    tx["day"] = tx[TIME_COL].dt.floor("D")
    tx["hour_fraction"] = tx[TIME_COL].dt.hour.astype(float) / 23.0
    grouped = (
        tx.groupby(["day", SRC_COL, DST_COL], sort=True)
        .agg(
            txn_count=("amount_abs", "size"),
            amount_log_sum=("amount_abs", lambda x: np.log1p(x).sum()),
            amount_log_mean=("amount_abs", lambda x: np.log1p(x).mean()),
            hour_mean=("hour_fraction", "mean"),
            self_loop_ratio=("self_loop", "mean"),
            non_positive_ratio=("non_positive_amount", "mean"),
        )
        .reset_index()
    )

    events = []
    for day, group in grouped.groupby("day", sort=True):
        src = torch.tensor(group[SRC_COL].map(id_to_idx).to_numpy(dtype=np.int64), dtype=torch.long)
        dst = torch.tensor(group[DST_COL].map(id_to_idx).to_numpy(dtype=np.int64), dtype=torch.long)
        day_index = float((pd.Timestamp(day) - start).total_seconds() / 86400.0)
        day_sin = float(np.sin(2 * np.pi * day_index / 7.0))
        day_cos = float(np.cos(2 * np.pi * day_index / 7.0))
        edge_features = np.column_stack(
            [
                group["txn_count"].map(np.log1p).to_numpy(dtype=np.float32),
                group["amount_log_sum"].to_numpy(dtype=np.float32),
                group["amount_log_mean"].to_numpy(dtype=np.float32),
                group["hour_mean"].to_numpy(dtype=np.float32),
                group["self_loop_ratio"].to_numpy(dtype=np.float32),
                group["non_positive_ratio"].to_numpy(dtype=np.float32),
                np.full(len(group), day_sin, dtype=np.float32),
                np.full(len(group), day_cos, dtype=np.float32),
            ]
        )
        events.append(
            {
                "src": src,
                "dst": dst,
                "edge_features": torch.from_numpy(edge_features),
                "day_index": torch.tensor(day_index / max(1.0, ROLLING_HISTORY_DAYS), dtype=torch.float32),
            }
        )
    return events, {
        "observation_start": str(start),
        "observation_end": str(end),
        "event_count": int(len(grouped)),
        "active_day_count": int(len(events)),
    }


class TGNTemporalMemory(nn.Module):
    """面向小样本反诈场景的轻量时间节点记忆模型。"""

    def __init__(self, node_count: int, side_dim: int, memory_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.node_count = node_count
        self.memory_dim = memory_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(memory_dim * 2 + 8, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, memory_dim),
        )
        self.gru = nn.GRUCell(memory_dim, memory_dim)
        self.side_encoder = nn.Sequential(
            nn.Linear(side_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(memory_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def initial_memory(self, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.node_count, self.memory_dim, device=device)

    def update(
        self,
        memory: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_features: torch.Tensor,
    ) -> torch.Tensor:
        src_memory = memory[src]
        dst_memory = memory[dst]
        forward = self.edge_encoder(torch.cat([src_memory, dst_memory, edge_features], dim=1))
        reverse = self.edge_encoder(torch.cat([dst_memory, src_memory, edge_features], dim=1))
        aggregate = torch.zeros_like(memory)
        aggregate.index_add_(0, src, forward)
        aggregate.index_add_(0, dst, reverse)
        active = torch.unique(torch.cat([src, dst]))
        updated = self.gru(aggregate[active], memory[active])
        return memory.index_copy(0, active, updated)

    def score(self, memory: torch.Tensor, side_features: torch.Tensor) -> torch.Tensor:
        side = self.side_encoder(side_features)
        return self.classifier(torch.cat([memory, side], dim=1)).squeeze(-1)


def run_stream(
    model: TGNTemporalMemory,
    events: list[dict[str, torch.Tensor]],
    side_features: torch.Tensor,
    device: torch.device,
    training: bool = False,
    labels: torch.Tensor | None = None,
    train_mask: torch.Tensor | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    criterion: nn.Module | None = None,
    supervision_stride: int = 7,
) -> tuple[torch.Tensor, list[float]]:
    memory = model.initial_memory(device)
    losses = []
    for index, event in enumerate(events):
        memory = model.update(
            memory,
            event["src"].to(device),
            event["dst"].to(device),
            event["edge_features"].to(device),
        )
        # 截断反向传播，保留时间顺序但避免构建数十万步的计算图。
        if training and ((index + 1) % supervision_stride == 0 or index == len(events) - 1):
            if optimizer is None or criterion is None or labels is None or train_mask is None:
                raise RuntimeError("训练流缺少监督参数。")
            logits = model.score(memory, side_features)
            loss = criterion(logits[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            memory = memory.detach()
    return memory, losses


@torch.no_grad()
def predict_stream(
    model: TGNTemporalMemory,
    events: list[dict[str, torch.Tensor]],
    side_features: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    memory = model.initial_memory(device)
    for event in events:
        memory = model.update(
            memory,
            event["src"].to(device),
            event["dst"].to(device),
            event["edge_features"].to(device),
        )
    return torch.sigmoid(model.score(memory, side_features)).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="轻量 TGN 风格时间事件流模型。")
    parser.add_argument("--drop-static-profile", action="store_true")
    parser.add_argument("--experiment-suffix", default=EXPERIMENT)
    parser.add_argument("--memory-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed()
    device = torch.device("cpu")

    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")[[ID_COL]]
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    id_to_idx = {int(account_id): idx for idx, account_id in enumerate(accounts[ID_COL])}

    frames = {}
    side_frames = {}
    for split in SPLITS:
        frame, side_cols = load_side_features(split, args.drop_static_profile)
        frame = accounts.merge(frame, on=ID_COL, how="left").fillna(0)
        frames[split] = frame
        side_frames[split] = frame[side_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler().fit(side_frames["train"].to_numpy(dtype=float))
    side_tensors = {
        split: torch.tensor(
            scaler.transform(side_frames[split].to_numpy(dtype=float)).astype("float32"),
            dtype=torch.float32,
            device=device,
        )
        for split in SPLITS
    }
    streams = {}
    stream_reports = {}
    for split in SPLITS:
        streams[split], stream_reports[split] = build_daily_event_stream(transactions, split, id_to_idx)

    train_target = frames["train"]["target"].to_numpy(dtype=int)
    train_mask = torch.tensor(train_target >= 0, dtype=torch.bool, device=device)
    train_labels = torch.tensor(np.where(train_target >= 0, train_target, 0), dtype=torch.float32, device=device)
    positives = max(1, int(train_target[train_target >= 0].sum()))
    negatives = max(1, int((train_target[train_target >= 0] == 0).sum()))

    model = TGNTemporalMemory(
        node_count=len(accounts),
        side_dim=len(side_cols),
        memory_dim=args.memory_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    valid_y = frames["valid"]["target_all_accounts"].to_numpy(dtype=int)
    best_state = None
    best_epoch = 0
    best_valid_pr = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        _, losses = run_stream(
            model,
            streams["train"],
            side_tensors["train"],
            device,
            training=True,
            labels=train_labels,
            train_mask=train_mask,
            optimizer=optimizer,
            criterion=criterion,
        )
        valid_score = predict_stream(model, streams["valid"], side_tensors["valid"], device)
        valid_pr = float(average_precision_score(valid_y, valid_score))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)) if losses else 0.0,
                "valid_all_accounts_pr_auc": valid_pr,
            }
        )
        if valid_pr > best_valid_pr + 1e-7:
            best_valid_pr = valid_pr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("TGN 未产生有效检查点。")
    model.load_state_dict(best_state)

    report = {
        "_metadata": {
            "model_family": "lightweight TGN-style temporal node memory",
            "experiment_suffix": args.experiment_suffix,
            "feature_policy": "daily transaction event stream + rolling dynamic node features",
            "event_policy": "daily src-dst aggregation; events processed chronologically",
            "rolling_history_days": ROLLING_HISTORY_DAYS,
            "side_feature_count": len(side_cols),
            "memory_dim": args.memory_dim,
            "hidden_dim": args.hidden_dim,
            "evaluation_policy": "Strategy A training excludes victims; formal metrics use all accounts.",
            "label_time_limitation": "Risk labels have no confirmation time; this is snapshot evaluation, not future-new-label validation.",
            "random_seed": RANDOM_SEED,
        },
        "status": "ok",
        "best_epoch": best_epoch,
        "best_valid_all_accounts_pr_auc": best_valid_pr,
        "event_streams": stream_reports,
    }
    predictions = []
    for split in SPLITS:
        score = predict_stream(model, streams[split], side_tensors[split], device)
        frame = frames[split]
        candidate_mask = frame["target"].ge(0).to_numpy()
        report[split] = evaluate(frame.loc[candidate_mask, "target"].to_numpy(dtype=int), score[candidate_mask])
        report[f"{split}_all_accounts"] = evaluate(frame["target_all_accounts"].to_numpy(dtype=int), score)
        predictions.append(pd.DataFrame({ID_COL: frame[ID_COL], "split": split, "target": frame["target_all_accounts"], "score": score}))

    stem = f"model10_tgn_{args.experiment_suffix}_strategy_A"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "memory_dim": args.memory_dim,
            "hidden_dim": args.hidden_dim,
            "side_feature_columns": side_cols,
            "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
            "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
            "node_count": len(accounts),
        },
        MODEL_DIR / f"{stem}.pt",
    )
    (MODEL_DIR / f"{stem}_metadata.json").write_text(
        json.dumps(report["_metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{stem}.csv", index=False)
    (METRIC_DIR / f"tgn_metrics_{args.experiment_suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(history).to_csv(METRIC_DIR / f"{stem}_training_history.csv", index=False)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
