"""真实的 PyTorch Geometric GraphSAGE 消融模型。"""

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
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

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


class GraphSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 1)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.norm1(x).relu()
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.norm2(x).relu()
        x = nn.functional.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x).squeeze(-1)


def ensure_dirs() -> None:
    for path in [METRIC_DIR, MODEL_DIR, PREDICTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def set_seed() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)


def evaluate(y_true: np.ndarray, score: np.ndarray) -> dict:
    result = {
        "auc": float(roc_auc_score(y_true, score)),
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
    }
    for rate in [0.01, 0.05]:
        k = max(1, int(np.ceil(len(y_true) * rate)))
        order = np.argsort(-score)[:k]
        hits = int(y_true[order].sum())
        positives = int(y_true.sum())
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


def load_node_frame(split: str, drop_static_profile: bool) -> pd.DataFrame:
    frame = pd.read_csv(FEATURE_DIR / f"stat_features_{split}.csv")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"graph_features_{split}.csv"), on=ID_COL, how="left")
    frame = frame.merge(pd.read_csv(FEATURE_DIR / f"dynamic_graph_features_{split}.csv"), on=ID_COL, how="left")
    labels = pd.read_csv(LABEL_DIR / "labels_all_strategies.csv")
    frame = frame.merge(
        labels[[ID_COL, "label_A_suspect_vs_other", "label_code", "label_text"]],
        on=ID_COL,
        how="inner",
    ).rename(columns={"label_A_suspect_vs_other": "target"})
    frame["target_all_accounts"] = frame["label_code"].eq(1).astype("int8")
    frame = frame[[c for c in frame.columns if not c.startswith("customer_type_")]]
    if drop_static_profile:
        frame = frame.drop(columns=[c for c in ["region_code", "account_age_months"] if c in frame])
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {ID_COL, "target", "target_all_accounts", "label_code", "label_text"}
    return [c for c in frame.columns if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])]


def observation_window(transactions: pd.DataFrame, split: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(SPLITS[split][1])
    start = max(pd.Timestamp(transactions[TIME_COL].min()), end - pd.Timedelta(days=ROLLING_HISTORY_DAYS))
    return start, end


def build_edge_index(
    transactions: pd.DataFrame,
    split: str,
    id_to_idx: dict[int, int],
) -> tuple[torch.Tensor, dict]:
    start, end = observation_window(transactions, split)
    tx = transactions.loc[
        transactions[TIME_COL].between(start, end),
        [SRC_COL, DST_COL],
    ].drop_duplicates()
    tx = tx[tx[SRC_COL].isin(id_to_idx) & tx[DST_COL].isin(id_to_idx)]
    forward = pd.DataFrame(
        {
            "src": tx[SRC_COL].map(id_to_idx).astype(int),
            "dst": tx[DST_COL].map(id_to_idx).astype(int),
        }
    )
    reverse = forward.rename(columns={"src": "dst", "dst": "src"})[["src", "dst"]]
    bidirectional = pd.concat([forward, reverse], ignore_index=True).drop_duplicates()
    edge_index = torch.tensor(bidirectional[["src", "dst"]].to_numpy().T, dtype=torch.long)
    report = {
        "observation_start": str(start),
        "observation_end": str(end),
        "unique_directed_transaction_edges": int(len(forward)),
        "message_passing_edges": int(edge_index.shape[1]),
    }
    return edge_index, report


def build_data(
    frame: pd.DataFrame,
    cols: list[str],
    scaler: StandardScaler,
    edge_index: torch.Tensor,
) -> Data:
    x = scaler.transform(frame[cols].to_numpy(dtype=float)).astype("float32")
    target = frame["target"].to_numpy(dtype=int)
    data = Data(
        x=torch.from_numpy(x),
        edge_index=edge_index,
        y=torch.from_numpy(np.where(target >= 0, target, 0).astype("float32")),
    )
    data.train_mask = torch.from_numpy(target >= 0)
    data.y_all_accounts = torch.from_numpy(frame["target_all_accounts"].to_numpy(dtype="float32"))
    return data


@torch.no_grad()
def predict(model: GraphSAGE, data: Data) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(data.x, data.edge_index)).cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch Geometric GraphSAGE 涉诈账户识别模型。")
    parser.add_argument("--drop-static-profile", action="store_true", help="额外删除地区和开户时长。")
    parser.add_argument("--experiment-suffix", default=EXPERIMENT)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    set_seed()
    device = torch.device("cpu")

    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")[[ID_COL]]
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    frames = {}
    for split in SPLITS:
        raw = load_node_frame(split, args.drop_static_profile)
        frames[split] = accounts.merge(raw, on=ID_COL, how="left").fillna(0)
    cols = feature_columns(frames["train"])
    scaler = StandardScaler().fit(frames["train"][cols].to_numpy(dtype=float))
    id_to_idx = {int(account_id): idx for idx, account_id in enumerate(accounts[ID_COL])}

    graph_reports = {}
    data = {}
    for split in SPLITS:
        edge_index, graph_reports[split] = build_edge_index(transactions, split, id_to_idx)
        data[split] = build_data(frames[split], cols, scaler, edge_index).to(device)

    model = GraphSAGE(len(cols), args.hidden_dim, args.dropout).to(device)
    train_target = frames["train"].loc[frames["train"]["target"].ge(0), "target"].to_numpy(dtype=int)
    positives = max(1, int(train_target.sum()))
    negatives = max(1, int((train_target == 0).sum()))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives / positives, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=5e-4)

    best_state = None
    best_epoch = 0
    best_valid_pr = -1.0
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data["train"].x, data["train"].edge_index)
        loss = criterion(logits[data["train"].train_mask], data["train"].y[data["train"].train_mask])
        loss.backward()
        optimizer.step()

        valid_score = predict(model, data["valid"])
        valid_y = frames["valid"]["target_all_accounts"].to_numpy(dtype=int)
        valid_pr = float(average_precision_score(valid_y, valid_score))
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "valid_all_accounts_pr_auc": valid_pr})
        if valid_pr > best_valid_pr + 1e-7:
            best_valid_pr = valid_pr
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            break

    if best_state is None:
        raise RuntimeError("GraphSAGE 未产生有效检查点。")
    model.load_state_dict(best_state)

    report = {
        "_metadata": {
            "model_family": "PyTorch Geometric GraphSAGE",
            "experiment_suffix": args.experiment_suffix,
            "feature_policy": (
                "stat + static graph + rolling dynamic graph; all static profile fields removed"
                if args.drop_static_profile
                else "stat + static graph + rolling dynamic graph; customer_type removed"
            ),
            "drop_static_profile": bool(args.drop_static_profile),
            "feature_count": len(cols),
            "graph_policy": "120-day rolling transaction graph; transaction edges made bidirectional for message passing",
            "edge_attribute_policy": "time and amount are encoded in dyn_* node features; vanilla SAGEConv consumes topology",
            "evaluation_policy": "Strategy A loss excludes victims; early stopping and formal metrics use all accounts",
            "label_time_limitation": "Risk labels have no confirmation time; this is snapshot evaluation, not future-new-label validation.",
            "random_seed": RANDOM_SEED,
            "torch_version": torch.__version__,
        },
        "status": "ok",
        "best_epoch": best_epoch,
        "best_valid_all_accounts_pr_auc": best_valid_pr,
        "graph_snapshots": graph_reports,
    }
    predictions = []
    for split, split_data in data.items():
        score = predict(model, split_data)
        frame = frames[split]
        candidate_mask = frame["target"].ge(0).to_numpy()
        candidate_y = frame.loc[candidate_mask, "target"].to_numpy(dtype=int)
        all_y = frame["target_all_accounts"].to_numpy(dtype=int)
        report[split] = evaluate(candidate_y, score[candidate_mask])
        report[f"{split}_all_accounts"] = evaluate(all_y, score)
        predictions.append(pd.DataFrame({ID_COL: frame[ID_COL], "split": split, "target": all_y, "score": score}))

    stem = f"model6_graphsage_{args.experiment_suffix}_strategy_A"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(cols),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "feature_columns": cols,
            "scaler_mean": torch.tensor(scaler.mean_, dtype=torch.float64),
            "scaler_scale": torch.tensor(scaler.scale_, dtype=torch.float64),
        },
        MODEL_DIR / f"{stem}.pt",
    )
    (MODEL_DIR / f"{stem}_metadata.json").write_text(
        json.dumps(report["_metadata"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.concat(predictions, ignore_index=True).to_csv(PREDICTION_DIR / f"{stem}.csv", index=False)
    pd.DataFrame(history).to_csv(METRIC_DIR / f"{stem}_training_history.csv", index=False)
    (METRIC_DIR / f"graphsage_metrics_{args.experiment_suffix}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
