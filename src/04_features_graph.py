import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import CLEAN_DIR, DST_COL, FEATURE_DIR, ID_COL, SPLITS, SRC_COL, TIME_COL  # noqa: E402

NEIGHBOR_BEHAVIOR_COLS = [
    "total_txn_count",
    "total_amount_sum",
    "out_amount_sum",
    "in_amount_sum",
    "in_out_amount_ratio",
    "burst_day_txn_ratio",
    "burst_day_amount_ratio",
    "counterparty_amount_top_ratio",
    "counterparty_txn_top_ratio",
    "fast_in_out_count_24h",
    "fast_in_out_balance_ratio_24h",
    "multi_in_one_out_count_24h",
    "max_prior_in_count_before_out_24h",
    "max_next_out_count_after_in_24h",
    "total_self_loop_count",
    "total_non_positive_amount_count",
]


def ensure_dirs() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).fillna(0.0)


class DSU:
    def __init__(self, nodes: list[int]) -> None:
        self.parent = {x: x for x in nodes}
        self.size = {x: 1 for x in nodes}

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def component_sizes(self) -> dict[int, int]:
        return {x: self.size[self.find(x)] for x in self.parent}


def pagerank(
    nodes: np.ndarray,
    edges: pd.DataFrame,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> pd.Series:
    node_to_idx = {int(n): i for i, n in enumerate(nodes)}
    n = len(nodes)
    out_neighbors: list[list[int]] = [[] for _ in range(n)]
    for src, dst in edges[[SRC_COL, DST_COL]].itertuples(index=False):
        if src in node_to_idx and dst in node_to_idx:
            out_neighbors[node_to_idx[int(src)]].append(node_to_idx[int(dst)])

    rank = np.full(n, 1.0 / n)
    base = (1.0 - damping) / n
    for _ in range(max_iter):
        new_rank = np.full(n, base)
        dangling_sum = 0.0
        for i, neigh in enumerate(out_neighbors):
            if neigh:
                share = damping * rank[i] / len(neigh)
                for j in neigh:
                    new_rank[j] += share
            else:
                dangling_sum += rank[i]
        if dangling_sum:
            new_rank += damping * dangling_sum / n
        if np.abs(new_rank - rank).sum() < tol:
            rank = new_rank
            break
        rank = new_rank
    return pd.Series(rank, index=nodes, name="pagerank")


def two_hop_counts(nodes: list[int], neighbors: dict[int, set[int]]) -> dict[int, int]:
    out = {}
    for node in nodes:
        hop2 = set()
        for nb in neighbors.get(node, set()):
            hop2.update(neighbors.get(nb, set()))
        hop2.discard(node)
        hop2.difference_update(neighbors.get(node, set()))
        out[node] = len(hop2)
    return out


def directed_two_hop_counts(nodes: list[int], neighbors: dict[int, set[int]]) -> dict[int, int]:
    out = {}
    for node in nodes:
        hop2 = set()
        for nb in neighbors.get(node, set()):
            hop2.update(neighbors.get(nb, set()))
        hop2.discard(node)
        out[node] = len(hop2)
    return out


def neighbor_behavior_features(
    nodes: np.ndarray,
    all_neighbors: dict[int, set[int]],
    out_neighbors: dict[int, set[int]],
    in_neighbors: dict[int, set[int]],
    stat_features: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    cols = []
    if stat_features is not None:
        cols = [c for c in NEIGHBOR_BEHAVIOR_COLS if c in stat_features.columns]
    if not cols:
        return pd.DataFrame({ID_COL: nodes})

    stat = stat_features.set_index(ID_COL)[cols].fillna(0)

    def summarize(node: int, mapping: dict[int, set[int]], prefix: str, use_max: bool) -> dict:
        neigh = list(mapping.get(node, set()))
        out = {}
        if not neigh:
            for col in cols:
                out[f"{prefix}_{col}_mean"] = 0.0
                if use_max:
                    out[f"{prefix}_{col}_max"] = 0.0
            return out
        vals = stat.reindex(neigh).fillna(0)
        means = vals.mean(axis=0)
        maxs = vals.max(axis=0)
        for col in cols:
            out[f"{prefix}_{col}_mean"] = float(means[col])
            if use_max:
                out[f"{prefix}_{col}_max"] = float(maxs[col])
        return out

    for node in nodes:
        node_i = int(node)
        row = {ID_COL: node_i}
        row.update(summarize(node_i, all_neighbors, "graph_nb", use_max=True))
        row.update(summarize(node_i, out_neighbors, "graph_out_nb", use_max=False))
        row.update(summarize(node_i, in_neighbors, "graph_in_nb", use_max=False))
        rows.append(row)
    return pd.DataFrame(rows)


def build_graph_features(
    accounts: pd.DataFrame,
    transactions: pd.DataFrame,
    start: str,
    end: str,
    stat_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    nodes = accounts[ID_COL].astype("int64").to_numpy()
    tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ][[SRC_COL, DST_COL]].copy()

    base = pd.DataFrame({ID_COL: nodes})
    if tx.empty:
        for col in [
            "graph_in_degree",
            "graph_out_degree",
            "graph_total_degree",
            "graph_reciprocal_neighbor_count",
            "graph_reciprocal_ratio",
            "graph_out_in_degree_ratio",
            "graph_in_out_degree_ratio",
            "graph_component_size",
            "graph_component_edge_count",
            "graph_component_density",
            "graph_two_hop_neighbor_count",
            "graph_cycle3_proxy_count",
            "graph_neighbor_degree_mean",
            "graph_neighbor_degree_max",
            "graph_neighbor_degree_std",
            "graph_low_degree_neighbor_ratio",
            "graph_hub_neighbor_ratio",
            "graph_out_two_hop_reach",
            "graph_in_two_hop_reach",
            "graph_path_through_proxy",
            "graph_fan_in_out_balance",
            "pagerank",
        ]:
            base[col] = 0.0
        return base

    out_degree = tx.groupby(SRC_COL)[DST_COL].nunique().rename("graph_out_degree")
    in_degree = tx.groupby(DST_COL)[SRC_COL].nunique().rename("graph_in_degree")
    base = base.merge(out_degree, left_on=ID_COL, right_index=True, how="left")
    base = base.merge(in_degree, left_on=ID_COL, right_index=True, how="left")
    base[["graph_out_degree", "graph_in_degree"]] = base[["graph_out_degree", "graph_in_degree"]].fillna(0)
    base["graph_total_degree"] = base["graph_out_degree"] + base["graph_in_degree"]

    pair = tx.drop_duplicates()
    pair_rev = pair.rename(columns={SRC_COL: DST_COL, DST_COL: SRC_COL})
    reciprocal = pair.merge(pair_rev, on=[SRC_COL, DST_COL], how="inner")
    reciprocal_count = reciprocal.groupby(SRC_COL)[DST_COL].nunique().rename("graph_reciprocal_neighbor_count")
    base = base.merge(reciprocal_count, left_on=ID_COL, right_index=True, how="left")
    base["graph_reciprocal_neighbor_count"] = base["graph_reciprocal_neighbor_count"].fillna(0)
    base["graph_reciprocal_ratio"] = safe_divide(base["graph_reciprocal_neighbor_count"], base["graph_total_degree"])
    base["graph_out_in_degree_ratio"] = safe_divide(base["graph_out_degree"], base["graph_in_degree"])
    base["graph_in_out_degree_ratio"] = safe_divide(base["graph_in_degree"], base["graph_out_degree"])

    dsu = DSU([int(x) for x in nodes])
    neighbors: dict[int, set[int]] = defaultdict(set)
    out_neighbors: dict[int, set[int]] = defaultdict(set)
    in_neighbors: dict[int, set[int]] = defaultdict(set)
    for src, dst in pair[[SRC_COL, DST_COL]].itertuples(index=False):
        src_i, dst_i = int(src), int(dst)
        dsu.union(src_i, dst_i)
        if src_i != dst_i:
            neighbors[src_i].add(dst_i)
            neighbors[dst_i].add(src_i)
            out_neighbors[src_i].add(dst_i)
            in_neighbors[dst_i].add(src_i)
    comp_size = pd.Series(dsu.component_sizes(), name="graph_component_size")
    base = base.merge(comp_size, left_on=ID_COL, right_index=True, how="left")

    component_edge_count: dict[int, int] = defaultdict(int)
    for src, dst in pair[[SRC_COL, DST_COL]].itertuples(index=False):
        component_edge_count[dsu.find(int(src))] += 1
    node_component_edges = {int(node): component_edge_count[dsu.find(int(node))] for node in nodes}
    base = base.merge(
        pd.Series(node_component_edges, name="graph_component_edge_count"),
        left_on=ID_COL,
        right_index=True,
        how="left",
    )
    max_edges = base["graph_component_size"] * (base["graph_component_size"] - 1)
    base["graph_component_density"] = safe_divide(base["graph_component_edge_count"], max_edges)

    hop2 = pd.Series(two_hop_counts([int(x) for x in nodes], neighbors), name="graph_two_hop_neighbor_count")
    base = base.merge(hop2, left_on=ID_COL, right_index=True, how="left")

    out_hop2 = pd.Series(
        directed_two_hop_counts([int(x) for x in nodes], out_neighbors),
        name="graph_out_two_hop_reach",
    )
    in_hop2 = pd.Series(
        directed_two_hop_counts([int(x) for x in nodes], in_neighbors),
        name="graph_in_two_hop_reach",
    )
    base = base.merge(out_hop2, left_on=ID_COL, right_index=True, how="left")
    base = base.merge(in_hop2, left_on=ID_COL, right_index=True, how="left")
    base["graph_path_through_proxy"] = base["graph_in_degree"] * base["graph_out_degree"]
    base["graph_fan_in_out_balance"] = 1.0 - (
        (base["graph_in_degree"] - base["graph_out_degree"]).abs()
        / (base["graph_in_degree"] + base["graph_out_degree"]).replace(0, np.nan)
    )

    # 第一版不做昂贵的三角/多步闭环枚举，用互惠转账数量作为 A-B-A 回流结构代理。
    base["graph_cycle3_proxy_count"] = base["graph_reciprocal_neighbor_count"].fillna(0)

    degree_map = dict(zip(base[ID_COL].astype("int64"), base["graph_total_degree"].astype(float)))
    hub_threshold = float(base["graph_total_degree"].quantile(0.95))
    neighbor_rows = []
    for node in nodes:
        node = int(node)
        neigh = neighbors.get(node, set())
        if not neigh:
            neighbor_rows.append(
                {
                    ID_COL: node,
                    "graph_neighbor_degree_mean": 0.0,
                    "graph_neighbor_degree_max": 0.0,
                    "graph_neighbor_degree_std": 0.0,
                    "graph_low_degree_neighbor_ratio": 0.0,
                    "graph_hub_neighbor_ratio": 0.0,
                }
            )
            continue
        degs = np.array([degree_map.get(int(nb), 0.0) for nb in neigh], dtype=float)
        neighbor_rows.append(
            {
                ID_COL: node,
                "graph_neighbor_degree_mean": float(degs.mean()),
                "graph_neighbor_degree_max": float(degs.max()),
                "graph_neighbor_degree_std": float(degs.std()),
                "graph_low_degree_neighbor_ratio": float((degs <= 1).mean()),
                "graph_hub_neighbor_ratio": float((degs >= hub_threshold).mean()) if hub_threshold > 0 else 0.0,
            }
        )
    base = base.merge(pd.DataFrame(neighbor_rows), on=ID_COL, how="left")

    pr = pagerank(nodes, pair)
    base = base.merge(pr, left_on=ID_COL, right_index=True, how="left")
    base = base.merge(
        neighbor_behavior_features(nodes, neighbors, out_neighbors, in_neighbors, stat_features),
        on=ID_COL,
        how="left",
    )
    base = base.fillna(0)
    return base


def maybe_build_node2vec(accounts: pd.DataFrame, transactions: pd.DataFrame, start: str, end: str, split: str) -> dict:
    try:
        import networkx as nx
        from node2vec import Node2Vec
    except Exception as exc:
        return {"split": split, "status": "skipped", "reason": f"缺少依赖: {type(exc).__name__}: {exc}"}

    tx = transactions[
        (transactions[TIME_COL] >= pd.Timestamp(start))
        & (transactions[TIME_COL] <= pd.Timestamp(end))
    ][[SRC_COL, DST_COL]].copy()
    graph = nx.from_pandas_edgelist(tx, source=SRC_COL, target=DST_COL, create_using=nx.Graph())
    graph.add_nodes_from(accounts[ID_COL].astype("int64").tolist())

    node2vec = Node2Vec(graph, dimensions=8, walk_length=8, num_walks=8, workers=1, quiet=True, seed=2026)
    model = node2vec.fit(window=5, min_count=1, batch_words=256)
    rows = []
    for account_id in accounts[ID_COL].astype("int64"):
        key = str(account_id)
        if key in model.wv:
            vec = model.wv[key]
        else:
            vec = np.zeros(8)
        rows.append([account_id, *vec])
    cols = [ID_COL] + [f"node2vec_{i:02d}" for i in range(8)]
    pd.DataFrame(rows, columns=cols).to_csv(FEATURE_DIR / f"node2vec_features_{split}.csv", index=False)
    return {"split": split, "status": "ok", "dimensions": 8}


def main() -> None:
    ensure_dirs()
    accounts = pd.read_csv(CLEAN_DIR / "clean_accounts.csv")
    transactions = pd.read_csv(CLEAN_DIR / "clean_transactions.csv", parse_dates=[TIME_COL])
    report = {"splits": {}, "node2vec": []}

    for split, (start, end) in SPLITS.items():
        stat_path = FEATURE_DIR / f"stat_features_{split}.csv"
        stat_features = pd.read_csv(stat_path) if stat_path.exists() else None
        feat = build_graph_features(accounts, transactions, start, end, stat_features)
        path = FEATURE_DIR / f"graph_features_{split}.csv"
        feat.to_csv(path, index=False)
        report["splits"][split] = {"rows": int(len(feat)), "cols": int(feat.shape[1]), "path": str(path)}
        report["node2vec"].append(maybe_build_node2vec(accounts, transactions, start, end, split))

    with (FEATURE_DIR / "graph_feature_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
