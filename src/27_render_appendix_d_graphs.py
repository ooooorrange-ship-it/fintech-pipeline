"""渲染附录 D 测试集真实资金图谱（每条边=一笔交易，文字直接写在边上）。"""

import math
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC", "Songti SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

TX = pd.read_csv("outputs/clean/clean_transactions.csv", parse_dates=["txn_time"])
LAB = pd.read_csv("outputs/labels/labels_all_strategies.csv")
SCORES = pd.read_csv("outputs/predictions/model11_validation_selected_best_strategy_A.csv")
SCORES = SCORES[SCORES["split"].eq("test")].set_index("account_id")["score"].to_dict()
LABEL = LAB.set_index("account_id")["label_text"].to_dict()

COLOR = {"嫌疑人": "#d64545", "受害人": "#f59e0b", "其它": "#2563eb", "未知": "#7f8c8d"}
IN_COLOR = "#e67e22"
OUT_COLOR = "#2f6fbf"
RAD_SETS = {
    1: [0.0],
    2: [0.46, -0.52],
    3: [0.44, -0.28, -0.62],
    4: [0.32, -0.38, 0.68, -0.78],
}


def raw_edges(account_id: int) -> list[dict]:
    sub = TX[(TX["src_account_id"] == account_id) | (TX["dst_account_id"] == account_id)].copy()
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "src": int(r["src_account_id"]),
                "dst": int(r["dst_account_id"]),
                "amount": float(r["amount_abs"]),
                "time": pd.Timestamp(r["txn_time"]),
            }
        )
    rows.sort(key=lambda x: x["time"])
    return rows


def bezier(p1: tuple, p2: tuple, rad: float, t: float) -> tuple[float, float]:
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    cx, cy = mx - rad * dy, my + rad * dx
    bx = (1 - t) ** 2 * x1 + 2 * t * (1 - t) * cx + t**2 * x2
    by = (1 - t) ** 2 * y1 + 2 * t * (1 - t) * cy + t**2 * y2
    return bx, by


def bezier_tangent(p1: tuple, p2: tuple, rad: float, t: float) -> tuple[float, float]:
    """返回二次贝塞尔曲线在 t 处的切向量。"""
    x1, y1 = p1
    x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2 - rad * dy, (y1 + y2) / 2 + rad * dx
    tx = 2 * (1 - t) * (cx - x1) + 2 * t * (x2 - cx)
    ty = 2 * (1 - t) * (cy - y1) + 2 * t * (y2 - cy)
    return tx, ty


def draw_node(ax, pos: tuple, aid: int, is_root: bool = False) -> None:
    x, y = pos
    label = LABEL.get(aid, "未知")
    color = COLOR.get(label, "#7f8c8d")
    r = 0.56 if is_root else 0.44
    ax.add_patch(Circle((x, y), r, facecolor=color, edgecolor="white", linewidth=3.4, zorder=6))
    if is_root:
        ax.add_patch(Circle((x, y), r + 0.13, facecolor="none", edgecolor=color,
                            linewidth=1.8, linestyle="--", zorder=5))
    ax.text(x, y, str(aid), ha="center", va="center", fontsize=15 if is_root else 12.5,
            fontweight="bold", color="white", zorder=7)
    score = SCORES.get(aid)
    caption = f"风险分 {score:.4f}" if score is not None else ""
    ax.text(x, y - r - 0.30, caption, ha="center", va="top", fontsize=9.2,
            color="#24344d", zorder=7, bbox=dict(boxstyle="round,pad=0.24", fc="white",
                                                 ec="#cbd5e1", lw=0.8, alpha=0.97))


def draw_edges(ax, fig, pos, edges: list[dict], root: int, label_fn, fontsize: float) -> None:
    pair_index = defaultdict(int)
    pair_count = defaultdict(int)
    for e in edges:
        pair_count[(min(e["src"], e["dst"]), max(e["src"], e["dst"]))] += 1
    max_amount = max(e["amount"] for e in edges) if edges else 1.0
    max_log = math.log1p(max_amount)
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    placed_boxes = []
    for e in edges:
        u, v = e["src"], e["dst"]
        p1, p2 = pos[u], pos[v]
        pair = (min(u, v), max(u, v))
        idx = pair_index[pair]
        pair_index[pair] += 1
        # RAD_SETS 的符号按“弦的上方”约定，绘制时根据边方向翻转，保证同对手的多条边分居弦两侧
        rad = RAD_SETS[pair_count[pair]][idx] * (1.0 if p2[0] >= p1[0] else -1.0)
        incoming = v == root
        color = IN_COLOR if incoming else OUT_COLOR
        width = 1.0 + 2.5 * math.log1p(e["amount"]) / max_log
        ru = 0.66 if u == root else 0.54
        rv = 0.66 if v == root else 0.54
        arrow = FancyArrowPatch(p1, p2, connectionstyle=f"arc3,rad={rad}",
                                arrowstyle="-|>,head_width=3.0,head_length=4.0",
                                mutation_scale=4, lw=width,
                                color=color, shrinkA=ru, shrinkB=rv, zorder=6)
        ax.add_patch(arrow)
        txt = label_fn(e, incoming)
        placed = None
        for t, offset in label_candidates():
            lx, ly = bezier(p1, p2, rad, t)
            tx, ty = bezier_tangent(p1, p2, rad, t)
            norm = math.hypot(tx, ty)
            lx += offset * (-ty / norm)
            ly += offset * (tx / norm)
            angle = math.degrees(math.atan2(ty, tx))
            tb = ax.text(lx, ly, txt, ha="center", va="center", fontsize=fontsize,
                         linespacing=1.12, color="#14222f", zorder=8,
                         rotation=angle, rotation_mode="anchor",
                         path_effects=[pe.withStroke(linewidth=width + 3.5, foreground="white")])
            bb = tb.get_window_extent(renderer)
            if not overlaps_node(bb, inv, pos, root) and not overlaps_text(bb, placed_boxes):
                placed = tb
                break
            tb.remove()
        if placed is None:
            raise RuntimeError(f"无法为账户 {root} 的边放置标签: {txt}")
        placed_boxes.append(placed.get_window_extent(renderer))


def label_candidates() -> list[tuple[float, float]]:
    """沿弧线从中间向两端扩展，必要时沿法线小幅偏移，避免文字重叠。"""
    out = []
    for offset in (0.0, 0.32, -0.32, 0.64, -0.64, 0.98, -0.98, 1.35, -1.35):
        for step in range(12):
            delta = 0.04 * (step + 1)
            if step % 2 == 0:
                out.append((0.50 + delta, offset))
            else:
                out.append((0.50 - delta, offset))
        out.append((0.50, offset))
    return out


def overlaps_node(bb, inv, pos: dict, root: int) -> bool:
    pts = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y0), (bb.x1, bb.y1), (bb.x0, bb.y1)])
    for aid, (cx, cy) in pos.items():
        radius = 0.66 if aid == root else 0.54
        margin = 0.16
        for px, py in pts:
            if math.hypot(px - cx, py - cy) < radius + margin:
                return True
    return False


def overlaps_text(bb, placed_boxes) -> bool:
    pad = 4
    for other in placed_boxes:
        if not (bb.x1 + pad <= other.x0 or other.x1 + pad <= bb.x0
                or bb.y1 + pad <= other.y0 or other.y1 + pad <= bb.y0):
            return True
    return False


def draw_frame(ax, title: str, subtitle: str, structure: str) -> None:
    ax.text(0, 4.30, title, ha="center", va="center", fontsize=16, fontweight="bold", color="#12222f")
    ax.text(0, 3.96, subtitle, ha="center", va="center", fontsize=10.5, color="#52616f")
    box = FancyBboxPatch((-4.0, -2.20), 8.0, 0.66, boxstyle="round,pad=0.08",
                         fc="#eef3f8", ec="#b7c4d2", lw=1.1)
    ax.add_patch(box)
    ax.text(0, -1.87, structure, ha="center", va="center", fontsize=10,
            fontweight="bold", color="#14222f")
    legend = [("确认嫌疑人", COLOR["嫌疑人"]), ("普通账户", COLOR["其它"]),
              ("转入", IN_COLOR), ("转出", OUT_COLOR)]
    lx = -4.0
    for text, color in legend:
        ax.scatter([lx + 0.30], [-3.05], s=90, color=color, edgecolors="white",
                   linewidths=1.4, zorder=6)
        ax.text(lx + 0.52, -3.05, text, ha="left", va="center", fontsize=9.5, color="#233142")
        lx += 0.52 + len(text) * 0.125 + 0.55
    ax.text(-4.0, -3.55, "每笔交易一条有向边 · 箭头 = 资金流向 · 线宽 = 单笔金额（对数）",
            ha="left", va="center", fontsize=9.2, color="#33475b")


def render(root: int, positions: dict, title: str, subtitle: str, structure: str,
           out_path: str, label_fn, fontsize: float = 8.2) -> None:
    edges = raw_edges(root)
    fig, ax = plt.subplots(figsize=(13.2, 8.6), dpi=170)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    ax.set_xlim(-4.8, 4.8)
    ax.set_ylim(-3.95, 4.75)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.canvas.draw()
    draw_edges(ax, fig, positions, edges, root, label_fn, fontsize)
    for aid, p in positions.items():
        draw_node(ax, p, aid, is_root=(aid == root))
    draw_frame(ax, title, subtitle, structure)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path, "edges", len(edges))


def fmt_amount(v: float) -> str:
    if v >= 1e4:
        return f"{v / 1e4:.2f}万"
    return f"{v:,.2f}"


def main() -> None:
    os.makedirs("docs/images", exist_ok=True)
    render(
        4379,
        {4379: (0, 0), 10662: (-2.6, 2.3), 10949: (2.6, 2.3)},
        "账户 4379 · 确认嫌疑人 · 模型风险分 0.9034",
        "测试集排序第 814 名 · 4 笔交易 · 2025-11-19",
        "结构：闭环回流 / 快进快出（15:37:53 转出 3000×2 → 15:38:14 原路回流，间隔 21 秒）",
        "docs/images/appendix_d_4379_loop.png",
        lambda e, inc: f"{'入' if inc else '出'} {e['time'].strftime('%H:%M:%S')} {e['amount']:,.0f}元",
        fontsize=7.0,
    )
    render(
        1740,
        {1740: (0, 0), 3863: (-2.6, 2.3), 7838: (2.6, 2.3)},
        "账户 1740 · 确认嫌疑人 · 模型风险分 0.7460",
        "测试集排序第 2827 名 · 6 笔交易 · 2025-07/11/12",
        "结构：星状汇聚 / 分散入账（3863、7838 各 3 笔转入，合计 18,828.88 元）",
        "docs/images/appendix_d_1740_star.png",
        lambda e, inc: f"{'入' if inc else '出'} {e['time'].strftime('%m-%d')} {e['amount']:,.0f}元",
        fontsize=7.0,
    )
    render(
        7265,
        {7265: (0, 0), 1137: (-2.6, 2.3), 7238: (2.6, 2.3)},
        "账户 7265 · 确认嫌疑人 · 模型风险分 0.5801",
        "测试集排序第 4906 名 · 6 笔交易 · 2025-11/12",
        "结构：星状试探 / 小额分散转出（向 1137、7238 各转出 3 笔，合计 2,633.40 元）",
        "docs/images/appendix_d_7265_star.png",
        lambda e, inc: f"{'入' if inc else '出'} {e['time'].strftime('%m-%d')} {e['amount']:,.0f}元",
        fontsize=7.0,
    )
    render(
        9928,
        {9928: (0, 0), 3842: (-2.6, 2.3), 5584: (2.6, 2.3)},
        "Top30 高风险巡检账户 9928 · 标签：其它 · 模型风险分 0.9797",
        "测试集排序第 73 名 · 8 笔交易 · 2025-07/08",
        "结构：双向闭环回流 / 资金归集（转入 4 笔合计 69 万、转出 4 笔合计 180 万）",
        "docs/images/appendix_d_9928_flow.png",
        lambda e, inc: f"{'入' if inc else '出'} {e['time'].strftime('%m-%d')} {fmt_amount(e['amount'])}",
        fontsize=6.6,
    )


if __name__ == "__main__":
    main()
