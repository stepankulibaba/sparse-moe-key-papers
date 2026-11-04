"""Рисунки корпуса, сравнения метрик и проверки устойчивости."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from .metrics import influence
from .storage import EXPERIMENTS, OUT, ROOT, TABLES, TEXTS, read_json

LABELS = TEXTS["figures"]
NAMES = LABELS["paper_names"]

BLUE, GOLD, INK = "#235789", "#BF8D24", "#192B3C"

SHORT = LABELS["short_names"]


def label(p):
    return NAMES.get(p["id"].split("/")[-1], p["title"].split(":")[0][:33])


def plot_corpus():
    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in papers}
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    with (TABLES / "edges.csv").open(encoding="utf-8-sig", newline="") as f:
        graph.add_edges_from((e["citing"], e["cited"]) for e in csv.DictReader(f))
    top = sorted(papers, key=lambda p: p["rank_pagerank"])[:35]
    sub = graph.subgraph([p["id"] for p in top]).copy()
    connected = max(nx.weakly_connected_components(sub), key=len)
    sub = sub.subgraph(connected).copy()
    highlights = summary["algorithmic_highlights"]
    decisions = read_json(ROOT / "settings.json")["selection"]
    final_choice = bool(decisions.get("final_papers"))
    if final_choice:
        highlights = [p["id"] for p in decisions["final_papers"]]
    positions = {}
    ordered_highlights = sorted(highlights, key=lambda u: (by_id[u]["date"], u))
    for i, u in enumerate(ordered_highlights):
        positions[u] = np.array([-1 + 0.5 * i, -0.65])
    ordinary = sorted((u for u in sub if u not in highlights), key=lambda u: (by_id[u]["date"], u))
    for i, u in enumerate(ordinary):
        row, col = divmod(i, 10)
        count = min(10, len(ordinary) - row * 10)
        positions[u] = np.array([-0.98 + 1.96 * col / max(1, count - 1), 0.12 + 0.4 * row])
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(10, 5.4), layout="constrained")
    nx.draw_networkx_edges(
        sub,
        positions,
        ax=ax,
        arrows=True,
        arrowsize=9,
        width=0.6,
        alpha=0.25,
        edge_color="#626b75",
        node_size=150,
        connectionstyle="arc3,rad=.05",
    )
    nx.draw_networkx_nodes(
        sub, positions, nodelist=ordinary, node_size=75, node_color="#376da4", edgecolors="white", linewidths=0.5, ax=ax
    )
    nx.draw_networkx_nodes(
        sub,
        positions,
        nodelist=highlights,
        node_size=210,
        node_color="#dfaa38",
        node_shape="s",
        edgecolors="#242424",
        linewidths=0.8,
        ax=ax,
    )
    for u in highlights:
        x, y = positions[u]
        ax.annotate(
            label(by_id[u]),
            (x, y),
            xytext=(0, -22),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.87, "pad": 1},
        )
    ax.set_title(
        LABELS["graph_title"].format(nodes=sub.number_of_nodes(), edges=sub.number_of_edges()),
        loc="left",
        fontsize=14,
        pad=16,
    )
    ax.text(
        0,
        -0.04,
        LABELS["graph_direction"] + (LABELS["selected_papers"] if final_choice else LABELS["algorithmic_top5"]),
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_axis_off()
    ax.margins(0.18)
    ax.set_ylim(-1.15, 1.2)
    fig.savefig(OUT / "citation_graph.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "citation_graph.svg", bbox_inches="tight")
    plt.close(fig)
    # Exact rank values; positions are ordinal with stable ID tie-breaking.
    shown = sorted(papers, key=lambda p: p["rank_pagerank"])[:12]
    arr = np.array([[p["rank_local_citations"], p["rank_pagerank"], p["rank_global_citations"]] for p in shown])
    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    ax.imshow(arr, cmap="Blues_r", vmin=1, vmax=max(20, int(arr.max())), aspect="auto")
    ax.set_yticks(range(len(shown)), [label(p) for p in shown])
    ax.set_xticks([0, 1, 2], ["Цитирования в корпусе", "PageRank", "Цитирования OpenAlex"])
    ax.tick_params(length=0, pad=9)
    for i in range(len(shown)):
        for j in range(3):
            ax.text(
                j,
                i,
                str(arr[i, j]),
                ha="center",
                va="center",
                color="white" if arr[i, j] < arr.max() * 0.42 else "#202020",
                fontsize=11,
            )
    ax.set_title(LABELS["ranking_title"], loc="left", fontsize=12, pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(OUT / "rank_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_metrics():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    rows = json.loads((TABLES / "custom_rankings.json").read_text(encoding="utf-8"))
    chosen = sorted(rows, key=lambda r: r["rank_branching_descendants"])[:12]
    fields = [
        "rank_local_citations",
        "rank_pagerank",
        "rank_katz_paths",
        "rank_unique_descendants",
        "rank_branching_descendants",
    ]
    arr = np.array([[r[f] for f in fields] for r in chosen])
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    ax.imshow(arr, cmap="Blues_r", vmin=1, vmax=max(20, int(arr.max())), aspect="auto")
    labels = [label(r) for r in chosen]
    labels = [s if not s.startswith("Taming") else "Stochastic Experts" for s in labels]
    ax.set_yticks(range(len(chosen)), labels)
    ax.set_xticks(range(5), ["Цитирования", "PageRank", "Katz", "Охват U", "Ветви B"])
    ax.tick_params(length=0, pad=9)
    for i in range(len(chosen)):
        for j in range(5):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="white" if arr[i, j] < 8 else "#202020")
    ax.set_title(LABELS["metric_comparison_title"], loc="left", fontsize=13, pad=15)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(OUT / "custom_rank_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    specs = [
        ("Цепочка", [("c", "b"), ("b", "a"), ("a", "v")], {"v": (0, 0), "a": (0, 1), "b": (0, 2), "c": (0, 3)}),
        ("Звезда", [("a", "v"), ("b", "v"), ("c", "v")], {"v": (0, 0), "a": (-1, 2), "b": (0, 2.6), "c": (1, 2)}),
        (
            "Ромб",
            [("a", "v"), ("b", "v"), ("c", "a"), ("c", "b")],
            {"v": (0, 0), "a": (-0.8, 1.4), "b": (0.8, 1.4), "c": (0, 3)},
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4.6), layout="constrained")
    for ax, (title, edges, positions) in zip(axes, specs):
        g = nx.DiGraph(edges)
        s = influence(g)["v"]
        nx.draw_networkx(
            g,
            positions,
            ax=ax,
            node_color=["#dfaa38" if u == "v" else "#376da4" for u in g],
            node_size=600,
            font_color="white",
            font_weight="bold",
            edgecolors="#253341",
            arrowsize=18,
            width=1.2,
        )
        ax.set_title(title, fontsize=14)
        ax.set_ylim(-1.1, 3.5)
        ax.set_xlim(-1.7, 1.7)
        ax.text(
            0,
            -0.7,
            f"c={s['direct_children']}   K={s['katz_paths']:.2f}\nU={s['unique_descendants']:.2f}   B={s['branching_descendants']:.2f}",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_axis_off()
    fig.suptitle(LABELS["counterexamples_title"], fontsize=13)
    fig.savefig(OUT / "metric_counterexamples.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def short(uid, title):
    full_title = "A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
    short_title = title.split(":")[0].replace(full_title, "").strip()
    return SHORT.get(uid.rsplit("/", 1)[-1], short_title)


def save(fig, name):
    fig.savefig(EXPERIMENTS / (name + ".png"), dpi=200, bbox_inches="tight")
    fig.savefig(EXPERIMENTS / (name + ".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison(selected, labels, metrics, lookup, cm):
    """Сравнить места лидеров по четырём метрикам."""
    data = np.array([[lookup["baseline", m, r["id"]] for m in metrics] for r in selected])
    fig, ax = plt.subplots(figsize=(12, 5.7))
    ax.imshow(np.maximum(0, 11 - data), cmap=cm, vmin=0, vmax=10, aspect="auto")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks(range(4), ["Цитирования c", "PageRank", "Ветвление B", "Дробный F"])
    ax.xaxis.tick_top()
    for i in range(len(labels)):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{data[i, j]:g}",
                ha="center",
                va="center",
                fontsize=15,
                color="white" if data[i, j] <= 3 else INK,
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=12)
    fig.subplots_adjust(left=0.26, right=0.98, top=0.86, bottom=0.09)
    fig.text(0.26, 0.02, LABELS["ranking_caption"], fontsize=11)
    save(fig, "ranking_comparison")


def plot_stability(selected, labels, stability, cm):
    """Показать частоты попадания в первую пятёрку при потере данных."""
    scenarios = ["edge_10", "edge_30", "edge_50", "bibliography_30"]
    freq = {(r["scenario"], r["metric"], r["id"]): float(r["top5_frequency"]) * 100 for r in stability}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.3), sharey=True)
    for ax, metric, label in zip(axes, ["pagerank", "branching"], ["PageRank", "Ветвление B"]):
        values = np.array([[freq[s, metric, r["id"]] for s in scenarios] for r in selected])
        ax.imshow(values, cmap=cm, vmin=0, vmax=100, aspect="auto")
        ax.set_title(label, fontsize=17, pad=15, weight="bold")
        ax.set_yticks(range(8), labels)
        ax.set_xticks(range(4), ["10%\nссылок", "30%\nссылок", "50%\nссылок", "30%\nбиблиогр."])
        ax.tick_params(length=0, pad=10)
        for i in range(8):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]:.1f}",
                    ha="center",
                    va="center",
                    color="white" if values[i, j] >= 65 else INK,
                    fontsize=12,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.88, bottom=0.17, wspace=0.10)
    fig.text(0.20, 0.015, LABELS["stability_caption"], fontsize=10)
    save(fig, "stability_top5")


def plot_ablations(selected, labels, lookup, cm):
    """Сравнить исходный граф с тремя способами исключения связей."""
    variants = ["baseline", "no_observed_shared_authors", "openalex_only", "no_review_outgoing_votes"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
    for ax, metric, label in zip(axes, ["pagerank", "branching"], ["PageRank", "Ветвление B"]):
        values = np.array([[lookup[v, metric, r["id"]] for v in variants] for r in selected[:6]])
        ax.imshow(np.maximum(0, 11 - values), cmap=cm, vmin=0, vmax=10, aspect="auto")
        ax.set_title(label, fontsize=17, pad=15, weight="bold")
        ax.set_yticks(range(6), labels[:6])
        ax.set_xticks(range(4), ["Исходный", "Без общих\nавторов*", "Только\nOpenAlex", "Без ссылок\nиз обзоров"])
        ax.tick_params(length=0, pad=10)
        for i in range(6):
            for j in range(4):
                ax.text(
                    j,
                    i,
                    f"{values[i,j]:g}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color="white" if values[i, j] <= 3 else INK,
                )
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.20, right=0.99, top=0.87, bottom=0.20, wspace=0.10)
    fig.text(0.20, 0.025, LABELS["ablation_caption"], fontsize=11)
    save(fig, "ablation_comparison")


def plot_correlations(summary):
    """Сравнить корреляции на всём корпусе и среди двадцати лидеров."""
    c = summary["correlations"][1:]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    y = np.arange(3)
    for offset, key, color, hatch, label in [
        (-0.17, "vs_local_all140", BLUE, None, "Все 140 статей"),
        (0.17, "vs_local_base_pagerank_top20", GOLD, "//", "20 лидеров исходного PageRank"),
    ]:
        values = [r[key] for r in c]
        ax.barh(y + offset, values, height=0.28, color=color, hatch=hatch, label=label)
        for yy, v in zip(y + offset, values):
            ax.text(
                v - 0.025,
                yy,
                f"{v:.3f}",
                color="white" if color == BLUE else INK,
                bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5},
                ha="right",
                va="center",
                weight="bold",
                fontsize=12,
            )
    ax.set_yticks(y, ["PageRank", "Ветвление B", "Дробный F"])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(LABELS["correlation_axis"])
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, frameon=False)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#BBBBBB")
    ax.spines["bottom"].set_color("#BBBBBB")
    ax.tick_params(axis="y", length=0, pad=10)
    fig.subplots_adjust(left=0.18, right=0.98, top=0.80, bottom=0.20)
    save(fig, "correlation_scope")


def plot_experiments():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    with (EXPERIMENTS / "rankings.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    with (EXPERIMENTS / "stability.csv").open(encoding="utf-8-sig") as f:
        stability = list(csv.DictReader(f))
    summary = json.loads((EXPERIMENTS / "summary.json").read_text(encoding="utf-8"))
    baseline = [r for r in rows if r["variant"] == "baseline" and r["metric"] == "pagerank"]
    baseline.sort(key=lambda r: (float(r["midrank"]), r["id"]))
    selected = baseline[:8]
    lookup = {(r["variant"], r["metric"], r["id"]): float(r["midrank"]) for r in rows}
    cm = LinearSegmentedColormap.from_list("quiet_blue", ["#FFFFFF", "#DCE8F2", BLUE])
    labels = [short(r["id"], r["title"]) for r in selected]
    metrics = ["local", "pagerank", "branching", "fractional"]
    plot_rank_comparison(selected, labels, metrics, lookup, cm)
    plot_stability(selected, labels, stability, cm)
    plot_ablations(selected, labels, lookup, cm)
    plot_correlations(summary)
    print("Exported four PNG/SVG figures to", EXPERIMENTS)


def build_figures(include_stress=True):
    for plot in [plot_corpus, plot_metrics] + ([plot_experiments] if include_stress else []):
        plt.rcdefaults()
        plot()
