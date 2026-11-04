"""Охват U, разнообразие ветвей B, Katz и текстовый пилот R.

U считает каждую достижимую цитирующую статью один раз. B уменьшает оценку,
если весь охват сосредоточен в одной ветви. Katz считает все пути.
R использует только сохранённые контексты с разметкой ассистента.
"""

import json
from collections import Counter, defaultdict

import networkx as nx
from scipy.stats import spearmanr

from .storage import DATA, ROOT, TABLES, TEXTS, load_contexts, read_csv, read_json, write_csv


def influence(graph, beta=0.5):
    """Посчитать U, B и Katz на графе без циклов.

    Для каждой оцениваемой статьи идём против стрелок цитирования.
    supports хранит прямые цитирующие статьи на кратчайших путях,
    а mass делит голос дальнего потомка поровну между этими ветвями.
    """
    if not 0 < beta < 1:
        raise ValueError("beta must be in (0,1)")
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Katz path propagation requires the chronological DAG")

    katz = dict.fromkeys(graph, 0.0)
    for u in nx.topological_sort(graph):
        for v in graph.successors(u):
            katz[v] += 1 + beta * katz[u]

    result = {}
    reverse = graph.reverse(copy=False)
    for v in sorted(graph):
        distance = nx.single_source_shortest_path_length(reverse, v)
        supports = {}
        mass = defaultdict(float)
        unique = 0.0
        for u, d in sorted(distance.items(), key=lambda x: (x[1], x[0])):
            if d == 0:
                continue
            if d == 1:
                supports[u] = {u}
            else:
                parents = (supports[t] for t in graph.successors(u) if distance.get(t) == d - 1)
                supports[u] = set().union(*parents)
            weight = beta ** (d - 1)
            unique += weight
            for first_branch in supports[u]:
                mass[first_branch] += weight / len(supports[u])

        concentration = sum((m / unique) ** 2 for m in mass.values()) if unique else 0.0
        diverse = unique * (1 - concentration) if unique else 0.0
        result[v] = {
            "unique_descendants": unique,
            "branching_descendants": max(0.0, diverse),
            "effective_branches": 1 / concentration if concentration else 0.0,
            "katz_paths": katz[v],
            "direct_children": graph.in_degree(v),
            "reachable_descendants": len(distance) - 1,
            "largest_branch_mass": max(mass.values(), default=0.0),
        }
    return result


def build_structural_scores():
    """Исключить рёбра против хронологии и сравнить три структурные оценки."""
    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in papers}

    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    removed = []
    for e in read_csv(TABLES / "edges.csv"):
        u, v = e["citing"], e["cited"]
        if by_id[u]["date"] < by_id[v]["date"]:
            removed.append(e)
        else:
            graph.add_edge(u, v)

    base = influence(graph)
    fields = [
        "id",
        "title",
        "year",
        "global_citations",
        "local_citations",
        "pagerank",
        "rank_global_citations",
        "rank_local_citations",
        "rank_pagerank",
    ]
    rows = [{k: p[k] for k in fields} | base[p["id"]] for p in papers]
    metrics = ["unique_descendants", "branching_descendants", "katz_paths"]
    for metric in metrics:
        for i, r in enumerate(sorted(rows, key=lambda r: (-r[metric], r["id"])), 1):
            r["rank_" + metric] = i

    sensitivity = []
    for beta in [0.25, 0.5, 0.75]:
        variant = influence(graph, beta)
        for metric in metrics:
            top = sorted(graph, key=lambda u: (-variant[u][metric], u))[:5]
            base_top = sorted(graph, key=lambda u: (-base[u][metric], u))[:5]
            sensitivity.append(
                {
                    "beta": beta,
                    "metric": metric,
                    "top5_ids": top,
                    "top5_overlap_with_beta_half": len(set(top) & set(base_top)),
                }
            )

    correlations = {}
    for metric in metrics:
        correlations[metric] = {
            comparison: float(spearmanr([r[metric] for r in rows], [r[comparison] for r in rows]).statistic)
            for comparison in ["local_citations", "pagerank"]
        }

    summary = {
        "status": "calculated_structural_proxies_not_verified_reuse",
        "vertices": len(graph),
        "chronological_edges": len(graph.edges),
        "removed_date_anomaly_edges": len(removed),
        "beta": 0.5,
        "correlations": correlations,
        "sensitivity": sensitivity,
        "top10": {
            metric: [r["id"] for r in sorted(rows, key=lambda r: r["rank_" + metric])[:10]] for metric in metrics
        },
        **TEXTS["structural_metrics"],
    }

    write_csv(TABLES / "custom_rankings.csv", sorted(rows, key=lambda r: r["rank_branching_descendants"]))
    (TABLES / "custom_rankings.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    (ROOT / "output/custom_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    nx.write_graphml(graph, ROOT / "output/chronological_graph.graphml")

    for metric in metrics:
        print("\n" + metric)
        for r in sorted(rows, key=lambda r: r["rank_" + metric])[:8]:
            print(r["rank_" + metric], round(r[metric], 3), r["title"])


def build_semantic_scores():
    """Суммировать разметку по парам статей, учитывая сильнейший контекст пары."""
    contexts = load_contexts()
    by_context = {r["context_id"]: r for r in contexts}
    config = read_json(ROOT / "settings.json")["context_review"]
    labels = {i: label for label, ids in config["labels"].items() for i in ids}
    if len(labels) != sum(len(ids) for ids in config["labels"].values()):
        raise ValueError("A context has several review labels in settings.json")

    flagged = {r["context_id"] for r in contexts if r["auto_label"] in ["use_signal", "extension_signal"]}
    if set(labels) != flagged:
        unreviewed = sorted(flagged - set(labels))
        stale = sorted(set(labels) - flagged)
        raise ValueError(f"Review labels do not match flagged contexts: unreviewed={unreviewed}, stale={stale}")

    reviewed = []
    for context_id, label in labels.items():
        r = by_context[context_id]
        reviewed.append(
            r
            | {
                "reviewed_label": label,
                "reviewer": "assistant_context_review",
                "review_note": config["notes"].get(context_id, ""),
            }
        )

    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in papers}
    coverage = read_csv(DATA / "context_coverage.csv")

    weights = {"extension": 2.0, "method_use": 1.0, "component_use": 0.5}
    edge_evidence = defaultdict(list)
    for r in reviewed:
        edge_evidence[r["citing"], r["cited"]].append(r)

    accepted_edges = []
    scores = defaultdict(float)
    counts = defaultdict(Counter)
    for (u, v), evidence in sorted(edge_evidence.items()):
        strongest = max(evidence, key=lambda r: weights.get(r["reviewed_label"], 0))
        score = weights.get(strongest["reviewed_label"], 0)
        if score:
            scores[v] += score
            counts[v][strongest["reviewed_label"]] += 1
            accepted_edges.append(
                {
                    "citing": u,
                    "cited": v,
                    "weight": score,
                    "label": strongest["reviewed_label"],
                    "context_id": strongest["context_id"],
                    "source_url": strongest["source_url"],
                    "reviewer": "assistant_context_review",
                    "chronological_anomaly": by_id[u]["date"] < by_id[v]["date"],
                }
            )

    rows = []
    for p in papers:
        incoming = [e for e in coverage if e["cited"] == p["id"]]
        rows.append(
            {
                "id": p["id"],
                "title": p["title"],
                "pilot_reuse_score": scores[p["id"]],
                "extension_edges": counts[p["id"]]["extension"],
                "method_use_edges": counts[p["id"]]["method_use"],
                "component_use_edges": counts[p["id"]]["component_use"],
                "incoming_citation_edges": len(incoming),
                "incoming_edges_with_contexts": sum(int(e["contexts"]) > 0 for e in incoming),
                "reviewed_incoming_edges": sum(v == p["id"] for u, v in edge_evidence),
                "rank_pagerank": p["rank_pagerank"],
                "rank_local_citations": p["rank_local_citations"],
                "status": "partial_assistant_review_not_complete_true_reuse_count",
            }
        )
    for i, r in enumerate(sorted(rows, key=lambda r: (-r["pilot_reuse_score"], r["id"])), 1):
        r["rank_pilot_reuse_score"] = i

    sensitivity = []
    for extension_weight in [1.0, 1.5, 2.0]:
        for component_weight in [0.0, 0.5, 1.0]:
            values = {
                r["id"]: extension_weight * r["extension_edges"]
                + r["method_use_edges"]
                + component_weight * r["component_use_edges"]
                for r in rows
            }
            sensitivity.append(
                {
                    "extension_weight": extension_weight,
                    "component_weight": component_weight,
                    "top5_ids": sorted(values, key=lambda u: (-values[u], u))[:5],
                }
            )

    positive_contexts = sum(r["reviewed_label"] in weights for r in reviewed)
    summary = {
        "contexts_extracted": len(contexts),
        "citation_edges": len(coverage),
        "edges_with_at_least_one_context": sum(int(e["contexts"]) > 0 for e in coverage),
        "reviewed_flagged_contexts": len(reviewed),
        "reviewed_unique_edges": len(edge_evidence),
        "positive_method_or_component_or_extension_contexts": positive_contexts,
        "accepted_reuse_edges": len(accepted_edges),
        "positive_scoring_papers": sum(r["pilot_reuse_score"] > 0 for r in rows),
        "review_label_counts": dict(Counter(labels.values())),
        "flagged_sample_adoption_fraction": positive_contexts / len(reviewed),
        "score": TEXTS["semantic_metrics"]["score"],
        "sensitivity": sensitivity,
        "limits": TEXTS["semantic_metrics"]["limits"],
    }

    write_csv(TABLES / "reviewed_citation_contexts.csv", reviewed)
    write_csv(TABLES / "reuse_edges.csv", accepted_edges)
    write_csv(TABLES / "semantic_rankings.csv", sorted(rows, key=lambda r: r["rank_pilot_reuse_score"]))
    (ROOT / "output/semantic_metrics_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    display_summary = {key: value for key, value in summary.items() if key not in ["sensitivity", "limits"]}
    print(json.dumps(display_summary, ensure_ascii=False, indent=2))
    for r in sorted(rows, key=lambda r: r["rank_pilot_reuse_score"])[:10]:
        print(r["rank_pilot_reuse_score"], r["pilot_reuse_score"], r["title"])


if __name__ == "__main__":
    build_structural_scores()
    build_semantic_scores()
