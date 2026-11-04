"""Построить граф и базовые рейтинги для зафиксированного корпуса.

Отбор и объединение версий сохранены в data/corpus.json и scope_audit.csv.
Ребро u -> v означает, что u цитирует v. Все численные оценки пересчитываются.
"""

import json

import networkx as nx
from scipy.stats import spearmanr

from .storage import (
    DATA,
    OUT,
    TABLES,
    bibliography_edges,
    normalize,
    read_csv,
    read_json,
    write_csv,
)

AS_OF = "2026-11-04"


def load_corpus():
    """Прочитать отобранные работы, версии и журнал тематического отбора."""
    source_summary = read_json(DATA / "provenance.json")["original_corpus_summary"]
    evidence_fields = {
        "observed_author_ids",
        "incomplete_author_metadata",
        "author_evidence_sources",
    }
    rows = [
        {k: v for k, v in paper.items() if k not in evidence_fields}
        for paper in read_json(DATA / "corpus.json")
    ]
    rows.sort(key=lambda paper: normalize(paper["title"]))
    alias = {
        version: paper["id"]
        for paper in rows
        for version in paper["version_ids"].split(";")
    }
    audit = read_csv(DATA / "scope_audit.csv")
    for row in audit:
        row["accepted"] = row["accepted"] == "True"
        row["has_abstract"] = row["has_abstract"] == "True"
    return rows, alias, audit, source_summary


def build_graph(rows, alias):
    """Объединить ссылки OpenAlex и точные совпадения названий в библиографиях."""
    graph = nx.DiGraph()
    graph.add_nodes_from(r["id"] for r in rows)
    by_id = {r["id"]: r for r in rows}
    temporal = []
    for r in rows:
        for ref in r["reference_ids"]:
            target = alias.get(ref)
            if target and target != r["id"]:
                graph.add_edge(r["id"], target)
    metadata_graph = graph.copy()
    provenance = {(u, v): {"openalex_referenced_works"} for u, v in graph.edges}
    pdf_edges = bibliography_edges(rows)
    write_csv(TABLES / "pdf_edges.csv", pdf_edges)
    for edge in pdf_edges:
        u = alias.get(edge["citing"], edge["citing"])
        v = alias.get(edge["cited"], edge["cited"])
        if u in graph and v in graph and u != v:
            graph.add_edge(u, v)
            provenance.setdefault((u, v), set()).add(edge["source"])
    for u, v in graph.edges:
        if by_id[u]["date"] < by_id[v]["date"]:
            temporal.append(
                {
                    "citing": u,
                    "cited": v,
                    "citing_date": by_id[u]["date"],
                    "cited_date": by_id[v]["date"],
                }
            )
    return graph, metadata_graph, provenance, temporal


def compare_pagerank_variants(graph, metadata_graph, rows, highlights, temporal):
    """Проверить влияние затухания, обзоров, источника связей и хронологии."""
    sensitivity = []
    for alpha in (0.5, 0.7, 0.85, 0.95):
        score = nx.pagerank(graph, alpha=alpha, tol=1e-12, max_iter=1000)
        top = sorted(rows, key=lambda r: (-score[r["id"]], r["id"]))[:5]
        sensitivity.append(
            {
                "variant": f"alpha={alpha}",
                "top5_ids": [r["id"] for r in top],
                "overlap_with_base": len(set(highlights) & {r["id"] for r in top}),
            }
        )
    no_reviews = graph.subgraph([r["id"] for r in rows if not r["is_review"]]).copy()
    scores = nx.pagerank(no_reviews, alpha=0.85, tol=1e-12, max_iter=1000)
    no_review_top = sorted(scores, key=lambda u: (-scores[u], u))[:5]
    sensitivity.append(
        {
            "variant": "exclude_title_marked_reviews",
            "top5_ids": no_review_top,
            "overlap_with_base": len(set(highlights) & set(no_review_top)),
        }
    )
    metadata_scores = nx.pagerank(metadata_graph, alpha=0.85, tol=1e-12, max_iter=1000)
    metadata_top = sorted(metadata_scores, key=lambda u: (-metadata_scores[u], u))[:5]
    sensitivity.append(
        {
            "variant": "openalex_edges_only",
            "top5_ids": metadata_top,
            "overlap_with_base": len(set(highlights) & set(metadata_top)),
        }
    )
    chronological_graph = graph.copy()
    chronological_graph.remove_edges_from((e["citing"], e["cited"]) for e in temporal)
    chrono_score = nx.pagerank(
        chronological_graph, alpha=0.85, tol=1e-12, max_iter=1000
    )
    chrono_top = sorted(chrono_score, key=lambda u: (-chrono_score[u], u))[:5]
    sensitivity.append(
        {
            "variant": "remove_date_anomalies",
            "top5_ids": chrono_top,
            "overlap_with_base": len(set(highlights) & set(chrono_top)),
        }
    )
    return sensitivity


def build():
    TABLES.mkdir(parents=True, exist_ok=True)
    rows, alias, audit, source_summary = load_corpus()
    graph, metadata_graph, provenance, temporal = build_graph(rows, alias)
    by_id = {row["id"]: row for row in rows}
    pr = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)
    for r in rows:
        r["local_citations"] = graph.in_degree(r["id"])
        r["pagerank"] = pr[r["id"]]
    metrics = ["global_citations", "local_citations", "pagerank", "citations_per_year"]
    for metric in metrics:
        for rank, r in enumerate(sorted(rows, key=lambda r: (-r[metric], r["id"])), 1):
            r["rank_" + metric] = rank
    main = sorted(rows, key=lambda r: r["rank_pagerank"])
    # Highlighted nodes are an algorithmic top five, not the student's final list.
    highlights = [r["id"] for r in main[:5]]
    sensitivity = compare_pagerank_variants(
        graph, metadata_graph, rows, highlights, temporal
    )
    summary = {
        "as_of_age_denominator": AS_OF,
        "raw_unique": source_summary["raw_unique"],
        "alias_records_examined": source_summary["alias_records_examined"],
        "accepted_records": sum(r["accepted"] for r in audit),
        "candidates": len(rows),
        "merged_records": sum(r["version_count"] - 1 for r in rows),
        "edges": graph.number_of_edges(),
        "openalex_edges": metadata_graph.number_of_edges(),
        "supplementary_edges": graph.number_of_edges()
        - metadata_graph.number_of_edges(),
        "isolates": nx.number_of_isolates(graph),
        "weak_components": nx.number_weakly_connected_components(graph),
        "missing_abstracts": sum(not r["abstract"] for r in rows),
        "zero_reference_records": sum(not r["reference_ids"] for r in rows),
        "review_count": sum(r["is_review"] for r in rows),
        "temporal_anomalies": len(temporal),
        "is_dag": nx.is_directed_acyclic_graph(graph),
        "pagerank_sum": sum(pr.values()),
        "algorithmic_highlights": highlights,
        "sensitivity": sensitivity,
        "spearman_global_vs_pagerank": float(
            spearmanr(
                [r["global_citations"] for r in rows], [r["pagerank"] for r in rows]
            ).statistic
        ),
        "spearman_local_vs_pagerank": float(
            spearmanr(
                [r["local_citations"] for r in rows], [r["pagerank"] for r in rows]
            ).statistic
        ),
        "top5_local_overlap_pagerank": len(
            set(highlights)
            & {
                r["id"]
                for r in sorted(rows, key=lambda r: r["rank_local_citations"])[:5]
            }
        ),
        "local_top5_boundary_value": sorted(
            (r["local_citations"] for r in rows), reverse=True
        )[4],
        "local_top5_boundary_tie_count": sum(
            r["local_citations"]
            == sorted((q["local_citations"] for q in rows), reverse=True)[4]
            for r in rows
        ),
        "top5_global_overlap_pagerank": len(
            set(highlights)
            & {
                r["id"]
                for r in sorted(rows, key=lambda r: r["rank_global_citations"])[:5]
            }
        ),
    }
    assert (
        len(rows) >= 100
    ), f"Only {len(rows)} eligible candidates; extend documented retrieval before reporting"
    assert len(by_id) == len(rows)
    assert abs(summary["pagerank_sum"] - 1) < 1e-9
    write_csv(TABLES / "corpus_audit.csv", sorted(audit, key=lambda r: r["id"]))
    csv_fields = [k for k in rows[0] if k != "reference_ids"]
    write_csv(TABLES / "papers.csv", main, csv_fields)
    write_csv(
        TABLES / "edges.csv",
        [
            {"citing": u, "cited": v, "sources": ";".join(sorted(provenance[(u, v)]))}
            for u, v in sorted(graph.edges)
        ],
        ["citing", "cited", "sources"],
    )
    write_csv(
        TABLES / "temporal_anomalies.csv",
        temporal,
        ["citing", "cited", "citing_date", "cited_date"],
    )
    write_csv(
        TABLES / "duplicates.csv",
        [r for r in rows if r["version_count"] > 1],
        ["id", "title", "version_count", "version_ids"],
    )
    (TABLES / "papers.json").write_text(
        json.dumps(main, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    nx.write_graphml(graph, OUT / "citation_graph.graphml")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nTOP 15: rank PR / global / local | title")
    for r in main[:15]:
        print(
            r["rank_pagerank"],
            r["rank_global_citations"],
            r["local_citations"],
            "|",
            r["title"],
        )
    return rows, graph, summary


if __name__ == "__main__":
    build()
