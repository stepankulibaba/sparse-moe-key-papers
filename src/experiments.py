"""Проверки устойчивости на фиксированном корпусе.

Все кандидаты сохраняются. Удаление библиографии убирает исходящие ссылки,
а сама статья и входящие в неё ссылки остаются. Внешние запросы не выполняются.
"""

import hashlib
import json

import networkx as nx
import numpy as np
from scipy.stats import rankdata, spearmanr

from .metrics import influence
from .storage import DATA, ROOT, TABLES, TEXTS, read_csv, read_json, write_csv

OUT = ROOT / "output" / "robustness"
METRICS = ["local", "pagerank", "branching", "fractional"]
SEED = 20261104
REPEATS = 500


def load_inputs():
    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    edges = read_csv(TABLES / "edges.csv")
    return papers, edges


def fractional(graph):
    """Разделить один голос статьи между её наблюдаемыми ссылками в корпусе."""
    result = dict.fromkeys(graph, 0.0)
    for u in graph:
        degree = graph.out_degree(u)
        if degree:
            for v in graph.successors(u):
                result[v] += 1 / degree
    return result


def ranking(values):
    """Средние ранги и равные доли места при ничьей на границе первой пятёрки."""
    values = np.round(np.asarray(values, dtype=float), 12)
    ranks = rankdata(-values, method="average")

    threshold = np.sort(values)[-min(5, len(values))]
    credit = (values > threshold).astype(float)
    boundary = values == threshold
    credit[boundary] = (min(5, len(values)) - credit.sum()) / boundary.sum()
    return ranks, credit


def scores(graph, ids, papers):
    """Строки матрицы: цитируемость, PageRank, ветви B и дробный подсчёт F."""
    chrono = graph.copy()
    chrono.remove_edges_from([(u, v) for u, v in graph.edges if papers[u]["date"] < papers[v]["date"]])
    structural = influence(chrono)
    pr = nx.pagerank(graph, alpha=0.85, tol=1e-12, max_iter=1000)
    f = fractional(graph)
    return np.array(
        [
            [graph.in_degree(u) for u in ids],
            [pr[u] for u in ids],
            [structural[u]["branching_descendants"] for u in ids],
            [f[u] for u in ids],
        ],
        dtype=float,
    )


def author_evidence(papers):
    """Объединить наблюдаемые ID авторов по всем версиям и сохранённым ответам.

    Так находятся только наблюдаемые общие ID. Отсутствие совпадения не доказывает независимость.
    Исходные файлы и ID версий остаются в аудите.
    """
    frozen = {p["id"]: p for p in read_json(DATA / "corpus.json")}
    authors = {p["id"]: set(frozen[p["id"]]["observed_author_ids"]) for p in papers}
    rows = [
        {
            "id": p["id"],
            "title": p["title"],
            "observed_author_count": len(authors[p["id"]]),
            "author_ids": ";".join(sorted(authors[p["id"]])),
            "missing_or_potentially_truncated": frozen[p["id"]]["incomplete_author_metadata"],
            "raw_sources": ";".join(frozen[p["id"]]["author_evidence_sources"]),
        }
        for p in papers
    ]
    return authors, rows


def run_simulations(ids, edges, by_id, base_ranks, base_credits):
    """Удалить связи или целые библиографии, сохранив всех кандидатов."""
    scenarios = [
        ("edge_10", "edge", 0.1),
        ("edge_30", "edge", 0.3),
        ("edge_50", "edge", 0.5),
        ("bibliography_30", "bibliography", 0.3),
    ]
    source_ids = sorted({u for u, _ in edges})
    source_index = {u: i for i, u in enumerate(source_ids)}
    edge_source_index = np.array([source_index[u] for u, _ in edges])
    rng = np.random.default_rng(SEED)

    all_ranks, all_credits, all_masks = [], [], []
    stability, trial_rows = [], []
    for scenario, mode, probability in scenarios:
        if mode == "edge":
            masks = rng.random((REPEATS, len(edges))) >= probability
        else:
            source_mask = rng.random((REPEATS, len(source_ids))) >= probability
            masks = source_mask[:, edge_source_index]

        ranks = np.empty((REPEATS, len(METRICS), len(ids)))
        credits = np.empty_like(ranks)
        for trial, mask in enumerate(masks):
            perturbed = nx.DiGraph()
            perturbed.add_nodes_from(ids)
            perturbed.add_edges_from(e for e, keep in zip(edges, mask) if keep)
            matrix = scores(perturbed, ids, by_id)
            for m in range(len(METRICS)):
                ranks[trial, m], credits[trial, m] = ranking(matrix[m])
            trial_rows.append(
                {
                    "scenario": scenario,
                    "trial": trial,
                    "retained_edges": int(mask.sum()),
                    "retained_bibliographies": sum(perturbed.out_degree(u) > 0 for u in source_ids),
                }
            )

        for m, metric in enumerate(METRICS):
            for j, uid in enumerate(ids):
                stability.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "id": uid,
                        "title": by_id[uid]["title"],
                        "base_midrank": base_ranks[m, j],
                        "base_top5_credit": base_credits[m, j],
                        "top5_frequency": credits[:, m, j].mean(),
                        "monte_carlo_se": credits[:, m, j].std(ddof=1) / np.sqrt(REPEATS),
                        "rank_p05": np.quantile(ranks[:, m, j], 0.05),
                        "rank_median": np.median(ranks[:, m, j]),
                        "rank_p95": np.quantile(ranks[:, m, j], 0.95),
                    }
                )

        all_ranks.append(ranks)
        all_credits.append(credits)
        all_masks.append(masks)
        print(f"{scenario}: {REPEATS} trials; mean retained edges {masks.sum(axis=1).mean():.1f}", flush=True)

    write_csv(OUT / "stability.csv", stability)
    write_csv(OUT / "trials.csv", trial_rows)
    np.savez_compressed(
        OUT / "simulations.npz",
        ids=np.array(ids),
        metrics=np.array(METRICS),
        scenarios=np.array([s[0] for s in scenarios]),
        edges=np.array(edges),
        ranks=np.array(all_ranks),
        top5_credit=np.array(all_credits),
        edge_keep=np.array(all_masks),
    )
    return scenarios, source_ids


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    papers, edge_rows = load_inputs()
    by_id = {p["id"]: p for p in papers}
    ids = sorted(by_id)
    edges = [(e["citing"], e["cited"]) for e in edge_rows]

    graph = nx.DiGraph()
    graph.add_nodes_from(ids)
    graph.add_edges_from(edges)
    base = scores(graph, ids, by_id)
    base_ranks = np.array([ranking(v)[0] for v in base])
    base_credits = np.array([ranking(v)[1] for v in base])

    authors, author_rows = author_evidence(papers)
    uncertain_authors = {r["id"] for r in author_rows if r["missing_or_potentially_truncated"]}
    write_csv(OUT / "author_audit.csv", author_rows)
    audit = []
    for e in edge_rows:
        shared = sorted(authors[e["citing"]] & authors[e["cited"]])
        both_observed = e["citing"] not in uncertain_authors and e["cited"] not in uncertain_authors
        audit.append(
            e
            | {
                "shared_author_ids": ";".join(shared),
                "observed_overlap": bool(shared),
                "both_bylines_observed_without_truncation_flag": both_observed,
            }
        )
    write_csv(OUT / "edge_author_audit.csv", audit)

    variants = {"baseline": graph}
    no_shared = graph.copy()
    no_shared.remove_edges_from([(e["citing"], e["cited"]) for e in audit if e["observed_overlap"]])
    variants["no_observed_shared_authors"] = no_shared
    oa = graph.copy()
    oa.remove_edges_from(
        [(e["citing"], e["cited"]) for e in edge_rows if "openalex_referenced_works" not in e["sources"]]
    )
    variants["openalex_only"] = oa
    # Набор кандидатов не меняется. Обзоры не голосуют, но остаются в рейтинге.
    no_review_votes = graph.copy()
    no_review_votes.remove_edges_from([(u, v) for u, v in edges if by_id[u]["is_review"]])
    variants["no_review_outgoing_votes"] = no_review_votes

    deterministic = []
    variant_summary = []
    for variant, g in variants.items():
        matrix = base if variant == "baseline" else scores(g, ids, by_id)
        for m, metric in enumerate(METRICS):
            ranks, credit = ranking(matrix[m])
            order = np.lexsort((np.array(ids), -matrix[m]))
            variant_summary.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "edges": g.number_of_edges(),
                    "top5_by_score_then_id": [ids[j] for j in order[:5]],
                    "top5_membership_mass_retained": float(np.minimum(base_credits[m], credit).sum()),
                }
            )
            for j, uid in enumerate(ids):
                deterministic.append(
                    {
                        "variant": variant,
                        "metric": metric,
                        "id": uid,
                        "title": by_id[uid]["title"],
                        "score": matrix[m, j],
                        "midrank": ranks[j],
                        "top5_credit": credit[j],
                    }
                )
    write_csv(OUT / "rankings.csv", deterministic)

    scenarios, source_ids = run_simulations(ids, edges, by_id, base_ranks, base_credits)

    correlations = []
    head = np.argsort(-base[1], kind="stable")[:20]
    for m, metric in enumerate(METRICS):
        correlations.append(
            {
                "metric": metric,
                "vs_local_all140": float(spearmanr(base[0], base[m]).statistic),
                "vs_local_base_pagerank_top20": float(spearmanr(base[0, head], base[m, head]).statistic),
            }
        )

    manifest_paths = [
        TABLES / "papers.json",
        TABLES / "edges.csv",
        ROOT / "src/metrics.py",
        ROOT / "src/experiments.py",
        ROOT / "texts.json",
        DATA / "corpus.json",
        DATA / "provenance.json",
    ]
    input_hashes = {}
    for path in manifest_paths:
        input_hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()

    summary = {
        "seed": SEED,
        "repeats_per_scenario": REPEATS,
        "total_perturbations": REPEATS * len(scenarios),
        "vertices": len(ids),
        "edges": len(edges),
        "citing_papers_with_observed_references": len(source_ids),
        "observed_shared_author_edges": len(edges) - no_shared.number_of_edges(),
        "papers_with_missing_or_potentially_truncated_authors": sum(
            r["missing_or_potentially_truncated"] for r in author_rows
        ),
        "papers_without_any_author_ids": sum(not authors[u] for u in ids),
        "edges_with_both_bylines_observed_without_truncation_flag": sum(
            e["both_bylines_observed_without_truncation_flag"] for e in audit
        ),
        "fractional_total_mass": float(base[3].sum()),
        "variants": variant_summary,
        "correlations": correlations,
        "scenarios": [{"name": n, "unit": u, "removal_probability": p} for n, u, p in scenarios],
        **TEXTS["experiments"],
        "input_sha256": input_hashes,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    display_fields = [
        "vertices",
        "edges",
        "observed_shared_author_edges",
        "papers_with_missing_or_potentially_truncated_authors",
        "correlations",
    ]
    display_summary = {key: summary[key] for key in display_fields}
    print(json.dumps(display_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
