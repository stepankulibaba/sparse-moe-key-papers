"""Проверки поведения методов, независимые численные расчёты и аудит входов."""

import csv
import hashlib
import importlib.metadata
import json
import platform
import unittest

import networkx as nx
import numpy as np
from scipy.stats import rankdata

from .experiments import fractional, ranking
from .metrics import influence
from .storage import (
    DATA,
    EXPERIMENTS,
    ROOT,
    TABLES,
    normalize,
    read_csv,
    read_json,
)


class InfluenceBehavior(unittest.TestCase):
    def test_chain_has_one_lineage(self):
        g = nx.DiGraph([("c", "b"), ("b", "a"), ("a", "root")])
        score = influence(g)["root"]
        self.assertAlmostEqual(score["unique_descendants"], 1.75)
        self.assertAlmostEqual(score["branching_descendants"], 0)
        self.assertAlmostEqual(score["effective_branches"], 1)

    def test_star_has_broad_direct_adoption_proxy(self):
        g = nx.DiGraph([(u, "root") for u in ["a", "b", "c"]])
        score = influence(g)["root"]
        self.assertAlmostEqual(score["unique_descendants"], 3)
        self.assertAlmostEqual(score["branching_descendants"], 2)
        self.assertAlmostEqual(score["effective_branches"], 3)

    def test_diamond_counts_shared_descendant_once(self):
        g = nx.DiGraph([("a", "root"), ("b", "root"), ("c", "a"), ("c", "b")])
        score = influence(g)["root"]
        self.assertAlmostEqual(score["unique_descendants"], 2.5)
        self.assertAlmostEqual(score["katz_paths"], 3)
        self.assertAlmostEqual(score["branching_descendants"], 1.25)

    def test_isolate_and_invalid_cycle(self):
        g = nx.DiGraph()
        g.add_node("alone")
        self.assertEqual(influence(g)["alone"]["unique_descendants"], 0)
        self.assertEqual(influence(g)["alone"]["branching_descendants"], 0)
        with self.assertRaises(ValueError):
            influence(nx.DiGraph([("a", "b"), ("b", "a")]))


def check_corpus():
    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "output/summary.json").read_text(encoding="utf-8"))
    with (TABLES / "edges.csv").open(encoding="utf-8-sig", newline="") as f:
        edges = list(csv.DictReader(f))
    ids = [p["id"] for p in papers]
    index = {u: i for i, u in enumerate(ids)}
    assert len(papers) >= 100 and len(index) == len(papers)
    assert len({normalize(p["title"]) for p in papers}) == len(papers)
    assert all("2017-01-01" <= p["date"] <= "2025-12-31" for p in papers)
    assert len(edges) == summary["edges"]
    assert len({(e["citing"], e["cited"]) for e in edges}) == len(edges)
    matrix = np.zeros((len(ids), len(ids)))
    for e in edges:
        assert (
            e["citing"] in index and e["cited"] in index and e["citing"] != e["cited"]
        )
        matrix[index[e["citing"]], index[e["cited"]]] = 1
    indegrees = matrix.sum(axis=0)
    assert all(p["local_citations"] == int(indegrees[i]) for i, p in enumerate(papers))
    outdegrees = matrix.sum(axis=1)
    for i in range(len(ids)):
        matrix[i, :] = matrix[i, :] / outdegrees[i] if outdegrees[i] else 1 / len(ids)
    # Solve the stationary equation directly, independently of NetworkX iteration.
    expected = np.linalg.solve(
        np.eye(len(ids)) - 0.85 * matrix.T, np.full(len(ids), 0.15 / len(ids))
    )
    observed = np.array([p["pagerank"] for p in papers])
    max_error = float(np.max(np.abs(expected - observed)))
    assert max_error < 1e-8, max_error
    assert abs(observed.sum() - 1) < 1e-9
    for metric in [
        "pagerank",
        "global_citations",
        "local_citations",
        "citations_per_year",
    ]:
        ordered = sorted(papers, key=lambda p: (-p[metric], p["id"]))
        assert all(p["rank_" + metric] == i for i, p in enumerate(ordered, 1))
    provenance = read_json(DATA / "provenance.json")
    manifest = provenance["openalex_requests"]
    assert len(manifest) == 30
    for path, expected_hash in provenance["input_sha256"].items():
        assert (
            hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected_hash
        ), path
    assert all("api_key=" not in item["url"] for item in manifest)
    pdf_edges = read_csv(TABLES / "pdf_edges.csv")
    bibliographies = {
        entry["arxiv"]: "\n".join(entry["lines"])
        for entry in read_json(DATA / "bibliographies.json")
    }
    for edge in pdf_edges:
        assert normalize(edge["matched_title"]) in normalize(
            bibliographies[edge["evidence_key"]]
        )
    # Check several high-impact edges directly against saved bibliography evidence.
    important = [
        ("W3040573126", "W2581624817"),
        ("W3119866685", "W2581624817"),
        ("W4402671950", "W4200634402"),
    ]
    for u, v in important:
        assert any(e["citing"].endswith(u) and e["cited"].endswith(v) for e in edges), (
            u,
            v,
        )
    versions = {}
    for package in ["networkx", "numpy", "scipy", "matplotlib"]:
        versions[package] = importlib.metadata.version(package)
    result = {
        "status": "passed",
        "candidate_count": len(papers),
        "edge_count": len(edges),
        "independent_pagerank_max_absolute_error": max_error,
        "normalized_input_hashes_checked": len(provenance["input_sha256"]),
        "original_response_hashes_retained": len(manifest),
        "pdf_edge_title_matches_checked": len(pdf_edges),
        "high_impact_edges_checked": len(important),
        "limitations": [
            "Exact-title matching can miss changed titles and PDF extraction errors.",
            "Bibliography title matching is not exhaustive human review of every source.",
            "Numerical checks do not validate the scientific interpretation of a ranking.",
        ],
        "python": platform.python_version(),
        "packages": versions,
    }
    (ROOT / "output/verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def check_metrics():
    rows = json.loads((TABLES / "custom_rankings.json").read_text(encoding="utf-8"))
    papers = json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    by_id = {p["id"]: p for p in papers}
    ids = [r["id"] for r in rows]
    index = {u: i for i, u in enumerate(ids)}
    n = len(ids)
    a = np.zeros((n, n))
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    with (TABLES / "edges.csv").open(encoding="utf-8-sig", newline="") as f:
        for e in csv.DictReader(f):
            u, v = e["citing"], e["cited"]
            if by_id[u]["date"] >= by_id[v]["date"]:
                a[index[u], index[v]] = 1
                dist[index[u], index[v]] = 1
    # Floyd-Warshall is independent of the per-target BFS used in the scorer.
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    weighted = np.zeros_like(dist)
    mask = (dist >= 1) & np.isfinite(dist)
    weighted[mask] = 0.5 ** (dist[mask] - 1)
    expected_unique = weighted.sum(axis=0)
    expected_katz = np.linalg.solve(np.eye(n) - 0.5 * a.T, np.ones(n)) - 1
    expected_katz /= 0.5
    error_u = max(
        abs(r["unique_descendants"] - expected_unique[i]) for i, r in enumerate(rows)
    )
    error_k = max(abs(r["katz_paths"] - expected_katz[i]) for i, r in enumerate(rows))
    assert error_u < 1e-9 and error_k < 1e-7, (error_u, error_k)
    for r in rows:
        assert 0 <= r["branching_descendants"] <= r["unique_descendants"] + 1e-9
    for metric in ["unique_descendants", "branching_descendants", "katz_paths"]:
        ordered = sorted(rows, key=lambda r: (-r[metric], r["id"]))
        assert all(r["rank_" + metric] == i for i, r in enumerate(ordered, 1))
    with (TABLES / "reuse_edges.csv").open(encoding="utf-8-sig", newline="") as f:
        edges = list(csv.DictReader(f))
    with (TABLES / "semantic_rankings.csv").open(encoding="utf-8-sig", newline="") as f:
        semantic = list(csv.DictReader(f))
    assert len({(e["citing"], e["cited"]) for e in edges}) == len(edges)
    assert all(e["citing"] in index and e["cited"] in index for e in edges)
    for r in semantic:
        expected = sum(float(e["weight"]) for e in edges if e["cited"] == r["id"])
        assert abs(float(r["pilot_reuse_score"]) - expected) < 1e-10
    result = {
        "status": "passed",
        "unique_reach_floyd_warshall_max_error": float(error_u),
        "katz_independent_linear_system_max_error": float(error_k),
        "branch_score_bounds_checked": len(rows),
        "semantic_edge_dedup_and_score_sum_checked": len(edges),
        "note": "Numerical/provenance checks; no independent validation of semantic annotation or scientific novelty.",
    }
    (ROOT / "output/extension_verification.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


class MethodTests(unittest.TestCase):
    def test_one_vote_per_source_and_dangling_nodes(self):
        g = nx.DiGraph([("a", "x"), ("a", "y"), ("b", "x")])
        g.add_node("isolated")
        self.assertEqual(
            fractional(g), {"a": 0.0, "x": 1.5, "y": 0.5, "b": 0.0, "isolated": 0.0}
        )

    def test_boundary_ties_have_equal_credit(self):
        ranks, credits = ranking([9, 8, 7, 6, 5, 5, 0])
        np.testing.assert_array_equal(ranks, [1, 2, 3, 4, 5.5, 5.5, 7])
        np.testing.assert_array_equal(credits, [1, 1, 1, 1, 0.5, 0.5, 0])
        np.testing.assert_allclose(ranking(np.zeros(140))[1], np.full(140, 5 / 140))


def check_experiments():
    s = json.loads((EXPERIMENTS / "summary.json").read_text(encoding="utf-8"))
    for path, expected in s["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    with np.load(EXPERIMENTS / "simulations.npz", allow_pickle=False) as z:
        ids, metrics, scenarios, edges = (
            z["ids"],
            z["metrics"],
            z["scenarios"],
            z["edges"],
        )
        ranks, credits, keep = z["ranks"], z["top5_credit"], z["edge_keep"]
    assert ranks.shape == (4, 500, 4, 140)
    assert keep.shape == (4, 500, 550)
    assert np.isfinite(ranks).all() and np.isfinite(credits).all()
    assert ((ranks >= 1) & (ranks <= 140)).all()
    assert ((credits >= 0) & (credits <= 1)).all()
    np.testing.assert_allclose(credits.sum(axis=-1), 5)
    np.testing.assert_allclose(ranks.sum(axis=-1), 140 * 141 / 2)
    source_ids = sorted(set(edges[:, 0]))
    source_indices = np.array([source_ids.index(u) for u in edges[:, 0]])
    rng = np.random.default_rng(s["seed"])
    for k, spec in enumerate(s["scenarios"]):
        n_units = len(edges) if spec["unit"] == "edge" else len(source_ids)
        expected = rng.random((500, n_units)) >= spec["removal_probability"]
        if spec["unit"] == "bibliography":
            expected = expected[:, source_indices]
        np.testing.assert_array_equal(keep[k], expected)

    # Rebuild the transition matrix independently, including dangling rows.
    def independent(mask):
        index = {u: j for j, u in enumerate(ids)}
        adjacency = np.zeros((len(ids), len(ids)))
        for (u, v), included in zip(edges, mask):
            if included:
                adjacency[index[u], index[v]] = 1
        counts = adjacency.sum(axis=1)
        transition = np.divide(
            adjacency,
            counts[:, None],
            out=np.zeros_like(adjacency),
            where=counts[:, None] != 0,
        )
        frac = transition.sum(axis=0)
        transition[counts == 0, :] = 1 / len(ids)
        pr = np.linalg.solve(
            np.eye(len(ids)) - 0.85 * transition.T, np.full(len(ids), 0.15 / len(ids))
        )
        return adjacency.sum(axis=0), pr, frac

    for k in range(4):
        for trial in [0, 499]:
            c, pr, f = independent(keep[k, trial])
            for values, m in [(c, 0), (pr, 1), (f, 3)]:
                np.testing.assert_array_equal(
                    rankdata(-np.round(values, 12)), ranks[k, trial, m]
                )

    with (EXPERIMENTS / "rankings.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    base = {(r["metric"], r["id"]): r for r in rows if r["variant"] == "baseline"}
    c, pr, f = independent(np.ones(len(edges), dtype=bool))
    papers = {
        p["id"]: p
        for p in json.loads((TABLES / "papers.json").read_text(encoding="utf-8"))
    }
    custom = {
        p["id"]: p
        for p in json.loads(
            (TABLES / "custom_rankings.json").read_text(encoding="utf-8")
        )
    }
    for j, uid in enumerate(ids):
        np.testing.assert_allclose(
            [
                float(base["local", uid]["score"]),
                float(base["pagerank", uid]["score"]),
                float(base["fractional", uid]["score"]),
            ],
            [c[j], pr[j], f[j]],
            atol=1e-10,
            rtol=0,
        )
        assert c[j] == papers[uid]["local_citations"]
        assert (
            abs(
                float(base["branching", uid]["score"])
                - custom[uid]["branching_descendants"]
            )
            < 1e-10
        )
    assert abs(f.sum() - len(source_ids)) < 1e-10
    with (EXPERIMENTS / "stability.csv").open(encoding="utf-8-sig") as file:
        stability = list(csv.DictReader(file))
    for row in stability:
        k = list(scenarios).index(row["scenario"])
        m = list(metrics).index(row["metric"])
        j = list(ids).index(row["id"])
        assert abs(float(row["top5_frequency"]) - credits[k, :, m, j].mean()) < 1e-12
        np.testing.assert_allclose(
            [float(row["rank_p05"]), float(row["rank_median"]), float(row["rank_p95"])],
            np.quantile(ranks[k, :, m, j], [0.05, 0.5, 0.95]),
        )
    with (EXPERIMENTS / "edge_author_audit.csv").open(encoding="utf-8-sig") as file:
        author_edges = list(csv.DictReader(file))
    assert (
        sum(bool(e["shared_author_ids"]) for e in author_edges)
        == s["observed_shared_author_edges"]
    )
    verified = {
        "status": "passed",
        "toy_tests": 2,
        "input_hashes_verified": len(s["input_sha256"]),
        "perturbations_checked": 2000,
        "independent_dense_pagerank_trial_checks": 8,
        "summary_rows_reconciled": len(stability),
        "checks": [
            "frozen inputs",
            "RNG replay and bibliography masks",
            "tie-aware ranks and mass conservation",
            "independent dense PageRank, local and fractional ranks",
            "baseline agrees with existing artifacts",
            "all reported frequencies and rank intervals match saved simulations",
        ],
    }
    (EXPERIMENTS / "verification.json").write_text(
        json.dumps(verified, indent=2), encoding="utf-8"
    )
    print(json.dumps(verified, indent=2))


def run_checks(include_stress=True):
    suite = unittest.TestSuite()
    for test_case in [InfluenceBehavior, MethodTests]:
        suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(test_case))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise RuntimeError("Behavior checks failed")
    check_corpus()
    check_metrics()
    if include_stress:
        check_experiments()
