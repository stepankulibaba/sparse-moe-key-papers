"""Independent numerical and saved-artifact checks for robustness experiments."""
import csv
import hashlib
import json
from pathlib import Path
import unittest

import networkx as nx
import numpy as np
from scipy.stats import rankdata

from robustness import OUT, ROOT, fractional, ranking


class MethodTests(unittest.TestCase):
    def test_one_vote_per_source_and_dangling_nodes(self):
        g = nx.DiGraph([('a', 'x'), ('a', 'y'), ('b', 'x')])
        g.add_node('isolated')
        self.assertEqual(fractional(g), {'a': 0., 'x': 1.5, 'y': .5, 'b': 0., 'isolated': 0.})

    def test_boundary_ties_have_equal_credit(self):
        ranks, credits = ranking([9, 8, 7, 6, 5, 5, 0])
        np.testing.assert_array_equal(ranks, [1, 2, 3, 4, 5.5, 5.5, 7])
        np.testing.assert_array_equal(credits, [1, 1, 1, 1, .5, .5, 0])
        np.testing.assert_allclose(ranking(np.zeros(140))[1], np.full(140, 5/140))


def main():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(MethodTests))
    assert result.wasSuccessful()
    s = json.loads((OUT / 'summary.json').read_text(encoding='utf-8'))
    for path, expected in s['input_sha256'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    with np.load(OUT / 'simulations.npz', allow_pickle=False) as z:
        ids, metrics, scenarios, edges = z['ids'], z['metrics'], z['scenarios'], z['edges']
        ranks, credits, keep = z['ranks'], z['top5_credit'], z['edge_keep']
    assert ranks.shape == (4, 500, 4, 140)
    assert keep.shape == (4, 500, 550)
    assert np.isfinite(ranks).all() and np.isfinite(credits).all()
    assert ((ranks >= 1) & (ranks <= 140)).all()
    assert ((credits >= 0) & (credits <= 1)).all()
    np.testing.assert_allclose(credits.sum(axis=-1), 5)
    np.testing.assert_allclose(ranks.sum(axis=-1), 140*141/2)
    source_ids = sorted(set(edges[:, 0]))
    source_indices = np.array([source_ids.index(u) for u in edges[:, 0]])
    rng = np.random.default_rng(s['seed'])
    for k, spec in enumerate(s['scenarios']):
        n_units = len(edges) if spec['unit'] == 'edge' else len(source_ids)
        expected = rng.random((500, n_units)) >= spec['removal_probability']
        if spec['unit'] == 'bibliography':
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
        transition = np.divide(adjacency, counts[:, None], out=np.zeros_like(adjacency), where=counts[:, None] != 0)
        frac = transition.sum(axis=0)
        transition[counts == 0, :] = 1/len(ids)
        pr = np.linalg.solve(np.eye(len(ids)) - .85*transition.T, np.full(len(ids), .15/len(ids)))
        return adjacency.sum(axis=0), pr, frac

    for k in range(4):
        for trial in [0, 499]:
            c, pr, f = independent(keep[k, trial])
            for values, m in [(c, 0), (pr, 1), (f, 3)]:
                np.testing.assert_array_equal(rankdata(-np.round(values, 12)), ranks[k, trial, m])

    with (OUT / 'rankings.csv').open(encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))
    base = {(r['metric'], r['id']): r for r in rows if r['variant'] == 'baseline'}
    c, pr, f = independent(np.ones(len(edges), dtype=bool))
    papers = {p['id']: p for p in json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))}
    custom = {p['id']: p for p in json.loads((ROOT/'data/custom_rankings.json').read_text(encoding='utf-8'))}
    for j, uid in enumerate(ids):
        np.testing.assert_allclose([float(base['local', uid]['score']), float(base['pagerank', uid]['score']), float(base['fractional', uid]['score'])], [c[j], pr[j], f[j]], atol=1e-10, rtol=0)
        assert c[j] == papers[uid]['local_citations']
        assert abs(float(base['branching', uid]['score']) - custom[uid]['branching_descendants']) < 1e-10
    assert abs(f.sum() - len(source_ids)) < 1e-10
    with (OUT / 'stability.csv').open(encoding='utf-8-sig') as file:
        stability = list(csv.DictReader(file))
    for row in stability:
        k = list(scenarios).index(row['scenario'])
        m = list(metrics).index(row['metric'])
        j = list(ids).index(row['id'])
        assert abs(float(row['top5_frequency']) - credits[k, :, m, j].mean()) < 1e-12
        np.testing.assert_allclose([float(row['rank_p05']), float(row['rank_median']), float(row['rank_p95'])], np.quantile(ranks[k, :, m, j], [.05, .5, .95]))
    with (OUT / 'edge_author_audit.csv').open(encoding='utf-8-sig') as file:
        author_edges = list(csv.DictReader(file))
    assert sum(bool(e['shared_author_ids']) for e in author_edges) == s['observed_shared_author_edges']
    verified = {'status': 'passed', 'toy_tests': result.testsRun,
                'input_hashes_verified': len(s['input_sha256']), 'perturbations_checked': 2000,
                'independent_dense_pagerank_trial_checks': 8, 'summary_rows_reconciled': len(stability),
                'checks': ['frozen inputs', 'RNG replay and bibliography masks', 'tie-aware ranks and mass conservation',
                           'independent dense PageRank, local and fractional ranks', 'baseline agrees with existing artifacts',
                           'all reported frequencies and rank intervals match saved simulations']}
    (OUT / 'verification.json').write_text(json.dumps(verified, indent=2), encoding='utf-8')
    print(json.dumps(verified, indent=2))


if __name__ == '__main__':
    main()
