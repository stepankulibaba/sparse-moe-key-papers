"""Behavioral counterexamples: chains, stars and a diamond have distinct meanings."""
import unittest
import networkx as nx
from custom_metrics import influence

class InfluenceBehavior(unittest.TestCase):
    def test_chain_has_one_lineage(self):
        g=nx.DiGraph([('c','b'),('b','a'),('a','root')])
        score=influence(g)['root']
        self.assertAlmostEqual(score['unique_descendants'],1.75)
        self.assertAlmostEqual(score['branching_descendants'],0)
        self.assertAlmostEqual(score['effective_branches'],1)
    def test_star_has_broad_direct_adoption_proxy(self):
        g=nx.DiGraph([(u,'root') for u in ['a','b','c']])
        score=influence(g)['root']
        self.assertAlmostEqual(score['unique_descendants'],3)
        self.assertAlmostEqual(score['branching_descendants'],2)
        self.assertAlmostEqual(score['effective_branches'],3)
    def test_diamond_counts_shared_descendant_once(self):
        g=nx.DiGraph([('a','root'),('b','root'),('c','a'),('c','b')])
        score=influence(g)['root']
        self.assertAlmostEqual(score['unique_descendants'],2.5)
        self.assertAlmostEqual(score['katz_paths'],3)
        self.assertAlmostEqual(score['branching_descendants'],1.25)
    def test_isolate_and_invalid_cycle(self):
        g=nx.DiGraph();g.add_node('alone')
        self.assertEqual(influence(g)['alone']['unique_descendants'],0)
        self.assertEqual(influence(g)['alone']['branching_descendants'],0)
        with self.assertRaises(ValueError):influence(nx.DiGraph([('a','b'),('b','a')]))

if __name__=='__main__':unittest.main()
