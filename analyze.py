"""Rebuild the corpus and all ranks from the frozen JSON cache (offline)."""
from __future__ import annotations
import csv
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date
from pathlib import Path
import networkx as nx
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUT = ROOT / 'output'
AS_OF = date(2026, 11, 4)  # publication-age denominator; counts are snapshot counts
NAMED = re.compile(r'^(gshard:|switch transformers:|glam: efficient scaling|mixtral of experts$|deepseekmoe:|deepseek[- ]v[23]:|deepseek[- ]v3 technical report$|base layers:|fastmoe:|deepspeed[- ]moe:|megablocks: efficient sparse|st[- ]moe: designing|sparse upcycling: training|mixture.of.experts with expert choice routing)', re.I)
MOE = re.compile(r'mixture[\s-]+of[\s-]+(?:[\w-]+[\s-]+)?experts?|sparse[\s-]+experts?|\bmoe\b|sparsely[\s-]+gated', re.I)
NLP = re.compile(r'language model|\bllms?\b|natural language|machine translation|language modeling|language modelling|\bnlp\b|text generation|text-to-text|vision.language|multimodal language', re.I)
BAD_TITLE = re.compile(r'clinical|medical|healthcare|ophthalm|dental|patient|diagnos|radiolog|cardiol|bibliometric|satellite networks', re.I)
REVIEW = re.compile(r'\bsurvey\b|\breview\b', re.I)

def normalize(title):
    return re.sub(r'[^a-z0-9]', '', unicodedata.normalize('NFKD', title).lower())

def abstract(p):
    inv = p.get('abstract_inverted_index') or {}
    return ' '.join(w for _, w in sorted((i, w) for w, indices in inv.items() for i in indices))

def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f'Empty table needs explicit columns: {path}')
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

def build():
    OUT.mkdir(exist_ok=True)
    records, alias_records, queries = {}, {}, defaultdict(set)
    for path in sorted((DATA / 'raw').glob('*.json')):
        payload = json.loads(path.read_text(encoding='utf-8'))
        for p in payload['body'].get('results', []):
            (alias_records if path.stem.startswith('alias_batch_') else records)[p['id']] = p
            queries[p['id']].add(path.stem)
    exclusions = json.loads((ROOT / 'scope_exclusions.json').read_text(encoding='utf-8'))
    audit, groups = [], defaultdict(list)
    for p in records.values():
        title, text = p['title'], p['title'] + ' ' + abstract(p)
        named = bool(NAMED.search(title))
        lexical = bool(MOE.search(text) and NLP.search(text))
        accepted = (named or lexical) and not BAD_TITLE.search(title)
        reason = 'named_architecture_or_system' if named else 'MoE_and_language_or_transformer_terms'
        if not accepted:
            reason = 'application_outside_architecture_scope' if BAD_TITLE.search(title) else 'insufficient_topic_evidence'
        if p['id'].rsplit('/', 1)[-1] in exclusions:
            accepted = False
            reason = 'scope_audit: ' + exclusions[p['id'].rsplit('/', 1)[-1]]
        audit.append({'id': p['id'], 'title': title, 'accepted': bool(accepted), 'rule': reason,
                      'has_abstract': bool(p.get('abstract_inverted_index')), 'queries': ';'.join(sorted(queries[p['id']]))})
        if accepted:
            groups[normalize(title)].append(p)
    selected_ids = {p['id'] for group in groups.values() for p in group}
    for p in alias_records.values():
        key=normalize(p['title'])
        if key in groups and p['id'] not in selected_ids:
            groups[key].append(p)
            selected_ids.add(p['id'])
            audit.append({'id':p['id'],'title':p['title'],'accepted':True,'rule':'resolved_cited_version_exact_title',
                          'has_abstract':bool(p.get('abstract_inverted_index')),'queries':';'.join(sorted(queries[p['id']]))})
    rows, alias = [], {}
    for key, versions in sorted(groups.items()):
        # Stable representative; never add global counts across versions.
        versions.sort(key=lambda p: (-p['cited_by_count'], p['id']))
        p = versions[0]
        for v in versions:
            alias[v['id']] = p['id']
        refs = set(r for v in versions for r in v.get('referenced_works', []))
        pub_date = min(v['publication_date'] for v in versions)
        age = max(1, (AS_OF - date.fromisoformat(pub_date)).days / 365.25)
        rows.append({'id': p['id'], 'title': p['title'], 'year': int(pub_date[:4]), 'date': pub_date,
                     'global_citations': max(v['cited_by_count'] for v in versions),
                     'citations_per_year': max(v['cited_by_count'] for v in versions) / age,
                     'doi': p.get('doi') or '', 'url': p['id'],
                     'authors': '; '.join(a['author']['display_name'] for a in p.get('authorships', [])),
                     'is_review': bool(REVIEW.search(p['title'])), 'version_count': len(versions),
                     'version_ids': ';'.join(v['id'] for v in versions),
                     'reference_ids': sorted(refs), 'abstract': abstract(p),
                     'queries': ';'.join(sorted(set(q for v in versions for q in queries[v['id']]))),
                     'reference_count': len(refs)})
    graph = nx.DiGraph()
    graph.add_nodes_from(r['id'] for r in rows)
    by_id = {r['id']: r for r in rows}
    temporal = []
    for r in rows:
        for ref in r['reference_ids']:
            target = alias.get(ref)
            if target and target != r['id']:
                graph.add_edge(r['id'], target)
    metadata_graph = graph.copy()
    provenance = {(u,v): {'openalex_referenced_works'} for u,v in graph.edges}
    pdf_edge_file = DATA / 'pdf_edges.csv'
    if pdf_edge_file.exists():
        with pdf_edge_file.open(encoding='utf-8-sig', newline='') as f:
            for edge in csv.DictReader(f):
                u, v = alias.get(edge['citing'],edge['citing']), alias.get(edge['cited'],edge['cited'])
                if u in graph and v in graph and u != v:
                    graph.add_edge(u,v)
                    provenance.setdefault((u,v),set()).add(edge['source'])
    for u, v in graph.edges:
        if by_id[u]['date'] < by_id[v]['date']:
            temporal.append({'citing': u, 'cited': v, 'citing_date': by_id[u]['date'], 'cited_date': by_id[v]['date']})
    pr = nx.pagerank(graph, alpha=.85, tol=1e-12, max_iter=1000)
    for r in rows:
        r['local_citations'] = graph.in_degree(r['id'])
        r['pagerank'] = pr[r['id']]
    metrics = ['global_citations', 'local_citations', 'pagerank', 'citations_per_year']
    for metric in metrics:
        for rank, r in enumerate(sorted(rows, key=lambda r: (-r[metric], r['id'])), 1):
            r['rank_' + metric] = rank
    main = sorted(rows, key=lambda r: r['rank_pagerank'])
    # Highlighted nodes are an algorithmic top five, not the student's final list.
    highlights = [r['id'] for r in main[:5]]
    sensitivity = []
    for alpha in (.5, .7, .85, .95):
        score = nx.pagerank(graph, alpha=alpha, tol=1e-12, max_iter=1000)
        top = sorted(rows, key=lambda r: (-score[r['id']], r['id']))[:5]
        sensitivity.append({'variant': f'alpha={alpha}', 'top5_ids': [r['id'] for r in top],
                            'overlap_with_base': len(set(highlights) & {r['id'] for r in top})})
    no_reviews = graph.subgraph([r['id'] for r in rows if not r['is_review']]).copy()
    scores = nx.pagerank(no_reviews, alpha=.85, tol=1e-12, max_iter=1000)
    no_review_top = sorted(scores, key=lambda u: (-scores[u], u))[:5]
    sensitivity.append({'variant': 'exclude_title_marked_reviews', 'top5_ids': no_review_top,
                        'overlap_with_base': len(set(highlights) & set(no_review_top))})
    metadata_scores = nx.pagerank(metadata_graph, alpha=.85, tol=1e-12, max_iter=1000)
    metadata_top = sorted(metadata_scores, key=lambda u: (-metadata_scores[u],u))[:5]
    sensitivity.append({'variant':'openalex_edges_only','top5_ids':metadata_top,
                        'overlap_with_base':len(set(highlights) & set(metadata_top))})
    chronological_graph = graph.copy()
    chronological_graph.remove_edges_from((e['citing'],e['cited']) for e in temporal)
    chrono_score = nx.pagerank(chronological_graph, alpha=.85, tol=1e-12, max_iter=1000)
    chrono_top = sorted(chrono_score,key=lambda u:(-chrono_score[u],u))[:5]
    sensitivity.append({'variant':'remove_date_anomalies','top5_ids':chrono_top,
                        'overlap_with_base':len(set(highlights) & set(chrono_top))})
    summary = {'as_of_age_denominator': AS_OF.isoformat(), 'raw_unique': len(records),
               'alias_records_examined':len(alias_records),
               'accepted_records': sum(r['accepted'] for r in audit), 'candidates': len(rows),
               'merged_records': sum(r['version_count'] - 1 for r in rows), 'edges': graph.number_of_edges(),
               'openalex_edges':metadata_graph.number_of_edges(),
               'supplementary_edges':graph.number_of_edges()-metadata_graph.number_of_edges(),
               'isolates': nx.number_of_isolates(graph), 'weak_components': nx.number_weakly_connected_components(graph),
               'missing_abstracts': sum(not r['abstract'] for r in rows),
               'zero_reference_records': sum(not r['reference_ids'] for r in rows),
               'review_count': sum(r['is_review'] for r in rows),
               'temporal_anomalies': len(temporal), 'is_dag': nx.is_directed_acyclic_graph(graph),
               'pagerank_sum': sum(pr.values()), 'algorithmic_highlights': highlights, 'sensitivity': sensitivity,
               'spearman_global_vs_pagerank': float(spearmanr([r['global_citations'] for r in rows], [r['pagerank'] for r in rows]).statistic),
               'spearman_local_vs_pagerank': float(spearmanr([r['local_citations'] for r in rows], [r['pagerank'] for r in rows]).statistic),
               'top5_local_overlap_pagerank':len(set(highlights) & {r['id'] for r in sorted(rows,key=lambda r:r['rank_local_citations'])[:5]}),
               'local_top5_boundary_value':sorted((r['local_citations'] for r in rows),reverse=True)[4],
               'local_top5_boundary_tie_count':sum(r['local_citations']==sorted((q['local_citations'] for q in rows),reverse=True)[4] for r in rows),
               'top5_global_overlap_pagerank': len(set(highlights) & {r['id'] for r in sorted(rows, key=lambda r:r['rank_global_citations'])[:5]})}
    assert len(rows) >= 100, f'Only {len(rows)} eligible candidates; extend documented retrieval before reporting'
    assert len(by_id) == len(rows)
    assert abs(summary['pagerank_sum'] - 1) < 1e-9
    write_csv(DATA / 'corpus_audit.csv', sorted(audit, key=lambda r:r['id']))
    csv_fields = [k for k in rows[0] if k != 'reference_ids']
    write_csv(DATA / 'papers.csv', main, csv_fields)
    write_csv(DATA / 'edges.csv', [{'citing':u, 'cited':v,'sources':';'.join(sorted(provenance[(u,v)]))} for u,v in sorted(graph.edges)], ['citing','cited','sources'])
    write_csv(DATA / 'temporal_anomalies.csv', temporal, ['citing','cited','citing_date','cited_date'])
    write_csv(DATA / 'duplicates.csv', [r for r in rows if r['version_count'] > 1], ['id','title','version_count','version_ids'])
    (DATA / 'papers.json').write_text(json.dumps(main, ensure_ascii=False, indent=2), encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    nx.write_graphml(graph, OUT / 'citation_graph.graphml')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\nTOP 15: rank PR / global / local | title')
    for r in main[:15]:
        print(r['rank_pagerank'], r['rank_global_citations'], r['local_citations'], '|', r['title'])
    return rows, graph, summary

if __name__ == '__main__':
    build()
