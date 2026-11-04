"""Independent numerical/provenance checks; exits nonzero on material errors."""
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import numpy as np
from analyze import normalize

ROOT=Path(__file__).resolve().parent

def main():
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    summary=json.loads((ROOT/'output/summary.json').read_text(encoding='utf-8'))
    with (ROOT/'data/edges.csv').open(encoding='utf-8-sig',newline='') as f:edges=list(csv.DictReader(f))
    ids=[p['id'] for p in papers]; index={u:i for i,u in enumerate(ids)}
    assert len(papers)>=100 and len(index)==len(papers)
    assert len({normalize(p['title']) for p in papers})==len(papers)
    assert all('2017-01-01'<=p['date']<='2025-12-31' for p in papers)
    assert len(edges)==summary['edges']
    assert len({(e['citing'],e['cited']) for e in edges})==len(edges)
    matrix=np.zeros((len(ids),len(ids)))
    for e in edges:
        assert e['citing'] in index and e['cited'] in index and e['citing']!=e['cited']
        matrix[index[e['citing']],index[e['cited']]]=1
    indegrees=matrix.sum(axis=0)
    assert all(p['local_citations']==int(indegrees[i]) for i,p in enumerate(papers))
    outdegrees=matrix.sum(axis=1)
    for i in range(len(ids)):
        matrix[i,:]=matrix[i,:]/outdegrees[i] if outdegrees[i] else 1/len(ids)
    # Solve the stationary equation directly, independently of NetworkX iteration.
    expected=np.linalg.solve(np.eye(len(ids))-.85*matrix.T,np.full(len(ids),.15/len(ids)))
    observed=np.array([p['pagerank'] for p in papers])
    max_error=float(np.max(np.abs(expected-observed)))
    assert max_error<1e-8, max_error
    assert abs(observed.sum()-1)<1e-9
    for metric in ['pagerank','global_citations','local_citations','citations_per_year']:
        ordered=sorted(papers,key=lambda p:(-p[metric],p['id']))
        assert all(p['rank_'+metric]==i for i,p in enumerate(ordered,1))
    manifest=json.loads((ROOT/'data/collection_manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        assert hashlib.sha256((ROOT/item['file']).read_bytes()).hexdigest()==item['sha256']
        assert 'api_key=' not in item['url']
    with (ROOT/'data/pdf_edges.csv').open(encoding='utf-8-sig',newline='') as f:pdf_edges=list(csv.DictReader(f))
    for e in pdf_edges:
        bib=(ROOT/e['evidence_file']).read_text(encoding='utf-8')
        assert normalize(e['matched_title']) in normalize(bib)
    # Check several high-impact edges directly against saved bibliography evidence.
    important=[('W3040573126','W2581624817'),('W3119866685','W2581624817'),('W4402671950','W4200634402')]
    for u,v in important:
        assert any(e['citing'].endswith(u) and e['cited'].endswith(v) for e in edges),(u,v)
    versions={}
    for package in ['requests','networkx','numpy','scipy','matplotlib','PyMuPDF']:
        versions[package]=importlib.metadata.version(package)
    result={'status':'passed','candidate_count':len(papers),'edge_count':len(edges),
            'independent_pagerank_max_absolute_error':max_error,'raw_files_hash_checked':len(manifest),
            'pdf_edge_title_matches_checked':len(pdf_edges),'high_impact_edges_checked':len(important),
            'limitations':['Exact-title matching can miss changed titles and PDF extraction errors.',
                'Bibliography title matching is not exhaustive human review of every source.',
                'Numerical checks do not validate the scientific interpretation of a ranking.'],
            'python':platform.python_version(),'packages':versions}
    (ROOT/'output/verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
