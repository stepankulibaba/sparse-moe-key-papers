"""Independent matrix/distance checks for newly added graph and pilot scores."""
import csv
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent

def main():
    rows=json.loads((ROOT/'data/custom_rankings.json').read_text(encoding='utf-8'))
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    by_id={p['id']:p for p in papers}
    ids=[r['id'] for r in rows];index={u:i for i,u in enumerate(ids)};n=len(ids)
    a=np.zeros((n,n));dist=np.full((n,n),np.inf);np.fill_diagonal(dist,0)
    with (ROOT/'data/edges.csv').open(encoding='utf-8-sig',newline='') as f:
        for e in csv.DictReader(f):
            u,v=e['citing'],e['cited']
            if by_id[u]['date']>=by_id[v]['date']:a[index[u],index[v]]=1;dist[index[u],index[v]]=1
    # Floyd-Warshall is independent of the per-target BFS used in the scorer.
    for k in range(n):dist=np.minimum(dist,dist[:,k,None]+dist[None,k,:])
    weighted=np.zeros_like(dist)
    mask=(dist>=1)&np.isfinite(dist)
    weighted[mask]=.5**(dist[mask]-1)
    expected_unique=weighted.sum(axis=0)
    expected_katz=np.linalg.solve(np.eye(n)-.5*a.T,np.ones(n))-1
    expected_katz/=.5
    error_u=max(abs(r['unique_descendants']-expected_unique[i]) for i,r in enumerate(rows))
    error_k=max(abs(r['katz_paths']-expected_katz[i]) for i,r in enumerate(rows))
    assert error_u<1e-9 and error_k<1e-7,(error_u,error_k)
    for r in rows:
        assert 0<=r['branching_descendants']<=r['unique_descendants']+1e-9
    for metric in ['unique_descendants','branching_descendants','katz_paths']:
        ordered=sorted(rows,key=lambda r:(-r[metric],r['id']))
        assert all(r['rank_'+metric]==i for i,r in enumerate(ordered,1))
    with (ROOT/'data/reuse_edges.csv').open(encoding='utf-8-sig',newline='') as f:edges=list(csv.DictReader(f))
    with (ROOT/'data/semantic_rankings.csv').open(encoding='utf-8-sig',newline='') as f:semantic=list(csv.DictReader(f))
    assert len({(e['citing'],e['cited']) for e in edges})==len(edges)
    assert all(e['citing'] in index and e['cited'] in index for e in edges)
    for r in semantic:
        expected=sum(float(e['weight']) for e in edges if e['cited']==r['id'])
        assert abs(float(r['pilot_reuse_score'])-expected)<1e-10
    result={'status':'passed','unique_reach_floyd_warshall_max_error':float(error_u),
            'katz_independent_linear_system_max_error':float(error_k),
            'branch_score_bounds_checked':len(rows),'semantic_edge_dedup_and_score_sum_checked':len(edges),
            'note':'Numerical/provenance checks; no independent validation of semantic annotation or scientific novelty.'}
    (ROOT/'output/extension_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
