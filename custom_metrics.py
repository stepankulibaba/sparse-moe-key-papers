"""Compare unique and branch-diverse downstream reach with path-counting Katz.

The graph is a CITATION graph, not a verified idea-inheritance graph. These are
structural proxies until text evidence distinguishes method use from mentions.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path
import networkx as nx
from scipy.stats import spearmanr
from analyze import write_csv

ROOT=Path(__file__).resolve().parent

def influence(graph,beta=.5):
    if not 0<beta<1:raise ValueError('beta must be in (0,1)')
    if not nx.is_directed_acyclic_graph(graph):raise ValueError('Katz path propagation requires the chronological DAG')
    result={}
    katz=dict.fromkeys(graph,0.0)
    for u in nx.topological_sort(graph):
        for v in graph.successors(u):katz[v]+=1+beta*katz[u]
    reverse=graph.reverse(copy=False)
    for v in sorted(graph):
        distance=nx.single_source_shortest_path_length(reverse,v)
        supports={}; mass=defaultdict(float); unique=0.0
        for u,d in sorted(distance.items(),key=lambda x:(x[1],x[0])):
            if d==0:continue
            if d==1:supports[u]={u}
            else:
                supports[u]=set().union(*(supports[t] for t in graph.successors(u) if distance.get(t)==d-1))
            weight=beta**(d-1)
            unique+=weight
            for first_branch in supports[u]:mass[first_branch]+=weight/len(supports[u])
        concentration=sum((m/unique)**2 for m in mass.values()) if unique else 0.0
        diverse=unique*(1-concentration) if unique else 0.0
        result[v]={'unique_descendants':unique,'branching_descendants':max(0.0,diverse),
                   'effective_branches':1/concentration if concentration else 0.0,
                   'katz_paths':katz[v],'direct_children':graph.in_degree(v),
                   'reachable_descendants':len(distance)-1,
                   'largest_branch_mass':max(mass.values(),default=0.0)}
        assert abs(sum(mass.values())-unique)<1e-8
        assert -1e-10<=diverse<=unique+1e-10
        assert katz[v]+1e-9>=unique
    return result

def main():
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    by_id={p['id']:p for p in papers}
    graph=nx.DiGraph();graph.add_nodes_from(by_id)
    removed=[]
    with (ROOT/'data/edges.csv').open(encoding='utf-8-sig',newline='') as f:
        for e in csv.DictReader(f):
            u,v=e['citing'],e['cited']
            if by_id[u]['date']<by_id[v]['date']:removed.append(e)
            else:graph.add_edge(u,v)
    assert nx.is_directed_acyclic_graph(graph)
    base=influence(graph)
    rows=[]
    for p in papers:
        rows.append({k:p[k] for k in ['id','title','year','global_citations','local_citations','pagerank','rank_global_citations','rank_local_citations','rank_pagerank']} | base[p['id']])
    metrics=['unique_descendants','branching_descendants','katz_paths']
    for metric in metrics:
        for i,r in enumerate(sorted(rows,key=lambda r:(-r[metric],r['id'])),1):r['rank_'+metric]=i
    sensitivity=[]
    for beta in [.25,.5,.75]:
        variant=influence(graph,beta)
        for metric in metrics:
            top=sorted(graph,key=lambda u:(-variant[u][metric],u))[:5]
            base_top=sorted(graph,key=lambda u:(-base[u][metric],u))[:5]
            sensitivity.append({'beta':beta,'metric':metric,'top5_ids':top,'top5_overlap_with_beta_half':len(set(top)&set(base_top))})
    correlations={}
    for metric in metrics:
        correlations[metric]={comparison:float(spearmanr([r[metric] for r in rows],[r[comparison] for r in rows]).statistic)
                              for comparison in ['local_citations','pagerank']}
    summary={'status':'calculated_structural_proxies_not_verified_reuse','vertices':len(graph),'chronological_edges':len(graph.edges),
             'removed_date_anomaly_edges':len(removed),'beta':.5,'correlations':correlations,'sensitivity':sensitivity,
             'top10':{metric:[r['id'] for r in sorted(rows,key=lambda r:r['rank_'+metric])[:10]] for metric in metrics},
             'definitions':{
                 'unique_descendants':'U_beta(v)=sum_{u: d(u,v)>=1} beta^(d(u,v)-1); each reachable citing paper counted once at shortest distance.',
                 'branching_descendants':'B_beta(v)=U_beta(v)*(1-sum_b p_b^2), p_b=M_b/U; allocate each unique descendant vote equally over first-generation branches lying on shortest paths.',
                 'katz_paths':'K_beta(v)=sum_{paths ending at v, length>=1} beta^(length-1); finite on chronological DAG. Standard comparator, not claimed original.'},
             'limitations':['Citation paths do not establish idea inheritance.',
                'B deliberately discounts one concentrated lineage, including important breakthroughs spread through one successor.',
                'Missing links and corpus selection affect all reachability measures.',
                'Paper branches are not independently verified research teams.',
                'No AI-authorship inference or stylistic penalty is used.']}
    write_csv(ROOT/'data/custom_rankings.csv',sorted(rows,key=lambda r:r['rank_branching_descendants']))
    (ROOT/'data/custom_rankings.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (ROOT/'output/custom_metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    nx.write_graphml(graph,ROOT/'output/chronological_graph.graphml')
    for metric in metrics:
        print('\n'+metric)
        for r in sorted(rows,key=lambda r:r['rank_'+metric])[:8]:print(r['rank_'+metric],round(r[metric],3),r['title'])

if __name__=='__main__':main()
