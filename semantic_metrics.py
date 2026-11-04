"""Pilot reuse score from assistant-reviewed contexts; missingness stays explicit."""
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
from analyze import write_csv

ROOT=Path(__file__).resolve().parent

def main():
    contexts=json.loads((ROOT/'data/citation_contexts.json').read_text(encoding='utf-8'))
    by_context={r['context_id']:r for r in contexts}
    config=json.loads((ROOT/'context_review_labels.json').read_text(encoding='utf-8'))
    labels={i:label for label,ids in config['labels'].items() for i in ids}
    assert len(labels)==sum(len(v) for v in config['labels'].values())
    flagged={r['context_id'] for r in contexts if r['auto_label'] in ['use_signal','extension_signal']}
    assert set(labels)==flagged, {'unreviewed_flagged':sorted(flagged-set(labels)),'stale_labels':sorted(set(labels)-flagged)}
    reviewed=[]
    for context_id,label in labels.items():
        r=by_context[context_id]
        reviewed.append(r | {'reviewed_label':label,'reviewer':'assistant_context_review','review_note':config['notes'].get(context_id,'')})
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    by_id={p['id']:p for p in papers}
    with (ROOT/'data/citation_intent_edges.csv').open(encoding='utf-8-sig',newline='') as f:coverage=list(csv.DictReader(f))
    weights={'extension':2.0,'method_use':1.0,'component_use':.5}
    edge_evidence=defaultdict(list)
    for r in reviewed:edge_evidence[r['citing'],r['cited']].append(r)
    accepted_edges=[];scores=defaultdict(float);counts=defaultdict(Counter)
    for (u,v),evidence in sorted(edge_evidence.items()):
        strongest=max(evidence,key=lambda r:weights.get(r['reviewed_label'],0))
        score=weights.get(strongest['reviewed_label'],0)
        if score:
            scores[v]+=score;counts[v][strongest['reviewed_label']]+=1
            accepted_edges.append({'citing':u,'cited':v,'weight':score,'label':strongest['reviewed_label'],
                                   'context_id':strongest['context_id'],'source_url':strongest['source_url'],
                                   'reviewer':'assistant_context_review','chronological_anomaly':by_id[u]['date']<by_id[v]['date']})
    rows=[]
    for p in papers:
        incoming=[e for e in coverage if e['cited']==p['id']]
        rows.append({'id':p['id'],'title':p['title'],'pilot_reuse_score':scores[p['id']],
                     'extension_edges':counts[p['id']]['extension'],'method_use_edges':counts[p['id']]['method_use'],
                     'component_use_edges':counts[p['id']]['component_use'],
                     'incoming_citation_edges':len(incoming),'incoming_edges_with_contexts':sum(int(e['contexts'])>0 for e in incoming),
                     'reviewed_incoming_edges':sum(v==p['id'] for u,v in edge_evidence),
                     'rank_pagerank':p['rank_pagerank'],'rank_local_citations':p['rank_local_citations'],
                     'status':'partial_assistant_review_not_complete_true_reuse_count'})
    for i,r in enumerate(sorted(rows,key=lambda r:(-r['pilot_reuse_score'],r['id'])),1):r['rank_pilot_reuse_score']=i
    sensitivity=[]
    for extension_weight in [1.0,1.5,2.0]:
        for component_weight in [0.0,.5,1.0]:
            values={r['id']:extension_weight*r['extension_edges']+r['method_use_edges']+component_weight*r['component_use_edges'] for r in rows}
            sensitivity.append({'extension_weight':extension_weight,'component_weight':component_weight,
                                'top5_ids':sorted(values,key=lambda u:(-values[u],u))[:5]})
    summary={'contexts_extracted':len(contexts),'citation_edges':len(coverage),
             'edges_with_at_least_one_context':sum(int(e['contexts'])>0 for e in coverage),
             'reviewed_flagged_contexts':len(reviewed),'reviewed_unique_edges':len(edge_evidence),
             'positive_method_or_component_or_extension_contexts':sum(r['reviewed_label'] in weights for r in reviewed),
             'accepted_reuse_edges':len(accepted_edges),'positive_scoring_papers':sum(r['pilot_reuse_score']>0 for r in rows),
             'review_label_counts':dict(Counter(labels.values())),
             'flagged_sample_adoption_fraction':sum(r['reviewed_label'] in weights for r in reviewed)/len(reviewed),
             'score':'R(v)=sum_u max_reviewed_context_weight(u,v); extension=2, method=1, component=0.5; each citing paper counted once.',
             'sensitivity':sensitivity,
             'limits':['Only heuristic-positive contexts were reviewed; recall and a held-out classifier quality are not estimated.',
                'The assistant made the context judgments; they are not independent human validation.',
                'Unreviewed or missing contexts are unknown, not confirmed non-use.',
                'Ranks compare observed evidence under unequal coverage, not complete true semantic influence.',
                'Weights are explicit modeling choices, not empirically calibrated values.']}
    write_csv(ROOT/'data/reviewed_citation_contexts.csv',reviewed)
    write_csv(ROOT/'data/reuse_edges.csv',accepted_edges)
    write_csv(ROOT/'data/semantic_rankings.csv',sorted(rows,key=lambda r:r['rank_pilot_reuse_score']))
    (ROOT/'output/semantic_metrics_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['sensitivity','limits']},ensure_ascii=False,indent=2))
    for r in sorted(rows,key=lambda r:r['rank_pilot_reuse_score'])[:10]:print(r['rank_pilot_reuse_score'],r['pilot_reuse_score'],r['title'])

if __name__=='__main__':main()
