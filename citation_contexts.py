"""Extract local citation contexts; conservatively flag explicit use/extension.

Regex flags are not validated semantic labels. All unflagged or inaccessible
edges remain UNKNOWN for reuse rather than becoming negative observations.
"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
import pymupdf
from analyze import write_csv

ROOT=Path(__file__).resolve().parent
ALIASES={
    'W2581624817':['sparsely-gated MoE','sparsely-gated mixture-of-experts'],
    'W3040573126':['GShard'],'W3119866685':['Switch Transformer','Switch Transformers'],
    'W4402671950':['DeepSeekMoE'],'W4200634402':['GLaM'],
    'W7133198209':['Expert Choice','expert-choice routing'],
    'W4390723197':['Mixtral'],'W4396815229':['DeepSeek-V2'],
    'W4225390052':['ST-MoE'],'W4224299101':['StableMoE'],
    'W4226515448':['DeepSpeed-MoE'],'W4310510250':['MegaBlocks'],
}
CITATION_YEARS={'W3119866685':[2021,2022],'W3040573126':[2020,2021],'W4200634402':[2021,2022]}
USE=re.compile(r'\b(?:we|our\s+(?:model|method|architecture|system|approach))\b.{0,120}?\b(?:adopt|use|utilize|employ|follow|incorporate|implement|reuse|integrate|based on)\b|\bfollowing\b.{0,80}?\bwe\b',re.I)
EXTEND=re.compile(r'\bwe\b.{0,100}?\b(?:extend|build (?:up)?on|generalize|modify|adapt)\b|\bour\b.{0,70}?\b(?:extends|builds (?:up)?on|generalizes)\b',re.I)
COMPARE=re.compile(r'\bbaselines?\b|\bwe compare\b|\bcompared (?:to|with|against)\b|\boutperforms?\b|\bcomparison (?:with|to)\b',re.I)
NEGATE=re.compile(r'\b(?:do not|does not|not use|not adopt|unlike|instead of|rather than)\b',re.I)

def normalized_offsets(text):
    normalized=[];offsets=[]
    for i,char in enumerate(text):
        for c in unicodedata.normalize('NFKD',char).lower():
            if c.isascii() and c.isalnum():normalized.append(c);offsets.append(i)
    return ''.join(normalized),offsets

def clean(text):
    text=unicodedata.normalize('NFKC',text)
    text=re.sub(r'(\w)-\n(\w)',r'\1\2',text)
    return re.sub(r'\s+',' ',text).strip()

def reference_number(bib,title):
    norm,offsets=normalized_offsets(bib)
    target,_=normalized_offsets(title)
    index=norm.find(target)
    if index<0:return None
    start=offsets[index]
    previous=list(re.finditer(r'(?m)^\s*\[(\d+)\]',bib[:start]))
    if previous and start-previous[-1].end()<1600:return previous[-1].group(1)
    return None

def aliases(p):
    result=ALIASES.get(p['id'].rsplit('/',1)[-1],[]).copy()
    prefix=p['title'].split(':',1)[0]
    if ':' in p['title'] and len(prefix)<=28 and len(prefix.split())<=3:result.append(prefix)
    return sorted(set(result),key=lambda s:(-len(s),s))

def heading_at(raw,position):
    prefix=raw[:position]
    found=list(re.finditer(r'(?m)^\s*(?:\d+(?:\.\d+)*[.)]?\s+)?(Related Work|Background|Introduction|Methods?|Methodology|Architecture|Approach|Experiments?|Evaluation|Conclusion|Model(?: Architecture)?|Training|Routing)(?:\s*\n|\s*$)',prefix,re.I))
    return found[-1].group(1) if found else 'unresolved'

def main():
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    by_id={p['id']:p for p in papers}
    manifest=json.loads((ROOT/'data/pdf_reference_manifest.json').read_text(encoding='utf-8'))
    pdf_map={p['id']:p['arxiv'] for p in manifest}
    with (ROOT/'data/edges.csv').open(encoding='utf-8-sig',newline='') as f:edges=list(csv.DictReader(f))
    outgoing=defaultdict(list)
    for edge in edges:outgoing[edge['citing']].append(edge['cited'])
    rows=[];edge_rows=[]
    for u,targets in sorted(outgoing.items()):
        arxiv=pdf_map.get(u)
        if not arxiv or not (path:=ROOT/'data/paper_pdfs'/f'{arxiv}.pdf').exists():
            for v in targets:edge_rows.append({'citing':u,'cited':v,'status':'no_pdf','contexts':0,'auto_label':'unknown'})
            continue
        with pymupdf.open(path) as doc:pages=[p.get_text() for p in doc]
        full='\n'.join(pages)
        refs=list(re.finditer(r'(?im)^\s*(?:\d+\s+)?references\s*$',full))
        refs=next((m for m in refs if m.start()>1000),None)
        if not refs:
            for v in targets:edge_rows.append({'citing':u,'cited':v,'status':'no_reference_boundary','contexts':0,'auto_label':'unknown'})
            continue
        body=full[:refs.start()];bib=full[refs.end():]
        for v in sorted(targets):
            p=by_id[v];number=reference_number(bib,p['title'])
            patterns=[]
            if number:
                # Numeric citation groups, e.g. [4, 9, 17]; no ranges inferred.
                patterns.append(('numeric',rf'\[(?:\d+\s*,\s*)*{number}(?:\s*,\s*\d+)*\]'))
            if p['authors']:
                first=p['authors'].split(';')[0].strip().split()[-1]
                if len(first)>=3:
                    years='|'.join(str(y) for y in CITATION_YEARS.get(v.rsplit('/',1)[-1],[p['year']]))
                    ambiguous=sum(bool(by_id[t]['authors']) and by_id[t]['authors'].split(';')[0].strip().split()[-1]==first for t in targets)>1
                    patterns.append(('author_year_ambiguous' if ambiguous else 'author_year',rf'\b{re.escape(first)}\b[^.\n]{{0,14}}(?:et\s+al\.?)?[^\n]{{0,10}}(?:{years})[a-z]?'))
            for alias in aliases(p):patterns.append(('name',re.escape(alias)))
            found=[]
            for mode,pattern in patterns:
                for m in re.finditer(pattern,body,re.I):
                    if any(abs(m.start()-old[0])<120 for old in found):continue
                    found.append((m.start(),m.end(),mode))
            edge_contexts=[]
            for start,end,mode in sorted(found):
                # Keep source text exact; classification sees only nearby sentence.
                left=max(0,start-430);right=min(len(body),end+430)
                excerpt=clean(body[left:right])
                center_left=max(body.rfind('. ',0,start),body.rfind('.\n',0,start),body.rfind('\n\n',0,start))
                center_left=max(center_left+2,start-280,0)
                stop=re.search(r'[.!?](?:\s|$)',body[end:])
                center_right=min(end+(stop.end() if stop else 250),end+350)
                sentence=clean(body[center_left:center_right])
                section=heading_at(body,start)
                if mode=='author_year_ambiguous' and not any(re.search(re.escape(a),sentence,re.I) for a in aliases(p)):label='unknown_author_year_collision'
                elif NEGATE.search(sentence):label='unknown_negated_or_contrast'
                elif COMPARE.search(sentence):label='comparison_or_baseline_signal'
                elif section.lower() in ['related work','background']:label='background_section'
                elif EXTEND.search(sentence):label='extension_signal'
                elif USE.search(sentence):label='use_signal'
                else:label='unknown'
                page=1
                offset=0
                for i,text in enumerate(pages,1):
                    if offset+len(text)>=start:page=i;break
                    offset+=len(text)+1
                context_id=hashlib.sha256(f'{u}|{v}|{start}'.encode()).hexdigest()[:16]
                row={'context_id':context_id,'citing':u,'cited':v,'citing_title':by_id[u]['title'],'cited_title':p['title'],
                     'pdf_page':page,'section':section,'match_mode':mode,'auto_label':label,
                     'sentence':sentence,'context':excerpt,'source_url':f'https://arxiv.org/pdf/{arxiv}#page={page}',
                     'status':'heuristic_unvalidated','body_char_start':start}
                rows.append(row);edge_contexts.append(row)
            labels={r['auto_label'] for r in edge_contexts}
            strongest=next((label for label in ['extension_signal','use_signal','comparison_or_baseline_signal','background_section'] if label in labels),'unknown')
            edge_rows.append({'citing':u,'cited':v,'status':'contexts_found' if found else 'no_context_match',
                              'contexts':len(found),'auto_label':strongest})
    write_csv(ROOT/'data/citation_contexts.csv',rows)
    write_csv(ROOT/'data/citation_intent_edges.csv',edge_rows)
    (ROOT/'data/citation_contexts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    counts=defaultdict(int)
    for r in rows:counts[r['auto_label']]+=1
    print(json.dumps({'edges':len(edge_rows),'contexts':len(rows),'auto_context_labels':dict(counts)},indent=2))

if __name__=='__main__':main()
