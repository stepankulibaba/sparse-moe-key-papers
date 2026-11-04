"""Recover within-corpus edges by exact titles in all available arXiv bibliographies.

Only positive matches are evidence. An unmatched title is NOT a negative citation.
Downloads are cached; OpenAlex remains the source of global citation counts.
"""
from __future__ import annotations
import concurrent.futures
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
import requests
import pymupdf
from analyze import normalize, write_csv

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'data' / 'paper_pdfs'

def fetch(item):
    paper, arxiv = item
    path = CACHE / f'{arxiv}.pdf'
    meta_path = CACHE / f'{arxiv}.json'
    try:
        if not path.exists():
            response = requests.get(f'https://arxiv.org/pdf/{arxiv}', timeout=65)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError('Response is not a PDF')
            path.write_bytes(response.content)
            meta_path.write_text(json.dumps({'url':response.url, 'retrieved_at_utc':datetime.now(timezone.utc).isoformat(),
                'sha256':hashlib.sha256(response.content).hexdigest()}, indent=2), encoding='utf-8')
            time.sleep(1.5)
        with pymupdf.open(path) as doc:
            pages = [p.get_text() for p in doc]
        full = '\n'.join(pages)
        headings = list(re.finditer(r'(?im)^\s*(?:\d+\s+)?references\s*$', full))
        if not headings:
            return {'id':paper['id'],'arxiv':arxiv,'status':'no_reference_heading','edges':[]}
        # First standalone heading after half of page one; ignore contents entries.
        heading = next((m for m in headings if m.start() > 1000), headings[0])
        bib = full[heading.end():]
        end = re.search(r'(?im)^\s*(?:appendix|appendices|A\s+Appendix|A\s+Additional|A\s+Experimental)\b', bib)
        if end:
            bib = bib[:end.start()]
        bib_norm = normalize(bib)
        return {'id':paper['id'],'arxiv':arxiv,'status':'parsed','bib_normalized':bib_norm,
                'bibliography_text':bib,'edges':[]}
    except (requests.RequestException, ValueError, RuntimeError) as exc:
        return {'id':paper['id'],'arxiv':arxiv,'status':'unavailable','error':str(exc),'edges':[]}

def main():
    CACHE.mkdir(exist_ok=True)
    papers = json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    raw = {p['id']:p for f in (ROOT/'data/raw').glob('*.json') for p in json.loads(f.read_text(encoding='utf-8'))['body'].get('results',[])}
    items = []
    for p in papers:
        versions = ' '.join(json.dumps(raw[v].get('locations',[])) for v in p['version_ids'].split(';'))
        matches = re.findall(r'arxiv.org/(?:abs|pdf)/(\d{4}\.\d{4,5})',versions,re.I)
        if matches:
            items.append((p, matches[0]))
    edges, manifest = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for index,result in enumerate(pool.map(fetch,items),1):
            text = result.pop('bibliography_text','')
            normalized = result.pop('bib_normalized','')
            if text:
                (CACHE / (result['arxiv'] + '.references.txt')).write_text(text, encoding='utf-8')
            for p in papers:
                target = normalize(p['title'])
                if p['id'] != result['id'] and len(target) >= 30 and target in normalized:
                    edges.append({'citing':result['id'], 'cited':p['id'], 'source':'arxiv_bibliography_exact_title',
                                  'source_url':f"https://arxiv.org/pdf/{result['arxiv']}",
                                  'matched_title':p['title'], 'evidence_file':f"data/paper_pdfs/{result['arxiv']}.references.txt"})
            result.pop('edges',None)
            result.pop('reference_start_pdf_page',None)
            manifest.append(result)
            print(f"{index}/{len(items)} {result['arxiv']} {result['status']} total_edges={len(edges)}",flush=True)
    write_csv(ROOT/'data/pdf_edges.csv',edges,['citing','cited','source','source_url','matched_title','evidence_file'])
    (ROOT/'data/pdf_reference_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__ == '__main__':
    main()
