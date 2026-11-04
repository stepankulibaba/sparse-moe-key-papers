"""Small, anonymous, cached OpenAlex collection. Never reads credentials."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw'
FIELDS = 'id,doi,title,publication_date,publication_year,cited_by_count,referenced_works,counts_by_year,abstract_inverted_index,authorships,primary_location,locations,type,ids'
QUERIES = [
    ('mixture_cited', 'mixture of experts', 'cited_by_count:desc', 200),
    ('mixture_recent', 'mixture of experts', 'publication_date:desc', 200),
    ('sparse_experts', 'sparse expert', 'cited_by_count:desc', 200),
    ('gshard', 'gshard', 'cited_by_count:desc', 30),
    ('switch', 'switch transformers', 'cited_by_count:desc', 30),
    ('glam', 'glam scaling', 'cited_by_count:desc', 20),
    ('mixtral', 'mixtral', 'cited_by_count:desc', 40),
    ('deepseek', 'deepseek', 'cited_by_count:desc', 100),
    ('base', 'BASE layers', 'cited_by_count:desc', 20),
    ('routing', 'expert choice routing', 'cited_by_count:desc', 30),
    ('fastmoe', 'fastmoe', 'cited_by_count:desc', 20),
    ('deepspeed', 'deepspeed moe', 'cited_by_count:desc', 20),
    ('megablocks', 'megablocks', 'cited_by_count:desc', 20),
    ('stmoe', 'st moe', 'cited_by_count:desc', 20),
    ('sparse_upcycling', 'sparse upcycling', 'cited_by_count:desc', 20),
]
SURVEYS = {'fedus2022': '2209.01667v1', 'cai2025': '2407.06204v3', 'mu_lin2025':'2503.07137v3'}

def request_cached(name: str, url: str, params: dict | None = None) -> dict:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f'{name}.json'
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    for attempt in range(4):
        response = requests.get(url, params=params, timeout=60, headers={'User-Agent': 'IntroAI-course-bibliometrics/1.0'})
        if response.status_code in (429, 500, 502, 503):
            time.sleep(min(3 * 2**attempt, 24))
            continue
        response.raise_for_status()
        result = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(), 'url': response.url,
                  'status_code': response.status_code, 'body': response.json()}
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, 'records=', len(result['body'].get('results', [])), flush=True)
        time.sleep(.5)
        return result
    raise RuntimeError(f'No successful response for {name}; existing cache remains usable')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--surveys-only', action='store_true')
    args = parser.parse_args()
    if not args.surveys_only:
        for name, term, order, limit in QUERIES:
            request_cached(name, 'https://api.openalex.org/works', {
                'filter': f'title.search:{term},from_publication_date:2017-01-01,to_publication_date:2025-12-31',
                'sort': order, 'per_page': limit, 'select': FIELDS})
    sources = ROOT / 'sources'
    sources.mkdir(exist_ok=True)
    for name, arxiv in SURVEYS.items():
        path = sources / f'{name}.pdf'
        url = f'https://arxiv.org/pdf/{arxiv}'
        if not path.exists():
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            if not response.content.startswith(b'%PDF'):
                raise ValueError(f'Not a PDF: {url}')
            path.write_bytes(response.content)
            (sources / f'{name}.provenance.json').write_text(json.dumps({
                'url': url, 'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
                'sha256': hashlib.sha256(response.content).hexdigest()}, indent=2), encoding='utf-8')
        import pymupdf
        with pymupdf.open(path) as doc:
            (sources / f'{name}.txt').write_text('\n'.join(
                f'\n=== PDF PAGE {i+1} ===\n{p.get_text()}' for i, p in enumerate(doc)), encoding='utf-8')
    files = list(RAW.glob('*.json'))
    manifest = [{'file': str(p.relative_to(ROOT)), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                 'url': (j := json.loads(p.read_text(encoding='utf-8')))['url'],
                 'retrieved_at_utc': j['retrieved_at_utc'],
                 'total_matches': j['body'].get('meta', {}).get('count'),
                 'returned': len(j['body'].get('results', []))} for p in files]
    (ROOT / 'data' / 'collection_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

if __name__ == '__main__':
    main()
