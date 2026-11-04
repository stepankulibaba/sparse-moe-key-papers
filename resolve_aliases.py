"""Fetch unresolved cited IDs in batches, to detect versions omitted by title search.

These records do not expand the candidate pool. Only exact title/DOI matches can
join an existing candidate. All other fetched records remain outside the corpus.
"""
import json
from pathlib import Path
from collect import request_cached, FIELDS

ROOT=Path(__file__).resolve().parent

def main():
    papers=json.loads((ROOT/'data/papers.json').read_text(encoding='utf-8'))
    known={v for p in papers for v in p['version_ids'].split(';')}
    ids=sorted({v for p in papers for v in p['reference_ids']} - known)
    for i in range(0,len(ids),100):
        batch=ids[i:i+100]
        request_cached(f'alias_batch_{i//100:03}', 'https://api.openalex.org/works', {
            'filter':'openalex_id:'+'|'.join(v.rsplit('/',1)[-1] for v in batch),
            'per_page':100,'select':FIELDS})

if __name__=='__main__':main()
