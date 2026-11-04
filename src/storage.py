"""Пути, чтение таблиц и сохранённые свидетельства цитирования.

data/ содержит только входы; все пересчитываемые файлы попадают в output/.
"""

import csv
import json
import re
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "output"
TABLES = OUT / "tables"
EXPERIMENTS = OUT / "robustness"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields=None):
    if not rows and fields is None:
        raise ValueError(f"Empty table needs explicit columns: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields or list(rows[0]), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def normalize(title):
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", title).lower())


def load_contexts():
    rows = read_csv(DATA / "contexts.csv")
    for row in rows:
        row["pdf_page"] = int(row["pdf_page"])
        row["body_char_start"] = int(row["body_char_start"])
    return rows


def bibliography_edges(papers):
    """Положительные совпадения названий; отсутствие совпадения не отрицает ссылку."""
    rows = []
    for source in read_json(DATA / "bibliographies.json"):
        bibliography = normalize("\n".join(source["lines"]))
        for paper in papers:
            title = normalize(paper["title"])
            if (
                paper["id"] != source["id"]
                and len(title) >= 30
                and title in bibliography
            ):
                rows.append(
                    {
                        "citing": source["id"],
                        "cited": paper["id"],
                        "source": "arxiv_bibliography_exact_title",
                        "source_url": f"https://arxiv.org/pdf/{source['arxiv']}",
                        "matched_title": paper["title"],
                        "evidence_key": source["arxiv"],
                    }
                )
    return rows
