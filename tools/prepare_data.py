"""Восстановить компактные входы из исходного снимка в истории Git.

Запуск из корня: python -X utf8 tools/prepare_data.py
Интернет не нужен. Требуются Git, NetworkX и SciPy.
Сначала исходный анализ выполняется во временной папке, затем сохраняются
корпус, аудит отбора и свидетельства цитирования без тяжёлого кэша API.
"""

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "527258167d3a1cdad876f5c14a3079c7c120621b"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def restore_original(destination):
    archive = subprocess.check_output(
        ["git", "archive", "--format=tar", SOURCE_COMMIT], cwd=ROOT
    )
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        for member in source.getmembers():
            if not member.isfile():
                continue
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unexpected archive path: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.extractfile(member).read())
    subprocess.run(
        [sys.executable, "-X", "utf8", "analyze.py"],
        cwd=destination,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def make_snapshot(original):
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    papers = read_json(original / "data/papers.json")
    version_to_paper = {
        version: paper["id"]
        for paper in papers
        for version in paper["version_ids"].split(";")
    }
    authors = {paper["id"]: set() for paper in papers}
    sources = {paper["id"]: set() for paper in papers}
    incomplete = set()
    requests = []
    for path in sorted((original / "data/raw").glob("*.json")):
        response = read_json(path)
        requests.append(
            {
                "file": path.relative_to(original).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "url": response["url"],
                "retrieved_at_utc": response["retrieved_at_utc"],
                "returned": len(response["body"].get("results", [])),
            }
        )
        for work in response["body"].get("results", []):
            key = version_to_paper.get(work["id"])
            if key is None:
                continue
            sources[key].add(path.name + ":" + work["id"])
            byline = work.get("authorships") or []
            if not byline or len(byline) >= 100:
                incomplete.add(key)
            for entry in byline:
                author_id = (entry.get("author") or {}).get("id")
                if author_id:
                    authors[key].add(author_id)
                else:
                    incomplete.add(key)

    corpus = []
    for paper in papers:
        # Ранги и локальные оценки вычисляются заново, а не входят во входы.
        row = {
            key: value
            for key, value in paper.items()
            if not key.startswith("rank_")
            and key not in ("pagerank", "local_citations")
        }
        row["observed_author_ids"] = sorted(authors[paper["id"]])
        row["incomplete_author_metadata"] = (
            paper["id"] in incomplete or not authors[paper["id"]]
        )
        row["author_evidence_sources"] = sorted(sources[paper["id"]])
        corpus.append(row)
    corpus.sort(key=lambda paper: paper["title"].casefold())
    save_json(data / "corpus.json", corpus)
    (data / "scope_audit.csv").write_bytes(
        (original / "data/corpus_audit.csv").read_bytes()
    )

    bibliographies = []
    for item in read_json(original / "data/pdf_reference_manifest.json"):
        item = item.copy()
        stem = original / "data/paper_pdfs" / item["arxiv"]
        # arXiv identifiers contain a dot; append the extension literally.
        metadata = Path(str(stem) + ".json")
        if metadata.exists():
            item.update(read_json(metadata))
        text = Path(str(stem) + ".references.txt")
        item["lines"] = (
            text.read_text(encoding="utf-8").split("\n") if text.exists() else []
        )
        bibliographies.append(item)
    save_json(data / "bibliographies.json", bibliographies)
    save_csv(data / "contexts.csv", read_json(original / "data/citation_contexts.json"))
    (data / "context_coverage.csv").write_bytes(
        (original / "data/citation_intent_edges.csv").read_bytes()
    )
    settings = {
        "selection": read_json(original / "author_decisions.json"),
        "context_review": read_json(original / "context_review_labels.json"),
    }
    save_json(ROOT / "settings.json", settings)
    provenance = {
        "source_commit": SOURCE_COMMIT,
        "snapshot_date": "2026-11-04",
        "scope": "Frozen normalized corpus. Original collection and filtering are reproducible from source_commit.",
        "original_corpus_summary": read_json(original / "output/summary.json"),
        "openalex_requests": requests,
        "input_sha256": {
            path.relative_to(ROOT)
            .as_posix(): hashlib.sha256(path.read_bytes())
            .hexdigest()
            for path in [
                data / name
                for name in (
                    "corpus.json",
                    "scope_audit.csv",
                    "bibliographies.json",
                    "contexts.csv",
                    "context_coverage.csv",
                )
            ]
        },
    }
    save_json(data / "provenance.json", provenance)
    print(
        "Prepared 140 papers, selection audit, bibliographies and citation contexts in data/."
    )


def main():
    with tempfile.TemporaryDirectory(prefix="moe-original-") as temporary:
        original = Path(temporary)
        restore_original(original)
        make_snapshot(original)


if __name__ == "__main__":
    main()
