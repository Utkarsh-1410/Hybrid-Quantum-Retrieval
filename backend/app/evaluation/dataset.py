from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import httpx

from app.evaluation.models import BenchmarkDataset, BenchmarkDocument

SCIFACT_URL = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)


def ensure_scifact_dataset(cache_dir: Path) -> Path:
    """Download and extract the official BEIR SciFact dataset if necessary."""
    dataset_dir = cache_dir / "scifact"
    if _is_dataset_directory(dataset_dir):
        return dataset_dir

    cache_dir.mkdir(parents=True, exist_ok=True)
    archive_path = cache_dir / "scifact.zip"
    if not archive_path.exists():
        partial_path = archive_path.with_suffix(".zip.part")
        try:
            with (
                httpx.stream(
                    "GET",
                    SCIFACT_URL,
                    follow_redirects=True,
                    timeout=120,
                ) as response,
                partial_path.open("wb") as destination,
            ):
                response.raise_for_status()
                for chunk in response.iter_bytes():
                    destination.write(chunk)
            partial_path.replace(archive_path)
        except Exception:
            partial_path.unlink(missing_ok=True)
            raise

    with ZipFile(archive_path) as archive:
        _safe_extract(archive, cache_dir)

    if not _is_dataset_directory(dataset_dir):
        raise ValueError("Downloaded archive does not contain a SciFact dataset")
    return dataset_dir


def load_beir_dataset(
    dataset_dir: Path,
    *,
    name: str = "scifact",
    split: str = "test",
) -> BenchmarkDataset:
    corpus_path = dataset_dir / "corpus.jsonl"
    queries_path = dataset_dir / "queries.jsonl"
    qrels_path = dataset_dir / "qrels" / f"{split}.tsv"
    missing = [
        path for path in (corpus_path, queries_path, qrels_path) if not path.is_file()
    ]
    if missing:
        names = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing BEIR dataset files: {names}")

    corpus: dict[str, BenchmarkDocument] = {}
    for record in _read_jsonl(corpus_path):
        document_id = str(record["_id"])
        corpus[document_id] = BenchmarkDocument(
            document_id=document_id,
            title=str(record.get("title", "")),
            text=str(record.get("text", "")),
        )

    queries = {
        str(record["_id"]): str(record["text"]) for record in _read_jsonl(queries_path)
    }
    qrels = _read_qrels(qrels_path)
    return BenchmarkDataset(
        name=name,
        corpus=corpus,
        queries=queries,
        qrels=qrels,
    )


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            yield value


def _read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        expected = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError(f"{path} has an invalid qrels header")
        for row in reader:
            query_id = row["query-id"]
            document_id = row["corpus-id"]
            qrels.setdefault(query_id, {})[document_id] = int(row["score"])
    return qrels


def _safe_extract(archive: ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe archive path: {member.filename}")
    archive.extractall(destination)


def _is_dataset_directory(path: Path) -> bool:
    return (
        (path / "corpus.jsonl").is_file()
        and (path / "queries.jsonl").is_file()
        and (path / "qrels" / "test.tsv").is_file()
    )
