import json
from datetime import date
from pathlib import Path

import pytest

from app.datasets.arxiv import (
    ArxivFilter,
    iter_arxiv_candidates,
    iter_arxiv_records,
)


def write_records(path: Path) -> None:
    records = [
        {
            "id": "2401.00001",
            "title": "  Quantum\n Retrieval ",
            "abstract": " A contextual search method. ",
            "authors": "A. Researcher",
            "categories": "cs.IR cs.AI",
            "update_date": "2024-01-02",
            "doi": "10.1/example",
            "versions": [
                {
                    "version": "v1",
                    "created": "Mon, 1 Jan 2024 00:00:00 GMT",
                }
            ],
        },
        {
            "id": "2401.00002",
            "title": "Physics",
            "abstract": "Collider result.",
            "authors": "B. Scientist",
            "categories": "hep-ph",
            "update_date": "2024-02-01",
            "versions": [],
        },
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_arxiv_streaming_filter_and_normalization(tmp_path: Path) -> None:
    path = tmp_path / "arxiv.json"
    write_records(path)

    records = list(
        iter_arxiv_records(
            path,
            filters=ArxivFilter(
                categories=("cs",),
                updated_from=date(2024, 1, 1),
            ),
        )
    )

    assert len(records) == 1
    assert records[0].title == "Quantum Retrieval"
    assert records[0].categories == ("cs.IR", "cs.AI")


def test_arxiv_candidate_ids_are_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "arxiv.json"
    write_records(path)

    first = list(iter_arxiv_candidates(path, limit=1))[0]
    second = list(iter_arxiv_candidates(path, limit=1))[0]

    assert first.chunk_id == second.chunk_id
    assert first.document_id == second.document_id
    assert first.url == "https://arxiv.org/abs/2401.00001"
    assert first.metadata["primary_category"] == "cs.IR"


def test_invalid_arxiv_record_can_be_skipped(tmp_path: Path) -> None:
    path = tmp_path / "arxiv.json"
    path.write_text('{"title": "missing id"}\n', encoding="utf-8")

    assert list(iter_arxiv_records(path, skip_invalid=True)) == []
    with pytest.raises(ValueError, match="Invalid arXiv record"):
        list(iter_arxiv_records(path))
