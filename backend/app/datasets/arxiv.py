from __future__ import annotations

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from app.domain.models import RetrievalCandidate
from app.schemas.documents import DocumentInput

WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class ArxivFilter:
    categories: tuple[str, ...] = ()
    updated_from: date | None = None
    updated_to: date | None = None
    require_abstract: bool = True

    def matches(self, record: ArxivRecord) -> bool:
        if self.require_abstract and not record.abstract:
            return False
        if self.categories and not any(
            _category_matches(category, requested)
            for category in record.categories
            for requested in self.categories
        ):
            return False
        if self.updated_from and (
            record.update_date is None or record.update_date < self.updated_from
        ):
            return False
        return not (
            self.updated_to
            and (record.update_date is None or record.update_date > self.updated_to)
        )


@dataclass(frozen=True, slots=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: str
    categories: tuple[str, ...]
    update_date: date | None
    doi: str | None = None
    journal_reference: str | None = None
    license_url: str | None = None
    comments: str | None = None
    versions: tuple[dict[str, str], ...] = ()

    @property
    def canonical_url(self) -> str:
        return f"https://arxiv.org/abs/{self.arxiv_id}"

    def to_document_input(self) -> DocumentInput:
        return DocumentInput(
            external_id=self.arxiv_id,
            title=self.title,
            content=self.abstract,
            canonical_url=self.canonical_url,
            source="arxiv",
            language="en",
            metadata=self.metadata(),
        )

    def to_retrieval_candidate(self) -> RetrievalCandidate:
        return RetrievalCandidate(
            chunk_id=uuid5(
                NAMESPACE_URL,
                f"arxiv:abstract:{self.arxiv_id}",
            ),
            document_id=uuid5(
                NAMESPACE_URL,
                f"arxiv:paper:{self.arxiv_id}",
            ),
            title=self.title,
            content=self.abstract,
            url=self.canonical_url,
            metadata=self.metadata(),
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "language": "en",
            "source": "arxiv",
            "arxiv_id": self.arxiv_id,
            "authors": self.authors,
            "categories": list(self.categories),
            "primary_category": (self.categories[0] if self.categories else None),
            "update_date": (self.update_date.isoformat() if self.update_date else None),
            "doi": self.doi,
            "journal_reference": self.journal_reference,
            "license": self.license_url,
            "comments": self.comments,
            "versions": list(self.versions),
        }


def iter_arxiv_records(
    path: Path,
    *,
    filters: ArxivFilter | None = None,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> Iterator[ArxivRecord]:
    """Stream records from the newline-delimited arXiv snapshot."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    active_filters = filters or ArxivFilter()
    yielded = 0

    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record is not a JSON object")
                record = parse_arxiv_record(payload)
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                if skip_invalid:
                    continue
                raise ValueError(
                    f"Invalid arXiv record at {path}:{line_number}"
                ) from None
            if not active_filters.matches(record):
                continue
            yield record
            yielded += 1
            if limit is not None and yielded >= limit:
                return


def iter_arxiv_documents(
    path: Path,
    *,
    filters: ArxivFilter | None = None,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> Iterator[DocumentInput]:
    for record in iter_arxiv_records(
        path,
        filters=filters,
        limit=limit,
        skip_invalid=skip_invalid,
    ):
        yield record.to_document_input()


def iter_arxiv_candidates(
    path: Path,
    *,
    filters: ArxivFilter | None = None,
    limit: int | None = None,
    skip_invalid: bool = False,
) -> Iterator[RetrievalCandidate]:
    for record in iter_arxiv_records(
        path,
        filters=filters,
        limit=limit,
        skip_invalid=skip_invalid,
    ):
        yield record.to_retrieval_candidate()


def parse_arxiv_record(payload: dict[str, Any]) -> ArxivRecord:
    arxiv_id = _required_text(payload, "id")
    categories = tuple(_text(payload.get("categories")).split())
    versions = tuple(
        {
            "version": _text(version.get("version")),
            "created": _text(version.get("created")),
        }
        for version in _mapping_sequence(payload.get("versions"))
    )
    return ArxivRecord(
        arxiv_id=arxiv_id,
        title=_required_text(payload, "title"),
        abstract=_text(payload.get("abstract")),
        authors=_text(payload.get("authors")),
        categories=categories,
        update_date=_parse_date(payload.get("update_date")),
        doi=_optional_text(payload.get("doi")),
        journal_reference=_optional_text(payload.get("journal-ref")),
        license_url=_optional_text(payload.get("license")),
        comments=_optional_text(payload.get("comments")),
        versions=versions,
    )


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = _text(payload.get(key))
    if not value:
        raise ValueError(f"{key} is required")
    return value


def _text(value: object) -> str:
    return WHITESPACE.sub(" ", value).strip() if isinstance(value, str) else ""


def _optional_text(value: object) -> str | None:
    text = _text(value)
    return text or None


def _parse_date(value: object) -> date | None:
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise TypeError("update_date must be a string")
    return date.fromisoformat(value)


def _mapping_sequence(value: object) -> Sequence[dict[str, Any]]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TypeError("versions must be a list of objects")
    return value


def _category_matches(category: str, requested: str) -> bool:
    return category == requested or category.startswith(f"{requested}.")
