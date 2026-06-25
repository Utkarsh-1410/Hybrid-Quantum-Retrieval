"""Streaming dataset adapters."""

from app.datasets.arxiv import (
    ArxivFilter,
    ArxivRecord,
    iter_arxiv_candidates,
    iter_arxiv_documents,
    iter_arxiv_records,
)

__all__ = [
    "ArxivFilter",
    "ArxivRecord",
    "iter_arxiv_candidates",
    "iter_arxiv_documents",
    "iter_arxiv_records",
]
