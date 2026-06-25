from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class DocumentInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    external_id: str | None = Field(default=None, max_length=512)
    title: str = Field(min_length=1, max_length=1000)
    content: str = Field(min_length=1)
    canonical_url: HttpUrl | None = None
    source: str | None = Field(default=None, max_length=255)
    language: str = Field(default="en", min_length=2, max_length=16)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=1000)


class IndexResponse(BaseModel):
    job_id: UUID
    status: str
    accepted_count: int


class IndexJobResponse(BaseModel):
    job_id: UUID
    status: str
    requested_count: int
    indexed_count: int
    skipped_count: int
    failed_count: int
    index_version: str | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
