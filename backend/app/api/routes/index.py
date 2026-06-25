from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.dependencies import get_indexer
from app.schemas.documents import IndexJobResponse, IndexRequest, IndexResponse
from app.services.protocols import DocumentIndexer

router = APIRouter(prefix="/index", tags=["indexing"])
IndexerDependency = Annotated[DocumentIndexer, Depends(get_indexer)]


@router.post("", response_model=IndexResponse, status_code=status.HTTP_202_ACCEPTED)
async def index_documents(
    payload: IndexRequest,
    indexer: IndexerDependency,
) -> IndexResponse:
    job_id = await indexer.submit(payload.documents)
    return IndexResponse(
        job_id=UUID(job_id),
        status="pending",
        accepted_count=len(payload.documents),
    )


@router.get("/{job_id}", response_model=IndexJobResponse)
async def get_index_job(
    job_id: UUID,
    indexer: IndexerDependency,
) -> IndexJobResponse:
    job = await indexer.get_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Indexing job not found",
        )
    return IndexJobResponse(
        job_id=job.job_id,
        status=job.status,
        requested_count=job.requested_count,
        indexed_count=job.indexed_count,
        skipped_count=job.skipped_count,
        failed_count=job.failed_count,
        index_version=job.index_version,
        error_message=job.error_message,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )
