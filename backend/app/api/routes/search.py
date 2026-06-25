from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_search_service
from app.schemas.search import SearchRequest, SearchResponse
from app.services.search import SearchService

router = APIRouter(prefix="/search", tags=["search"])

SearchServiceDependency = Annotated[
    SearchService,
    Depends(get_search_service),
]


@router.post("", response_model=SearchResponse)
async def search(
    payload: SearchRequest,
    service: SearchServiceDependency,
) -> SearchResponse:
    print("SEARCH ENDPOINT HIT")
    return await service.search(payload)