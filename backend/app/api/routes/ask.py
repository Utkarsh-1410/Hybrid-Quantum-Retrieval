from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_ask_service
from app.schemas.ask import AskRequest, AskResponse
from app.services.ask import AskService

router = APIRouter(prefix="/ask", tags=["rag"])
AskServiceDependency = Annotated[AskService, Depends(get_ask_service)]


@router.post("", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    service: AskServiceDependency,
) -> AskResponse:
    return await service.ask(payload)
