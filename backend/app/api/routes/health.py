from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def readiness(request: Request) -> dict[str, str]:
    configured = bool(getattr(request.app.state, "search_service", None))
    return {"status": "ready" if configured else "not_ready"}
