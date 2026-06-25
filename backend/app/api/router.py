from fastapi import APIRouter

from app.api.routes import ask, index, search

api_router = APIRouter()
api_router.include_router(index.router)
api_router.include_router(search.router)
api_router.include_router(ask.router)
