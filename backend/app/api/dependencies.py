'''from typing import cast

from fastapi import Request

from app.services.ask import AskService
from app.services.protocols import DocumentIndexer
from app.services.search import SearchService


def get_search_service(request: Request) -> SearchService:
    return cast(SearchService, request.app.state.search_service)


def get_ask_service(request: Request) -> AskService:
    return cast(AskService, request.app.state.ask_service)


def get_indexer(request: Request) -> DocumentIndexer:
    return cast(DocumentIndexer, request.app.state.indexer)
'''

from typing import cast

from fastapi import Request

from app.services.ask import AskService
from app.services.protocols import DocumentIndexer
from app.services.search import SearchService


def get_search_service(request: Request) -> SearchService:
    print("DEPENDENCY CALLED")
    print(type(request.app.state.search_service))
    return cast(SearchService, request.app.state.search_service)


def get_ask_service(request: Request) -> AskService:
    return cast(AskService, request.app.state.ask_service)


def get_indexer(request: Request) -> DocumentIndexer:
    return cast(DocumentIndexer, request.app.state.indexer)