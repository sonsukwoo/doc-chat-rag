"""문서 업로드와 파이프라인 실행을 위한 FastAPI 앱."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.documents import router as documents_router
from .routes.pipeline import router as pipeline_router
from .routes.review import router as review_router


def create_app() -> FastAPI:
    """서비스용 FastAPI 앱을 생성한다."""
    app = FastAPI(
        title="Doc Chat RAG Backend",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(documents_router)
    app.include_router(pipeline_router)
    app.include_router(review_router)
    return app


app = create_app()
