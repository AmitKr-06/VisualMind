"""FastAPI application — VisualMind REST API + Web UI."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.api.dependencies import (
    get_workflow, get_metadata, get_retriever,
    get_figure_matcher, get_image_analyzer,
)
from src.api.routes import ask_router, upload_router
from src.api.schemas import HealthResponse, ConfigResponse
from src.api.ui import router as ui_router

logger = logging.getLogger("visualmind.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting VisualMind API — warming up...")
    logger.info("=" * 60)

    get_retriever()
    get_figure_matcher()
    get_image_analyzer()
    get_workflow()

    # Start the cleanup task (module 10)
    import asyncio
    from src.api.cleanup import cleanup_loop
    task = asyncio.create_task(cleanup_loop())

    logger.info("=" * 60)
    logger.info("VisualMind API ready")
    logger.info("=" * 60)
    yield
    task.cancel()
    logger.info("Shutting down VisualMind API")


app = FastAPI(
    title="VisualMind API",
    description="Multimodal RAG pipeline for NCERT Class 10 Science.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    meta = get_metadata()
    return HealthResponse(
        status="ok", graph="VisualMindGraph", version="v1",
        nodes=meta["nodes"], models=meta["models"],
    )


@app.get("/config", response_model=ConfigResponse, tags=["meta"])
def config():
    meta = get_metadata()
    policy = meta["policy"]
    return ConfigResponse(
        chunk_file=meta["chunk_file"],
        k_chunks=policy["k_chunks"],
        n_figures=policy["n_figures"],
        tau_text=policy["tau_text"],
        choice1=meta["choice1"],
        choice2=meta["choice2"],
        dense_model=meta["models"]["dense"],
        reranker_model=meta["models"]["reranker"],
        answer_model_chain=meta["models"]["answer_model_chain"],
        fallback_provider=meta["models"].get("fallback_provider"),
    )


# Routers
app.include_router(ask_router)
app.include_router(upload_router)
app.include_router(ui_router)

# Static files (CSS + JS)
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")