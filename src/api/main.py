"""
FastAPI application — VisualMind REST API.

Endpoints:
  GET  /health      — liveness
  GET  /config      — frozen pipeline metadata
  POST /ask         — single question
  POST /ask/batch   — batch of questions

Run with:
  uvicorn src.api_app:app --reload
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.dependencies import (
    get_workflow, get_metadata, get_retriever,
    get_figure_matcher, get_image_analyzer,
)
from src.api.routes import ask_router
from src.api.schemas import HealthResponse, ConfigResponse

logger = logging.getLogger("visualmind.api.main")


# ============================================================
# Lifespan — warm up singletons at startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=" * 60)
    logger.info("Starting VisualMind API — warming up...")
    logger.info("=" * 60)

    # Build every singleton ONCE so requests are fast
    get_retriever()
    get_figure_matcher()
    get_image_analyzer()
    get_workflow()

    logger.info("=" * 60)
    logger.info("VisualMind API ready")
    logger.info("=" * 60)
    yield
    logger.info("Shutting down VisualMind API")


# ============================================================
# App
# ============================================================
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


# ============================================================
# Health + Config
# ============================================================
@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    meta = get_metadata()
    return HealthResponse(
        status="ok",
        graph="VisualMindGraph",
        version="v1",
        nodes=meta["nodes"],
        models=meta["models"],
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


# ============================================================
# Mount routers
# ============================================================
app.include_router(ask_router)