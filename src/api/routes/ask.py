"""/ask and /ask/batch endpoints."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    AskRequest, AskResponse,
    BatchRequest, BatchResponse,
)
from src.api.service import ask, ask_batch

logger = logging.getLogger("visualmind.api.routes.ask")

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    """Ask a single question."""
    try:
        out = ask(
            question=req.question,
            thread_id=req.thread_id,
            session_id=req.session_id,
            image_path=req.image_path,
        )
        return AskResponse(**out)
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask/batch", response_model=BatchResponse)
def ask_batch_endpoint(req: BatchRequest):
    """Ask multiple questions in one request."""
    try:
        out = ask_batch(req.questions)
        return BatchResponse(**out)
    except Exception as e:
        logger.exception("ask_batch failed")
        raise HTTPException(status_code=500, detail=str(e))