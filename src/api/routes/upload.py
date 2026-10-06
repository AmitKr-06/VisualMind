"""
POST /upload — accept a PDF or image, ingest it, register a session.

Uses FastAPI's File/Form for multipart uploads.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.config import UPLOADS_DIR
from src.api.schemas import UploadResponse
from src.api.session_store import (
    create_session, register_file, get_session, list_sessions, delete_session,
    Session,
)
from src.ingestion.runtime_ingest import ingest_uploaded_pdf

logger = logging.getLogger("visualmind.api.routes.upload")

router = APIRouter(tags=["upload"])

# Allowed file types
ALLOWED_PDF  = {"application/pdf"}
ALLOWED_IMG  = {"image/jpeg", "image/png", "image/webp"}


def _save_upload(file: UploadFile, session_id: str) -> Path:
    """Save the uploaded file to data/uploads/{session_id}/."""
    dir_ = UPLOADS_DIR / session_id
    dir_.mkdir(parents=True, exist_ok=True)
    safe_name = Path(file.filename or "file").name
    dst = dir_ / safe_name
    # Async read into bytes, then write (Starlette handles the streaming)
    import shutil
    with dst.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return dst


@router.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(..., description="PDF or image to upload"),
    session_id: Optional[str] = Form(None, description="Optional session id"),
):
    """Upload a PDF or image, ingest it, and attach it to a session."""
    try:
        # 1. Session
        sid = session_id or f"session-{uuid.uuid4().hex[:12]}"
        sess = get_session(sid) or create_session(sid)

        # 2. Save file
        saved = _save_upload(file, sid)
        content_type = (file.content_type or "").lower()

        # 3. Route by type
        if content_type in ALLOWED_PDF:
            # PDF ingestion
            chunks = ingest_uploaded_pdf(saved)
            kind = "pdf"
        elif content_type in ALLOWED_IMG:
            # Image ingestion — Gemini Vision description becomes a chunk
            from src.vision.image_analyzer import analyze_query_image
            from src.ingestion.preprocessing import Chunk

            analysis = analyze_query_image(saved)
            if analysis is None:
                raise HTTPException(422, "Vision model could not read this image")

            chunk = Chunk(
                chunk_id=f"upload_{sid}_{saved.stem}",
                text=analysis.get("description", "") or saved.stem,
                source=saved.name,
                page=1,
                section="uploaded image",
                chunk_type="image",
            )
            chunks = [chunk]
            kind = "image"
        else:
            raise HTTPException(
                415,
                f"Unsupported file type: {content_type}. "
                f"Allowed: PDF, JPEG, PNG, WEBP.",
            )

        # 4. Register
        register_file(sid, saved, kind, chunks, rebuild=True)
        sess = get_session(sid)

        return UploadResponse(
            session_id=sid,
            filename=saved.name,
            kind=kind,
            chunks_added=len(chunks),
            total_chunks=sess.n_chunks if sess else len(chunks),
            ready=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload failed")
        raise HTTPException(500, str(e))


@router.get("/sessions")
def sessions_list():
    """List all active sessions."""
    from src.api.schemas import SessionInfo, SessionListResponse
    from src.api.session_store import get_session as _get

    out = []
    for sid in list_sessions():
        s = _get(sid)
        if s is None:
            continue
        out.append(SessionInfo(
            session_id=sid,
            n_files=len(s.files),
            n_chunks=s.n_chunks,
            age_seconds=round(s.age_seconds(), 1),
        ))
    return SessionListResponse(sessions=out, count=len(out))


@router.delete("/sessions/{session_id}")
def sessions_delete(session_id: str):
    """Delete a session and its files."""
    ok = delete_session(session_id)
    if not ok:
        raise HTTPException(404, f"Session not found: {session_id}")
    return {"status": "deleted", "session_id": session_id}