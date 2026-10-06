"""
Session registry — tracks uploaded documents per session.

Each session has its own retriever, built from the uploaded chunks.
Sessions expire after SESSION_TTL_HOURS (default 24h).
"""
from __future__ import annotations

import logging
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from src.config import UPLOADS_DIR, SESSION_TTL_HOURS, MAX_SESSIONS
from src.exception import CustomException

logger = logging.getLogger("visualmind.api.session")


# ============================================================
# Data
# ============================================================
@dataclass
class UploadedFile:
    name: str
    path: Path
    kind: str                 # "pdf" | "image"
    n_chunks: int
    uploaded_at: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    files: List[UploadedFile] = field(default_factory=list)
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    retriever: Any = None      # HybridRetriever | None

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def age_seconds(self) -> float:
        return time.time() - self.created_at


# ============================================================
# Store
# ============================================================
_STORE: Dict[str, Session] = {}
_LOCK = Lock()


# ============================================================
# API
# ============================================================
def create_session(session_id: Optional[str] = None) -> Session:
    """Create (or return) a session."""
    try:
        sid = session_id or f"session-{uuid.uuid4().hex[:12]}"
        with _LOCK:
            if sid in _STORE:
                return _STORE[sid]
            # Enforce max sessions
            if len(_STORE) >= MAX_SESSIONS:
                _evict_oldest()
            sess = Session(session_id=sid)
            _STORE[sid] = sess
        logger.info(f"Session created: {sid}")
        return sess

    except Exception as e:
        raise CustomException(e, sys) from e


def get_session(session_id: str) -> Optional[Session]:
    with _LOCK:
        return _STORE.get(session_id)


def delete_session(session_id: str) -> bool:
    """Delete the session and its uploaded files."""
    with _LOCK:
        sess = _STORE.pop(session_id, None)
    if sess is None:
        return False
    # Wipe files
    upload_dir = UPLOADS_DIR / session_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    logger.info(f"Session deleted: {session_id}")
    return True


def list_sessions() -> List[str]:
    with _LOCK:
        return list(_STORE.keys())


def cleanup_expired(max_age_hours: int = SESSION_TTL_HOURS) -> int:
    """Delete sessions older than max_age_hours. Returns count removed."""
    max_age = max_age_hours * 3600
    with _LOCK:
        expired = [sid for sid, s in _STORE.items() if s.age_seconds() > max_age]
    for sid in expired:
        delete_session(sid)
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")
    return len(expired)


def _evict_oldest():
    """Remove the oldest session to make room."""
    if not _STORE:
        return
    oldest = min(_STORE.values(), key=lambda s: s.created_at)
    logger.warning(f"Evicting oldest session: {oldest.session_id}")
    delete_session(oldest.session_id)


# ============================================================
# Session add file
# ============================================================
def register_file(
    session_id: str,
    file_path: Path,
    kind: str,
    chunks: List[Any],
    rebuild: bool = True,
) -> Session:
    """
    Add an uploaded file's chunks to a session, and rebuild its retriever.
    """
    try:
        sess = get_session(session_id)
        if sess is None:
            sess = create_session(session_id)

        # Convert Chunk dataclass → dict
        new_chunks = [c.to_dict() if hasattr(c, "to_dict") else c for c in chunks]

        sess.files.append(UploadedFile(
            name=file_path.name,
            path=file_path,
            kind=kind,
            n_chunks=len(new_chunks),
        ))
        sess.chunks.extend(new_chunks)

        if rebuild:
            _rebuild_retriever(sess)

        logger.info(
            f"Session {session_id}: +{len(new_chunks)} chunks "
            f"(total {sess.n_chunks})"
        )
        return sess

    except Exception as e:
        raise CustomException(e, sys) from e


def _rebuild_retriever(sess: Session):
    """Rebuild the session's retriever from its accumulated chunks."""
    from src.retrieval.session_retriever import build_session_retriever

    if not sess.chunks:
        sess.retriever = None
        return

    logger.info(f"Rebuilding retriever for session {sess.session_id} "
                f"({len(sess.chunks)} chunks)")
    sess.retriever = build_session_retriever(sess.chunks, with_reranker=True)