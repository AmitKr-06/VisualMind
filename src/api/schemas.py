"""Pydantic schemas for the API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ============================================================
# Requests
# ============================================================
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000,
                          description="The student's question")
    thread_id: Optional[str] = Field(
        None,
        description="Optional thread id for multi-turn checkpointing",
    )
    session_id: Optional[str] = Field(
        None,
        description="Optional session id for scoped retrieval (module 10 uploads)",
    )
    image_path: Optional[str] = Field(
        None,
        description="Optional server-side path to an image (for internal testing)",
    )


class BatchRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1, max_length=50)


# ============================================================
# Responses
# ============================================================
class AskResponse(BaseModel):
    status: str = Field(..., description="ok | no_evidence | out_of_scope | citation_problem | error")
    question: str
    answer: str
    citations: List[str] = Field(default_factory=list)
    used_chunks: List[str] = Field(default_factory=list)
    used_figures: List[str] = Field(default_factory=list)
    confidence: float = 0.0
    answerable: bool = False
    retries: int = 0
    route: str = "text"
    elapsed_s: float = 0.0
    thread_id: Optional[str] = None


class BatchResponse(BaseModel):
    results: List[AskResponse]
    count: int
    elapsed_s: float


class HealthResponse(BaseModel):
    status: str = "ok"
    graph: str = "VisualMindGraph"
    version: str = "v1"
    nodes: List[str] = Field(default_factory=list)
    models: Dict[str, Any] = Field(default_factory=dict)


class ConfigResponse(BaseModel):
    chunk_file: str
    k_chunks: int
    n_figures: int
    tau_text: float
    choice1: str
    choice2: str
    dense_model: str
    reranker_model: str
    answer_model_chain: List[str]
    fallback_provider: Optional[str] = None