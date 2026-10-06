"""
VMState — the LangGraph state schema.

Every node receives the state and returns a partial update.
Fields annotated with Annotated[list, operator.add] are APPENDED across nodes.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class VMState(TypedDict, total=False):
    """The state passed between nodes."""

    # --- Input ---
    question: str
    image_path: Optional[str]
    thread_id: str

    # --- Router ---
    route: str                       # "text" | "image" | "mixed"
    route_reason: str

    # --- Retrieval ---
    package: Optional[Dict[str, Any]]
    package_status: str              # "ok" | "no_evidence" | "out_of_scope"

    # --- Answer ---
    answer: Optional[str]
    citations:    Annotated[List[str], operator.add]
    used_chunks:  Annotated[List[str], operator.add]
    used_figures: Annotated[List[str], operator.add]
    confidence: float
    answerable: bool

    # --- Reflection / retry ---
    citation_errors: List[str]
    retries: int
    max_retries: int
    needs_retry: bool

    # --- Final ---
    final: Optional[Dict[str, Any]]
    notes: Annotated[List[str], operator.add]