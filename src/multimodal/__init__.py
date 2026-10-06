"""Multimodal — fuse text + figures into a unified evidence package."""

from .query_router import classify_query, QueryType
from .fusion import fuse_chunks_and_figures, fuse_rrf
from .evidence_package import (
    build_evidence_package,
    validate_package,
    EvidencePackage,
)

__all__ = [
    "classify_query",
    "QueryType",
    "fuse_chunks_and_figures",
    "fuse_rrf",
    "build_evidence_package",
    "validate_package",
    "EvidencePackage",
]