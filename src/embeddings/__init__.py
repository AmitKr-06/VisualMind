"""Embeddings — dense vectors for chunks and queries."""

from .embedding import (
    load_embedding_model,
    embed_texts,
    embed_query,
    embed_chunks,
    save_embeddings,
    load_embeddings,
    embedding_cache_path,
    EmbeddingResult,
)

__all__ = [
    "load_embedding_model",
    "embed_texts",
    "embed_query",
    "embed_chunks",
    "save_embeddings",
    "load_embeddings",
    "embedding_cache_path",
    "EmbeddingResult",
]