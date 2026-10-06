"""
Embeddings module — turns text into vectors for retrieval.

Uses BAAI/bge-base-en-v1.5 (768-dim).
- Chunks are embedded WITHOUT a prefix (the model was trained this way).
- Queries are embedded WITH the prefix "Represent this sentence for searching relevant passages: ".

Cache: .npy files under data/embeddings/.
The cache filename includes a short hash of the chunk texts, so the
embeddings are invalidated automatically when the chunks change.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.exception import CustomException

logger = logging.getLogger("visualmind.embeddings")


# ============================================================
# Constants
# ============================================================
DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
QUERY_PREFIX  = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 768


# ============================================================
# Data structures
# ============================================================
@dataclass
class EmbeddingResult:
    """The output of an embedding run."""
    model_name: str
    dim: int
    n_vectors: int
    vectors: np.ndarray            # shape (n_vectors, dim)
    source_file: str               # path to the chunks JSONL
    chunk_ids: List[str]           # parallel to vectors

    def __post_init__(self):
        assert self.vectors.shape == (self.n_vectors, self.dim), \
            f"shape mismatch: {self.vectors.shape} vs ({self.n_vectors}, {self.dim})"
        assert len(self.chunk_ids) == self.n_vectors, \
            f"chunk_ids count {len(self.chunk_ids)} != n_vectors {self.n_vectors}"

    def summary(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dim": self.dim,
            "n_vectors": self.n_vectors,
            "source_file": self.source_file,
            "mean_norm": round(float(np.linalg.norm(self.vectors, axis=1).mean()), 4),
        }


# ============================================================
# Model
# ============================================================
_model_cache: Dict[str, SentenceTransformer] = {}


def load_embedding_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load (and cache) a SentenceTransformer model.

    Args:
        model_name: HuggingFace model id.

    Returns:
        The loaded model.
    """
    try:
        if model_name in _model_cache:
            logger.info(f"Using cached model: {model_name}")
            return _model_cache[model_name]

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading embedding model: {model_name} on {device}")

        model = SentenceTransformer(model_name, device=device)
        _model_cache[model_name] = model

        dim = model.get_sentence_embedding_dimension()
        logger.info(f"  loaded — dim={dim}, max_seq_length={model.max_seq_length}")
        return model

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Embedding
# ============================================================
def embed_texts(
    texts: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    normalize: bool = True,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Embed a list of texts. No query prefix is applied.

    Args:
        texts: List of strings.
        model_name: SentenceTransformer model id.
        batch_size: Encode batch size.
        normalize: L2-normalize the vectors (required for cosine similarity via dot product).
        show_progress: Show a progress bar.

    Returns:
        np.ndarray of shape (len(texts), dim), dtype float32.
    """
    try:
        if not texts:
            return np.zeros((0, EMBEDDING_DIM), dtype="float32")

        model = load_embedding_model(model_name)
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return vectors.astype("float32")

    except Exception as e:
        raise CustomException(e, sys) from e


def embed_query(
    query: str,
    model_name: str = DEFAULT_MODEL,
) -> np.ndarray:
    """
    Embed a single query with the recommended query prefix.

    Args:
        query: The user's question.
        model_name: SentenceTransformer model id.

    Returns:
        np.ndarray of shape (dim,).
    """
    try:
        model = load_embedding_model(model_name)
        vec = model.encode(
            [QUERY_PREFIX + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]
        return vec.astype("float32")

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Cache helpers
# ============================================================
def _hash_chunks(chunk_ids: List[str], texts: List[str]) -> str:
    """Short hash of the chunk texts (first + last 16 chars of the SHA1)."""
    h = hashlib.sha1()
    for cid, t in zip(chunk_ids, texts):
        h.update(cid.encode())
        h.update(t.encode())
    return h.hexdigest()[:16]


def embedding_cache_path(
    source_file: Path | str,
    model_name: str = DEFAULT_MODEL,
) -> Path:
    """
    Return the cache path for the embeddings of a chunks file.

    The filename includes a short hash of the model name, so different
    models for the same chunks don't collide.
    """
    source_file = Path(source_file)
    model_slug = model_name.replace("/", "_").replace("-", "_")
    return source_file.parent.parent / "embeddings" / f"{source_file.stem}_{model_slug}.npy"


def save_embeddings(
    result: EmbeddingResult,
    output_path: Optional[Path | str] = None,
) -> Path:
    """
    Save an EmbeddingResult to disk.

    Saves:
      - <path>.npy        — the vectors
      - <path>.meta.json  — model, dim, source_file, chunk_ids (order matters!)
    """
    try:
        if output_path is None:
            output_path = embedding_cache_path(result.source_file, result.model_name)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, result.vectors)

        meta = {
            "model_name": result.model_name,
            "dim": result.dim,
            "n_vectors": result.n_vectors,
            "source_file": str(result.source_file),
            "chunk_ids": result.chunk_ids,
        }
        meta_path = output_path.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        logger.info(f"Saved embeddings: {output_path} ({result.n_vectors} vectors, {result.dim} dims)")
        logger.info(f"Saved metadata : {meta_path}")
        return output_path

    except Exception as e:
        raise CustomException(e, sys) from e


def load_embeddings(
    source_file: Path | str,
    model_name: str = DEFAULT_MODEL,
) -> Optional[EmbeddingResult]:
    """
    Load embeddings from the cache. Returns None if the cache file doesn't exist.
    """
    try:
        path = embedding_cache_path(source_file, model_name)
        if not path.exists():
            return None

        vectors = np.load(path)
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            logger.warning(f"Meta file missing for {path}, ignoring cache")
            return None

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return EmbeddingResult(
            model_name=meta["model_name"],
            dim=meta["dim"],
            n_vectors=meta["n_vectors"],
            vectors=vectors,
            source_file=meta["source_file"],
            chunk_ids=meta["chunk_ids"],
        )

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# End-to-end: chunks JSONL -> embeddings
# ============================================================
def _load_chunks_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_chunks(
    chunks_jsonl: Path | str,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    force_recompute: bool = False,
) -> EmbeddingResult:
    """
    Load a chunks JSONL, embed all chunks, and save the result.

    Args:
        chunks_jsonl: Path to the chunks JSONL file.
        model_name: SentenceTransformer model id.
        batch_size: Encode batch size.
        force_recompute: Ignore the existing cache and re-embed.

    Returns:
        EmbeddingResult
    """
    try:
        chunks_jsonl = Path(chunks_jsonl)
        if not chunks_jsonl.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_jsonl}")

        # 1. Try cache
        if not force_recompute:
            cached = load_embeddings(chunks_jsonl, model_name)
            if cached is not None:
                logger.info(f"Loaded cached embeddings ({cached.n_vectors} vectors, {cached.dim} dims)")
                return cached

        # 2. Load chunks
        chunks = _load_chunks_jsonl(chunks_jsonl)
        if not chunks:
            raise ValueError(f"No chunks in {chunks_jsonl}")

        chunk_ids = [c["chunk_id"] for c in chunks]
        texts     = [c["text"]     for c in chunks]
        logger.info(f"Loaded {len(chunks)} chunks from {chunks_jsonl}")

        # 3. Embed
        logger.info(f"Embedding {len(texts)} chunks with {model_name}...")
        vectors = embed_texts(texts, model_name=model_name, batch_size=batch_size)

        result = EmbeddingResult(
            model_name=model_name,
            dim=vectors.shape[1],
            n_vectors=vectors.shape[0],
            vectors=vectors,
            source_file=str(chunks_jsonl),
            chunk_ids=chunk_ids,
        )

        # 4. Save
        save_embeddings(result)
        return result

    except Exception as e:
        raise CustomException(e, sys) from e