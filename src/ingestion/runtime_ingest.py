"""
Runtime ingestion — load a single uploaded PDF into memory chunks.

Reuses the existing loader + cleaner + chunker, but does NOT save anything
to disk. The chunks are handed to the session store.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

from src.exception import CustomException
from src.ingestion.document_loader import load_pdf
from src.ingestion.preprocessing import chunk_document, Chunk

logger = logging.getLogger("visualmind.ingestion.runtime")


def ingest_uploaded_pdf(
    pdf_path: Path | str,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Load a single uploaded PDF, clean it, and chunk it.

    Returns the Chunk list in memory. Nothing is written to disk.
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Ingesting uploaded PDF: {pdf_path.name}")
        doc = load_pdf(pdf_path)
        chunks = chunk_document(doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        logger.info(f"  → {len(chunks)} chunks from {pdf_path.name}")
        return chunks

    except Exception as e:
        raise CustomException(e, sys) from e