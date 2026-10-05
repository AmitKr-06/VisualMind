"""
Document loader — extracts raw text from PDF files.

This module mirrors the Exp 1 pipeline but is production-ready:
- Uses PyMuPDF (fitz) for reliable text extraction
- Returns structured PDFDocument objects with per-page text
- Fails loudly with CustomException on any error
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf  # PyMuPDF
from src.exception import CustomException

logger = logging.getLogger("visualmind.ingestion.loader")


# ============================================================
# Data structures
# ============================================================
@dataclass
class PDFPage:
    """One page of a PDF document."""
    page_number: int             # 1-based
    text: str                    # raw extracted text
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)


@dataclass
class PDFDocument:
    """A complete PDF with per-page text."""
    name: str                    # filename (e.g. "doc1_life_processes.pdf")
    path: Path
    pages: List[PDFPage] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return sum(p.char_count for p in self.pages)

    @property
    def n_pages(self) -> int:
        return len(self.pages)


# ============================================================
# Public API
# ============================================================
def load_pdf(pdf_path: Path | str) -> PDFDocument:
    """
    Load a single PDF file and extract text from every page.

    Args:
        pdf_path: Path to the PDF.

    Returns:
        PDFDocument with all pages populated.

    Raises:
        CustomException on any failure (missing file, corrupt PDF, etc.).
    """
    try:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Loading PDF: {pdf_path.name}")

        doc = pymupdf.open(pdf_path)
        pages: List[PDFPage] = []

        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            pages.append(PDFPage(page_number=i, text=text))

        doc.close()

        result = PDFDocument(name=pdf_path.name, path=pdf_path, pages=pages)
        logger.info(
            f"  Loaded {result.n_pages} pages, {result.total_chars:,} chars"
        )
        return result

    except Exception as e:
        raise CustomException(e, sys) from e


def load_all_pdfs(pdf_dir: Path | str) -> Dict[str, PDFDocument]:
    """
    Load every PDF in a directory.

    Args:
        pdf_dir: Directory containing PDFs.

    Returns:
        Dict mapping filename -> PDFDocument.
    """
    try:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            logger.warning(f"No PDFs found in {pdf_dir}")
            return {}

        logger.info(f"Loading {len(pdfs)} PDFs from {pdf_dir}")

        documents: Dict[str, PDFDocument] = {}
        for pdf in pdfs:
            documents[pdf.name] = load_pdf(pdf)

        logger.info(f"Loaded {len(documents)} documents")
        return documents

    except Exception as e:
        raise CustomException(e, sys) from e


def list_documents(pdf_dir: Path | str) -> List[Dict]:
    """
    List PDFs in a directory with basic metadata (without extracting text).

    Args:
        pdf_dir: Directory containing PDFs.

    Returns:
        List of dicts: [{"name", "path", "size_kb"}, ...]
    """
    try:
        pdf_dir = Path(pdf_dir)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")

        out = []
        for pdf in sorted(pdf_dir.glob("*.pdf")):
            out.append({
                "name": pdf.name,
                "path": str(pdf),
                "size_kb": round(pdf.stat().st_size / 1024, 1),
            })
        return out

    except Exception as e:
        raise CustomException(e, sys) from e