"""
Text preprocessing — cleans raw PDF text and chunks it for retrieval.

Cleaning rules:
- Remove running page headers ("Science82", "Life Processes 95")
- Remove the "Reprint 2022-23" footer
- Collapse repeated captions ("Figure 5.2Figure 5.2" → "Figure 5.2")
- Fix broken bullets ("/square6" → "- ")
- Rebuild paragraphs from broken lines
- Keep Activities, Examples, Tables, Exercises whole

Chunking:
- Splits by size with overlap
- Preserves sentence boundaries
- Maps every chunk back to a source page
"""
from __future__ import annotations

import json
import logging
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.exception import CustomException
from src.ingestion.document_loader import PDFDocument

logger = logging.getLogger("visualmind.ingestion.preprocessing")


# ============================================================
# Constants
# ============================================================
SECTION_TITLES = {
    "doc1_life_processes.pdf": {
        "5.1": "5.1 WHAT ARE LIFE PROCESSES?",
        "5.2": "5.2 NUTRITION",
        "5.3": "5.3 RESPIRATION",
        "5.4": "5.4 TRANSPORTATION",
        "5.5": "5.5 EXCRETION",
    },
    "doc2_electricity.pdf": {
        "11.1": "11.1 ELECTRIC CURRENT AND CIRCUIT",
        "11.2": "11.2 ELECTRIC POTENTIAL AND POTENTIAL DIFFERENCE",
        "11.3": "11.3 CIRCUIT DIAGRAM",
        "11.4": "11.4 OHM'S LAW",
        "11.5": "11.5 FACTORS ON WHICH THE RESISTANCE OF A CONDUCTOR DEPENDS",
        "11.6": "11.6 RESISTANCE OF A SYSTEM OF RESISTORS",
        "11.7": "11.7 HEATING EFFECT OF ELECTRIC CURRENT",
        "11.8": "11.8 ELECTRIC POWER",
    },
}

SEPARATORS = ["\n\n", "\n", ".", "?", ":", " ", ""]


# ============================================================
# Data structures
# ============================================================
@dataclass
class Chunk:
    """One retrieval chunk."""
    chunk_id: str
    text: str
    source: str                 # PDF filename
    page: int                   # 1-based page number
    section: str                # section title (e.g. "5.2 NUTRITION")
    chunk_type: str             # "content", "activity", "example", "table", "exercise", "summary"
    char_count: int = 0

    def __post_init__(self):
        self.char_count = len(self.text)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================
# Text cleaning
# ============================================================
def clean_text(text: str) -> str:
    """
    Clean raw PDF text. Mirrors the Exp 1 rules.

    Args:
        text: Raw text from one PDF page.

    Returns:
        Cleaned text.
    """
    try:
        # 1. Running page header (e.g. "Science82", "Life Processes 95",
        #    "Science194 194194 194194")
        #    FIX: balanced parentheses; the digits variant is captured by the
        #    inner (?:[ \n\t]*\d{1,6})* group, not by a stray `)` inside `[]`.
        text = re.sub(
            r"\s*(?:Science|Life Processes|Electricity)\s*\d{1,3}(?:[ \n\t]*\d{1,6})*[ \t]*\n",
            " ",
            text,
        )

        # 2. Chapter-opening page:
        #    "Life Processes 5CHAPTER How do..."  →  "How do..."
        text = re.sub(
            r"\s*(?:Life Processes|Electricity)\s*\d{1,2}\s*CHAPTER\s*([A-Z])\s*([a-z]+)",
            r"\1\2",
            text,
        )

        # 3. Collapse repeated captions ("Figure 5.2Figure 5.2" → "Figure 5.2")
        text = re.sub(r"((?:Figure|Activity|Table)\s*\d+\.\d+)\1+", r"\1", text)

        # 4. Broken bullets and reprint footer
        text = text.replace("/square6", "- ")
        text = text.replace("...", "..")
        text = re.sub(r"Reprint\s+\d{4}-\d{2}", " ", text)

        # 5. Drop page-icon "?" lines
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        lines = [ln for ln in lines if ln not in ("?", "QUESTIONS")]

        # 6. Rebuild paragraphs from broken lines
        caption_start = re.compile(r"^(?:Figure|Activity|Table)\s*\d+\.\d+")
        block_start   = re.compile(r"^(?:\d{1,2}\.\s*[A-Z]|Example\s\d+\.\d+|Solution\b)")
        heading       = re.compile(r"^\d+\.\d+\s+[A-Z][A-Z ,'\-]{4,}")

        paragraphs: List[str] = []
        for line in lines:
            if not paragraphs:
                paragraphs.append(line)
                continue
            prev = paragraphs[-1]
            starts_new = (
                line.startswith("- ")
                or prev.endswith((".", "?", "!", ":"))
                or caption_start.match(line)
                or block_start.match(line)
                or heading.match(line)
                or heading.match(prev)
            )
            if starts_new:
                paragraphs.append(line)
            else:
                paragraphs[-1] = prev + " " + line

        return "\n\n".join(paragraphs).strip()

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Heading detection
# ============================================================
def _extract_section(text: str) -> Optional[str]:
    """Return the section title from the first heading in the chunk."""
    try:
        m = re.match(r"^(\d+\.\d+)\s+([A-Z][A-Z ,'\-]{4,})", text.strip())
        if m:
            return f"{m.group(1)} {m.group(2).strip()}"
        return None
    except Exception:
        return None


def _chunk_type(text: str) -> str:
    """Classify a chunk by its content."""
    try:
        t = text.strip()
        if re.match(r"^Activity\s\d+\.\d+", t):
            return "activity"
        if re.match(r"^Example\s\d+\.\d+", t):
            return "example"
        if re.match(r"^Table\s\d+\.\d+", t):
            return "table"
        if re.match(r"^EXERCISES", t, re.IGNORECASE):
            return "exercise"
        if re.match(r"^What you have learnt", t, re.IGNORECASE):
            return "summary"
        return "content"
    except Exception:
        return "content"


# ============================================================
# Chunking
# ============================================================
def chunk_document(
    doc: PDFDocument,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Chunk]:
    """
    Clean and chunk a single PDF document.

    Args:
        doc: A loaded PDFDocument.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        List of Chunk objects.
    """
    try:
        logger.info(f"Chunking {doc.name}: {doc.n_pages} pages, size={chunk_size}, overlap={chunk_overlap}")

        # Clean each page separately (keeps page numbers)
        cleaned_pages = [
            (page.page_number, clean_text(page.text))
            for page in doc.pages
        ]

        # Join all pages with a marker; track offsets
        full_text = ""
        page_offsets: List[tuple] = []  # (start_offset, page_number)
        for page_num, text in cleaned_pages:
            if not text:
                continue
            if full_text:
                full_text += "\n\n"
            page_offsets.append((len(full_text), page_num))
            full_text += text

        # Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=SEPARATORS,
            keep_separator="end",
            add_start_index=True,
        )
        raw_chunks = splitter.create_documents([full_text])

        # Build Chunk objects
        chunks: List[Chunk] = []
        stem = Path(doc.name).stem
        for i, rc in enumerate(raw_chunks):
            start = rc.metadata.get("start_index", 0)
            # Find the closest page offset
            page_num = 1
            for offset, pn in page_offsets:
                if offset <= start:
                    page_num = pn
                else:
                    break

            text = rc.page_content.strip()
            if not text:
                continue

            chunk = Chunk(
                chunk_id=f"{stem}_{chunk_size}_{i:04d}",
                text=text,
                source=doc.name,
                page=page_num,
                section=_extract_section(text) or "Chapter opening",
                chunk_type=_chunk_type(text),
            )
            chunks.append(chunk)

        logger.info(f"  → {len(chunks)} chunks")
        return chunks

    except Exception as e:
        raise CustomException(e, sys) from e


def chunk_all_documents(
    documents: Dict[str, PDFDocument],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> Dict[str, List[Chunk]]:
    """
    Chunk every loaded document.

    Args:
        documents: Dict of filename -> PDFDocument.
        chunk_size: Target chunk size in characters.
        chunk_overlap: Overlap between chunks in characters.

    Returns:
        Dict of filename -> list of Chunk objects.
    """
    try:
        out: Dict[str, List[Chunk]] = {}
        for name, doc in documents.items():
            out[name] = chunk_document(doc, chunk_size, chunk_overlap)
        total = sum(len(v) for v in out.values())
        logger.info(f"Chunked {len(out)} documents → {total} total chunks")
        return out
    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Persistence
# ============================================================
def save_chunks_jsonl(chunks_by_doc: Dict[str, List[Chunk]], output_path: Path | str) -> int:
    """
    Save chunks to a JSONL file (one JSON object per line).

    Args:
        chunks_by_doc: Dict of filename -> list of Chunk.
        output_path: Where to write the JSONL file.

    Returns:
        Number of chunks written.
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        n = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for chunks in chunks_by_doc.values():
                for c in chunks:
                    f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
                    n += 1
        logger.info(f"Saved {n} chunks to {output_path}")
        return n
    except Exception as e:
        raise CustomException(e, sys) from e