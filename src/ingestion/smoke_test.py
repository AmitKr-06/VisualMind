"""
Smoke test for the ingestion module.

Run from the project root:
    python -m src.ingestion.smoke_test
"""
import logging
import sys
from pathlib import Path

# Add project root to sys.path so "src.*" imports work
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401 — sets up logging
from src.ingestion.document_loader import list_documents, load_all_pdfs
from src.ingestion.preprocessing import chunk_all_documents, save_chunks_jsonl

logger = logging.getLogger("visualmind.ingestion.smoke_test")


def main():
    logger.info("=" * 70)
    logger.info("INGESTION SMOKE TEST")
    logger.info("=" * 70)

    # 1. Locate documents
    raw_dir = PROJECT_ROOT / "data" / "raw" / "documents"
    if not raw_dir.exists():
        logger.error(f"Missing folder: {raw_dir}")
        sys.exit(1)

    docs = list_documents(raw_dir)
    logger.info(f"Found {len(docs)} PDFs in {raw_dir}")
    for d in docs:
        logger.info(f"  {d['name']}  ({d['size_kb']} KB)")

    if not docs:
        logger.error("No PDFs found — put doc1_life_processes.pdf and doc2_electricity.pdf in data/raw/documents/")
        sys.exit(1)

    # 2. Load PDFs
    logger.info("")
    logger.info("Loading PDFs...")
    documents = load_all_pdfs(raw_dir)

    # 3. Chunk
    logger.info("")
    logger.info("Cleaning and chunking...")
    chunks_by_doc = chunk_all_documents(documents, chunk_size=1000, chunk_overlap=150)

    # 4. Summary
    logger.info("")
    logger.info("SUMMARY")
    logger.info("-" * 70)
    for name, chunks in chunks_by_doc.items():
        types = {}
        for c in chunks:
            types[c.chunk_type] = types.get(c.chunk_type, 0) + 1
        logger.info(f"  {name}:")
        logger.info(f"    chunks: {len(chunks)}")
        logger.info(f"    types : {types}")
        logger.info(f"    avg chars: {sum(c.char_count for c in chunks) // len(chunks)}")

    # 5. Save to JSONL (for verification only — real pipeline uses Exp 1 output)
    out_path = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    n = save_chunks_jsonl(chunks_by_doc, out_path)
    logger.info("")
    logger.info(f"Wrote {n} chunks to {out_path}")

    # 6. Show 3 sample chunks
    logger.info("")
    logger.info("Sample chunks:")
    sample_types = ["activity", "example", "content"]
    shown = {t: False for t in sample_types}
    for chunks in chunks_by_doc.values():
        for c in chunks:
            if c.chunk_type in shown and not shown[c.chunk_type]:
                logger.info("")
                logger.info(f"  [{c.chunk_type}] {c.chunk_id} | p{c.page} | {c.section}")
                logger.info(f"    {c.text[:200]}...")
                shown[c.chunk_type] = True

    logger.info("")
    logger.info("=" * 70)
    logger.info("SMOKE TEST COMPLETE ")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()