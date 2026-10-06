"""
Smoke test for the multimodal module.

Run from the project root:
    python -m src.multimodal.smoke_test
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.retrieval import get_retriever
from src.vision.figure_manifest import load_figure_manifest
from src.vision.matchers import FigureMatcher
from src.vision.image_analyzer import analyze_query_image
from src.multimodal import (
    classify_query, QueryType,
    build_evidence_package, validate_package,
    fuse_rrf,
)

logger = logging.getLogger("visualmind.multimodal.smoke_test")


def _show_package(pkg, label):
    logger.info(f"  {label}:")
    logger.info(f"    status  : {pkg['status']}")
    logger.info(f"    chunks  : {len(pkg['chunks'])}")
    logger.info(f"    figures : {len(pkg['figures'])}")
    if pkg["notes"]:
        for n in pkg["notes"]:
            logger.info(f"    note    : {n}")
    for c in pkg["chunks"]:
        logger.info(f"      [{c['rank']}] {c['chunk_id']} | {c['chunk_type']} | p{c['page']}")
    for f in pkg["figures"]:
        logger.info(f"      [f{f['rank']}] {f['fig_key']} | Figure {f['figure_id']}")


def main():
    logger.info("=" * 70)
    logger.info("MULTIMODAL SMOKE TEST")
    logger.info("=" * 70)

    # --- Test 1: query router ---
    logger.info("")
    logger.info("Test 1 — Query router")
    r1 = classify_query(question="What is Ohm's law?")
    r2 = classify_query(image_path="fake.jpg")
    r3 = classify_query(question="Describe this", image_path="fake.jpg")
    r4 = classify_query()
    logger.info(f"  text       → {r1.type.value}   ({r1.reason})")
    logger.info(f"  image      → {r2.type.value}   ({r2.reason})")
    logger.info(f"  mixed      → {r3.type.value}   ({r3.reason})")
    logger.info(f"  empty      → {r4.type.value}   ({r4.reason})")
    assert r1.type == QueryType.TEXT
    assert r2.type == QueryType.IMAGE
    assert r3.type == QueryType.MIXED
    assert r4.type == QueryType.TEXT

    # --- Test 2: RRF fusion ---
    logger.info("")
    logger.info("Test 2 — RRF fusion")
    a = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
    b = [{"chunk_id": "c2"}, {"chunk_id": "c1"}, {"chunk_id": "c4"}]
    fused = fuse_rrf([a, b], id_key="chunk_id")
    logger.info(f"  fused order: {[x['chunk_id'] for x in fused]}")
    assert fused[0]["chunk_id"] in ("c1", "c2")

    # --- Test 3: text-only package ---
    logger.info("")
    logger.info("Test 3 — Build package for a text question")
    chunks_file = PROJECT_ROOT / "data" / "interim" / "chunks_smoke_test.jsonl"
    retriever = get_retriever(chunks_file, with_reranker=False)

    manifest = load_figure_manifest(PROJECT_ROOT / "data" / "extracted")
    matcher = FigureMatcher(manifest)

    pkg = build_evidence_package(
        question="What is Ohm's law?",
        text_retriever=retriever,
        figure_matcher=matcher,
    )
    _show_package(pkg, "text package")
    assert pkg["status"] == "ok"
    assert len(pkg["chunks"]) >= 1
    errs = validate_package(pkg)
    assert not errs, f"package validation failed: {errs}"

    # --- Test 4: no-evidence package ---
    logger.info("")
    logger.info("Test 4 — Build package for out-of-scope text")
    pkg2 = build_evidence_package(
        question="Who won the cricket world cup in 2011?",
        text_retriever=retriever,
        figure_matcher=matcher,
    )
    _show_package(pkg2, "off-topic package")
    assert pkg2["status"] in ("no_evidence", "ok")  # depends on retriever scores
    if pkg2["status"] == "no_evidence":
        assert pkg2["chunks"] == []
        assert pkg2["figures"] == []
        logger.info("  → correctly refused")

    # --- Test 5: image-only package ---
    logger.info("")
    logger.info("Test 5 — Build package for an image-only query")
    query_img = PROJECT_ROOT / "data" / "eval" / "images" / "test" / "in_5.1_phone_01.jpg"
    if not query_img.exists():
        candidates = list((PROJECT_ROOT / "data" / "eval" / "images" / "test").glob("in_*"))
        if candidates:
            query_img = candidates[0]

    if query_img.exists():
        pkg3 = build_evidence_package(
            question=None,
            image_path=str(query_img),
            text_retriever=retriever,
            figure_matcher=matcher,
            image_analyzer=analyze_query_image,
        )
        _show_package(pkg3, "image package")
        assert pkg3["status"] in ("ok", "no_evidence", "out_of_scope")
    else:
        logger.warning("  no test image found — skipped")

    logger.info("")
    logger.info("=" * 70)
    logger.info("MULTIMODAL SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()