"""
Smoke test for the vision module.

Run from the project root:
    python -m src.vision.smoke_test
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.vision.figure_manifest import load_figure_manifest
from src.vision.image_analyzer import describe_figure, analyze_query_image
from src.vision.matchers import FigureMatcher, match_query_to_figures

logger = logging.getLogger("visualmind.vision.smoke_test")


def main():
    logger.info("=" * 70)
    logger.info("VISION SMOKE TEST")
    logger.info("=" * 70)

    extract_dir = PROJECT_ROOT / "data" / "extracted"

    # 1. Load manifest
    logger.info("")
    logger.info("Test 1 — Load figure manifest")
    manifest = load_figure_manifest(extract_dir)
    logger.info(f"  → loaded {len(manifest)} figures")
    assert len(manifest) > 0, "manifest should not be empty"

    # Show a couple
    for e in list(manifest)[:3]:
        logger.info(f"    {e.fig_key} | Figure {e.figure_id} | {e.caption[:60]}")

    # 2. Describe one figure (uses Gemini Vision)
    logger.info("")
    logger.info("Test 2 — Describe a figure (Gemini Vision)")
    fig = manifest.get_by_id("5.1")
    assert fig is not None, "Figure 5.1 must exist"
    png = fig.png_path(manifest.fig_dir)
    assert png.exists(), f"PNG not found: {png}"

    desc = describe_figure(png, fig.caption)
    assert desc is not None, "describe_figure should return a dict"
    logger.info(f"  → provider : {desc.get('_provider')}")
    logger.info(f"  → model    : {desc.get('_model')}")
    logger.info(f"  → cached   : {desc.get('_cached')}")
    logger.info(f"  → type     : {desc.get('visual_type')}")
    logger.info(f"  → text[:100]: {desc.get('description', '')[:100]!r}")
    assert desc.get("description"), "description should be non-empty"

    # 3. Analyze a query image
    logger.info("")
    logger.info("Test 3 — Analyze a query image")
    query_img = PROJECT_ROOT / "data" / "eval" / "images" / "test" / "in_5.1_phone_01.jpg"
    if not query_img.exists():
        # fallback: try any test image
        candidates = list((PROJECT_ROOT / "data" / "eval" / "images" / "test").glob("in_*"))
        if candidates:
            query_img = candidates[0]
    if query_img.exists():
        analysis = analyze_query_image(query_img)
        if analysis:
            logger.info(f"  → in_scope : {analysis.get('in_scope')}")
            logger.info(f"  → reason   : {analysis.get('reason', '')[:80]}")
            logger.info(f"  → desc     : {analysis.get('description', '')[:100]}")
        else:
            logger.warning("  → analysis returned None")
    else:
        logger.warning(f"  → no query image found — skipping")

    # 4. Build the CLIP matcher
    logger.info("")
    logger.info("Test 4 — Build CLIP matcher")
    matcher = FigureMatcher(manifest)
    logger.info(f"  → matcher ready with {len(matcher.fig_keys)} indexed figures")

    # 5. Rank by text
    logger.info("")
    logger.info("Test 5 — Rank figures by text (\"diagram of a leaf\")")
    results = matcher.rank_by_text("diagram of a leaf", k=5)
    for r in results:
        logger.info(f"    [{r['rank']}] {r['fig_key']:<40s} "
                    f"Figure {r['figure_id']:<5s} | sim={r['score']:.4f} | "
                    f"{r['caption'][:50]}")

    # 6. Fusion
    logger.info("")
    logger.info("Test 6 — Fusion (text only)")
    fused = match_query_to_figures(
        matcher,
        text="cross-section of a leaf",
        k=3,
        strategy="fusion",
    )
    for r in fused:
        logger.info(f"    [{r['rank']}] {r['fig_key']:<40s} | rrf={r.get('rrf_score', 0):.4f}")

    logger.info("")
    logger.info("=" * 70)
    logger.info("VISION SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()