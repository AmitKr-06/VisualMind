"""
Smoke test for the LLM module.

Run from the project root:
    python -m src.llm.smoke_test
"""
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.logger import logging as _init_logging  # noqa: F401
from src.llm import call_llm, get_llm
from src.llm.base import LLMError, QuotaExhausted, TransientError

logger = logging.getLogger("visualmind.llm.smoke_test")


def _show(label: str, resp):
    logger.info(f"  → {label}")
    logger.info(f"    provider : {resp.provider}")
    logger.info(f"    model    : {resp.model}")
    logger.info(f"    cached   : {resp.cached}")
    logger.info(f"    seconds  : {resp.seconds}")
    logger.info(f"    text[:80]: {resp.text[:80]!r}")


def main():
    logger.info("=" * 70)
    logger.info("LLM SMOKE TEST")
    logger.info("=" * 70)

    # --- Test 1: Gemini direct ---
    logger.info("")
    logger.info("Test 1 — Gemini direct")
    prompt = 'What is 2+2? Reply with JSON {"result": 4, "explanation": "..."}'
    try:
        resp = call_llm(prompt, provider="gemini")
        _show("gemini ok", resp)
    except LLMError as e:
        logger.warning(f"  gemini failed: {e}")
        logger.info("  continuing with the router chain")

    # --- Test 2: Router (auto) ---
    logger.info("")
    logger.info("Test 2 — Router (gemini → groq fallback)")
    try:
        resp = call_llm(prompt)
        _show("router ok", resp)
    except LLMError as e:
        logger.error(f"  router failed: {e}")
        sys.exit(1)

    # --- Test 3: Cache hit ---
    logger.info("")
    logger.info("Test 3 — Cache hit")
    resp2 = call_llm(prompt)
    _show("router cached", resp2)
    assert resp2.cached, "second call should be cached"
    assert resp2.text == resp.text, "cached text should match"

    # --- Test 4: Groq direct ---
    logger.info("")
    logger.info("Test 4 — Groq direct")
    try:
        client = get_llm("groq")
        resp3 = client.generate('What is the capital of France? Reply with JSON {"city": "..."}')
        _show("groq ok", resp3)
    except Exception as e:
        logger.warning(f"  groq failed: {e}")

    # --- Test 5: Non-JSON mode ---
    logger.info("")
    logger.info("Test 5 — Non-JSON mode")
    resp4 = call_llm(
        "Reply with the single word OK.",
        json_mode=False,
    )
    _show("plain text", resp4)

    # --- Test 6: Prompt template ---
    logger.info("")
    logger.info("Test 6 — Prompt template (no LLM call)")
    from src.llm.prompts import build_answer_prompt
    p = build_answer_prompt(
        question="What is Ohm's law?",
        chunks_block="[doc2_0016] Ohm's law states...",
        figures_block="(no figures)",
    )
    assert "What is Ohm's law?" in p
    assert "[doc2_0016]" in p
    assert "answerable" in p
    logger.info("  → template built, placeholders replaced")

    logger.info("")
    logger.info("=" * 70)
    logger.info("LLM SMOKE TEST COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()