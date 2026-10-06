"""
Prompt formatting for the answer node.

Wraps the ANSWER_PROMPT_TEMPLATE from src/llm/prompts.py and provides
helpers to serialize an evidence package into text blocks.
"""
from __future__ import annotations

from typing import Any, Dict

from src.llm.prompts import build_answer_prompt


# ============================================================
# Chunk / figure formatting
# ============================================================
def format_chunks_block(package: Dict[str, Any]) -> str:
    """Render the package's chunks as a text block for the prompt."""
    chunks = package.get("chunks", [])
    if not chunks:
        return "(no evidence chunks)"
    lines = []
    for c in chunks:
        lines.append(f"[{c['chunk_id']}] (page {c.get('page', '?')}, {c.get('chunk_type', 'content')})")
        lines.append(c.get("text", "").strip())
        lines.append("")
    return "\n".join(lines).strip()


def format_figures_block(package: Dict[str, Any]) -> str:
    """Render the package's figures as a text block."""
    figs = package.get("figures", [])
    if not figs:
        return "(no figures in this package)"
    lines = []
    for f in figs:
        lines.append(f"[{f['fig_key']}] Figure {f.get('figure_id', '?')} — {f.get('caption', '')}")
        desc = f.get("description", "")
        if desc:
            lines.append(desc)
        lines.append("")
    return "\n".join(lines).strip()


# ============================================================
# Public
# ============================================================
def build_answer_prompt_from_package(
    question: str,
    package: Dict[str, Any],
) -> str:
    """
    Build the full answer prompt from a question + evidence package.
    """
    return build_answer_prompt(
        question=question,
        chunks_block=format_chunks_block(package),
        figures_block=format_figures_block(package),
    )