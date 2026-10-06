"""
Prompt templates.

All templates use `__PLACEHOLDER__` markers instead of `{name}` so
that JSON in the prompt (like the output schema) doesn't break `.format()`.

Always build the final prompt with `.replace()`, not `.format()`.
"""
from __future__ import annotations


# ============================================================
# Answer generation (the core RAG prompt)
# ============================================================
ANSWER_PROMPT_TEMPLATE = """You are a study assistant for NCERT Class 10 Science (Life Processes / Electricity).

You will be given:
- A student's question.
- A set of EVIDENCE CHUNKS, each labelled with its chunk_id.
- Optionally, a set of FIGURES, each labelled with its fig_key and its caption/description.

STRICT RULES (any answer that breaks them is rejected):
1. Answer ONLY using facts that appear in the evidence chunks. Do not use outside knowledge.
2. Every sentence in your answer MUST end with one or more citations in square brackets, e.g. [chunk_id].
3. ONLY cite chunk_ids that appear in the EVIDENCE CHUNKS section below. Never invent a chunk_id, never cite a chunk_id that is not listed.
4. If the evidence is not sufficient to answer, set "answerable": false and explain in one sentence.
5. Keep the answer 2-6 sentences. Simple, textbook style.
6. If a formula is present in the evidence, quote it exactly as it appears.

Return ONLY a JSON object with exactly these keys:
{
  "answerable": true or false,
  "answer": "the answer text with [chunk_id] citations",
  "used_chunks": ["chunk_id1", "chunk_id2"],
  "used_figures": [],
  "confidence": 0.0
}

QUESTION:
__QUESTION__

EVIDENCE CHUNKS:
__CHUNKS__

FIGURES (optional):
__FIGURES__
"""


# ============================================================
# Figure description (for Exp 5 / module 5)
# ============================================================
FIGURE_PROMPT_TEMPLATE = """You are looking at ONE figure cut out of a school science textbook (class 10: life processes / electricity).
The textbook caption is: "__CAPTION__"
Describe only what you can SEE in the image. Do not add facts from memory.
Return ONE JSON object (not a list, not an array) with exactly these keys:
"visual_type": one of "diagram", "circuit diagram", "graph", "flowchart", "photo", "table", "other"
"description": 2 to 4 sentences: what the figure shows and how its parts relate to each other
"labels": list of the text labels you can actually read in the image, copied exactly ([] if none)
"entities": list of the main objects, organs or components shown
"relationships": list of short statements that are visible, e.g. "guard cells surround the stomatal pore" ([] if none)
"values": list of numbers or quantities you can read, with units, e.g. "6 V" ([] if none)
"crop_ok": true or false
"uncertain": list of things you are not sure about ([] if none)
Return only the JSON object."""


# ============================================================
# Query description (for Exp 5 image-to-figure)
# ============================================================
QUERY_PROMPT_TEMPLATE = """A student uploaded this image to a science tutor. Describe only what you can SEE.
Decide whether it is a diagram, graph, circuit or photo that belongs to a class 10 science chapter on LIFE PROCESSES (nutrition, respiration, transportation, excretion) or ELECTRICITY (current, potential difference, resistance, circuits, heating effect).
Return JSON with exactly these keys:
"in_scope": true or false
"reason": one short sentence
"visual_type": one of "diagram", "circuit diagram", "graph", "flowchart", "photo", "table", "other"
"description": 2 to 4 sentences: what the image shows
"labels": list of the text labels you can actually read ([] if none)
"entities": list of the main objects, organs or components shown
"relationships": list of short statements that are visible ([] if none)
"values": list of numbers or quantities you can read ([] if none)
"crop_ok": true
"uncertain": list of things you are not sure about ([] if none)
Return only the JSON."""


# ============================================================
# Contextualizer (for follow-up rewriting)
# ============================================================
CONTEXTUALIZE_PROMPT_TEMPLATE = """Rewrite the FOLLOW-UP question so it is a standalone question.

PREVIOUS QUESTION:
__PREV__

FOLLOW-UP:
__FOLLOW__

Return ONLY the rewritten standalone question, no explanation, no quotes."""


# ============================================================
# Convenience builders
# ============================================================
def build_answer_prompt(question: str, chunks_block: str, figures_block: str) -> str:
    return (ANSWER_PROMPT_TEMPLATE
            .replace("__QUESTION__", question)
            .replace("__CHUNKS__", chunks_block)
            .replace("__FIGURES__", figures_block))


def build_figure_prompt(caption: str) -> str:
    return FIGURE_PROMPT_TEMPLATE.replace("__CAPTION__", caption)


def build_query_prompt() -> str:
    return QUERY_PROMPT_TEMPLATE


def build_contextualize_prompt(prev_question: str, followup: str) -> str:
    return (CONTEXTUALIZE_PROMPT_TEMPLATE
            .replace("__PREV__", prev_question)
            .replace("__FOLLOW__", followup))