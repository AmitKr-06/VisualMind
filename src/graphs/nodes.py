"""
The five graph nodes.

Each node takes the state and returns a partial update:
  - router_node       classifies input
  - retrieval_node    builds the evidence package
  - answer_node       calls the LLM
  - reflection_node   checks citation integrity, decides retry
  - responder_node    formats the final dict
"""
from __future__ import annotations

import json
import logging
import re
import sys
from typing import Any, Dict, List

from src.exception import CustomException
from src.graphs.prompts import build_answer_prompt_from_package
from src.graphs.state import VMState
from src.llm import call_llm
from src.multimodal import build_evidence_package

logger = logging.getLogger("visualmind.graphs.nodes")


# ============================================================
# Citation extraction
# ============================================================
CITATION_RE = re.compile(r"\[([a-zA-Z0-9_\.]+)\]")


def extract_citations(text: str) -> List[str]:
    return CITATION_RE.findall(text or "")


# ============================================================
# Answer parsing
# ============================================================
def parse_answer_response(text: str) -> Dict[str, Any] | None:
    """Robust JSON parser for the LLM's answer response."""
    if not isinstance(text, str):
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    obj = None
    try:
        obj = json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, list) and len(obj) == 1 and isinstance(obj[0], dict):
        obj = obj[0]
    if not isinstance(obj, dict):
        return None

    obj.setdefault("answerable", False)
    obj.setdefault("answer", "")
    obj.setdefault("used_chunks", [])
    obj.setdefault("used_figures", [])
    obj.setdefault("confidence", 0.0)
    for k in ("used_chunks", "used_figures"):
        if not isinstance(obj[k], list):
            obj[k] = [str(obj[k])] if obj[k] else []
    return obj


# ============================================================
# Node 1 — router
# ============================================================
def router_node(state: VMState) -> Dict[str, Any]:
    q = (state.get("question") or "").strip()
    img = state.get("image_path")

    if img and q:
        route, reason = "mixed", "image + question provided"
    elif img:
        route, reason = "image", "image only"
    elif q:
        route, reason = "text", "text-only question"
    else:
        route, reason = "text", "empty input → default to text"

    return {
        "route": route,
        "route_reason": reason,
        "retries": 0,
        "max_retries": 2,
        "citation_errors": [],
        "needs_retry": False,
        "notes": [f"router: {route} ({reason})"],
    }


# ============================================================
# Node 2 — retrieval (factory)
# ============================================================
def retrieval_node(state: VMState) -> Dict[str, Any]:
    """
    Placeholder — the real node is created by `make_retrieval_node()`,
    which binds the retriever / matcher / analyzer dependencies.
    """
    raise RuntimeError(
        "retrieval_node must be called with dependencies injected via closure. "
        "See src/graphs/workflow.py for the factory pattern."
    )


def make_retrieval_node(
    text_retriever,
    figure_matcher=None,
    image_analyzer=None,
    k_chunks: int = 3,
    n_figures: int = 2,
    tau_text: float = 0.115,
):
    """Factory that binds dependencies into the retrieval node."""
    def _node(state: VMState) -> Dict[str, Any]:
        try:
            question = state.get("question") or ""
            image_path = state.get("image_path")

            pkg = build_evidence_package(
                question=question or None,
                image_path=image_path,
                text_retriever=text_retriever,
                figure_matcher=figure_matcher,
                image_analyzer=image_analyzer,
                k_chunks=k_chunks,
                n_figures=n_figures,
                tau_text=tau_text,
            )

            return {
                "package": pkg,
                "package_status": pkg["status"],
                "notes": [
                    f"retrieval: status={pkg['status']} "
                    f"chunks={len(pkg['chunks'])} figures={len(pkg['figures'])}"
                ],
            }
        except Exception as e:
            raise CustomException(e, sys) from e
    return _node


# ============================================================
# Node 3 — answer
# ============================================================
def answer_node(state: VMState) -> Dict[str, Any]:
    try:
        pkg = state.get("package")
        q = state.get("question") or ""

        # Refusal fast-path
        if pkg is None or pkg.get("status") != "ok":
            return {
                "answer": "I don't have enough evidence in the provided chapters to answer this.",
                "citations": [],
                "used_chunks": [],
                "used_figures": [],
                "confidence": 0.0,
                "answerable": False,
                "notes": [
                    f"answer: refused (package_status={pkg['status'] if pkg else 'None'})"
                ],
            }

        # Build prompt and call LLM
        if not q:
            # Image-only: use the first figure's description as the question
            figs = pkg.get("figures", [])
            if figs and figs[0].get("description"):
                q = "What does this image show?"
            else:
                q = "Describe the image."

        prompt = build_answer_prompt_from_package(q, pkg)
        resp = call_llm(prompt, json_mode=True)
        parsed = parse_answer_response(resp.text)

        if parsed is None:
            return {
                "answer": "The model returned an unreadable response.",
                "citations": [],
                "used_chunks": [],
                "used_figures": [],
                "confidence": 0.0,
                "answerable": False,
                "notes": ["answer: parse error"],
            }

        return {
            "answer": parsed["answer"],
            "citations": extract_citations(parsed["answer"]),
            "used_chunks": parsed.get("used_chunks", []),
            "used_figures": parsed.get("used_figures", []),
            "confidence": float(parsed.get("confidence", 0.0)),
            "answerable": bool(parsed.get("answerable", False)),
            "notes": [
                f"answer: status=ok conf={parsed.get('confidence', 0):.2f} "
                f"provider={resp.provider} cached={resp.cached}"
            ],
        }
    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Node 4 — reflection
# ============================================================
def reflection_node(state: VMState) -> Dict[str, Any]:
    try:
        pkg = state.get("package") or {}
        cites = state.get("citations") or []
        retries = state.get("retries", 0)
        max_r = state.get("max_retries", 2)

        valid_ids = (
            {c["chunk_id"] for c in pkg.get("chunks", [])}
            | {f["fig_key"] for f in pkg.get("figures", [])}
        )
        bad = sorted(set(c for c in cites if c not in valid_ids))

        needs_retry = False
        reason = "citations ok"

        if state.get("package_status") != "ok":
            reason = f"package_status={state.get('package_status')}"
        elif not cites and state.get("answerable", False):
            needs_retry, reason = True, "answerable but no citations"
        elif bad:
            needs_retry, reason = True, f"bad citations: {bad[:3]}"

        if retries >= max_r:
            needs_retry = False
            reason += f" (retries={retries} >= max={max_r})"

        return {
            "citation_errors": bad,
            "needs_retry": needs_retry,
            "retries": retries + 1 if needs_retry else retries,
            "notes": [f"reflection: {reason}"],
        }
    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Node 5 — responder  (FIXED)
# ============================================================
def responder_node(state: VMState) -> Dict[str, Any]:
    try:
        ps = state.get("package_status", "ok")
        answerable = state.get("answerable")

        # --- Determine final status ---
        if ps == "no_evidence":
            final_status = "no_evidence"
        elif ps == "out_of_scope":
            final_status = "out_of_scope"
        elif state.get("citation_errors"):
            final_status = "citation_problem"
        elif answerable is False:
            # LLM judged the evidence insufficient → treat as no_evidence
            final_status = "no_evidence"
        else:
            final_status = "ok"

        # --- Refusal: rewrite answer text and clear citations ---
        answer_text = state.get("answer") or ""
        citations   = state.get("citations") or []

        if final_status == "no_evidence":
            answer_text = "I don't have enough evidence in the provided chapters to answer this."
            citations = []
        elif final_status == "out_of_scope":
            answer_text = "This question is outside the scope of the provided chapters."
            citations = []

        final = {
            "status":       final_status,
            "question":     state.get("question"),
            "answer":       answer_text,
            "citations":    citations,
            "used_chunks":  state.get("used_chunks", []),
            "used_figures": state.get("used_figures", []),
            "confidence":   float(state.get("confidence", 0.0)),
            "answerable":   bool(state.get("answerable", False)),
            "retries":      state.get("retries", 0),
            "route":        state.get("route", "text"),
            "notes":        state.get("notes", []),
        }
        return {"final": final, "notes": [f"responder: status={final_status}"]}
    except Exception as e:
        raise CustomException(e, sys) from e