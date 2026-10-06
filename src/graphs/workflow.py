"""
Build and compile the VisualMindGraph.

Dependencies (text_retriever, figure_matcher, image_analyzer) are injected
via the `build_workflow()` factory. The compiled graph is cached for reuse.
"""
from __future__ import annotations

import logging
import sys
from typing import Any, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.exception import CustomException
from src.graphs.nodes import (
    answer_node,
    make_retrieval_node,
    reflection_node,
    responder_node,
    router_node,
)
from src.graphs.state import VMState

logger = logging.getLogger("visualmind.graphs.workflow")


# ============================================================
# Cache
# ============================================================
_WORKFLOW = None
_META: dict = {}


# ============================================================
# Builder
# ============================================================
def build_workflow(
    text_retriever,
    figure_matcher=None,
    image_analyzer=None,
    k_chunks: int = 3,
    n_figures: int = 2,
    tau_text: float = 0.115,
    use_memory: bool = True,
):
    """
    Build and compile the VisualMind state machine.

    Args:
        text_retriever: A HybridRetriever (from module 3). Required.
        figure_matcher: A FigureMatcher (from module 5). Optional.
        image_analyzer: A callable analyze_query_image(path) (from module 5). Optional.
        k_chunks: Number of answer chunks per package.
        n_figures: Number of figures per package.
        tau_text: Evidence threshold below which we return no_evidence.
        use_memory: Attach a MemorySaver checkpointer (enables multi-turn).

    Returns:
        Compiled LangGraph graph.
    """
    try:
        logger.info("Building VisualMind workflow...")

        retrieval_node = make_retrieval_node(
            text_retriever=text_retriever,
            figure_matcher=figure_matcher,
            image_analyzer=image_analyzer,
            k_chunks=k_chunks,
            n_figures=n_figures,
            tau_text=tau_text,
        )

        graph = StateGraph(VMState)
        graph.add_node("router",     router_node)
        graph.add_node("retrieval",  retrieval_node)
        graph.add_node("answer",     answer_node)
        graph.add_node("reflection", reflection_node)
        graph.add_node("responder",  responder_node)

        graph.add_edge(START,       "router")
        graph.add_edge("router",    "retrieval")
        graph.add_edge("retrieval", "answer")
        graph.add_edge("answer",    "reflection")

        graph.add_conditional_edges(
            "reflection",
            lambda s: "answer" if s.get("needs_retry") else "responder",
            {"answer": "answer", "responder": "responder"},
        )
        graph.add_edge("responder", END)

        if use_memory:
            compiled = graph.compile(checkpointer=MemorySaver())
        else:
            compiled = graph.compile()

        logger.info("VisualMind workflow compiled")
        return compiled

    except Exception as e:
        raise CustomException(e, sys) from e


# ============================================================
# Cached access
# ============================================================
def get_workflow(
    text_retriever,
    figure_matcher=None,
    image_analyzer=None,
    **kwargs,
):
    """Build once and cache."""
    global _WORKFLOW, _META
    if _WORKFLOW is not None:
        return _WORKFLOW

    _WORKFLOW = build_workflow(
        text_retriever=text_retriever,
        figure_matcher=figure_matcher,
        image_analyzer=image_analyzer,
        **kwargs,
    )
    _META = {
        "nodes": ["router", "retrieval", "answer", "reflection", "responder"],
        "conditional_edges": {"reflection": ["answer", "responder"]},
        "checkpointer": "MemorySaver",
    }
    return _WORKFLOW


def clear_workflow_cache():
    global _WORKFLOW, _META
    _WORKFLOW = None
    _META = {}
    logger.info("Workflow cache cleared")


# ============================================================
# Convenience — run once
# ============================================================
def run_once(
    question: Optional[str] = None,
    image_path: Optional[str] = None,
    thread_id: str = "default",
) -> dict:
    """Invoke the cached workflow once. Requires get_workflow() first."""
    if _WORKFLOW is None:
        raise RuntimeError("Workflow not initialized. Call get_workflow() first.")

    config = {"configurable": {"thread_id": thread_id}}
    state_in = {
        "question":     question or "",
        "image_path":   image_path,
        "thread_id":    thread_id,
        "citations":    [],
        "used_chunks":  [],
        "used_figures": [],
        "notes":        [],
    }
    out = _WORKFLOW.invoke(state_in, config=config)
    return out["final"]