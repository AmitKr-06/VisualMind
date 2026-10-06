"""LLM layer — unified interface to Gemini + Groq with fallback."""

from .base import BaseLLM, LLMResponse
from .gemini_client import GeminiClient
from .groq_client import GroqClient
from .llm_router import call_llm, get_llm
from .prompts import (
    ANSWER_PROMPT_TEMPLATE,
    FIGURE_PROMPT_TEMPLATE,
    QUERY_PROMPT_TEMPLATE,
    CONTEXTUALIZE_PROMPT_TEMPLATE,
)

__all__ = [
    "BaseLLM",
    "LLMResponse",
    "GeminiClient",
    "GroqClient",
    "call_llm",
    "get_llm",
    "ANSWER_PROMPT_TEMPLATE",
    "FIGURE_PROMPT_TEMPLATE",
    "QUERY_PROMPT_TEMPLATE",
    "CONTEXTUALIZE_PROMPT_TEMPLATE",
]