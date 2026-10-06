"""
LLM router — walks through providers in order.

Default chain:
  1. Gemini  (primary, text + vision)
  2. Groq    (fallback, text only)

Vision-aware: if `image_path` is provided, providers that don't
support vision are skipped automatically.

Every call is cached on disk, so re-runs are instant.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from src.exception import CustomException
from src.llm.base import (
    BaseLLM, LLMResponse, LLMError, QuotaExhausted, TransientError,
)
from src.llm.gemini_client import GeminiClient
from src.llm.groq_client import GroqClient

logger = logging.getLogger("visualmind.llm.router")


# ============================================================
# Singleton registry
# ============================================================
_CLIENTS: Dict[str, BaseLLM] = {}
_ROUTER: Optional[List[BaseLLM]] = None


def _default_cache_dir() -> Path:
    from src.config import VLM_DIR
    return VLM_DIR


def get_llm(
    provider: str,
    cache_dir: Optional[Path | str] = None,
) -> BaseLLM:
    """
    Return a cached LLM client for the given provider.

    Args:
        provider: "gemini" or "groq".
        cache_dir: Optional override for the disk cache.

    Returns:
        A BaseLLM instance.
    """
    try:
        cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        key = f"{provider}::{cache_dir}"
        if key in _CLIENTS:
            return _CLIENTS[key]

        if provider == "gemini":
            client = GeminiClient(cache_dir=cache_dir)
        elif provider == "groq":
            client = GroqClient(cache_dir=cache_dir)
        else:
            raise ValueError(f"Unknown provider: {provider}")

        _CLIENTS[key] = client
        return client

    except Exception as e:
        raise CustomException(e, sys) from e


def _get_router() -> List[BaseLLM]:
    """Return the fallback chain, building it lazily."""
    global _ROUTER
    if _ROUTER is not None:
        return _ROUTER

    chain: List[BaseLLM] = []

    # 1. Gemini (primary — text + vision)
    try:
        chain.append(get_llm("gemini"))
    except Exception as e:
        logger.warning(f"Gemini unavailable: {e}")

    # 2. Groq (fallback — text only)
    try:
        chain.append(get_llm("groq"))
    except Exception as e:
        logger.warning(f"Groq unavailable: {e}")

    if not chain:
        raise RuntimeError("No LLM providers available. Check .env keys.")

    _ROUTER = chain
    logger.info(f"LLM router chain: {[c.name for c in chain]}")
    return chain


def clear_router_cache():
    """Clear the client + router singleton cache (useful for tests)."""
    global _ROUTER
    _CLIENTS.clear()
    _ROUTER = None
    logger.info("LLM router cache cleared")


# ============================================================
# Public API
# ============================================================
def call_llm(
    prompt: str,
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    json_mode: bool = True,
    retries: int = 3,
    image_path: Optional[str] = None,
) -> LLMResponse:
    """
    Call the LLM with automatic fallback.

    Args:
        prompt: The prompt.
        provider: Force a specific provider ("gemini" or "groq").
        model: Override the model for that provider.
        json_mode: Ask for strict JSON output.
        retries: Retries per provider.
        image_path: Optional image (Gemini only). When set, the router
                    skips providers that don't support vision.

    Returns:
        LLMResponse

    Raises:
        LLMError: if no provider succeeded.
    """
    try:
        # ----------------------------------------------------
        # Explicit provider → no fallback
        # ----------------------------------------------------
        if provider:
            client = get_llm(provider)

            # Guard: user asked for a provider that can't do vision
            if image_path and not client.supports_vision():
                raise LLMError(
                    f"Provider {provider!r} does not support image input. "
                    f"Use 'gemini' for vision tasks."
                )

            return client.generate(
                prompt,
                model=model,
                json_mode=json_mode,
                retries=retries,
                image_path=image_path,
            )

        # ----------------------------------------------------
        # Automatic chain
        # ----------------------------------------------------
        chain = _get_router()
        last_err: Optional[Exception] = None
        tried: List[str] = []

        for client in chain:
            # Skip providers that can't handle images
            if image_path and not client.supports_vision():
                logger.debug(f"{client.name} skipped (no vision support)")
                continue

            tried.append(client.name)
            try:
                logger.debug(f"Trying provider: {client.name}")
                return client.generate(
                    prompt,
                    model=model,
                    json_mode=json_mode,
                    retries=retries,
                    image_path=image_path,
                )
            except (QuotaExhausted, TransientError, LLMError) as e:
                last_err = e
                logger.warning(
                    f"  {client.name} failed: {type(e).__name__} — trying next provider"
                )
                continue

        if not tried:
            if image_path:
                raise LLMError(
                    "No vision-capable provider available for the image request. "
                    "Gemini must be configured in .env."
                )
            raise LLMError("No LLM providers available.")

        raise LLMError(
            f"All LLM providers failed ({', '.join(tried)}). Last error: {last_err}"
        )

    except LLMError:
        raise
    except Exception as e:
        raise CustomException(e, sys) from e