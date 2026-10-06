"""
Base LLM interface + result types.

Every LLM provider implements `BaseLLM.generate(prompt, json_mode=True)`.
The router in `llm_router.py` walks through providers in order,
returning the first successful response.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger("visualmind.llm.base")


# ============================================================
# Exceptions
# ============================================================
class LLMError(Exception):
    """Base error for all LLM failures."""


class QuotaExhausted(LLMError):
    """Raised when a provider's daily quota is used up — try the next provider."""


class TransientError(LLMError):
    """Raised on 5xx / timeout / rate-limit; worth retrying the SAME provider."""


# ============================================================
# Response type
# ============================================================
@dataclass
class LLMResponse:
    """The result of one LLM call."""
    text: str
    model: str
    seconds: float
    cached: bool = False
    provider: str = ""
    raw: Optional[Dict[str, Any]] = field(default=None, repr=False)

    def __repr__(self) -> str:
        return (
            f"LLMResponse(provider={self.provider!r}, model={self.model!r}, "
            f"seconds={self.seconds:.2f}, cached={self.cached}, "
            f"text_len={len(self.text)})"
        )


# ============================================================
# Base class
# ============================================================
class BaseLLM(ABC):
    """Abstract base for all LLM providers."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        json_mode: bool = True,
        retries: int = 3,
        image_path: Optional[str] = None,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt.

        Args:
            prompt: The text prompt.
            model: Override the default model.
            json_mode: If True, ask the provider for strict JSON output.
            retries: How many times to retry on TransientError.
            image_path: Optional path to an image for vision-capable providers.

        Returns:
            LLMResponse

        Raises:
            QuotaExhausted — if the daily quota is used up.
            TransientError — if a retryable failure occurs.
            LLMError       — for all other errors.
        """
        raise NotImplementedError

    def supports_vision(self) -> bool:
        """Whether this provider can accept image_path."""
        return False