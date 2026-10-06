"""
Gemini client — the primary LLM provider.

Features:
- Disk-cached responses (keyed on model + prompt + image bytes)
- Retry on transient errors (5xx, timeout) with backoff
- Detects daily quota exhaustion → raises QuotaExhausted
- Optional image input (for vision tasks)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.exception import CustomException
from src.llm.base import (
    BaseLLM, LLMResponse, LLMError, QuotaExhausted, TransientError,
)

logger = logging.getLogger("visualmind.llm.gemini")


# ============================================================
# Constants
# ============================================================
GEMINI_MODEL_CHAIN = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
]

MIN_INTERVAL = 7.0   # seconds between calls (free tier)


# ============================================================
# Cache
# ============================================================
def _cache_key(model: str, prompt: str, image_bytes: Optional[bytes]) -> str:
    h = hashlib.sha1(model.encode() + b"||" + prompt.encode())
    if image_bytes:
        h.update(b"||")
        h.update(image_bytes)
    return h.hexdigest()


# ============================================================
# Helpers
# ============================================================
def _load_env_key() -> str:
    load_dotenv(override=True)
    key = os.environ.get("GEMINI_API_KEY", "").strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("GEMINI_API_KEY missing from .env")
    return key


def _mime_from_bytes(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


# ============================================================
# Client
# ============================================================
class GeminiClient(BaseLLM):
    name = "gemini"

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        model_chain: Optional[List[str]] = None,
    ):
        try:
            self.api_key = _load_env_key()
            self.client = genai.Client(api_key=self.api_key)
            self.model_chain = model_chain or list(GEMINI_MODEL_CHAIN)
            self.default_model = self.model_chain[0]

            self.cache_dir = Path(cache_dir) if cache_dir else None
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            self._last_call = 0.0
            logger.info(f"GeminiClient initialized | models: {self.model_chain}")
            logger.info(f"  key ends with: ...{self.api_key[-4:]}")

        except Exception as e:
            raise CustomException(e, sys) from e

    # --------------------------------------------------------
    def supports_vision(self) -> bool:
        return True

    # --------------------------------------------------------
    def _wait_rate_limit(self):
        wait = MIN_INTERVAL - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    # --------------------------------------------------------
    def _cache_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    # --------------------------------------------------------
    def _try_one_model(
        self,
        model: str,
        prompt: str,
        json_mode: bool,
        image_bytes: Optional[bytes],
        retries: int,
    ) -> LLMResponse:
        """Try a single Gemini model with retries. Raises on failure."""
        cache_key = _cache_key(model, prompt, image_bytes)
        cache_path = self._cache_path(cache_key)

        # Cache hit
        if cache_path and cache_path.exists():
            d = json.loads(cache_path.read_text(encoding="utf-8"))
            logger.debug(f"  cache hit ({cache_path.name})")
            return LLMResponse(
                text=d["text"], model=model, seconds=d["seconds"],
                cached=True, provider="gemini",
            )

        last_err: Optional[Exception] = None
        n429 = 0

        for attempt in range(retries):
            self._wait_rate_limit()
            t0 = time.time()
            self._last_call = t0

            try:
                contents: List[Any] = [prompt]
                if image_bytes:
                    contents = [
                        types.Part.from_bytes(
                            data=image_bytes,
                            mime_type=_mime_from_bytes(image_bytes),
                        ),
                        prompt,
                    ]

                config = (
                    types.GenerateContentConfig(response_mime_type="application/json")
                    if json_mode else None
                )

                resp = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
                text = (resp.text or "").strip()
                if not text:
                    raise RuntimeError("empty answer")

                seconds = round(time.time() - t0, 2)

                # Cache on success
                if cache_path:
                    cache_path.write_text(
                        json.dumps({"model": model, "prompt": prompt,
                                    "text": text, "seconds": seconds}),
                        encoding="utf-8",
                    )
                return LLMResponse(
                    text=text, model=model, seconds=seconds,
                    cached=False, provider="gemini",
                )

            except Exception as e:
                last_err = e
                msg = str(e).lower()

                # Quota / rate limit
                if "429" in msg or "resource_exhausted" in msg:
                    n429 += 1
                    daily = any(x in msg.replace(" ", "")
                                for x in ("perday", "requestsperday", "dailylimit"))
                    if daily:
                        logger.warning(f"  {model}: DAILY quota exhausted")
                        raise QuotaExhausted(f"{model}: daily quota exhausted")
                    if n429 >= 3:
                        logger.warning(f"  {model}: too many 429s, giving up")
                        raise QuotaExhausted(f"{model}: too many 429s")
                    time.sleep(15 * n429)
                    continue

                # Transient
                transient = any(
                    x in msg for x in
                    ("503", "500", "unavailable", "overloaded",
                     "timeout", "internal")
                )
                logger.warning(f"  {model} attempt {attempt + 1}/{retries}: {str(e)[:120]}")
                if not transient:
                    raise LLMError(f"{model}: {e}")
                time.sleep(5 * (attempt + 1))

        raise TransientError(f"{model}: all retries failed — {last_err}")

    # --------------------------------------------------------
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
        Generate a completion. Walks the model chain on failure.

        Order:
        1. Try `model` (or default)
        2. If QuotaExhausted → next model in chain
        3. If TransientError → next model in chain
        4. If all models exhausted → raise the last error
        """
        try:
            image_bytes = None
            if image_path:
                image_bytes = Path(image_path).read_bytes()

            models_to_try = [model] if model else list(self.model_chain)
            last_err: Optional[Exception] = None

            for m in models_to_try:
                try:
                    logger.debug(f"Trying Gemini model: {m}")
                    return self._try_one_model(m, prompt, json_mode, image_bytes, retries)
                except (QuotaExhausted, TransientError) as e:
                    last_err = e
                    logger.warning(f"  {m} failed: {type(e).__name__} — trying next model")
                    continue
                except LLMError:
                    raise  # non-recoverable, propagate immediately

            raise LLMError(f"All Gemini models exhausted. Last error: {last_err}")

        except (QuotaExhausted, TransientError, LLMError):
            raise
        except Exception as e:
            raise CustomException(e, sys) from e