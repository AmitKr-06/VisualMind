"""
Groq client — the fallback LLM provider.

Uses the OpenAI-compatible Chat Completions API at api.groq.com.
Reads GROQ_API_KEY from .env.

Model chain:
  1. openai/gpt-oss-20b      (fast, strong)
  2. openai/gpt-oss-120b     (higher quality)
  3. qwen/qwen3.8-27b        (alternative)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.exception import CustomException
from src.llm.base import (
    BaseLLM, LLMResponse, LLMError, QuotaExhausted, TransientError,
)

logger = logging.getLogger("visualmind.llm.groq")


# ============================================================
# Constants
# ============================================================
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MODEL_CHAIN = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]


# ============================================================
# Client
# ============================================================
class GroqClient(BaseLLM):
    name = "groq"

    def __init__(
        self,
        cache_dir: Optional[Path | str] = None,
        default_model: str = DEFAULT_GROQ_MODEL,
    ):
        try:
            load_dotenv(override=True)
            key = os.environ.get("GROQ_API_KEY", "").strip().strip('"').strip("'")
            if not key:
                raise RuntimeError("GROQ_API_KEY missing from .env — Groq fallback disabled")

            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError("openai package not installed — run: pip install openai")

            self.client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
            self.default_model = default_model
            self.model_chain = list(GROQ_MODEL_CHAIN)
            self.cache_dir = Path(cache_dir) if cache_dir else None
            if self.cache_dir:
                self.cache_dir.mkdir(parents=True, exist_ok=True)

            self._last_call = 0.0
            logger.info(f"GroqClient initialized | model: {default_model}")
            logger.info(f"  chain: {self.model_chain}")
            logger.info(f"  key ends with: ...{key[-4:]}")

        except Exception as e:
            raise CustomException(e, sys) from e

    def supports_vision(self) -> bool:
        return False

    # --------------------------------------------------------
    def _cache_path(self, model: str, prompt: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        key = hashlib.sha1(("groq||" + model + "||" + prompt).encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    # --------------------------------------------------------
    def _try_one_model(
        self,
        model: str,
        prompt: str,
        json_mode: bool,
        retries: int,
    ) -> LLMResponse:
        """Try a single Groq model with retries. Raises on failure."""
        cache_path = self._cache_path(model, prompt)

        # Cache hit
        if cache_path and cache_path.exists():
            d = json.loads(cache_path.read_text(encoding="utf-8"))
            return LLMResponse(
                text=d["text"], model=model, seconds=d["seconds"],
                cached=True, provider="groq",
            )

        last_err = None
        for attempt in range(retries):
            t0 = time.time()
            try:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You always return valid JSON."},
                        {"role": "user",   "content": prompt},
                    ],
                    "temperature": 0.2,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                resp = self.client.chat.completions.create(**kwargs)
                text = resp.choices[0].message.content.strip()
                if not text:
                    raise RuntimeError("empty answer")

                seconds = round(time.time() - t0, 2)

                if cache_path:
                    cache_path.write_text(
                        json.dumps({"model": model, "prompt": prompt,
                                    "text": text, "seconds": seconds}),
                        encoding="utf-8",
                    )
                return LLMResponse(
                    text=text, model=model, seconds=seconds,
                    cached=False, provider="groq",
                )

            except Exception as e:
                last_err = e
                msg = str(e).lower()

                # Quota / rate limit → try a different model
                if "429" in msg or "quota" in msg or "rate" in msg:
                    raise QuotaExhausted(f"groq:{model}: {e}")

                # Model not found / retired → try a different model
                if "404" in msg or "model_not_found" in msg or "does not exist" in msg:
                    raise TransientError(f"groq:{model}: model not found — trying next")

                logger.warning(f"  groq:{model} attempt {attempt + 1}/{retries}: {str(e)[:120]}")
                time.sleep(3 * (attempt + 1))

        raise TransientError(f"groq:{model}: all retries failed — {last_err}")

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
        try:
            if image_path:
                raise LLMError("GroqClient does not support image input")

            models_to_try = [model] if model else self.model_chain
            last_err = None

            for m in models_to_try:
                try:
                    logger.debug(f"Trying Groq model: {m}")
                    return self._try_one_model(m, prompt, json_mode, retries)
                except (QuotaExhausted, TransientError) as e:
                    last_err = e
                    logger.warning(f"  {m} failed — trying next model")
                    continue

            raise LLMError(f"All Groq models failed. Last error: {last_err}")

        except (QuotaExhausted, TransientError, LLMError):
            raise
        except Exception as e:
            raise CustomException(e, sys) from e