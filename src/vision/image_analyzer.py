"""
Image analyzer — describes figures and analyzes user-uploaded query images.

Uses the LLM router (Gemini Vision) with the prompts from src/llm/prompts.py.
Parses the JSON response into a strict schema.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exception import CustomException
from src.llm import call_llm
from src.llm.prompts import build_figure_prompt, build_query_prompt

logger = logging.getLogger("visualmind.vision.analyzer")


# ============================================================
# Response parsing
# ============================================================
DEFAULT_VISUAL = {
    "visual_type": "other",
    "description": "",
    "labels": [],
    "entities": [],
    "relationships": [],
    "values": [],
    "crop_ok": True,
    "uncertain": [],
    "in_scope": None,
    "reason": "",
}


def parse_visual_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse the JSON response from a vision call. Returns a dict or None.

    Handles: markdown fences, single-element arrays, missing keys.
    """
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

    # Fill defaults
    for k, default in DEFAULT_VISUAL.items():
        if k not in obj or obj[k] is None:
            obj[k] = default

    # Coerce list fields
    for k in ("labels", "entities", "relationships", "values", "uncertain"):
        if not isinstance(obj[k], list):
            obj[k] = [str(obj[k])] if obj[k] else []
        obj[k] = [str(x) for x in obj[k]]

    # Coerce strings
    for k in ("description", "visual_type", "reason"):
        if not isinstance(obj[k], str):
            obj[k] = str(obj[k])

    return obj


# ============================================================
# Public API
# ============================================================
def describe_figure(
    image_path: Path | str,
    caption: str,
    force_recompute: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Describe a textbook figure using Gemini Vision.

    Args:
        image_path: Path to the figure PNG.
        caption: The textbook caption for the figure.
        force_recompute: Ignore the disk cache.

    Returns:
        Parsed dict with the schema in DEFAULT_VISUAL, or None on failure.
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Figure not found: {image_path}")

        prompt = build_figure_prompt(caption)

        resp = call_llm(
            prompt,
            image_path=str(image_path),
            json_mode=True,
        )

        parsed = parse_visual_response(resp.text)
        if parsed is None:
            logger.warning(f"Could not parse figure response for {image_path.name}")
            return None

        parsed["_provider"] = resp.provider
        parsed["_model"] = resp.model
        parsed["_seconds"] = resp.seconds
        parsed["_cached"] = resp.cached
        return parsed

    except Exception as e:
        raise CustomException(e, sys) from e


def analyze_query_image(
    image_path: Path | str,
    force_recompute: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Analyze a user-uploaded query image (diagram, sketch, textbook photo).

    Returns:
        Parsed dict with keys: in_scope, reason, visual_type, description, ...
    """
    try:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Query image not found: {image_path}")

        prompt = build_query_prompt()

        resp = call_llm(
            prompt,
            image_path=str(image_path),
            json_mode=True,
        )

        parsed = parse_visual_response(resp.text)
        if parsed is None:
            logger.warning(f"Could not parse query response for {image_path.name}")
            return None

        # Normalize in_scope
        if parsed.get("in_scope") is not None:
            parsed["in_scope"] = (
                parsed["in_scope"] is True
                or str(parsed["in_scope"]).strip().lower() == "true"
            )

        parsed["_provider"] = resp.provider
        parsed["_model"] = resp.model
        parsed["_seconds"] = resp.seconds
        parsed["_cached"] = resp.cached
        return parsed

    except Exception as e:
        raise CustomException(e, sys) from e