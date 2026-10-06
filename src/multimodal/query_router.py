"""
Query router — classify an incoming request as text / image / mixed.

Rules:
- image_path + question → "mixed"
- image_path only        → "image"
- question only          → "text"
- neither                → "text" (will be refused by the pipeline)
"""
from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from src.exception import CustomException

logger = logging.getLogger("visualmind.multimodal.router")


# ============================================================
# Types
# ============================================================
class QueryType(str, Enum):
    TEXT  = "text"
    IMAGE = "image"
    MIXED = "mixed"


@dataclass
class RoutedQuery:
    type: QueryType
    question: str
    image_path: Optional[str]
    reason: str

    def is_text_only(self) -> bool:
        return self.type == QueryType.TEXT


# ============================================================
# Visual intent regex
# ============================================================
VISUAL_WORDS = re.compile(
    r"\b(figure|fig|diagram|picture|image|draw|sketch|illustrat\w*|"
    r"graph|plot|circle|look like|shown|shows?|depict\w*)\b",
    re.I,
)

ID_FIGURE = re.compile(r"\b(fig(?:ure)?)\.?\s*(\d+\.\d+)", re.I)


def has_visual_intent(question: str) -> bool:
    """Does the question ask about a figure or use visual words?"""
    return bool(VISUAL_WORDS.search(question)) or bool(ID_FIGURE.search(question))


# ============================================================
# Router
# ============================================================
def classify_query(
    question: Optional[str] = None,
    image_path: Optional[str] = None,
) -> RoutedQuery:
    """
    Classify an incoming query.

    Args:
        question: Optional text question.
        image_path: Optional path to an uploaded image.

    Returns:
        RoutedQuery
    """
    try:
        q = (question or "").strip()
        img = image_path if image_path else None

        if img and q:
            r = RoutedQuery(
                type=QueryType.MIXED,
                question=q,
                image_path=str(img),
                reason="image + question",
            )
        elif img:
            r = RoutedQuery(
                type=QueryType.IMAGE,
                question="",
                image_path=str(img),
                reason="image only",
            )
        elif q:
            r = RoutedQuery(
                type=QueryType.TEXT,
                question=q,
                image_path=None,
                reason="text-only",
            )
        else:
            r = RoutedQuery(
                type=QueryType.TEXT,
                question="",
                image_path=None,
                reason="empty input → default to text (will be refused)",
            )

        logger.debug(f"Router: type={r.type.value} | {r.reason}")
        return r

    except Exception as e:
        raise CustomException(e, sys) from e