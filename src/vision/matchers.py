"""
Figure matchers — map a query image or question to the best-matching figures.

Three strategies (matches Exp 5 "Choice 2"):
  E: description + text retrieval (uses src/retrieval + src/llm)
  F: CLIP image embedding (fast, no API call)
  G: fusion of E and F with RRF

We use CLIP for F, and the analysis from image_analyzer + retrieval for E.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

from src.exception import CustomException
from src.vision.figure_manifest import FigureEntry, FigureManifest

logger = logging.getLogger("visualmind.vision.matchers")


# ============================================================
# Constants
# ============================================================
CLIP_MODEL_NAME = "clip-ViT-B-32"
RRF_K = 60


# ============================================================
# Figure matcher
# ============================================================
class FigureMatcher:
    """CLIP-based figure matcher + text-retrieval fusion."""

    def __init__(
        self,
        manifest: FigureManifest,
        clip_model_name: str = CLIP_MODEL_NAME,
        device: Optional[str] = None,
    ):
        try:
            from sentence_transformers import SentenceTransformer

            self.manifest = manifest
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

            logger.info(f"Loading CLIP model: {clip_model_name} on {self.device}")
            self.clip = SentenceTransformer(clip_model_name, device=self.device)

            # Pre-compute figure image embeddings
            self.fig_keys = []
            self.fig_embeddings = []
            for entry in manifest:
                png = entry.png_path(manifest.fig_dir)
                if not png.exists():
                    logger.warning(f"PNG missing: {png}")
                    continue
                try:
                    img = Image.open(png).convert("RGB")
                    emb = self.clip.encode([img], normalize_embeddings=True,
                                            show_progress_bar=False)[0]
                    self.fig_keys.append(entry.fig_key)
                    self.fig_embeddings.append(emb)
                except Exception as e:
                    logger.warning(f"Failed to embed {png.name}: {e}")

            self.fig_embeddings = (
                np.array(self.fig_embeddings, dtype="float32")
                if self.fig_embeddings else np.zeros((0, 512), dtype="float32")
            )
            logger.info(f"Figure image embeddings: {self.fig_embeddings.shape}")

        except Exception as e:
            raise CustomException(e, sys) from e

    # --------------------------------------------------------
    def rank_by_image(
        self,
        image_path: Path | str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank figures by CLIP image similarity."""
        try:
            img = Image.open(image_path).convert("RGB")
            q_emb = self.clip.encode([img], normalize_embeddings=True,
                                      show_progress_bar=False)[0]

            if self.fig_embeddings.shape[0] == 0:
                return []

            sims = self.fig_embeddings @ q_emb
            order = np.argsort(-sims)[:k]

            out = []
            for rank, i in enumerate(order, start=1):
                key = self.fig_keys[int(i)]
                entry = self.manifest.get_by_key(key)
                out.append({
                    "fig_key": key,
                    "figure_id": entry.figure_id if entry else "",
                    "caption": entry.caption if entry else "",
                    "score": round(float(sims[int(i)]), 4),
                    "rank": rank,
                    "retriever": "clip",
                })
            return out

        except Exception as e:
            raise CustomException(e, sys) from e

    # --------------------------------------------------------
    def rank_by_text(
        self,
        text: str,
        k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rank figures by text similarity against caption + description."""
        try:
            texts = []
            keys = []
            for entry in self.manifest:
                t = entry.caption
                if entry.description:
                    t += "\n" + entry.description
                texts.append(t)
                keys.append(entry.fig_key)

            if not texts:
                return []

            q_emb = self.clip.encode([text], normalize_embeddings=True,
                                      show_progress_bar=False)[0]
            fig_emb = self.clip.encode(texts, normalize_embeddings=True,
                                        show_progress_bar=False)

            sims = fig_emb @ q_emb
            order = np.argsort(-sims)[:k]

            out = []
            for rank, i in enumerate(order, start=1):
                key = keys[int(i)]
                entry = self.manifest.get_by_key(key)
                out.append({
                    "fig_key": key,
                    "figure_id": entry.figure_id if entry else "",
                    "caption": entry.caption if entry else "",
                    "score": round(float(sims[int(i)]), 4),
                    "rank": rank,
                    "retriever": "clip_text",
                })
            return out

        except Exception as e:
            raise CustomException(e, sys) from e


# ============================================================
# RRF fusion
# ============================================================
def _rrf_fuse(
    lists: List[List[Dict[str, Any]]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """Fuse multiple ranked lists of {fig_key, ...} with RRF."""
    fused: Dict[str, Dict[str, Any]] = {}
    for lst in lists:
        for r, item in enumerate(lst, start=1):
            key = item["fig_key"]
            entry = fused.setdefault(key, {"item": item, "score": 0.0})
            entry["score"] += 1.0 / (k + r)

    ranked = sorted(fused.values(), key=lambda e: -e["score"])
    out = []
    for i, e in enumerate(ranked, start=1):
        item = dict(e["item"])
        item["rank"] = i
        item["rrf_score"] = round(e["score"], 6)
        out.append(item)
    return out


def match_query_to_figures(
    matcher: FigureMatcher,
    image_path: Optional[Path | str] = None,
    text: Optional[str] = None,
    k: int = 5,
    strategy: str = "fusion",
) -> List[Dict[str, Any]]:
    """
    Match a query (image and/or text) to the best figures.

    Args:
        matcher: A FigureMatcher instance.
        image_path: Optional query image.
        text: Optional query text.
        k: Number of results.
        strategy: "clip" | "text" | "fusion".

    Returns:
        Ranked list of figures.
    """
    try:
        if strategy == "clip":
            if not image_path:
                raise ValueError("strategy='clip' requires image_path")
            return matcher.rank_by_image(image_path, k=k)

        if strategy == "text":
            if not text:
                raise ValueError("strategy='text' requires text")
            return matcher.rank_by_text(text, k=k)

        # Fusion
        lists = []
        if image_path:
            lists.append(matcher.rank_by_image(image_path, k=k * 2))
        if text:
            lists.append(matcher.rank_by_text(text, k=k * 2))

        if not lists:
            raise ValueError("Provide at least one of image_path or text")

        return _rrf_fuse(lists, k=RRF_K)[:k]

    except Exception as e:
        raise CustomException(e, sys) from e