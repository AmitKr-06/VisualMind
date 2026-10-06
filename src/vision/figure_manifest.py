"""
Figure manifest — loads the frozen figure metadata from Exp 4/5.

Provides:
- Loaded FigureEntry objects (one per figure)
- Fast lookup by fig_key or figure_id
- Caption + description + PNG path for each figure
- CAP_LINKS mapping: chunk_id -> [fig_key] (which chunk contains each caption)
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.exception import CustomException

logger = logging.getLogger("visualmind.vision.manifest")


# ============================================================
# Constants
# ============================================================
DEFAULT_MANIFEST_V1 = "figures_v1.jsonl"   # base manifest from Exp 4
DEFAULT_MANIFEST_V2 = "figures_v2.json"    # reviewed manifest from Exp 5


# ============================================================
# Data structures
# ============================================================
@dataclass
class FigureEntry:
    """One figure, with metadata + paths."""
    fig_key: str                    # e.g. "doc1_life_processes_fig5.1"
    figure_id: str                  # e.g. "5.1"
    doc: str                        # e.g. "doc1_life_processes.pdf"
    page: int                       # 1-based page number
    caption: str                    # textbook caption
    file: str                       # PNG filename, e.g. "doc1_life_processes_p04_fig5.1.png"

    # From the reviewed manifest (may be empty)
    description: str = ""
    visual_type: str = "other"
    labels: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    relationships: List[str] = field(default_factory=list)
    values: List[str] = field(default_factory=list)
    crop_ok: bool = True

    # Review verdict from Exp 5
    description_usable: bool = False
    review_verdict: str = ""

    def png_path(self, fig_dir: Path) -> Path:
        """Absolute path to the PNG for this figure."""
        return fig_dir / self.file

    def summary(self) -> Dict[str, Any]:
        return {
            "fig_key": self.fig_key,
            "figure_id": self.figure_id,
            "doc": self.doc,
            "page": self.page,
            "caption": self.caption[:80],
            "has_description": bool(self.description),
            "has_png": True,
            "usable": self.description_usable,
        }


# ============================================================
# Manifest
# ============================================================
class FigureManifest:
    """Loaded, indexed set of figures."""

    def __init__(self, entries: List[FigureEntry], fig_dir: Path):
        self.entries = entries
        self.fig_dir = Path(fig_dir)
        self.by_key = {e.fig_key: e for e in entries}
        self.by_id  = {e.figure_id: e for e in entries}
        self.cap_links: Dict[str, List[str]] = {}   # chunk_id -> [fig_key]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)

    def get_by_key(self, fig_key: str) -> Optional[FigureEntry]:
        return self.by_key.get(fig_key)

    def get_by_id(self, figure_id: str) -> Optional[FigureEntry]:
        return self.by_id.get(figure_id)

    def describe_all(self) -> List[str]:
        """Return a list of 'Figure X.Y: <caption>' strings for LLM prompts."""
        return [f"Figure {e.figure_id}: {e.caption}" for e in self.entries]


# ============================================================
# Loading
# ============================================================
def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSON or JSONL file. Detects format automatically:
    - If the file starts with '[' → standard JSON array.
    - Otherwise → treat as JSONL (one JSON object per line).
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    # Standard JSON array/object
    if text[0] in "[{":
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
            return [data]
        except json.JSONDecodeError:
            # Fall through to JSONL parsing
            pass

    # JSONL: one object per line
    out: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping malformed line in {path.name}: {e}")
            continue
    return out


def load_figure_manifest(
    extract_dir: Path | str,
    fig_dir: Optional[Path | str] = None,
) -> FigureManifest:
    """
    Load the frozen figure manifest.

    Merges:
    - figures_v1.jsonl   (base metadata from Exp 4)
    - figures_v2.json    (reviewed descriptions from Exp 5)

    Args:
        extract_dir: Where the manifest files live (e.g. data/extracted/).
        fig_dir: Where the PNG crops live. Defaults to extract_dir/figures.

    Returns:
        FigureManifest
    """
    try:
        extract_dir = Path(extract_dir)
        fig_dir = Path(fig_dir) if fig_dir else extract_dir / "figures"

        v1_path = extract_dir / DEFAULT_MANIFEST_V1
        v2_path = extract_dir / DEFAULT_MANIFEST_V2

        if not v1_path.exists():
            raise FileNotFoundError(f"Missing base manifest: {v1_path}")

        v1 = _load_jsonl(v1_path)
        v2 = _load_json(v2_path) if v2_path.exists() else []

        # Index v2 by fig_key
        v2_by_key = {r.get("fig_key"): r for r in v2 if r.get("fig_key")}

        entries: List[FigureEntry] = []
        for r in v1:
            key = r.get("fig_key")
            if not key:
                continue

            v2r = v2_by_key.get(key, {})
            visual = v2r.get("visual") or {}
            review = v2r.get("review") or {}

            entries.append(FigureEntry(
                fig_key=key,
                figure_id=str(r.get("figure_id", "")),
                doc=r.get("doc", ""),
                page=int(r.get("page", 1)),
                caption=r.get("caption", ""),
                file=r.get("file", ""),
                description=v2r.get("description", "") or visual.get("description", ""),
                visual_type=visual.get("visual_type", "other"),
                labels=list(visual.get("labels") or []),
                entities=list(visual.get("entities") or []),
                relationships=list(visual.get("relationships") or []),
                values=list(visual.get("values") or []),
                crop_ok=bool(visual.get("crop_ok", True)),
                description_usable=bool(v2r.get("description_usable", False)),
                review_verdict=str(review.get("verdict", "")),
            ))

        manifest = FigureManifest(entries, fig_dir)

        # Build CAP_LINKS
        for r in v1:
            cap_ids = r.get("caption_chunk_ids", {})
            for size, cids in cap_ids.items():
                for cid in cids:
                    manifest.cap_links.setdefault(cid, []).append(r["fig_key"])

        logger.info(
            f"Figure manifest loaded: {len(manifest)} figures | "
            f"{len(manifest.cap_links)} caption→chunk links"
        )
        return manifest

    except Exception as e:
        raise CustomException(e, sys) from e