"""Vision — figure descriptions, image analysis, and image-to-figure matching."""

from .figure_manifest import (
    FigureManifest,
    FigureEntry,
    load_figure_manifest,
)
from .image_analyzer import (
    describe_figure,
    analyze_query_image,
    parse_visual_response,
)
from .matchers import (
    FigureMatcher,
    match_query_to_figures,
)

__all__ = [
    "FigureManifest",
    "FigureEntry",
    "load_figure_manifest",
    "describe_figure",
    "analyze_query_image",
    "parse_visual_response",
    "FigureMatcher",
    "match_query_to_figures",
]