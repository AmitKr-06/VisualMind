"""Central configuration: paths, versions, and tunables."""
import json
from pathlib import Path

# --- Paths ---
PROJECT_ROOT   = Path(__file__).resolve().parent.parent
DATA_DIR       = PROJECT_ROOT / "data"
PROCESSED_DIR  = DATA_DIR / "processed"
EVAL_DIR       = DATA_DIR / "eval"
RESULTS_DIR    = DATA_DIR / "results"
EXTRACT_DIR    = DATA_DIR / "extracted"
FIG_DIR        = EXTRACT_DIR / "figures"
VLM_DIR        = DATA_DIR / "vision_cache"
UPLOADS_DIR    = DATA_DIR / "uploads"

# Ensure folders exist
for _d in (PROCESSED_DIR, EVAL_DIR, RESULTS_DIR, EXTRACT_DIR,
           FIG_DIR, VLM_DIR, UPLOADS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Versions ---
EXP5 = "v1"
EXP6 = "v1"
EXP7 = "v1"
EXP8 = "v1"
EXP9 = "v1"
APP_VERSION = "v1"

# --- Files ---
HANDOFF5_PATH = RESULTS_DIR / f"exp5_handoff_{EXP5}.json"
HANDOFF9_PATH = RESULTS_DIR / f"exp9_handoff_{EXP9}.json"
FROZEN_PATH   = RESULTS_DIR / "final_config_v5_frozen.json"

# --- Upload config (for module 10) ---
MAX_UPLOAD_MB     = 50
MAX_SESSIONS      = 100
SESSION_TTL_HOURS = 24


def load_config() -> dict:
    """
    Load the frozen pipeline config + handoffs.

    Merges fields from every expN_handoff_v1.json we can find, then
    falls back to final_config_v5_frozen.json for anything missing.
    The frozen config is the source of truth.
    """
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(f"Missing {FROZEN_PATH}. Run Exp 5 freeze first.")

    frozen = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))

    # Merge all handoffs (last one wins per field)
    merged_handoff: dict = {}
    for name in ("exp5_handoff_v1.json", "exp6_handoff_v1.json",
                 "exp7_handoff_v1.json", "exp8_handoff_v1.json",
                 "exp9_handoff_v1.json"):
        p = RESULTS_DIR / name
        if p.exists():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(d, dict):
                    # Only merge known "pipeline" keys to avoid clobbering
                    for k in ("chunk_file", "policy", "models",
                              "choice1", "choice2", "figure_dir"):
                        if k in d:
                            merged_handoff[k] = d[k]
            except Exception:
                pass

    # -------- Fallbacks from frozen --------
    if "chunk_file" not in merged_handoff:
        cv = frozen.get("chunk_version", "v3")
        cs = frozen.get("chunk_size", 1000)
        candidate = f"chunks_{cv}_{cs}.jsonl"
        # If that file exists, use it; otherwise try chunks.jsonl
        if (PROCESSED_DIR / candidate).exists():
            merged_handoff["chunk_file"] = candidate
        else:
            merged_handoff["chunk_file"] = "chunks.jsonl"

    if "policy" not in merged_handoff:
        merged_handoff["policy"] = frozen.get("policy", {
            "k_chunks": 3,
            "n_figures": 2,
            "tau_text": 0.115,
            "choice1": "A",
            "choice2": "E",
            "tau_image": {"E": 0.4803, "F": 0.0},
        })

    if "models" not in merged_handoff:
        merged_handoff["models"] = {
            "dense": frozen.get("models", {}).get("dense", "BAAI/bge-base-en-v1.5"),
            "reranker": frozen.get("models", {}).get("reranker", "BAAI/bge-reranker-base"),
            "answer_model_chain": [
                "gemini-3.1-flash-lite",
                "gemini-3-flash-preview",
                "gemini-3.5-flash-lite",
                "gemini-3.5-flash",
            ],
            "fallback_provider": "groq",
        }

    if "choice1" not in merged_handoff:
        merged_handoff["choice1"] = frozen.get("choice1", "A")
    if "choice2" not in merged_handoff:
        merged_handoff["choice2"] = frozen.get("choice2", "E")
    if "figure_dir" not in merged_handoff:
        merged_handoff["figure_dir"] = str(EXTRACT_DIR / "figures")

    # Frozen is the source of truth; handoffs override pipeline fields.
    merged = {**frozen, **merged_handoff}
    return merged