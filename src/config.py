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
    """Load the frozen pipeline config + Exp 9 handoff."""
    if not HANDOFF9_PATH.exists():
        raise FileNotFoundError(f"Missing {HANDOFF9_PATH}. Run Exp 9 first.")
    if not FROZEN_PATH.exists():
        raise FileNotFoundError(f"Missing {FROZEN_PATH}. Run Exp 5 freeze first.")
    handoff = json.loads(HANDOFF9_PATH.read_text(encoding="utf-8"))
    frozen  = json.loads(FROZEN_PATH.read_text(encoding="utf-8"))
    return {"handoff": handoff, "frozen": frozen}