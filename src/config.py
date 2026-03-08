"""
src/config.py  –  Centralized hyperparameters & path definitions.

All threshold constants are defined here so that detector.py, model.py, and
evaluate.py never contain magic numbers.
"""

import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Absolute Path Definitions
# ─────────────────────────────────────────────────────────────────────────────
# BASE_DIR resolves to the project root (one level up from src/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Assets
ASSETS_DIR      = BASE_DIR / "assets"
ALARM_PATH      = ASSETS_DIR / "alarm.wav"

# Raw Data
DATA_DIR        = BASE_DIR / "data"
ARCHIVE_PATH    = DATA_DIR / "raw" / "archive.zip"
EXTRACT_DIR     = DATA_DIR / "raw" / "archive_extracted"

# Prepared Datasets
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EYE_DATA_DIR       = PROCESSED_DATA_DIR / "data_eyes"
YAWN_DATA_DIR      = PROCESSED_DATA_DIR / "data_yawns"

# Models and Cascades
MODELS_DIR          = BASE_DIR / "models"
EYE_MODEL_PATH      = MODELS_DIR / "eye_cnn.h5"
YAWN_MODEL_PATH     = MODELS_DIR / "yawn_cnn.h5"
YAWN_CLASS_MAP_PATH = MODELS_DIR / "yawn_class_indices.json"

CASCADES_DIR         = ASSETS_DIR / "haarcascades"
FACE_CASCADE_PATH    = CASCADES_DIR / "haarcascade_frontalface_alt2.xml"
LEFT_EYE_CASCADE_PATH  = CASCADES_DIR / "haarcascade_lefteye_2splits.xml"
RIGHT_EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_righteye_2splits.xml"

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters: Training
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE_EYES  = (24, 24)
IMG_SIZE_YAWNS = (64, 64)
BATCH_SIZE     = 256
EPOCHS         = 20           # raised from 2 – EarlyStopping will cut it short anyway

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters: Eye Detection
# ─────────────────────────────────────────────────────────────────────────────
EYE_CLOSED_CLASS_INDEX = 0
# CNN confidence required before we treat an eye as "closed"
EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD = 0.90

# ── Display threshold ──────────────────────────────────────────────────────
# Show "Eyes: Closed" (red) only once the eye has been continuously closed
# for at least this many seconds.  Below this → show "Eyes: Open" (green).
EYE_CLOSED_DISPLAY_SECONDS = 0.3          # NEW — was missing in original

# ── Alarm threshold ────────────────────────────────────────────────────────
# Trigger the drowsiness alarm once eyes are CONTINUOUSLY closed this long.
EYE_DROWSY_SECONDS_THRESHOLD = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters: Yawn Detection
# ─────────────────────────────────────────────────────────────────────────────
# CNN confidence to START counting a mouth as open
YAWN_PROBABILITY_THRESHOLD = 0.45
# CNN confidence to END / release a yawn (hysteresis – must drop below this)
YAWN_END_PROBABILITY_THRESHOLD = 0.30

# ── Display threshold ──────────────────────────────────────────────────────
# Show "Yawning" only once the mouth has been continuously open for this long.
# Below this → show "Not Yawning".
YAWN_OPEN_DISPLAY_SECONDS = 0.25          # NEW (renamed from YAWN_OPEN_SECONDS_THRESHOLD for clarity)

# How long mouth must stay open before the yawn EVENT is officially counted
YAWN_OPEN_SECONDS_THRESHOLD = 0.25       # kept identical to display threshold

# Time the mouth must stay CLOSED before the yawn is considered "released"
YAWN_RELEASE_SECONDS_THRESHOLD = 0.2

# Minimum gap between two counted yawn events (debounce)
YAWN_MIN_GAP_SECONDS = 0.4

# ── Alarm threshold ────────────────────────────────────────────────────────
# Trigger the drowsiness alarm when this many valid yawns have been counted.
YAWN_EVENT_LIMIT = 2


# ─────────────────────────────────────────────────────────────────────────────
# Shared CLI argument parser (used by model.py and evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    """Shared args parser for train/eval modules invoked via run.py."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["eyes", "yawns"],
        help="Choose which model branch to run.",
    )
    args, _ = parser.parse_known_args()
    return args
