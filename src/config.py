import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR             = BASE_DIR / "assets"
ALARM_PATH             = ASSETS_DIR / "alarm.wav"
CASCADES_DIR           = ASSETS_DIR / "haarcascades"
FACE_CASCADE_PATH      = CASCADES_DIR / "haarcascade_frontalface_alt2.xml"
LEFT_EYE_CASCADE_PATH  = CASCADES_DIR / "haarcascade_lefteye_2splits.xml"
RIGHT_EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_righteye_2splits.xml"

DATA_DIR           = BASE_DIR / "data"
ARCHIVE_PATH       = DATA_DIR / "raw" / "archive.zip"
EXTRACT_DIR        = DATA_DIR / "raw" / "archive_extracted"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EYE_DATA_DIR       = PROCESSED_DATA_DIR / "data_eyes"
YAWN_DATA_DIR      = PROCESSED_DATA_DIR / "data_yawns"

MODELS_DIR          = BASE_DIR / "models"
EYE_MODEL_PATH      = MODELS_DIR / "eye_cnn.h5"
YAWN_MODEL_PATH     = MODELS_DIR / "yawn_cnn.h5"
YAWN_CLASS_MAP_PATH = MODELS_DIR / "yawn_class_indices.json"

# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE_EYES  = (24, 24)
IMG_SIZE_YAWNS = (64, 64)
BATCH_SIZE     = 256
EPOCHS         = 20   # EarlyStopping will cut short when val_loss stops improving

# ─────────────────────────────────────────────────────────────────────────────
# Eye detection thresholds
# ─────────────────────────────────────────────────────────────────────────────
EYE_CLOSED_CLASS_INDEX = 0

# CNN confidence required to classify a single eye as closed
EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD = 0.90

# Eyes must be continuously closed for this long before the LABEL changes
# Requirement: >= 0.3 s
EYE_CLOSED_DISPLAY_SECONDS = 0.3

# Eyes must be continuously closed for this long to TRIGGER THE ALARM
# Requirement: >= 3.0 s
EYE_DROWSY_SECONDS_THRESHOLD = 3.0

# ─────────────────────────────────────────────────────────────────────────────
# Yawn detection thresholds
# ─────────────────────────────────────────────────────────────────────────────

# CNN confidence required to START treating the mouth as open.
# Raised from 0.45 → 0.65 to reject talking / sighing (FIX for false counts).
YAWN_PROBABILITY_THRESHOLD = 0.65

# CNN confidence that must be sustained to END / release a yawn.
# Must be at least 0.25 below start so there is a clear hysteresis band.
# 0.65 - 0.30 = 0.35 gap  (FIX: was only 0.15, causing oscillation).
YAWN_END_PROBABILITY_THRESHOLD = 0.30

# Mouth must stay ABOVE YAWN_PROBABILITY_THRESHOLD for this many continuous
# seconds before the yawn is COUNTED.
# Raised from 0.25 s → 1.5 s  (FIX: 0.25 s = ~7 frames, too easy to trigger
# from a brief wide-mouth expression).
# A genuine yawn mouth-open phase lasts 2-6 s; 1.5 s is the minimum gate.
YAWN_OPEN_SECONDS_THRESHOLD = 1.5

# Mouth must be ABOVE threshold for this long before the LABEL shows "Yawning".
# Requirement: >= 0.25 s  (unchanged — display responds faster than count).
YAWN_OPEN_DISPLAY_SECONDS = 0.25

# After yawn_in_progress becomes True, the mouth must stay BELOW
# YAWN_END_PROBABILITY_THRESHOLD for this long before the yawn is "released".
# Raised from 0.2 s → 1.5 s  (FIX: 0.2 s is too short; brief lip movements
# after a yawn kept re-triggering).
YAWN_RELEASE_SECONDS_THRESHOLD = 1.5

# Minimum wall-clock gap between two counted yawn events.
# Raised from 0.4 s → 3.0 s  (FIX: 0.4 s let a single physical yawn be
# counted twice if the EMA briefly dipped and recovered).
YAWN_MIN_GAP_SECONDS = 3.0

# Number of valid yawns that TRIGGERS THE ALARM.
# Requirement: >= 2
YAWN_EVENT_LIMIT = 2

# ─────────────────────────────────────────────────────────────────────────────
# Shared CLI parser (used by model.py and evaluate.py via run.py)
# ─────────────────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=str, required=True,
        choices=["eyes", "yawns"],
        help="Choose which model branch to run.",
    )
    args, _ = parser.parse_known_args()
    return args
