import argparse
from pathlib import Path

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

ASSETS_DIR        = BASE_DIR / "assets"
ALARM_PATH        = ASSETS_DIR / "alarm.wav"
CASCADES_DIR      = ASSETS_DIR / "haarcascades"
FACE_CASCADE_PATH = CASCADES_DIR / "haarcascade_frontalface_alt2.xml"

# Eye cascade paths — kept so model.py / data_prep.py that may reference
# them do not break, but they are NOT used inside the detection loop.
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

# =============================================================================
# TRAINING
# =============================================================================
IMG_SIZE_EYES  = (24, 24)
IMG_SIZE_YAWNS = (64, 64)
BATCH_SIZE     = 256
EPOCHS         = 20

# =============================================================================
# EYE DETECTION THRESHOLDS
# =============================================================================

# Which output index of the eye CNN corresponds to "closed".
# Keras flow_from_directory assigns indices alphabetically:
#   "closed" < "open"  →  closed=0, open=1
# We probe the trained model at startup to confirm (see _infer_eye_closed_class_index).
EYE_CLOSED_CLASS_INDEX = 0

# Average CNN confidence (across both eye crops) needed to call eyes "closed".
# Fixed-proportion crops are clean and consistent, so 0.55 is a reliable midpoint.
EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD = 0.55

# Show "Eyes: Closed" label only after eyes have been continuously closed
# for this many seconds.  Requirement: 0.25 s
EYE_CLOSED_DISPLAY_SECONDS = 0.25

# Trigger drowsiness alarm after eyes have been continuously closed
# for this many seconds.  Requirement: 3.0 s
EYE_DROWSY_SECONDS_THRESHOLD = 3.0

# How many consecutive frames the CNN must call "closed" before the
# duration timer starts accumulating.  2 frames ≈ 0.067 s at 30 fps —
# absorbs single-frame CNN noise without adding perceptible latency.
# (Was 3, which added 0.1 s, pushing perceived latency to 0.4 s > 0.3 s requirement.)
EYE_CONSEC_FRAMES = 2

# =============================================================================
# YAWN DETECTION THRESHOLDS
# =============================================================================

# CNN confidence above which the mouth is considered "open / yawning".
# Talking pushes the CNN to ~0.30–0.40; genuine yawns reach ~0.60–0.95.
# 0.45 sits cleanly between the two.
YAWN_PROBABILITY_THRESHOLD = 0.45

# CNN confidence below which the mouth is considered "clearly closed".
# Hysteresis gap = 0.45 − 0.20 = 0.25 — wide enough to prevent oscillation.
YAWN_END_PROBABILITY_THRESHOLD = 0.20

# Mouth must stay above YAWN_PROBABILITY_THRESHOLD for this many continuous
# seconds before a yawn is counted.
# Requirement: mouth half-open for ≥ 0.2 s  — using 0.3 s for a small guard.
YAWN_OPEN_SECONDS_THRESHOLD = 0.3

# Show "Yawning" label after mouth has been above threshold for this long.
# Requirement: 0.3 s
YAWN_OPEN_DISPLAY_SECONDS = 0.3

# After a yawn is counted, mouth must stay below YAWN_END_PROBABILITY_THRESHOLD
# for this long before the lock is released and the next yawn can be counted.
YAWN_RELEASE_SECONDS_THRESHOLD = 0.8

# Minimum wall-clock gap between two counted yawn events.
# Requirement: 2.0 s
YAWN_MIN_GAP_SECONDS = 2.0

# Number of counted yawns that triggers the alarm.  Requirement: 2
YAWN_EVENT_LIMIT = 2

# =============================================================================
# SHARED CLI PARSER  (used by model.py / evaluate.py via run.py)
# =============================================================================
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", type=str, required=True,
        choices=["eyes", "yawns"],
        help="Which model to train or evaluate.",
    )
    args, _ = parser.parse_known_args()
    return args
