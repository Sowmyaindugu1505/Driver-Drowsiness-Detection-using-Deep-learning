import argparse
from pathlib import Path

# -------------------------
# Absolute Path Definitions
# -------------------------
# BASE_DIR is now one level up from the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Assets
ASSETS_DIR = BASE_DIR / "assets"
ALARM_PATH = ASSETS_DIR / "alarm.wav"

# Raw Data
DATA_DIR = BASE_DIR / "data"
ARCHIVE_PATH = DATA_DIR / "raw" / "archive.zip"
EXTRACT_DIR = DATA_DIR / "raw" / "archive_extracted"

# Prepared Datasets
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EYE_DATA_DIR = PROCESSED_DATA_DIR / "data_eyes"
YAWN_DATA_DIR = PROCESSED_DATA_DIR / "data_yawns"

# Models and Cascades
MODELS_DIR = BASE_DIR / "models"
EYE_MODEL_PATH = MODELS_DIR / "eye_cnn.h5"
YAWN_MODEL_PATH = MODELS_DIR / "yawn_cnn.h5"
YAWN_CLASS_MAP_PATH = MODELS_DIR / "yawn_class_indices.json"

CASCADES_DIR = ASSETS_DIR / "haarcascades"
FACE_CASCADE_PATH = CASCADES_DIR / "haarcascade_frontalface_alt2.xml"
LEFT_EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_lefteye_2splits.xml"
RIGHT_EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_righteye_2splits.xml"

# -------------------------
# Hyperparameters: Training
# -------------------------
IMG_SIZE_EYES = (24, 24)
IMG_SIZE_YAWNS = (64, 64)
BATCH_SIZE = 256
EPOCHS = 2

# -------------------------
# Hyperparameters: Detection
# -------------------------
EYE_CLOSED_CLASS_INDEX = 0
EYE_FULLY_CLOSED_PROBABILITY_THRESHOLD = 0.90
EYE_DROWSY_SECONDS_THRESHOLD = 3.0
YAWN_PROBABILITY_THRESHOLD = 0.45
YAWN_END_PROBABILITY_THRESHOLD = 0.30
YAWN_OPEN_SECONDS_THRESHOLD = 0.3
YAWN_RELEASE_SECONDS_THRESHOLD = 0.2
YAWN_MIN_GAP_SECONDS = 0.4
YAWN_EVENT_LIMIT = 2


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
