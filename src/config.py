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
FACE_CASCADE_PATH = CASCADES_DIR / "haarcascade_frontalface_default.xml"
EYE_CASCADE_PATH = CASCADES_DIR / "haarcascade_eye.xml"

# -------------------------
# Hyperparameters: Training
# -------------------------
IMG_SIZE_EYES = (24, 24)
IMG_SIZE_YAWNS = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15

# -------------------------
# Hyperparameters: Detection
# -------------------------
EYE_DROWSY_SCORE_THRESHOLD = 15
YAWN_PROBABILITY_THRESHOLD = 0.7
YAWN_CONSECUTIVE_FRAMES = 8
YAWN_EVENT_LIMIT = 3
