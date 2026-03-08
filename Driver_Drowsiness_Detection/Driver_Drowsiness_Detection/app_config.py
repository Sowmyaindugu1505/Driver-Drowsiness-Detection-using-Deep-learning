from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Core assets
ALARM_PATH = BASE_DIR / "alarm.wav"

# Data locations
ARCHIVE_PATH = BASE_DIR / "archive.zip"
EXTRACT_DIR = BASE_DIR / "datasets" / "archive_extracted"
EYE_DATA_DIR = BASE_DIR / "data"
YAWN_DATA_DIR = BASE_DIR / "data_yawn"

# Model locations
MODELS_DIR = BASE_DIR / "models"
EYE_MODEL_PATH = MODELS_DIR / "cnnCat2.h5"
YAWN_MODEL_PATH = MODELS_DIR / "yawn_cnn.h5"
YAWN_CLASS_MAP_PATH = MODELS_DIR / "yawn_class_indices.json"
