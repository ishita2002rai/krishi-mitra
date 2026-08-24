"""Shared constants and paths for the Krishi-Mitra classifier (Increment 1)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Canonical PlantVillage tomato class names (as used in the "color" split of
# the abdallahalidev/plantvillage-dataset Kaggle mirror). prepare_data.py
# matches folders to these names case/underscore-insensitively, since some
# mirrors use "Tomato_Bacterial_spot" instead of "Tomato___Bacterial_spot".
TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42
VAL_SPLIT = 0.2


def normalize_class_name(name: str) -> str:
    """Collapse naming variants (___/_/space, case) to one comparable key."""
    return name.lower().replace("___", "_").replace(" ", "_").replace("-", "_")
