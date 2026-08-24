"""Shared constants and paths for the Krishi-Mitra classifier (Increment 1) and
agent loop (Increment 2)."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "scans.db"
UPLOADS_DIR = DB_DIR / "uploads"

# Canonical PlantVillage class names (as used in the "color" split of the
# abdallahalidev/plantvillage-dataset Kaggle mirror). prepare_data.py matches
# folders to these names case/underscore-insensitively, since some mirrors use
# "Tomato_Bacterial_spot" instead of "Tomato___Bacterial_spot".
#
# Covers 11 crop species / 35 classes. Excludes Blueberry, Raspberry, and
# Soybean, which only have "healthy" images in this dataset (no disease
# examples), so a model trained on them could never actually diagnose
# anything for those crops.
PLANT_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
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


def display_name(class_name: str) -> str:
    """Human-readable form of a PlantVillage class name, e.g. 'Tomato___Early_blight' -> 'Early blight'."""
    return class_name.split("___")[-1].replace("_", " ").strip()


def crop_name(class_name: str) -> str:
    """Human-readable crop name from a PlantVillage class name, e.g.
    'Corn_(maize)___Common_rust_' -> 'Corn (maize)', 'Pepper,_bell___healthy' -> 'Pepper, bell'."""
    return class_name.split("___")[0].replace("_", " ").strip()


# Sorted, de-duplicated crop names covered by PLANT_CLASSES -- used to tell farmers
# what the app supports and to help the agent notice likely out-of-scope photos.
SUPPORTED_CROPS = sorted({crop_name(c) for c in PLANT_CLASSES})


# Increment 2: agent loop
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.85  # below this top-1 softmax confidence, ask a clarifying question
# llama-3.3-70b-versatile was retired from Groq's catalog; gpt-oss-120b is the
# closest available equivalent in capability tier.
GROQ_MODEL = "openai/gpt-oss-120b"

# Increment 3: weather context
FORECAST_DAYS = 3

# Increment 4: treatment plan + frontend + storage
TREATMENT_DISCLAIMER = (
    "This is general guidance, not a substitute for advice from your local agricultural "
    "extension officer -- confirm chemical products and dosages locally before applying."
)
