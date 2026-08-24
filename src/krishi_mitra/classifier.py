"""Load the Increment 1 trained model and run top-k predictions on a single image."""

import json
from functools import lru_cache

import numpy as np
import tensorflow as tf

from krishi_mitra.config import IMG_SIZE, MODELS_DIR, TOP_K

MODEL_PATH = MODELS_DIR / "mobilenetv2_tomato.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"


@lru_cache(maxsize=1)
def load_classifier():
    if not MODEL_PATH.exists():
        raise SystemExit(f"No trained model found at {MODEL_PATH}. Run scripts/train_classifier.py first.")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def predict_topk(image_path: str, k: int = TOP_K) -> list:
    """Return the k most likely classes for an image, as [{"class": str, "confidence": float}, ...]."""
    model, class_names = load_classifier()
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    batch = np.expand_dims(tf.keras.utils.img_to_array(img), axis=0)

    probs = model.predict(batch, verbose=0)[0]
    top_indices = np.argsort(probs)[::-1][:k]
    return [{"class": class_names[i], "confidence": float(probs[i])} for i in top_indices]
