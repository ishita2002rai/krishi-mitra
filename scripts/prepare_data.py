"""Download the PlantVillage dataset from Kaggle and split the configured
classes into data/processed/{train,val} for scripts/train_classifier.py.

Usage:
    python scripts/prepare_data.py
"""

import os
import random
import shutil
from pathlib import Path

from dotenv import load_dotenv

from krishi_mitra.config import (
    PLANT_CLASSES,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
    TRAIN_DIR,
    VAL_DIR,
    VAL_SPLIT,
    normalize_class_name,
)
from krishi_mitra.utils import save_json, set_seed

KAGGLE_DATASET = "abdallahalidev/plantvillage-dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def download_dataset() -> Path:
    """Download + unzip the Kaggle PlantVillage dataset into RAW_DIR, if not already present."""
    load_dotenv()
    has_token = bool(os.environ.get("KAGGLE_API_TOKEN"))
    has_legacy_key = bool(os.environ.get("KAGGLE_USERNAME")) and bool(os.environ.get("KAGGLE_KEY"))
    has_token_file = Path("~/.kaggle/access_token").expanduser().exists()
    if not (has_token or has_legacy_key or has_token_file):
        raise SystemExit(
            "Missing Kaggle credentials. Copy .env.example to .env and fill in "
            "KAGGLE_API_TOKEN with a token from https://www.kaggle.com/settings -> API."
        )

    # The kaggle package reads credentials from the environment at import time,
    # so it must be imported only after load_dotenv() has populated os.environ.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    marker = RAW_DIR / ".downloaded"
    if marker.exists():
        print(f"Dataset already downloaded at {RAW_DIR} (delete {marker} to re-download).")
        return RAW_DIR

    print(f"Downloading Kaggle dataset '{KAGGLE_DATASET}' ...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True, quiet=False)
    marker.touch()
    return RAW_DIR


def find_color_dir(raw_dir: Path) -> Path:
    """Locate the 'color' split folder inside the extracted dataset (layout varies by mirror)."""
    candidates = [p for p in raw_dir.rglob("color") if p.is_dir()]
    if not candidates:
        raise SystemExit(
            f"Could not find a 'color' folder under {raw_dir}. "
            "Check the extracted dataset layout and update find_color_dir()."
        )
    return candidates[0]


def collect_class_images(color_dir: Path) -> dict:
    """Map each PLANT_CLASSES entry to the list of image paths in its matching folder."""
    normalized_to_actual = {
        normalize_class_name(p.name): p for p in color_dir.iterdir() if p.is_dir()
    }

    class_to_images = {}
    missing = []
    for class_name in PLANT_CLASSES:
        folder = normalized_to_actual.get(normalize_class_name(class_name))
        if folder is None:
            missing.append(class_name)
            continue
        class_to_images[class_name] = [
            p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
        ]

    if missing:
        raise SystemExit(f"Could not find folders for classes: {missing}")

    return class_to_images


def split_and_copy(class_to_images: dict) -> dict:
    """Stratified train/val split per class, copying files into data/processed/{train,val}/<class>/."""
    rng = random.Random(SEED)
    counts = {}

    for class_name, images in class_to_images.items():
        images = sorted(images)  # deterministic order before shuffling
        rng.shuffle(images)
        n_val = max(1, int(len(images) * VAL_SPLIT))
        val_images, train_images = images[:n_val], images[n_val:]

        for split_images, split_dir in ((train_images, TRAIN_DIR), (val_images, VAL_DIR)):
            out_dir = split_dir / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            for src in split_images:
                shutil.copyfile(src, out_dir / src.name)

        counts[class_name] = {"train": len(train_images), "val": len(val_images)}
        print(f"{class_name}: {len(train_images)} train / {len(val_images)} val")

    return counts


def main():
    set_seed(SEED)
    raw_dir = download_dataset()
    color_dir = find_color_dir(raw_dir)
    class_to_images = collect_class_images(color_dir)
    counts = split_and_copy(class_to_images)
    save_json(counts, PROCESSED_DIR / "split_counts.json")
    print(f"\nDone. Split counts saved to {PROCESSED_DIR / 'split_counts.json'}")


if __name__ == "__main__":
    main()
