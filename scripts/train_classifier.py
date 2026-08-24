"""Train a MobileNetV2 transfer-learning classifier on the tomato PlantVillage
classes prepared by scripts/prepare_data.py. Saves the trained model plus
accuracy/precision/recall/F1 metrics and a confusion matrix for the
evaluation section.

Usage:
    python scripts/train_classifier.py
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from krishi_mitra.config import BATCH_SIZE, IMG_SIZE, MODELS_DIR, OUTPUTS_DIR, PROCESSED_DIR, SEED, TRAIN_DIR, VAL_DIR
from krishi_mitra.utils import save_json, set_seed

HEAD_EPOCHS = 10
FINE_TUNE_EPOCHS = 8
FINE_TUNE_AT_LAYER = 100  # unfreeze MobileNetV2 layers from this index onward
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5
CHECKPOINT_PATH = MODELS_DIR / "mobilenetv2_tomato.keras"


def load_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, image_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED, label_mode="int"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED,
        label_mode="int",
        shuffle=False,
    )
    class_names = train_ds.class_names

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def compute_class_weights(class_names) -> dict:
    """Inverse-frequency class weights from prepare_data.py's split_counts.json,
    so the ~14x imbalance between tomato classes (e.g. mosaic virus vs. yellow leaf
    curl virus) doesn't get drowned out by the majority classes during training."""
    counts_path = PROCESSED_DIR / "split_counts.json"
    with open(counts_path) as f:
        split_counts = json.load(f)

    train_counts = np.array([split_counts[name]["train"] for name in class_names], dtype=np.float64)
    total = train_counts.sum()
    num_classes = len(class_names)
    weights = total / (num_classes * train_counts)
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes: int):
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentation",
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augmentation(inputs)
    x = preprocess(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs), base_model


def make_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(CHECKPOINT_PATH), monitor="val_loss", save_best_only=True),
    ]


def plot_history(history_head, history_fine, path):
    acc = history_head.history["accuracy"] + history_fine.history["accuracy"]
    val_acc = history_head.history["val_accuracy"] + history_fine.history["val_accuracy"]
    loss = history_head.history["loss"] + history_fine.history["loss"]
    val_loss = history_head.history["val_loss"] + history_fine.history["val_loss"]
    split_at = len(history_head.history["accuracy"]) - 0.5

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(acc, label="train")
    axes[0].plot(val_acc, label="val")
    axes[0].axvline(split_at, color="gray", linestyle="--", label="fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()

    axes[1].plot(loss, label="train")
    axes[1].plot(val_loss, label="val")
    axes[1].axvline(split_at, color="gray", linestyle="--")
    axes[1].set_title("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def evaluate(model, val_ds, class_names):
    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "per_class": report,
    }
    return metrics, cm


def main():
    set_seed(SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, class_names = load_datasets()
    save_json(class_names, MODELS_DIR / "class_names.json")
    class_weight = compute_class_weights(class_names)

    model, base_model = build_model(num_classes=len(class_names))
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    print("Phase 1: training classification head (base frozen)...")
    history_head = model.fit(
        train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=make_callbacks(), class_weight=class_weight
    )

    print(f"Phase 2: fine-tuning top layers of MobileNetV2 (from layer {FINE_TUNE_AT_LAYER})...")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(FINE_TUNE_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    initial_epoch = history_head.epoch[-1] + 1
    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=initial_epoch + FINE_TUNE_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=make_callbacks(),
        class_weight=class_weight,
    )

    plot_history(history_head, history_fine, OUTPUTS_DIR / "training_history.png")

    metrics, cm = evaluate(model, val_ds, class_names)
    save_json(metrics, OUTPUTS_DIR / "metrics.json")
    plot_confusion_matrix(cm, class_names, OUTPUTS_DIR / "confusion_matrix.png")

    print(f"\nFinal validation accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['f1_macro']:.4f} | Weighted F1: {metrics['f1_weighted']:.4f}")
    print(f"Model saved to {CHECKPOINT_PATH}")
    print(f"Metrics saved to {OUTPUTS_DIR / 'metrics.json'}")
    print(f"Confusion matrix saved to {OUTPUTS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
