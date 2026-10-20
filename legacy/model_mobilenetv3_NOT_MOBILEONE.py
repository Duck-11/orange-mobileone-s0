"""Xây dựng và huấn luyện trực tiếp MobileNetV3Small."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


def build_mobilenetv3_small(image_size: int, dropout: float):
    """Tạo model nhị phân GOOD/BAD từ MobileNetV3Small pretrained ImageNet."""
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="image")
    backbone = tf.keras.applications.MobileNetV3Small(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
        include_preprocessing=True,
    )
    backbone.trainable = False

    # training=False giữ BatchNormalization ở chế độ inference khi fine-tune.
    x = backbone(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout, name="dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="probability_bad")(x)

    model = tf.keras.Model(inputs, outputs, name="orange_mobilenetv3_small")
    return model, backbone


def compile_model(model, learning_rate: float) -> None:
    """Compile model với các metric cần thiết trong lúc train."""
    import tensorflow as tf

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision_bad"),
            tf.keras.metrics.Recall(name="recall_bad"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )


def enable_fine_tuning(backbone, last_layers: int) -> None:
    """Mở các tầng cuối để fine-tune, giữ BatchNormalization đóng băng."""
    import tensorflow as tf

    if last_layers <= 0:
        raise ValueError("fine_tune_last_layers phải lớn hơn 0.")

    backbone.trainable = True
    cutoff = max(0, len(backbone.layers) - last_layers)
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= cutoff and not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )

    trainable_count = sum(layer.trainable for layer in backbone.layers)
    LOGGER.info(
        "Fine-tune %d/%d layer của MobileNetV3Small.",
        trainable_count,
        len(backbone.layers),
    )


def _callbacks(
    model_path: Path,
    log_path: Path,
    patience: int,
    reduce_lr_patience: int,
    append_log: bool,
    shared_checkpoint=None,
):
    """Tạo callback ngắn gọn, checkpoint theo validation AUC."""
    import tensorflow as tf

    checkpoint = shared_checkpoint or tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        verbose=1,
    )
    callbacks = [
        checkpoint,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=patience,
            restore_best_weights=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.3,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(log_path, append=append_log),
    ]
    return callbacks, checkpoint


def train_two_stage(
    model,
    backbone,
    train_dataset,
    validation_dataset,
    class_weight: dict[int, float],
    training_config: dict[str, Any],
    output_dir: Path,
):
    """Train classifier head rồi fine-tune các tầng cuối."""
    model_dir = output_dir / "models"
    log_dir = output_dir / "logs"
    model_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "best_model.keras"
    training_log_path = log_dir / "training_log.csv"

    compile_model(model, float(training_config["head_learning_rate"]))
    head_callbacks, checkpoint = _callbacks(
        model_path=best_model_path,
        log_path=training_log_path,
        patience=int(training_config["early_stopping_patience"]),
        reduce_lr_patience=int(training_config["reduce_lr_patience"]),
        append_log=False,
    )

    LOGGER.info("Bắt đầu giai đoạn 1: train classifier head.")
    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(training_config["head_epochs"]),
        class_weight=class_weight,
        callbacks=head_callbacks,
        verbose=1,
    )

    enable_fine_tuning(backbone, int(training_config["fine_tune_last_layers"]))
    compile_model(model, float(training_config["fine_tune_learning_rate"]))
    fine_callbacks, _ = _callbacks(
        model_path=best_model_path,
        log_path=training_log_path,
        patience=int(training_config["early_stopping_patience"]),
        reduce_lr_patience=int(training_config["reduce_lr_patience"]),
        append_log=True,
        shared_checkpoint=checkpoint,
    )

    LOGGER.info("Bắt đầu giai đoạn 2: fine-tune MobileNetV3Small.")
    fine_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=int(training_config["fine_tune_epochs"]),
        class_weight=class_weight,
        callbacks=fine_callbacks,
        verbose=1,
    )

    if not best_model_path.exists():
        raise RuntimeError("Không tạo được best_model.keras sau khi train.")

    final_model_path = model_dir / "final_model.keras"
    model.save(final_model_path)

    combined_history = combine_histories(head_history.history, fine_history.history)
    combined_history["fine_tune_start_epoch"] = len(
        next(iter(head_history.history.values()), [])
    )
    return best_model_path, final_model_path, combined_history


def combine_histories(
    head_history: dict[str, list[float]],
    fine_history: dict[str, list[float]],
) -> dict[str, Any]:
    """Ghép lịch sử hai giai đoạn theo cùng tên metric."""
    keys = sorted(set(head_history) | set(fine_history))
    combined: dict[str, Any] = {}
    for key in keys:
        combined[key] = list(head_history.get(key, [])) + list(fine_history.get(key, []))
    return combined


def save_model_summary(model, path: Path) -> None:
    """Lưu kiến trúc model dạng text."""
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def load_keras_model(path: Path):
    """Nạp model Keras không có custom layer/metric."""
    import tensorflow as tf

    return tf.keras.models.load_model(path, compile=False)
