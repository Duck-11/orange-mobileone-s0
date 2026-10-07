"""Đánh giá model, chọn threshold và tạo báo cáo chuẩn hóa."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

LOGGER = logging.getLogger(__name__)


def predict_frame(model, dataset, frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Dự đoán theo đúng thứ tự frame và trả thời gian trung bình mỗi ảnh."""
    start = time.perf_counter()
    probabilities = model.predict(dataset, verbose=1).reshape(-1).astype(np.float64)
    elapsed = time.perf_counter() - start

    if len(probabilities) != len(frame):
        raise RuntimeError(
            f"Số dự đoán ({len(probabilities)}) không khớp số ảnh ({len(frame)})."
        )
    return np.clip(probabilities, 0.0, 1.0), elapsed * 1000.0 / max(len(frame), 1)


def _metrics_at_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Tính đầy đủ metric GOOD=0, BAD=1 tại một threshold."""
    y_pred = (probabilities >= threshold).astype(np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    try:
        roc_auc = float(roc_auc_score(y_true, probabilities))
    except ValueError:
        roc_auc = float("nan")

    metrics: dict[str, Any] = {
        "threshold": float(threshold),
        "total_images": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_good": float(precision[0]),
        "recall_good": float(recall[0]),
        "f1_good": float(f1[0]),
        "support_good": int(support[0]),
        "precision_bad": float(precision[1]),
        "recall_bad": float(recall[1]),
        "f1_bad": float(f1[1]),
        "support_bad": int(support[1]),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "roc_auc": roc_auc,
        "true_good_predicted_good": int(tn),
        "good_predicted_bad": int(fp),
        "bad_predicted_good": int(fn),
        "true_bad_predicted_bad": int(tp),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return metrics, y_pred


def choose_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> tuple[float, pd.DataFrame]:
    """Chọn threshold trên validation theo F1 BAD, rồi Recall BAD, rồi Accuracy."""
    if not 0 <= minimum < maximum <= 1:
        raise ValueError("Khoảng threshold phải nằm trong [0, 1].")
    if step <= 0:
        raise ValueError("threshold.step phải lớn hơn 0.")

    rows: list[dict[str, Any]] = []
    thresholds = np.arange(minimum, maximum + step / 2.0, step)
    for threshold in thresholds:
        metrics, _ = _metrics_at_threshold(y_true, probabilities, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": metrics["accuracy"],
                "precision_bad": metrics["precision_bad"],
                "recall_bad": metrics["recall_bad"],
                "f1_bad": metrics["f1_bad"],
                "false_negative_bad": metrics["bad_predicted_good"],
                "false_positive_bad": metrics["good_predicted_bad"],
            }
        )

    analysis = pd.DataFrame(rows)
    best = analysis.sort_values(
        ["f1_bad", "recall_bad", "accuracy", "false_negative_bad"],
        ascending=[False, False, False, True],
    ).iloc[0]
    return float(best["threshold"]), analysis


def _save_confusion_matrix(matrix: np.ndarray, path: Path, normalized: bool = False) -> None:
    """Vẽ confusion matrix."""
    display_matrix = matrix.astype(np.int64)
    values_format = "d"
    title = "Confusion Matrix"
    if normalized:
        display_matrix = matrix.astype(float)
        row_sums = display_matrix.sum(axis=1, keepdims=True)
        display_matrix = np.divide(
            display_matrix,
            row_sums,
            out=np.zeros_like(display_matrix),
            where=row_sums != 0,
        )
        values_format = ".2f"
        title = "Normalized Confusion Matrix"

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(display_matrix, display_labels=["GOOD", "BAD"])
    display.plot(ax=axis, values_format=values_format, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_roc_pr_plots(y_true: np.ndarray, probabilities: np.ndarray, report_dir: Path) -> None:
    """Lưu ROC và Precision-Recall curve."""
    if len(np.unique(y_true)) < 2:
        return

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
    roc_auc = roc_auc_score(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.4f}")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Ngẫu nhiên")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(report_dir / "roc_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(recall, precision)
    axis.set_xlabel("Recall BAD")
    axis.set_ylabel("Precision BAD")
    axis.set_title("Precision-Recall Curve")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(report_dir / "precision_recall_curve.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_training_plots(history: dict[str, Any], output_dir: Path) -> None:
    """Vẽ loss, accuracy và AUC của hai giai đoạn train."""
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    fine_start = int(history.get("fine_tune_start_epoch", 0))

    for metric, title, file_name in [
        ("loss", "Loss", "loss_curve.png"),
        ("accuracy", "Accuracy", "accuracy_curve.png"),
        ("auc", "AUC", "auc_curve.png"),
    ]:
        train_values = history.get(metric, [])
        validation_values = history.get(f"val_{metric}", [])
        if not train_values:
            continue
        epochs = np.arange(1, len(train_values) + 1)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.plot(epochs, train_values, label="Train")
        if validation_values:
            axis.plot(epochs, validation_values, label="Validation")
        if 0 < fine_start < len(epochs):
            axis.axvline(fine_start + 0.5, linestyle="--", label="Bắt đầu fine-tune")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(title)
        axis.set_title(f"Training {title}")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / file_name, dpi=180, bbox_inches="tight")
        plt.close(figure)


def save_threshold_analysis(analysis: pd.DataFrame, output_dir: Path) -> None:
    """Lưu bảng và biểu đồ threshold trên validation."""
    report_dir = output_dir / "reports" / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(report_dir / "threshold_analysis.csv", index=False, encoding="utf-8-sig")

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(analysis["threshold"], analysis["precision_bad"], label="Precision BAD")
    axis.plot(analysis["threshold"], analysis["recall_bad"], label="Recall BAD")
    axis.plot(analysis["threshold"], analysis["f1_bad"], label="F1 BAD")
    axis.set_xlabel("Threshold")
    axis.set_ylabel("Score")
    axis.set_title("Validation Threshold Analysis")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(report_dir / "threshold_analysis.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_dataset_report(
    dataset_name: str,
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    average_inference_ms: float,
    output_dir: Path,
    model_name: str = "MobileNetV3Small",
) -> dict[str, Any]:
    """Lưu báo cáo đầy đủ cho một dataset."""
    safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in dataset_name)
    report_dir = output_dir / "reports" / safe_name
    report_dir.mkdir(parents=True, exist_ok=True)

    y_true = frame["label"].astype(int).to_numpy()
    metrics, y_pred = _metrics_at_threshold(y_true, probabilities, threshold)
    metrics.update(
        {
            "dataset": dataset_name,
            "model": model_name,
            "average_inference_ms_per_image": float(average_inference_ms),
        }
    )

    predictions = frame.copy()
    predictions["probability_good"] = 1.0 - probabilities
    predictions["probability_bad"] = probabilities
    predictions["predicted_label"] = y_pred
    predictions["predicted_class"] = np.where(y_pred == 1, "BAD", "GOOD")
    predictions["correct"] = predictions["predicted_label"] == predictions["label"].astype(int)
    predictions.to_csv(report_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    per_original_class = (
        predictions.groupby(["original_class", "binary_label"], as_index=False)
        .agg(
            total_images=("image_path", "count"),
            correct_images=("correct", "sum"),
            average_probability_bad=("probability_bad", "mean"),
        )
    )
    per_original_class["accuracy"] = (
        per_original_class["correct_images"] / per_original_class["total_images"]
    )
    per_original_class.to_csv(
        report_dir / "per_original_class.csv", index=False, encoding="utf-8-sig"
    )

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (report_dir / "classification_report.txt").write_text(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["GOOD", "BAD"],
            digits=6,
            zero_division=0,
        ),
        encoding="utf-8",
    )

    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int32)
    _save_confusion_matrix(matrix, report_dir / "confusion_matrix.png")
    _save_confusion_matrix(
        matrix, report_dir / "normalized_confusion_matrix.png", normalized=True
    )
    _save_roc_pr_plots(y_true, probabilities, report_dir)

    LOGGER.info(
        "%s | Accuracy %.4f | Recall BAD %.4f | F1 BAD %.4f",
        dataset_name,
        metrics["accuracy"],
        metrics["recall_bad"],
        metrics["f1_bad"],
    )
    return metrics


def append_summary(
    metrics_rows: list[dict[str, Any]],
    output_dir: Path,
    parameter_count: int,
    model_size_mb: float,
) -> Path:
    """Tạo bảng tổng hợp để nhóm ghép kết quả các backbone."""
    rows: list[dict[str, Any]] = []
    for metrics in metrics_rows:
        rows.append(
            {
                "backbone": metrics.get("model", "MobileNetV3Small"),
                "dataset": metrics["dataset"],
                "accuracy": metrics["accuracy"],
                "precision_bad": metrics["precision_bad"],
                "recall_bad": metrics["recall_bad"],
                "f1_bad": metrics["f1_bad"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc": metrics["roc_auc"],
                "good_predicted_bad": metrics["good_predicted_bad"],
                "bad_predicted_good": metrics["bad_predicted_good"],
                "threshold": metrics["threshold"],
                "parameter_count": parameter_count,
                "keras_model_size_mb": model_size_mb,
                "average_inference_ms_per_image": metrics[
                    "average_inference_ms_per_image"
                ],
            }
        )
    path = output_dir / "backbone_result.csv"
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
    return path
