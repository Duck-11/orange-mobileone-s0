"""CLI ngắn gọn cho project MobileNetV3Small phân loại cam GOOD/BAD."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from orange.data import (
    build_tf_dataset,
    calculate_class_weight,
    load_split_csvs,
    normalize_label,
    save_splits,
    scan_labeled_folder,
    set_global_seed,
    split_dataset,
)
from orange.evaluation import (
    append_summary,
    choose_threshold,
    predict_frame,
    save_dataset_report,
    save_threshold_analysis,
    save_training_plots,
)
from orange.model import (
    build_mobilenetv3_small,
    load_keras_model,
    save_model_summary,
    train_two_stage,
)

LOGGER = logging.getLogger("orange_mobilenetv3")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: Path) -> dict[str, Any]:
    """Đọc và kiểm tra config.yaml."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy config: {path}")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("config.yaml phải là một YAML object.")

    required_sections = ["project", "data", "training", "threshold"]
    missing = [name for name in required_sections if name not in config]
    if missing:
        raise ValueError(f"config.yaml thiếu các phần: {missing}")

    ratios = [
        float(config["data"]["train_ratio"]),
        float(config["data"]["validation_ratio"]),
        float(config["data"]["test_ratio"]),
    ]
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError(f"Tổng tỷ lệ train/validation/test phải bằng 1, hiện là {sum(ratios)}")

    class_mapping = config["data"].get("class_mapping", {})
    labels = {normalize_label(value) for value in class_mapping.values()}
    if labels != {0, 1}:
        raise ValueError("data.class_mapping phải tạo đủ hai nhãn GOOD và BAD.")

    for item in config.get("external_datasets", []) or []:
        if not item.get("name") or not item.get("path") or not item.get("class_mapping"):
            raise ValueError("Mỗi external dataset cần name, path và class_mapping.")
        external_labels = {normalize_label(value) for value in item["class_mapping"].values()}
        if external_labels != {0, 1}:
            raise ValueError(
                f"External dataset {item.get('name')} phải có cả GOOD và BAD."
            )

    return config


def output_dir_from(config: dict[str, Any]) -> Path:
    return Path(config["project"].get("output_dir", "outputs")).expanduser().resolve()


def prepare_data(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Quét dataset gốc, chia train/validation/test và lưu CSV."""
    data_config = config["data"]
    output_dir = output_dir_from(config)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest, invalid = scan_labeled_folder(
        root=data_config["source_dir"],
        class_mapping=data_config["class_mapping"],
        allowed_extensions=data_config.get(
            "allowed_extensions", [".jpg", ".jpeg", ".png"]
        ),
        validate_images=bool(data_config.get("validate_images", True)),
    )

    train_df, validation_df, test_df = split_dataset(
        manifest,
        train_ratio=float(data_config["train_ratio"]),
        validation_ratio=float(data_config["validation_ratio"]),
        test_ratio=float(data_config["test_ratio"]),
        seed=int(config["project"]["seed"]),
    )
    save_splits(
        output_dir,
        manifest,
        invalid,
        train_df,
        validation_df,
        test_df,
    )

    LOGGER.info("Đã chia dữ liệu: train=%d, validation=%d, test=%d", len(train_df), len(validation_df), len(test_df))
    LOGGER.info("Ảnh lỗi bị bỏ qua: %d", len(invalid))
    for split_name, frame in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        counts = frame["binary_label"].value_counts().to_dict()
        LOGGER.info("%s: %s", split_name, counts)

    return train_df, validation_df, test_df


def build_datasets(
    config: dict[str, Any],
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """Tạo ba tf.data pipeline."""
    data_config = config["data"]
    training_config = config["training"]
    common = {
        "image_size": int(data_config["image_size"]),
        "batch_size": int(training_config["batch_size"]),
        "seed": int(config["project"]["seed"]),
        "cache_dataset": bool(training_config.get("cache_dataset", False)),
    }
    return (
        build_tf_dataset(train_df, training=True, **common),
        build_tf_dataset(validation_df, training=False, **common),
        build_tf_dataset(test_df, training=False, **common),
    )


def get_or_create_splits(config: dict[str, Any]):
    """Đọc split có sẵn; báo rõ nếu chưa có."""
    return load_split_csvs(output_dir_from(config))


def train_and_evaluate(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Train MobileNetV3Small và tạo báo cáo internal test."""
    output_dir = output_dir_from(config)
    train_df, validation_df, test_df = get_or_create_splits(config)
    train_ds, validation_ds, test_ds = build_datasets(
        config, train_df, validation_df, test_df
    )

    class_weight = calculate_class_weight(train_df)
    LOGGER.info("Class weight: GOOD=%.4f, BAD=%.4f", class_weight[0], class_weight[1])

    model, backbone = build_mobilenetv3_small(
        image_size=int(config["data"]["image_size"]),
        dropout=float(config["training"]["dropout"]),
    )
    save_model_summary(model, output_dir / "model_summary.txt")

    best_model_path, _, history = train_two_stage(
        model=model,
        backbone=backbone,
        train_dataset=train_ds,
        validation_dataset=validation_ds,
        class_weight=class_weight,
        training_config=config["training"],
        output_dir=output_dir,
    )
    save_training_plots(history, output_dir)

    best_model = load_keras_model(best_model_path)
    validation_probabilities, validation_latency = predict_frame(
        best_model, validation_ds, validation_df
    )
    threshold, threshold_analysis = choose_threshold(
        validation_df["label"].astype(int).to_numpy(),
        validation_probabilities,
        minimum=float(config["threshold"]["min"]),
        maximum=float(config["threshold"]["max"]),
        step=float(config["threshold"]["step"]),
    )
    save_threshold_analysis(threshold_analysis, output_dir)
    threshold_payload = {
        "selected_on": "validation",
        "final_threshold": threshold,
        "validation_average_inference_ms_per_image": validation_latency,
    }
    (output_dir / "selected_threshold.json").write_text(
        json.dumps(threshold_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    LOGGER.info("Threshold chọn trên validation: %.4f", threshold)

    test_probabilities, test_latency = predict_frame(best_model, test_ds, test_df)
    internal_metrics = save_dataset_report(
        dataset_name="internal_test",
        frame=test_df,
        probabilities=test_probabilities,
        threshold=threshold,
        average_inference_ms=test_latency,
        output_dir=output_dir,
    )

    model_size_mb = best_model_path.stat().st_size / (1024 * 1024)
    summary_path = append_summary(
        [internal_metrics],
        output_dir,
        parameter_count=int(best_model.count_params()),
        model_size_mb=float(model_size_mb),
    )
    LOGGER.info("Bảng kết quả: %s", summary_path)
    return [internal_metrics]


def read_threshold(output_dir: Path) -> float:
    """Đọc threshold đã chọn trên validation."""
    path = output_dir / "selected_threshold.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có {path}. Hãy train model trước bằng `python main.py run` hoặc `train`."
        )
    return float(json.loads(path.read_text(encoding="utf-8"))["final_threshold"])


def evaluate_external_datasets(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Chạy hàng loạt tất cả external dataset trong config.yaml."""
    external_items = config.get("external_datasets", []) or []
    if not external_items:
        LOGGER.info("Không có external_datasets trong config.yaml; bỏ qua.")
        return []

    output_dir = output_dir_from(config)
    model_path = output_dir / "models" / "best_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model đã train: {model_path}")

    model = load_keras_model(model_path)
    threshold = read_threshold(output_dir)
    results: list[dict[str, Any]] = []

    for item in external_items:
        dataset_path = Path(str(item["path"])).expanduser()
        if not dataset_path.exists():
            LOGGER.warning(
                "Bỏ qua external dataset %s vì chưa tìm thấy đường dẫn: %s",
                item["name"],
                dataset_path,
            )
            continue
        LOGGER.info("Đang đánh giá external dataset: %s", item["name"])
        frame, invalid = scan_labeled_folder(
            root=dataset_path,
            class_mapping=item["class_mapping"],
            allowed_extensions=config["data"].get(
                "allowed_extensions", [".jpg", ".jpeg", ".png"]
            ),
            validate_images=bool(config["data"].get("validate_images", True)),
        )
        if not invalid.empty:
            invalid_dir = output_dir / "reports" / str(item["name"])
            invalid_dir.mkdir(parents=True, exist_ok=True)
            invalid.to_csv(
                invalid_dir / "invalid_images.csv", index=False, encoding="utf-8-sig"
            )

        dataset = build_tf_dataset(
            frame,
            image_size=int(config["data"]["image_size"]),
            batch_size=int(config["training"]["batch_size"]),
            training=False,
            seed=int(config["project"]["seed"]),
            cache_dataset=False,
        )
        probabilities, latency = predict_frame(model, dataset, frame)
        metrics = save_dataset_report(
            dataset_name=str(item["name"]),
            frame=frame,
            probabilities=probabilities,
            threshold=threshold,
            average_inference_ms=latency,
            output_dir=output_dir,
        )
        results.append(metrics)

    internal_path = output_dir / "reports" / "internal_test" / "metrics.json"
    all_results: list[dict[str, Any]] = []
    if internal_path.exists():
        all_results.append(json.loads(internal_path.read_text(encoding="utf-8")))
    all_results.extend(results)

    summary_path = append_summary(
        all_results,
        output_dir,
        parameter_count=int(model.count_params()),
        model_size_mb=float(model_path.stat().st_size / (1024 * 1024)),
    )
    LOGGER.info("Đã cập nhật bảng tổng hợp: %s", summary_path)
    return results


def predict_one(config: dict[str, Any], image_path: Path) -> None:
    """Dự đoán một ảnh khi cần kiểm tra nhanh."""
    output_dir = output_dir_from(config)
    model_path = output_dir / "models" / "best_model.keras"
    if not image_path.exists():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")

    frame = pd.DataFrame(
        [
            {
                "image_path": str(image_path.resolve()),
                "file_name": image_path.name,
                "original_class": "unknown",
                "label": 0,
                "binary_label": "UNKNOWN",
            }
        ]
    )
    dataset = build_tf_dataset(
        frame,
        image_size=int(config["data"]["image_size"]),
        batch_size=1,
        training=False,
        seed=int(config["project"]["seed"]),
        cache_dataset=False,
    )
    model = load_keras_model(model_path)
    probabilities, latency = predict_frame(model, dataset, frame)
    probability_bad = float(probabilities[0])
    threshold = read_threshold(output_dir)
    predicted = "BAD" if probability_bad >= threshold else "GOOD"

    print(f"Probability GOOD : {1.0 - probability_bad:.6f}")
    print(f"Probability BAD  : {probability_bad:.6f}")
    print(f"Predicted class  : {predicted}")
    print(f"Threshold used   : {threshold:.6f}")
    print(f"Inference time   : {latency:.3f} ms")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MobileNetV3Small phân loại cam GOOD/BAD."
    )
    parser.add_argument("--config", default="config.yaml", help="Đường dẫn config.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Quét và chia dataset gốc.")
    subparsers.add_parser("train", help="Train và đánh giá internal test.")
    subparsers.add_parser(
        "evaluate-all", help="Đánh giá tất cả external dataset trong config."
    )
    subparsers.add_parser(
        "run", help="Prepare + train + internal test + tất cả external dataset."
    )

    predict_parser = subparsers.add_parser("predict", help="Dự đoán một ảnh.")
    predict_parser.add_argument("--image", required=True, help="Đường dẫn ảnh cần dự đoán")
    return parser


def main() -> int:
    configure_logging()
    args = build_parser().parse_args()
    try:
        config = load_config(Path(args.config).expanduser().resolve())
        set_global_seed(int(config["project"]["seed"]))

        if args.command == "prepare":
            prepare_data(config)
        elif args.command == "train":
            train_and_evaluate(config)
        elif args.command == "evaluate-all":
            evaluate_external_datasets(config)
        elif args.command == "run":
            prepare_data(config)
            train_and_evaluate(config)
            evaluate_external_datasets(config)
        elif args.command == "predict":
            predict_one(config, Path(args.image).expanduser().resolve())
        else:
            raise ValueError(f"Lệnh không hỗ trợ: {args.command}")

        LOGGER.info("HOÀN TẤT. Kết quả nằm tại: %s", output_dir_from(config))
        return 0
    except KeyboardInterrupt:
        LOGGER.error("Đã dừng theo yêu cầu người dùng.")
        return 130
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("LỖI: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
