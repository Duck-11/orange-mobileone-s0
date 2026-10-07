from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

def convert(keras_model_path: Path, threshold_json_path: Path, quantize: bool) -> None:
    import tensorflow as tf

    if not keras_model_path.exists():
        raise FileNotFoundError(f"Không tìm thấy model đã train: {keras_model_path}")

    print(f"Đang load model: {keras_model_path}")

    model = tf.keras.models.load_model(keras_model_path,compile=False)

    # BƯỚC 1: Export Keras -> SavedModel
    saved_model_dir = Path(tempfile.mkdtemp(prefix="orange_saved_model_"))

    print(f"Đang export SavedModel: {saved_model_dir}")

    model.export(saved_model_dir)

    # BƯỚC 2: SavedModel -> TFLite
    print("Đang convert SavedModel -> TFLite...")

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print(
            "Chế độ: dynamic-range quantization "
            "(nhẹ hơn, có thể lệch nhẹ kết quả)."
        )
    else:
        print("Chế độ: float32 ""(không lượng tử hóa).")

    tflite_model = converter.convert()

    # BƯỚC 3: Lưu model.tflite
    output_path = APP_DIR / "model.tflite"

    output_path.write_bytes(tflite_model)

    size_mb = output_path.stat().st_size / (1024 * 1024)

    print(
        f"Đã lưu: {output_path} "
        f"({size_mb:.2f} MB)"
    )

    # BƯỚC 4: Copy threshold
    if threshold_json_path.exists():

        payload = json.loads(
            threshold_json_path.read_text(
                encoding="utf-8"
            )
        )

        threshold = float(
            payload["final_threshold"]
        )

        threshold_path = APP_DIR / "threshold.json"

        threshold_path.write_text(
            json.dumps(
                {
                    "threshold": threshold
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"Đã lưu threshold.json "
            f"(threshold = {threshold:.4f})."
        )

    else:
        print(
            f"[Cảnh báo] Không tìm thấy "
            f"{threshold_json_path}"
        )

    # Dọn thư mục tạm
    shutil.rmtree(
        saved_model_dir,
        ignore_errors=True
    )

    print()
    print("=" * 60)
    print("CONVERT HOÀN TẤT")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description="Export model.tflite + threshold.json"
    )

    parser.add_argument(
        "--keras-model",
        type=str,
        default="training/outputs/models/best_model.keras",
        help="Đường dẫn model .keras đã train.",
    )

    parser.add_argument(
        "--threshold-json",
        type=str,
        default="training/outputs/selected_threshold.json",
        help="Đường dẫn selected_threshold.json.",
    )

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Bật dynamic-range quantization.",
    )

    return parser


if __name__ == "__main__":

    args = build_parser().parse_args()

    convert(
        keras_model_path=Path(
            args.keras_model
        ).expanduser().resolve(),

        threshold_json_path=Path(
            args.threshold_json
        ).expanduser().resolve(),

        quantize=args.quantize,
    )