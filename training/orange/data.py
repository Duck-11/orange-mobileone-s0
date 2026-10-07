"""Chuẩn bị dữ liệu và tạo tf.data pipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)
LABEL_TO_INT = {"GOOD": 0, "BAD": 1}
INT_TO_LABEL = {0: "GOOD", 1: "BAD"}


def normalize_label(value: Any) -> int:
    """Chuẩn hóa nhãn cấu hình về 0=GOOD, 1=BAD."""
    if isinstance(value, bool):
        raise ValueError("Nhãn không được là kiểu boolean.")
    if isinstance(value, (int, np.integer)):
        label = int(value)
        if label in (0, 1):
            return label
    text = str(value).strip().upper()
    if text in LABEL_TO_INT:
        return LABEL_TO_INT[text]
    raise ValueError(f"Nhãn không hợp lệ: {value!r}. Chỉ dùng GOOD/BAD hoặc 0/1.")


def set_global_seed(seed: int) -> None:
    """Đặt seed cho Python, NumPy và TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ModuleNotFoundError:
        LOGGER.warning("Chưa cài TensorFlow; chỉ đặt seed cho Python và NumPy.")


def resolve_dataset_root(path: str | Path, class_mapping: dict[str, Any]) -> Path:
    """Nhận thư mục lớp hoặc tự tìm thư mục test bên trong thư mục cha."""
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {root}")

    expected = {name.lower() for name in class_mapping}
    direct_dirs = {item.name.lower() for item in root.iterdir() if item.is_dir()}
    if expected.intersection(direct_dirs):
        return root

    test_dir = root / "test"
    if test_dir.is_dir():
        test_dirs = {item.name.lower() for item in test_dir.iterdir() if item.is_dir()}
        if expected.intersection(test_dirs):
            return test_dir.resolve()

    raise ValueError(
        f"Không tìm thấy thư mục lớp trong {root}. "
        f"Các lớp được cấu hình: {sorted(class_mapping)}"
    )


def _is_valid_image(path: Path) -> bool:
    """Kiểm tra nhanh ảnh có đọc được hay không."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def scan_labeled_folder(
    root: str | Path,
    class_mapping: dict[str, Any],
    allowed_extensions: list[str] | tuple[str, ...],
    validate_images: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Quét dataset dạng mỗi lớp là một thư mục.

    Returns:
        valid_df: image_path, file_name, original_class, label, binary_label.
        invalid_df: image_path, reason.
    """
    normalized_mapping = {
        str(folder).strip(): normalize_label(label)
        for folder, label in class_mapping.items()
    }
    dataset_root = resolve_dataset_root(root, normalized_mapping)
    extensions = {str(ext).lower() for ext in allowed_extensions}

    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, str]] = []

    candidates: list[tuple[Path, str, int]] = []
    for folder_name, label in normalized_mapping.items():
        class_dir = dataset_root / folder_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Thiếu thư mục lớp bắt buộc: {class_dir}")
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                candidates.append((path, folder_name, label))

    if not candidates:
        raise ValueError(f"Không tìm thấy ảnh .jpg/.jpeg/.png trong {dataset_root}")

    for path, folder_name, label in tqdm(candidates, desc="Đang quét ảnh", unit="ảnh"):
        if validate_images and not _is_valid_image(path):
            invalid_rows.append({"image_path": str(path.resolve()), "reason": "Ảnh không đọc được"})
            continue
        valid_rows.append(
            {
                "image_path": str(path.resolve()),
                "file_name": path.name,
                "original_class": folder_name,
                "label": int(label),
                "binary_label": INT_TO_LABEL[int(label)],
            }
        )

    valid_df = pd.DataFrame(valid_rows)
    invalid_df = pd.DataFrame(invalid_rows, columns=["image_path", "reason"])

    if valid_df.empty:
        raise ValueError("Không có ảnh hợp lệ sau khi kiểm tra.")
    if valid_df["label"].nunique() != 2:
        raise ValueError("Dataset phải có cả hai nhãn GOOD và BAD.")

    return valid_df, invalid_df


def split_dataset(
    frame: pd.DataFrame,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chia train/validation/test có phân tầng theo lớp gốc."""
    total = train_ratio + validation_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Tổng tỷ lệ phải bằng 1.0, hiện tại là {total}")
    if min(train_ratio, validation_ratio, test_ratio) <= 0:
        raise ValueError("Các tỷ lệ train/validation/test phải lớn hơn 0.")

    counts = frame["original_class"].value_counts()
    too_small = counts[counts < 3]
    if not too_small.empty:
        raise ValueError(
            "Mỗi lớp gốc cần ít nhất 3 ảnh để chia 3 tập. "
            f"Các lớp chưa đủ: {too_small.to_dict()}"
        )

    train_df, remaining_df = train_test_split(
        frame,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=frame["original_class"],
    )

    validation_share = validation_ratio / (validation_ratio + test_ratio)
    validation_df, test_df = train_test_split(
        remaining_df,
        train_size=validation_share,
        random_state=seed,
        shuffle=True,
        stratify=remaining_df["original_class"],
    )

    return (
        train_df.sort_values("image_path").reset_index(drop=True),
        validation_df.sort_values("image_path").reset_index(drop=True),
        test_df.sort_values("image_path").reset_index(drop=True),
    )


def save_splits(
    output_dir: Path,
    manifest: pd.DataFrame,
    invalid: pd.DataFrame,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Lưu manifest và ba tập CSV."""
    split_dir = output_dir / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)

    split_frames = {
        "train": train_df.copy(),
        "validation": validation_df.copy(),
        "test": test_df.copy(),
    }
    manifest_with_split = manifest.copy()
    manifest_with_split["split"] = ""
    for split_name, split_frame in split_frames.items():
        manifest_with_split.loc[
            manifest_with_split["image_path"].isin(split_frame["image_path"]), "split"
        ] = split_name
        split_frame.to_csv(split_dir / f"{split_name}.csv", index=False, encoding="utf-8-sig")

    manifest_with_split.to_csv(
        split_dir / "dataset_manifest.csv", index=False, encoding="utf-8-sig"
    )
    invalid.to_csv(split_dir / "invalid_images.csv", index=False, encoding="utf-8-sig")


def load_split_csvs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Đọc ba file split đã tạo."""
    split_dir = output_dir / "splits"
    paths = [split_dir / "train.csv", split_dir / "validation.csv", split_dir / "test.csv"]
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Chưa có dữ liệu đã chia. Hãy chạy `python main.py prepare` trước. "
            f"Thiếu: {missing}"
        )
    return tuple(pd.read_csv(path) for path in paths)  # type: ignore[return-value]


def calculate_class_weight(train_df: pd.DataFrame) -> dict[int, float]:
    """Tính class weight cân bằng từ tập train."""
    counts = train_df["label"].astype(int).value_counts().to_dict()
    if 0 not in counts or 1 not in counts:
        raise ValueError("Tập train phải có cả GOOD và BAD.")
    total = len(train_df)
    return {
        0: float(total / (2.0 * counts[0])),
        1: float(total / (2.0 * counts[1])),
    }


def build_tf_dataset(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
    cache_dataset: bool = False,
):
    """Tạo tf.data.Dataset đọc ảnh theo batch."""
    import tensorflow as tf

    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].astype(np.float32).to_numpy()

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = not training
    dataset = dataset.with_options(options)

    def decode(path, label):
        content = tf.io.read_file(path)
        image = tf.io.decode_image(content, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size], antialias=True)
        image = tf.cast(image, tf.float32)
        return image, tf.reshape(tf.cast(label, tf.float32), [1])

    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_dataset:
        dataset = dataset.cache()

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(frame), 4096), seed=seed, reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)

    if training:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal", seed=seed),
                tf.keras.layers.RandomRotation(0.04, fill_mode="reflect", seed=seed + 1),
                tf.keras.layers.RandomZoom(0.10, fill_mode="reflect", seed=seed + 2),
                tf.keras.layers.RandomTranslation(
                    0.05, 0.05, fill_mode="reflect", seed=seed + 3
                ),
                tf.keras.layers.RandomContrast(0.10, seed=seed + 4),
            ],
            name="train_augmentation",
        )
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.prefetch(tf.data.AUTOTUNE)
