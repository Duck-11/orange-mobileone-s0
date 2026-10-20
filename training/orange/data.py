"""Chuẩn bị dữ liệu và tạo PyTorch DataLoader (GOOD=0, BAD=1)."""

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
    """Đặt seed cho Python, NumPy và PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ModuleNotFoundError:
        LOGGER.warning("Chưa cài PyTorch; chỉ đặt seed cho Python và NumPy.")


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


def read_rgb(path: str, image_size: int) -> np.ndarray:
    """Đọc ảnh -> RGB uint8 HxWx3, resize INTER_AREA.

    Dùng đúng cách resize/đổi màu của app.py (cv2.INTER_AREA, BGR->RGB) để train và
    deploy cùng một tiền xử lý. np.fromfile giúp đọc được đường dẫn có dấu trên Windows.
    """
    import cv2

    data = np.fromfile(path, dtype=np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise OSError(f"Không đọc được ảnh: {path}")
    bgr = cv2.resize(bgr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def augment(rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Augmentation tương đương bản Keras cũ: flip ngang, xoay ~±0.04*2π rad,
    zoom ±10%, tịnh tiến ±5%, contrast ±10% (viền reflect)."""
    import cv2

    size = rgb.shape[0]
    if rng.random() < 0.5:
        rgb = rgb[:, ::-1]
    angle = rng.uniform(-0.04, 0.04) * 360.0
    scale = rng.uniform(0.9, 1.1)
    tx, ty = rng.uniform(-0.05, 0.05, size=2) * size
    matrix = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), angle, scale)
    matrix[:, 2] += (tx, ty)
    out = cv2.warpAffine(
        np.ascontiguousarray(rgb), matrix, (size, size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    factor = rng.uniform(0.9, 1.1)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = np.clip((out - mean) * factor + mean, 0.0, 255.0)
    return out


try:  # module-level để DataLoader worker trên Windows (spawn) pickle được
    from torch.utils.data import Dataset as _TorchDataset
except ModuleNotFoundError:  # cho phép import khi chưa cài torch
    _TorchDataset = object


class OrangeDataset(_TorchDataset):
    """Dataset PyTorch: trả (tensor float32 CHW giá trị 0..255, nhãn float).
    Mean/std KHÔNG chuẩn hoá ở đây — đã nằm trong model (orange.model)."""

    def __init__(self, frame: pd.DataFrame, image_size: int, training: bool, seed: int):
        self.paths = frame["image_path"].astype(str).tolist()
        self.labels = frame["label"].astype(np.float32).to_numpy()
        self.image_size, self.training, self.seed = image_size, training, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        import torch

        rgb = read_rgb(self.paths[index], self.image_size)
        if self.training:
            rng = np.random.default_rng([self.seed, self.epoch, index])
            array = augment(rgb, rng)
        else:
            array = rgb.astype(np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
        return tensor, torch.tensor(self.labels[index], dtype=torch.float32)


def build_loader(
    frame: pd.DataFrame,
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
    num_workers: int = 0,
):
    """DataLoader. Eval: không shuffle -> thứ tự dự đoán khớp frame."""
    import torch
    from torch.utils.data import DataLoader

    dataset = OrangeDataset(frame, image_size, training, seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=training,
        generator=generator if training else None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    loader.orange_dataset = dataset
    return loader
