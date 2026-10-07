"""Đánh giá model.onnx trên một bộ ảnh bất kỳ (đã chia sẵn theo thư mục lớp).

Đặt file này cạnh app.py và model.onnx (thư mục gốc project MobileOne-S0), rồi chạy:

    python eval_folder.py --root "D:\\NCKH\\test_moi"

Cấu trúc thư mục --root: mỗi thư mục con là một lớp, ví dụ
    test_moi/fresh_orange/*.jpg      -> GOOD
    test_moi/rotten_orange/*.jpg     -> BAD
hoặc đơn giản hơn: test_moi/good/*.jpg và test_moi/bad/*.jpg

Thư mục có tên lạ: dùng --map, ví dụ  --map "tot=0,hong=1"  (0 = GOOD, 1 = BAD)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from app import OrangeClassifier, load_threshold

# Tên thư mục -> nhãn nhị phân (0 = GOOD, 1 = BAD). Khớp với training/dataset.py
# sau khi đã đổi green_orange thành BAD. Nếu bạn để green là GOOD, đổi số ở đây.
DEFAULT_MAP = {
    "fresh_orange": 0,
    "green_orange": 1,
    "blackspotted_orange": 1,
    "cankered_orange": 1,
    "rotten_orange": 1,
    "good": 0,
    "bad": 1,
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_extra_map(text: str | None) -> dict[str, int]:
    result: dict[str, int] = {}
    if not text:
        return result
    for pair in text.split(","):
        name, value = pair.split("=")
        result[name.strip().lower()] = int(value)
    return result


def read_image(path: Path) -> np.ndarray | None:
    # np.fromfile + imdecode để đọc được đường dẫn có dấu / khoảng trắng trên Windows
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Thư mục chứa các thư mục lớp.")
    parser.add_argument("--threshold", type=float, default=None, help="Ghi đè threshold (mặc định lấy threshold.json).")
    parser.add_argument("--map", type=str, default=None, help='Thêm ánh xạ tên thư mục, ví dụ "tot=0,hong=1".')
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[Lỗi] Không thấy thư mục: {root}")
        return 1

    mapping = dict(DEFAULT_MAP)
    mapping.update(parse_extra_map(args.map))

    items: list[tuple[Path, int, str]] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        key = sub.name.lower()
        if key not in mapping:
            print(f"[Bỏ qua] Thư mục '{sub.name}' không có trong ánh xạ. Dùng --map \"{sub.name}=0\" hoặc =1.")
            continue
        for f in sorted(sub.iterdir()):
            if f.suffix.lower() in IMAGE_EXTS:
                items.append((f, mapping[key], sub.name))

    if not items:
        print("[Lỗi] Không tìm thấy ảnh nào để đánh giá.")
        return 1

    threshold = args.threshold if args.threshold is not None else load_threshold()
    classifier = OrangeClassifier()

    y_true: list[int] = []
    y_prob: list[float] = []
    folders: list[str] = []
    batch: list[np.ndarray] = []
    meta: list[tuple[int, str]] = []
    skipped = 0

    def flush() -> None:
        if not batch:
            return
        probs = classifier.predict_probability_bad(batch)
        for (label, folder), p in zip(meta, probs):
            y_true.append(label)
            y_prob.append(float(p))
            folders.append(folder)
        batch.clear()
        meta.clear()

    for path, label, folder in items:
        img = read_image(path)
        if img is None:
            skipped += 1
            continue
        batch.append(img)
        meta.append((label, folder))
        if len(batch) >= args.batch_size:
            flush()
    flush()

    yt = np.array(y_true)
    yp = (np.array(y_prob) >= threshold).astype(int)

    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    total = len(yt)
    acc = (tp + tn) / total
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    print("\n===== KẾT QUẢ =====")
    print(f"Số ảnh đánh giá : {total}  (không đọc được: {skipped})")
    print(f"Threshold       : {threshold:.2f}  (xác suất BAD >= threshold -> BAD)")
    print(f"Accuracy        : {acc:.4f}")
    print(f"BAD  precision  : {prec:.4f}   recall: {rec:.4f}   F1: {f1:.4f}")
    print("\nMa trận nhầm lẫn (hàng = thật, cột = dự đoán):")
    print("              pred GOOD   pred BAD")
    print(f"true GOOD     {tn:9d}   {fp:8d}")
    print(f"true BAD      {fn:9d}   {tp:8d}")

    print("\nĐộ chính xác theo từng thư mục:")
    folder_arr = np.array(folders)
    for name in sorted(set(folders)):
        mask = folder_arr == name
        correct = int((yt[mask] == yp[mask]).sum())
        print(f"  {name:22s} {correct:5d}/{int(mask.sum()):5d}  = {correct / int(mask.sum()):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
