"""Liệt kê ảnh bị đoán sai trong một thư mục chia theo lớp (cạnh app.py/model.onnx).
  python list_errors.py --root "D:\\NCKH\\test_moi"
  python list_errors.py --root "D:\\NCKH\\dataset_3" --map "fresh=0,rotten=1"   (thư mục tên lạ)
  python list_errors.py --root ... --threshold 0.5
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
from app import OrangeClassifier, load_threshold
from eval_folder import DEFAULT_MAP, IMAGE_EXTS, parse_extra_map, read_image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--map", type=str, default=None, help='Ví dụ "fresh=0,rotten=1" (0=GOOD, 1=BAD)')
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()

    mapping = dict(DEFAULT_MAP)
    mapping.update(parse_extra_map(a.map))
    thr = a.threshold if a.threshold is not None else load_threshold()

    items = []
    for sub in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
        key = sub.name.lower()
        if key not in mapping:
            print(f"[Bỏ qua] Thư mục '{sub.name}' không có trong ánh xạ. Dùng --map \"{sub.name}=0\" hoặc =1.")
            continue
        items += [(f, mapping[key], sub.name) for f in sorted(sub.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
    if not items:
        print("[Lỗi] Không có ảnh nào được đánh giá. Kiểm tra --root và --map.")
        return 1

    clf = OrangeClassifier()
    checked = errors = 0
    for i in range(0, len(items), a.batch_size):
        chunk, imgs, meta = items[i:i + a.batch_size], [], []
        for f, truth, folder in chunk:
            img = read_image(f)
            if img is not None:
                imgs.append(img); meta.append((f, truth, folder))
        for (f, truth, folder), p in zip(meta, clf.predict_probability_bad(imgs)):
            checked += 1
            pred = int(float(p) >= thr)
            if pred != truth:
                errors += 1
                print(f"SAI | thật={'BAD' if truth else 'GOOD'} dự đoán={'BAD' if pred else 'GOOD'} "
                      f"| p_BAD={float(p):.3f} | {folder}\\{f.name}")
    print(f"\nThreshold {thr:.2f}: {errors} ảnh sai / {checked} ảnh đã đánh giá.")
    return 0


if __name__ == "__main__":
    sys.exit(main())