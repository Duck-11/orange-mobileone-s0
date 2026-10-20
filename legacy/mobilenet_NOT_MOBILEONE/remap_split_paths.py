"""Đổi tiền tố đường dẫn ảnh trong outputs/splits/*.csv (giữ nguyên membership train/val/test).

Dùng (từ thư mục training/):
  python remap_split_paths.py --old "E:/dataset" --new "D:/NCKH/dataset"
Chỉ GHI khi mọi ảnh sau khi đổi đều tồn tại. Backup: *.csv.bak_remap
"""
import argparse, shutil, sys
from pathlib import Path
import pandas as pd


def remap(p: str, old: str, new: str) -> str:
    pn = str(p).replace("\\", "/")
    o = old.replace("\\", "/").rstrip("/")
    if pn.lower().startswith(o.lower() + "/"):
        pn = new.replace("\\", "/").rstrip("/") + pn[len(o):]
    return str(Path(pn))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--splits", default="outputs/splits")
    a = ap.parse_args()
    d = Path(a.splits)
    names = ["train", "validation", "test", "dataset_manifest", "invalid_images"]
    frames, missing = {}, []
    for n in names:
        f = d / f"{n}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if "image_path" in df.columns and len(df):
            df["image_path"] = df["image_path"].map(lambda p: remap(p, a.old, a.new))
            if n != "invalid_images":
                missing += [p for p in df["image_path"] if not Path(p).exists()]
        frames[n] = (f, df)
    print({n: len(df) for n, (_, df) in frames.items()})
    if missing:
        print(f"DỪNG: {len(missing)} ảnh không tồn tại sau khi đổi. Ví dụ:")
        for p in missing[:5]:
            print("  ", p)
        print("Kiểm tra lại --new (đúng thư mục chứa fresh_orange, rotten_orange, ...).")
        return 1
    for n, (f, df) in frames.items():
        shutil.copy(f, f.with_name(f.name + ".bak_remap"))
        df.to_csv(f, index=False, encoding="utf-8-sig")
    print("OK: đã đổi đường dẫn, mọi ảnh trong train/validation/test đều tồn tại.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
