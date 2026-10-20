"""Chọn lại threshold = ĐIỂM GIỮA vùng hòa điểm tốt nhất (dài nhất) trong threshold_analysis.csv.
Không cần train lại, không cần chạy model. Chạy từ thư mục gốc project:
  python recompute_threshold.py
Ghi: training/outputs/selected_threshold.json (backup .bak_threshold) và threshold.json (cạnh app.py).
"""
import json, shutil, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "training/outputs/reports/validation/threshold_analysis.csv"
SEL = ROOT / "training/outputs/selected_threshold.json"
APP = ROOT / "threshold.json"


def plateau_midpoint(df: pd.DataFrame):
    df = df.reset_index(drop=True)
    best = df.sort_values(["f1_bad", "recall_bad", "accuracy", "false_negative_bad"],
                          ascending=[False, False, False, True], kind="stable").iloc[0]
    tied = np.where(np.isclose(df["f1_bad"], best["f1_bad"]) & np.isclose(df["recall_bad"], best["recall_bad"])
                    & np.isclose(df["accuracy"], best["accuracy"])
                    & (df["false_negative_bad"] == best["false_negative_bad"]))[0]
    runs = np.split(tied, np.where(np.diff(tied) > 1)[0] + 1)
    run = max(runs, key=len)
    return float(df["threshold"].iloc[run[len(run) // 2]]), float(df["threshold"].iloc[run[0]]), float(df["threshold"].iloc[run[-1]])


def main() -> int:
    if not CSV.exists():
        print("Không thấy", CSV); return 1
    df = pd.read_csv(CSV)
    mid, lo, hi = plateau_midpoint(df)
    mid = round(mid, 2)
    print(f"Vùng hòa điểm dài nhất: {lo:.2f} -> {hi:.2f}; chọn điểm giữa = {mid:.2f}")
    data = json.loads(SEL.read_text(encoding="utf-8")) if SEL.exists() else {}
    shutil.copy(SEL, SEL.with_name(SEL.name + ".bak_threshold")) if SEL.exists() else None
    data.update({"selected_on": "validation (điểm giữa vùng hòa điểm)", "final_threshold": mid,
                 "plateau_range": [round(lo, 2), round(hi, 2)]})
    SEL.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    APP.write_text(json.dumps({"threshold": mid}, indent=2), encoding="utf-8")
    print("Đã ghi", SEL, "và", APP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
