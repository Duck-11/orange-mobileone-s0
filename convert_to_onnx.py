"""Train checkpoint (.pt) -> reparameterize (deploy) -> model.onnx FP32 + kiểm tra parity.
Thay convert_to_tflite.py. Chạy từ thư mục gốc project:
  python convert_to_onnx.py --checkpoint training/outputs/models/best_model.pt
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from pathlib import Path
import numpy as np, torch

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR / "training"))
from orange.model import DeployPBad, load_checkpoint, reparameterize, IMAGE_SIZE  # noqa

PARITY_TOL = 1e-4   # công bố TRƯỚC khi đo


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--threshold-json", type=Path, default=APP_DIR / "training/outputs/selected_threshold.json")
    a = ap.parse_args()

    train_m = load_checkpoint(a.checkpoint)
    deploy_m = reparameterize(train_m)
    torch.manual_seed(0)
    x = torch.rand(8, 3, IMAGE_SIZE, IMAGE_SIZE) * 255.0
    with torch.no_grad():
        d_tr = (train_m.p_bad(x) - deploy_m.p_bad(x)).abs().max().item()
    print(f"[parity] train vs deploy (PyTorch FP32) max|dp_BAD| = {d_tr:.2e}  (tol {PARITY_TOL})")
    if d_tr > PARITY_TOL:
        sys.exit("FAIL: reparameterize parity vượt dung sai — không export.")

    onnx_path = APP_DIR / "model.onnx"
    torch.onnx.export(DeployPBad(deploy_m).eval(), x[:1], str(onnx_path), opset_version=17,
                      input_names=["image_rgb_0_255_nchw"], output_names=["p_bad"],
                      dynamic_axes={"image_rgb_0_255_nchw": {0: "batch"}, "p_bad": {0: "batch"}})
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image_rgb_0_255_nchw": x.numpy()})[0].reshape(-1)
    with torch.no_grad():
        ref = deploy_m.p_bad(x).numpy()
    d_onnx = float(np.abs(out - ref).max())
    print(f"[parity] ONNX vs PyTorch deploy max|dp_BAD| = {d_onnx:.2e}  (tol {PARITY_TOL})")
    if d_onnx > PARITY_TOL:
        sys.exit("FAIL: ONNX parity vượt dung sai.")

    meta = {"model_id": "MobileOne-S0", "timm_name": train_m.timm_name, "dtype": "float32",
            "input": "RGB NCHW float32 0..255 (mean/std nằm trong graph)", "norm_cfg": train_m.norm_cfg,
            "labels": {"GOOD": 0, "BAD": 1}, "checkpoint_train": str(a.checkpoint),
            "checkpoint_train_sha256": sha(a.checkpoint), "onnx_sha256": sha(onnx_path),
            "parity_train_vs_deploy": d_tr, "parity_onnx_vs_deploy": d_onnx, "parity_tol": PARITY_TOL}
    (APP_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    if a.threshold_json.exists():
        thr = float(json.loads(a.threshold_json.read_text(encoding="utf-8"))["final_threshold"])
        (APP_DIR / "threshold.json").write_text(json.dumps({"threshold": thr}, indent=2), encoding="utf-8")
        print(f"[threshold] {thr}")
    print("OK ->", onnx_path)


if __name__ == "__main__":
    main()
