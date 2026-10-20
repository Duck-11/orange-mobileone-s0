"""Parity đường train (data.read_rgb + PyTorch) vs đường app (OrangeClassifier + ONNX) trên CÙNG ảnh.
Bỏ qua nếu chưa có model.onnx / best_model.pt. Có thể đặt biến môi trường ORANGE_PARITY_DIR để dùng ảnh thật."""
import os, pathlib, sys
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2"); torch = pytest.importorskip("torch"); pytest.importorskip("onnxruntime")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "training"))
CKPT, ONNX = ROOT / "training/outputs/models/best_model.pt", ROOT / "model.onnx"
pytestmark = pytest.mark.skipif(not (CKPT.exists() and ONNX.exists()), reason="chưa có best_model.pt/model.onnx")
TOL = 1e-4


def _images(tmp_path):
    d = os.environ.get("ORANGE_PARITY_DIR")
    if d:
        by = {}
        for f in sorted(pathlib.Path(d).rglob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                by.setdefault(f.parent.name, []).append(f)
        fs = []
        for i in range(max((len(v) for v in by.values()), default=0)):
            for k in sorted(by):
                if i < len(by[k]) and len(fs) < 16:
                    fs.append(by[k][i])
        if fs:
            return fs
    rng = np.random.default_rng(0)
    out = []
    for i in range(8):
        p = tmp_path / f"s{i}.png"
        cv2.imwrite(str(p), rng.integers(0, 255, (120, 150, 3), dtype=np.uint8))
        out.append(p)
    return out


def test_train_path_equals_app_path(tmp_path):
    from orange.data import read_rgb
    from orange.model import load_checkpoint, reparameterize
    from app import OrangeClassifier
    files = _images(tmp_path)
    model = reparameterize(load_checkpoint(CKPT))
    x = torch.stack([torch.from_numpy(read_rgb(str(f), 224).astype(np.float32).transpose(2, 0, 1)) for f in files])
    train_model = load_checkpoint(CKPT)
    with torch.no_grad():
        ref = model.p_bad(x).numpy()
        logit_diff = (train_model(x) - model(x)).abs().max().item()
    assert logit_diff <= 1e-3, logit_diff   # train vs deploy trên logit
    bgr = [cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR) for f in files]
    got = OrangeClassifier().predict_probability_bad(bgr)
    assert np.abs(ref - got).max() <= TOL, np.abs(ref - got).max()