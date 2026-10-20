"""Smoke test pipeline train PyTorch bằng dữ liệu giả (không cần dataset thật, không cần mạng).
Chạy từ thư mục training/:  pytest ../tests/test_pipeline.py -v   (hoặc copy tests vào cạnh)"""
import sys, pathlib
import numpy as np
import pytest
cv2 = pytest.importorskip("cv2"); pd = pytest.importorskip("pandas")
torch = pytest.importorskip("torch"); pytest.importorskip("timm")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from orange import data as D, model as M, trainer as T, evaluation as E  # noqa


@pytest.fixture()
def toy(tmp_path):
    rng = np.random.default_rng(0)
    mapping = {"fresh_orange": "GOOD", "rotten_orange": "BAD"}
    for folder, base in (("fresh_orange", 200), ("rotten_orange", 60)):
        (tmp_path / folder).mkdir()
        for i in range(12):
            img = np.clip(rng.normal(base, 20, (80, 80, 3)), 0, 255).astype(np.uint8)
            cv2.imwrite(str(tmp_path / folder / f"{i}.png"), img)
    frame, invalid = D.scan_labeled_folder(tmp_path, mapping, [".png"])
    assert invalid.empty
    return frame


def test_read_and_augment_shapes(toy):
    rgb = D.read_rgb(toy.image_path[0], 224)
    assert rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8
    aug = D.augment(rgb, np.random.default_rng(1))
    assert aug.shape == (224, 224, 3) and 0 <= aug.min() and aug.max() <= 255


def test_loader_order_and_range(toy):
    ld = D.build_loader(toy, 224, 8, training=False, seed=0)
    x, y = next(iter(ld))
    assert x.shape[1:] == (3, 224, 224) and x.max() > 1.0   # 0..255, chưa chia 255
    assert y.tolist() == toy.label.astype(float).tolist()[:len(y)]


def test_two_stage_train_smoke(toy, tmp_path):
    m = M.OrangeMobileOneS0(pretrained=False)
    ld = D.build_loader(toy, 224, 8, training=True, seed=0)
    vl = D.build_loader(toy, 224, 8, training=False, seed=0)
    cfg = dict(head_epochs=1, fine_tune_epochs=1, head_learning_rate=1e-3, fine_tune_learning_rate=1e-5,
               fine_tune_last_stages=2, early_stopping_patience=5, reduce_lr_patience=2, reduce_lr_factor=0.5)
    best, hist = T.train_two_stage(m, ld, vl, D.calculate_class_weight(toy), cfg, tmp_path, torch.device("cpu"))
    assert best.exists() and len(hist["val_loss"]) == 2
    # BN phải còn frozen sau fine-tune
    assert all(not p.requires_grad for mod in m.modules()
               if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm) for p in mod.parameters())
    m2 = M.load_checkpoint(best)
    probs, _ = E.predict_frame(m2, vl, toy)
    assert probs.shape == (len(toy),) and ((probs >= 0) & (probs <= 1)).all()
