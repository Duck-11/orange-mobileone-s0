"""Lưu checkpoint_deploy (đã reparameterize) RIÊNG với checkpoint_train, kèm parity trên batch ảnh.
  python tools/deploy_checkpoint.py --checkpoint training\\outputs\\models\\best_model.pt
  python tools/deploy_checkpoint.py --checkpoint ... --images "D:\\NCKH\\test_moi"   (ảnh thật; không có thì dùng ảnh giả seed cố định)
Dung sai công bố TRƯỚC khi đo: TOL = 1e-4 trên |p_BAD(train) - p_BAD(deploy)|. Vượt dung sai -> không lưu."""
import argparse, datetime, hashlib, json, pathlib, sys
import numpy as np
import torch

TOL = 1e-4          # trên |p_BAD(train) - p_BAD(deploy)|
LOGIT_TOL = 1e-3    # trên |logit(train) - logit(deploy)| (không bị sigmoid bão hoà che mất sai lệch)
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from orange.model import OrangeMobileOneS0, load_checkpoint, reparameterize, IMAGE_SIZE  # noqa
from orange.data import read_rgb  # noqa


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def load_deploy_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    m = reparameterize(OrangeMobileOneS0(pretrained=False, timm_name=ck["timm_name"]))
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def batch(images_dir, n=32):
    """Lấy ảnh xen kẽ giữa các thư mục con (để có cả GOOD lẫn BAD), không lấy 32 ảnh đầu của một lớp."""
    if images_dir:
        by = {}
        for f in sorted(pathlib.Path(images_dir).rglob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                by.setdefault(f.parent.name, []).append(f)
        files = []
        for i in range(max((len(v) for v in by.values()), default=0)):
            for k in sorted(by):
                if i < len(by[k]) and len(files) < n:
                    files.append(by[k][i])
        if files:
            xs = [torch.from_numpy(read_rgb(str(f), IMAGE_SIZE).astype(np.float32).transpose(2, 0, 1)) for f in files]
            return torch.stack(xs), [f"{f.parent.name}/{f.name}" for f in files]
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 3, IMAGE_SIZE, IMAGE_SIZE, generator=g) * 255.0, ["synthetic_seed0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, type=pathlib.Path)
    ap.add_argument("--images", default=None)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    a = ap.parse_args()
    out = a.out or a.checkpoint.with_name("checkpoint_deploy.pt")

    train_m = load_checkpoint(a.checkpoint)
    deploy_m = reparameterize(train_m)
    x, names = batch(a.images)
    with torch.no_grad():
        p_t, p_d = train_m.p_bad(x), deploy_m.p_bad(x)
        l_t, l_d = train_m(x), deploy_m(x)
    diff = (p_t - p_d).abs().max().item()
    logit_diff = (l_t - l_d).abs().max().item()
    saturated = float(((p_t < 1e-6) | (p_t > 1 - 1e-6)).float().mean())
    real = a.images and names[0] != "synthetic_seed0"
    print(f"[parity] {len(x)} ảnh ({'thật' if real else 'giả'}): max|dp_BAD| = {diff:.2e} (tol {TOL}); "
          f"max|dlogit| = {logit_diff:.2e} (tol {LOGIT_TOL}); p_BAD bão hoà (<1e-6 hoặc >1-1e-6): {saturated:.0%}")
    if saturated > 0.5:
        print("[cảnh báo] >50% ảnh có p_BAD bão hoà: parity trên p_BAD kém ý nghĩa, xem parity trên logit.")
    if diff > TOL or logit_diff > LOGIT_TOL:
        sys.exit("FAIL: vượt dung sai — không lưu checkpoint_deploy.")

    torch.save({"state_dict": deploy_m.state_dict(), "timm_name": train_m.timm_name, "reparameterized": True,
                "norm_cfg": train_m.norm_cfg, "labels": {"GOOD": 0, "BAD": 1}}, out)
    again = load_deploy_checkpoint(out)
    with torch.no_grad():
        reload_diff = (deploy_m.p_bad(x) - again.p_bad(x)).abs().max().item()
    meta = {"checkpoint_train": str(a.checkpoint), "checkpoint_train_sha256": sha(a.checkpoint),
            "checkpoint_deploy": str(out), "checkpoint_deploy_sha256": sha(out),
            "reparameterized_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "parity_tolerance": TOL, "logit_tolerance": LOGIT_TOL, "parity_train_vs_deploy_max_abs": diff, "parity_logit_max_abs": logit_diff, "p_bad_saturated_fraction": saturated, "parity_deploy_reload_max_abs": reload_diff,
            "parity_batch_size": len(x), "parity_images": names[:50]}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()