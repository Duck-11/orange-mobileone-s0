"""pytest tests/ -v  (cần torch + timm; pretrained=False nên không cần mạng, trừ test nguồn)."""
import json, pathlib, sys
import pytest
torch = pytest.importorskip("torch"); pytest.importorskip("timm")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from orange import model as M  # noqa


def _randomize_bn(model):
    g = torch.Generator().manual_seed(0)
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


def test_is_mobileone_s0_not_mobilenet():
    m = M.OrangeMobileOneS0(pretrained=False)
    assert m.backbone.pretrained_cfg["architecture"] == "mobileone_s0"
    names = " ".join(type(x).__name__ for x in m.modules()).lower()
    assert "mobilenet" not in names
    src = (ROOT / "training/orange/model.py").read_text(encoding="utf-8")
    assert "tensorflow" not in src and "applications" not in src
    assert m.backbone.num_features == 1024
    n = sum(p.numel() for p in m.backbone.parameters())
    assert n == 4_268_272, n  # S0 ≈5.3M gồm classifier 1000 lớp; siết biên sau khi in số thật


def test_label_mapping_and_range():
    assert M.LABELS == {"GOOD": 0, "BAD": 1}
    p = M.OrangeMobileOneS0(pretrained=False).eval().p_bad(torch.rand(2, 3, 224, 224) * 255)
    assert p.shape == (2,) and ((p >= 0) & (p <= 1)).all()


def test_train_vs_deploy_parity():
    torch.manual_seed(1)
    m = M.OrangeMobileOneS0(pretrained=False).eval()
    _randomize_bn(m)
    d = M.reparameterize(m)
    assert not any(isinstance(x, torch.nn.modules.batchnorm._BatchNorm) for x in d.backbone.modules())
    x = torch.rand(4, 3, 224, 224) * 255
    with torch.no_grad():
        diff = (m.p_bad(x) - d.p_bad(x)).abs().max().item()
    assert diff <= 1e-4, diff


def test_preprocessing_matches_pretrained_cfg():
    m = M.OrangeMobileOneS0(pretrained=False).eval()
    cfg = m.backbone.pretrained_cfg
    x = torch.rand(1, 3, 224, 224) * 255
    ref = (x / 255.0 - torch.tensor(cfg["mean"]).view(1, 3, 1, 1)) / torch.tensor(cfg["std"]).view(1, 3, 1, 1)
    got = (x - m.pix_mean) / m.pix_std
    assert torch.allclose(ref, got, atol=1e-5)


def test_source_proof_file():  # MOBILEONE_SOURCE_IDENTITY_PASS — chạy tools/prove_source.py trước
    f = ROOT / "training/orange/mobileone_source.json"
    assert f.exists(), "Chạy: python tools/prove_source.py"
    d = json.loads(f.read_text(encoding="utf-8"))
    assert d["variant"] == "S0" and len(d.get("weights_sha256", "")) == 64
