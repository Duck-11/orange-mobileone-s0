"""Chứng minh nguồn MobileOne-S0 -> training/orange/mobileone_source.json (cần mạng + torch + timm).
python tools/prove_source.py"""
import hashlib, json, sys, pathlib
import timm, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from training.orange.model import TIMM_NAME, OrangeMobileOneS0

m = OrangeMobileOneS0(pretrained=True)
cfg = m.backbone.pretrained_cfg
info = {
    "variant": "S0", "timm_name": TIMM_NAME, "timm_version": timm.__version__,
    "torch_version": torch.__version__,
    "pretrained_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
    "backbone_parameter_count": sum(p.numel() for p in m.backbone.parameters()),
    "backbone_constructor": f"timm.create_model('{TIMM_NAME}', num_classes=0)",
    "license": "VERIFY_MANUALLY: LICENSE của timm + apple/ml-mobileone + model card HF",
}
try:
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(cfg["hf_hub_id"], "model.safetensors")
    info["weights_source"] = f"https://huggingface.co/{cfg['hf_hub_id']}"
    info["weights_file"] = p
    info["weights_sha256"] = hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
except Exception as e:  # noqa
    info["weights_hash_error"] = repr(e)
out = pathlib.Path("training/orange/mobileone_source.json")
out.write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
pathlib.Path("outputs").mkdir(exist_ok=True)
pathlib.Path("outputs/model_summary.txt").write_text(str(m) + f"\n\nbackbone_params={info['backbone_parameter_count']}\n", encoding="utf-8")
print(json.dumps(info, indent=2, ensure_ascii=False))
