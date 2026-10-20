"""Sinh bằng chứng cấu trúc MobileOne-S0: class hierarchy, tham số & output shape theo stage, số nhánh.
Chạy từ thư mục gốc project:  python tools/provenance_report.py
Ghi: outputs/provenance/model_hierarchy.json và model_hierarchy.md  (chỉ số liệu đo được, không suy diễn)."""
import json, pathlib, sys
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training"))
from orange.model import OrangeMobileOneS0, TIMM_NAME  # noqa

m = OrangeMobileOneS0(pretrained=False).eval()
shapes, hooks = {}, []


def hook(name):
    def fn(_mod, _inp, out):
        shapes[name] = list(out.shape) if torch.is_tensor(out) else None
    return fn


def n_params(mod):
    return sum(p.numel() for p in mod.parameters())


rows = []
for name, child in m.backbone.named_children():
    hooks.append(child.register_forward_hook(hook(name)))
stages = getattr(m.backbone, "stages", None)
if stages is not None:
    for i, st in enumerate(stages.children()):
        hooks.append(st.register_forward_hook(hook(f"stages.{i}")))
with torch.no_grad():
    m(torch.zeros(1, 3, 224, 224))
for h in hooks:
    h.remove()

for name, child in m.backbone.named_children():
    rows.append({"module": name, "class": type(child).__name__, "params": n_params(child), "output_shape": shapes.get(name)})
if stages is not None:
    for i, st in enumerate(stages.children()):
        blocks = [b for b in st.modules() if type(b).__name__ == "MobileOneBlock"]
        k = getattr(blocks[0], "conv_kxk", None) if blocks else None
        rows.append({"module": f"stages.{i}", "class": type(st).__name__, "params": n_params(st),
                     "output_shape": shapes.get(f"stages.{i}"), "mobileone_blocks": len(blocks),
                     "conv_kxk_branches_first_block": len(k) if k is not None else None})

info = {
    "timm_name": TIMM_NAME,
    "backbone_class": type(m.backbone).__name__,
    "backbone_mro": [c.__name__ for c in type(m.backbone).__mro__],
    "wrapper_class": type(m).__name__,
    "num_features": m.backbone.num_features,
    "backbone_params": n_params(m.backbone),
    "total_params_with_head": n_params(m),
    "mobileone_block_count": sum(1 for b in m.modules() if type(b).__name__ == "MobileOneBlock"),
    "input_shape": [1, 3, 224, 224],
    "modules": rows,
}
out = ROOT / "outputs" / "provenance"
out.mkdir(parents=True, exist_ok=True)
(out / "model_hierarchy.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
md = ["# Cấu trúc MobileOne-S0 (đo bằng forward 1x3x224x224)", "",
      f"- timm: `{TIMM_NAME}`; backbone class: `{info['backbone_class']}`; MobileOneBlock: {info['mobileone_block_count']}",
      f"- tham số backbone {info['backbone_params']:,}; toàn model {info['total_params_with_head']:,}", "",
      "| module | class | params | output shape | blocks | nhánh kxk (block đầu) |", "|---|---|---|---|---|---|"]
for r in rows:
    md.append(f"| {r['module']} | {r['class']} | {r['params']:,} | {r['output_shape']} | "
              f"{r.get('mobileone_blocks', '')} | {r.get('conv_kxk_branches_first_block', '')} |")
(out / "model_hierarchy.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md))
