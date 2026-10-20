"""Train MobileOne-S0 theo 2 giai đoạn (head -> fine-tune), BatchNorm đóng băng,
best checkpoint theo val_loss TOÀN CỤC xuyên cả hai giai đoạn."""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from orange.model import OrangeMobileOneS0, freeze_batchnorm, save_checkpoint

LOGGER = logging.getLogger(__name__)


def pick_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def set_head_only(model: OrangeMobileOneS0) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in list(model.head.parameters()):
        p.requires_grad = True


def set_fine_tune(model: OrangeMobileOneS0, last_stages: int) -> None:
    set_head_only(model)
    stages = getattr(model.backbone, "stages", None)
    if stages is None:
        raise RuntimeError("backbone timm không có thuộc tính .stages — kiểm tra phiên bản timm.")
    chosen = list(stages.children())[-last_stages:]
    for stage in chosen:
        for p in stage.parameters():
            p.requires_grad = True
    freeze_batchnorm(model)  # BN luôn đóng băng (tham số + running stats)
    LOGGER.info("Fine-tune %d stage cuối (BN vẫn frozen).", len(chosen))


def _enter_train_mode(model: torch.nn.Module) -> None:
    model.train()
    freeze_batchnorm(model)  # model.train() bật lại BN -> phải ép eval mỗi epoch


def _safe_auc(y: np.ndarray, p: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


@torch.no_grad()
def evaluate_loader(model, loader, device) -> tuple[float, float, float]:
    """(val_loss không trọng số, accuracy@0.5, AUC)."""
    model.eval()
    losses, probs, ys = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item())
        probs.append(torch.sigmoid(logits).cpu().numpy())
        ys.append(y.cpu().numpy())
    p, t = np.concatenate(probs), np.concatenate(ys)
    return float(sum(losses) / len(t)), float(((p >= 0.5) == (t >= 0.5)).mean()), _safe_auc(t, p)


def _run_phase(name, model, train_loader, val_loader, class_weight, device, epochs, lr,
               patience, lr_patience, lr_factor, state, history, best_path, extra_meta):
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    LOGGER.info("=== GIAI ĐOẠN %s: train %d tham số | lr=%g | tối đa %d epoch ===", name.upper(), n_train, lr, epochs)
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=lr_factor, patience=lr_patience)
    w = torch.tensor([class_weight[0], class_weight[1]], dtype=torch.float32, device=device)
    phase_best, bad_epochs, ran, durations = float("inf"), 0, 0, []
    label = {"head": "HEAD", "fine_tune": "FINE-TUNE"}.get(name, name.upper())

    for ep in range(1, epochs + 1):
        state["epoch"] += 1
        ran += 1
        ds = getattr(train_loader, "orange_dataset", None)
        if ds is not None:
            ds.set_epoch(state["epoch"])
        _enter_train_mode(model)
        t0, tot, seen, probs, ys = time.perf_counter(), 0.0, 0, [], []
        bar = tqdm(train_loader, desc=f"{label} {ep}/{epochs} đang train", leave=False, ncols=90)
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            per = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            loss = (per * w[y.long()]).mean()   # class_weight như Keras
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            tot += loss.item() * len(y)
            seen += len(y)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            ys.append(y.cpu().numpy())
        p, t = np.concatenate(probs), np.concatenate(ys)
        tr_loss, tr_acc, tr_auc = tot / seen, float(((p >= 0.5) == (t >= 0.5)).mean()), _safe_auc(t, p)
        v_loss, v_acc, v_auc = evaluate_loader(model, val_loader, device)
        scheduler.step(v_loss)

        for k, v in [("loss", tr_loss), ("accuracy", tr_acc), ("auc", tr_auc), ("val_loss", v_loss),
                     ("val_accuracy", v_acc), ("val_auc", v_auc), ("lr", optimizer.param_groups[0]["lr"])]:
            history[k].append(v)
        history["phase"].append(name)
        durations.append(time.perf_counter() - t0)
        left = (epochs - ep) * (sum(durations) / len(durations)) / 60.0
        LOGGER.info(
            "[%s %2d/%d] Train: loss %.4f, acc %5.1f%% | Val: loss %.4f, acc %5.1f%%, auc %.4f | "
            "%.0fs/epoch | còn tối đa ~%.0f phút (giai đoạn này)",
            label, ep, epochs, tr_loss, tr_acc * 100, v_loss, v_acc * 100, v_auc,
            durations[-1], left)

        if v_loss < state["best_val_loss"]:     # best TOÀN CỤC xuyên 2 phase
            state["best_val_loss"] = v_loss
            save_checkpoint(model, best_path, {**extra_meta, "epoch": state["epoch"], "phase": name,
                                               "val_loss": v_loss})
            LOGGER.info("    * Val loss tốt nhất từ đầu đến giờ -> đã lưu best_model.pt (%.4f)", v_loss)
        if v_loss < phase_best - 1e-12:
            phase_best, bad_epochs = v_loss, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                LOGGER.info("[%s] Dừng sớm: val_loss không cải thiện %d epoch liên tiếp.", label, patience)
                break
    return ran


def train_two_stage(model: OrangeMobileOneS0, train_loader, validation_loader, class_weight,
                    training_config: dict[str, Any], output_dir: Path, device: torch.device,
                    extra_meta: dict[str, Any] | None = None):
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    best_path = models_dir / "best_model.pt"
    model.to(device)
    history: dict[str, Any] = {k: [] for k in
                               ["loss", "accuracy", "auc", "val_loss", "val_accuracy", "val_auc", "lr", "phase"]}
    state = {"epoch": 0, "best_val_loss": float("inf")}
    c = training_config
    common = dict(patience=int(c["early_stopping_patience"]), lr_patience=int(c["reduce_lr_patience"]),
                  lr_factor=float(c.get("reduce_lr_factor", 0.5)), state=state, history=history,
                  best_path=best_path, extra_meta=extra_meta or {})

    LOGGER.info("Cách đọc: Train = trên ảnh đang học; Val = trên ảnh chưa học (mới phản ánh chất lượng thật). "
                "loss càng thấp càng tốt; acc/auc càng gần 100%%/1.0 càng tốt.")
    set_head_only(model)
    freeze_batchnorm(model)
    n1 = _run_phase("head", model, train_loader, validation_loader, class_weight, device,
                    int(c["head_epochs"]), float(c["head_learning_rate"]), **common)
    history["fine_tune_start_epoch"] = n1

    ck = torch.load(best_path, map_location=device, weights_only=True)          # khôi phục best trước khi fine-tune
    model.load_state_dict(ck["state_dict"])
    set_fine_tune(model, int(c["fine_tune_last_stages"]))
    _run_phase("fine_tune", model, train_loader, validation_loader, class_weight, device,
               int(c["fine_tune_epochs"]), float(c["fine_tune_learning_rate"]), **common)

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_path, history