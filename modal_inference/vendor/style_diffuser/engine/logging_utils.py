from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import re
import time
from typing import Any, Dict, Optional

import torch


def format_train_progress(
    step: int,
    total: int,
    loss: float,
    started_at: float,
    extras: Optional[Dict[str, float]] = None,
) -> str:
    now = time.time()
    elapsed = max(now - started_at, 1e-6)
    rate = elapsed / max(step, 1)
    remaining = rate * max(total - step, 0)
    eta = time.strftime("%m-%d %H:%M", time.localtime(now + remaining))
    remain_h = int(remaining // 3600)
    remain_m = int((remaining % 3600) // 60)
    remain_s = int(remaining % 60)
    pct = 100.0 * step / total if total else 0.0
    clock = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    extra_str = ""
    if extras:
        extra_str = " " + " ".join(f"{k}={v:.4f}" for k, v in extras.items())
    return (
        f"[{clock}] {step}/{total} ({pct:.1f}%) "
        f"loss={loss:.4f}{extra_str} {rate:.2f}s/step "
        f"剩余 {remain_h:02d}:{remain_m:02d}:{remain_s:02d} 预计 {eta}"
    )


def log_train_progress(message: str) -> None:
    logging.info(message)
    for handler in logging.root.handlers:
        handler.flush()
    print(message, flush=True)


def resolve_mixed_precision(requested: str) -> str:
    """在 CPU 或不支持 BF16 的老卡上安全降级混合精度，避免启动崩溃。"""
    mp = str(requested or "no").lower()
    if mp == "no":
        return "no"
    if not torch.cuda.is_available():
        return "no"
    if mp == "bf16" and not torch.cuda.is_bf16_supported():
        return "fp16"
    return mp


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    wsd_decay_ratio: float = 0.15,
    min_lr_ratio: float = 0.05,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    现代学习率调度器：
    - wsd:      Warmup -> Stable 恒定平台 -> 尾部余弦衰减（随时可续训或早停收尾）
    - cosine:   Warmup -> 余弦退火至 min_lr_ratio
    - linear:   Warmup -> 线性衰减至 0
    - constant: Warmup -> 恒定学习率
    """
    warmup = max(0, int(num_warmup_steps))
    total = max(1, int(num_training_steps))
    decay_steps = max(1, int(round(total * float(wsd_decay_ratio))))
    stable_end = max(warmup, total - decay_steps)

    def lr_lambda(current_step: int) -> float:
        if warmup > 0 and current_step < warmup:
            return float(current_step + 1) / float(warmup)
        if scheduler_type == "constant":
            return 1.0
        if scheduler_type == "linear":
            progress = float(current_step - warmup) / float(max(1, total - warmup))
            return max(0.0, 1.0 - progress)
        if scheduler_type == "cosine":
            progress = min(
                1.0, float(current_step - warmup) / float(max(1, total - warmup))
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
        # wsd (Warmup-Stable-Decay)
        if current_step < stable_end:
            return 1.0
        progress = min(
            1.0, float(current_step - stable_end) / float(max(1, total - stable_end))
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def write_stage_done(stage_dir: str | Path, payload: Dict[str, Any]) -> Path:
    path = Path(stage_dir) / "stage_done.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = dict(payload)
    record["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    with path.open("w", encoding="utf-8") as fh:
        json.dump(record, fh, ensure_ascii=False, indent=2)
    return path


def read_stage_done(stage_dir: str | Path) -> Optional[Dict[str, Any]]:
    path = Path(stage_dir) / "stage_done.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def find_latest_step_dir(stage_dir: str | Path) -> Optional[Path]:
    root = Path(stage_dir)
    if not root.is_dir():
        return None
    pattern = re.compile(r"^global_step_(\d+)$")
    best_step = -1
    best_path: Optional[Path] = None
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = pattern.match(entry.name)
        if match:
            step = int(match.group(1))
            if step > best_step and (entry / "pixel_dit.pth").is_file():
                best_step = step
                best_path = entry
    return best_path


def find_latest_scr_ckpt(scr_dir: str | Path) -> Optional[Path]:
    root = Path(scr_dir)
    if not root.is_dir():
        return None
    pattern = re.compile(r"^scr_(\d+)\.pth$")
    best_step = -1
    best_path: Optional[Path] = None
    for entry in root.iterdir():
        if not entry.is_file():
            continue
        match = pattern.match(entry.name)
        if match:
            step = int(match.group(1))
            if step > best_step:
                best_step = step
                best_path = entry
    return best_path
