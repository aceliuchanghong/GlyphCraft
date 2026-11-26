from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

import torch
import torch.nn as nn


class ModelEMA:
    """
    v3 向量化融合 EMA（torch._foreach_lerp_ 单次内核融合更新）：
    - 仅追踪 requires_grad=True 的可训练参数（跳过冻结 DINOv2，零冗余显存）；
    - 支持 sync_to(target_model) 零拷贝无克隆同步至 Phase 2 持久 BF16 教师副本。
    """

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup_steps: int = 1000,
    ):
        self.decay = float(decay)
        self.warmup_steps = max(0, int(warmup_steps))
        self.num_updates = 0
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().float().clone()

    def current_decay(self) -> float:
        if self.decay <= 0.0:
            return 0.0
        if self.warmup_steps <= 0:
            return self.decay
        std_decay = (1.0 + self.num_updates) / (10.0 + self.num_updates)
        if self.num_updates < self.warmup_steps:
            progress = float(self.num_updates) / float(self.warmup_steps)
            return min(self.decay * progress, std_decay)
        return min(self.decay, std_decay)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        d = self.current_decay()
        one_minus_d = 1.0 - d

        shadow_list: List[torch.Tensor] = []
        target_list: List[torch.Tensor] = []
        same_dtype = True
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            buf = self.shadow[name]
            if buf.device != param.device:
                buf = buf.to(device=param.device)
                self.shadow[name] = buf
            p_det = param.detach()
            if p_det.dtype != buf.dtype:
                same_dtype = False
                p_det = p_det.float()
            shadow_list.append(buf)
            target_list.append(p_det)

        if not shadow_list:
            return
        if same_dtype and hasattr(torch, "_foreach_lerp_"):
            torch._foreach_lerp_(shadow_list, target_list, one_minus_d)
        else:
            for buf, tgt in zip(shadow_list, target_list):
                buf.mul_(d).add_(tgt, alpha=one_minus_d)

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": {k: v.cpu() for k, v in self.shadow.items()},
        }

    def load_state_dict(
        self, state: Dict[str, object], device: Optional[torch.device] = None
    ) -> None:
        self.decay = float(state.get("decay", self.decay))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.num_updates = int(state.get("num_updates", 0))
        loaded_shadow = state.get("shadow", {})
        if isinstance(loaded_shadow, dict):
            for k, v in loaded_shadow.items():
                if k in self.shadow:
                    dst_dev = device if device is not None else self.shadow[k].device
                    self.shadow[k] = v.detach().float().to(dst_dev)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        """零克隆批量将 EMA 权重写入目标模型（兼容 BF16 持久教师同步）。"""
        dst_list: List[torch.Tensor] = []
        src_list: List[torch.Tensor] = []
        same_dtype = True
        for name, param in model.named_parameters():
            if name not in self.shadow:
                continue
            src = self.shadow[name]
            if src.device != param.device:
                src = src.to(device=param.device)
                self.shadow[name] = src
            if src.dtype != param.dtype:
                same_dtype = False
                param.data.copy_(src.to(dtype=param.dtype))
            else:
                dst_list.append(param.data)
                src_list.append(src)
        if dst_list and same_dtype and hasattr(torch, "_foreach_copy_"):
            torch._foreach_copy_(dst_list, src_list)
        elif dst_list:
            for dst, src in zip(dst_list, src_list):
                dst.copy_(src)

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """临时把 EMA 权重挂到模型上做验证推理，退出后无损恢复原始权重。"""
        backup: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    backup[name] = param.data.detach().clone()
                    param.data.copy_(
                        self.shadow[name].to(device=param.device, dtype=param.dtype)
                    )
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in backup:
                        param.data.copy_(backup[name])
