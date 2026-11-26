from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


class _OfflineLiteBackbone(nn.Module):
    """
    离线轻量占位视觉编码器（仅用于 vision_backbone='none' 的纯离线单元测试）。
    提供与 DINOv2 完全一致的 (cls_token, patch_tokens, multi_layer_states) 输出协议。
    """

    def __init__(self, hidden_size: int = 128, patch_size: int = 14, num_layers: int = 12):
        super().__init__()
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                )
                for _ in range(4)
            ]
        )
        gen = torch.Generator().manual_seed(2026)
        for p in self.parameters():
            nn.init.normal_(p, mean=0.0, std=0.02, generator=gen)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        feat = self.patch_embed(pixel_values)
        tokens = feat.flatten(2).transpose(1, 2)
        cls = tokens.mean(dim=1, keepdim=True)
        h = torch.cat([cls, tokens], dim=1)

        layer_outputs: List[torch.Tensor] = [h]
        for idx in range(self.num_layers):
            blk = self.blocks[idx % len(self.blocks)]
            h = h + 0.1 * blk(h)
            layer_outputs.append(h)

        final_cls = h[:, 0]
        final_patches = h[:, 1:]
        return final_cls, final_patches, layer_outputs


class FrozenVisionBackbone(nn.Module):
    """
    冻结视觉基础模型统一封装（默认 facebook/dinov2-small，v3 原生支持 BF16 计算与全量特征预缓存）。
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov2-small",
        input_size: int = 140,
    ):
        super().__init__()
        self.model_name = str(model_name).strip()
        self.input_size = int(input_size)
        self.is_offline_stub = self.model_name.lower() in ("none", "stub", "offline")

        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        if self.is_offline_stub:
            self.backbone = _OfflineLiteBackbone(hidden_size=128, patch_size=14, num_layers=12)
            self.hidden_size = 128
            self.patch_size = 14
            self.num_layers = 12
        else:
            from transformers import AutoModel

            logger.info("Loading frozen vision backbone: %s", self.model_name)
            try:
                self.backbone = AutoModel.from_pretrained(
                    self.model_name, local_files_only=True
                )
            except Exception:
                try:
                    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
                    self.backbone = AutoModel.from_pretrained(self.model_name)
                except Exception as exc:
                    hint = (
                        "若处于内网或国内网络环境，可设置环境变量 HF_ENDPOINT=https://hf-mirror.com "
                        "（并设置 HF_HUB_DISABLE_XET=1），或将 model.vision_backbone / "
                        "scr.vision_backbone 指向本地已下载模型目录。"
                    )
                    raise RuntimeError(
                        f"无法加载视觉基座模型 '{self.model_name}': {exc}\n提示: {hint}"
                    ) from exc
            cfg = self.backbone.config
            self.hidden_size = int(cfg.hidden_size)
            self.patch_size = int(getattr(cfg, "patch_size", 14))
            self.num_layers = int(getattr(cfg, "num_hidden_layers", 12))

        self.grid_size = self.input_size // self.patch_size
        self.num_patch_tokens = self.grid_size * self.grid_size
        self.backbone.requires_grad_(False)
        self.backbone.eval()

    def set_compute_dtype(self, dtype: torch.dtype) -> "FrozenVisionBackbone":
        """将冻结骨干切换至 BF16/FP16 以释放显存并成倍加速前向（保持 _mean/_std 为 FP32）。"""
        self.backbone.to(dtype=dtype)
        return self

    def train(self, mode: bool = True):
        super().train(False)
        self.backbone.eval()
        return self

    def preprocess(self, images_norm: torch.Tensor) -> torch.Tensor:
        x = (images_norm.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(
                x,
                size=(self.input_size, self.input_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp(0.0, 1.0)
        mean = self._mean.to(device=x.device, dtype=x.dtype)
        std = self._std.to(device=x.device, dtype=x.dtype)
        return (x - mean) / std

    def _run_backbone(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        param = next(self.backbone.parameters())
        px = pixel_values.to(device=param.device, dtype=param.dtype)
        if self.is_offline_stub:
            cls, patches, layers = self.backbone(px)
            return cls, patches, tuple(layers) if output_hidden_states else None

        out = self.backbone(
            pixel_values=px,
            output_hidden_states=output_hidden_states,
        )
        seq = out.last_hidden_state
        cls = seq[:, 0]
        patches = seq[:, -self.num_patch_tokens :]
        hidden_states = out.hidden_states if output_hidden_states else None
        return cls, patches, hidden_states

    def extract_style_features(
        self,
        images_norm: torch.Tensor,
        no_grad: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ctx = torch.no_grad() if no_grad else torch.enable_grad()
        with ctx:
            px = self.preprocess(images_norm)
            cls, patches, _ = self._run_backbone(px, output_hidden_states=False)
        return cls, patches

    def extract_repa_targets(self, target_images_norm: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            px = self.preprocess(target_images_norm)
            _, patches, _ = self._run_backbone(px, output_hidden_states=False)
            patches = F.normalize(patches.float(), dim=-1)
            b, _, d = patches.shape
            return patches.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)

    def _pool_multilayer(
        self,
        hidden_states: Sequence[torch.Tensor],
        layers: Sequence[int],
    ) -> List[torch.Tensor]:
        max_idx = len(hidden_states) - 1
        feats: List[torch.Tensor] = []
        for layer_idx in layers:
            clamped = min(max(int(layer_idx), 1), max_idx)
            seq = hidden_states[clamped].float()
            cls = seq[:, 0]
            patches = seq[:, -self.num_patch_tokens :]
            mean_p = patches.mean(dim=1)
            std_p = patches.var(dim=1, unbiased=False).clamp_min(1e-6).sqrt()
            feats.append(torch.cat([cls, mean_p, std_p], dim=-1))
        return feats

    def extract_multilayer_features(
        self,
        images_norm: torch.Tensor,
        layers: Sequence[int] = (3, 6, 9, 12),
    ) -> List[torch.Tensor]:
        px = self.preprocess(images_norm)
        _, _, hidden_states = self._run_backbone(px, output_hidden_states=True)
        assert hidden_states is not None
        return self._pool_multilayer(hidden_states, layers=layers)

    @torch.no_grad()
    def extract_all_features(
        self,
        images_norm: torch.Tensor,
        layers: Optional[Sequence[int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        """单次前向同时提取 (cls, patches, multilayer_pooled)，用于启动期一次性预缓存。"""
        need_multi = layers is not None and len(layers) > 0
        px = self.preprocess(images_norm)
        cls, patches, hidden_states = self._run_backbone(
            px, output_hidden_states=need_multi
        )
        multi = (
            self._pool_multilayer(hidden_states, layers=layers)
            if need_multi and hidden_states is not None and layers is not None
            else None
        )
        return cls, patches, multi


class VisionFeatureCache:
    """
    v3 冻结视觉基座全量预缓存器（CPU FP16 紧凑存储，按索引向量化搬运至 GPU）：
    - 在 Stage 0 / Phase 1 / Phase 2 启动时用大 batch(128) 一次性跑完静态数据集原图的 DINOv2 表征；
    - 彻底消除训练期间每步数百次的重复冻结 DINOv2 前向，Stage 0 提速 10x+、Phase 2 提速 3x+。
    """

    def __init__(
        self,
        path_to_idx: Dict[str, int],
        cls_feats: Optional[torch.Tensor] = None,
        patch_feats: Optional[torch.Tensor] = None,
        multilayer_feats: Optional[torch.Tensor] = None,
        grid_size: int = 10,
    ):
        self.path_to_idx = path_to_idx
        self.cls_feats = cls_feats  # [N, D] fp16
        self.patch_feats = patch_feats  # [N, G*G, D] fp16
        self.multilayer_feats = multilayer_feats  # [N, num_layers, 3*D] fp16
        self.grid_size = int(grid_size)

    @classmethod
    @torch.no_grad()
    def build(
        cls,
        backbone: FrozenVisionBackbone,
        paths: Sequence[Path | str],
        image_loader,
        device: torch.device,
        need_style_patches: bool = True,
        multilayer_layers: Optional[Sequence[int]] = None,
        batch_size: int = 128,
    ) -> "VisionFeatureCache":
        unique_strs: List[str] = []
        path_to_idx: Dict[str, int] = {}
        for p in paths:
            ps = str(p)
            if ps not in path_to_idx:
                path_to_idx[ps] = len(unique_strs)
                unique_strs.append(ps)

        n_total = len(unique_strs)
        if n_total == 0:
            return cls(path_to_idx={}, grid_size=backbone.grid_size)

        backbone.eval()
        cls_list: List[torch.Tensor] = []
        patch_list: List[torch.Tensor] = []
        multi_list: List[torch.Tensor] = []

        for start in range(0, n_total, batch_size):
            chunk_paths = unique_strs[start : start + batch_size]
            imgs = torch.stack([image_loader(p) for p in chunk_paths], dim=0).to(
                device=device, non_blocking=True
            )
            c_out, p_out, m_out = backbone.extract_all_features(
                imgs, layers=multilayer_layers
            )
            if need_style_patches:
                cls_list.append(c_out.detach().to(device="cpu", dtype=torch.float16))
                patch_list.append(p_out.detach().to(device="cpu", dtype=torch.float16))
            if m_out is not None:
                stacked = torch.stack(m_out, dim=1)  # [b, L, 3*D]
                multi_list.append(stacked.detach().to(device="cpu", dtype=torch.float16))

        cls_t = torch.cat(cls_list, dim=0) if cls_list else None
        patch_t = torch.cat(patch_list, dim=0) if patch_list else None
        multi_t = torch.cat(multi_list, dim=0) if multi_list else None
        return cls(
            path_to_idx=path_to_idx,
            cls_feats=cls_t,
            patch_feats=patch_t,
            multilayer_feats=multi_t,
            grid_size=backbone.grid_size,
        )

    def _lookup_indices(self, paths: Iterable[str]) -> torch.Tensor:
        return torch.tensor(
            [self.path_to_idx[p] for p in paths], dtype=torch.long
        )

    def get_style_dino(
        self,
        flat_paths: Sequence[str],
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.cls_feats is not None and self.patch_feats is not None
        idx = self._lookup_indices(flat_paths)
        c = self.cls_feats[idx].to(device=device, dtype=dtype, non_blocking=True)
        p = self.patch_feats[idx].to(device=device, dtype=dtype, non_blocking=True)
        return c, p

    def get_repa_grids(
        self,
        paths: Sequence[str],
        device: torch.device,
    ) -> torch.Tensor:
        assert self.patch_feats is not None
        idx = self._lookup_indices(paths)
        patches = self.patch_feats[idx].to(device=device, dtype=torch.float32, non_blocking=True)
        patches = F.normalize(patches, dim=-1)
        b, _, d = patches.shape
        return patches.transpose(1, 2).reshape(b, d, self.grid_size, self.grid_size)

    def get_multilayer(
        self,
        flat_paths: Sequence[str],
        device: torch.device,
    ) -> List[torch.Tensor]:
        assert self.multilayer_feats is not None
        idx = self._lookup_indices(flat_paths)
        # [N, num_layers, 3*D] -> list of num_layers tensors [N, 3*D]
        feats = self.multilayer_feats[idx].to(device=device, dtype=torch.float32, non_blocking=True)
        return [feats[:, l_idx] for l_idx in range(feats.shape[1])]


# 进程内缓存，防止同一进程内主模型、REPA 与 SCR 重复加载多份冻结 DINOv2 占显存
_BACKBONE_CACHE: Dict[Tuple[str, int, str], FrozenVisionBackbone] = {}


def get_shared_vision_backbone(
    model_name: str,
    input_size: int = 140,
    device: torch.device | str = "cpu",
) -> FrozenVisionBackbone:
    dev = torch.device(device)
    key = (str(model_name).strip(), int(input_size), str(dev))
    if key not in _BACKBONE_CACHE:
        backbone = FrozenVisionBackbone(model_name=model_name, input_size=input_size).to(dev)
        backbone.eval()
        _BACKBONE_CACHE[key] = backbone
    return _BACKBONE_CACHE[key]


def clear_vision_backbone_cache() -> None:
    _BACKBONE_CACHE.clear()
