from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple
import yaml

import torch

from ..config import DataConfig, ModelConfig, ProjectConfig, SCRConfig
from .ema import ModelEMA
from .pixel_dit import PixelDiTFlowModel
from .scr import SCRv2
from .vision_backbone import FrozenVisionBackbone, get_shared_vision_backbone


def build_vision_backbone(
    model_name: str,
    input_size: int = 140,
    device: torch.device | str = "cpu",
    shared: bool = False,
) -> FrozenVisionBackbone:
    if shared:
        return get_shared_vision_backbone(
            model_name=model_name, input_size=input_size, device=device
        )
    return FrozenVisionBackbone(model_name=model_name, input_size=input_size).to(device)


def build_flow_model(
    model_cfg: ModelConfig,
    vision_backbone: Optional[FrozenVisionBackbone] = None,
    device: torch.device | str = "cpu",
) -> PixelDiTFlowModel:
    model_cfg.validate()
    if vision_backbone is None:
        vision_backbone = build_vision_backbone(
            model_name=model_cfg.vision_backbone,
            input_size=model_cfg.vision_input_size,
            device=device,
            shared=False,
        )
    return PixelDiTFlowModel(model_cfg=model_cfg, vision_backbone=vision_backbone).to(device)


def build_scr(
    scr_cfg: Optional[SCRConfig] = None,
    vision_backbone: Optional[FrozenVisionBackbone] = None,
    mode: str = "training",
    device: torch.device | str = "cpu",
) -> SCRv2:
    cfg = scr_cfg if scr_cfg is not None else SCRConfig()
    if vision_backbone is None:
        vision_backbone = build_vision_backbone(
            model_name=cfg.vision_backbone,
            input_size=cfg.vision_input_size,
            device=device,
            shared=False,
        )
    return SCRv2(
        vision_backbone=vision_backbone,
        feature_layers=cfg.feature_layers,
        proj_dim=cfg.proj_dim,
        temperature=cfg.temperature,
        mode=mode,
        neg_chunk=cfg.neg_chunk,
    ).to(device)


def save_flow_checkpoint(
    save_dir: str | Path,
    model: PixelDiTFlowModel,
    cfg: ProjectConfig,
    ema: Optional[ModelEMA] = None,
    global_step: Optional[int] = None,
) -> Path:
    """
    保存现代 PixelDiT-Flow 轻量化权重包：
    - pixel_dit.pth: 仅含可训练参数（不含冻结 DINOv2）
    - ema.pth:       EMA 平滑权重（推理默认优先加载）
    - model_config.yaml: 完整架构与预处理元数据
    """
    out = Path(save_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.trainable_state_dict(), out / "pixel_dit.pth")
    if ema is not None:
        torch.save(ema.state_dict(), out / "ema.pth")

    dims = cfg.model.resolved_dims()
    meta = {
        "format": "pixel_dit_flow_v3",
        "resolution": cfg.data.resolution,
        "resize_mode": cfg.data.resize_mode,
        "pad_ratio": cfg.data.pad_ratio,
        "interpolation": cfg.data.interpolation,
        "background_color": list(cfg.data.background_color),
        "max_style_refs": cfg.data.max_style_refs,
        "model": {
            "preset": cfg.model.preset,
            "patch_size": cfg.model.patch_size,
            "hidden_size": dims["hidden_size"],
            "depth": dims["depth"],
            "num_heads": dims["num_heads"],
            "mlp_dim": dims["mlp_dim"],
            "ada_dim": dims["ada_dim"],
            "pixel_depth": dims["pixel_depth"],
            "pixel_dim": dims["pixel_dim"],
            "repa_layer": dims["repa_layer"],
            "style_latent_tokens": dims["style_latent_tokens"],
            "style_resampler_depth": dims["style_resampler_depth"],
            "stroke_tokens_k": dims["stroke_tokens_k"],
            "use_spatial_warper": bool(cfg.model.use_spatial_warper),
            "vision_backbone": cfg.model.vision_backbone,
            "vision_input_size": cfg.model.vision_input_size,
            "style_token_drop_prob": cfg.model.style_token_drop_prob,
        },
        "has_ema": ema is not None,
    }
    if global_step is not None:
        meta["global_step"] = int(global_step)
    with (out / "model_config.yaml").open("w", encoding="utf-8") as fh:
        yaml.safe_dump(meta, fh, allow_unicode=True, sort_keys=False)
    return out


def save_scr_checkpoint(
    ckpt_path: str | Path,
    scr: SCRv2,
    scr_cfg: SCRConfig,
    global_step: int,
) -> Path:
    path = Path(ckpt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "scr_v2",
        "global_step": int(global_step),
        "vision_backbone": scr_cfg.vision_backbone,
        "vision_input_size": scr_cfg.vision_input_size,
        "feature_layers": list(scr_cfg.feature_layers),
        "proj_dim": scr_cfg.proj_dim,
        "temperature": scr_cfg.temperature,
        "state_dict": scr.trainable_state_dict(),
    }
    torch.save(payload, path)
    return path


def load_scr_checkpoint(
    ckpt_path: str | Path,
    vision_backbone: Optional[FrozenVisionBackbone] = None,
    mode: str = "refinement",
    neg_chunk: int = 8,
    map_location: str | torch.device = "cpu",
) -> SCRv2:
    payload = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "scr_v2":
        raise ValueError(
            f"SCR 权重 '{ckpt_path}' 不是有效的新版 SCR 2.0 (scr_v2) 格式。"
        )
    scr_cfg = SCRConfig(
        vision_backbone=str(payload["vision_backbone"]),
        vision_input_size=int(payload["vision_input_size"]),
        feature_layers=tuple(int(x) for x in payload["feature_layers"]),
        proj_dim=int(payload["proj_dim"]),
        temperature=float(payload["temperature"]),
        neg_chunk=neg_chunk,
    )
    scr = build_scr(
        scr_cfg=scr_cfg,
        vision_backbone=vision_backbone,
        mode=mode,
        device=map_location,
    )
    scr.load_trainable_state_dict(payload["state_dict"], strict=True)
    return scr


def load_checkpoint_metadata(ckpt_dir: str | Path) -> Tuple[ModelConfig, DataConfig]:
    meta_path = Path(ckpt_dir) / "model_config.yaml"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"权重目录 '{ckpt_dir}' 缺少 model_config.yaml（仅支持 PixelDiT-Flow v3 权重）。"
        )
    with meta_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    if raw.get("format") != "pixel_dit_flow_v3":
        raise ValueError(
            f"权重目录 '{ckpt_dir}' 格式为 {raw.get('format')!r}，与当前 PixelDiT-Flow v3 不兼容。"
        )
    mraw = raw.get("model", {})
    model_cfg = ModelConfig(
        preset=str(mraw.get("preset", "B")),
        patch_size=int(mraw.get("patch_size", 8)),
        hidden_size=int(mraw["hidden_size"]) if mraw.get("hidden_size") is not None else None,
        depth=int(mraw["depth"]) if mraw.get("depth") is not None else None,
        num_heads=int(mraw["num_heads"]) if mraw.get("num_heads") is not None else None,
        mlp_dim=int(mraw["mlp_dim"]) if mraw.get("mlp_dim") is not None else None,
        ada_dim=int(mraw["ada_dim"]) if mraw.get("ada_dim") is not None else None,
        pixel_depth=int(mraw["pixel_depth"]) if mraw.get("pixel_depth") is not None else None,
        pixel_dim=int(mraw["pixel_dim"]) if mraw.get("pixel_dim") is not None else None,
        repa_layer=int(mraw["repa_layer"]) if mraw.get("repa_layer") is not None else None,
        style_latent_tokens=(
            int(mraw["style_latent_tokens"])
            if mraw.get("style_latent_tokens") is not None
            else None
        ),
        style_resampler_depth=(
            int(mraw["style_resampler_depth"])
            if mraw.get("style_resampler_depth") is not None
            else None
        ),
        stroke_tokens_k=(
            int(mraw["stroke_tokens_k"])
            if mraw.get("stroke_tokens_k") is not None
            else None
        ),
        use_spatial_warper=bool(mraw.get("use_spatial_warper", True)),
        vision_backbone=str(mraw.get("vision_backbone", "facebook/dinov2-small")),
        vision_input_size=int(mraw.get("vision_input_size", 140)),
        style_token_drop_prob=float(mraw.get("style_token_drop_prob", 0.15)),
    )
    data_cfg = DataConfig(
        resolution=int(raw.get("resolution", 128)),
        resize_mode=str(raw.get("resize_mode", "pad")),
        pad_ratio=float(raw.get("pad_ratio", 0.0)),
        interpolation=str(raw.get("interpolation", "bilinear")),
        background_color=tuple(
            int(c) for c in raw.get("background_color", (255, 255, 255))
        ),
        max_style_refs=int(raw.get("max_style_refs", 1)),
    )
    return model_cfg, data_cfg


def load_flow_weights(
    model: PixelDiTFlowModel,
    ckpt_dir: str | Path,
    use_ema: bool = False,
    ema: Optional[ModelEMA] = None,
    map_location: str | torch.device = "cpu",
) -> None:
    ckpt = Path(ckpt_dir)
    weight_path = ckpt / "pixel_dit.pth"
    if not weight_path.is_file():
        raise FileNotFoundError(f"权重目录 '{ckpt}' 缺少核心权重文件: pixel_dit.pth")
    state = torch.load(weight_path, map_location=map_location, weights_only=True)
    model.load_trainable_state_dict(state, strict=True)

    ema_path = ckpt / "ema.pth"
    if ema_path.is_file():
        ema_state = torch.load(ema_path, map_location=map_location, weights_only=True)
        if ema is not None:
            ema.load_state_dict(ema_state, device=torch.device(map_location))
        if use_ema:
            shadow = ema_state.get("shadow", {})
            if isinstance(shadow, dict) and shadow:
                model.load_trainable_state_dict(shadow, strict=False)


def load_flow_checkpoint(
    ckpt_dir: str | Path,
    model_cfg: Optional[ModelConfig] = None,
    use_ema: bool = True,
    map_location: str | torch.device = "cpu",
) -> Tuple[PixelDiTFlowModel, ModelConfig, DataConfig]:
    meta_model, meta_data = load_checkpoint_metadata(ckpt_dir)
    final_model_cfg = model_cfg if model_cfg is not None else meta_model
    model = build_flow_model(model_cfg=final_model_cfg, device=map_location)
    load_flow_weights(
        model=model,
        ckpt_dir=ckpt_dir,
        use_ema=use_ema,
        map_location=map_location,
    )
    return model, final_model_cfg, meta_data
