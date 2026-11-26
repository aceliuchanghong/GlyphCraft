from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
import torch

from ..config import ProjectConfig
from ..data.index import PreviewSample
from ..data.preprocess import SquareResizeOrPad, build_image_transform, load_rgb_image
from ..models.ema import ModelEMA
from ..models.pixel_dit import PixelDiTFlowModel
from ..sampling.pipeline import StyleDiffuserFlowPipeline


def _compose_preview_grid(
    rows: List[Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]],
    resolution: int,
    step_label: str,
    fast_steps: int,
    hd_steps: int,
) -> Image.Image:
    """将多组 (Content | Style Ref | Fast | HD | Ground Truth) 拼成五列双档验证对比网格。"""
    pad = 4
    header_h = 26
    cols = 5
    grid_w = cols * resolution + (cols + 1) * pad
    grid_h = header_h + len(rows) * (resolution + pad) + pad

    canvas = Image.new("RGB", (grid_w, grid_h), (236, 238, 242))
    draw = ImageDraw.Draw(canvas)

    headers = [
        "Content",
        "Style Ref",
        f"Fast {fast_steps}st ({step_label})",
        f"HD {hd_steps}st",
        "Ground Truth",
    ]
    for col_idx, text in enumerate(headers):
        x = pad + col_idx * (resolution + pad) + 4
        draw.text((x, 6), text, fill=(32, 36, 42))

    for row_idx, quint in enumerate(rows):
        y = header_h + row_idx * (resolution + pad)
        for col_idx, tile in enumerate(quint):
            x = pad + col_idx * (resolution + pad)
            if tile.size != (resolution, resolution):
                tile = tile.resize((resolution, resolution), Image.Resampling.BILINEAR)
            canvas.paste(tile, (x, y))

    return canvas


def _log_grid_to_tensorboard(
    accelerator,
    grid_pil: Image.Image,
    tag: str,
    global_step: int,
) -> None:
    if accelerator is None:
        return
    try:
        tb_tracker = accelerator.get_tracker("tensorboard", unwrap=True)
    except Exception:
        tb_tracker = None
    if tb_tracker is None or not hasattr(tb_tracker, "add_image"):
        return
    arr = np.asarray(grid_pil, dtype=np.float32) / 255.0
    chw = np.transpose(arr, (2, 0, 1))
    tb_tracker.add_image(tag, chw, global_step=global_step)


def render_validation_grid(
    model: PixelDiTFlowModel,
    preview_samples: Sequence[PreviewSample],
    cfg: ProjectConfig,
    device: torch.device | str,
    output_path: str | Path,
    global_step: int,
    ema: Optional[ModelEMA] = None,
    fast_steps: int = 4,
    hd_steps: int = 10,
    content_scale: Optional[float] = None,
    style_scale: Optional[float] = None,
    time_shift: Optional[float] = None,
    accelerator=None,
    tb_tag: str = "val/preview_grid",
) -> Optional[Path]:
    """训练期同屏渲染固定噪声种子的 5 列对比图：Content | Style Ref | Fast (1-4步) | HD (8-10步) | GT。"""
    if not preview_samples:
        return None

    was_training = model.training
    model.eval()

    resolution = cfg.data.resolution
    resizer = SquareResizeOrPad(
        resolution=resolution,
        resize_mode=cfg.data.resize_mode,
        pad_ratio=cfg.data.pad_ratio,
        interpolation=cfg.data.interpolation,
        background_color=cfg.data.background_color,
    )
    to_tensor = build_image_transform(cfg.data, normalize=True)
    torch_device = torch.device(device)
    w_c = (
        float(content_scale)
        if content_scale is not None
        else cfg.sampling.content_guidance_scale
    )
    w_s = (
        float(style_scale)
        if style_scale is not None
        else cfg.sampling.style_guidance_scale
    )

    shift = (
        float(time_shift) if time_shift is not None else float(cfg.sampling.time_shift)
    )

    pipe = StyleDiffuserFlowPipeline(model=model, sampling_cfg=cfg.sampling)
    ema_ctx = ema.average_parameters(model) if ema is not None else nullcontext()

    rows: List[Tuple[Image.Image, Image.Image, Image.Image, Image.Image, Image.Image]] = []
    try:
        with ema_ctx, torch.no_grad():
            for idx, sample in enumerate(preview_samples):
                content_pil = load_rgb_image(
                    sample.content_path, background_color=cfg.data.background_color
                )
                style_pil = load_rgb_image(
                    sample.style_ref_path, background_color=cfg.data.background_color
                )
                target_pil = load_rgb_image(
                    sample.target_path, background_color=cfg.data.background_color
                )

                content_tensor = to_tensor(content_pil).unsqueeze(0).to(torch_device)
                style_tensor = to_tensor(style_pil).unsqueeze(0).to(torch_device)
                style_pack = pipe.encode_style(
                    style_tensor, target_hw=(resolution, resolution)
                )

                sample_seed = [int(cfg.seed) + idx]
                pred_fast = pipe.generate(
                    content_images=content_tensor,
                    style_pack=style_pack,
                    num_inference_steps=max(1, int(fast_steps)),
                    mode="fast",
                    content_guidance_scale=w_c,
                    style_guidance_scale=w_s,
                    time_shift=shift,
                    seeds=sample_seed,
                )[0]

                pred_hd = pipe.generate(
                    content_images=content_tensor,
                    style_pack=style_pack,
                    num_inference_steps=max(1, int(hd_steps)),
                    mode="hd",
                    content_guidance_scale=w_c,
                    style_guidance_scale=w_s,
                    time_shift=shift,
                    seeds=sample_seed,
                )[0]

                rows.append(
                    (
                        resizer(content_pil),
                        resizer(style_pil),
                        pred_fast,
                        pred_hd,
                        resizer(target_pil),
                    )
                )
    finally:
        if was_training:
            model.train()

    grid_img = _compose_preview_grid(
        rows=rows,
        resolution=resolution,
        step_label=f"step {global_step}",
        fast_steps=max(1, int(fast_steps)),
        hd_steps=max(1, int(hd_steps)),
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    grid_img.save(out)
    _log_grid_to_tensorboard(accelerator, grid_img, tag=tb_tag, global_step=global_step)
    return out
