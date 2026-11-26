from __future__ import annotations

from pathlib import Path
import time
from typing import List, Optional, Sequence, Tuple, Union
import yaml

from PIL import Image, ImageDraw
import torch
from tqdm.auto import tqdm

from ..config import DataConfig, ModelConfig, SamplingConfig
from ..data.index import VALID_IMAGE_EXTENSIONS
from ..data.preprocess import SquareResizeOrPad, build_image_transform, load_rgb_image
from ..models.build import load_flow_checkpoint
from ..sampling.pipeline import StyleDiffuserFlowPipeline


def load_inference_pipeline(
    ckpt_dir: str | Path,
    device: str = "cuda:0",
    sampling_cfg: Optional[SamplingConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    model_cfg: Optional[ModelConfig] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[StyleDiffuserFlowPipeline, ModelConfig, DataConfig]:
    sampling = sampling_cfg if sampling_cfg is not None else SamplingConfig()
    model, resolved_model_cfg, meta_data_cfg = load_flow_checkpoint(
        ckpt_dir=ckpt_dir,
        model_cfg=model_cfg,
        use_ema=sampling.use_ema,
        map_location="cpu",
    )
    resolved_data_cfg = data_cfg if data_cfg is not None else meta_data_cfg
    torch_device = torch.device(device)
    if dtype is None and torch_device.type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if dtype is not None:
        model.to(device=torch_device, dtype=dtype)
        model.vision_backbone.set_compute_dtype(dtype)
    else:
        model.to(torch_device)
    model.eval()

    pipe = StyleDiffuserFlowPipeline(model=model, sampling_cfg=sampling)
    return pipe, resolved_model_cfg, resolved_data_cfg


def _save_comparison_strip(
    content_pil: Image.Image,
    style_pil: Image.Image,
    output_pil: Image.Image,
    data_cfg: DataConfig,
    save_path: Path,
    mode_label: str = "",
) -> Path:
    res = data_cfg.resolution
    resizer = SquareResizeOrPad(
        resolution=res,
        resize_mode=data_cfg.resize_mode,
        pad_ratio=data_cfg.pad_ratio,
        interpolation=data_cfg.interpolation,
        background_color=data_cfg.background_color,
    )
    pad = 4
    header_h = 24
    canvas = Image.new("RGB", (res * 3 + pad * 4, res + header_h + pad), (236, 238, 242))
    draw = ImageDraw.Draw(canvas)
    gen_label = f"Generated ({mode_label})" if mode_label else "Generated"
    for idx, (label, img) in enumerate(
        (
            ("Content", resizer(content_pil)),
            ("Style Ref", resizer(style_pil)),
            (gen_label, output_pil.resize((res, res), Image.Resampling.BILINEAR)),
        )
    ):
        x = pad + idx * (res + pad)
        draw.text((x + 4, 4), label, fill=(32, 36, 42))
        canvas.paste(img, (x, header_h))
    canvas.save(save_path)
    return save_path


def infer_images(
    pipe: StyleDiffuserFlowPipeline,
    model_cfg: ModelConfig,
    data_cfg: DataConfig,
    sampling_cfg: SamplingConfig,
    style_image: Union[Image.Image, Sequence[Image.Image]],
    content_items: Sequence[Tuple[str, Image.Image]],
    output_dir: Optional[str | Path] = None,
    batch_size: int = 4,
    seed: Optional[int] = 123,
    device: str = "cuda:0",
    save_comparison: bool = True,
    sample_indices: Optional[Sequence[int]] = None,
) -> List[Tuple[str, Image.Image]]:
    """
    通用纯图像域批量推理：
    - 支持单张或多张风格参考图（K-shot，整批仅编码 1 次并缓存 KV）；
    - 支持 fast（1-4 步免 CFG 极速）与 hd（8-10 步 Dual-CFG + APG 高画质）双档推理；
    - 按每张样本的全局索引 `seed + sample_idx` 独立采样初始噪声，跨 batch_size / 多卡分片结果完全一致。
    """
    if not content_items:
        return []

    if sample_indices is not None:
        if len(sample_indices) != len(content_items):
            raise ValueError(
                f"sample_indices 长度 ({len(sample_indices)}) 必须等于 content_items 长度 ({len(content_items)})。"
            )
        resolved_indices = [int(i) for i in sample_indices]
    else:
        resolved_indices = list(range(len(content_items)))

    torch_device = torch.device(device)
    model_dtype = next(pipe.model.parameters()).dtype
    bsz = max(1, int(batch_size))
    to_tensor = build_image_transform(data_cfg, normalize=True)

    style_pils = (
        [style_image]
        if isinstance(style_image, Image.Image)
        else list(style_image)
    )
    if not style_pils:
        raise ValueError("必须至少提供 1 张风格参考图。")

    # [1, K, 3, H, W]
    style_tensor = (
        torch.stack([to_tensor(s) for s in style_pils], dim=0)
        .unsqueeze(0)
        .to(device=torch_device, dtype=model_dtype)
    )
    res = data_cfg.resolution
    resolved_mode = sampling_cfg.resolved_mode()

    results: List[Tuple[str, Image.Image]] = []
    start = time.time()
    with torch.no_grad():
        style_pack = pipe.encode_style(style_tensor, target_hw=(res, res))
        chunk_starts = list(range(0, len(content_items), bsz))
        for start_idx in tqdm(chunk_starts, desc="Sampling"):
            chunk = content_items[start_idx : start_idx + bsz]
            chunk_indices = resolved_indices[start_idx : start_idx + bsz]
            content_tensor = torch.stack(
                [to_tensor(img) for _, img in chunk], dim=0
            ).to(device=torch_device, dtype=model_dtype)

            chunk_seeds = (
                [int(seed) + s_idx for s_idx in chunk_indices]
                if seed is not None
                else None
            )

            images = pipe.generate(
                content_images=content_tensor,
                style_pack=style_pack,
                num_inference_steps=sampling_cfg.num_inference_steps,
                mode=resolved_mode,
                solver=sampling_cfg.solver,
                content_guidance_scale=sampling_cfg.content_guidance_scale,
                style_guidance_scale=sampling_cfg.style_guidance_scale,
                use_apg=sampling_cfg.use_apg,
                guidance_interval=sampling_cfg.guidance_interval,
                time_shift=sampling_cfg.time_shift,
                seeds=chunk_seeds,
            )
            results.extend(zip((name for name, _ in chunk), images))

    elapsed = time.time() - start
    print(
        f"Finished sampling {len(results)} image(s) "
        f"[{resolved_mode}, {sampling_cfg.num_inference_steps} steps, {len(style_pils)} style ref(s)] "
        f"in {elapsed:.2f}s.",
        flush=True,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, img in results:
            stem = Path(name).stem if "." in name else name
            img.save(out / f"{stem}.png")
        if save_comparison and results:
            _save_comparison_strip(
                content_pil=content_items[0][1],
                style_pil=style_pils[0],
                output_pil=results[0][1],
                data_cfg=data_cfg,
                save_path=out / "comparison.png",
                mode_label=f"{resolved_mode} {sampling_cfg.num_inference_steps}st",
            )
        meta = {
            "resolution": data_cfg.resolution,
            "resize_mode": data_cfg.resize_mode,
            "mode": resolved_mode,
            "num_inference_steps": sampling_cfg.num_inference_steps,
            "solver": sampling_cfg.solver,
            "content_guidance_scale": sampling_cfg.content_guidance_scale,
            "style_guidance_scale": sampling_cfg.style_guidance_scale,
            "use_apg": sampling_cfg.use_apg,
            "time_shift": sampling_cfg.time_shift,
            "num_style_refs": len(style_pils),
            "seed": seed,
            "seed_mode": "per_sample" if seed is not None else "random",
            "count": len(results),
            "elapsed_sec": round(elapsed, 4),
        }
        with (out / "sampling_meta.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(meta, fh, allow_unicode=True, sort_keys=False)

    return results


def infer_from_paths(
    ckpt_dir: str | Path,
    style_image_path: Union[str, Path, Sequence[str | Path]],
    output_dir: str | Path,
    content_image_path: Optional[str | Path] = None,
    content_dir: Optional[str | Path] = None,
    sampling_cfg: Optional[SamplingConfig] = None,
    data_cfg: Optional[DataConfig] = None,
    batch_size: int = 4,
    seed: Optional[int] = 123,
    device: str = "cuda:0",
    save_comparison: bool = True,
) -> List[Tuple[str, Image.Image]]:
    if not content_image_path and not content_dir:
        raise ValueError("必须提供 content_image_path 或 content_dir 之一。")

    sampling = sampling_cfg if sampling_cfg is not None else SamplingConfig()
    pipe, resolved_model_cfg, resolved_data_cfg = load_inference_pipeline(
        ckpt_dir=ckpt_dir,
        device=device,
        sampling_cfg=sampling,
        data_cfg=data_cfg,
    )

    bg = resolved_data_cfg.background_color
    if isinstance(style_image_path, (str, Path)):
        style_paths = [Path(style_image_path)]
    else:
        style_paths = [Path(p) for p in style_image_path]
    style_pils = [load_rgb_image(p, background_color=bg) for p in style_paths]

    paths: List[Path] = []
    if content_image_path:
        paths.append(Path(content_image_path))
    if content_dir:
        cdir = Path(content_dir)
        paths.extend(
            p
            for p in sorted(cdir.iterdir(), key=lambda x: x.name)
            if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS
        )

    if not paths:
        raise FileNotFoundError("未找到可供推理的内容图。")

    items = [(p.stem, load_rgb_image(p, background_color=bg)) for p in paths]
    return infer_images(
        pipe=pipe,
        model_cfg=resolved_model_cfg,
        data_cfg=resolved_data_cfg,
        sampling_cfg=sampling,
        style_image=style_pils,
        content_items=items,
        output_dir=output_dir,
        batch_size=batch_size,
        seed=seed,
        device=device,
        save_comparison=save_comparison,
    )
