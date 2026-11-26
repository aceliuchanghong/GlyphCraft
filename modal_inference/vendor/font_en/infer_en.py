from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch

from style_diffuser.config import SamplingConfig, load_config
from style_diffuser.data.preprocess import load_rgb_image
from style_diffuser.engine.infer import infer_images, load_inference_pipeline
from style_diffuser.engine.logging_utils import find_latest_step_dir

try:
    from .charset import ASCII_94, glyph_stem, load_ascii_charset, stem_to_char
    from .render_en import (
        DEFAULT_BASELINE_RATIO,
        DEFAULT_CAP_HEIGHT_RATIO,
        calibrate_font_em_baseline,
        load_cmap_codepoints,
        open_freetype_font,
        render_char_em_baseline,
        render_sentence_preview,
    )
    from .to_ttf import build_ttf_from_glyph_dir
except ImportError:
    from charset import ASCII_94, glyph_stem, load_ascii_charset, stem_to_char
    from render_en import (
        DEFAULT_BASELINE_RATIO,
        DEFAULT_CAP_HEIGHT_RATIO,
        calibrate_font_em_baseline,
        load_cmap_codepoints,
        open_freetype_font,
        render_char_em_baseline,
        render_sentence_preview,
    )
    from to_ttf import build_ttf_from_glyph_dir


def _build_content_items(
    chars: Sequence[str],
    resolution: int,
    content_ttf: Optional[str | Path] = None,
    content_dir: Optional[str | Path] = None,
    baseline_ratio: float = DEFAULT_BASELINE_RATIO,
    cap_height_ratio: float = DEFAULT_CAP_HEIGHT_RATIO,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> List[Tuple[str, Image.Image]]:
    items: List[Tuple[str, Image.Image]] = []

    if content_dir is not None:
        cdir = Path(content_dir)
        for ch in chars:
            stem = glyph_stem(ch)
            img_path = cdir / f"{stem}.png"
            if not img_path.exists():
                print(f"[WARN] content_dir 缺少 {ch!r} ({img_path.name})，跳过。")
                continue
            items.append((stem, load_rgb_image(img_path, background_color=background_color)))
        return items

    if content_ttf is None:
        raise ValueError("必须提供 --content-ttf 或 --content-dir 之一。")

    cpath = Path(content_ttf)
    cmap = load_cmap_codepoints(cpath)
    calib = calibrate_font_em_baseline(
        font_path=cpath,
        chars=ASCII_94,
        image_size=resolution,
        baseline_ratio=baseline_ratio,
        cap_height_ratio=cap_height_ratio,
    )
    font = open_freetype_font(cpath, size=calib.font_size)
    for ch in chars:
        if cmap is not None and ord(ch) not in cmap:
            print(f"[WARN] 内容原型字体不含 {ch!r}，跳过。")
            continue
        img = render_char_em_baseline(
            font=font,
            char=ch,
            image_size=resolution,
            baseline_y=calib.baseline_y,
        )
        if img is None:
            continue
        items.append((glyph_stem(ch), img))
    return items


def _load_style_images(
    style_image_paths: Optional[Sequence[str | Path]],
    style_ttf: Optional[str | Path],
    style_chars: str,
    resolution: int,
    baseline_ratio: float,
    cap_height_ratio: float,
    background_color: Tuple[int, int, int],
) -> List[Image.Image]:
    pils: List[Image.Image] = []
    if style_image_paths:
        for p in style_image_paths:
            pils.append(load_rgb_image(p, background_color=background_color))
    if style_ttf is not None:
        spath = Path(style_ttf)
        calib = calibrate_font_em_baseline(
            font_path=spath,
            chars=ASCII_94,
            image_size=resolution,
            baseline_ratio=baseline_ratio,
            cap_height_ratio=cap_height_ratio,
        )
        font = open_freetype_font(spath, size=calib.font_size)
        for ch in load_ascii_charset(chars=style_chars):
            img = render_char_em_baseline(
                font=font,
                char=ch,
                image_size=resolution,
                baseline_y=calib.baseline_y,
            )
            if img is not None:
                pils.append(img)
    if not pils:
        raise ValueError("必须提供至少 1 张 --style-image，或提供 --style-ttf。")
    return pils


def infer_english_font(
    output_dir: str | Path,
    config_path: Optional[str | Path] = None,
    ckpt_dir: Optional[str | Path] = None,
    style_image_paths: Optional[Union[str, Path, Sequence[str | Path]]] = None,
    style_ttf: Optional[str | Path] = None,
    style_chars: str = "Hgp",
    content_ttf: Optional[str | Path] = None,
    content_dir: Optional[str | Path] = None,
    text: Optional[str] = None,
    baseline_ratio: float = DEFAULT_BASELINE_RATIO,
    cap_height_ratio: float = DEFAULT_CAP_HEIGHT_RATIO,
    batch_size: int = 16,
    mode: str = "auto",
    steps: Optional[int] = None,
    solver: Optional[str] = None,
    content_scale: Optional[float] = None,
    style_scale: Optional[float] = None,
    time_shift: Optional[float] = None,
    use_apg: Optional[bool] = None,
    use_ema: Optional[bool] = None,
    seed: int = 123,
    device: str = "cuda:0",
    save_comparison: bool = True,
    preview_text: str = "Hamburgefontsiv 0123456789 !?.,'\"-@",
    pack_ttf: Optional[str | Path] = None,
    family_name: str = "StyleDiffuserEN",
    style_name: str = "Regular",
) -> List[Tuple[str, Image.Image]]:
    """
    英文专用推理入口：
    - 默认生成全部 94 个可打印 ASCII（传 --text 则仅生成指定字符）；
    - 内容图与可选 --style-ttf 均严格使用与训练完全一致的 em_baseline 标尺；
    - 自动输出 preview_sentence.png（带基线参考线的整行拼词图），并支持一键 --pack-ttf 打包 .ttf。
    """
    data_cfg = None
    sampling_cfg = SamplingConfig()
    resolved_ckpt = ckpt_dir

    if config_path:
        cfg = load_config(config_path)
        data_cfg = cfg.data
        sampling_cfg = cfg.sampling
        if not resolved_ckpt:
            latest = find_latest_step_dir(cfg.phase2_dir) or find_latest_step_dir(
                cfg.phase1_dir
            )
            if latest is None:
                raise FileNotFoundError(
                    f"未在 '{cfg.phase2_dir}' 或 '{cfg.phase1_dir}' 找到权重，请显式传入 --ckpt-dir。"
                )
            resolved_ckpt = latest

    if not resolved_ckpt:
        raise ValueError("必须提供 --ckpt-dir 或 --config。")

    ckpt_path = Path(resolved_ckpt)
    if config_path is None:
        bundled_cfg = ckpt_path / "resolved_config.yaml"
        if bundled_cfg.is_file():
            cfg = load_config(bundled_cfg)
            data_cfg = cfg.data
            sampling_cfg = cfg.sampling

    if content_dir is None and content_ttf is None:
        bundled_content = ckpt_path / "content"
        fallback_content = Path("data/font_en_128/content")
        if bundled_content.is_dir() and any(bundled_content.glob("*.png")):
            content_dir = bundled_content
        elif fallback_content.is_dir() and any(fallback_content.glob("*.png")):
            content_dir = fallback_content

    sampling_cfg.mode = mode
    if steps is not None:
        sampling_cfg.num_inference_steps = int(steps)
    if solver is not None:
        sampling_cfg.solver = solver
    if content_scale is not None:
        sampling_cfg.content_guidance_scale = float(content_scale)
    if style_scale is not None:
        sampling_cfg.style_guidance_scale = float(style_scale)
    if time_shift is not None:
        sampling_cfg.time_shift = float(time_shift)
    if use_apg is not None:
        sampling_cfg.use_apg = bool(use_apg)
    if use_ema is not None:
        sampling_cfg.use_ema = bool(use_ema)

    t0 = time.perf_counter()
    pipe, model_cfg, resolved_data_cfg = load_inference_pipeline(
        ckpt_dir=resolved_ckpt,
        device=device,
        sampling_cfg=sampling_cfg,
        data_cfg=data_cfg,
    )

    chars = load_ascii_charset(chars=text) if text is not None else list(ASCII_94)
    content_items = _build_content_items(
        chars=chars,
        resolution=resolved_data_cfg.resolution,
        content_ttf=content_ttf,
        content_dir=content_dir,
        baseline_ratio=baseline_ratio,
        cap_height_ratio=cap_height_ratio,
        background_color=resolved_data_cfg.background_color,
    )
    if not content_items:
        raise ValueError("没有可推理的有效内容字符。")

    if isinstance(style_image_paths, (str, Path)):
        style_list: Optional[Sequence[str | Path]] = [style_image_paths]
    else:
        style_list = style_image_paths

    style_pils = _load_style_images(
        style_image_paths=style_list,
        style_ttf=style_ttf,
        style_chars=style_chars,
        resolution=resolved_data_cfg.resolution,
        baseline_ratio=baseline_ratio,
        cap_height_ratio=cap_height_ratio,
        background_color=resolved_data_cfg.background_color,
    )

    out_path = Path(output_dir)
    results = infer_images(
        pipe=pipe,
        model_cfg=model_cfg,
        data_cfg=resolved_data_cfg,
        sampling_cfg=sampling_cfg,
        style_image=style_pils,
        content_items=content_items,
        output_dir=out_path,
        batch_size=batch_size,
        seed=seed,
        device=device,
        save_comparison=save_comparison,
    )

    glyph_map: Dict[str, Image.Image] = {}
    for stem, pil_img in results:
        ch = stem_to_char(stem)
        if ch is not None:
            glyph_map[ch] = pil_img

    if glyph_map:
        out_path.mkdir(parents=True, exist_ok=True)
        sentence = preview_text if text is None else text
        preview = render_sentence_preview(
            glyph_map,
            text=sentence,
            image_size=resolved_data_cfg.resolution,
            baseline_y=int(round(resolved_data_cfg.resolution * baseline_ratio)),
        )
        preview.save(out_path / "preview_sentence.png")

    if pack_ttf is not None:
        build_ttf_from_glyph_dir(
            input_dir=out_path,
            output_ttf=pack_ttf,
            family_name=family_name,
            style_name=style_name,
            baseline_ratio=baseline_ratio,
        )

    elapsed = time.perf_counter() - t0
    ttf_note = f", TTF -> {pack_ttf}" if pack_ttf is not None else ""
    print(
        f"[infer_en] 完成 {len(results)} 字推理 | ckpt={resolved_ckpt} | device={device} "
        f"| mode={sampling_cfg.mode} steps={sampling_cfg.num_inference_steps} "
        f"| 耗时 {elapsed:.2f}s | 输出={out_path}{ttf_note}"
    )

    return results


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="纯英文 94-ASCII 字体推理与一键打包 TTF 脚本（em_baseline 固定基线对齐）。"
    )
    parser.add_argument("--config", default=None, help="项目 YAML 或 resolved_config.yaml 配置。")
    parser.add_argument("--ckpt-dir", default=None, help="权重目录（省略时从 --config 自动选最新权重）。")
    parser.add_argument(
        "--style-image",
        action="append",
        default=None,
        help="风格参考字图路径（可重复传入 1~3 张，须为同款 em_baseline 对齐字图）。",
    )
    parser.add_argument(
        "--style-ttf",
        default=None,
        help="可选：直接传入未见英文 TTF 作为风格源（自动按 em_baseline 渲染 --style-chars 作为参考图）。",
    )
    parser.add_argument(
        "--style-chars",
        default="Hgp",
        help="配合 --style-ttf 使用的参考字符（默认 'Hgp'，同时覆盖大写/小写下伸部/曲线结构）。",
    )
    parser.add_argument("--content-ttf", default=None, help="内容原型 TTF 路径（如 Inter-Regular.ttf）。")
    parser.add_argument("--content-dir", default=None, help="已预渲染好的 content/ 目录（与 --content-ttf 二选一）。")
    parser.add_argument(
        "--text",
        default=None,
        help="待生成英文字符串；省略时默认生成全部 94 个可打印 ASCII。",
    )
    parser.add_argument("--output-dir", required=True, help="输出单字 PNG 与预览图目录。")
    parser.add_argument("--baseline-ratio", type=float, default=DEFAULT_BASELINE_RATIO)
    parser.add_argument("--cap-height-ratio", type=float, default=DEFAULT_CAP_HEIGHT_RATIO)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mode", choices=("auto", "fast", "hd"), default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--solver", choices=("euler", "heun", "heun_fast"), default=None)
    parser.add_argument("--content-scale", type=float, default=None)
    parser.add_argument("--style-scale", type=float, default=None)
    parser.add_argument("--time-shift", type=float, default=None)
    parser.add_argument("--no-apg", action="store_true")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--no-comparison", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--pack-ttf",
        default=None,
        help="可选：推理结束后自动将全部输出字图打包为可安装 .ttf 的目标路径。",
    )
    parser.add_argument("--family-name", default="StyleDiffuserEN", help="打包 TTF 的家族名称。")
    parser.add_argument("--style-name", default="Regular", help="打包 TTF 的样式名称。")
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    infer_english_font(
        output_dir=args.output_dir,
        config_path=args.config,
        ckpt_dir=args.ckpt_dir,
        style_image_paths=args.style_image,
        style_ttf=args.style_ttf,
        style_chars=args.style_chars,
        content_ttf=args.content_ttf,
        content_dir=args.content_dir,
        text=args.text,
        baseline_ratio=args.baseline_ratio,
        cap_height_ratio=args.cap_height_ratio,
        batch_size=args.batch_size,
        mode=args.mode,
        steps=args.steps,
        solver=args.solver,
        content_scale=args.content_scale,
        style_scale=args.style_scale,
        time_shift=args.time_shift,
        use_apg=False if args.no_apg else None,
        use_ema=False if args.no_ema else None,
        seed=args.seed,
        device=args.device,
        save_comparison=not args.no_comparison,
        pack_ttf=args.pack_ttf,
        family_name=args.family_name,
        style_name=args.style_name,
    )


if __name__ == "__main__":
    main()
