from __future__ import annotations

import argparse
import io
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree as ET

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    from .charset import stem_to_char
    from .render_en import DEFAULT_BASELINE_RATIO, DEFAULT_EM_RATIO
except ImportError:
    from charset import stem_to_char
    from render_en import DEFAULT_BASELINE_RATIO, DEFAULT_EM_RATIO

RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _require_ttf_deps():
    try:
        import vtracer
    except ImportError as exc:
        raise ImportError(
            "缺少 vtracer 依赖，请先运行: python scripts/sync_env.py --extra font --extra icons"
        ) from exc
    try:
        from fontTools.fontBuilder import FontBuilder
        from fontTools.pens.areaPen import AreaPen
        from fontTools.pens.cu2quPen import Cu2QuPen
        from fontTools.pens.recordingPen import RecordingPen
        from fontTools.pens.reverseContourPen import ReverseContourPen
        from fontTools.pens.transformPen import TransformPen
        from fontTools.pens.ttGlyphPen import TTGlyphPen
        from fontTools.svgLib.path import parse_path
    except ImportError as exc:
        raise ImportError(
            "缺少 fonttools 依赖，请先运行: python scripts/sync_env.py --extra font --extra icons"
        ) from exc
    return (
        vtracer,
        FontBuilder,
        AreaPen,
        Cu2QuPen,
        RecordingPen,
        ReverseContourPen,
        TransformPen,
        TTGlyphPen,
        parse_path,
    )


_TRANSLATE_RE = re.compile(
    r"translate\(\s*([+-]?\d+(?:\.\d+)?)(?:[\s,]+([+-]?\d+(?:\.\d+)?))?\s*\)"
)


def _extract_svg_paths(svg_str: str) -> List[Tuple[str, float, float]]:
    """从 vtracer 输出的 SVG 中解析所有 <path d='...'> 及其可选 translate(tx, ty) 偏移。"""
    root = ET.fromstring(svg_str)
    out: List[Tuple[str, float, float]] = []
    for el in root.iter():
        if not el.tag.endswith("path"):
            continue
        d = el.attrib.get("d", "").strip()
        if not d:
            continue
        dx, dy = 0.0, 0.0
        tf = el.attrib.get("transform", "")
        if tf:
            m = _TRANSLATE_RE.search(tf)
            if m:
                dx = float(m.group(1))
                dy = float(m.group(2)) if m.group(2) is not None else 0.0
        out.append((d, dx, dy))
    return out


def _clean_binary_for_vectorize(
    img: Image.Image,
    upscale: int = 4,
    threshold: int = 160,
    pixel_mode: bool = False,
) -> Tuple[bytes, int, int, Optional[Tuple[int, int]]]:
    """
    将白底黑字单字图上采样并转为干净白底黑字 PNG 字节流，返回 (png_bytes, w, h, ink_x_range)。
    """
    gray = img.convert("L")
    resample = Image.Resampling.NEAREST if pixel_mode else Image.Resampling.BICUBIC
    w, h = gray.width * upscale, gray.height * upscale
    up = gray.resize((w, h), resample) if upscale > 1 else gray

    arr = np.asarray(up)
    # 干净二值化，消除扩散模型残余浅灰底噪，确保 vtracer 轮廓边缘锐利无杂点
    binary = np.where(arr < threshold, 0, 255).astype(np.uint8)
    _, xs = np.where(binary == 0)
    if xs.size == 0:
        return b"", w, h, None

    rgb = Image.fromarray(binary, mode="L").convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, format="PNG")
    return buf.getvalue(), w, h, (int(xs.min()), int(xs.max()))


def _glyph_png_to_ttglyph(
    img: Image.Image,
    upm: int = 1000,
    baseline_ratio: float = DEFAULT_BASELINE_RATIO,
    em_ratio: float = DEFAULT_EM_RATIO,
    upscale: int = 4,
    threshold: int = 160,
    sidebearing: int = 55,
    min_advance: int = 240,
    filter_speckle: int = 6,
    curve_mode: str = "spline",
) -> Tuple[object, int, int]:
    """
    将单张 em_baseline PNG 转换为 TrueType Glyph 与 (advance_width, lsb)。
    严格按 baseline_ratio 将基线映射到 TTF y=0。
    """
    (
        vtracer,
        _,
        AreaPen,
        Cu2QuPen,
        RecordingPen,
        ReverseContourPen,
        TransformPen,
        TTGlyphPen,
        parse_path,
    ) = _require_ttf_deps()

    pixel_mode = curve_mode == "polygon"
    png_bytes, up_w, up_h, x_range = _clean_binary_for_vectorize(
        img,
        upscale=upscale,
        threshold=threshold,
        pixel_mode=pixel_mode,
    )
    if x_range is None or not png_bytes:
        pen = TTGlyphPen(None)
        return pen.glyph(), min_advance, 0

    svg_str = vtracer.convert_raw_image_to_svg(
        png_bytes,
        img_format="png",
        colormode="binary",
        mode=curve_mode,
        filter_speckle=filter_speckle,
        corner_threshold=60,
        length_threshold=3.5,
        splice_threshold=45,
        path_precision=3,
    )

    svg_paths = _extract_svg_paths(svg_str)
    if not svg_paths:
        pen = TTGlyphPen(None)
        return pen.glyph(), min_advance, 0

    # 坐标线性换算：
    #   x_ttf = (x_local + dx - x_min_px) * scale + sidebearing
    #   y_ttf = (baseline_up_y - (y_local + dy)) * scale  =>  基线 y_svg == baseline_up_y 精确落到 y_ttf == 0
    scale = float(upm) / (float(em_ratio) * float(up_h))
    baseline_up_y = float(baseline_ratio) * float(up_h)
    x_min_px, x_max_px = x_range

    # 先录制经仿射变换后的三次贝塞尔路径，检验总有向面积以修正 TrueType 绕向（外轮廓顺时针 => 面积 < 0）
    rec = RecordingPen()
    for d_str, dx, dy in svg_paths:
        tx = float(sidebearing) + (dx - float(x_min_px)) * scale
        ty = (baseline_up_y - dy) * scale
        tpen = TransformPen(rec, (scale, 0.0, 0.0, -scale, tx, ty))
        parse_path(d_str, tpen)

    area_pen = AreaPen()
    rec.replay(area_pen)
    needs_reverse = area_pen.value > 0

    tt_pen = TTGlyphPen(None)
    cu2qu_pen = Cu2QuPen(tt_pen, max_err=1.0, reverse_direction=False)
    if needs_reverse:
        rec.replay(ReverseContourPen(cu2qu_pen))
    else:
        rec.replay(cu2qu_pen)

    glyph = tt_pen.glyph()
    ink_w_upm = int(round((x_max_px - x_min_px + 1) * scale))
    advance_width = max(min_advance, ink_w_upm + 2 * int(sidebearing))
    lsb = int(sidebearing)
    return glyph, advance_width, lsb


def build_ttf_from_glyph_dir(
    input_dir: str | Path,
    output_ttf: str | Path,
    family_name: str = "StyleDiffuserEN",
    style_name: str = "Regular",
    upm: int = 1000,
    ascent: int = 850,
    descent: int = -250,
    baseline_ratio: float = DEFAULT_BASELINE_RATIO,
    em_ratio: float = DEFAULT_EM_RATIO,
    upscale: int = 4,
    threshold: int = 160,
    sidebearing: int = 55,
    space_width: int = 300,
    filter_speckle: int = 6,
    curve_mode: str = "spline",
) -> Path:
    """
    将 94-ASCII 生成图目录（u0021.png ~ u007E.png）批量矢量化并打包为标准可安装 TrueType (.ttf)。
    """
    (_, FontBuilder, *_, TTGlyphPen, _) = _require_ttf_deps()

    in_path = Path(input_dir)
    if not in_path.is_dir():
        raise FileNotFoundError(f"输入字图目录不存在: {in_path}")

    char_to_path: Dict[str, Path] = {}
    for p in sorted(in_path.iterdir(), key=lambda x: x.name):
        if not p.is_file() or p.suffix.lower() not in RASTER_EXTENSIONS:
            continue
        ch = stem_to_char(p.stem)
        if ch is None:
            continue
        char_to_path[ch] = p

    if not char_to_path:
        raise FileNotFoundError(f"未在 '{in_path}' 找到任何可识别的 ASCII 字图（如 u0041.png）。")

    glyph_order: List[str] = [".notdef", "space"]
    cmap: Dict[int, str] = {0x20: "space"}
    glyphs: Dict[str, object] = {}
    metrics: Dict[str, Tuple[int, int]] = {}

    # 构造空 .notdef 与 space 字形
    empty_pen = TTGlyphPen(None)
    glyphs[".notdef"] = empty_pen.glyph()
    metrics[".notdef"] = (500, 0)

    space_pen = TTGlyphPen(None)
    glyphs["space"] = space_pen.glyph()
    metrics["space"] = (int(space_width), 0)

    for ch in tqdm(sorted(char_to_path.keys(), key=ord), desc="Vectorizing glyphs to TTF"):
        gname = f"uni{ord(ch):04X}"
        with Image.open(char_to_path[ch]) as raw:
            img = raw.convert("RGB")
        glyph, adv, lsb = _glyph_png_to_ttglyph(
            img=img,
            upm=upm,
            baseline_ratio=baseline_ratio,
            em_ratio=em_ratio,
            upscale=upscale,
            threshold=threshold,
            sidebearing=sidebearing,
            filter_speckle=filter_speckle,
            curve_mode=curve_mode,
        )
        glyph_order.append(gname)
        cmap[ord(ch)] = gname
        glyphs[gname] = glyph
        metrics[gname] = (adv, lsb)

    fb = FontBuilder(int(upm), isTTF=True)
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=int(ascent), descent=int(descent))
    fb.setupNameTable(
        {
            "familyName": family_name,
            "styleName": style_name,
            "fullName": f"{family_name} {style_name}",
            "psName": f"{family_name.replace(' ', '')}-{style_name.replace(' ', '')}",
        }
    )
    fb.setupOS2(
        sTypoAscender=int(ascent),
        sTypoDescender=int(descent),
        usWinAscent=int(ascent),
        usWinDescent=abs(int(descent)),
        sxHeight=500,
        sCapHeight=700,
    )
    fb.setupPost()

    out_path = Path(output_ttf)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fb.save(str(out_path))
    print(f"已成功打包 TTF 字体: {out_path}（包含 {len(char_to_path)} 个可见字形 + 空格）")
    return out_path


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(
        description="英文生成字图打包工具：vtracer 矢量化 + fontTools 固定基线打包为可安装 .ttf 字体。"
    )
    parser.add_argument("--input-dir", required=True, help="infer_en.py 输出的单字 PNG 目录（含 u0021.png~u007E.png）。")
    parser.add_argument("--output-ttf", required=True, help="输出 .ttf 文件路径。")
    parser.add_argument("--family-name", default="StyleDiffuserEN", help="字体家族名（默认 StyleDiffuserEN）。")
    parser.add_argument("--style-name", default="Regular", help="字体样式名（默认 Regular）。")
    parser.add_argument("--upm", type=int, default=1000, help="Units Per Em（默认 1000）。")
    parser.add_argument("--baseline-ratio", type=float, default=DEFAULT_BASELINE_RATIO, help="与训练一致的基线比例（默认 0.75）。")
    parser.add_argument("--em-ratio", type=float, default=DEFAULT_EM_RATIO, help="与训练一致的 Em 比例（默认 0.625）。")
    parser.add_argument("--upscale", type=int, default=4, help="矢量化前上采样倍数（默认 4，曲线更顺滑）。")
    parser.add_argument("--threshold", type=int, default=160, help="灰度二值化阈值 0~255（默认 160）。")
    parser.add_argument("--sidebearing", type=int, default=55, help="左右字距留白 UPM 单位（默认 55）。")
    parser.add_argument("--space-width", type=int, default=300, help="空格 advanceWidth（默认 300）。")
    parser.add_argument("--filter-speckle", type=int, default=6, help="vtracer 去噪点面积阈值（默认 6）。")
    parser.add_argument(
        "--curve-mode",
        choices=("spline", "polygon"),
        default="spline",
        help="曲线拟合模式：普通字体用 spline，像素体可选 polygon（默认 spline）。",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main() -> None:
    args = parse_args()
    build_ttf_from_glyph_dir(
        input_dir=args.input_dir,
        output_ttf=args.output_ttf,
        family_name=args.family_name,
        style_name=args.style_name,
        upm=args.upm,
        baseline_ratio=args.baseline_ratio,
        em_ratio=args.em_ratio,
        upscale=args.upscale,
        threshold=args.threshold,
        sidebearing=args.sidebearing,
        space_width=args.space_width,
        filter_speckle=args.filter_speckle,
        curve_mode=args.curve_mode,
    )


if __name__ == "__main__":
    main()
