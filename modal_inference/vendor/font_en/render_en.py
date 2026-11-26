from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from .charset import ASCII_94, CORE_ALNUM_62, DISTINCT_CASE_PAIRS, LOWER_26, UPPER_26
except ImportError:
    from charset import ASCII_94, CORE_ALNUM_62, DISTINCT_CASE_PAIRS, LOWER_26, UPPER_26

# 128x128 默认标准坐标系约定（与 to_ttf.py 严格共用）：
#   baseline_y = round(image_size * 0.75) = 96px（上留 96px 给大写/上伸部/花饰，下留 32px 给下伸部）
#   target_cap_ratio = 0.44 -> 128px 下标准大写 H 约 56px（对应 UPM=1000 中的 700 单位）
DEFAULT_IMAGE_SIZE: int = 128
DEFAULT_BASELINE_RATIO: float = 0.75
DEFAULT_CAP_HEIGHT_RATIO: float = 0.44
DEFAULT_EM_RATIO: float = 0.625  # 128px -> 80px = 1000 UPM（12.5 UPM/px；56px CapHeight = 700 UPM）
DEFAULT_MARGIN_RATIO: float = 0.05  # 128px -> 6px 四周安全留白

_CAP_PROBE_CHARS = ("H", "E", "I", "M", "T", "N")


@dataclass
class FontRenderCalibration:
    """单套英文字体的全局统一 em_baseline 渲染标尺（全 94 字严格共用）。"""

    font_path: str
    image_size: int
    baseline_y: int
    font_size: int
    cap_height_px: float
    em_px: float
    clamped: bool
    weight_axis: Optional[float] = None
    width_axis: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AllCapsReport:
    """纯全大写字体自动检测结果。"""

    is_all_caps: bool
    reason: str
    cmap_same_pairs: int
    mean_shape_iou: float
    mean_height_ratio: float


def load_cmap_mapping(font_path: str | Path) -> Optional[Dict[int, str]]:
    """读取字体 cmap（codepoint -> glyphName），损坏时降级返回 None。"""
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return None
    try:
        with TTFont(str(font_path), fontNumber=0, lazy=True) as tt:
            cmap = tt.getBestCmap()
            if cmap is None:
                return {}
            return dict(cmap)
    except Exception:
        return None


def load_cmap_codepoints(font_path: str | Path) -> Optional[Set[int]]:
    mapping = load_cmap_mapping(font_path)
    return None if mapping is None else set(mapping.keys())


def open_freetype_font(
    font_path: str | Path,
    size: int,
    weight_axis: Optional[float] = None,
    width_axis: Optional[float] = None,
) -> ImageFont.FreeTypeFont:
    """打开 TTF/OTF 字体；若为可变字体且指定了 wght/wdth 轴值，自动应用轴实例。"""
    font = ImageFont.truetype(str(font_path), size=max(8, int(size)))
    if weight_axis is None and width_axis is None:
        return font
    try:
        axes = font.get_variation_axes()
    except Exception:
        return font
    if not axes:
        return font

    axis_values: List[float] = []
    changed = False
    for axis in axes:
        raw_name = axis.get("name", b"")
        name = (
            raw_name.decode("utf-8", errors="ignore").lower()
            if isinstance(raw_name, bytes)
            else str(raw_name).lower()
        )
        lo = float(axis.get("minimum", 0.0))
        hi = float(axis.get("maximum", 1000.0))
        val = float(axis.get("default", lo))
        if weight_axis is not None and ("weight" in name or name == "wght"):
            val = min(max(float(weight_axis), lo), hi)
            changed = True
        elif width_axis is not None and ("width" in name or name == "wdth"):
            val = min(max(float(width_axis), lo), hi)
            changed = True
        axis_values.append(val)

    if changed:
        try:
            font.set_variation_by_axes(axis_values)
        except Exception:
            pass
    return font


def _render_ink_bounds_at_baseline(
    font: ImageFont.FreeTypeFont,
    char: str,
    probe_canvas: int = 384,
    baseline_y: int = 256,
    ink_threshold: int = 248,
) -> Optional[Tuple[int, int, int, int]]:
    """
    使用 Pillow anchor='ls' 在临时画布测量真实墨迹相对基线 (0, 0) 的精确包围盒：
    返回 (x_min, y_min_rel_baseline, x_max, y_max_rel_baseline)。
    其中 y < 0 表示基线以上，y > 0 表示基线以下。
    """
    anchor_x = probe_canvas // 3
    canvas = Image.new("L", (probe_canvas, probe_canvas), 255)
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((anchor_x, baseline_y), char, font=font, fill=0, anchor="ls")
    except Exception:
        return None
    arr = np.asarray(canvas)
    ys, xs = np.where(arr < ink_threshold)
    if ys.size == 0:
        return None
    return (
        int(xs.min()) - anchor_x,
        int(ys.min()) - baseline_y,
        int(xs.max()) - anchor_x,
        int(ys.max()) - baseline_y,
    )


def _render_normalized_mask(
    font: ImageFont.FreeTypeFont,
    char: str,
    box_size: int = 32,
) -> Optional[Tuple[np.ndarray, int, int]]:
    """渲染单字归一化二值掩码及原始宽高，用于全大写字形相似度对比。"""
    canvas = Image.new("L", (192, 192), 255)
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((64, 140), char, font=font, fill=0, anchor="ls")
    except Exception:
        return None
    arr = np.asarray(canvas)
    ys, xs = np.where(arr < 240)
    if ys.size == 0:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = Image.fromarray(arr[y0:y1, x0:x1], mode="L").resize(
        (box_size, box_size), Image.Resampling.BILINEAR
    )
    mask = np.asarray(crop) < 200
    return mask, (x1 - x0), (y1 - y0)


def detect_all_caps_font(
    font_path: str | Path,
    weight_axis: Optional[float] = None,
    width_axis: Optional[float] = None,
    cmap_same_threshold: int = 18,
    iou_threshold: float = 0.80,
    height_ratio_threshold: float = 0.88,
) -> AllCapsReport:
    """
    自动检测是否为纯全大写字体（All-Caps）：
    1. 若 cmap 中 >=18 组大小写字母指向同一 glyphName -> 直接判为纯全大写；
    2. 否则对比 9 组必不同形的字母对 (A/a, B/b, D/d, E/e, G/g, H/h, N/n, Q/q, R/r)：
       若归一化形状平均 IoU >= 0.80 且小写/大写高度比 >= 0.88 -> 判为纯全大写。
    """
    cmap = load_cmap_mapping(font_path)
    same_pairs = 0
    if cmap:
        for u, l in zip(UPPER_26, LOWER_26):
            ug = cmap.get(ord(u))
            lg = cmap.get(ord(l))
            if ug is not None and lg is not None and ug == lg:
                same_pairs += 1
        if same_pairs >= cmap_same_threshold:
            return AllCapsReport(
                is_all_caps=True,
                reason=f"cmap 大小写共享同一字形 ({same_pairs}/26)",
                cmap_same_pairs=same_pairs,
                mean_shape_iou=1.0,
                mean_height_ratio=1.0,
            )

    font = open_freetype_font(
        font_path, size=72, weight_axis=weight_axis, width_axis=width_axis
    )
    ious: List[float] = []
    h_ratios: List[float] = []
    for u, l in DISTINCT_CASE_PAIRS:
        ru = _render_normalized_mask(font, u)
        rl = _render_normalized_mask(font, l)
        if ru is None or rl is None:
            continue
        mu, _, hu = ru
        ml, _, hl = rl
        inter = int(np.logical_and(mu, ml).sum())
        union = int(np.logical_or(mu, ml).sum())
        if union == 0 or hu <= 0:
            continue
        ious.append(inter / union)
        h_ratios.append(hl / hu)

    if len(ious) < 5:
        return AllCapsReport(
            is_all_caps=True,
            reason=f"有效大小写对比样本不足 ({len(ious)}/9)",
            cmap_same_pairs=same_pairs,
            mean_shape_iou=0.0,
            mean_height_ratio=0.0,
        )

    mean_iou = float(np.mean(ious))
    mean_hr = float(np.mean(h_ratios))
    is_ac = (mean_iou >= iou_threshold) and (mean_hr >= height_ratio_threshold)
    return AllCapsReport(
        is_all_caps=is_ac,
        reason=(
            f"大小写轮廓高度一致 (IoU={mean_iou:.3f}, H_ratio={mean_hr:.3f})"
            if is_ac
            else "ok"
        ),
        cmap_same_pairs=same_pairs,
        mean_shape_iou=mean_iou,
        mean_height_ratio=mean_hr,
    )


def calibrate_font_em_baseline(
    font_path: str | Path,
    chars: Sequence[str] = ASCII_94,
    image_size: int = DEFAULT_IMAGE_SIZE,
    baseline_ratio: float = DEFAULT_BASELINE_RATIO,
    cap_height_ratio: float = DEFAULT_CAP_HEIGHT_RATIO,
    margin_ratio: float = DEFAULT_MARGIN_RATIO,
    weight_axis: Optional[float] = None,
    width_axis: Optional[float] = None,
) -> FontRenderCalibration:
    """
    为单套字体计算全 94 字共享的唯一缩放标尺：
    - 固定基线 baseline_y = round(image_size * baseline_ratio)（128px 下为 96px）
    - 以大写字母实测 CapHeight 锚定目标视觉高度（128px 下 ~56px），让不同 UPM 留白习惯的字体
      视觉字号一致；
    - 再依据全字集实测上伸部/下伸部/宽度做防裁切安全约束（带下限保护，防极端单标点把整套字缩太小）。
    """
    baseline_y = int(round(image_size * baseline_ratio))
    margin = max(4, int(round(image_size * margin_ratio)))
    target_cap = float(image_size) * float(cap_height_ratio)

    probe_size = 100
    probe_font = open_freetype_font(
        font_path, size=probe_size, weight_axis=weight_axis, width_axis=width_axis
    )
    cmap = load_cmap_codepoints(font_path)

    cap_heights: List[float] = []
    for ch in _CAP_PROBE_CHARS:
        if cmap is not None and ord(ch) not in cmap:
            continue
        b = _render_ink_bounds_at_baseline(probe_font, ch)
        if b is not None and b[1] < 0:
            cap_heights.append(float(-b[1]))

    ref_cap = float(np.median(cap_heights)) if cap_heights else 72.0
    ref_cap = max(ref_cap, 18.0)
    ideal_scale = target_cap / ref_cap

    # 扫描核心 62 字母数字 + 全字符集的最大上下左右跨度
    above_core: List[float] = []
    below_core: List[float] = []
    width_core: List[float] = []
    above_all: List[float] = []
    below_all: List[float] = []
    width_all: List[float] = []

    core_set = set(CORE_ALNUM_62)
    for ch in chars:
        if cmap is not None and ord(ch) not in cmap:
            continue
        b = _render_ink_bounds_at_baseline(probe_font, ch)
        if b is None:
            continue
        x0, y0, x1, y1 = b
        a = max(0.0, float(-y0))
        d = max(0.0, float(y1))
        w = max(1.0, float(x1 - x0 + 1))
        above_all.append(a)
        below_all.append(d)
        width_all.append(w)
        if ch in core_set:
            above_core.append(a)
            below_core.append(d)
            width_core.append(w)

    avail_above = float(baseline_y - margin)
    avail_below = float(image_size - baseline_y - margin)
    avail_width = float(image_size - 2 * margin)

    # 核心字母数字严格不裁切；全字符集取 98 分位防极个别夸张符号把整套字缩成芝麻
    max_a = max(
        max(above_core, default=ref_cap),
        float(np.percentile(above_all, 98)) if above_all else ref_cap,
    )
    max_d = max(
        max(below_core, default=1.0),
        float(np.percentile(below_all, 98)) if below_all else 1.0,
    )
    max_w = max(
        max(width_core, default=ref_cap),
        float(np.percentile(width_all, 98)) if width_all else ref_cap,
    )

    safe_scale = min(
        ideal_scale,
        avail_above / max(max_a, 1.0),
        avail_below / max(max_d, 1.0),
        avail_width / max(max_w, 1.0),
    )
    # 下限保护：不低于理想比例的 55%，避免极端花体整体过小
    min_allowed_scale = ideal_scale * 0.55
    final_scale = max(safe_scale, min_allowed_scale)
    clamped = final_scale < ideal_scale - 1e-4

    font_size = max(12, int(round(probe_size * final_scale)))
    actual_cap_px = ref_cap * (font_size / probe_size)
    em_px = float(image_size) * DEFAULT_EM_RATIO

    return FontRenderCalibration(
        font_path=str(Path(font_path).resolve()),
        image_size=int(image_size),
        baseline_y=baseline_y,
        font_size=font_size,
        cap_height_px=round(actual_cap_px, 2),
        em_px=round(em_px, 2),
        clamped=clamped,
        weight_axis=weight_axis,
        width_axis=width_axis,
    )


def render_char_em_baseline(
    font: ImageFont.FreeTypeFont,
    char: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    baseline_y: Optional[int] = None,
    ink_threshold: int = 250,
) -> Optional[Image.Image]:
    """
    以固定基线 baseline_y 渲染单字符为白底黑字 RGB 正方形图：
    - 垂直方向严格锁定基线（绝不做单字垂直居中），完整保留大写/x-height/上下伸部/标点高度；
    - 水平方向按真实墨迹包围盒精确居中（斜体/花体探出部分也不会偏出画布边缘）。
    """
    if baseline_y is None:
        baseline_y = int(round(image_size * DEFAULT_BASELINE_RATIO))

    # 在更大的草稿画布上以 anchor='ls' 渲染，避免斜体负左轴承或长拖尾被提前截断
    pad_x = image_size
    pad_y = image_size // 2
    scratch_w = image_size + 2 * pad_x
    scratch_h = image_size + 2 * pad_y
    scratch_baseline = pad_y + baseline_y

    scratch = Image.new("L", (scratch_w, scratch_h), 255)
    draw = ImageDraw.Draw(scratch)
    try:
        draw.text((pad_x, scratch_baseline), char, font=font, fill=0, anchor="ls")
    except Exception:
        return None

    arr = np.asarray(scratch)
    ys, xs = np.where(arr < ink_threshold)
    if ys.size == 0:
        return None

    # 水平按真实墨迹中心对齐到画布水平中线；垂直方向严格按 pad_y 裁切，保证基线钉死在 baseline_y
    ink_cx = 0.5 * (float(xs.min()) + float(xs.max()))
    crop_x0 = int(round(ink_cx - image_size / 2.0))
    crop_x0 = max(0, min(scratch_w - image_size, crop_x0))
    crop_y0 = pad_y

    cropped = arr[crop_y0 : crop_y0 + image_size, crop_x0 : crop_x0 + image_size]
    if cropped.min() >= ink_threshold:
        return None
    return Image.fromarray(cropped, mode="L").convert("RGB")


def render_sentence_preview(
    glyph_images: Dict[str, Image.Image],
    text: str = "Hamburgefontsiv 0123456789 !?.,'\"-@",
    image_size: int = DEFAULT_IMAGE_SIZE,
    baseline_y: Optional[int] = None,
    letter_spacing: int = 6,
    space_width: int = 34,
    draw_baseline_guide: bool = True,
) -> Image.Image:
    """
    将单字图按水平墨迹紧凑拼接成一整行英文预览图（严格保持同一 y 基线），
    用于肉眼一眼检查基线齐不齐、大小写比例和字重是否均匀。
    """
    if baseline_y is None:
        baseline_y = int(round(image_size * DEFAULT_BASELINE_RATIO))

    crops: List[Optional[Image.Image]] = []
    total_w = 24
    for ch in text:
        if ch == " ":
            crops.append(None)
            total_w += space_width
            continue
        img = glyph_images.get(ch)
        if img is None:
            continue
        gray = np.asarray(img.convert("L"))
        _, xs = np.where(gray < 245)
        if xs.size == 0:
            continue
        x0 = max(0, int(xs.min()) - 1)
        x1 = min(gray.shape[1], int(xs.max()) + 2)
        strip = img.convert("RGB").crop((x0, 0, x1, image_size))
        crops.append(strip)
        total_w += strip.width + letter_spacing
    total_w += 24

    canvas = Image.new("RGB", (max(total_w, image_size), image_size), (255, 255, 255))
    if draw_baseline_guide:
        d = ImageDraw.Draw(canvas)
        d.line([(0, baseline_y), (canvas.width, baseline_y)], fill=(235, 200, 200), width=1)

    cur_x = 24
    for strip in crops:
        if strip is None:
            cur_x += space_width
            continue
        # 保留浅红参考线可见，采用最小值融合（白底不遮挡）
        dst = np.asarray(canvas.crop((cur_x, 0, cur_x + strip.width, image_size)))
        src = np.asarray(strip)
        blended = np.minimum(dst, src)
        canvas.paste(Image.fromarray(blended, mode="RGB"), (cur_x, 0))
        cur_x += strip.width + letter_spacing

    return canvas
