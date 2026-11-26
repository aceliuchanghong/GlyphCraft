from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Sequence, Tuple

from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF

from ..config import DataConfig

_PIL_RESAMPLE = {
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "nearest": Image.Resampling.NEAREST,
}


def has_transparency(img: Image.Image) -> bool:
    if img.mode in ("RGBA", "LA", "PA"):
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def to_rgb_with_background(
    img: Image.Image,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """将任意模式图像转为 RGB；若含透明通道则合成到指定底色，避免透明区域变黑。"""
    if has_transparency(img):
        rgba = img.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (*background_color, 255))
        return Image.alpha_composite(canvas, rgba).convert("RGB")
    if img.mode != "RGB":
        return img.convert("RGB")
    return img.copy()


def load_rgb_image(
    path: str | Path,
    background_color: Tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    with Image.open(path) as img:
        return to_rgb_with_background(img, background_color=background_color)


class SquareResizeOrPad:
    """将 RGB 图像缩放到正方形分辨率，支持直接拉伸（stretch）或保持比例居中补边（pad）。"""

    def __init__(
        self,
        resolution: int,
        resize_mode: str = "pad",
        pad_ratio: float = 0.0,
        interpolation: str = "bilinear",
        background_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.resolution = int(resolution)
        self.resize_mode = resize_mode
        self.pad_ratio = float(pad_ratio)
        self.resample = _PIL_RESAMPLE[interpolation]
        self.background_color = tuple(int(c) for c in background_color)

    def __call__(self, img: Image.Image) -> Image.Image:
        rgb = to_rgb_with_background(img, background_color=self.background_color)
        if self.resize_mode == "stretch":
            if rgb.size == (self.resolution, self.resolution):
                return rgb
            return rgb.resize((self.resolution, self.resolution), resample=self.resample)

        inner = max(1, int(round(self.resolution * (1.0 - 2.0 * self.pad_ratio))))
        w, h = rgb.size
        if w <= 0 or h <= 0:
            return Image.new("RGB", (self.resolution, self.resolution), self.background_color)

        scale = inner / float(max(w, h))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = rgb.resize((new_w, new_h), resample=self.resample)

        canvas = Image.new("RGB", (self.resolution, self.resolution), self.background_color)
        offset_x = (self.resolution - new_w) // 2
        offset_y = (self.resolution - new_h) // 2
        canvas.paste(resized, (offset_x, offset_y))
        return canvas


def build_image_transform(
    data_cfg: DataConfig,
    normalize: bool = True,
) -> transforms.Compose:
    steps = [
        SquareResizeOrPad(
            resolution=data_cfg.resolution,
            resize_mode=data_cfg.resize_mode,
            pad_ratio=data_cfg.pad_ratio,
            interpolation=data_cfg.interpolation,
            background_color=data_cfg.background_color,
        ),
        transforms.ToTensor(),
    ]
    if normalize:
        steps.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    return transforms.Compose(steps)


def apply_paired_color_jitter(
    tensors_norm: Sequence[torch.Tensor],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    hue_range: float = 0.25,
    sat_range: float = 0.25,
    val_range: float = 0.15,
) -> list[torch.Tensor]:
    """
    成对色彩增强：对同一条训练样本的所有 style 参考图与 target 目标图施加完全一致的随机
    色相/饱和度/亮度偏移（并保护纯背景色不被亮度偏移染灰），大幅提升彩色资产的未见色板泛化。
    输入/输出均为 [-1, 1] 归一化张量 [3, H, W]。
    """
    hue_delta = random.uniform(-hue_range, hue_range) if hue_range > 0 else 0.0
    sat_factor = random.uniform(max(0.1, 1.0 - sat_range), 1.0 + sat_range) if sat_range > 0 else 1.0
    val_factor = random.uniform(max(0.5, 1.0 - val_range), 1.0 + val_range) if val_range > 0 else 1.0

    bg = torch.tensor(
        [c / 255.0 for c in background_color], dtype=torch.float32
    ).view(3, 1, 1)

    out: list[torch.Tensor] = []
    for t in tensors_norm:
        img01 = (t.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        bg_mask = (img01 - bg).abs().amax(dim=0, keepdim=True) < (6.0 / 255.0)
        aug = img01
        if abs(hue_delta) > 1e-4:
            aug = TF.adjust_hue(aug, hue_delta)
        if abs(sat_factor - 1.0) > 1e-4:
            aug = TF.adjust_saturation(aug, sat_factor)
        if abs(val_factor - 1.0) > 1e-4:
            aug = TF.adjust_brightness(aug, val_factor)
        aug = torch.where(bg_mask, img01, aug).clamp(0.0, 1.0)
        out.append((aug * 2.0 - 1.0).to(dtype=t.dtype))
    return out


def _exact_int_shift(
    x: torch.Tensor,
    dx: int,
    dy: int,
    fill_vals: list[float],
) -> torch.Tensor:
    """零插值模糊的精确整数像素平移（保持 1px 锐利边缘 100% 无损）。"""
    if dx == 0 and dy == 0:
        return x
    c, h, w = x.shape
    bg = torch.tensor(fill_vals, dtype=x.dtype, device=x.device).view(c, 1, 1)
    out = bg.expand(c, h, w).clone()
    src_x0 = max(0, -dx)
    src_x1 = min(w, w - dx)
    dst_x0 = max(0, dx)
    dst_x1 = min(w, w + dx)
    src_y0 = max(0, -dy)
    src_y1 = min(h, h - dy)
    dst_y0 = max(0, dy)
    dst_y1 = min(h, h + dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[:, dst_y0:dst_y1, dst_x0:dst_x1] = x[:, src_y0:src_y1, src_x0:src_x1]
    return out


def apply_paired_spatial_aug(
    tensors_norm: Sequence[torch.Tensor],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    hflip_prob: float = 0.0,
    scale: float = 1.0,
    shift_ratio: float = 0.03,
    interpolation: str = "bilinear",
) -> list[torch.Tensor]:
    """
    成对空间几何增强：对一组 [-1, 1] 张量施加一致的随机水平翻转 + 可选等比缩放 + 整数像素平移，
    空白外围严格用 background_color 填充。
    当 scale == 1.0 时采用精确整数切片平移，彻底避免双线性重采样造成的 1px 边缘发糊。
    """
    do_hflip = hflip_prob > 0.0 and random.random() < float(hflip_prob)
    _, h, w = tensors_norm[0].shape
    max_dx = int(round(w * float(shift_ratio)))
    max_dy = int(round(h * float(shift_ratio)))
    dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
    dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0

    fill_vals = [(float(c) / 127.5) - 1.0 for c in background_color]
    interp_mode = (
        TF.InterpolationMode.NEAREST
        if interpolation == "nearest"
        else TF.InterpolationMode.BILINEAR
    )
    sc = float(scale)
    need_scale = abs(sc - 1.0) > 1e-4

    out: list[torch.Tensor] = []
    for t in tensors_norm:
        x = t
        if do_hflip:
            x = TF.hflip(x)
        if need_scale:
            x = TF.affine(
                x,
                angle=0.0,
                translate=[dx, dy],
                scale=sc,
                shear=[0.0, 0.0],
                interpolation=interp_mode,
                fill=fill_vals,
            )
        elif dx != 0 or dy != 0:
            x = _exact_int_shift(x, dx=dx, dy=dy, fill_vals=fill_vals)
        out.append(x)
    return out


def apply_paired_morph_shear_aspect(
    tensors_norm: Sequence[torch.Tensor],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    max_shear_deg: float = 8.0,
    aspect_range: float = 0.10,
) -> list[torch.Tensor]:
    """
    v3 成对斜切/字面比例微几何风格增广（仅同步作用于 target + style_refs，保持 content 原图不动）：
    - shear_y 模拟仿宋横画左低右高「抗肩」斜势，shear_x 模拟斜体/行书倾角，(sx, sy) 模拟中宫瘦长/宽扁；
    - 用纯背景边界 padding 避免黑边伪影，并按倾角自适应留边防裁切。
    """
    if not tensors_norm:
        return []
    shx_rad = math.radians(random.uniform(-max_shear_deg, max_shear_deg)) if max_shear_deg > 0 else 0.0
    shy_rad = math.radians(random.uniform(-max_shear_deg, max_shear_deg)) if max_shear_deg > 0 else 0.0
    shx = math.tan(shx_rad)
    shy = math.tan(shy_rad)

    sx = random.uniform(1.0 - aspect_range, 1.0 + aspect_range) if aspect_range > 0 else 1.0
    sy = random.uniform(1.0 - aspect_range, 1.0 + aspect_range) if aspect_range > 0 else 1.0
    # 轻微收缩保安全边距，避免斜切后字角越界
    safety = 1.0 / (1.0 + 0.35 * (abs(shx) + abs(shy)))
    inv_sx = 1.0 / max(sx * safety, 0.70)
    inv_sy = 1.0 / max(sy * safety, 0.70)

    theta = torch.tensor(
        [[[inv_sx, -shx * inv_sy, 0.0], [-shy * inv_sx, inv_sy, 0.0]]],
        dtype=torch.float32,
        device=tensors_norm[0].device,
    )
    _, h, w = tensors_norm[0].shape
    grid = F.affine_grid(theta, size=(1, 1, h, w), align_corners=False)

    bg = background_to_norm_tensor(
        background_color, device=tensors_norm[0].device, dtype=torch.float32
    )  # [1, 3, 1, 1]
    out: list[torch.Tensor] = []
    for t in tensors_norm:
        x = t.float().unsqueeze(0)
        # 在偏移零点后执行 grid_sample（zeros padding 等价于严格填充 background_color，无边界拖尾）
        warped = F.grid_sample(
            x - bg,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        ) + bg
        out.append(warped.squeeze(0).clamp(-1.0, 1.0).to(dtype=t.dtype))
    return out


def apply_paired_pixel_grid_quantize(
    tensors_norm: Sequence[torch.Tensor],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    grid_sizes: Sequence[int] = (24, 32, 40, 48),
) -> list[torch.Tensor]:
    """
    v3 成对离散像素格量化风格增广（仅同步作用于 target + style_refs，保持 content 平滑原图不动）：
    - 用区域均值 + 峰值反推前景纯色（完整保留单色细线、Duotone 浅灰与 RGB 彩块，彻底消灭灰晕抗锯齿）；
    - 最近邻上采样还原为清晰锐利的像素阶梯折线（24/32/40/48px 离散格），零样本泛化 pixelarticons 等像素风。
    """
    if not tensors_norm:
        return []
    g = int(random.choice(list(grid_sizes)))
    bg = torch.tensor(
        [c / 255.0 for c in background_color],
        dtype=torch.float32,
        device=tensors_norm[0].device,
    ).view(1, 3, 1, 1)

    out: list[torch.Tensor] = []
    for t in tensors_norm:
        _, h, w = t.shape
        img = (t.float() * 0.5 + 0.5).clamp(0.0, 1.0).unsqueeze(0)  # [1, 3, H, W]
        ink = (img - bg).abs().amax(dim=1, keepdim=True)  # [1, 1, H, W]
        img_avg = F.adaptive_avg_pool2d(img, (g, g))
        ink_avg = F.adaptive_avg_pool2d(ink, (g, g))
        ink_max = F.adaptive_max_pool2d(ink, (g, g))

        is_fg = (ink_avg > 0.22) & (ink_max > 0.25)
        alpha = (ink_avg / ink_max.clamp_min(1e-3)).clamp(0.22, 1.0)
        pure_fg = (bg + (img_avg - bg) / alpha).clamp(0.0, 1.0)
        grid_rgb = torch.where(is_fg, pure_fg, bg.expand_as(pure_fg))
        pixelated = F.interpolate(grid_rgb, size=(h, w), mode="nearest").squeeze(0)
        out.append((pixelated * 2.0 - 1.0).clamp(-1.0, 1.0).to(dtype=t.dtype))
    return out


def _ink_mask(
    tensor_norm: torch.Tensor,
    background_color: Tuple[int, int, int],
    tol: float = 24.0 / 255.0,
) -> torch.Tensor:
    """前景墨迹掩码 [H, W]：与背景色的最大通道差超过 tol。"""
    img01 = (tensor_norm.detach().float() * 0.5 + 0.5).clamp(0.0, 1.0)
    bg = torch.tensor(
        [c / 255.0 for c in background_color],
        dtype=img01.dtype,
        device=img01.device,
    ).view(3, 1, 1)
    return (img01 - bg).abs().amax(dim=0) > tol


def _thick_stroke_ratio(
    tensor_norm: torch.Tensor,
    background_color: Tuple[int, int, int],
) -> float:
    """
    1px 腐蚀后仍存活的墨迹占比。
    3px 及以上笔画明显 > 0，1~2px 游丝会被 3×3 最小值滤波抹掉并返回 0。
    """
    ink = _ink_mask(tensor_norm, background_color)
    ink_sum = float(ink.sum().item())
    if ink_sum < 12.0:
        return 0.0
    ink_f = ink.float().view(1, 1, int(ink.shape[0]), int(ink.shape[1]))
    eroded = -F.max_pool2d(-ink_f, kernel_size=3, stride=1, padding=1)
    return float(eroded.sum().item()) / ink_sum


def _sample_cut_spans(
    length: int,
    count: int,
    gap: int,
    lo: float,
    hi: float,
    rng: random.Random,
) -> list[Tuple[int, int]]:
    """在 [lo, hi] 比例带内抽互不重叠的切缝区间，区间之间至少留 4px。"""
    lo_i = int(round(length * lo))
    hi_i = int(round(length * hi))
    if hi_i - lo_i <= gap + 1:
        return []
    spans: list[Tuple[int, int]] = []
    for _ in range(max(0, int(count))):
        placed = False
        for _try in range(8):
            start = rng.randint(lo_i, hi_i - gap)
            end = start + gap
            if all(end + 4 <= a or start >= b + 4 for a, b in spans):
                spans.append((start, end))
                placed = True
                break
        if not placed:
            break
    return spans


def apply_paired_stencil_cut(
    tensors_norm: Sequence[torch.Tensor],
    background_color: Tuple[int, int, int] = (255, 255, 255),
    gap_choices: Sequence[int] = (2, 3),
    thick_ratio_min: float = 0.18,
    rng: random.Random | None = None,
) -> list[torch.Tensor]:
    """
    成对模板切缝（仅由调用方传入 target + style_refs，content 不进入本函数）：
    - 在字身中段同步切 1~2 条水平白缝，并以一定概率再切 1 条垂直白缝（缝宽 2~3px）；
    - 只把缝内像素写成背景色，不平移、不缩放，em_baseline 坐标保持不动；
    - 1~2px 游丝（腐蚀后几乎不存活）整张跳过，避免把细花体笔画擦掉。
    未发生改动的张量原样返回，调用方可用身份比较判断是否需要重算特征。
    """
    if not tensors_norm:
        return []
    picker = rng if rng is not None else random
    gaps = [int(g) for g in gap_choices if int(g) >= 1]
    if not gaps:
        return list(tensors_norm)

    _, height, width = tensors_norm[0].shape
    gap = int(picker.choice(gaps))
    n_h = 1 if picker.random() < 0.55 else 2
    n_v = 1 if picker.random() < 0.65 else 0
    # 水平缝落在大写字身（约 y=0.34H~0.68H，128px 下覆盖基线 y=96 以上的字腔），不挪动基线。
    h_cuts = _sample_cut_spans(height, n_h, gap, 0.34, 0.68, picker)
    v_cuts = _sample_cut_spans(width, n_v, gap, 0.32, 0.68, picker)
    if not h_cuts and not v_cuts:
        return list(tensors_norm)

    bg_vals = [(float(c) / 127.5) - 1.0 for c in background_color]
    out: list[torch.Tensor] = []
    changed = False
    for tensor in tensors_norm:
        if _thick_stroke_ratio(tensor, background_color) < float(thick_ratio_min):
            out.append(tensor)
            continue
        canvas = tensor.clone()
        bg = torch.tensor(bg_vals, dtype=canvas.dtype, device=canvas.device).view(3, 1, 1)
        for y0, y1 in h_cuts:
            canvas[:, y0:y1, :] = bg
        for x0, x1 in v_cuts:
            canvas[:, :, x0:x1] = bg
        out.append(canvas)
        changed = True
    if not changed:
        return list(tensors_norm)
    return out


def background_to_norm_tensor(
    background_color: Tuple[int, int, int] = (255, 255, 255),
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """将 [0, 255] 背景色映射为 [-1, 1] 归一化空间张量，形状 [1, 3, 1, 1]。"""
    vals = [(float(c) / 127.5) - 1.0 for c in background_color]
    return torch.tensor(vals, device=device, dtype=dtype).view(1, 3, 1, 1)
