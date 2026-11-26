from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .vision_backbone import FrozenVisionBackbone


def _rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """FP32 内部精确归一回转输入精度（杜绝 autocast(bf16) 下 Q/K 泄漏为 FP32 阻塞 FlashAttention-2）。"""
    out = F.rms_norm(
        x.float(),
        (x.shape[-1],),
        weight.float() if weight is not None else None,
        eps,
    )
    return out.to(dtype=x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _rms_norm(x, self.weight, self.eps)


class SinusoidalScalarEmbedder(nn.Module):
    """连续标量嵌入（用于流时间 t、Shortcut 步长 d、嵌入式引导强度 w）。"""

    def __init__(self, hidden_size: int, freq_dim: int = 256, max_period: float = 10000.0):
        super().__init__()
        self.freq_dim = freq_dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def _sinusoidal(self, val: torch.Tensor) -> torch.Tensor:
        half = self.freq_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, dtype=torch.float32, device=val.device)
            / max(half, 1)
        )
        args = val.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.freq_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, val: torch.Tensor) -> torch.Tensor:
        emb = self._sinusoidal(val)
        return self.mlp(emb.to(dtype=self.mlp[0].weight.dtype))


class RotaryEmbedding3D(nn.Module):
    """
    3D Rotary Position Embedding (y, x, slot)：
    - (y, x): 空间坐标（Target 与 Content 共享严格对齐的 2D 网格坐标）
    - slot:   角色槽位偏移（Target=0, Content=1, StyleLatent=2）
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        if head_dim < 16 or head_dim % 8 != 0:
            raise ValueError(f"3D RoPE 要求 head_dim >= 16 且为 8 的倍数，收到 {head_dim}")
        self.head_dim = head_dim
        dim_slot = max(4, ((head_dim // 4) // 2) * 2)
        rem = head_dim - dim_slot
        dim_y = (rem // 4) * 2
        dim_x = rem - dim_y
        assert dim_y % 2 == 0 and dim_x % 2 == 0 and dim_slot % 2 == 0
        self.dims = (dim_y, dim_x, dim_slot)

        for name, d in zip(("y", "x", "s"), self.dims):
            inv_freq = 1.0 / (
                theta ** (torch.arange(0, d, 2, dtype=torch.float32) / float(d))
            )
            self.register_buffer(f"inv_freq_{name}", inv_freq, persistent=False)

    def build_cos_sin(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if coords.ndim == 2:
            coords = coords.unsqueeze(0)
        coords_f = coords.float()
        half_freqs = []
        for idx, name in enumerate(("y", "x", "s")):
            inv_f = getattr(self, f"inv_freq_{name}").to(device=coords_f.device)
            ang = coords_f[..., idx : idx + 1] * inv_f.view(1, 1, -1)
            half_freqs.append(ang)
        half = torch.cat(half_freqs, dim=-1)
        freqs = torch.cat([half, half], dim=-1)
        return freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)

    @staticmethod
    def apply_rope(
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        d = x.shape[-1]
        x_f = x.float()
        x1, x2 = x_f[..., : d // 2], x_f[..., d // 2 :]
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x_f * cos.float() + rotated * sin.float()).to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w12 = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


class PixelContentStem(nn.Module):
    """
    轻量无谱归一化像素内容编码器：
    - 输出全分辨率 pixel_feat [B, pixel_dim, H, W] 供空间形变对齐与像素精修头使用；
    - 输出 patch 对齐的 content_tokens [B, N, hidden_size]。
    """

    def __init__(self, hidden_size: int, pixel_dim: int, patch_size: int):
        super().__init__()
        groups = min(8, pixel_dim)
        while pixel_dim % groups != 0:
            groups -= 1
        self.stem = nn.Sequential(
            nn.Conv2d(3, pixel_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
        )
        self.res_conv = nn.Conv2d(3, pixel_dim, kernel_size=1)
        self.to_patch = nn.Conv2d(
            pixel_dim, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.patch_norm = RMSNorm(hidden_size)

    def forward(self, content_images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pix = self.stem(content_images) + self.res_conv(content_images)
        patches = self.to_patch(pix).flatten(2).transpose(1, 2)
        return self.patch_norm(patches), pix


class StyleMicroStatsEncoder(nn.Module):
    """
    v3 显式 48 维色彩 + 微几何与笔触拓扑编码器：
    - [0..31]  全局 RGB 均值/标准差/极值 + 四象限均值 + 5 档亮度占比(Duotone) + 色度 + 边缘密度；
    - [32..39] 8 方向梯度直方图 (HOG, 0..pi)；
    - [40]     正交阶梯各向异性比 min(|gx|,|gy|)/max(|gx|,|gy|) —— 瞬间区分像素阶梯折线(~0)与平滑45度斜线(>0.35)；
    - [41]     边缘二值锐度比（区分硬边像素/矢量 vs 抗锯齿/柔焦）；
    - [42]     拉普拉斯高频角点能量比；
    - [43]     笔画端点/断裂密度（显式编码虚线/断续线与离散对角像素块）；
    - [44]     前景墨迹宽高比（捕捉瘦长仿宋 vs 宽扁体中宫比例）；
    - [45]     近水平笔画倾角响应（精确量化仿宋横画左低右高「抗肩」斜势）；
    - [46]     近竖直笔画倾角响应（精确量化斜体 Italic / 行书倾斜）；
    - [47]     Stride-2/Stride-1 周期阶梯能量比。
    """

    STATS_DIM = 48

    def __init__(self, hidden_size: int, ada_dim: int):
        super().__init__()
        self.token_mlp = nn.Sequential(
            nn.Linear(self.STATS_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(self.STATS_DIM, ada_dim),
            nn.SiLU(),
            nn.Linear(ada_dim, ada_dim),
        )

    @staticmethod
    def extract_stats(images: torch.Tensor) -> torch.Tensor:
        # images: [N, 3, H, W] in [-1, 1]
        x = images.float()
        n, _, h, w = x.shape
        device = x.device

        mean = x.mean(dim=(2, 3))
        std = x.var(dim=(2, 3), unbiased=False).clamp_min(1e-6).sqrt()
        vmin = x.amin(dim=(2, 3))
        vmax = x.amax(dim=(2, 3))
        quad = F.adaptive_avg_pool2d(x, (2, 2)).flatten(1)  # [N, 12]

        # 5 档亮度占比（idx 27 为 b3 浅灰 Duotone 占比）
        lum = 0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]  # [N, H, W]
        b0 = (lum < -0.70).float().mean(dim=(1, 2)).unsqueeze(-1)
        b1 = ((lum >= -0.70) & (lum < -0.20)).float().mean(dim=(1, 2)).unsqueeze(-1)
        b2 = ((lum >= -0.20) & (lum < 0.30)).float().mean(dim=(1, 2)).unsqueeze(-1)
        b3 = ((lum >= 0.30) & (lum < 0.85)).float().mean(dim=(1, 2)).unsqueeze(-1)
        b4 = (lum >= 0.85).float().mean(dim=(1, 2)).unsqueeze(-1)

        chroma = 0.5 * (x.amax(dim=1) - x.amin(dim=1))
        chroma_mean = chroma.mean(dim=(1, 2)).unsqueeze(-1)
        chroma_max = chroma.amax(dim=(1, 2)).unsqueeze(-1)

        dx_raw = lum[:, :, 1:] - lum[:, :, :-1]
        dy_raw = lum[:, 1:, :] - lum[:, :-1, :]
        dx1 = dx_raw.abs().mean(dim=(1, 2)).unsqueeze(-1)
        dy1 = dy_raw.abs().mean(dim=(1, 2)).unsqueeze(-1)
        edge_density = 0.5 * (dx1 + dy1)

        # ---- v3 新增 16 维微几何与拓扑特征 ----
        lum_4d = lum.unsqueeze(1)
        # Sobel 梯度（抗 1px 混叠，用于方向直方图与仿宋/斜体倾角检测）
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) * 0.25
        ky = kx.transpose(2, 3).contiguous()
        klap = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
            device=device,
            dtype=torch.float32,
        ).view(1, 1, 3, 3) * 0.25

        gx = F.conv2d(lum_4d, kx, padding=1).squeeze(1)  # [N, H, W]
        gy = F.conv2d(lum_4d, ky, padding=1).squeeze(1)
        lap = F.conv2d(lum_4d, klap, padding=1).squeeze(1)

        mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
        mag_sum = mag.sum(dim=(1, 2)).clamp_min(1e-4)

        # 32..39: 8 方向梯度直方图 (0..pi)
        ang = torch.atan2(gy, gx) % math.pi  # [0, pi)
        bin_centers = torch.linspace(0.0, math.pi * (7.0 / 8.0), 8, device=device)
        # 软三角插值分箱，平滑且可微
        ang_diff = torch.abs(ang.unsqueeze(-1) - bin_centers.view(1, 1, 1, 8))
        ang_diff = torch.minimum(ang_diff, math.pi - ang_diff)
        bin_w = (1.0 - ang_diff / (math.pi / 8.0)).clamp_min(0.0)
        hog = (bin_w * mag.unsqueeze(-1)).sum(dim=(1, 2)) / mag_sum.unsqueeze(-1)  # [N, 8]

        # 40: 正交阶梯各向异性比（用 1px 差分对齐网格边界，像素阶梯处 min(|dx|,|dy|)≈0）
        adx = F.pad(dx_raw.abs(), (0, 1, 0, 0))
        ady = F.pad(dy_raw.abs(), (0, 0, 0, 1))
        e_max = torch.maximum(adx, ady)
        e_min = torch.minimum(adx, ady)
        e_mask = (e_max > 0.15).float()
        axis_aniso = (
            ((e_min / e_max.clamp_min(1e-4)) * e_mask).sum(dim=(1, 2))
            / e_mask.sum(dim=(1, 2)).clamp_min(1.0)
        ).unsqueeze(-1)

        # 41: 硬边二值锐度比（区分 0 抗锯齿像素风 vs 平滑抗锯齿曲线）
        active_e = (e_max > 0.06).float()
        hard_e = (e_max > 0.85).float()
        crispness = (
            hard_e.sum(dim=(1, 2)) / active_e.sum(dim=(1, 2)).clamp_min(1.0)
        ).unsqueeze(-1)

        # 42: 高频拉普拉斯角点比
        lap_ratio = (
            lap.abs().mean(dim=(1, 2)) / mag.mean(dim=(1, 2)).clamp_min(1e-3)
        ).clamp(0.0, 4.0).unsqueeze(-1) * 0.25

        # 43: 笔画端点 / 虚线断裂 / 孤立像素角点密度
        # 根据四角估计背景亮度，兼容白底黑字与深底亮色图标
        border_lum = (
            lum[:, 0, :].mean(dim=1)
            + lum[:, -1, :].mean(dim=1)
            + lum[:, :, 0].mean(dim=1)
            + lum[:, :, -1].mean(dim=1)
        ) * 0.25
        fg_mask = ((lum - border_lum.view(n, 1, 1)).abs() > 0.25).float()
        nbr_sum = F.avg_pool2d(fg_mask.unsqueeze(1), 3, stride=1, padding=1).squeeze(1) * 9.0
        endpoints = ((fg_mask > 0.5) & (nbr_sum <= 3.5)).float()
        discontinuity = (
            endpoints.sum(dim=(1, 2)) / fg_mask.sum(dim=(1, 2)).clamp_min(1.0)
        ).clamp(0.0, 1.0).unsqueeze(-1)

        # 44: 前景墨迹宽高展布比（区分瘦长中宫 vs 宽扁字形）
        ys = torch.linspace(-1.0, 1.0, h, device=device).view(1, h, 1)
        xs = torch.linspace(-1.0, 1.0, w, device=device).view(1, 1, w)
        fg_area = fg_mask.sum(dim=(1, 2)).clamp_min(1.0)
        cx = (fg_mask * xs).sum(dim=(1, 2)) / fg_area
        cy = (fg_mask * ys).sum(dim=(1, 2)) / fg_area
        sx = (
            ((fg_mask * (xs - cx.view(n, 1, 1)).pow(2)).sum(dim=(1, 2)) / fg_area)
            .clamp_min(1e-5)
            .sqrt()
        )
        sy = (
            ((fg_mask * (ys - cy.view(n, 1, 1)).pow(2)).sum(dim=(1, 2)) / fg_area)
            .clamp_min(1e-5)
            .sqrt()
        )
        aspect_spread = ((sx - sy) / (sx + sy).clamp_min(1e-4)).unsqueeze(-1)

        # 45: 近水平笔画抗肩倾角（仿宋左低右高为正，平直黑体为 0）
        h_mask = ((gy.abs() > 1.5 * gx.abs()) & (mag > 0.12)).float()
        h_sign = torch.where(gy >= 0.0, 1.0, -1.0)
        horiz_slant = (
            ((gx * h_sign / mag.clamp_min(1e-4)) * h_mask * mag).sum(dim=(1, 2))
            / (h_mask * mag).sum(dim=(1, 2)).clamp_min(1e-4)
        ).clamp(-1.0, 1.0).unsqueeze(-1)

        # 46: 近竖直笔画斜体倾角（右倾 Italic 为正）
        v_mask = ((gx.abs() > 1.5 * gy.abs()) & (mag > 0.12)).float()
        v_sign = torch.where(gx >= 0.0, 1.0, -1.0)
        vert_slant = (
            ((gy * v_sign / mag.clamp_min(1e-4)) * v_mask * mag).sum(dim=(1, 2))
            / (v_mask * mag).sum(dim=(1, 2)).clamp_min(1e-4)
        ).clamp(-1.0, 1.0).unsqueeze(-1)

        # 47: Stride-2 / Stride-1 周期阶梯与短虚线频率响应
        dx2 = (lum[:, :, 2:] - lum[:, :, :-2]).abs().mean(dim=(1, 2)).unsqueeze(-1)
        dy2 = (lum[:, 2:, :] - lum[:, :-2, :]).abs().mean(dim=(1, 2)).unsqueeze(-1)
        step_ratio = (0.5 * (dx2 + dy2) / edge_density.clamp_min(1e-3)).clamp(0.0, 3.0) / 3.0

        return torch.cat(
            [
                mean,
                std,
                vmin,
                vmax,
                quad,
                b0,
                b1,
                b2,
                b3,
                b4,
                chroma_mean,
                chroma_max,
                edge_density,
                hog,
                axis_aniso,
                crispness,
                lap_ratio,
                discontinuity,
                aspect_spread,
                horiz_slant,
                vert_slant,
                step_ratio,
            ],
            dim=-1,
        )  # [N, 48]

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        stats = self.extract_stats(images).to(dtype=self.token_mlp[0].weight.dtype)
        return self.token_mlp(stats).unsqueeze(1), self.global_mlp(stats)


# 保持向后别名导入兼容
ColorStatsEncoder = StyleMicroStatsEncoder


class _ResamplerCrossBlock(nn.Module):
    """Perceiver 风格重采样交叉注意力块（将变长原始多尺度风格 Token 压缩至固定长度高密度潜变量）。"""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_norm_in = RMSNorm(dim)
        self.kv_norm_in = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.ffn_norm = RMSNorm(dim)
        self.mlp = SwiGLU(dim, mlp_dim)

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, lq, d = latents.shape
        lk = context.shape[1]
        q = self.q_proj(self.q_norm_in(latents)).view(b, lq, self.num_heads, self.head_dim)
        kv = self.kv_proj(self.kv_norm_in(context)).view(b, lk, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(b, lq, d)
        latents = latents + self.out_proj(attn)
        latents = latents + self.mlp(self.ffn_norm(latents))
        return latents


class _StrokeDictionaryResampler(nn.Module):
    """轻量高分辨率前景笔触字典提取器（从 stride=2 的 Top-P 前景高梯度像素块浓缩为 M 个语义对齐笔触 Token）。"""

    def __init__(self, pixel_dim: int, num_tokens: int = 32, top_p: int = 256):
        super().__init__()
        self.pixel_dim = pixel_dim
        self.num_tokens = int(num_tokens)
        self.top_p = int(top_p)
        self.num_heads = 4 if pixel_dim >= 16 and pixel_dim % 4 == 0 else 2
        self.head_dim = pixel_dim // self.num_heads

        groups = min(8, pixel_dim)
        while pixel_dim % groups != 0:
            groups -= 1
        self.stroke_conv = nn.Sequential(
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
        )
        self.queries = nn.Parameter(torch.randn(1, self.num_tokens, pixel_dim) * 0.02)
        self.q_norm_in = RMSNorm(pixel_dim)
        self.kv_norm_in = RMSNorm(pixel_dim)
        self.q_proj = nn.Linear(pixel_dim, pixel_dim, bias=False)
        self.kv_proj = nn.Linear(pixel_dim, pixel_dim * 2, bias=False)
        self.out_proj = nn.Linear(pixel_dim, pixel_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, stem_feat: torch.Tensor, raw_images: torch.Tensor) -> torch.Tensor:
        """
        stem_feat:  [BK, C_pix, H, W]
        raw_images: [BK, 3, H, W] in [-1, 1]
        返回: [BK, M, C_pix]
        """
        hi = self.stroke_conv(stem_feat)  # [BK, C_pix, H/2, W/2]
        bk, c, hh, ww = hi.shape

        # 计算与四角背景的对比度 + 梯度显著性，自动剔除 75%~85% 纯白空白背景块
        with torch.no_grad():
            img_s = F.avg_pool2d(raw_images.float(), kernel_size=2, stride=2)
            lum = 0.299 * img_s[:, 0] + 0.587 * img_s[:, 1] + 0.114 * img_s[:, 2]
            border = (
                lum[:, 0, :].mean(dim=1)
                + lum[:, -1, :].mean(dim=1)
                + lum[:, :, 0].mean(dim=1)
                + lum[:, :, -1].mean(dim=1)
            ) * 0.25
            fg_sal = (lum - border.view(bk, 1, 1)).abs()
            dx = F.pad((lum[:, :, 1:] - lum[:, :, :-1]).abs(), (0, 1, 0, 0))
            dy = F.pad((lum[:, 1:, :] - lum[:, :-1, :]).abs(), (0, 0, 0, 1))
            saliency = (fg_sal + 2.0 * (dx + dy)).reshape(bk, hh * ww)
            p_sel = min(self.top_p, hh * ww)
            top_idx = saliency.topk(p_sel, dim=1, largest=True, sorted=False).indices

        hi_flat = hi.flatten(2).transpose(1, 2)  # [BK, hh*ww, C_pix]
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, c)
        fg_tokens = torch.gather(hi_flat, dim=1, index=gather_idx)  # [BK, P, C_pix]

        q_lat = self.queries.to(dtype=fg_tokens.dtype).expand(bk, -1, -1)
        q = self.q_proj(self.q_norm_in(q_lat)).view(bk, self.num_tokens, self.num_heads, self.head_dim)
        kv = self.kv_proj(self.kv_norm_in(fg_tokens)).view(bk, p_sel, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(bk, self.num_tokens, c)
        return q_lat + self.out_proj(attn)


class StyleTokenEncoder(nn.Module):
    """
    v3 多尺度前景笔触 + Perceiver 重采样风格编码器：
    - 每张参考图独立提取并重采样为固定 L_lat(64) 个高密度风格 Token + M(32) 个高清前景笔触 Token
      + 全局风格摘要 + 内容路由结构键（交由 Content-Aware Router 在 encode_condition 中按内容动态加权）；
    - 支持传入预缓存的 BF16 DINOv2 特征以实现训练期零重复视觉基座前向。
    """

    def __init__(
        self,
        hidden_size: int,
        ada_dim: int,
        patch_size: int,
        vision_dim: int,
        vision_grid_size: int,
        pixel_dim: int = 32,
        num_heads: int = 12,
        style_latent_tokens: int = 64,
        resampler_depth: int = 2,
        stroke_tokens_k: int = 32,
        max_pixel_grid: int = 16,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.vision_grid_size = vision_grid_size
        self.max_pixel_grid = max_pixel_grid
        self.style_latent_tokens = int(style_latent_tokens)
        self.stroke_tokens_k = int(stroke_tokens_k)

        groups = min(8, pixel_dim)
        while pixel_dim % groups != 0:
            groups -= 1
        # 1px 局部 3x3 纹理/笔触 Stem
        self.pixel_stem = nn.Sequential(
            nn.Conv2d(3, pixel_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=3, padding=1),
            nn.GroupNorm(groups, pixel_dim),
            nn.SiLU(),
        )
        self.pixel_stem_res = nn.Conv2d(3, pixel_dim, kernel_size=1)
        self.pixel_patch = nn.Conv2d(
            pixel_dim, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.pixel_norm = RMSNorm(hidden_size)

        # 高清 stride=2 前景笔触字典提取器（直供 PixelRefinementHead v3）
        self.stroke_resampler = _StrokeDictionaryResampler(
            pixel_dim=pixel_dim,
            num_tokens=self.stroke_tokens_k,
        )

        # 48 维色彩 + 微几何与拓扑统计编码器
        self.color_encoder = StyleMicroStatsEncoder(hidden_size=hidden_size, ada_dim=ada_dim)

        self.dino_patch_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_size),
            RMSNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dino_cls_proj = nn.Sequential(
            nn.Linear(vision_dim, ada_dim),
            nn.SiLU(),
            nn.Linear(ada_dim, ada_dim),
        )

        # 用于 Content-Aware 多参考路由的结构/语种亲和键
        self.ref_router_key = nn.Sequential(
            nn.Linear(ada_dim * 2, ada_dim),
            nn.SiLU(),
            nn.Linear(ada_dim, ada_dim),
        )

        # 2 层 Perceiver 风格重采样器（把 ~357 个原始空间风格 token 压缩为固定 64 个高密度潜变量）
        self.style_queries = nn.Parameter(
            torch.randn(1, self.style_latent_tokens, hidden_size) * 0.02
        )
        resampler_mlp = max(hidden_size * 2, 512)
        self.resampler_blocks = nn.ModuleList(
            [
                _ResamplerCrossBlock(
                    dim=hidden_size,
                    num_heads=num_heads,
                    mlp_dim=resampler_mlp,
                )
                for _ in range(int(resampler_depth))
            ]
        )
        self.resampler_norm = RMSNorm(hidden_size)

    def forward(
        self,
        style_images: torch.Tensor,
        vision_backbone: FrozenVisionBackbone,
        target_grid_hw: Tuple[int, int],
        token_drop_prob: float = 0.0,
        precomputed_dino: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        style_images: [B, 3, H, W] 或 [B, K, 3, H, W]
        precomputed_dino: 可选 (dino_cls [B*K, D_vis], dino_patches [B*K, G*G, D_vis])
        返回:
            ref_latents:        [B, K, L_lat, D]
            ref_global:         [B, K, ada_dim]
            ref_struct_key:     [B, K, ada_dim]
            ref_stroke_latents: [B, K, M, pixel_dim]
        """
        if style_images.ndim == 4:
            style_images = style_images.unsqueeze(1)
        b, k_refs, c, h, w = style_images.shape
        flat = style_images.reshape(b * k_refs, c, h, w)
        device = style_images.device
        dtype = self.pixel_patch.weight.dtype

        # 1. 1px 全分辨率纹理 Stem -> 粗 Patch + 高清 stride=2 前景笔触字典
        stem_feat = self.pixel_stem(flat) + self.pixel_stem_res(flat)
        pix_feat = self.pixel_patch(stem_feat)
        hp, wp = pix_feat.shape[-2:]
        if hp > self.max_pixel_grid or wp > self.max_pixel_grid:
            new_h = min(hp, self.max_pixel_grid)
            new_w = min(wp, self.max_pixel_grid)
            pix_feat = F.adaptive_avg_pool2d(pix_feat, (new_h, new_w))
        pix_tokens = self.pixel_norm(pix_feat.flatten(2).transpose(1, 2))  # [BK, N_pix, D]

        stroke_latents = self.stroke_resampler(stem_feat, flat)  # [BK, M, C_pix]

        # 2. 48 维色彩与微几何统计
        stats_tok, stats_glob = self.color_encoder(flat)  # [BK, 1, D], [BK, ada_dim]

        # 3. 冻结 DINOv2 表征（优先复用预缓存特征，零开销）
        if precomputed_dino is not None:
            dino_cls, dino_patches = precomputed_dino
            dino_cls = dino_cls.to(device=device, dtype=dtype)
            dino_patches = dino_patches.to(device=device, dtype=dtype)
        else:
            dino_cls_raw, dino_patches_raw = vision_backbone.extract_style_features(
                flat, no_grad=True
            )
            dino_cls = dino_cls_raw.to(dtype=dtype)
            dino_patches = dino_patches_raw.to(dtype=dtype)

        dino_tokens = self.dino_patch_proj(dino_patches)  # [BK, G*G, D]
        dino_glob = self.dino_cls_proj(dino_cls)  # [BK, ada_dim]

        # 4. 训练期随机遮罩部分空间风格 token（防内容形状泄露，始终保留 stats_tok）
        if self.training and token_drop_prob > 0.0:
            n_pix = pix_tokens.shape[1]
            n_keep_pix = max(1, int(round(n_pix * (1.0 - token_drop_prob))))
            keep_pix = torch.randperm(n_pix, device=device)[:n_keep_pix]
            pix_tokens = pix_tokens[:, keep_pix]

            n_dino = dino_tokens.shape[1]
            n_keep_dino = max(1, int(round(n_dino * (1.0 - token_drop_prob))))
            keep_dino = torch.randperm(n_dino, device=device)[:n_keep_dino]
            dino_tokens = dino_tokens[:, keep_dino]

        # 5. Perceiver 重采样为每张参考图固定 L_lat 个高密度潜变量
        raw_ctx = torch.cat([pix_tokens, dino_tokens, stats_tok], dim=1)
        latents = self.style_queries.to(dtype=raw_ctx.dtype).expand(b * k_refs, -1, -1)
        for blk in self.resampler_blocks:
            latents = blk(latents, raw_ctx)
        latents = self.resampler_norm(latents)

        ref_global = (dino_glob + stats_glob).view(b, k_refs, -1)
        ref_struct_key = self.ref_router_key(
            torch.cat([dino_glob, stats_glob], dim=-1)
        ).view(b, k_refs, -1)
        ref_latents = latents.view(b, k_refs, self.style_latent_tokens, -1)
        ref_stroke_latents = stroke_latents.view(b, k_refs, self.stroke_tokens_k, -1)
        return ref_latents, ref_global, ref_struct_key, ref_stroke_latents


class StyleSpatialWarper(nn.Module):
    """
    v3 零初始化可学习仿射 + 低频残差流场对齐器：
    - 仅约 ~0.35M 参数，初始精确恒等于单位矩阵 I（零训练风险）；
    - 解锁仿宋横画抗肩斜势（vertical shear）、英文斜体（horizontal shear）、中宫瘦长/宽扁缩放
      以及局部偏旁比例对齐；当 style 或 content 置空时自动退化为恒等映射。
    """

    def __init__(self, hidden_size: int, ada_dim: int):
        super().__init__()
        self.affine_head = nn.Sequential(
            nn.Linear(ada_dim * 2, ada_dim),
            nn.SiLU(),
            nn.Linear(ada_dim, 6),
        )
        nn.init.zeros_(self.affine_head[2].weight)
        nn.init.zeros_(self.affine_head[2].bias)

        self.flow_in = nn.Linear(hidden_size, 64)
        self.flow_style = nn.Linear(ada_dim, 64)
        self.flow_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.flow_conv[3].weight)
        nn.init.zeros_(self.flow_conv[3].bias)

    @torch.compiler.disable
    def forward(
        self,
        content_tokens: torch.Tensor,
        content_pixel_feat: torch.Tensor,
        content_summary: torch.Tensor,
        style_global: torch.Tensor,
        grid_hw: Tuple[int, int],
        disable_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, _, h, w = content_pixel_feat.shape
        hp, wp = grid_hw
        device = content_pixel_feat.device
        dtype = content_pixel_feat.dtype

        joint = torch.cat([content_summary, style_global], dim=-1)
        a = self.affine_head(joint).float()  # [B, 6]
        if disable_mask is not None:
            a = torch.where(disable_mask.view(b, 1), torch.zeros_like(a), a)

        sx = 1.0 + 0.18 * torch.tanh(a[:, 0])
        sy = 1.0 + 0.18 * torch.tanh(a[:, 1])
        shx = 0.25 * torch.tanh(a[:, 2])
        shy = 0.25 * torch.tanh(a[:, 3])
        tx = 0.08 * torch.tanh(a[:, 4])
        ty = 0.08 * torch.tanh(a[:, 5])

        row0 = torch.stack([sx, shx, tx], dim=-1)
        row1 = torch.stack([shy, sy, ty], dim=-1)
        theta = torch.stack([row0, row1], dim=1)  # [B, 2, 3]

        # 低频局部残差位移场（在 Patch 网格 Hp x Wp 上预测并双线性平滑插值）
        c_map = (
            self.flow_in(content_tokens)
            + self.flow_style(style_global).unsqueeze(1)
        ).transpose(1, 2).reshape(b, 64, hp, wp)
        delta_flow = 0.08 * torch.tanh(self.flow_conv(c_map).float())  # [B, 2, Hp, Wp]
        if disable_mask is not None:
            delta_flow = torch.where(
                disable_mask.view(b, 1, 1, 1),
                torch.zeros_like(delta_flow),
                delta_flow,
            )

        # 光滑正则（抑制局部折叠与无效漂移）
        tv_x = (delta_flow[:, :, :, 1:] - delta_flow[:, :, :, :-1]).pow(2).mean()
        tv_y = (delta_flow[:, :, 1:, :] - delta_flow[:, :, :-1, :]).pow(2).mean()
        warp_reg = 0.05 * a.pow(2).mean() + 0.5 * (tv_x + tv_y) + 0.1 * delta_flow.pow(2).mean()

        # 构造高精度 FP32 采样网格，再采样回原精度
        base_grid_pix = F.affine_grid(theta, size=(b, 1, h, w), align_corners=False)
        flow_pix = F.interpolate(
            delta_flow, size=(h, w), mode="bilinear", align_corners=False
        ).permute(0, 2, 3, 1)
        grid_pix = (base_grid_pix + flow_pix).clamp(-1.5, 1.5)

        warped_pix = F.grid_sample(
            content_pixel_feat.float(),
            grid_pix,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).to(dtype=dtype)

        base_grid_tok = F.affine_grid(theta, size=(b, 1, hp, wp), align_corners=False)
        grid_tok = (base_grid_tok + delta_flow.permute(0, 2, 3, 1)).clamp(-1.5, 1.5)
        tok_map = content_tokens.transpose(1, 2).reshape(b, -1, hp, wp)
        warped_tok = (
            F.grid_sample(
                tok_map.float(),
                grid_tok,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            .flatten(2)
            .transpose(1, 2)
            .to(dtype=dtype)
        )
        return warped_tok, warped_pix, warp_reg


class PixelDiTBlock(nn.Module):
    """
    非对称干净条件 DiT 块（Clean-Condition Asymmetric Attention, 纯 BF16 FlashAttention-2 路径）：
    - 条件流 [Content(256) ; StyleLatent(64)] 仅 320 tokens，与加噪时间步 t 解耦；
    - 目标流 [Noisy x_t(256)] 受 AdaLN-Zero(t, d, w, style_global) 调制并联合关注 [Target ; Cached Cond]。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_dim: int,
        ada_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.mlp = SwiGLU(hidden_size, mlp_dim)

        self.cond_norm1 = RMSNorm(hidden_size)
        self.cond_norm2 = RMSNorm(hidden_size)

        self.tgt_norm1 = RMSNorm(hidden_size, elementwise_affine=False)
        self.tgt_norm2 = RMSNorm(hidden_size, elementwise_affine=False)
        self.ada_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ada_dim, hidden_size * 6),
        )
        nn.init.zeros_(self.ada_modulation[1].weight)
        nn.init.zeros_(self.ada_modulation[1].bias)

    def _project_qkv(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, l, _ = x.shape
        qkv = self.qkv(x).view(b, l, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        q = RotaryEmbedding3D.apply_rope(q, cos, sin)
        k = RotaryEmbedding3D.apply_rope(k, cos, sin)
        return q, k, v

    def forward_cond(
        self,
        cond_tokens: torch.Tensor,
        cond_cos: torch.Tensor,
        cond_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.cond_norm1(cond_tokens)
        q, k, v = self._project_qkv(h, cond_cos, cond_sin)
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape_as(cond_tokens)
        cond_tokens = cond_tokens + self.proj(attn_out)
        cond_tokens = cond_tokens + self.mlp(self.cond_norm2(cond_tokens))
        return cond_tokens, (k, v)

    def forward_target(
        self,
        tgt_tokens: torch.Tensor,
        tgt_cos: torch.Tensor,
        tgt_sin: torch.Tensor,
        ada_cond: torch.Tensor,
        cond_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.ada_modulation(ada_cond).unsqueeze(1).chunk(6, dim=-1)
        )
        h = self.tgt_norm1(tgt_tokens) * (1.0 + scale_msa) + shift_msa
        q_t, k_t, v_t = self._project_qkv(h, tgt_cos, tgt_sin)

        k_c, v_c = cond_kv
        k_all = torch.cat([k_t, k_c], dim=2)
        v_all = torch.cat([v_t, v_c], dim=2)
        attn_out = F.scaled_dot_product_attention(q_t, k_all, v_all)
        attn_out = attn_out.transpose(1, 2).reshape_as(tgt_tokens)
        tgt_tokens = tgt_tokens + gate_msa * self.proj(attn_out)

        h_mlp = self.tgt_norm2(tgt_tokens) * (1.0 + scale_mlp) + shift_mlp
        tgt_tokens = tgt_tokens + gate_mlp * self.mlp(h_mlp)
        return tgt_tokens


class PixelRefinementBlock(nn.Module):
    """
    PixelDiT v3 双层像素精修块：
    - 局部 p×p 像素窗口在自注意力的同时，直接交叉关注高清前景风格笔触字典（stroke_tokens），
      让精修头逐像素复现阶梯像素格、虚线断口、尖锐折角与干笔飞白；
    - 跨边界 3×3 Depthwise Conv + SwiGLU 彻底消除 Patch 接缝。
    """

    def __init__(self, pixel_dim: int, patch_size: int):
        super().__init__()
        self.pixel_dim = pixel_dim
        self.patch_size = patch_size
        self.num_heads = 4 if pixel_dim >= 16 and pixel_dim % 4 == 0 else 2
        self.head_dim = pixel_dim // self.num_heads

        self.norm1 = RMSNorm(pixel_dim, elementwise_affine=False)
        self.qkv = nn.Linear(pixel_dim, pixel_dim * 3, bias=False)
        self.stroke_norm = RMSNorm(pixel_dim)
        self.stroke_kv = nn.Linear(pixel_dim, pixel_dim * 2, bias=False)
        self.proj = nn.Linear(pixel_dim, pixel_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

        self.norm2 = RMSNorm(pixel_dim, elementwise_affine=False)
        self.dwconv = nn.Conv2d(
            pixel_dim, pixel_dim, kernel_size=3, padding=1, groups=pixel_dim
        )
        self.pw1 = nn.Conv2d(pixel_dim, pixel_dim * 4, kernel_size=1)
        self.pw2 = nn.Conv2d(pixel_dim * 2, pixel_dim, kernel_size=1)

        self.pix_ada = nn.Sequential(
            nn.SiLU(),
            nn.Linear(pixel_dim * 2, pixel_dim * 6),
        )
        nn.init.zeros_(self.pix_ada[1].weight)
        nn.init.zeros_(self.pix_ada[1].bias)

    def forward(
        self,
        h_pix: torch.Tensor,
        patch_cond: torch.Tensor,
        style_stroke_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_pix:               [B, C_pix, H, W]
        patch_cond:          [B, Hp, Wp, 2 * C_pix]
        style_stroke_tokens: [B, M, C_pix]
        """
        b, c, h, w = h_pix.shape
        p = self.patch_size
        hp, wp = h // p, w // p
        num_win = hp * wp
        m = style_stroke_tokens.shape[1]

        mods = self.pix_ada(patch_cond).reshape(b * num_win, 1, c * 6)
        shift1, scale1, gate1, shift2, scale2, gate2 = mods.chunk(6, dim=-1)

        windows = (
            h_pix.view(b, c, hp, p, wp, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b * num_win, p * p, c)
        )
        x = self.norm1(windows) * (1.0 + scale1) + shift1
        qkv = self.qkv(x).view(b * num_win, p * p, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # 投影高清前景笔触字典并广播到每个局部像素窗口
        skv = self.stroke_kv(self.stroke_norm(style_stroke_tokens)).view(
            b, m, 2, self.num_heads, self.head_dim
        )
        ks, vs = skv.unbind(dim=2)
        ks = self.k_norm(ks).transpose(1, 2)  # [B, H, M, Dh]
        vs = vs.transpose(1, 2)
        ks = ks.unsqueeze(1).expand(b, num_win, -1, -1, -1).reshape(b * num_win, self.num_heads, m, self.head_dim)
        vs = vs.unsqueeze(1).expand(b, num_win, -1, -1, -1).reshape(b * num_win, self.num_heads, m, self.head_dim)

        k_all = torch.cat([k, ks], dim=2)
        v_all = torch.cat([v, vs], dim=2)
        attn = F.scaled_dot_product_attention(q, k_all, v_all)
        attn = attn.transpose(1, 2).reshape(b * num_win, p * p, c)
        windows = windows + gate1 * self.proj(attn)

        x2 = self.norm2(windows) * (1.0 + scale2) + shift2
        x2_map = (
            x2.view(b, hp, wp, p, p, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h, w)
        )
        u, v_g = self.pw1(self.dwconv(x2_map)).chunk(2, dim=1)
        ffn_map = self.pw2(F.silu(u) * v_g)
        ffn_win = (
            ffn_map.view(b, c, hp, p, wp, p)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(b * num_win, p * p, c)
        )
        windows = windows + gate2 * ffn_win

        return (
            windows.view(b, hp, wp, p, p, c)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, c, h, w)
        )


class PixelRefinementHead(nn.Module):
    """
    PixelDiT v3 像素级精修头（无 x_t 直连捷径 + 像素级空间内容门控 + 高清前景笔触交叉注意力）：
    - 像素级空间门控 c_gate_map(y, x) 允许主干在虚线断口或像素阶梯缺口处局部关断平滑内容残差；
    - 每个局部像素精修块交叉关注参考图的高清前景笔触字典，精准合成 1px 折线、像素块与细笔画。
    """

    def __init__(
        self,
        hidden_size: int,
        ada_dim: int,
        pixel_dim: int,
        pixel_depth: int,
        patch_size: int,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.pixel_dim = pixel_dim

        self.patch_to_pixels = nn.Linear(hidden_size, patch_size * patch_size * pixel_dim)
        self.content_proj = nn.Conv2d(pixel_dim, pixel_dim, kernel_size=1)
        # 全局通道门控偏置（初值 -1.0）+ 逐像素空间门控（初值 0，随训练学会在虚线/折线缺口处局部关断内容边）
        self.content_gate = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ada_dim, pixel_dim),
        )
        nn.init.zeros_(self.content_gate[1].weight)
        nn.init.constant_(self.content_gate[1].bias, -1.0)

        self.spatial_gate_conv = nn.Sequential(
            nn.Conv2d(pixel_dim * 2, pixel_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(pixel_dim, pixel_dim, kernel_size=1),
        )
        nn.init.zeros_(self.spatial_gate_conv[2].weight)
        nn.init.zeros_(self.spatial_gate_conv[2].bias)

        self.patch_cond_proj = nn.Linear(hidden_size, pixel_dim)
        self.time_cond_proj = nn.Linear(ada_dim, pixel_dim)

        self.blocks = nn.ModuleList(
            [
                PixelRefinementBlock(pixel_dim=pixel_dim, patch_size=patch_size)
                for _ in range(pixel_depth)
            ]
        )
        groups = min(8, pixel_dim)
        while pixel_dim % groups != 0:
            groups -= 1
        self.out_norm = nn.GroupNorm(groups, pixel_dim)
        self.out_conv = nn.Conv2d(pixel_dim, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        content_pixel_feat: torch.Tensor,
        patch_tokens: torch.Tensor,
        ada_cond: torch.Tensor,
        style_stroke_tokens: torch.Tensor,
    ) -> torch.Tensor:
        b, _, h, w = content_pixel_feat.shape
        p = self.patch_size
        hp, wp = h // p, w // p

        sem_pix = (
            self.patch_to_pixels(patch_tokens)
            .view(b, hp, wp, p, p, self.pixel_dim)
            .permute(0, 5, 1, 3, 2, 4)
            .reshape(b, self.pixel_dim, h, w)
        )
        c_proj = self.content_proj(content_pixel_feat)
        gate_logits = self.spatial_gate_conv(
            torch.cat([sem_pix, c_proj], dim=1)
        ) + self.content_gate(ada_cond).view(b, self.pixel_dim, 1, 1)
        c_gate_map = torch.sigmoid(gate_logits)
        h_pix = sem_pix + c_gate_map * c_proj

        p_cond = self.patch_cond_proj(patch_tokens).view(b, hp, wp, self.pixel_dim)
        t_cond = (
            self.time_cond_proj(ada_cond)
            .view(b, 1, 1, self.pixel_dim)
            .expand(b, hp, wp, self.pixel_dim)
        )
        patch_cond = torch.cat([p_cond, t_cond], dim=-1)

        for block in self.blocks:
            h_pix = block(h_pix, patch_cond, style_stroke_tokens)

        return self.out_conv(F.silu(self.out_norm(h_pix)))


@dataclass
class StylePack:
    """v3 单套风格的缓存编码（一次编码，支持按不同 Content 内容原型动态多参考路由）。"""

    ref_latents: torch.Tensor  # [B, K, L_lat, D]
    ref_global: torch.Tensor  # [B, K, ada_dim]
    ref_struct_key: torch.Tensor  # [B, K, ada_dim]
    ref_stroke_latents: torch.Tensor  # [B, K, M, pixel_dim]
    target_grid_hw: Tuple[int, int]


@dataclass
class ConditionCache:
    """v3 固定长度（320 Tokens）干净条件流逐层 KV 缓存。"""

    kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]]
    content_tokens: torch.Tensor
    content_pixel_feat: torch.Tensor
    style_global: torch.Tensor
    style_stroke_tokens: torch.Tensor
    target_grid_hw: Tuple[int, int]
    warp_reg: Optional[torch.Tensor] = None


class PixelDiTFlowModel(nn.Module):
    """
    2026 PixelDiT-Flow v3 主模型：
    - 48 维显式微几何与笔触拓扑编码（正交阶梯比 / 仿宋抗肩倾角 / 斜体倾角 / 虚线断裂度）；
    - Perceiver 风格重采样器将任意 K 张参考压缩为固定 64 个高密度风格潜变量（条件流从 1327 -> 320，提速 4x+，原生满血 torch.compile）；
    - Content-Aware 多参考路由器消除中英混输/多图 2-Shot 串味；
    - 零初始化 StyleSpatialWarper 解锁仿宋斜势、斜体与中宫比例形变；
    - PixelRefinementHead v3 引入像素级空间内容门控 + 高清前景笔触交叉注意力，守住阶梯折线与虚线断口。
    """

    def __init__(
        self,
        model_cfg: ModelConfig,
        vision_backbone: FrozenVisionBackbone,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        dims = model_cfg.resolved_dims()

        self.patch_size = int(model_cfg.patch_size)
        self.hidden_size = dims["hidden_size"]
        self.depth = dims["depth"]
        self.num_heads = dims["num_heads"]
        self.head_dim = self.hidden_size // self.num_heads
        self.mlp_dim = dims["mlp_dim"]
        self.ada_dim = dims["ada_dim"]
        self.pixel_depth = dims["pixel_depth"]
        self.pixel_dim = dims["pixel_dim"]
        self.repa_layer = dims["repa_layer"]
        self.style_latent_tokens = dims["style_latent_tokens"]
        self.style_resampler_depth = dims["style_resampler_depth"]
        self.stroke_tokens_k = dims["stroke_tokens_k"]
        self.use_spatial_warper = bool(model_cfg.use_spatial_warper)
        self.style_token_drop_prob = float(model_cfg.style_token_drop_prob)

        self.vision_backbone = vision_backbone
        self.vision_dim = int(vision_backbone.hidden_size)

        # 目标图 Patchify + 第 0 层形变对齐后内容骨架注入
        self.x_patch_embed = nn.Conv2d(
            3, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size
        )
        self.x_patch_norm = RMSNorm(self.hidden_size)
        self.tgt_content_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        nn.init.normal_(self.tgt_content_proj.weight, mean=0.0, std=0.02)

        # 内容与风格编码器
        self.content_stem = PixelContentStem(
            hidden_size=self.hidden_size,
            pixel_dim=self.pixel_dim,
            patch_size=self.patch_size,
        )
        self.style_encoder = StyleTokenEncoder(
            hidden_size=self.hidden_size,
            ada_dim=self.ada_dim,
            patch_size=self.patch_size,
            vision_dim=self.vision_dim,
            vision_grid_size=vision_backbone.grid_size,
            pixel_dim=self.pixel_dim,
            num_heads=self.num_heads,
            style_latent_tokens=self.style_latent_tokens,
            resampler_depth=self.style_resampler_depth,
            stroke_tokens_k=self.stroke_tokens_k,
        )

        # Content-Aware 多参考路由器（按内容原型特征 + 48维几何统计生成查询向量，消除跨语种 2-Shot 串味）
        self.content_summary_proj = nn.Sequential(
            nn.Linear(self.hidden_size + StyleMicroStatsEncoder.STATS_DIM, self.ada_dim),
            nn.SiLU(),
            nn.Linear(self.ada_dim, self.ada_dim),
        )
        self.router_dim = 32
        self.slot_router_q = nn.Linear(self.ada_dim, self.style_latent_tokens * self.router_dim)
        self.slot_router_k = nn.Linear(self.ada_dim, self.router_dim)

        # 零初始化仿射 + 平滑残差空间形变对齐器
        self.spatial_warper = (
            StyleSpatialWarper(hidden_size=self.hidden_size, ada_dim=self.ada_dim)
            if self.use_spatial_warper
            else None
        )

        # 独立双条件 CFG 的可学习置空表征
        self.null_content_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.null_style_token = nn.Parameter(torch.zeros(1, 1, self.hidden_size))
        self.null_style_global = nn.Parameter(torch.zeros(1, self.ada_dim))
        nn.init.normal_(self.null_content_token, std=0.02)
        nn.init.normal_(self.null_style_token, std=0.02)

        # 连续流时间 t / Shortcut 步长 d / 嵌入式双引导强度 (w_c, w_s)
        self.t_embedder = SinusoidalScalarEmbedder(self.ada_dim)
        self.d_embedder = SinusoidalScalarEmbedder(self.ada_dim)
        self.wc_embedder = SinusoidalScalarEmbedder(self.ada_dim)
        self.ws_embedder = SinusoidalScalarEmbedder(self.ada_dim)
        for emb in (self.d_embedder, self.wc_embedder, self.ws_embedder):
            nn.init.normal_(emb.mlp[2].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(emb.mlp[2].bias)

        # 3D RoPE 与 Patch-Level DiT 主干
        self.rope = RotaryEmbedding3D(self.head_dim)
        self.blocks = nn.ModuleList(
            [
                PixelDiTBlock(
                    hidden_size=self.hidden_size,
                    num_heads=self.num_heads,
                    mlp_dim=self.mlp_dim,
                    ada_dim=self.ada_dim,
                )
                for _ in range(self.depth)
            ]
        )
        self.final_norm = RMSNorm(self.hidden_size)

        # REPA 表征对齐投影头（训练期生效，推理不调用）
        self.repa_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, self.vision_dim),
        )

        # Pixel-Level v3 像素精修通路
        self.pixel_head = PixelRefinementHead(
            hidden_size=self.hidden_size,
            ada_dim=self.ada_dim,
            pixel_dim=self.pixel_dim,
            pixel_depth=self.pixel_depth,
            patch_size=self.patch_size,
        )
        # 选择性 torch.compile 缓存（普通 dict，不注册为子模块，确保 state_dict 键名 100% 纯净无 _orig_mod 前缀）
        self._compiled_fns: Dict[str, Any] = {}

    def enable_selective_compile(self) -> None:
        """
        对三大计算热点子图启用 torch.compile（style_encoder / 12层条件主干 / 12层目标主干+PixelHead）：
        - 保留原生 nn.Module 参数树与 state_dict 键名，零 _orig_mod 污染，EMA/Checkpoint/DDP 100% 兼容；
        - 仅在 train() 模式下激活编译子图，避免 eval 预览不同 batch_size 触发重编译抖动；
        - 避开 PyTorch 2.6 Inductor 在动态条件拼接处的 SymPy 边界 bug。
        """
        self._compiled_fns["style_encoder"] = torch.compile(self.style_encoder)
        self._compiled_fns["cond_blocks"] = torch.compile(self._run_cond_blocks)
        self._compiled_fns["forward_with_cache"] = torch.compile(
            self._forward_with_cache_impl
        )

    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        return {
            k: v
            for k, v in self.state_dict().items()
            if not k.startswith("vision_backbone.")
        }

    def load_trainable_state_dict(
        self,
        state_dict: Dict[str, torch.Tensor],
        strict: bool = True,
    ) -> None:
        cleaned = {
            k: v for k, v in state_dict.items() if not k.startswith("vision_backbone.")
        }
        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        missing_trainable = [k for k in missing if not k.startswith("vision_backbone.")]
        if strict and (missing_trainable or unexpected):
            raise RuntimeError(
                f"加载 PixelDiTFlowModel v3 权重失败: missing={missing_trainable[:8]}, "
                f"unexpected={unexpected[:8]}"
            )

    def _build_grid_coords(
        self,
        grid_hw: Tuple[int, int],
        slot: float,
        device: torch.device,
    ) -> torch.Tensor:
        hp, wp = grid_hw
        ys = torch.arange(hp, device=device, dtype=torch.float32)
        xs = torch.arange(wp, device=device, dtype=torch.float32)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        sl = torch.full((hp * wp,), slot, device=device, dtype=torch.float32)
        return torch.stack([gy.reshape(-1), gx.reshape(-1), sl], dim=-1)

    def _build_style_latent_coords(
        self,
        grid_hw: Tuple[int, int],
        device: torch.device,
    ) -> torch.Tensor:
        """为固定 L_lat 个风格潜变量构造均匀映射到目标网格坐标系的 (y, x, slot=2) 坐标。"""
        tgt_h, tgt_w = grid_hw
        n = self.style_latent_tokens
        side = max(1, int(math.ceil(math.sqrt(n))))
        ys = torch.linspace(0.0, max(tgt_h - 1, 0), steps=side, device=device)
        xs = torch.linspace(0.0, max(tgt_w - 1, 0), steps=side, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        yx = torch.stack([gy.reshape(-1), gx.reshape(-1)], dim=-1)[:n]
        sl = torch.full((n, 1), 2.0, device=device, dtype=torch.float32)
        return torch.cat([yx, sl], dim=-1)

    def _build_ada_cond(
        self,
        t: torch.Tensor,
        d: Optional[torch.Tensor],
        style_global: torch.Tensor,
        w_content: Optional[torch.Tensor] = None,
        w_style: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz = t.shape[0]
        device = t.device
        dtype = style_global.dtype
        if d is None:
            d = torch.zeros(bsz, device=device, dtype=torch.float32)
        if w_content is None:
            w_content = torch.ones(bsz, device=device, dtype=torch.float32)
        if w_style is None:
            w_style = torch.ones(bsz, device=device, dtype=torch.float32)

        base = self.t_embedder(t * 1000.0).to(dtype=dtype) + style_global
        has_shortcut = (d > 1e-6).to(dtype=dtype).unsqueeze(-1)
        sc_emb = (
            self.d_embedder(d * 1000.0).to(dtype=dtype)
            + self.wc_embedder(w_content).to(dtype=dtype)
            + self.ws_embedder(w_style).to(dtype=dtype)
        )
        return base + has_shortcut * sc_emb

    def encode_style(
        self,
        style_images: torch.Tensor,
        target_hw: Tuple[int, int],
        token_drop_prob: Optional[float] = None,
        precomputed_dino: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> StylePack:
        grid_hw = (target_hw[0] // self.patch_size, target_hw[1] // self.patch_size)
        drop_p = (
            self.style_token_drop_prob
            if token_drop_prob is None
            else float(token_drop_prob)
        )
        style_enc_fn = (
            self._compiled_fns["style_encoder"]
            if (self.training and "style_encoder" in self._compiled_fns)
            else self.style_encoder
        )
        ref_latents, ref_global, ref_struct_key, ref_stroke_latents = style_enc_fn(
            style_images=style_images,
            vision_backbone=self.vision_backbone,
            target_grid_hw=grid_hw,
            token_drop_prob=drop_p if self.training else 0.0,
            precomputed_dino=precomputed_dino,
        )
        return StylePack(
            ref_latents=ref_latents,
            ref_global=ref_global,
            ref_struct_key=ref_struct_key,
            ref_stroke_latents=ref_stroke_latents,
            target_grid_hw=grid_hw,
        )

    @staticmethod
    def expand_style_pack(style_pack: StylePack, batch_size: int) -> StylePack:
        if style_pack.ref_latents.shape[0] == batch_size:
            return style_pack
        if style_pack.ref_latents.shape[0] != 1:
            raise ValueError(
                f"无法将 batch={style_pack.ref_latents.shape[0]} 的 StylePack 扩展到 {batch_size}"
            )
        return StylePack(
            ref_latents=style_pack.ref_latents.expand(batch_size, -1, -1, -1).contiguous(),
            ref_global=style_pack.ref_global.expand(batch_size, -1, -1).contiguous(),
            ref_struct_key=style_pack.ref_struct_key.expand(batch_size, -1, -1).contiguous(),
            ref_stroke_latents=style_pack.ref_stroke_latents.expand(
                batch_size, -1, -1, -1
            ).contiguous(),
            target_grid_hw=style_pack.target_grid_hw,
        )

    @staticmethod
    def expand_condition_cache(cache: ConditionCache, batch_size: int) -> ConditionCache:
        """将 batch=1 的单例缓存（如无条件 uncond_cache）零拷贝广播至目标 batch_size。"""
        if cache.content_tokens.shape[0] == batch_size:
            return cache
        if cache.content_tokens.shape[0] != 1:
            raise ValueError(
                f"无法将 batch={cache.content_tokens.shape[0]} 的 ConditionCache 扩展到 {batch_size}"
            )
        return ConditionCache(
            kv_per_layer=[
                (k.expand(batch_size, -1, -1, -1), v.expand(batch_size, -1, -1, -1))
                for k, v in cache.kv_per_layer
            ],
            content_tokens=cache.content_tokens.expand(batch_size, -1, -1),
            content_pixel_feat=cache.content_pixel_feat.expand(batch_size, -1, -1, -1),
            style_global=cache.style_global.expand(batch_size, -1),
            style_stroke_tokens=cache.style_stroke_tokens.expand(batch_size, -1, -1),
            target_grid_hw=cache.target_grid_hw,
            warp_reg=cache.warp_reg,
        )

    @staticmethod
    def cat_condition_caches(caches: Sequence[ConditionCache]) -> ConditionCache:
        num_layers = len(caches[0].kv_per_layer)
        merged_kv = []
        for l_idx in range(num_layers):
            k_cat = torch.cat([c.kv_per_layer[l_idx][0] for c in caches], dim=0)
            v_cat = torch.cat([c.kv_per_layer[l_idx][1] for c in caches], dim=0)
            merged_kv.append((k_cat, v_cat))
        return ConditionCache(
            kv_per_layer=merged_kv,
            content_tokens=torch.cat([c.content_tokens for c in caches], dim=0),
            content_pixel_feat=torch.cat([c.content_pixel_feat for c in caches], dim=0),
            style_global=torch.cat([c.style_global for c in caches], dim=0),
            style_stroke_tokens=torch.cat([c.style_stroke_tokens for c in caches], dim=0),
            target_grid_hw=caches[0].target_grid_hw,
            warp_reg=None,
        )

    @staticmethod
    def slice_condition_cache(cache: ConditionCache, start: int, end: int) -> ConditionCache:
        return ConditionCache(
            kv_per_layer=[(k[start:end], v[start:end]) for k, v in cache.kv_per_layer],
            content_tokens=cache.content_tokens[start:end],
            content_pixel_feat=cache.content_pixel_feat[start:end],
            style_global=cache.style_global[start:end],
            style_stroke_tokens=cache.style_stroke_tokens[start:end],
            target_grid_hw=cache.target_grid_hw,
            warp_reg=cache.warp_reg,
        )

    def _route_multi_refs(
        self,
        content_summary: torch.Tensor,
        style_pack: StylePack,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Content-Aware 多参考自适应路由：
        - K == 1 时零开销直通；
        - K >= 2 时按内容结构/语种亲和度分别计算全局路由权重与逐槽位(64槽)路由权重，
          根除中英混输或异构图标 2-Shot 双向串味，同时保持输出固定为 64 个风格 Token。
        """
        k_refs = style_pack.ref_latents.shape[1]
        if k_refs == 1:
            return (
                style_pack.ref_latents[:, 0],
                style_pack.ref_global[:, 0],
                style_pack.ref_stroke_latents[:, 0],
            )

        b = content_summary.shape[0]
        # 全局亲和度 [B, K]
        g_logits = (
            content_summary.float().unsqueeze(1) * style_pack.ref_struct_key.float()
        ).sum(dim=-1) / math.sqrt(float(self.ada_dim))
        w_glob = F.softmax(g_logits, dim=-1).to(dtype=style_pack.ref_global.dtype)

        style_global = (w_glob.unsqueeze(-1) * style_pack.ref_global).sum(dim=1)
        stroke_tokens = (
            w_glob.view(b, k_refs, 1, 1) * style_pack.ref_stroke_latents
        ).sum(dim=1)

        # 逐潜变量槽位路由 [B, L_lat, K]：允许部分槽位共享全局字重/色彩，部分槽位专挑同语种/同结构参考
        q_slot = self.slot_router_q(content_summary).view(
            b, self.style_latent_tokens, self.router_dim
        )
        k_ref = self.slot_router_k(style_pack.ref_struct_key)  # [B, K, router_dim]
        slot_logits = (
            torch.einsum("bld,bkd->blk", q_slot.float(), k_ref.float())
            / math.sqrt(float(self.router_dim))
            + g_logits.unsqueeze(1)
        )
        w_slot = F.softmax(slot_logits, dim=-1).to(dtype=style_pack.ref_latents.dtype)
        style_tokens = torch.einsum("blk,bkld->bld", w_slot, style_pack.ref_latents)
        return style_tokens, style_global, stroke_tokens

    def encode_condition(
        self,
        content_images: torch.Tensor,
        style_pack: StylePack,
        drop_content: Optional[torch.Tensor] = None,
        drop_style: Optional[torch.Tensor] = None,
    ) -> ConditionCache:
        """
        将形变对齐后的 Content(256) 与路由压缩后的 Style(64) 拼装为固定 320 Tokens 跑完 12 层干净条件流。
        """
        b, _, h, w = content_images.shape
        device = content_images.device
        grid_hw = (h // self.patch_size, w // self.patch_size)
        style_pack = self.expand_style_pack(style_pack, b)

        content_tokens, content_pixel_feat = self.content_stem(content_images)
        c_stats = StyleMicroStatsEncoder.extract_stats(content_images).to(
            dtype=content_tokens.dtype
        )
        content_summary = self.content_summary_proj(
            torch.cat([content_tokens.mean(dim=1), c_stats], dim=-1)
        )

        style_tokens, style_global, stroke_tokens = self._route_multi_refs(
            content_summary=content_summary,
            style_pack=style_pack,
        )

        # 无分支 GPU 置空掩码处理（杜绝 CPU-GPU 同步与 Dynamo 图断裂）
        if drop_style is not None:
            m_tok = drop_style.view(b, 1, 1)
            m_glob = drop_style.view(b, 1)
            null_s = self.null_style_token.to(dtype=style_tokens.dtype).expand_as(style_tokens)
            null_g = self.null_style_global.to(dtype=style_global.dtype).expand_as(style_global)
            style_tokens = torch.where(m_tok, null_s, style_tokens)
            style_global = torch.where(m_glob, null_g, style_global)
            stroke_tokens = torch.where(m_tok, torch.zeros_like(stroke_tokens), stroke_tokens)

        warp_reg: Optional[torch.Tensor] = None
        if self.spatial_warper is not None:
            disable_warp: Optional[torch.Tensor] = None
            if drop_content is not None and drop_style is not None:
                disable_warp = drop_content | drop_style
            elif drop_content is not None:
                disable_warp = drop_content
            elif drop_style is not None:
                disable_warp = drop_style
            content_tokens, content_pixel_feat, warp_reg = self.spatial_warper(
                content_tokens=content_tokens,
                content_pixel_feat=content_pixel_feat,
                content_summary=content_summary,
                style_global=style_global,
                grid_hw=grid_hw,
                disable_mask=disable_warp,
            )

        if drop_content is not None:
            m_tok = drop_content.view(b, 1, 1)
            m_pix = drop_content.view(b, 1, 1, 1)
            null_c = self.null_content_token.to(dtype=content_tokens.dtype).expand_as(
                content_tokens
            )
            content_tokens = torch.where(m_tok, null_c, content_tokens)
            content_pixel_feat = torch.where(
                m_pix, torch.zeros_like(content_pixel_feat), content_pixel_feat
            )

        content_coords = self._build_grid_coords(grid_hw, slot=1.0, device=device)
        style_coords = self._build_style_latent_coords(grid_hw, device=device)
        cond_tokens = torch.cat([content_tokens, style_tokens], dim=1)  # 固定 256 + 64 = 320
        cond_coords = torch.cat([content_coords, style_coords], dim=0)
        cond_cos, cond_sin = self.rope.build_cos_sin(cond_coords)

        cond_blocks_fn = (
            self._compiled_fns["cond_blocks"]
            if (self.training and "cond_blocks" in self._compiled_fns)
            else self._run_cond_blocks
        )
        kv_per_layer = cond_blocks_fn(cond_tokens, cond_cos, cond_sin)

        return ConditionCache(
            kv_per_layer=kv_per_layer,
            content_tokens=content_tokens,
            content_pixel_feat=content_pixel_feat,
            style_global=style_global,
            style_stroke_tokens=stroke_tokens,
            target_grid_hw=grid_hw,
            warp_reg=warp_reg,
        )

    def _run_cond_blocks(
        self,
        cond_tokens: torch.Tensor,
        cond_cos: torch.Tensor,
        cond_sin: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        kv_per_layer: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for block in self.blocks:
            cond_tokens, kv = block.forward_cond(cond_tokens, cond_cos, cond_sin)
            kv_per_layer.append(kv)
        return kv_per_layer

    def _forward_with_cache_impl(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_cache: ConditionCache,
        d: Optional[torch.Tensor] = None,
        w_content: Optional[torch.Tensor] = None,
        w_style: Optional[torch.Tensor] = None,
        return_repa: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x_t.shape
        device = x_t.device
        grid_hw = (h // self.patch_size, w // self.patch_size)

        tgt_tokens = (
            self.x_patch_norm(self.x_patch_embed(x_t).flatten(2).transpose(1, 2))
            + self.tgt_content_proj(cond_cache.content_tokens)
        )
        tgt_coords = self._build_grid_coords(grid_hw, slot=0.0, device=device)
        tgt_cos, tgt_sin = self.rope.build_cos_sin(tgt_coords)

        ada_cond = self._build_ada_cond(
            t=t,
            d=d,
            style_global=cond_cache.style_global,
            w_content=w_content,
            w_style=w_style,
        )

        repa_feat: Optional[torch.Tensor] = None
        for idx, block in enumerate(self.blocks):
            tgt_tokens = block.forward_target(
                tgt_tokens=tgt_tokens,
                tgt_cos=tgt_cos,
                tgt_sin=tgt_sin,
                ada_cond=ada_cond,
                cond_kv=cond_cache.kv_per_layer[idx],
            )
            if return_repa and (idx + 1) == self.repa_layer:
                repa_feat = self.repa_projector(tgt_tokens)

        tgt_tokens = self.final_norm(tgt_tokens)
        x0_pred = self.pixel_head(
            content_pixel_feat=cond_cache.content_pixel_feat,
            patch_tokens=tgt_tokens,
            ada_cond=ada_cond,
            style_stroke_tokens=cond_cache.style_stroke_tokens,
        )

        if return_repa:
            assert repa_feat is not None
            return x0_pred, repa_feat
        return x0_pred

    def forward_with_cache(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond_cache: ConditionCache,
        d: Optional[torch.Tensor] = None,
        w_content: Optional[torch.Tensor] = None,
        w_style: Optional[torch.Tensor] = None,
        return_repa: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        fwc_fn = (
            self._compiled_fns["forward_with_cache"]
            if (self.training and "forward_with_cache" in self._compiled_fns)
            else self._forward_with_cache_impl
        )
        return fwc_fn(
            x_t=x_t,
            t=t,
            cond_cache=cond_cache,
            d=d,
            w_content=w_content,
            w_style=w_style,
            return_repa=return_repa,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        content_images: torch.Tensor,
        style_images: torch.Tensor,
        d: Optional[torch.Tensor] = None,
        w_content: Optional[torch.Tensor] = None,
        w_style: Optional[torch.Tensor] = None,
        drop_content: Optional[torch.Tensor] = None,
        drop_style: Optional[torch.Tensor] = None,
        return_repa: bool = False,
        return_warp_reg: bool = False,
        precomputed_style_dino: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor | Tuple[torch.Tensor, ...]:
        style_pack = self.encode_style(
            style_images=style_images,
            target_hw=(x_t.shape[-2], x_t.shape[-1]),
            precomputed_dino=precomputed_style_dino,
        )
        cond_cache = self.encode_condition(
            content_images=content_images,
            style_pack=style_pack,
            drop_content=drop_content,
            drop_style=drop_style,
        )
        out = self.forward_with_cache(
            x_t=x_t,
            t=t,
            cond_cache=cond_cache,
            d=d,
            w_content=w_content,
            w_style=w_style,
            return_repa=return_repa,
        )
        if return_warp_reg:
            reg = (
                cond_cache.warp_reg
                if cond_cache.warp_reg is not None
                else x_t.new_zeros((), dtype=torch.float32)
            )
            if return_repa:
                x0_pred, repa_feat = out  # type: ignore[misc]
                return x0_pred, repa_feat, reg
            return out, reg
        return out
