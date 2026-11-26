from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def apply_time_shift(
    u: torch.Tensor,
    shift: float = 1.0,
) -> torch.Tensor:
    """
    SD3 / Flux / PixelDiT 像素空间分辨率时间偏移（Resolution Shift）：
    t = (alpha * u) / (1 + (alpha - 1) * u)，保持端点 0 -> 0、1 -> 1 严格不变，
    alpha > 1 时将更多训练与采样预算分配到高噪起笔区（弥补原始像素空间相比 VAE Latent 的高冗余）。
    """
    alpha = float(shift)
    if abs(alpha - 1.0) < 1e-6:
        return u
    u_f = u.float()
    return (alpha * u_f) / (1.0 + (alpha - 1.0) * u_f).clamp_min(1e-6)


def sample_logit_normal_timesteps(
    batch_size: int,
    device: torch.device,
    mean: float = 0.0,
    std: float = 1.0,
    eps: float = 1e-3,
    time_shift: float = 1.0,
    uniform_prob: float = 0.0,
    low_t_prob: float = 0.0,
    low_t_max: float = 0.3,
) -> torch.Tensor:
    """
    像素空间流匹配时间步采样：
    - 以 (1 - uniform_prob) 采样 Logit-Normal(mean, std)，以 uniform_prob 混入 Uniform(0, 1) 保障 t->1.0 起笔端覆盖；
    - 通过 apply_time_shift(u, time_shift) 做像素分辨率高噪偏移，使高噪骨架生成段获得充足训练密度；
    - low_t_prob > 0 时，按该概率把已经偏移后的 t 换成 Uniform(eps, low_t_max)。
      只降 time_shift 到 2.5~3 并加大 uniform_prob，t<0.3 仍只有约 7%，而且会削掉高噪起笔；
      这段替换直接补低噪细节，主分布保持原样。
    """
    u_ln = torch.sigmoid(
        torch.randn(batch_size, device=device, dtype=torch.float32) * float(std)
        + float(mean)
    )
    p_unif = float(uniform_prob)
    if p_unif > 0.0:
        u_unif = torch.rand(batch_size, device=device, dtype=torch.float32)
        use_unif = torch.rand(batch_size, device=device) < p_unif
        u = torch.where(use_unif, u_unif, u_ln)
    else:
        u = u_ln
    t = apply_time_shift(u, shift=time_shift).clamp(eps, 1.0)
    p_low = float(low_t_prob)
    if p_low > 0.0:
        span = max(float(low_t_max) - float(eps), 1e-4)
        t_low = float(eps) + torch.rand(batch_size, device=device, dtype=torch.float32) * span
        use_low = torch.rand(batch_size, device=device) < p_low
        t = torch.where(use_low, t_low, t)
    return t


def jit_velocity_from_x0(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    t: torch.Tensor,
    t_min: float = 0.05,
) -> torch.Tensor:
    """
    JiT 像素空间 x0-prediction 转速度场（Li & He, 2025）：
    v = (x_t - x0) / max(t, t_min)，兼顾低维图像流形精度与 t->0 数值稳定。
    """
    denom = t.float().clamp_min(t_min).view(-1, 1, 1, 1)
    return (x_t.float() - x0_pred.float()) / denom


def flow_matching_velocity_loss(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    x0_target: torch.Tensor,
    t: torch.Tensor,
    t_min: float = 0.05,
    loss_t_floor: float = 1.0,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    无偏有界流匹配损失（像素空间 JiT x0-prediction 稳定版）：
    - 全局最优解严格为 x0_pred == E[x0 | x_t, c, s]；
    - 当 loss_t_floor > 0 时，用软限幅权重 w(t) = (1 + c^2) / (max(t, t_min)^2 + c^2) 替代
      无界 1/max(t, 0.05)^2（默认 c=1.0 时高低噪权重比收敛至 2x，彻底防止低噪盲去噪梯度垄断，
      让高噪起笔段 t in [0.85, 1.0] 与低噪精修段均衡获得有效梯度）；
    - 当 loss_t_floor == 0 时，严格退化为原始速度空间 MSE ||v_pred - v_target||^2。
    """
    t_eff = t.float().clamp_min(t_min)
    c = max(0.0, float(loss_t_floor))
    if c > 0.0:
        t_weight = (1.0 + c * c) / (t_eff.pow(2) + c * c)
    else:
        t_weight = 1.0 / t_eff.pow(2)
    per_sample = (x0_pred.float() - x0_target.float()).pow(2).mean(dim=(1, 2, 3)) * t_weight
    if weights is not None:
        w = weights.float().to(per_sample.device)
        return (per_sample * w).sum() / w.sum().clamp_min(1e-6)
    return per_sample.mean()


def repa_alignment_loss(
    repa_pred_tokens: torch.Tensor,
    dino_target_grid: torch.Tensor,
    target_grid_hw: Tuple[int, int],
) -> torch.Tensor:
    """
    REPA（Representation Alignment, ICLR 2025）中间层语义对齐损失：
    - repa_pred_tokens: [B, Hp*Wp, D_vis]（主干第 repa_layer 层经 MLP 投影后的特征）
    - dino_target_grid: [B, D_vis, G, G]（冻结 DINOv2 提取的干净目标图空间表征）
    """
    b, _, d_vis = repa_pred_tokens.shape
    hp, wp = target_grid_hw
    if dino_target_grid.shape[-2:] != (hp, wp):
        target_aligned = F.interpolate(
            dino_target_grid.float(),
            size=(hp, wp),
            mode="bilinear",
            align_corners=False,
        )
    else:
        target_aligned = dino_target_grid.float()

    target_tokens = target_aligned.flatten(2).transpose(1, 2)  # [B, Hp*Wp, D_vis]
    pred_norm = F.normalize(repa_pred_tokens.float(), dim=-1)
    tgt_norm = F.normalize(target_tokens, dim=-1)
    cos_sim = (pred_norm * tgt_norm).sum(dim=-1)
    return (1.0 - cos_sim).mean()


def compute_snr_gate(t: torch.Tensor, t_max: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    低中噪声门控权重（仅在 t <= t_max 激活图像级感知/边缘/SCR 损失，彻底消除高噪区梯度污染）：
    返回:
        active_mask: [B] bool
        weights:     [B] float in [0, 1]，按 (1 - t / t_max)^2 平滑过渡
    """
    t_f = t.float()
    active = t_f <= t_max
    w = ((1.0 - t_f / max(t_max, 1e-4)).clamp_min(0.0)).pow(2)
    return active, w


class PerceptualDISTSLoss(nn.Module):
    """
    现代 DISTS 风格结构 + 纹理感知损失（取代 2014 原始未校准 VGG16 MSE）：
    - 将多层 VGG 特征分解为纹理均值相似度 l(mu) 与几何结构协方差相似度 s(sigma, cov)；
    - 仅在低中噪声子样本（t <= t_max）上前向计算，节省 ~50% 计算量且梯度极稳。
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.VGG16_Weights.DEFAULT if pretrained else None
        try:
            vgg = torchvision.models.vgg16(weights=weights)
        except Exception:
            # 离线环境安全回退
            vgg = torchvision.models.vgg16(weights=None)

        self.stages = nn.ModuleList(
            [
                nn.Sequential(*vgg.features[:4]),
                nn.Sequential(*vgg.features[4:9]),
                nn.Sequential(*vgg.features[9:16]),
                nn.Sequential(*vgg.features[16:23]),
            ]
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    def train(self, mode: bool = True):
        super().train(False)
        return self

    def _extract(self, x_norm: torch.Tensor) -> list[torch.Tensor]:
        param_dtype = next(self.stages[0].parameters()).dtype
        x = (x_norm.float() * 0.5 + 0.5).clamp(0.0, 1.0)
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        feats = [x]
        h = x.to(dtype=param_dtype)
        for stage in self.stages:
            h = stage(h)
            feats.append(h.float())
        return feats

    @staticmethod
    def _dists_pair(fx: torch.Tensor, fy: torch.Tensor, c1: float = 1e-6, c2: float = 1e-6) -> torch.Tensor:
        """返回每张图的 DISTS 距离 [B]（范围约 [0, 2]，越小越相似）。"""
        mx = fx.mean(dim=(2, 3), keepdim=True)
        my = fy.mean(dim=(2, 3), keepdim=True)
        vx = ((fx - mx) ** 2).mean(dim=(2, 3), keepdim=True)
        vy = ((fy - my) ** 2).mean(dim=(2, 3), keepdim=True)
        cxy = ((fx - mx) * (fy - my)).mean(dim=(2, 3), keepdim=True)

        l_term = (2.0 * mx * my + c1) / (mx.pow(2) + my.pow(2) + c1)
        s_term = (2.0 * cxy + c2) / (vx + vy + c2)
        dist = 1.0 - 0.5 * (l_term + s_term)
        return dist.mean(dim=(1, 2, 3))

    def forward(
        self,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 0.5,
    ) -> torch.Tensor:
        active, weights = compute_snr_gate(t, t_max=t_max)
        if not active.any():
            return pred_x0.new_zeros((), dtype=torch.float32)

        px = pred_x0[active].clamp(-1.0, 1.0)
        tx = target_x0[active]
        w = weights[active]

        f_pred = self._extract(px)
        with torch.no_grad():
            f_tgt = self._extract(tx)

        total = torch.zeros(px.shape[0], device=px.device, dtype=torch.float32)
        for fp, ft in zip(f_pred, f_tgt):
            total = total + self._dists_pair(fp, ft)
        total = total / float(len(f_pred))
        return (total * w).sum() / w.sum().clamp_min(1e-6)


class EdgeFrequencyLoss(nn.Module):
    """
    高频边缘 + Charbonnier 像素保真损失（专治字体细笔画、UI 图标抗锯齿边缘、像素画发虚）：
    仅在 t <= t_max 低中噪步按 (1 - t/t_max)^2 加权生效。
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3).contiguous()
        laplacian = torch.tensor(
            [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("sobel_x", sobel_x.repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer("sobel_y", sobel_y.repeat(3, 1, 1, 1), persistent=False)
        self.register_buffer("laplacian", laplacian.repeat(3, 1, 1, 1), persistent=False)

    def _charbonnier(self, diff: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(diff.pow(2) + self.eps * self.eps).mean(dim=(1, 2, 3))

    def forward(
        self,
        pred_x0: torch.Tensor,
        target_x0: torch.Tensor,
        t: torch.Tensor,
        t_max: float = 0.5,
    ) -> torch.Tensor:
        active, weights = compute_snr_gate(t, t_max=t_max)
        if not active.any():
            return pred_x0.new_zeros((), dtype=torch.float32)

        px = pred_x0[active].float().clamp(-1.0, 1.0)
        tx = target_x0[active].float()
        w = weights[active]

        dev = px.device
        sx = self.sobel_x.to(dev)
        sy = self.sobel_y.to(dev)
        lap = self.laplacian.to(dev)

        l_pix = self._charbonnier(px - tx)
        l_gx = self._charbonnier(F.conv2d(px, sx, padding=1, groups=3) - F.conv2d(tx, sx, padding=1, groups=3))
        l_gy = self._charbonnier(F.conv2d(px, sy, padding=1, groups=3) - F.conv2d(tx, sy, padding=1, groups=3))
        l_lap = self._charbonnier(F.conv2d(px, lap, padding=1, groups=3) - F.conv2d(tx, lap, padding=1, groups=3))

        per_sample = 0.5 * l_pix + 0.25 * (l_gx + l_gy) + 0.25 * l_lap
        return (per_sample * w).sum() / w.sum().clamp_min(1e-6)
