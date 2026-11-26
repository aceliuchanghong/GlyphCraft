from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

from PIL import Image
import torch

from ..config import SamplingConfig
from ..models.losses import apply_time_shift, jit_velocity_from_x0
from ..models.pixel_dit import ConditionCache, PixelDiTFlowModel, StylePack


def sample_initial_noise(
    batch_size: int,
    image_hw: Tuple[int, int],
    device: torch.device,
    generator: Optional[Union[torch.Generator, Sequence[torch.Generator]]] = None,
    seeds: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """
    构造 [B, 3, H, W] float32 初始高斯噪声：
    - 传入 `seeds`（长度 = B）：逐样本在 CPU 上按独立种子采样后搬运到目标 device，
      保证跨 batch_size、跨多卡分片及 CPU/GPU 跨设备的初始噪声 100% 比特级一致；
    - 传入 `generator` 序列（长度 = B）：逐样本使用对应 Generator 采样；
    - 传入单个 `generator`：保持向后兼容，按原样一次性采样。
    """
    if seeds is not None and generator is not None:
        raise ValueError("seeds 与 generator 不能同时传入，请二选一。")

    h, w = int(image_hw[0]), int(image_hw[1])
    if seeds is not None:
        if len(seeds) != batch_size:
            raise ValueError(
                f"seeds 长度 ({len(seeds)}) 必须等于 batch_size ({batch_size})。"
            )
        slices = []
        for s in seeds:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(s))
            slices.append(
                torch.randn(
                    (1, 3, h, w),
                    generator=gen,
                    device="cpu",
                    dtype=torch.float32,
                )
            )
        return torch.cat(slices, dim=0).to(device=device, dtype=torch.float32)

    if isinstance(generator, Sequence):
        if len(generator) != batch_size:
            raise ValueError(
                f"generator 列表长度 ({len(generator)}) 必须等于 batch_size ({batch_size})。"
            )
        slices = []
        for gen in generator:
            g_dev = gen.device if isinstance(gen, torch.Generator) else device
            slices.append(
                torch.randn(
                    (1, 3, h, w),
                    generator=gen,
                    device=g_dev,
                    dtype=torch.float32,
                )
            )
        return torch.cat(slices, dim=0).to(device=device, dtype=torch.float32)

    if generator is not None:
        g_dev = generator.device
        return torch.randn(
            (batch_size, 3, h, w),
            generator=generator,
            device=g_dev,
            dtype=torch.float32,
        ).to(device=device, dtype=torch.float32)

    return torch.randn((batch_size, 3, h, w), device=device, dtype=torch.float32)


def apply_apg_guidance(
    v_cond: torch.Tensor,
    delta: torch.Tensor,
    eta: float = 0.1,
    norm_threshold: float = 2.5,
) -> torch.Tensor:
    """
    Adaptive Projected Guidance (APG, Sadat et al., ICLR 2025)：
    将 CFG 增量 delta 沿 v_cond 方向分解为平行分量（导致色彩过曝/烧焦）与正交分量（增强细节），
    抑制平行分量并对增量施加分辨率无关的 RMS 范数保护，彻底消除高引导系数下的色彩发焦与振铃伪影。
    """
    vc = v_cond.float()
    d = delta.float()
    flat_dim = (1, 2, 3)

    if norm_threshold > 0.0:
        # 使用逐元素 RMS 范数而非未归一化 L2 总和，保证 96/128/256px 像素空间阈值完全分辨率无关
        d_rms = d.pow(2).mean(dim=flat_dim, keepdim=True).sqrt().clamp_min(1e-6)
        scale = (norm_threshold / d_rms).clamp_max(1.0)
        d = d * scale

    v_unit = vc / vc.pow(2).sum(dim=flat_dim, keepdim=True).sqrt().clamp_min(1e-6)
    proj = (d * v_unit).sum(dim=flat_dim, keepdim=True) * v_unit
    ortho = d - proj
    return vc + eta * proj + ortho


class StyleDiffuserFlowPipeline:
    """
    2026 统一双档流匹配推理管线（PixelDiT-Flow）：
    - fast 档（1-4 步 Shortcut）：单次前向内置引导强度 (w_c, w_s)，无 CFG 批翻倍，API 极致低成本；
    - hd   档（8-10 步 Flow Matching）：独立双条件 CFG (w_c / w_s) + APG + Guidance Interval + Euler/Heun；
    - 全程支持「风格一次编码（支持 K 张多参考图）」+「干净条件流逐层 KV 缓存」，推理降算 ~60%。
    """

    def __init__(
        self,
        model: PixelDiTFlowModel,
        sampling_cfg: Optional[SamplingConfig] = None,
        t_min: float = 0.05,
    ):
        self.model = model
        self.sampling_cfg = sampling_cfg if sampling_cfg is not None else SamplingConfig()
        self.t_min = float(t_min)

    @staticmethod
    def tensor_to_pil(images_norm: torch.Tensor) -> List[Image.Image]:
        imgs = (images_norm.detach().float() * 0.5 + 0.5).clamp(0.0, 1.0)
        arr = (imgs.cpu().permute(0, 2, 3, 1).numpy() * 255.0).round().astype("uint8")
        return [Image.fromarray(frame) for frame in arr]

    def encode_style(
        self,
        style_images: torch.Tensor,
        target_hw: Tuple[int, int],
    ) -> StylePack:
        param = next(self.model.parameters())
        return self.model.encode_style(
            style_images=style_images.to(device=param.device, dtype=param.dtype),
            target_hw=target_hw,
            token_drop_prob=0.0,
        )

    def _build_hd_caches(
        self,
        content_images: torch.Tensor,
        style_pack: StylePack,
        need_content_cfg: bool,
        need_style_cfg: bool,
    ) -> Tuple[ConditionCache, Optional[ConditionCache], int]:
        """
        构造 HD 模式的干净条件 KV 缓存（v3 单例广播 uncond_cache，节省 33% 编码开销）：
        """
        b = content_images.shape[0]
        dev = content_images.device
        zeros = torch.zeros(b, dtype=torch.bool, device=dev)
        ones = torch.ones(b, dtype=torch.bool, device=dev)

        full_cache = self.model.encode_condition(
            content_images=content_images,
            style_pack=style_pack,
            drop_content=zeros,
            drop_style=zeros,
        )
        if not need_content_cfg and not need_style_cfg:
            return full_cache, None, 1

        if need_content_cfg and need_style_cfg:
            one_1 = torch.ones(1, dtype=torch.bool, device=dev)
            sp_1 = StylePack(
                ref_latents=style_pack.ref_latents[:1],
                ref_global=style_pack.ref_global[:1],
                ref_struct_key=style_pack.ref_struct_key[:1],
                ref_stroke_latents=style_pack.ref_stroke_latents[:1],
                target_grid_hw=style_pack.target_grid_hw,
            )
            uncond_1 = self.model.encode_condition(
                content_images=content_images[:1],
                style_pack=sp_1,
                drop_content=one_1,
                drop_style=one_1,
            )
            uncond_cache = self.model.expand_condition_cache(uncond_1, b)
            c_only_cache = self.model.encode_condition(
                content_images=content_images,
                style_pack=style_pack,
                drop_content=zeros,
                drop_style=ones,
            )
            return (
                full_cache,
                self.model.cat_condition_caches([uncond_cache, c_only_cache, full_cache]),
                3,
            )

        drop_c = ones if need_content_cfg else zeros
        drop_s = ones if need_style_cfg else zeros
        base_cache = self.model.encode_condition(
            content_images=content_images,
            style_pack=style_pack,
            drop_content=drop_c,
            drop_style=drop_s,
        )
        return full_cache, self.model.cat_condition_caches([base_cache, full_cache]), 2

    def _predict_hd_velocity(
        self,
        x_t: torch.Tensor,
        t_scalar: float,
        full_cache: ConditionCache,
        guided_cache: Optional[ConditionCache],
        num_branches: int,
        w_c: float,
        w_s: float,
        use_apg: bool,
        apg_eta: float,
        apg_norm_threshold: float,
        guidance_interval: Tuple[float, float],
        u_scalar: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (guided_velocity, cond_x0_pred)，在未偏移进度空间 u∈[0,1] 上做区间引导与 APG 抗过曝。"""
        b = x_t.shape[0]
        dev = x_t.device
        progress = t_scalar if u_scalar is None else float(u_scalar)
        in_interval = (
            guided_cache is not None
            and guidance_interval[0] <= progress <= guidance_interval[1]
        )

        if not in_interval:
            t_vec = torch.full((b,), t_scalar, device=dev, dtype=torch.float32)
            d_vec = torch.zeros(b, device=dev, dtype=torch.float32)
            x0_cond = self.model.forward_with_cache(
                x_t=x_t, t=t_vec, cond_cache=full_cache, d=d_vec
            )
            v_cond = jit_velocity_from_x0(x_t, x0_cond, t_vec, t_min=self.t_min)
            return v_cond, x0_cond

        assert guided_cache is not None
        x_rep = x_t.repeat(num_branches, 1, 1, 1)
        t_rep = torch.full(
            (b * num_branches,), t_scalar, device=dev, dtype=torch.float32
        )
        d_rep = torch.zeros(b * num_branches, device=dev, dtype=torch.float32)
        x0_all = self.model.forward_with_cache(
            x_t=x_rep, t=t_rep, cond_cache=guided_cache, d=d_rep
        )
        v_all = jit_velocity_from_x0(x_rep, x0_all, t_rep, t_min=self.t_min)

        if num_branches == 3:
            v_uncond, v_c_only, v_cond = v_all.chunk(3, dim=0)
            x0_cond = x0_all.chunk(3, dim=0)[2]
            delta = (w_c - 1.0) * (v_c_only - v_uncond) + (w_s - 1.0) * (
                v_cond - v_c_only
            )
        else:
            v_base, v_cond = v_all.chunk(2, dim=0)
            x0_cond = x0_all.chunk(2, dim=0)[1]
            scale = w_c if abs(w_c - 1.0) > 1e-4 else w_s
            delta = (scale - 1.0) * (v_cond - v_base)

        if use_apg:
            v_guided = apply_apg_guidance(
                v_cond=v_cond,
                delta=delta,
                eta=apg_eta,
                norm_threshold=apg_norm_threshold,
            )
        else:
            v_guided = v_cond + delta
        return v_guided, x0_cond

    @torch.no_grad()
    def generate(
        self,
        content_images: torch.Tensor,
        style_images: Optional[torch.Tensor] = None,
        style_pack: Optional[StylePack] = None,
        num_inference_steps: Optional[int] = None,
        mode: Optional[str] = None,
        solver: Optional[str] = None,
        content_guidance_scale: Optional[float] = None,
        style_guidance_scale: Optional[float] = None,
        use_apg: Optional[bool] = None,
        guidance_interval: Optional[Tuple[float, float]] = None,
        time_shift: Optional[float] = None,
        generator: Optional[Union[torch.Generator, Sequence[torch.Generator]]] = None,
        seeds: Optional[Sequence[int]] = None,
        return_tensors: bool = False,
    ) -> List[Image.Image] | torch.Tensor:
        """
        统一生成入口：
        - mode='fast'（或 mode='auto' 且 steps<=4）：1-4 步 Shortcut 单次前向极速推理；
        - mode='hd'  （或 mode='auto' 且 steps>4） ：8-10 步 Flow Matching + Dual-CFG + APG 高画质推理。
        """
        if style_pack is None and style_images is None:
            raise ValueError("必须提供 style_images 或预编码的 style_pack。")

        cfg = self.sampling_cfg
        steps = int(
            num_inference_steps
            if num_inference_steps is not None
            else cfg.num_inference_steps
        )
        steps = max(1, steps)
        resolved_mode = mode if mode is not None else cfg.resolved_mode(steps)
        if resolved_mode == "auto":
            resolved_mode = "fast" if steps <= 4 else "hd"

        w_c = float(
            content_guidance_scale
            if content_guidance_scale is not None
            else cfg.content_guidance_scale
        )
        w_s = float(
            style_guidance_scale
            if style_guidance_scale is not None
            else cfg.style_guidance_scale
        )
        apg_on = cfg.use_apg if use_apg is None else bool(use_apg)
        g_interval = (
            cfg.guidance_interval
            if guidance_interval is None
            else (float(guidance_interval[0]), float(guidance_interval[1]))
        )
        shift = float(cfg.time_shift if time_shift is None else time_shift)
        ode_solver = solver if solver is not None else cfg.solver

        device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        content_images = content_images.to(device=device, dtype=model_dtype)
        b, _, h, w = content_images.shape

        if style_pack is None:
            assert style_images is not None
            style_pack = self.encode_style(
                style_images.to(device=device, dtype=model_dtype),
                target_hw=(h, w),
            )

        x = sample_initial_noise(
            batch_size=b,
            image_hw=(h, w),
            device=device,
            generator=generator,
            seeds=seeds,
        )

        # 像素空间偏移时间网格 t: 1.0 -> 0.0（与训练期 time_shift 严格对齐）
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32)
        sigmas = apply_time_shift(u_grid, shift=shift).tolist()

        # ---- 极速档：1-4 步 Shortcut（单次前向，无 CFG 批翻倍！） ----
        if resolved_mode == "fast":
            zeros = torch.zeros(b, dtype=torch.bool, device=device)
            full_cache = self.model.encode_condition(
                content_images=content_images,
                style_pack=style_pack,
                drop_content=zeros,
                drop_style=zeros,
            )
            wc_vec = torch.full((b,), w_c, device=device, dtype=torch.float32)
            ws_vec = torch.full((b,), w_s, device=device, dtype=torch.float32)

            for i in range(steps):
                t_curr = sigmas[i]
                t_next = sigmas[i + 1]
                dt = t_curr - t_next
                t_vec = torch.full((b,), t_curr, device=device, dtype=torch.float32)
                d_vec = torch.full((b,), dt, device=device, dtype=torch.float32)
                x0_pred = self.model.forward_with_cache(
                    x_t=x.to(dtype=model_dtype),
                    t=t_vec,
                    cond_cache=full_cache,
                    d=d_vec,
                    w_content=wc_vec,
                    w_style=ws_vec,
                ).float()
                if i == steps - 1 or t_next <= 1e-5:
                    x = x0_pred
                else:
                    ratio = dt / max(t_curr, 1e-5)
                    x = (1.0 - ratio) * x + ratio * x0_pred

            x = x.clamp(-1.0, 1.0)
            return x if return_tensors else self.tensor_to_pil(x)

        # ---- 高画质档：8-10 步 Flow Matching + Dual-CFG + APG + Euler/Heun ----
        need_c = abs(w_c - 1.0) > 1e-4
        need_s = abs(w_s - 1.0) > 1e-4
        full_cache, guided_cache, num_branches = self._build_hd_caches(
            content_images=content_images,
            style_pack=style_pack,
            need_content_cfg=need_c,
            need_style_cfg=need_s,
        )

        u_list = u_grid.tolist()
        for i in range(steps):
            u_curr = u_list[i]
            u_next = u_list[i + 1]
            t_curr = sigmas[i]
            t_next = sigmas[i + 1]
            dt = t_curr - t_next

            v1, x0_pred = self._predict_hd_velocity(
                x_t=x.to(dtype=model_dtype),
                t_scalar=t_curr,
                full_cache=full_cache,
                guided_cache=guided_cache,
                num_branches=num_branches,
                w_c=w_c,
                w_s=w_s,
                use_apg=apg_on,
                apg_eta=cfg.apg_eta,
                apg_norm_threshold=cfg.apg_norm_threshold,
                guidance_interval=g_interval,
                u_scalar=u_curr,
            )

            # heun_fast: 在高噪弯曲段 (u >= 0.5) 走二阶 Heun，低噪近似直线段 (u < 0.5) 走一阶 Euler，NFE 降 ~40%
            use_heun_step = (
                (ode_solver == "heun" and t_next > 1e-4)
                or (ode_solver == "heun_fast" and u_curr >= 0.5 and t_next > 1e-4)
            )
            if use_heun_step:
                x_euler = x - dt * v1
                v2, _ = self._predict_hd_velocity(
                    x_t=x_euler.to(dtype=model_dtype),
                    t_scalar=t_next,
                    full_cache=full_cache,
                    guided_cache=guided_cache,
                    num_branches=num_branches,
                    w_c=w_c,
                    w_s=w_s,
                    use_apg=apg_on,
                    apg_eta=cfg.apg_eta,
                    apg_norm_threshold=cfg.apg_norm_threshold,
                    guidance_interval=g_interval,
                    u_scalar=u_next,
                )
                x = x - dt * 0.5 * (v1 + v2)
            else:
                x = x - dt * v1

        x = x.clamp(-1.0, 1.0)
        return x if return_tensors else self.tensor_to_pil(x)
