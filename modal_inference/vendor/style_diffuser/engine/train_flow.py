from __future__ import annotations

import logging
import math
from pathlib import Path
import random
import sys
import time
from typing import List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from ..config import FlowStageConfig, ProjectConfig, save_config
from ..data.datasets import MatrixDiffusionDataset, collate_matrix_batch
from ..data.index import MatrixIndex, try_load_explicit_val_index
from ..models.build import (
    build_flow_model,
    load_flow_weights,
    load_scr_checkpoint,
    save_flow_checkpoint,
)
from ..models.ema import ModelEMA
from ..models.losses import (
    EdgeFrequencyLoss,
    PerceptualDISTSLoss,
    apply_time_shift,
    compute_snr_gate,
    flow_matching_velocity_loss,
    jit_velocity_from_x0,
    repa_alignment_loss,
    sample_logit_normal_timesteps,
)
from ..models.vision_backbone import VisionFeatureCache
from ..sampling.pipeline import apply_apg_guidance
from .logging_utils import (
    build_lr_scheduler,
    find_latest_scr_ckpt,
    find_latest_step_dir,
    format_train_progress,
    log_train_progress,
    resolve_mixed_precision,
    write_stage_done,
)
from .validation import render_validation_grid


def resolve_phase2_inputs(cfg: ProjectConfig) -> Tuple[Path, Optional[Path]]:
    """解析 Phase 2 依赖的 Phase 1 权重目录与 SCR 2.0 权重路径（支持 'auto' 自动衔接）。"""
    p1_raw = cfg.phase2.phase1_ckpt_dir
    if not p1_raw or p1_raw == "auto":
        p1_dir = find_latest_step_dir(cfg.phase1_dir)
        if p1_dir is None:
            raise FileNotFoundError(
                f"phase2.phase1_ckpt_dir='auto' 未在 '{cfg.phase1_dir}' 找到已保存的 global_step_* 权重。"
            )
    else:
        p1_dir = Path(p1_raw)
        if not (p1_dir / "pixel_dit.pth").is_file():
            raise FileNotFoundError(
                f"指定的 Phase 1 权重目录无效（缺少 pixel_dit.pth）: {p1_dir}"
            )

    if cfg.phase2.sc_coefficient <= 0.0:
        return p1_dir, None

    scr_raw = cfg.phase2.scr_ckpt_path
    if not scr_raw or scr_raw == "auto":
        scr_path = find_latest_scr_ckpt(cfg.scr_dir)
        if scr_path is None and cfg.scr.pretrained_ckpt_path:
            scr_path = Path(cfg.scr.pretrained_ckpt_path)
        if scr_path is None or not scr_path.is_file():
            raise FileNotFoundError(
                f"phase2.scr_ckpt_path='auto' 未在 '{cfg.scr_dir}' 找到 scr_*.pth，"
                "且未设置有效的 scr.pretrained_ckpt_path。"
            )
    else:
        scr_path = Path(scr_raw)
        if not scr_path.is_file():
            raise FileNotFoundError(f"指定的 SCR 权重文件不存在: {scr_path}")

    return p1_dir, scr_path


def _sample_dual_cfg_masks(
    bsz: int,
    drop_content_prob: float,
    drop_style_prob: float,
    drop_both_prob: float,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(bsz, device=device)
    p1 = float(drop_content_prob)
    p2 = p1 + float(drop_style_prob)
    p3 = p2 + float(drop_both_prob)
    drop_c = (u < p1) | ((u >= p2) & (u < p3))
    drop_s = ((u >= p1) & (u < p2)) | ((u >= p2) & (u < p3))
    return drop_c, drop_s


def _resolve_style_dino_cache(
    feat_cache: Optional[VisionFeatureCache],
    backbone,
    style_images: torch.Tensor,
    style_paths_2d: Sequence[Sequence[str]],
    style_clean: Optional[torch.Tensor],
    k_curr: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """对未经过色彩/微几何变形的干净风格图直接命中预缓存 DINOv2 特征，变形样本局部按需计算。"""
    if feat_cache is None:
        return None
    b = style_images.shape[0]
    if style_clean is None or bool(style_clean.all()):
        flat_paths = [row[k] for row in style_paths_2d for k in range(k_curr)]
        return feat_cache.get_style_dino(flat_paths, device=device, dtype=dtype)
    if not bool(style_clean.any()):
        return None

    clean_cpu = style_clean.detach().cpu().tolist()
    clean_b = [i for i, ok in enumerate(clean_cpu) if ok]
    dirty_b = [i for i, ok in enumerate(clean_cpu) if not ok]

    clean_paths = [style_paths_2d[i][k] for i in clean_b for k in range(k_curr)]
    c_cls, c_pat = feat_cache.get_style_dino(clean_paths, device=device, dtype=dtype)

    dirty_idx = torch.tensor(dirty_b, dtype=torch.long, device=device)
    dirty_imgs = style_images.index_select(0, dirty_idx)
    dirty_flat = dirty_imgs.reshape(-1, *dirty_imgs.shape[2:])
    d_cls, d_pat = backbone.extract_style_features(dirty_flat, no_grad=True)
    d_cls = d_cls.to(device=device, dtype=dtype)
    d_pat = d_pat.to(device=device, dtype=dtype)

    out_cls = torch.empty((b, k_curr, c_cls.shape[-1]), device=device, dtype=dtype)
    out_pat = torch.empty(
        (b, k_curr, c_pat.shape[-2], c_pat.shape[-1]), device=device, dtype=dtype
    )
    clean_idx = torch.tensor(clean_b, dtype=torch.long, device=device)
    out_cls.index_copy_(0, clean_idx, c_cls.view(len(clean_b), k_curr, -1))
    out_cls.index_copy_(0, dirty_idx, d_cls.view(len(dirty_b), k_curr, -1))
    out_pat.index_copy_(
        0, clean_idx, c_pat.view(len(clean_b), k_curr, c_pat.shape[-2], c_pat.shape[-1])
    )
    out_pat.index_copy_(
        0, dirty_idx, d_pat.view(len(dirty_b), k_curr, d_pat.shape[-2], d_pat.shape[-1])
    )
    return (
        out_cls.reshape(b * k_curr, -1),
        out_pat.reshape(b * k_curr, out_pat.shape[-2], out_pat.shape[-1]),
    )


def _resolve_repa_grids(
    feat_cache: Optional[VisionFeatureCache],
    backbone,
    x0_imgs: torch.Tensor,
    target_paths: Sequence[str],
    target_clean: Optional[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    if feat_cache is None:
        return backbone.extract_repa_targets(x0_imgs)
    if target_clean is None or bool(target_clean.all()):
        return feat_cache.get_repa_grids(target_paths, device=device)
    if not bool(target_clean.any()):
        return backbone.extract_repa_targets(x0_imgs)

    clean_cpu = target_clean.detach().cpu().tolist()
    clean_i = [i for i, ok in enumerate(clean_cpu) if ok]
    dirty_i = [i for i, ok in enumerate(clean_cpu) if not ok]

    c_grid = feat_cache.get_repa_grids([target_paths[i] for i in clean_i], device=device)
    dirty_t = torch.tensor(dirty_i, dtype=torch.long, device=device)
    d_grid = backbone.extract_repa_targets(x0_imgs.index_select(0, dirty_t))

    out = torch.empty(
        (x0_imgs.shape[0], *c_grid.shape[1:]), device=device, dtype=torch.float32
    )
    clean_t = torch.tensor(clean_i, dtype=torch.long, device=device)
    out.index_copy_(0, clean_t, c_grid.float())
    out.index_copy_(0, dirty_t, d_grid.float())
    return out


def _resolve_scr_raw_feats(
    feat_cache: Optional[VisionFeatureCache],
    backbone,
    imgs: torch.Tensor,
    paths: Sequence[str],
    clean_mask: Optional[torch.Tensor],
    layers: Sequence[int],
    device: torch.device,
) -> Optional[List[torch.Tensor]]:
    if feat_cache is None:
        return None
    if clean_mask is None or bool(clean_mask.all()):
        return feat_cache.get_multilayer(paths, device=device)
    if not bool(clean_mask.any()):
        return None

    clean_cpu = clean_mask.detach().cpu().tolist()
    clean_i = [i for i, ok in enumerate(clean_cpu) if ok]
    dirty_i = [i for i, ok in enumerate(clean_cpu) if not ok]

    c_layers = feat_cache.get_multilayer([paths[i] for i in clean_i], device=device)
    dirty_t = torch.tensor(dirty_i, dtype=torch.long, device=device)
    with torch.no_grad():
        d_layers = backbone.extract_multilayer_features(
            imgs.index_select(0, dirty_t), layers=layers
        )
    clean_t = torch.tensor(clean_i, dtype=torch.long, device=device)
    bsz = imgs.shape[0]
    merged: List[torch.Tensor] = []
    for cf, df in zip(c_layers, d_layers):
        buf = torch.empty((bsz, cf.shape[-1]), device=device, dtype=torch.float32)
        buf.index_copy_(0, clean_t, cf.float())
        buf.index_copy_(0, dirty_t, df.float())
        merged.append(buf)
    return merged


@torch.no_grad()
def _compute_shortcut_teacher_velocity(
    teacher_model,
    x_t: torch.Tensor,
    t_vec: torch.Tensor,
    d_big: float | torch.Tensor,
    d_half: float | torch.Tensor,
    content_images: torch.Tensor,
    style_images: torch.Tensor,
    w_c: torch.Tensor,
    w_s: torch.Tensor,
    t_min: float,
    use_apg: bool = True,
    apg_eta: float = 0.1,
    apg_norm_threshold: float = 2.5,
    dt1: Optional[torch.Tensor] = None,
    dt2: Optional[torch.Tensor] = None,
    use_instant_teacher: Optional[bool] = None,
    u_vec: Optional[torch.Tensor] = None,
    u_mid_vec: Optional[torch.Tensor] = None,
    guidance_interval: Tuple[float, float] = (0.25, 1.0),
    precomputed_style_dino: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """
    v3 极速免 JVP Shortcut 自洽 + Dual-CFG 蒸馏教师目标：
    - 支持复用预缓存 DINOv2 风格特征；
    - uncond_cache 仅以 batch=1 计算并零拷贝广播，节省 33% 教师条件流开销。
    """
    was_training = teacher_model.training
    teacher_model.eval()
    try:
        b, _, h, w = x_t.shape
        dev = x_t.device
        t_dtype = next(teacher_model.parameters()).dtype

        x_in = x_t.to(dtype=t_dtype)
        c_in = content_images.to(dtype=t_dtype)
        s_in = style_images.to(dtype=t_dtype)

        def _as_vec(val: float | torch.Tensor) -> torch.Tensor:
            if isinstance(val, torch.Tensor):
                return val.to(device=dev, dtype=torch.float32)
            return torch.full((b,), float(val), device=dev, dtype=torch.float32)

        d_big_vec = _as_vec(d_big)
        d_half_vec = _as_vec(d_half)
        dt1_vec = _as_vec(dt1) if dt1 is not None else (d_big_vec * 0.5)
        dt2_vec = _as_vec(dt2) if dt2 is not None else (d_big_vec * 0.5)
        instant = (
            bool(use_instant_teacher)
            if use_instant_teacher is not None
            else bool((d_half_vec <= 0.0).all())
        )

        style_pack = teacher_model.encode_style(
            style_images=s_in,
            target_hw=(h, w),
            token_drop_prob=0.0,
            precomputed_dino=precomputed_style_dino,
        )
        zeros = torch.zeros(b, dtype=torch.bool, device=dev)
        full_cache = teacher_model.encode_condition(
            content_images=c_in,
            style_pack=style_pack,
            drop_content=zeros,
            drop_style=zeros,
        )

        merged_cfg_cache = None
        if instant:
            ones_b = torch.ones(b, dtype=torch.bool, device=dev)
            # 单例 batch=1 计算无条件分支再零拷贝广播至 batch b
            one_1 = torch.ones(1, dtype=torch.bool, device=dev)
            sp_1 = type(style_pack)(
                ref_latents=style_pack.ref_latents[:1],
                ref_global=style_pack.ref_global[:1],
                ref_struct_key=style_pack.ref_struct_key[:1],
                ref_stroke_latents=style_pack.ref_stroke_latents[:1],
                target_grid_hw=style_pack.target_grid_hw,
            )
            uncond_1 = teacher_model.encode_condition(
                content_images=c_in[:1],
                style_pack=sp_1,
                drop_content=one_1,
                drop_style=one_1,
            )
            uncond_cache = teacher_model.expand_condition_cache(uncond_1, b)
            c_only_cache = teacher_model.encode_condition(
                content_images=c_in,
                style_pack=style_pack,
                drop_content=zeros,
                drop_style=ones_b,
            )
            merged_cfg_cache = teacher_model.cat_condition_caches(
                [uncond_cache, c_only_cache, full_cache]
            )

        def eval_step(
            x_cur: torch.Tensor,
            t_cur: torch.Tensor,
            d_sub: torch.Tensor,
            u_cur: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            if instant:
                assert merged_cfg_cache is not None
                x_rep = x_cur.repeat(3, 1, 1, 1)
                t_rep = t_cur.repeat(3)
                d_rep = torch.zeros_like(t_rep)
                x0_all = teacher_model.forward_with_cache(
                    x_t=x_rep, t=t_rep, cond_cache=merged_cfg_cache, d=d_rep
                )
                v_all = jit_velocity_from_x0(x_rep, x0_all, t_rep, t_min=t_min)
                v_u, v_c_only, v_cond = v_all.chunk(3, dim=0)
                wc_4 = w_c.view(b, 1, 1, 1).float()
                ws_4 = w_s.view(b, 1, 1, 1).float()
                delta = (wc_4 - 1.0) * (v_c_only - v_u) + (ws_4 - 1.0) * (
                    v_cond - v_c_only
                )
                if use_apg:
                    v_guided = apply_apg_guidance(
                        v_cond=v_cond,
                        delta=delta,
                        eta=apg_eta,
                        norm_threshold=apg_norm_threshold,
                    )
                else:
                    v_guided = v_cond + delta
                if u_cur is not None:
                    gi_lo, gi_hi = guidance_interval
                    in_gi = ((u_cur >= gi_lo) & (u_cur <= gi_hi)).view(b, 1, 1, 1)
                    return torch.where(in_gi, v_guided, v_cond)
                return v_guided

            x0_p = teacher_model.forward_with_cache(
                x_t=x_cur,
                t=t_cur,
                cond_cache=full_cache,
                d=d_sub,
                w_content=w_c,
                w_style=w_s,
            )
            return jit_velocity_from_x0(x_cur, x0_p, t_cur, t_min=t_min)

        d1_sub = dt1_vec if dt1 is not None else d_half_vec
        d2_sub = dt2_vec if dt2 is not None else d_half_vec
        dt1_4 = dt1_vec.view(b, 1, 1, 1)
        dt2_4 = dt2_vec.view(b, 1, 1, 1)

        v1 = eval_step(x_in, t_vec, d1_sub, u_cur=u_vec)
        x_mid = (x_in.float() - dt1_4 * v1).to(dtype=t_dtype)
        t_mid = (t_vec - dt1_vec).clamp_min(t_min)
        v2 = eval_step(x_mid, t_mid, d2_sub, u_cur=u_mid_vec)
        return (dt1_4 * v1 + dt2_4 * v2) / (dt1_4 + dt2_4).clamp_min(1e-6)
    finally:
        if was_training:
            teacher_model.train()


def run_train_flow(cfg: ProjectConfig, phase: int = 1) -> Path:
    if phase not in (1, 2):
        raise ValueError(f"phase 必须为 1 或 2，收到 {phase}")

    cfg.validate()
    is_phase2 = phase == 2
    stage_cfg: FlowStageConfig = cfg.phase2 if is_phase2 else cfg.phase1
    stage_name = "phase2" if is_phase2 else "phase1"
    stage_dir = cfg.phase2_dir if is_phase2 else cfg.phase1_dir
    logging_dir = stage_dir / cfg.runtime.logging_dir

    mp = resolve_mixed_precision(stage_cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=stage_cfg.gradient_accumulation_steps,
        mixed_precision=mp,
        log_with=cfg.runtime.report_to,
        project_dir=str(logging_dir),
    )

    if accelerator.is_main_process:
        stage_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        filename=str(stage_dir / f"{stage_name}_training.log"),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_flow_model(model_cfg=cfg.model, device="cpu")
    ema = (
        ModelEMA(
            model,
            decay=stage_cfg.ema_decay,
            warmup_steps=stage_cfg.ema_warmup_steps,
        )
        if stage_cfg.ema_decay > 0.0
        else None
    )

    scr = None
    if is_phase2:
        p1_ckpt_dir, scr_ckpt_path = resolve_phase2_inputs(cfg)
        load_flow_weights(
            model,
            p1_ckpt_dir,
            use_ema=True,
            ema=ema,
            map_location="cpu",
        )
        if accelerator.is_main_process:
            print(
                f"Phase 2 loaded Phase 1 weights from {p1_ckpt_dir}\n"
                f"Phase 2 SCR checkpoint: {scr_ckpt_path or '(disabled)'}",
                flush=True,
            )
        if scr_ckpt_path is not None:
            same_backbone = (
                cfg.scr.vision_backbone == cfg.model.vision_backbone
                and cfg.scr.vision_input_size == cfg.model.vision_input_size
            )
            scr = load_scr_checkpoint(
                scr_ckpt_path,
                vision_backbone=model.vision_backbone if same_backbone else None,
                mode="refinement",
                neg_chunk=cfg.phase2.neg_chunk,
                map_location="cpu",
            )
            scr.requires_grad_(False)
    elif stage_cfg.resume_ckpt_dir:
        load_flow_weights(
            model,
            stage_cfg.resume_ckpt_dir,
            use_ema=False,
            ema=ema,
            map_location="cpu",
        )
        if accelerator.is_main_process:
            print(f"Resumed {stage_name} from {stage_cfg.resume_ckpt_dir}", flush=True)

    perceptual_loss = (
        PerceptualDISTSLoss(
            pretrained=(cfg.model.vision_backbone.lower() not in ("none", "stub", "offline"))
        )
        if stage_cfg.perceptual_coefficient > 0.0
        else None
    )
    edge_loss_fn = (
        EdgeFrequencyLoss()
        if (
            stage_cfg.edge_coefficient > 0.0
            or (is_phase2 and cfg.phase2.shortcut_edge_coefficient > 0.0)
        )
        else None
    )

    index = MatrixIndex(
        data_root=cfg.data.root,
        phase=cfg.data.phase,
        layout=cfg.data.layout,
        require_content=True,
    )
    train_styles, holdout_styles = index.split_styles(
        val_holdout_styles=cfg.data.val_holdout_styles, seed=cfg.seed
    )

    explicit_val = try_load_explicit_val_index(cfg.data.root, layout=cfg.data.layout)
    if explicit_val is not None:
        preview_samples = explicit_val.select_preview_samples(
            num_samples=cfg.data.val_num_samples, seed=cfg.seed
        )
    else:
        preview_samples = index.select_preview_samples(
            num_samples=cfg.data.val_num_samples,
            seed=cfg.seed,
            preferred_styles=holdout_styles if holdout_styles else None,
        )

    use_scr_data = bool(is_phase2 and scr is not None)
    dataset = MatrixDiffusionDataset(
        index=index,
        data_cfg=cfg.data,
        use_scr=use_scr_data,
        num_neg=cfg.phase2.num_neg if use_scr_data else 0,
        allowed_styles=train_styles,
    )

    if accelerator.is_main_process:
        trainable_m = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        ram_flag = "RAM" if dataset._ram_cache is not None else "Disk"
        msg = (
            f"PixelDiT-Flow v3 ({cfg.model.preset}, {trainable_m:.2f}M trainable, "
            f"res={cfg.data.resolution}, patch={cfg.model.patch_size}, "
            f"style_latents={model.style_latent_tokens}, io={ram_flag}, mp={mp}) "
            f"stage={stage_name}: {len(dataset)} samples, "
            f"{len(dataset.style_to_valid_paths)} valid styles, "
            f"{len(preview_samples)} preview pairs."
        )
        print(msg, flush=True)
        logging.info(msg)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=stage_cfg.train_batch_size,
        num_workers=stage_cfg.num_workers,
        drop_last=len(dataset) >= stage_cfg.train_batch_size,
        persistent_workers=stage_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_matrix_batch,
    )

    learning_rate = stage_cfg.learning_rate
    if stage_cfg.scale_lr:
        learning_rate = (
            learning_rate
            * stage_cfg.gradient_accumulation_steps
            * stage_cfg.train_batch_size
            * accelerator.num_processes
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(stage_cfg.adam_beta1, stage_cfg.adam_beta2),
        weight_decay=stage_cfg.adam_weight_decay,
        eps=stage_cfg.adam_epsilon,
        fused=torch.cuda.is_available(),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=stage_cfg.lr_scheduler,
        num_warmup_steps=stage_cfg.lr_warmup_steps
        * stage_cfg.gradient_accumulation_steps,
        num_training_steps=stage_cfg.max_train_steps
        * stage_cfg.gradient_accumulation_steps,
        wsd_decay_ratio=stage_cfg.wsd_decay_ratio,
    )

    model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, dataloader, lr_scheduler
    )

    unwrapped_base = accelerator.unwrap_model(model)
    use_bf16_fast = mp == "bf16" and torch.cuda.is_available()
    if use_bf16_fast:
        unwrapped_base.vision_backbone.set_compute_dtype(torch.bfloat16)

    if scr is not None:
        scr = scr.to(accelerator.device)
        if use_bf16_fast and scr.vision_backbone is not unwrapped_base.vision_backbone:
            scr.vision_backbone.set_compute_dtype(torch.bfloat16)
    if perceptual_loss is not None:
        perceptual_loss = perceptual_loss.to(accelerator.device)
        if use_bf16_fast:
            perceptual_loss.stages.to(dtype=torch.bfloat16)
    if edge_loss_fn is not None:
        edge_loss_fn = edge_loss_fn.to(accelerator.device)

    # 启动期一次性预提取所有风格/目标图的 DINOv2 表征（Phase 2 同时缓存 SCR 多层特征）
    feat_cache: Optional[VisionFeatureCache] = None
    if cfg.data.feature_cache:
        t_cache = time.time()
        cache_paths = dataset.collect_style_target_paths()
        need_multi = (
            scr.feature_layers
            if (is_phase2 and scr is not None and scr.vision_backbone is unwrapped_base.vision_backbone)
            else None
        )
        feat_cache = VisionFeatureCache.build(
            backbone=unwrapped_base.vision_backbone,
            paths=cache_paths,
            image_loader=dataset.load_clean_norm,
            device=accelerator.device,
            need_style_patches=True,
            multilayer_layers=need_multi,
            batch_size=128,
        )
        neg_skipped = (
            dataset.enable_cached_neg_skip() if need_multi is not None else False
        )
        if accelerator.is_main_process:
            print(
                f"{stage_name.upper()} VisionFeatureCache built for {len(cache_paths)} images "
                f"(multilayer={'yes' if need_multi else 'no'}"
                f"{', zero-copy neg skip' if neg_skipped else ''}) in {time.time() - t_cache:.2f}s.",
                flush=True,
            )

    # Phase 2 持久 BF16 EMA 教师副本（彻底消灭每步 430MB 克隆备份 + 双次整模拷贝）
    teacher_model = None
    if is_phase2 and cfg.phase2.shortcut_enabled and cfg.phase2.shortcut_ratio > 0.0:
        teacher_model = build_flow_model(
            model_cfg=cfg.model,
            vision_backbone=unwrapped_base.vision_backbone,
            device=accelerator.device,
        )
        teacher_model.requires_grad_(False)
        teacher_model.eval()
        if use_bf16_fast:
            for name, p in teacher_model.named_parameters():
                if not name.startswith("vision_backbone."):
                    p.data = p.data.to(dtype=torch.bfloat16)
        if ema is not None:
            ema.copy_to(teacher_model)
        else:
            teacher_model.load_trainable_state_dict(
                unwrapped_base.trainable_state_dict(), strict=True
            )

    # 对三大计算热点子图启用选择性 torch.compile（state_dict 键名 100% 纯净，零 _orig_mod 污染）
    compiled_active = False
    forward_model = model
    if (
        stage_cfg.use_compile
        and hasattr(torch, "compile")
        and torch.cuda.is_available()
        and sys.platform != "win32"
    ):
        try:
            unwrapped_base.enable_selective_compile()
            compiled_active = True
            if accelerator.is_main_process:
                print(
                    f"{stage_name.upper()} selective torch.compile enabled for PixelDiT-Flow v3.",
                    flush=True,
                )
        except Exception as exc:
            if accelerator.is_main_process:
                print(f"[WARN] torch.compile 初始化失败，安全回退 eager 模式: {exc}", flush=True)

    if accelerator.is_main_process:
        accelerator.init_trackers(f"{cfg.project_name}_{stage_name}")
        save_config(cfg, stage_dir / "resolved_config.yaml")

    progress_bar = tqdm(
        range(stage_cfg.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc=f"{stage_name.upper()} Flow v3",
    )

    num_update_steps_per_epoch = max(
        1, math.ceil(len(dataloader) / stage_cfg.gradient_accumulation_steps)
    )
    num_train_epochs = math.ceil(stage_cfg.max_train_steps / num_update_steps_per_epoch)

    shortcut_steps_sorted = (
        sorted(set(int(s) for s in cfg.phase2.shortcut_steps if int(s) >= 1))
        if is_phase2 and cfg.phase2.shortcut_enabled
        else []
    )
    max_sc_steps = max(shortcut_steps_sorted) if shortcut_steps_sorted else 8

    global_step = 0
    train_start = time.time()
    last_saved_step = 0
    last_preview_step = -1
    latest_ckpt_dir: Optional[Path] = None
    latest_preview_path: Optional[Path] = None

    for _ in range(num_train_epochs):
        accum = {
            "total": 0.0,
            "fm": 0.0,
            "repa": 0.0,
            "dists": 0.0,
            "edge": 0.0,
            "scr": 0.0,
            "shortcut": 0.0,
        }

        for samples in dataloader:
            model.train()
            unwrapped_model = accelerator.unwrap_model(model)
            content_images = samples["content_image"]
            style_images = samples["style_images"]
            target_images = samples["target_image"]

            style_paths_all = samples.get("style_image_paths")
            target_paths_all = samples.get("target_image_path")
            style_clean_all = samples.get("style_clean")
            target_clean_all = samples.get("target_clean")

            k_max = style_images.shape[1]
            if k_max > 1:
                k_curr = random.randint(1, k_max)
                style_images = style_images[:, :k_curr]
                if style_paths_all is not None:
                    style_paths_all = [row[:k_curr] for row in style_paths_all]
                if k_curr > 1:
                    # 逐样本随机打乱 k_curr 张参考图顺序，防止路由器依赖槽位位置偷懒
                    perm_list = [
                        torch.randperm(k_curr).tolist()
                        for _ in range(style_images.shape[0])
                    ]
                    perm_t = torch.tensor(
                        perm_list, dtype=torch.long, device=style_images.device
                    )
                    gather_idx = perm_t.view(
                        style_images.shape[0], k_curr, 1, 1, 1
                    ).expand_as(style_images)
                    style_images = torch.gather(style_images, 1, gather_idx)
                    if style_paths_all is not None:
                        style_paths_all = [
                            [row[j] for j in p]
                            for row, p in zip(style_paths_all, perm_list)
                        ]
            else:
                k_curr = 1

            bsz = target_images.shape[0]
            dev = target_images.device
            h, w = target_images.shape[-2:]
            grid_hw = (h // unwrapped_model.patch_size, w // unwrapped_model.patch_size)
            dino_dtype = torch.bfloat16 if use_bf16_fast else torch.float32

            with accelerator.accumulate(model):
                run_shortcut = (
                    is_phase2
                    and cfg.phase2.shortcut_enabled
                    and cfg.phase2.shortcut_ratio > 0.0
                    and len(shortcut_steps_sorted) > 0
                    and bsz >= 2
                )
                n_sc = (
                    min(bsz - 1, max(1, int(round(bsz * cfg.phase2.shortcut_ratio))))
                    if run_shortcut
                    else 0
                )
                n_fm = bsz - n_sc

                # ---------- 1. HD Flow Matching 分支（d = 0）----------
                c_fm = content_images[:n_fm]
                s_fm = style_images[:n_fm]
                x0_fm = target_images[:n_fm]

                pre_s_dino_fm = (
                    _resolve_style_dino_cache(
                        feat_cache=feat_cache,
                        backbone=unwrapped_model.vision_backbone,
                        style_images=s_fm,
                        style_paths_2d=style_paths_all[:n_fm],
                        style_clean=style_clean_all[:n_fm] if style_clean_all is not None else None,
                        k_curr=k_curr,
                        device=dev,
                        dtype=dino_dtype,
                    )
                    if style_paths_all is not None
                    else None
                )

                noise_fm = torch.randn_like(x0_fm)
                t_fm = sample_logit_normal_timesteps(
                    n_fm,
                    device=dev,
                    mean=stage_cfg.logit_normal_mean,
                    std=stage_cfg.logit_normal_std,
                    time_shift=stage_cfg.time_shift,
                    uniform_prob=stage_cfg.uniform_t_prob,
                    low_t_prob=stage_cfg.low_t_prob,
                    low_t_max=stage_cfg.low_t_max,
                )
                t4 = t_fm.view(n_fm, 1, 1, 1)
                x_t_fm = (1.0 - t4) * x0_fm + t4 * noise_fm

                drop_c, drop_s = _sample_dual_cfg_masks(
                    n_fm,
                    drop_content_prob=stage_cfg.drop_content_prob,
                    drop_style_prob=stage_cfg.drop_style_prob,
                    drop_both_prob=stage_cfg.drop_both_prob,
                    device=dev,
                )
                want_repa = stage_cfg.repa_coefficient > 0.0
                want_warp_reg = (
                    stage_cfg.warp_reg_coefficient > 0.0
                    and unwrapped_model.spatial_warper is not None
                )
                out_fm = forward_model(
                    x_t=x_t_fm,
                    t=t_fm,
                    content_images=c_fm,
                    style_images=s_fm,
                    d=torch.zeros(n_fm, device=dev, dtype=torch.float32),
                    drop_content=drop_c,
                    drop_style=drop_s,
                    return_repa=want_repa,
                    return_warp_reg=want_warp_reg,
                    precomputed_style_dino=pre_s_dino_fm,
                )
                warp_reg = None
                if want_repa and want_warp_reg:
                    x0_pred_fm, repa_feat, warp_reg = out_fm
                elif want_repa:
                    x0_pred_fm, repa_feat = out_fm
                elif want_warp_reg:
                    x0_pred_fm, warp_reg = out_fm
                    repa_feat = None
                else:
                    x0_pred_fm, repa_feat = out_fm, None

                fm_loss = flow_matching_velocity_loss(
                    x_t=x_t_fm,
                    x0_pred=x0_pred_fm,
                    x0_target=x0_fm,
                    t=t_fm,
                    t_min=stage_cfg.t_min,
                    loss_t_floor=stage_cfg.loss_t_floor,
                )
                loss = fm_loss
                if warp_reg is not None:
                    loss = loss + stage_cfg.warp_reg_coefficient * warp_reg

                full_cond = (~drop_c) & (~drop_s)

                # REPA 中间层对齐损失（优先复用预缓存 DINOv2 表征）
                repa_val = 0.0
                if want_repa and repa_feat is not None and full_cond.any():
                    fc_idx = full_cond.nonzero(as_tuple=False).squeeze(-1).tolist()
                    dino_grid = _resolve_repa_grids(
                        feat_cache=feat_cache,
                        backbone=unwrapped_model.vision_backbone,
                        x0_imgs=x0_fm[full_cond],
                        target_paths=[target_paths_all[i] for i in fc_idx]
                        if target_paths_all is not None
                        else [],
                        target_clean=target_clean_all[:n_fm][full_cond]
                        if (target_clean_all is not None and target_paths_all is not None)
                        else None,
                        device=dev,
                    )
                    repa_loss = repa_alignment_loss(
                        repa_pred_tokens=repa_feat[full_cond],
                        dino_target_grid=dino_grid,
                        target_grid_hw=grid_hw,
                    )
                    loss = loss + stage_cfg.repa_coefficient * repa_loss
                    repa_val = float(repa_loss.detach().item())

                # SNR 门控 DISTS 结构+纹理感知损失
                dists_val = 0.0
                if (
                    perceptual_loss is not None
                    and stage_cfg.perceptual_coefficient > 0.0
                    and full_cond.any()
                ):
                    d_loss = perceptual_loss(
                        pred_x0=x0_pred_fm[full_cond],
                        target_x0=x0_fm[full_cond],
                        t=t_fm[full_cond],
                        t_max=stage_cfg.perceptual_t_max,
                    )
                    loss = loss + stage_cfg.perceptual_coefficient * d_loss
                    dists_val = float(d_loss.detach().item())

                # SNR 门控高频边缘 + Charbonnier 损失
                edge_val = 0.0
                if (
                    edge_loss_fn is not None
                    and stage_cfg.edge_coefficient > 0.0
                    and full_cond.any()
                ):
                    e_loss = edge_loss_fn(
                        pred_x0=x0_pred_fm[full_cond],
                        target_x0=x0_fm[full_cond],
                        t=t_fm[full_cond],
                        t_max=stage_cfg.resolved_edge_t_max(),
                    )
                    loss = loss + stage_cfg.edge_coefficient * e_loss
                    edge_val = float(e_loss.detach().item())

                # Phase 2: SNR 门控 SCR 2.0（真实正负样本 100% 命中预缓存，仅 x0_pred 跑 BF16 DINOv2 反传）
                scr_val = 0.0
                if is_phase2 and scr is not None and cfg.phase2.sc_coefficient > 0.0:
                    active_snr, snr_w = compute_snr_gate(
                        t_fm, t_max=stage_cfg.perceptual_t_max
                    )
                    valid_scr = active_snr & full_cond
                    if valid_scr.any():
                        v_idx = valid_scr.nonzero(as_tuple=False).squeeze(-1).tolist()
                        neg_images_all = samples.get("neg_images")
                        neg_fm = neg_images_all[:n_fm][valid_scr] if neg_images_all is not None else None
                        pos_raw = None
                        neg_raw = None
                        if (
                            feat_cache is not None
                            and feat_cache.multilayer_feats is not None
                            and target_paths_all is not None
                        ):
                            pos_raw = _resolve_scr_raw_feats(
                                feat_cache=feat_cache,
                                backbone=scr.vision_backbone,
                                imgs=x0_fm[valid_scr],
                                paths=[target_paths_all[i] for i in v_idx],
                                clean_mask=target_clean_all[:n_fm][valid_scr]
                                if target_clean_all is not None
                                else None,
                                layers=scr.feature_layers,
                                device=dev,
                            )
                            neg_clean_all = samples.get("neg_clean")
                            neg_paths_all = samples.get("neg_image_paths")
                            if (
                                neg_paths_all is not None
                                and (
                                    neg_clean_all is None
                                    or bool(neg_clean_all[:n_fm][valid_scr].all())
                                )
                            ):
                                flat_neg_paths = [
                                    p for i in v_idx for p in neg_paths_all[i]
                                ]
                                neg_raw = feat_cache.get_multilayer(
                                    flat_neg_paths, device=dev
                                )

                        s_emb, p_emb, n_emb = scr(
                            sample_imgs=x0_pred_fm[valid_scr].clamp(-1.0, 1.0),
                            pos_imgs=None if pos_raw is not None else x0_fm[valid_scr],
                            neg_imgs=None if neg_raw is not None else neg_fm,
                            pos_raw_feats=pos_raw,
                            neg_raw_feats=neg_raw,
                            num_neg=cfg.phase2.num_neg,
                        )
                        sc_loss = scr.calculate_nce_loss(
                            sample_s=s_emb,
                            pos_s=p_emb,
                            neg_s=n_emb,
                            sample_weights=snr_w[valid_scr],
                        )
                        loss = loss + cfg.phase2.sc_coefficient * sc_loss
                        scr_val = float(sc_loss.detach().item())

                # ---------- 2. Phase 2 渐进式 Shortcut + 边缘一致性蒸馏分支（d > 0）----------
                shortcut_val = 0.0
                if n_sc > 0:
                    c_sc = content_images[n_fm:]
                    s_sc = style_images[n_fm:]
                    x0_sc = target_images[n_fm:]

                    pre_s_dino_sc = (
                        _resolve_style_dino_cache(
                            feat_cache=feat_cache,
                            backbone=unwrapped_model.vision_backbone,
                            style_images=s_sc,
                            style_paths_2d=style_paths_all[n_fm:],
                            style_clean=style_clean_all[n_fm:]
                            if style_clean_all is not None
                            else None,
                            k_curr=k_curr,
                            device=dev,
                            dtype=dino_dtype,
                        )
                        if style_paths_all is not None
                        else None
                    )

                    # 渐进式课程采样：前 25% 步以 70% 概率优先训稳最细粒度 N=8 瞬时教师，消除 2 步残影
                    curr_progress = float(global_step) / float(
                        max(1, stage_cfg.max_train_steps)
                    )
                    if (
                        curr_progress < float(cfg.phase2.shortcut_curriculum_ratio)
                        and random.random() < 0.70
                    ):
                        step_n = max_sc_steps
                    else:
                        step_n = random.choice(shortcut_steps_sorted)

                    u_step = 1.0 / float(step_n)
                    k_idx = torch.randint(1, step_n + 1, (n_sc,), device=dev)
                    u_curr = k_idx.float() * u_step
                    u_next = (k_idx - 1).float() * u_step
                    u_mid = 0.5 * (u_curr + u_next)

                    t_sc = apply_time_shift(u_curr, shift=stage_cfg.time_shift)
                    t_mid_sc = apply_time_shift(u_mid, shift=stage_cfg.time_shift)
                    t_next_sc = apply_time_shift(u_next, shift=stage_cfg.time_shift)
                    d_big_vec = t_sc - t_next_sc
                    dt1_vec = t_sc - t_mid_sc
                    dt2_vec = t_mid_sc - t_next_sc
                    use_instant = step_n >= max_sc_steps

                    noise_sc = torch.randn_like(x0_sc)
                    t_sc4 = t_sc.view(n_sc, 1, 1, 1)
                    x_t_sc = (1.0 - t_sc4) * x0_sc + t_sc4 * noise_sc

                    wc_lo, wc_hi = cfg.phase2.distill_content_scale_range
                    ws_lo, ws_hi = cfg.phase2.distill_style_scale_range
                    wc_sc = torch.empty(n_sc, device=dev).uniform_(wc_lo, wc_hi)
                    ws_sc = torch.empty(n_sc, device=dev).uniform_(ws_lo, ws_hi)

                    # 直接调用持久 BF16 教师副本（零 clone/copy 开销）
                    t_eval_model = teacher_model if teacher_model is not None else unwrapped_model
                    v_teacher = _compute_shortcut_teacher_velocity(
                        teacher_model=t_eval_model,
                        x_t=x_t_sc,
                        t_vec=t_sc,
                        d_big=d_big_vec,
                        d_half=0.0 if use_instant else dt1_vec,
                        content_images=c_sc,
                        style_images=s_sc,
                        w_c=wc_sc,
                        w_s=ws_sc,
                        t_min=stage_cfg.t_min,
                        use_apg=cfg.sampling.use_apg,
                        apg_eta=cfg.sampling.apg_eta,
                        apg_norm_threshold=cfg.sampling.apg_norm_threshold,
                        dt1=dt1_vec,
                        dt2=dt2_vec,
                        use_instant_teacher=use_instant,
                        u_vec=u_curr,
                        u_mid_vec=u_mid,
                        guidance_interval=cfg.sampling.guidance_interval,
                        precomputed_style_dino=pre_s_dino_sc,
                    )

                    x0_stu = forward_model(
                        x_t=x_t_sc,
                        t=t_sc,
                        content_images=c_sc,
                        style_images=s_sc,
                        d=d_big_vec,
                        w_content=wc_sc,
                        w_style=ws_sc,
                        return_repa=False,
                        precomputed_style_dino=pre_s_dino_sc,
                    )
                    x0_teacher = (
                        x_t_sc.float()
                        - t_sc.clamp_min(stage_cfg.t_min).view(n_sc, 1, 1, 1)
                        * v_teacher.float()
                    ).clamp(-1.5, 1.5)
                    sc_boot_loss = flow_matching_velocity_loss(
                        x_t=x_t_sc,
                        x0_pred=x0_stu,
                        x0_target=x0_teacher.detach(),
                        t=t_sc,
                        t_min=stage_cfg.t_min,
                        loss_t_floor=stage_cfg.loss_t_floor,
                    )
                    if (
                        edge_loss_fn is not None
                        and float(cfg.phase2.shortcut_edge_coefficient) > 0.0
                    ):
                        sc_edge = edge_loss_fn(
                            pred_x0=x0_stu,
                            target_x0=x0_teacher.detach().clamp(-1.0, 1.0),
                            t=t_sc,
                            t_max=stage_cfg.resolved_edge_t_max(),
                        )
                        sc_boot_loss = (
                            sc_boot_loss
                            + float(cfg.phase2.shortcut_edge_coefficient) * sc_edge
                        )
                    loss = loss + cfg.phase2.shortcut_coefficient * sc_boot_loss
                    shortcut_val = float(sc_boot_loss.detach().item())

                denom = float(stage_cfg.gradient_accumulation_steps)
                accum["total"] += (
                    float(accelerator.gather(loss.detach().repeat(bsz)).mean().item())
                    / denom
                )
                accum["fm"] += float(fm_loss.detach().item()) / denom
                accum["repa"] += repa_val / denom
                accum["dists"] += dists_val / denom
                accum["edge"] += edge_val / denom
                accum["scr"] += scr_val / denom
                accum["shortcut"] += shortcut_val / denom

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, stage_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                if ema is not None:
                    ema.update(unwrapped_base)
                    if teacher_model is not None:
                        ema.copy_to(teacher_model)
                elif teacher_model is not None:
                    teacher_model.load_trainable_state_dict(
                        unwrapped_base.trainable_state_dict(), strict=True
                    )

                progress_bar.update(1)
                global_step += 1
                lr_now = lr_scheduler.get_last_lr()[0]

                metrics = {
                    "train/total_loss": accum["total"],
                    "train/fm_loss": accum["fm"],
                    "train/repa_loss": accum["repa"],
                    "train/dists_loss": accum["dists"],
                    "train/edge_loss": accum["edge"],
                    "train/lr": lr_now,
                }
                if is_phase2:
                    metrics["train/scr_loss"] = accum["scr"]
                    metrics["train/shortcut_loss"] = accum["shortcut"]
                accelerator.log(metrics, step=global_step)

                if (
                    accelerator.is_main_process
                    and global_step > 0
                    and global_step % stage_cfg.log_interval == 0
                ):
                    extras = {
                        "fm": accum["fm"],
                        "repa": accum["repa"],
                        "dists": accum["dists"],
                        "edge": accum["edge"],
                    }
                    if is_phase2:
                        extras["scr"] = accum["scr"]
                        extras["sc"] = accum["shortcut"]
                    log_train_progress(
                        format_train_progress(
                            step=global_step,
                            total=stage_cfg.max_train_steps,
                            loss=accum["total"],
                            started_at=train_start,
                            extras=extras,
                        )
                    )
                for k_m in accum:
                    accum[k_m] = 0.0

                if (
                    accelerator.is_main_process
                    and global_step % stage_cfg.ckpt_interval == 0
                ):
                    latest_ckpt_dir = save_flow_checkpoint(
                        save_dir=stage_dir / f"global_step_{global_step}",
                        model=unwrapped_base,
                        cfg=cfg,
                        ema=ema,
                        global_step=global_step,
                    )
                    last_saved_step = global_step
                    msg = (
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                        f"Saved {stage_name} checkpoint at step {global_step} -> {latest_ckpt_dir}"
                    )
                    logging.info(msg)
                    print(msg, flush=True)

                if (
                    accelerator.is_main_process
                    and stage_cfg.eval_interval > 0
                    and preview_samples
                    and global_step % stage_cfg.eval_interval == 0
                ):
                    eval_model = teacher_model if (teacher_model is not None and stage_cfg.use_ema_for_eval) else unwrapped_base
                    eval_ema = None if (eval_model is teacher_model) else (ema if stage_cfg.use_ema_for_eval else None)
                    latest_preview_path = render_validation_grid(
                        model=eval_model,
                        preview_samples=preview_samples,
                        cfg=cfg,
                        device=accelerator.device,
                        output_path=stage_dir / "previews" / f"step_{global_step:07d}.png",
                        global_step=global_step,
                        ema=eval_ema,
                        fast_steps=stage_cfg.eval_fast_steps,
                        hd_steps=stage_cfg.eval_hd_steps,
                        content_scale=stage_cfg.eval_content_scale,
                        style_scale=stage_cfg.eval_style_scale,
                        time_shift=stage_cfg.time_shift,
                        accelerator=accelerator,
                        tb_tag=f"{stage_name}/preview_grid",
                    )
                    last_preview_step = global_step
                    print(
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                        f"Saved {stage_name} dual-mode preview grid -> {latest_preview_path}",
                        flush=True,
                    )

            progress_bar.set_postfix(
                step_loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step >= stage_cfg.max_train_steps:
                break
        if global_step >= stage_cfg.max_train_steps:
            break

    if accelerator.is_main_process:
        if global_step > last_saved_step or latest_ckpt_dir is None:
            latest_ckpt_dir = save_flow_checkpoint(
                save_dir=stage_dir / f"global_step_{global_step}",
                model=unwrapped_base,
                cfg=cfg,
                ema=ema,
                global_step=global_step,
            )
        if stage_cfg.eval_interval > 0 and preview_samples and global_step > last_preview_step:
            eval_model = teacher_model if (teacher_model is not None and stage_cfg.use_ema_for_eval) else unwrapped_base
            eval_ema = None if (eval_model is teacher_model) else (ema if stage_cfg.use_ema_for_eval else None)
            latest_preview_path = render_validation_grid(
                model=eval_model,
                preview_samples=preview_samples,
                cfg=cfg,
                device=accelerator.device,
                output_path=stage_dir / "previews" / f"step_{global_step:07d}.png",
                global_step=global_step,
                ema=eval_ema,
                fast_steps=stage_cfg.eval_fast_steps,
                hd_steps=stage_cfg.eval_hd_steps,
                content_scale=stage_cfg.eval_content_scale,
                style_scale=stage_cfg.eval_style_scale,
                time_shift=stage_cfg.time_shift,
                accelerator=accelerator,
                tb_tag=f"{stage_name}/preview_grid",
            )
        write_stage_done(
            stage_dir,
            {
                "stage": stage_name,
                "global_step": global_step,
                "ckpt_dir": str(latest_ckpt_dir),
                "preview": str(latest_preview_path) if latest_preview_path else None,
                "compiled": compiled_active,
            },
        )

    accelerator.end_training()
    assert latest_ckpt_dir is not None
    return latest_ckpt_dir
