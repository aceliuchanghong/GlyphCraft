from __future__ import annotations

import logging
import math
from pathlib import Path
import time
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed

from ..config import ProjectConfig, save_config
from ..data.datasets import MatrixSCRDataset, collate_matrix_batch
from ..data.index import MatrixIndex
from ..models.build import build_scr, load_scr_checkpoint, save_scr_checkpoint
from ..models.vision_backbone import VisionFeatureCache
from .logging_utils import (
    build_lr_scheduler,
    format_train_progress,
    log_train_progress,
    resolve_mixed_precision,
    write_stage_done,
)


def _resolve_multilayer_batch(
    feat_cache: VisionFeatureCache | None,
    backbone,
    images: torch.Tensor,
    paths: Sequence[str],
    clean_mask: torch.Tensor | None,
    layers: Sequence[int],
    device: torch.device,
) -> List[torch.Tensor] | None:
    """命中预缓存的干净样本零开销查表，仅对增广变形样本跑 BF16 DINOv2。"""
    if feat_cache is None:
        return None
    if clean_mask is None or bool(clean_mask.all()):
        return feat_cache.get_multilayer(paths, device=device)
    if not bool(clean_mask.any()):
        return None

    clean_cpu = clean_mask.detach().cpu().tolist()
    clean_idx = [i for i, ok in enumerate(clean_cpu) if ok]
    dirty_idx = [i for i, ok in enumerate(clean_cpu) if not ok]

    cached_layers = feat_cache.get_multilayer([paths[i] for i in clean_idx], device=device)
    dirty_t = torch.tensor(dirty_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        dirty_layers = backbone.extract_multilayer_features(
            images.index_select(0, dirty_t), layers=layers
        )

    bsz = images.shape[0]
    clean_t = torch.tensor(clean_idx, dtype=torch.long, device=device)
    merged: List[torch.Tensor] = []
    for c_feat, d_feat in zip(cached_layers, dirty_layers):
        buf = torch.empty((bsz, c_feat.shape[-1]), device=device, dtype=torch.float32)
        buf.index_copy_(0, clean_t, c_feat.float())
        buf.index_copy_(0, dirty_t, d_feat.float())
        merged.append(buf)
    return merged


def run_train_scr(cfg: ProjectConfig) -> Path:
    """
    训练轻量化 SCR 2.0 风格对比度量器（v3：BF16 冻结基座 + 全量 DINOv2 特征预缓存 + RAM 缓存，提速 10x+）。
    """
    cfg.validate()
    scr_cfg = cfg.scr
    stage_dir = cfg.scr_dir
    logging_dir = stage_dir / cfg.runtime.logging_dir

    mp = resolve_mixed_precision(scr_cfg.mixed_precision)
    accelerator = Accelerator(
        gradient_accumulation_steps=scr_cfg.gradient_accumulation_steps,
        mixed_precision=mp,
        log_with=cfg.runtime.report_to,
        project_dir=str(logging_dir),
    )

    if accelerator.is_main_process:
        stage_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    logging.basicConfig(
        filename=str(stage_dir / "scr_training.log"),
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        force=True,
    )

    if cfg.seed is not None:
        set_seed(cfg.seed)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if scr_cfg.resume_ckpt_path:
        scr = load_scr_checkpoint(
            scr_cfg.resume_ckpt_path,
            mode="training",
            neg_chunk=scr_cfg.neg_chunk,
            map_location="cpu",
        )
        if accelerator.is_main_process:
            print(f"Resumed SCR 2.0 weights from {scr_cfg.resume_ckpt_path}", flush=True)
    else:
        scr = build_scr(scr_cfg=scr_cfg, mode="training", device="cpu")

    index = MatrixIndex(
        data_root=cfg.data.root,
        phase=cfg.data.phase,
        layout=cfg.data.layout,
        require_content=False,
    )
    train_styles, _ = index.split_styles(
        val_holdout_styles=cfg.data.val_holdout_styles, seed=cfg.seed
    )
    dataset = MatrixSCRDataset(
        index=index,
        data_cfg=cfg.data,
        num_neg=scr_cfg.num_neg,
        allowed_styles=train_styles,
    )

    if len(dataset) < scr_cfg.train_batch_size:
        raise ValueError(
            f"SCR 有效样本总数（{len(dataset)}）小于 scr.train_batch_size（{scr_cfg.train_batch_size}）。"
        )

    if accelerator.is_main_process:
        trainable_m = sum(p.numel() for p in scr.parameters() if p.requires_grad) / 1e6
        ram_flag = "RAM" if dataset._ram_cache is not None else "Disk"
        msg = (
            f"SCR 2.0 v3 ({scr_cfg.vision_backbone}, {trainable_m:.2f}M trainable params) "
            f"loaded from '{index.styles_dir}' (layout={index.layout}, io={ram_flag}): "
            f"{len(dataset)} samples, {dataset.num_styles} styles, "
            f"{len(dataset.valid_contents)} valid contents "
            f"(skipped {dataset.skipped_contents} contents with < {scr_cfg.num_neg + 1} styles), "
            f"mixed_precision={mp}."
        )
        print(msg, flush=True)
        logging.info(msg)

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=scr_cfg.train_batch_size,
        num_workers=scr_cfg.num_workers,
        drop_last=len(dataset) >= scr_cfg.train_batch_size,
        persistent_workers=scr_cfg.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_matrix_batch,
    )

    learning_rate = scr_cfg.learning_rate
    if scr_cfg.scale_lr:
        learning_rate = (
            learning_rate
            * scr_cfg.gradient_accumulation_steps
            * scr_cfg.train_batch_size
            * accelerator.num_processes
        )

    trainable_params = [p for p in scr.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        betas=(scr_cfg.adam_beta1, scr_cfg.adam_beta2),
        weight_decay=scr_cfg.adam_weight_decay,
        eps=scr_cfg.adam_epsilon,
        fused=torch.cuda.is_available(),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=scr_cfg.lr_scheduler,
        num_warmup_steps=scr_cfg.lr_warmup_steps * scr_cfg.gradient_accumulation_steps,
        num_training_steps=scr_cfg.max_train_steps * scr_cfg.gradient_accumulation_steps,
    )

    scr, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        scr, optimizer, dataloader, lr_scheduler
    )

    unwrapped_scr = accelerator.unwrap_model(scr)
    if mp == "bf16" and torch.cuda.is_available():
        unwrapped_scr.vision_backbone.set_compute_dtype(torch.bfloat16)

    # 启动期一次性预提取全数据集冻结 DINOv2 多层特征
    feat_cache: VisionFeatureCache | None = None
    if cfg.data.feature_cache:
        t_cache = time.time()
        all_paths = dataset.collect_all_image_paths()
        feat_cache = VisionFeatureCache.build(
            backbone=unwrapped_scr.vision_backbone,
            paths=all_paths,
            image_loader=dataset.load_clean_norm,
            device=accelerator.device,
            need_style_patches=False,
            multilayer_layers=scr_cfg.feature_layers,
            batch_size=128,
        )
        neg_skipped = dataset.enable_cached_neg_skip()
        if accelerator.is_main_process:
            print(
                f"SCR VisionFeatureCache built for {len(all_paths)} images in {time.time() - t_cache:.2f}s"
                f"{' (zero-copy neg skip enabled)' if neg_skipped else ''}.",
                flush=True,
            )

    if accelerator.is_main_process:
        accelerator.init_trackers(f"{cfg.project_name}_scr")
        save_config(cfg, stage_dir / "resolved_config.yaml")

    progress_bar = tqdm(
        range(scr_cfg.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="SCR 2.0 Steps",
    )

    num_update_steps_per_epoch = max(
        1, math.ceil(len(dataloader) / scr_cfg.gradient_accumulation_steps)
    )
    num_train_epochs = math.ceil(scr_cfg.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    train_start = time.time()
    last_saved_step = 0
    latest_ckpt: Path | None = None

    for _ in range(num_train_epochs):
        train_loss = 0.0
        for samples in dataloader:
            scr.train()
            anchor_images = samples["target_image"]
            pos_images = samples["pos_image"]
            neg_images = samples.get("neg_images")
            dev = anchor_images.device

            anchor_raw = _resolve_multilayer_batch(
                feat_cache=feat_cache,
                backbone=unwrapped_scr.vision_backbone,
                images=anchor_images,
                paths=samples["target_image_path"],
                clean_mask=samples.get("anchor_clean"),
                layers=scr_cfg.feature_layers,
                device=dev,
            )
            pos_raw = _resolve_multilayer_batch(
                feat_cache=feat_cache,
                backbone=unwrapped_scr.vision_backbone,
                images=pos_images,
                paths=samples["pos_image_path"],
                clean_mask=samples.get("pos_clean"),
                layers=scr_cfg.feature_layers,
                device=dev,
            )
            neg_raw = None
            neg_clean = samples.get("neg_clean")
            if feat_cache is not None and (neg_clean is None or bool(neg_clean.all())):
                flat_neg_paths = [p for row in samples["neg_image_paths"] for p in row]
                neg_raw = feat_cache.get_multilayer(flat_neg_paths, device=dev)

            with accelerator.accumulate(scr):
                sample_emb, pos_emb, neg_emb = scr(
                    sample_imgs=None if anchor_raw is not None else anchor_images,
                    pos_imgs=None if pos_raw is not None else pos_images,
                    neg_imgs=None if neg_raw is not None else neg_images,
                    sample_raw_feats=anchor_raw,
                    pos_raw_feats=pos_raw,
                    neg_raw_feats=neg_raw,
                    num_neg=scr_cfg.num_neg,
                )
                loss = unwrapped_scr.calculate_nce_loss(
                    sample_s=sample_emb,
                    pos_s=pos_emb,
                    neg_s=neg_emb,
                )

                avg_loss = accelerator.gather(
                    loss.detach().repeat(anchor_images.shape[0])
                ).mean()
                train_loss += avg_loss.item() / scr_cfg.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, scr_cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                lr_now = lr_scheduler.get_last_lr()[0]
                accelerator.log(
                    {"train/scr_loss": train_loss, "train/lr": lr_now},
                    step=global_step,
                )

                if (
                    accelerator.is_main_process
                    and global_step > 0
                    and global_step % scr_cfg.log_interval == 0
                ):
                    log_train_progress(
                        format_train_progress(
                            step=global_step,
                            total=scr_cfg.max_train_steps,
                            loss=train_loss,
                            started_at=train_start,
                        )
                    )
                train_loss = 0.0

                if (
                    accelerator.is_main_process
                    and global_step % scr_cfg.ckpt_interval == 0
                ):
                    latest_ckpt = save_scr_checkpoint(
                        stage_dir / f"scr_{global_step}.pth",
                        scr=unwrapped_scr,
                        scr_cfg=scr_cfg,
                        global_step=global_step,
                    )
                    last_saved_step = global_step
                    msg = (
                        f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}] "
                        f"Saved SCR 2.0 checkpoint at step {global_step} -> {latest_ckpt}"
                    )
                    logging.info(msg)
                    print(msg, flush=True)

            progress_bar.set_postfix(
                step_loss=f"{loss.detach().item():.4f}",
                lr=f"{lr_scheduler.get_last_lr()[0]:.2e}",
            )

            if global_step >= scr_cfg.max_train_steps:
                break
        if global_step >= scr_cfg.max_train_steps:
            break

    if accelerator.is_main_process:
        if global_step > last_saved_step or latest_ckpt is None:
            latest_ckpt = save_scr_checkpoint(
                stage_dir / f"scr_{global_step}.pth",
                scr=unwrapped_scr,
                scr_cfg=scr_cfg,
                global_step=global_step,
            )
        write_stage_done(
            stage_dir,
            {
                "stage": "scr",
                "global_step": global_step,
                "ckpt_path": str(latest_ckpt),
            },
        )

    accelerator.end_training()
    assert latest_ckpt is not None
    return latest_ckpt
