from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple
import yaml

SUPPORTED_LAYOUTS = ("auto", "standard", "legacy")
SUPPORTED_RESIZE_MODES = ("stretch", "pad")
SUPPORTED_INTERPOLATIONS = ("bilinear", "bicubic", "nearest")
SUPPORTED_MIXED_PRECISION = ("no", "fp16", "bf16")
SUPPORTED_PRESETS = ("T", "S", "B", "custom")
SUPPORTED_LR_SCHEDULERS = ("wsd", "cosine", "constant", "linear")
SUPPORTED_SAMPLING_MODES = ("auto", "fast", "hd")
SUPPORTED_SOLVERS = ("euler", "heun", "heun_fast")

# PixelDiT-Flow v3 预设规格表（均保持 d_head = 64 以满血命中 FlashAttention-2 / SDPA）
MODEL_PRESETS = {
    "T": {
        "hidden_size": 256,
        "depth": 6,
        "num_heads": 4,
        "mlp_dim": 640,
        "ada_dim": 128,
        "pixel_depth": 1,
        "pixel_dim": 16,
        "repa_layer": 2,
        "style_latent_tokens": 32,
        "style_resampler_depth": 1,
        "stroke_tokens_k": 16,
    },
    "S": {
        "hidden_size": 576,
        "depth": 12,
        "num_heads": 9,
        "mlp_dim": 1536,
        "ada_dim": 256,
        "pixel_depth": 2,
        "pixel_dim": 24,
        "repa_layer": 4,
        "style_latent_tokens": 64,
        "style_resampler_depth": 2,
        "stroke_tokens_k": 32,
    },
    "B": {
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_dim": 2048,
        "ada_dim": 256,
        "pixel_depth": 2,
        "pixel_dim": 32,
        "repa_layer": 4,
        "style_latent_tokens": 64,
        "style_resampler_depth": 2,
        "stroke_tokens_k": 32,
    },
}


@dataclass
class DataConfig:
    root: str = "data/example"
    layout: str = "auto"
    phase: str = "train"
    resolution: int = 128
    resize_mode: str = "pad"
    pad_ratio: float = 0.0
    interpolation: str = "bilinear"
    background_color: Tuple[int, int, int] = (255, 255, 255)
    val_holdout_styles: int = 2
    val_num_samples: int = 4
    # 原生 K-shot 多参考风格训练（1 表示 1-shot；>1 时按 batch 随机采样 1..K 张同风格不同内容图）
    max_style_refs: int = 1
    # 彩色资产成对色彩增强（对 style 与 target 同步施加随机色相/饱和度/亮度偏移，泛化未见色板）
    paired_color_aug: bool = False
    color_aug_hue: float = 0.25
    color_aug_sat: float = 0.25
    color_aug_val: float = 0.15
    # 成对空间几何增强（对 content/target 同步缩放平移及可选翻转，style 同步缩放保线宽 + 独立平移翻转，
    # 打破小样本按坐标查表过拟合，强迫 3D RoPE 真正逐像素追踪几何并迁移纯风格笔触）
    paired_spatial_aug: bool = False
    spatial_aug_prob: float = 0.5
    spatial_aug_hflip: bool = False
    spatial_aug_scale: Tuple[float, float] = (1.0, 1.0)
    spatial_aug_shift: float = 0.025
    # v3 成对微几何风格增广（仅同步作用于 style_refs + target，保持 content 原图不动）：
    # - paired_morph_aug / morph_shear_prob: 随机斜切 + 纵横字面缩放。英文 em_baseline 训练必须关闭。
    # - morph_pixelate_prob 与 stencil_cut_prob 与斜切开关解耦，关闭字面缩放时仍可单独打开。
    paired_morph_aug: bool = False
    morph_shear_prob: float = 0.0
    morph_shear_deg: float = 8.0
    morph_aspect_range: float = 0.10
    morph_pixelate_prob: float = 0.0
    morph_pixelate_grids: Tuple[int, ...] = (24, 32, 40, 48)
    # 成对模板切缝：仅同步作用于 target + style_refs，content 保持实心。
    # 与 paired_morph_aug 解耦，以便在关闭斜切/字面缩放（守住 em_baseline）时仍能合成断缝。
    stencil_cut_prob: float = 0.0
    stencil_cut_gaps: Tuple[int, ...] = (2, 3)
    # v3 跨语种多参考防串味增广（仅在数据集自动检测到 >=2 种 Unicode 语系如中+英且 max_style_refs>=2 时生效）：
    # 以该概率在次要参考槽位混入异风格异语种干扰参考（k=1 时永不混入），强迫 Content-Aware 路由器学会按语种隔离风格
    mixed_script_ref_prob: float = 0.25
    # v3 极速内存与特征缓存（中小数据集零磁盘 I/O + 冻结 DINOv2 特征预缓存）
    ram_cache_max_images: int = 20000
    feature_cache: bool = True
    # 可选风格重采样权重表 JSON（style_id -> 整数重复倍率 1..8），用于 Phase 2 对稀缺/偏门风格加权课程采样
    style_weights_path: Optional[str] = None

    def validate(self, patch_size: int = 8) -> None:
        if self.layout not in SUPPORTED_LAYOUTS:
            raise ValueError(
                f"data.layout 必须是 {SUPPORTED_LAYOUTS} 之一，当前为 {self.layout!r}"
            )
        if self.resolution < 32 or self.resolution % patch_size != 0:
            raise ValueError(
                f"data.resolution 必须 >= 32 且能被 model.patch_size({patch_size}) 整除，"
                f"当前为 {self.resolution}"
            )
        if self.resize_mode not in SUPPORTED_RESIZE_MODES:
            raise ValueError(
                f"data.resize_mode 必须是 {SUPPORTED_RESIZE_MODES} 之一，当前为 {self.resize_mode!r}"
            )
        if not (0.0 <= float(self.pad_ratio) < 0.45):
            raise ValueError(
                f"data.pad_ratio 必须在 [0.0, 0.45) 范围内，当前为 {self.pad_ratio}"
            )
        if self.interpolation not in SUPPORTED_INTERPOLATIONS:
            raise ValueError(
                f"data.interpolation 必须是 {SUPPORTED_INTERPOLATIONS} 之一，当前为 {self.interpolation!r}"
            )
        if len(self.background_color) != 3 or any(
            not (0 <= int(c) <= 255) for c in self.background_color
        ):
            raise ValueError(
                f"data.background_color 必须是 3 个 0~255 的整数，当前为 {self.background_color!r}"
            )
        if self.val_holdout_styles < 0:
            raise ValueError("data.val_holdout_styles 不能为负数")
        if self.val_num_samples < 0:
            raise ValueError("data.val_num_samples 不能为负数")
        if self.max_style_refs < 1:
            raise ValueError("data.max_style_refs 必须 >= 1")
        if len(self.spatial_aug_scale) != 2 or not (
            0.5 <= float(self.spatial_aug_scale[0]) <= float(self.spatial_aug_scale[1]) <= 1.5
        ):
            raise ValueError(
                f"data.spatial_aug_scale 需满足 0.5 <= lo <= hi <= 1.5，当前为 {self.spatial_aug_scale!r}"
            )
        if not (0.0 <= float(self.spatial_aug_shift) < 0.3):
            raise ValueError(
                f"data.spatial_aug_shift 需位于 [0.0, 0.3)，当前为 {self.spatial_aug_shift}"
            )
        if not (0.0 <= float(self.spatial_aug_prob) <= 1.0):
            raise ValueError(
                f"data.spatial_aug_prob 需位于 [0.0, 1.0]，当前为 {self.spatial_aug_prob}"
            )
        if not (0.0 <= float(self.morph_shear_prob) <= 1.0):
            raise ValueError(
                f"data.morph_shear_prob 需位于 [0.0, 1.0]，当前为 {self.morph_shear_prob}"
            )
        if not (0.0 <= float(self.morph_pixelate_prob) <= 1.0):
            raise ValueError(
                f"data.morph_pixelate_prob 需位于 [0.0, 1.0]，当前为 {self.morph_pixelate_prob}"
            )
        if float(self.morph_shear_deg) < 0.0 or float(self.morph_shear_deg) > 25.0:
            raise ValueError(
                f"data.morph_shear_deg 需位于 [0.0, 25.0]，当前为 {self.morph_shear_deg}"
            )
        if float(self.morph_aspect_range) < 0.0 or float(self.morph_aspect_range) > 0.35:
            raise ValueError(
                f"data.morph_aspect_range 需位于 [0.0, 0.35]，当前为 {self.morph_aspect_range}"
            )
        if not self.morph_pixelate_grids or any(int(g) < 8 for g in self.morph_pixelate_grids):
            raise ValueError(
                f"data.morph_pixelate_grids 必须为非空且元素 >= 8，当前为 {self.morph_pixelate_grids!r}"
            )
        if not (0.0 <= float(self.stencil_cut_prob) <= 1.0):
            raise ValueError(
                f"data.stencil_cut_prob 需位于 [0.0, 1.0]，当前为 {self.stencil_cut_prob}"
            )
        if not self.stencil_cut_gaps or any(not (1 <= int(g) <= 6) for g in self.stencil_cut_gaps):
            raise ValueError(
                f"data.stencil_cut_gaps 必须为非空且元素位于 [1, 6]，当前为 {self.stencil_cut_gaps!r}"
            )
        if self.style_weights_path is not None and not str(self.style_weights_path).strip():
            raise ValueError("data.style_weights_path 不能为空字符串")


@dataclass
class ModelConfig:
    preset: str = "B"
    patch_size: int = 8
    hidden_size: Optional[int] = None
    depth: Optional[int] = None
    num_heads: Optional[int] = None
    mlp_dim: Optional[int] = None
    ada_dim: Optional[int] = None
    pixel_depth: Optional[int] = None
    pixel_dim: Optional[int] = None
    vision_backbone: str = "facebook/dinov2-small"
    vision_input_size: int = 140
    style_token_drop_prob: float = 0.15
    repa_layer: Optional[int] = None
    # v3 架构核心参数：固定长度 Perceiver 风格重采样器 + 前景笔触字典 + 空间形变对齐器
    style_latent_tokens: Optional[int] = None
    style_resampler_depth: Optional[int] = None
    stroke_tokens_k: Optional[int] = None
    use_spatial_warper: bool = True

    def resolved_dims(self) -> dict[str, int]:
        """将 preset 默认值与显式覆盖字段合并为完整的网络维度表。"""
        base = MODEL_PRESETS.get(self.preset, MODEL_PRESETS["B"])
        return {
            "hidden_size": int(
                self.hidden_size if self.hidden_size is not None else base["hidden_size"]
            ),
            "depth": int(self.depth if self.depth is not None else base["depth"]),
            "num_heads": int(
                self.num_heads if self.num_heads is not None else base["num_heads"]
            ),
            "mlp_dim": int(self.mlp_dim if self.mlp_dim is not None else base["mlp_dim"]),
            "ada_dim": int(self.ada_dim if self.ada_dim is not None else base["ada_dim"]),
            "pixel_depth": int(
                self.pixel_depth if self.pixel_depth is not None else base["pixel_depth"]
            ),
            "pixel_dim": int(
                self.pixel_dim if self.pixel_dim is not None else base["pixel_dim"]
            ),
            "repa_layer": int(
                self.repa_layer if self.repa_layer is not None else base["repa_layer"]
            ),
            "style_latent_tokens": int(
                self.style_latent_tokens
                if self.style_latent_tokens is not None
                else base["style_latent_tokens"]
            ),
            "style_resampler_depth": int(
                self.style_resampler_depth
                if self.style_resampler_depth is not None
                else base["style_resampler_depth"]
            ),
            "stroke_tokens_k": int(
                self.stroke_tokens_k
                if self.stroke_tokens_k is not None
                else base["stroke_tokens_k"]
            ),
        }

    def validate(self) -> None:
        if self.preset not in SUPPORTED_PRESETS:
            raise ValueError(
                f"model.preset 必须是 {SUPPORTED_PRESETS} 之一，当前为 {self.preset!r}"
            )
        if self.patch_size not in (4, 8, 16):
            raise ValueError(
                f"model.patch_size 必须是 4 / 8 / 16 之一，当前为 {self.patch_size}"
            )
        dims = self.resolved_dims()
        if dims["hidden_size"] % dims["num_heads"] != 0:
            raise ValueError(
                f"hidden_size({dims['hidden_size']}) 必须能被 num_heads({dims['num_heads']}) 整除"
            )
        head_dim = dims["hidden_size"] // dims["num_heads"]
        if head_dim < 16 or head_dim % 8 != 0:
            raise ValueError(
                f"3D RoPE 要求单头维度 head_dim >= 16 且为 8 的倍数，当前为 {head_dim}"
            )
        if dims["depth"] < 1:
            raise ValueError("model.depth 必须 >= 1")
        if dims["pixel_depth"] < 1:
            raise ValueError("model.pixel_depth 必须 >= 1")
        if dims["pixel_dim"] < 8 or dims["pixel_dim"] % 4 != 0:
            raise ValueError("model.pixel_dim 必须 >= 8 且为 4 的倍数")
        if not (1 <= dims["repa_layer"] <= dims["depth"]):
            raise ValueError(
                f"model.repa_layer({dims['repa_layer']}) 必须在 [1, depth({dims['depth']})] 范围内"
            )
        if not (0.0 <= float(self.style_token_drop_prob) < 1.0):
            raise ValueError("model.style_token_drop_prob 必须在 [0.0, 1.0) 范围内")
        if self.vision_input_size < 28 or self.vision_input_size % 14 != 0:
            raise ValueError(
                f"model.vision_input_size 必须为 14 的正倍数（适配 DINOv2 patch=14），当前为 {self.vision_input_size}"
            )
        if dims["style_latent_tokens"] < 4:
            raise ValueError("model.style_latent_tokens 必须 >= 4")
        if dims["style_resampler_depth"] < 1:
            raise ValueError("model.style_resampler_depth 必须 >= 1")
        if dims["stroke_tokens_k"] < 4:
            raise ValueError("model.stroke_tokens_k 必须 >= 4")


@dataclass
class SCRConfig:
    """SCR 2.0：冻结 DINOv2 多层特征 + 轻量多层投影头（无 BatchNorm，跨内容正样本）。"""

    enabled: bool = True
    pretrained_ckpt_path: Optional[str] = None
    resume_ckpt_path: Optional[str] = None
    vision_backbone: str = "facebook/dinov2-small"
    vision_input_size: int = 140
    feature_layers: Tuple[int, ...] = (3, 6, 9, 12)
    proj_dim: int = 256
    num_neg: int = 16
    temperature: float = 0.07
    neg_chunk: int = 8
    train_batch_size: int = 16
    num_workers: int = 0
    max_train_steps: int = 10000
    ckpt_interval: int = 2000
    log_interval: int = 50
    learning_rate: float = 3e-4
    scale_lr: bool = False
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 500
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"

    def validate(self, resolution: int) -> None:
        if self.num_neg < 1:
            raise ValueError(f"scr.num_neg 必须 >= 1，当前为 {self.num_neg}")
        if self.train_batch_size < 1:
            raise ValueError(f"scr.train_batch_size 必须 >= 1，当前为 {self.train_batch_size}")
        if resolution < 32:
            raise ValueError(f"SCR 要求 resolution >= 32，当前为 {resolution}")
        if self.mixed_precision not in SUPPORTED_MIXED_PRECISION:
            raise ValueError(
                f"scr.mixed_precision 必须是 {SUPPORTED_MIXED_PRECISION} 之一"
            )
        if self.lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
            raise ValueError(
                f"scr.lr_scheduler 必须是 {SUPPORTED_LR_SCHEDULERS} 之一"
            )
        if not self.feature_layers:
            raise ValueError("scr.feature_layers 不能为空")
        if self.max_train_steps < 1:
            raise ValueError("scr.max_train_steps 必须 >= 1")
        if self.ckpt_interval < 1:
            raise ValueError("scr.ckpt_interval 必须 >= 1")


@dataclass
class FlowStageConfig:
    """PixelDiT-Flow 主训练阶段配置（Rectified Flow + JiT x0-pred + REPA + EMA + Dual-CFG）。"""

    enabled: bool = True
    resume_ckpt_dir: Optional[str] = None
    train_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    num_workers: int = 0
    max_train_steps: int = 80000
    ckpt_interval: int = 10000
    eval_interval: int = 2000
    eval_fast_steps: int = 4
    eval_hd_steps: int = 10
    eval_content_scale: float = 2.0
    eval_style_scale: float = 2.5
    log_interval: int = 50
    learning_rate: float = 2e-4
    scale_lr: bool = False
    lr_scheduler: str = "wsd"
    lr_warmup_steps: int = 2000
    wsd_decay_ratio: float = 0.15
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    use_compile: bool = False
    # EMA（指数移动平均权重）
    ema_decay: float = 0.999
    ema_warmup_steps: int = 1000
    use_ema_for_eval: bool = True
    # Flow Matching / JiT 像素空间时间偏移采样与有界损失权重
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0
    time_shift: float = 6.0
    uniform_t_prob: float = 0.2
    # 在偏移后的 t 上额外替换出一段 Uniform(0, low_t_max)，专补 1~2px 断缝/像素角的低噪梯度。
    low_t_prob: float = 0.0
    low_t_max: float = 0.3
    t_min: float = 0.05
    loss_t_floor: float = 1.0
    # 独立双条件 CFG Dropout（分离骨架与风格引导）
    drop_content_prob: float = 0.10
    drop_style_prob: float = 0.10
    drop_both_prob: float = 0.08
    # 现代多目标损失系数
    repa_coefficient: float = 0.5
    perceptual_coefficient: float = 0.1
    edge_coefficient: float = 0.1
    perceptual_t_max: float = 0.9
    # <=0 时边缘损失沿用 perceptual_t_max。单独放大后，1~2px 断缝在中低噪步仍有门控权重，
    # 感知损失与 SCR 的门控保持不变。
    edge_t_max: float = 0.0
    warp_reg_coefficient: float = 0.01

    def validate(self, stage_name: str = "phase1") -> None:
        if self.train_batch_size < 1:
            raise ValueError(f"{stage_name}.train_batch_size 必须 >= 1")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"{stage_name}.gradient_accumulation_steps 必须 >= 1")
        if self.max_train_steps < 1:
            raise ValueError(f"{stage_name}.max_train_steps 必须 >= 1")
        if self.ckpt_interval < 1:
            raise ValueError(f"{stage_name}.ckpt_interval 必须 >= 1")
        if self.eval_interval < 0:
            raise ValueError(f"{stage_name}.eval_interval 不能为负数")
        if self.mixed_precision not in SUPPORTED_MIXED_PRECISION:
            raise ValueError(
                f"{stage_name}.mixed_precision 必须是 {SUPPORTED_MIXED_PRECISION} 之一"
            )
        if self.lr_scheduler not in SUPPORTED_LR_SCHEDULERS:
            raise ValueError(
                f"{stage_name}.lr_scheduler 必须是 {SUPPORTED_LR_SCHEDULERS} 之一"
            )
        drop_sum = (
            float(self.drop_content_prob)
            + float(self.drop_style_prob)
            + float(self.drop_both_prob)
        )
        if not (0.0 <= drop_sum < 1.0):
            raise ValueError(
                f"{stage_name} 的 CFG dropout 概率之和必须位于 [0, 1)，当前为 {drop_sum:.3f}"
            )
        if not (0.0 <= float(self.ema_decay) < 1.0):
            raise ValueError(f"{stage_name}.ema_decay 必须位于 [0, 1)")
        if not (0.0 < float(self.t_min) < 0.5):
            raise ValueError(f"{stage_name}.t_min 必须位于 (0, 0.5)")
        if float(self.time_shift) <= 0.0:
            raise ValueError(f"{stage_name}.time_shift 必须 > 0")
        if not (0.0 <= float(self.uniform_t_prob) <= 1.0):
            raise ValueError(f"{stage_name}.uniform_t_prob 必须位于 [0, 1]")
        if not (0.0 <= float(self.low_t_prob) <= 1.0):
            raise ValueError(f"{stage_name}.low_t_prob 必须位于 [0, 1]")
        if not (0.05 <= float(self.low_t_max) <= 1.0):
            raise ValueError(f"{stage_name}.low_t_max 必须位于 [0.05, 1]，当前为 {self.low_t_max}")
        if float(self.loss_t_floor) < 0.0:
            raise ValueError(f"{stage_name}.loss_t_floor 不能为负数")
        if float(self.edge_t_max) < 0.0 or float(self.edge_t_max) > 1.0:
            raise ValueError(
                f"{stage_name}.edge_t_max 必须位于 [0, 1]，当前为 {self.edge_t_max}"
            )

    def resolved_edge_t_max(self) -> float:
        if float(self.edge_t_max) > 0.0:
            return float(self.edge_t_max)
        return float(self.perceptual_t_max)


@dataclass
class Phase2Config(FlowStageConfig):
    """Phase 2：SCR 2.0 风格对比精调 + Shortcut 双档（1–4 步极速 + 8–10 步高画质）联合微调。"""

    phase1_ckpt_dir: str = "auto"
    scr_ckpt_path: str = "auto"
    num_neg: int = 16
    sc_coefficient: float = 0.02
    temperature: float = 0.07
    neg_chunk: int = 8
    # Shortcut 自洽 + CFG 蒸馏（一套权重同时支持 1-4 步免 CFG 极速与 8-10 步满血高画质）
    shortcut_enabled: bool = True
    shortcut_ratio: float = 0.35
    shortcut_coefficient: float = 1.0
    shortcut_edge_coefficient: float = 0.15
    shortcut_curriculum_ratio: float = 0.25
    shortcut_steps: Tuple[int, ...] = (1, 2, 4, 8)
    distill_content_scale_range: Tuple[float, float] = (1.2, 3.0)
    distill_style_scale_range: Tuple[float, float] = (1.5, 3.5)
    # Phase 2 默认训练步数与学习率
    max_train_steps: int = 20000
    ckpt_interval: int = 2000
    eval_interval: int = 2000
    learning_rate: float = 5e-5
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 1000
    repa_coefficient: float = 0.25

    def validate(self, stage_name: str = "phase2") -> None:
        super().validate(stage_name=stage_name)
        if self.num_neg < 1:
            raise ValueError(f"{stage_name}.num_neg 必须 >= 1")
        if not (0.0 <= float(self.shortcut_ratio) <= 1.0):
            raise ValueError(f"{stage_name}.shortcut_ratio 必须在 [0, 1] 之间")
        if self.shortcut_enabled and not self.shortcut_steps:
            raise ValueError(f"{stage_name}.shortcut_steps 不能为空")


@dataclass
class SamplingConfig:
    """统一双档流匹配采样配置：fast（1-4 步免 CFG 极速） / hd（8-10 步 Dual-CFG + APG）。"""

    mode: str = "auto"
    num_inference_steps: int = 10
    solver: str = "heun_fast"
    content_guidance_scale: float = 2.0
    style_guidance_scale: float = 2.5
    use_apg: bool = True
    apg_eta: float = 0.1
    apg_norm_threshold: float = 2.5
    # 在未偏移进度空间 u∈[0,1] 上生效：默认前 75% 高中噪步开启双 CFG，末段 25% 纯无引导精修防过冲颗粒
    guidance_interval: Tuple[float, float] = (0.25, 1.0)
    time_shift: float = 6.0
    use_kv_cache: bool = True
    use_ema: bool = True

    def resolved_mode(self, steps: Optional[int] = None) -> str:
        n = int(steps if steps is not None else self.num_inference_steps)
        if self.mode == "auto":
            return "fast" if n <= 4 else "hd"
        return self.mode

    def validate(self) -> None:
        if self.mode not in SUPPORTED_SAMPLING_MODES:
            raise ValueError(
                f"sampling.mode 必须是 {SUPPORTED_SAMPLING_MODES} 之一，当前为 {self.mode!r}"
            )
        if self.solver not in SUPPORTED_SOLVERS:
            raise ValueError(
                f"sampling.solver 必须是 {SUPPORTED_SOLVERS} 之一，当前为 {self.solver!r}"
            )
        if self.num_inference_steps < 1:
            raise ValueError("sampling.num_inference_steps 必须 >= 1")
        lo, hi = self.guidance_interval
        if not (0.0 <= float(lo) <= float(hi) <= 1.0):
            raise ValueError(
                f"sampling.guidance_interval 需满足 0 <= lo <= hi <= 1，当前为 {self.guidance_interval}"
            )
        if float(self.time_shift) <= 0.0:
            raise ValueError("sampling.time_shift 必须 > 0")


@dataclass
class RuntimeConfig:
    output_root: str = "runs"
    report_to: str = "tensorboard"
    use_accelerate: bool = False
    logging_dir: str = "logs"


@dataclass
class ProjectConfig:
    project_name: str = "default_project"
    seed: int = 123
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    scr: SCRConfig = field(default_factory=SCRConfig)
    phase1: FlowStageConfig = field(default_factory=FlowStageConfig)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def project_dir(self) -> Path:
        return Path(self.runtime.output_root) / self.project_name

    @property
    def scr_dir(self) -> Path:
        return self.project_dir / "scr"

    @property
    def phase1_dir(self) -> Path:
        return self.project_dir / "phase1"

    @property
    def phase2_dir(self) -> Path:
        return self.project_dir / "phase2"

    @property
    def outputs_dir(self) -> Path:
        return self.project_dir / "outputs"

    def validate(self) -> None:
        if not self.project_name or not self.project_name.strip():
            raise ValueError("project_name 不能为空")
        self.model.validate()
        self.data.validate(patch_size=self.model.patch_size)
        self.scr.validate(resolution=self.data.resolution)
        self.phase1.validate(stage_name="phase1")
        self.phase2.validate(stage_name="phase2")
        self.sampling.validate()

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["data"]["background_color"] = list(self.data.background_color)
        raw["data"]["spatial_aug_scale"] = list(self.data.spatial_aug_scale)
        raw["data"]["morph_pixelate_grids"] = list(self.data.morph_pixelate_grids)
        raw["data"]["stencil_cut_gaps"] = list(self.data.stencil_cut_gaps)
        raw["scr"]["feature_layers"] = list(self.scr.feature_layers)
        raw["phase2"]["shortcut_steps"] = list(self.phase2.shortcut_steps)
        raw["phase2"]["distill_content_scale_range"] = list(
            self.phase2.distill_content_scale_range
        )
        raw["phase2"]["distill_style_scale_range"] = list(
            self.phase2.distill_style_scale_range
        )
        raw["sampling"]["guidance_interval"] = list(self.sampling.guidance_interval)
        return raw


def _coerce_section(cls, raw: Optional[dict[str, Any]]):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise TypeError(f"配置段 {cls.__name__} 必须为字典，收到 {type(raw).__name__}")
    valid_names = {f.name for f in fields(cls)}
    unknown = sorted(set(raw.keys()) - valid_names)
    if unknown:
        raise ValueError(f"配置段 {cls.__name__} 包含未知字段: {unknown}")
    kwargs = dict(raw)
    # 兼容 PyYAML 1.1 将不带小数点的科学计数法（如 5e-5）解析为 str 的情况
    field_defaults = {f.name: f.default for f in fields(cls)}
    for fname, fval in list(kwargs.items()):
        if isinstance(fval, str) and isinstance(field_defaults.get(fname), float):
            try:
                kwargs[fname] = float(fval)
            except ValueError:
                pass
    if cls is DataConfig:
        if "background_color" in kwargs:
            kwargs["background_color"] = tuple(int(c) for c in kwargs["background_color"])
        if "spatial_aug_scale" in kwargs:
            kwargs["spatial_aug_scale"] = tuple(float(v) for v in kwargs["spatial_aug_scale"])
        if "morph_pixelate_grids" in kwargs:
            kwargs["morph_pixelate_grids"] = tuple(int(g) for g in kwargs["morph_pixelate_grids"])
        if "stencil_cut_gaps" in kwargs:
            kwargs["stencil_cut_gaps"] = tuple(int(g) for g in kwargs["stencil_cut_gaps"])
    if cls is SCRConfig and "feature_layers" in kwargs:
        kwargs["feature_layers"] = tuple(int(v) for v in kwargs["feature_layers"])
    if cls is Phase2Config:
        if "shortcut_steps" in kwargs:
            kwargs["shortcut_steps"] = tuple(int(v) for v in kwargs["shortcut_steps"])
        if "distill_content_scale_range" in kwargs:
            kwargs["distill_content_scale_range"] = tuple(
                float(v) for v in kwargs["distill_content_scale_range"]
            )
        if "distill_style_scale_range" in kwargs:
            kwargs["distill_style_scale_range"] = tuple(
                float(v) for v in kwargs["distill_style_scale_range"]
            )
    if cls is SamplingConfig and "guidance_interval" in kwargs:
        kwargs["guidance_interval"] = tuple(float(v) for v in kwargs["guidance_interval"])
    return cls(**kwargs)


def config_from_dict(raw: dict[str, Any]) -> ProjectConfig:
    if not isinstance(raw, dict):
        raise TypeError("顶层 YAML 配置必须为字典。")
    valid_top = {f.name for f in fields(ProjectConfig)}
    unknown = sorted(set(raw.keys()) - valid_top)
    if unknown:
        raise ValueError(f"顶层配置包含未知字段: {unknown}")

    cfg = ProjectConfig(
        project_name=raw.get("project_name", "default_project"),
        seed=int(raw.get("seed", 123)),
        data=_coerce_section(DataConfig, raw.get("data")),
        model=_coerce_section(ModelConfig, raw.get("model")),
        scr=_coerce_section(SCRConfig, raw.get("scr")),
        phase1=_coerce_section(FlowStageConfig, raw.get("phase1")),
        phase2=_coerce_section(Phase2Config, raw.get("phase2")),
        sampling=_coerce_section(SamplingConfig, raw.get("sampling")),
        runtime=_coerce_section(RuntimeConfig, raw.get("runtime")),
    )
    cfg.validate()
    return cfg


def apply_overrides(raw: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """应用 CLI `--set a.b=value` 覆盖项，值按 YAML 语义解析。"""
    merged = yaml.safe_load(yaml.safe_dump(raw)) or {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 参数格式错误（需要 key=value）: {item!r}")
        key_path, value_str = item.split("=", 1)
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            raise ValueError(f"--set 键名为空: {item!r}")
        parsed_value = yaml.safe_load(value_str)
        cursor = merged
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = parsed_value
    return merged


def load_config(
    config_path: str | Path,
    overrides: Optional[Iterable[str]] = None,
) -> ProjectConfig:
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = apply_overrides(raw, overrides)
    return config_from_dict(raw)


def save_config(cfg: ProjectConfig, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(
            cfg.to_dict(),
            fh,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
    return path
