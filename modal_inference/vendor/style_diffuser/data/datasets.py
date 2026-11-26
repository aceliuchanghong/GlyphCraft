from __future__ import annotations

from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig
from .index import MatrixIndex
from .preprocess import (
    SquareResizeOrPad,
    apply_paired_color_jitter,
    apply_paired_morph_shear_aspect,
    apply_paired_pixel_grid_quantize,
    apply_paired_spatial_aug,
    apply_paired_stencil_cut,
    build_image_transform,
    load_rgb_image,
)


def _apply_topology_style_augs(
    tensors: Sequence[torch.Tensor],
    data_cfg: DataConfig,
    background_color: Tuple[int, int, int],
) -> Tuple[list[torch.Tensor], bool]:
    """
    仅作用于风格侧张量（target + 同源 style_refs，或 SCR 的 anchor + pos）。
    像素格量化与模板切缝都不依赖 paired_morph_aug，因此可以在守住 em_baseline 时单独打开。
    """
    group = list(tensors)
    dirty = False
    if (
        float(data_cfg.stencil_cut_prob) > 0.0
        and random.random() < float(data_cfg.stencil_cut_prob)
    ):
        before = group
        group = apply_paired_stencil_cut(
            group,
            background_color=background_color,
            gap_choices=data_cfg.stencil_cut_gaps,
        )
        if any(a is not b for a, b in zip(before, group)):
            dirty = True
    if (
        float(data_cfg.morph_pixelate_prob) > 0.0
        and random.random() < float(data_cfg.morph_pixelate_prob)
    ):
        group = apply_paired_pixel_grid_quantize(
            group,
            background_color=background_color,
            grid_sizes=data_cfg.morph_pixelate_grids,
        )
        dirty = True
    return group, dirty


def collate_matrix_batch(batch: List[Dict]) -> Dict:
    batched: Dict = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            batched[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], bool):
            batched[key] = torch.tensor(values, dtype=torch.bool)
        else:
            batched[key] = values
    return batched


def _preload_ram_uint8_cache(
    paths: Sequence[Path],
    data_cfg: DataConfig,
) -> Optional[Dict[str, torch.Tensor]]:
    """
    v3 极速内存图像缓存：
    若数据集唯一图片数 <= ram_cache_max_images，则在启动期一次性解码缩放并常驻 CPU uint8 张量
    （10,000 张 128x128 RGB 仅占 ~490MB RAM），训练期间彻底零磁盘 I/O 与零 PNG 解码开销。
    """
    max_imgs = int(getattr(data_cfg, "ram_cache_max_images", 20000))
    if max_imgs <= 0:
        return None
    unique_paths: Dict[str, Path] = {}
    for p in paths:
        unique_paths[str(p)] = p
    if len(unique_paths) > max_imgs:
        return None

    resizer = SquareResizeOrPad(
        resolution=data_cfg.resolution,
        resize_mode=data_cfg.resize_mode,
        pad_ratio=data_cfg.pad_ratio,
        interpolation=data_cfg.interpolation,
        background_color=data_cfg.background_color,
    )
    cache: Dict[str, torch.Tensor] = {}
    for ps, p in unique_paths.items():
        img = resizer(load_rgb_image(p, background_color=data_cfg.background_color))
        arr = np.asarray(img, dtype=np.uint8)
        cache[ps] = torch.from_numpy(arr.copy()).permute(2, 0, 1).contiguous()
    return cache


def _uint8_to_norm_tensor(img_u8: torch.Tensor) -> torch.Tensor:
    return img_u8.float().mul_(2.0 / 255.0).sub_(1.0)


def detect_content_script_group(content_id: str) -> Optional[str]:
    """
    从 content_id 识别单字符所属 Unicode 语系（仅支持单字符或 uXXXX/U+XXXX 码点命名；
    图标/插画词条如 'bookmark'/'square-pen' 自动返回 None，不触发跨语种干扰采样）。
    """
    cid = str(content_id).strip()
    if not cid:
        return None
    cp: Optional[int] = None
    if len(cid) == 1:
        cp = ord(cid)
    else:
        low = cid.lower()
        hex_part = None
        if low.startswith("u+") and len(low) >= 6:
            hex_part = low[2:]
        elif low.startswith("u") and len(low) in (5, 6, 7):
            hex_part = low[1:]
        if hex_part is not None and all(ch in "0123456789abcdef" for ch in hex_part):
            try:
                val = int(hex_part, 16)
                if 0x20 <= val <= 0x10FFFF:
                    cp = val
            except ValueError:
                cp = None
    if cp is None:
        return None

    if (
        0x4E00 <= cp <= 0x9FFF
        or 0x3400 <= cp <= 0x4DBF
        or 0xF900 <= cp <= 0xFAFF
        or 0x20000 <= cp <= 0x2A6DF
    ):
        return "cjk"
    if 0x3040 <= cp <= 0x30FF:
        return "kana"
    if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF:
        return "hangul"
    if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A) or (0x00C0 <= cp <= 0x024F):
        return "latin"
    if 0x30 <= cp <= 0x39:
        return "digit"
    return None


class MatrixDiffusionDataset(Dataset):
    """
    v3 通用二维矩阵流匹配数据集：
    - 原生支持 K-shot 多参考采样、RAM 零磁盘 I/O 缓存、成对色彩增强、零模糊整数平移；
    - 新增成对微几何风格增广（paired_morph_aug：斜切抗肩/字面缩放 + 离散像素格量化）；
    - 精确返回每条样本的 style_clean / target_clean / neg_clean 标志供 VisionFeatureCache 100% 无误差命中。
    """

    def __init__(
        self,
        index: MatrixIndex,
        data_cfg: DataConfig,
        use_scr: bool = False,
        num_neg: int = 16,
        allowed_styles: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.index = index
        self.data_cfg = data_cfg
        self.use_scr = bool(use_scr)
        self.num_neg = int(num_neg)
        self.max_style_refs = max(1, int(data_cfg.max_style_refs))
        self.background_color = data_cfg.background_color
        self._skip_neg_tensors = False

        if self.use_scr and self.num_neg < 1:
            raise ValueError(f"use_scr=True 时 num_neg 必须 >= 1，当前为 {self.num_neg}")

        self.norm_transform = build_image_transform(data_cfg, normalize=True)

        self.style_to_valid_paths: Dict[str, List[Path]] = {}
        self.content_to_valid_styles: Dict[str, Dict[str, Path]] = {}
        self.style_script_paths: Dict[str, Dict[str, List[Path]]] = {}
        self.script_to_style_paths: Dict[str, Dict[str, List[Path]]] = {}
        self._has_multi_scripts: bool = False
        self.samples: List[Tuple[str, str, Path]] = []
        self.skipped_missing_content = 0
        self.skipped_single_style = 0
        self.skipped_insufficient_neg = 0

        self._build_samples(allowed_styles=allowed_styles)

        all_paths = self.collect_all_image_paths()
        self._ram_cache: Optional[Dict[str, torch.Tensor]] = _preload_ram_uint8_cache(
            all_paths, data_cfg=data_cfg
        )

    def enable_cached_neg_skip(self) -> bool:
        """在确认已构建 SCR VisionFeatureCache 且负样本无缩放/翻转增广时，跳过每步 16x 负样本像素搬运。"""
        s_lo, s_hi = self.data_cfg.spatial_aug_scale
        can_skip = bool(
            self.use_scr
            and not self.data_cfg.spatial_aug_hflip
            and abs(float(s_lo) - 1.0) <= 1e-4
            and abs(float(s_hi) - 1.0) <= 1e-4
        )
        self._skip_neg_tensors = can_skip
        return can_skip

    def _build_samples(self, allowed_styles: Optional[Set[str]]) -> None:
        selected_styles = (
            sorted(s for s in self.index.style_ids if s in allowed_styles)
            if allowed_styles is not None
            else list(self.index.style_ids)
        )

        eligible_style_items: Dict[str, Dict[str, Path]] = {}
        for style_id in selected_styles:
            raw_items = self.index.style_to_items.get(style_id, {})
            matched: Dict[str, Path] = {}
            for content_id, img_path in raw_items.items():
                if content_id in self.index.content_paths:
                    matched[content_id] = img_path
                else:
                    self.skipped_missing_content += 1
            if len(matched) < 2:
                self.skipped_single_style += 1
                continue
            eligible_style_items[style_id] = matched
            self.style_to_valid_paths[style_id] = list(matched.values())

        for style_id, items in eligible_style_items.items():
            for content_id, img_path in items.items():
                self.content_to_valid_styles.setdefault(content_id, {})[style_id] = img_path
                grp = detect_content_script_group(content_id)
                if grp is not None:
                    self.style_script_paths.setdefault(style_id, {}).setdefault(grp, []).append(
                        img_path
                    )
                    self.script_to_style_paths.setdefault(grp, {}).setdefault(
                        style_id, []
                    ).append(img_path)

        self._has_multi_scripts = len(self.script_to_style_paths) >= 2

        weights_map: Dict[str, int] = {}
        if getattr(self.data_cfg, "style_weights_path", None):
            w_path = Path(str(self.data_cfg.style_weights_path))
            if not w_path.is_file():
                raise FileNotFoundError(f"data.style_weights_path 文件不存在: {w_path}")
            import json

            raw_w = json.loads(w_path.read_text(encoding="utf-8"))
            for k_s, v_w in raw_w.items():
                weights_map[str(k_s)] = max(1, min(8, int(round(float(v_w)))))

        min_styles_for_neg = self.num_neg + 1 if self.use_scr else 1
        for style_id, items in sorted(eligible_style_items.items()):
            repeat = weights_map.get(style_id, 1)
            for content_id, img_path in sorted(items.items()):
                if len(self.content_to_valid_styles[content_id]) < min_styles_for_neg:
                    self.skipped_insufficient_neg += 1
                    continue
                for _ in range(repeat):
                    self.samples.append((style_id, content_id, img_path))

        if not self.samples:
            max_cover = max(
                (len(m) for m in self.content_to_valid_styles.values()),
                default=0,
            )
            raise ValueError(
                f"在 '{self.index.styles_dir}' 中未找到可用于训练的有效样本："
                f"可用风格 {len(eligible_style_items)} 套，单内容最多覆盖 {max_cover} 套风格"
                + (
                    f"，但 Phase 2 设置了 num_neg={self.num_neg}（要求至少 {min_styles_for_neg} 套风格）。"
                    if self.use_scr
                    else "（每套风格至少需要 2 张与 content/ 匹配的图片）。"
                )
            )

    def collect_all_image_paths(self) -> List[Path]:
        """返回所有训练阶段涉及的风格/目标/内容图唯一路径列表。"""
        seen: Dict[str, Path] = {}
        for paths in self.style_to_valid_paths.values():
            for p in paths:
                seen[str(p)] = p
        for cid in self.content_to_valid_styles:
            cp = self.index.content_paths.get(cid)
            if cp is not None:
                seen[str(cp)] = cp
        return list(seen.values())

    def collect_style_target_paths(self) -> List[Path]:
        """仅返回风格/目标图路径（供 VisionFeatureCache 预提 DINOv2 表征）。"""
        seen: Dict[str, Path] = {}
        for paths in self.style_to_valid_paths.values():
            for p in paths:
                seen[str(p)] = p
        return list(seen.values())

    def load_clean_norm(self, path: Path | str) -> torch.Tensor:
        ps = str(path)
        if self._ram_cache is not None and ps in self._ram_cache:
            return _uint8_to_norm_tensor(self._ram_cache[ps])
        return self.norm_transform(
            load_rgb_image(path, background_color=self.background_color)
        )

    def _sample_style_refs(
        self,
        style_id: str,
        target_path: Path,
        content_id: str,
    ) -> Tuple[List[Path], bool]:
        """
        返回 (ref_paths, has_decoy_ref)：
        - 若触发跨语种干扰采样（mixed_script_ref_prob）：
          slot 0 保证为「同风格 + 同语系」（从而 k_curr==1 永不被污染）；
          slot 1 为「异风格 + 异语系」干扰参考，训练端在 k_curr>=2 时再随机打乱槽位顺序，
          迫使 Content-Aware 路由器学会按语系结构严格隔离中/英参考风格。
        """
        candidates = [p for p in self.style_to_valid_paths[style_id] if p != target_path]
        mixed_p = float(self.data_cfg.mixed_script_ref_prob)
        if (
            self.max_style_refs >= 2
            and self._has_multi_scripts
            and mixed_p > 0.0
            and random.random() < mixed_p
        ):
            grp = detect_content_script_group(content_id)
            if grp is not None:
                same_cands = [
                    p
                    for p in self.style_script_paths.get(style_id, {}).get(grp, [])
                    if p != target_path
                ]
                other_groups = [g for g in self.script_to_style_paths if g != grp]
                if same_cands and other_groups:
                    og = random.choice(other_groups)
                    decoy_styles = [
                        s for s in self.script_to_style_paths[og] if s != style_id
                    ]
                    if decoy_styles:
                        same_ref = random.choice(same_cands)
                        decoy_sid = random.choice(decoy_styles)
                        decoy_ref = random.choice(self.script_to_style_paths[og][decoy_sid])
                        picked = [same_ref, decoy_ref]
                        extra_pool = [p for p in candidates if p != same_ref] or candidates
                        while len(picked) < self.max_style_refs:
                            picked.append(random.choice(extra_pool))
                        return picked, True

        if len(candidates) >= self.max_style_refs:
            return random.sample(candidates, self.max_style_refs), False
        picked = list(candidates)
        while len(picked) < self.max_style_refs:
            picked.append(random.choice(candidates))
        return picked, False

    def __getitem__(self, idx: int) -> Dict:
        style_id, content_id, target_path = self.samples[idx]

        content_tensor = self.load_clean_norm(self.index.content_paths[content_id])
        target_tensor = self.load_clean_norm(target_path)

        ref_paths, has_decoy_ref = self._sample_style_refs(
            style_id, target_path, content_id
        )
        style_tensors = [self.load_clean_norm(p) for p in ref_paths]

        neg_paths: List[Path] = []
        neg_tensors: Optional[List[torch.Tensor]] = None
        if self.use_scr:
            style_map = self.content_to_valid_styles[content_id]
            neg_candidates = [
                path for s_id, path in style_map.items() if s_id != style_id
            ]
            neg_paths = random.sample(neg_candidates, self.num_neg)
            if not self._skip_neg_tensors:
                neg_tensors = [self.load_clean_norm(p) for p in neg_paths]

        style_clean = True
        target_clean = True
        neg_clean = True

        # 1. 成对空间整数平移/缩放
        if self.data_cfg.paired_spatial_aug and random.random() < float(
            self.data_cfg.spatial_aug_prob
        ):
            s_lo, s_hi = self.data_cfg.spatial_aug_scale
            shared_scale = (
                random.uniform(float(s_lo), float(s_hi)) if s_hi > s_lo else float(s_lo)
            )
            hflip_p = 0.5 if self.data_cfg.spatial_aug_hflip else 0.0
            spatial_group = [content_tensor, target_tensor]
            if neg_tensors is not None:
                spatial_group.extend(neg_tensors)
            shifted_group = apply_paired_spatial_aug(
                spatial_group,
                background_color=self.background_color,
                hflip_prob=hflip_p,
                scale=shared_scale,
                shift_ratio=self.data_cfg.spatial_aug_shift,
                interpolation=self.data_cfg.interpolation,
            )
            content_tensor, target_tensor = shifted_group[0], shifted_group[1]
            target_clean = False
            if neg_tensors is not None:
                neg_tensors = shifted_group[2:]
                if abs(shared_scale - 1.0) > 1e-4 or hflip_p > 0.0:
                    neg_clean = False
            if abs(shared_scale - 1.0) > 1e-4:
                style_tensors = [
                    apply_paired_spatial_aug(
                        [st],
                        background_color=self.background_color,
                        hflip_prob=hflip_p,
                        scale=shared_scale,
                        shift_ratio=0.0,
                        interpolation=self.data_cfg.interpolation,
                    )[0]
                    for st in style_tensors
                ]
                style_clean = False

        # 2. v3 成对微几何风格增广（仅同步作用于 target + 同源 style_refs，保持 content 与异源干扰参考不动）
        if self.data_cfg.paired_morph_aug:
            if (
                float(self.data_cfg.morph_shear_prob) > 0.0
                and random.random() < float(self.data_cfg.morph_shear_prob)
            ):
                if has_decoy_ref and len(style_tensors) >= 2:
                    same_group = [target_tensor, style_tensors[0], *style_tensors[2:]]
                    morphed = apply_paired_morph_shear_aspect(
                        same_group,
                        background_color=self.background_color,
                        max_shear_deg=float(self.data_cfg.morph_shear_deg),
                        aspect_range=float(self.data_cfg.morph_aspect_range),
                    )
                    target_tensor = morphed[0]
                    style_tensors = [morphed[1], style_tensors[1], *morphed[2:]]
                else:
                    morphed = apply_paired_morph_shear_aspect(
                        [target_tensor, *style_tensors],
                        background_color=self.background_color,
                        max_shear_deg=float(self.data_cfg.morph_shear_deg),
                        aspect_range=float(self.data_cfg.morph_aspect_range),
                    )
                    target_tensor = morphed[0]
                    style_tensors = morphed[1:]
                target_clean = False
                style_clean = False

        # 像素格与模板切缝只改 target + style_refs，content 保持实心原图。
        topo_group, topo_dirty = _apply_topology_style_augs(
            [target_tensor, *style_tensors],
            self.data_cfg,
            self.background_color,
        )
        if topo_dirty:
            target_tensor = topo_group[0]
            style_tensors = topo_group[1:]
            target_clean = False
            style_clean = False

        # 3. 成对色彩增强
        if self.data_cfg.paired_color_aug:
            augmented = apply_paired_color_jitter(
                [target_tensor, *style_tensors],
                background_color=self.background_color,
                hue_range=self.data_cfg.color_aug_hue,
                sat_range=self.data_cfg.color_aug_sat,
                val_range=self.data_cfg.color_aug_val,
            )
            target_tensor = augmented[0]
            style_tensors = augmented[1:]
            target_clean = False
            style_clean = False

        style_stack = torch.stack(style_tensors, dim=0)  # [K, 3, H, W]

        sample = {
            "content_image": content_tensor,
            "style_image": style_tensors[0],
            "style_images": style_stack,
            "target_image": target_tensor,
            "target_image_path": str(target_path),
            "style_image_paths": [str(p) for p in ref_paths],
            "style_clean": style_clean,
            "target_clean": target_clean,
            "neg_clean": neg_clean,
            "style_id": style_id,
            "content_id": content_id,
        }

        if neg_tensors is not None:
            sample["neg_images"] = torch.stack(neg_tensors, dim=0)
        if neg_paths:
            sample["neg_image_paths"] = [str(p) for p in neg_paths]

        return sample

    def __len__(self) -> int:
        return len(self.samples)


class MatrixSCRDataset(Dataset):
    """
    SCR 2.0 跨内容风格对比预训练数据集（v3 支持 RAM 缓存、成对微几何风格增广与 VisionFeatureCache 命中标志）：
    - anchor_image: (风格 S, 内容 C1)
    - pos_image:    (风格 S, 内容 C2 != C1)
    - neg_images:   (风格 S_neg != S, 内容 C1)
    """

    def __init__(
        self,
        index: MatrixIndex,
        data_cfg: DataConfig,
        num_neg: int = 16,
        allowed_styles: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.index = index
        self.data_cfg = data_cfg
        self.num_neg = int(num_neg)
        self.background_color = data_cfg.background_color
        self.norm_transform = build_image_transform(data_cfg, normalize=True)
        self._skip_neg_tensors = False

        if self.num_neg < 1:
            raise ValueError(f"num_neg 必须 >= 1，当前为 {self.num_neg}")

        self.content_to_styles: Dict[str, Dict[str, Path]] = {}
        self.style_to_paths: Dict[str, List[Path]] = {}
        all_styles: Set[str] = set()

        for style_id in self.index.style_ids:
            if allowed_styles is not None and style_id not in allowed_styles:
                continue
            items = self.index.style_to_items.get(style_id, {})
            if len(items) < 2:
                continue
            all_styles.add(style_id)
            self.style_to_paths[style_id] = list(items.values())
            for content_id, img_path in items.items():
                self.content_to_styles.setdefault(content_id, {})[style_id] = img_path

        min_required = self.num_neg + 1
        self.samples: List[Tuple[str, str, Path]] = []
        self.valid_contents: List[str] = []
        self.skipped_contents = 0

        for content_id, style_map in sorted(self.content_to_styles.items()):
            if len(style_map) < min_required:
                self.skipped_contents += 1
                continue
            self.valid_contents.append(content_id)
            for style_id, img_path in sorted(style_map.items()):
                self.samples.append((style_id, content_id, img_path))

        self.num_styles = len(all_styles)

        if not self.samples:
            max_cover = max(
                (len(m) for m in self.content_to_styles.values()),
                default=0,
            )
            raise ValueError(
                f"在 '{self.index.styles_dir}' 中未找到可用于 SCR 训练的有效样本："
                f"检测到 {self.num_styles} 套可用风格（>=2张）、{len(self.content_to_styles)} 个内容对象，"
                f"单内容最多覆盖 {max_cover} 套风格，但 num_neg={self.num_neg} 要求至少 "
                f"{min_required} 套风格。请增加风格数量或调小 scr.num_neg。"
            )

        self._ram_cache: Optional[Dict[str, torch.Tensor]] = _preload_ram_uint8_cache(
            self.collect_all_image_paths(), data_cfg=data_cfg
        )

    def enable_cached_neg_skip(self) -> bool:
        """在确认已构建 SCR VisionFeatureCache 且负样本无缩放/翻转增广时，跳过每步负样本像素张量打包。"""
        s_lo, s_hi = self.data_cfg.spatial_aug_scale
        can_skip = bool(
            not self.data_cfg.spatial_aug_hflip
            and abs(float(s_lo) - 1.0) <= 1e-4
            and abs(float(s_hi) - 1.0) <= 1e-4
        )
        self._skip_neg_tensors = can_skip
        return can_skip

    def collect_all_image_paths(self) -> List[Path]:
        seen: Dict[str, Path] = {}
        for paths in self.style_to_paths.values():
            for p in paths:
                seen[str(p)] = p
        return list(seen.values())

    def load_clean_norm(self, path: Path | str) -> torch.Tensor:
        ps = str(path)
        if self._ram_cache is not None and ps in self._ram_cache:
            return _uint8_to_norm_tensor(self._ram_cache[ps])
        return self.norm_transform(
            load_rgb_image(path, background_color=self.background_color)
        )

    def __getitem__(self, idx: int) -> Dict:
        style_id, content_id, anchor_path = self.samples[idx]
        anchor_image = self.load_clean_norm(anchor_path)

        same_style_others = [
            p for p in self.style_to_paths[style_id] if p != anchor_path
        ]
        pos_path = random.choice(same_style_others)
        pos_image = self.load_clean_norm(pos_path)

        style_map = self.content_to_styles[content_id]
        neg_candidates = [
            path for s_id, path in style_map.items() if s_id != style_id
        ]
        chosen_neg = random.sample(neg_candidates, self.num_neg)
        neg_list = (
            None if self._skip_neg_tensors else [self.load_clean_norm(p) for p in chosen_neg]
        )

        anchor_clean = True
        pos_clean = True
        neg_clean = True

        if self.data_cfg.paired_spatial_aug and random.random() < float(
            self.data_cfg.spatial_aug_prob
        ):
            s_lo, s_hi = self.data_cfg.spatial_aug_scale
            shared_scale = (
                random.uniform(float(s_lo), float(s_hi)) if s_hi > s_lo else float(s_lo)
            )
            hflip_p = 0.5 if self.data_cfg.spatial_aug_hflip else 0.0
            if abs(shared_scale - 1.0) > 1e-4 or hflip_p > 0.0:
                if neg_list is None:
                    neg_list = [self.load_clean_norm(p) for p in chosen_neg]
                shifted_anchor_negs = apply_paired_spatial_aug(
                    [anchor_image, *neg_list],
                    background_color=self.background_color,
                    hflip_prob=hflip_p,
                    scale=shared_scale,
                    shift_ratio=self.data_cfg.spatial_aug_shift,
                    interpolation=self.data_cfg.interpolation,
                )
                anchor_image = shifted_anchor_negs[0]
                neg_list = shifted_anchor_negs[1:]
                pos_image = apply_paired_spatial_aug(
                    [pos_image],
                    background_color=self.background_color,
                    hflip_prob=hflip_p,
                    scale=shared_scale,
                    shift_ratio=self.data_cfg.spatial_aug_shift,
                    interpolation=self.data_cfg.interpolation,
                )[0]
                anchor_clean = False
                pos_clean = False
                neg_clean = False

        # v3 成对微几何风格增广：对 (anchor, pos) 同步施加，让 SCR 学会将斜势/像素格视作合法风格维度
        if self.data_cfg.paired_morph_aug:
            if (
                float(self.data_cfg.morph_shear_prob) > 0.0
                and random.random() < float(self.data_cfg.morph_shear_prob)
            ):
                anchor_image, pos_image = apply_paired_morph_shear_aspect(
                    [anchor_image, pos_image],
                    background_color=self.background_color,
                    max_shear_deg=float(self.data_cfg.morph_shear_deg),
                    aspect_range=float(self.data_cfg.morph_aspect_range),
                )
                anchor_clean = False
                pos_clean = False

        # 像素格与模板切缝同步作用于 (anchor, pos)，负样本保持原风格。
        topo_pair, topo_dirty = _apply_topology_style_augs(
            [anchor_image, pos_image],
            self.data_cfg,
            self.background_color,
        )
        if topo_dirty:
            anchor_image, pos_image = topo_pair
            anchor_clean = False
            pos_clean = False

        if self.data_cfg.paired_color_aug:
            anchor_image, pos_image = apply_paired_color_jitter(
                [anchor_image, pos_image],
                background_color=self.background_color,
                hue_range=self.data_cfg.color_aug_hue,
                sat_range=self.data_cfg.color_aug_sat,
                val_range=self.data_cfg.color_aug_val,
            )
            anchor_clean = False
            pos_clean = False

        out = {
            "target_image": anchor_image,
            "pos_image": pos_image,
            "target_image_path": str(anchor_path),
            "pos_image_path": str(pos_path),
            "neg_image_paths": [str(p) for p in chosen_neg],
            "anchor_clean": anchor_clean,
            "pos_clean": pos_clean,
            "neg_clean": neg_clean,
            "style_id": style_id,
            "content_id": content_id,
        }
        if neg_list is not None:
            out["neg_images"] = torch.stack(neg_list, dim=0)
        return out

    def __len__(self) -> int:
        return len(self.samples)
