from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


@dataclass(frozen=True)
class PreviewSample:
    content_id: str
    style_id: str
    content_path: Path
    style_ref_path: Path
    target_path: Path


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def _resolve_dirs(
    root: Path,
    phase: str,
    layout: str,
) -> Tuple[str, Optional[Path], Path]:
    """解析内容图目录与风格目标图目录，自动兼容标准协议与旧版 FontDiffuser 协议。"""
    candidates: List[Tuple[str, Path, Path]] = []
    if layout in ("auto", "standard"):
        candidates.extend(
            [
                ("standard", root / phase / "content", root / phase / "styles"),
                ("standard", root / "content", root / "styles"),
            ]
        )
    if layout in ("auto", "legacy"):
        candidates.extend(
            [
                ("legacy", root / phase / "ContentImage", root / phase / "TargetImage"),
                ("legacy", root / "ContentImage", root / "TargetImage"),
            ]
        )

    for detected_layout, content_dir, styles_dir in candidates:
        if styles_dir.is_dir():
            resolved_content = content_dir if content_dir.is_dir() else None
            return detected_layout, resolved_content, styles_dir

    checked = ", ".join(str(c[2]) for c in candidates)
    raise FileNotFoundError(
        f"在 '{root}' 下未找到风格目标目录（layout={layout!r}, phase={phase!r}）。"
        f"已检查: {checked}"
    )


def _parse_content_id(
    style_id: str,
    filename: str,
    layout: str,
    known_contents: Set[str],
) -> str:
    stem = Path(filename).stem
    if not stem:
        return ""
    if layout == "standard":
        if stem in known_contents or not known_contents:
            # 若用户在 standard 目录下不小心沿用了 style+content 命名且内容表匹配去掉前缀后的名字，也兼容
            prefix = f"{style_id}+"
            if (
                stem not in known_contents
                and stem.startswith(prefix)
                and stem[len(prefix) :] in known_contents
            ):
                return stem[len(prefix) :]
            return stem
        prefix = f"{style_id}+"
        if stem.startswith(prefix) and stem[len(prefix) :] in known_contents:
            return stem[len(prefix) :]
        return stem

    # legacy 布局：优先剥离 "{style_id}+"，其次剥离首个 "+"
    prefix = f"{style_id}+"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    if "+" in stem:
        _, content = stem.split("+", 1)
        return content
    return stem


class MatrixIndex:
    """
    标准矩阵数据集索引器。

    同时支持：
    1. 标准协议：
       data_root/[phase/]content/<content_id>.<ext>
       data_root/[phase/]styles/<style_id>/<content_id>.<ext>
    2. 旧版 FontDiffuser 协议：
       data_root/[phase/]ContentImage/<content_id>.<ext>
       data_root/[phase/]TargetImage/<style_id>/[<style_id>+]<content_id>.<ext>
    """

    def __init__(
        self,
        data_root: str | Path,
        phase: str = "train",
        layout: str = "auto",
        require_content: bool = True,
    ):
        self.root = Path(data_root)
        self.phase = phase
        self.requested_layout = layout
        self.layout, self.content_dir, self.styles_dir = _resolve_dirs(
            self.root, phase=phase, layout=layout
        )

        if require_content and self.content_dir is None:
            expected = (
                self.styles_dir.parent
                / ("content" if self.layout == "standard" else "ContentImage")
            )
            raise FileNotFoundError(
                f"未找到内容原型目录 '{expected}'（已检测到风格目录 '{self.styles_dir}'）。"
            )

        self.content_paths: Dict[str, Path] = {}
        self.style_to_items: Dict[str, Dict[str, Path]] = {}
        self.content_to_styles: Dict[str, Dict[str, Path]] = {}
        self.orphan_targets: List[Tuple[str, str, Path]] = []

        self._scan()

    def _scan(self) -> None:
        if self.content_dir is not None and self.content_dir.is_dir():
            for entry in sorted(self.content_dir.iterdir(), key=lambda p: p.name):
                if _is_image_file(entry):
                    self.content_paths[entry.stem] = entry

        known_contents = set(self.content_paths.keys())

        for style_dir in sorted(self.styles_dir.iterdir(), key=lambda p: p.name):
            if not style_dir.is_dir():
                continue
            style_id = style_dir.name
            items: Dict[str, Path] = {}
            for entry in sorted(style_dir.iterdir(), key=lambda p: p.name):
                if not _is_image_file(entry):
                    continue
                content_id = _parse_content_id(
                    style_id=style_id,
                    filename=entry.name,
                    layout=self.layout,
                    known_contents=known_contents,
                )
                if not content_id:
                    continue
                items[content_id] = entry
                if known_contents and content_id not in known_contents:
                    self.orphan_targets.append((style_id, content_id, entry))
            if items:
                self.style_to_items[style_id] = items
                for content_id, img_path in items.items():
                    self.content_to_styles.setdefault(content_id, {})[style_id] = img_path

    @property
    def style_ids(self) -> List[str]:
        return sorted(self.style_to_items.keys())

    @property
    def content_ids(self) -> List[str]:
        return sorted(self.content_paths.keys())

    @property
    def total_target_images(self) -> int:
        return sum(len(v) for v in self.style_to_items.values())

    def split_styles(
        self,
        val_holdout_styles: int = 0,
        seed: int = 123,
    ) -> Tuple[Set[str], Set[str]]:
        """将风格集合确定性划分为训练集风格与留出验证风格。"""
        all_styles = list(self.style_ids)
        if val_holdout_styles <= 0:
            return set(all_styles), set()
        if val_holdout_styles >= len(all_styles):
            raise ValueError(
                f"val_holdout_styles={val_holdout_styles} 大于等于总风格数 {len(all_styles)}，"
                "请调小 val_holdout_styles。"
            )
        rng = random.Random(seed)
        shuffled = list(all_styles)
        rng.shuffle(shuffled)
        val_styles = set(shuffled[:val_holdout_styles])
        train_styles = set(shuffled[val_holdout_styles:])
        return train_styles, val_styles

    def select_preview_samples(
        self,
        num_samples: int,
        seed: int = 123,
        preferred_styles: Optional[Sequence[str] | Set[str]] = None,
    ) -> List[PreviewSample]:
        """确定性挑选训练期验证对比网格样本（Content | Style Ref | Pred | GT）。"""
        if num_samples <= 0 or not self.content_paths:
            return []

        candidate_styles = (
            sorted(s for s in preferred_styles if s in self.style_to_items)
            if preferred_styles
            else list(self.style_ids)
        )
        if not candidate_styles:
            candidate_styles = list(self.style_ids)

        rng = random.Random(seed)
        pool: List[PreviewSample] = []
        for style_id in candidate_styles:
            items = self.style_to_items.get(style_id, {})
            matched_cids = sorted(c for c in items if c in self.content_paths)
            if len(matched_cids) < 2:
                continue
            for idx, cid in enumerate(matched_cids):
                # 选同风格的另一张图作为参考风格图
                ref_cid = matched_cids[(idx + 1) % len(matched_cids)]
                pool.append(
                    PreviewSample(
                        content_id=cid,
                        style_id=style_id,
                        content_path=self.content_paths[cid],
                        style_ref_path=items[ref_cid],
                        target_path=items[cid],
                    )
                )

        if not pool:
            return []

        # 尽量覆盖不同风格和不同内容
        rng.shuffle(pool)
        selected: List[PreviewSample] = []
        used_styles: Set[str] = set()
        used_contents: Set[str] = set()
        for sample in pool:
            if len(selected) >= num_samples:
                break
            if sample.style_id not in used_styles and sample.content_id not in used_contents:
                selected.append(sample)
                used_styles.add(sample.style_id)
                used_contents.add(sample.content_id)

        if len(selected) < num_samples:
            chosen_keys = {(s.style_id, s.content_id) for s in selected}
            for sample in pool:
                if len(selected) >= num_samples:
                    break
                key = (sample.style_id, sample.content_id)
                if key not in chosen_keys:
                    selected.append(sample)
                    chosen_keys.add(key)

        return selected


def try_load_explicit_val_index(
    data_root: str | Path,
    layout: str = "auto",
) -> Optional[MatrixIndex]:
    """如果数据集根目录下存在显式 val/ 分割，返回其 MatrixIndex，否则返回 None。"""
    root = Path(data_root)
    val_candidates = [
        root / "val" / "styles",
        root / "val" / "TargetImage",
    ]
    if not any(p.is_dir() for p in val_candidates):
        return None
    try:
        return MatrixIndex(data_root=root, phase="val", layout=layout, require_content=True)
    except FileNotFoundError:
        return None
