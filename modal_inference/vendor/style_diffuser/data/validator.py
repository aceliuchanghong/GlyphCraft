from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Dict, List, Optional

from PIL import Image

from ..config import ProjectConfig
from .index import MatrixIndex, try_load_explicit_val_index
from .preprocess import has_transparency


@dataclass
class ValidationReport:
    ok: bool
    layout: str
    data_root: str
    content_dir: Optional[str]
    styles_dir: str
    has_explicit_val: bool
    num_contents: int
    num_styles: int
    num_train_styles: int
    num_val_holdout_styles: int
    total_target_images: int
    matrix_density: float
    orphan_target_count: int
    single_image_styles: List[str]
    max_num_neg_100pct: int
    max_num_neg_90pct: int
    max_num_neg_any: int
    images_checked: int
    corrupted_images: List[str]
    transparent_image_count: int
    non_square_image_count: int
    top_resolutions: List[Dict[str, object]]
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, ensure_ascii=False, indent=2)
        return path


def _quantile_max_neg(coverages: List[int], ratio: float) -> int:
    if not coverages:
        return 0
    sorted_cov = sorted(coverages, reverse=True)
    idx = min(len(sorted_cov) - 1, max(0, int(len(sorted_cov) * ratio) - 1))
    return max(0, sorted_cov[idx] - 1)


def validate_dataset(
    cfg: ProjectConfig,
    full_scan: bool = False,
    max_image_checks: int = 500,
) -> ValidationReport:
    errors: List[str] = []
    warnings: List[str] = []

    try:
        cfg.validate()
    except Exception as exc:
        errors.append(f"配置校验失败: {exc}")

    need_content = bool(cfg.phase1.enabled or cfg.phase2.enabled)
    try:
        index = MatrixIndex(
            data_root=cfg.data.root,
            phase=cfg.data.phase,
            layout=cfg.data.layout,
            require_content=need_content,
        )
    except Exception as exc:
        return ValidationReport(
            ok=False,
            layout="unknown",
            data_root=str(cfg.data.root),
            content_dir=None,
            styles_dir="",
            has_explicit_val=False,
            num_contents=0,
            num_styles=0,
            num_train_styles=0,
            num_val_holdout_styles=0,
            total_target_images=0,
            matrix_density=0.0,
            orphan_target_count=0,
            single_image_styles=[],
            max_num_neg_100pct=0,
            max_num_neg_90pct=0,
            max_num_neg_any=0,
            images_checked=0,
            corrupted_images=[],
            transparent_image_count=0,
            non_square_image_count=0,
            top_resolutions=[],
            errors=errors + [str(exc)],
            warnings=warnings,
        )

    explicit_val = try_load_explicit_val_index(cfg.data.root, layout=cfg.data.layout)

    # 切分训练/留出风格
    try:
        train_styles, holdout_styles = index.split_styles(
            val_holdout_styles=cfg.data.val_holdout_styles,
            seed=cfg.seed,
        )
    except ValueError as exc:
        errors.append(str(exc))
        train_styles, holdout_styles = set(index.style_ids), set()

    # 统计矩阵覆盖度（基于训练风格；SCR 2.0 跨内容正样本与扩散训练均要求每风格 >=2 张图）
    single_image_styles: List[str] = []
    train_content_coverage: Dict[str, int] = {}
    matched_target_count = 0
    for style_id in sorted(train_styles):
        items = index.style_to_items.get(style_id, {})
        matched = [
            cid for cid in items if (not index.content_paths or cid in index.content_paths)
        ]
        if len(matched) < 2:
            single_image_styles.append(style_id)
            continue
        for cid in matched:
            train_content_coverage[cid] = train_content_coverage.get(cid, 0) + 1
            matched_target_count += 1

    if single_image_styles:
        warnings.append(
            f"发现 {len(single_image_styles)} 套风格可用图片数 < 2（无法抽取同风格参考图，扩散训练时将自动跳过）: "
            f"{single_image_styles[:5]}"
        )

    if index.orphan_targets:
        warnings.append(
            f"发现 {len(index.orphan_targets)} 张目标图在 content 目录中无对应内容原型（扩散训练时将自动跳过）。"
        )

    coverages = list(train_content_coverage.values())
    max_neg_100 = max(0, min(coverages) - 1) if coverages else 0
    max_neg_90 = _quantile_max_neg(coverages, 0.90)
    max_neg_any = max(0, max(coverages) - 1) if coverages else 0

    denom = max(1, len(train_styles) * max(1, len(index.content_paths) or len(train_content_coverage)))
    density = matched_target_count / float(denom)

    # 检查 SCR / Phase 1 / Phase 2 数据充分性
    if cfg.scr.enabled:
        scr_valid_samples = sum(c for c in coverages if c >= cfg.scr.num_neg + 1)
        if scr_valid_samples == 0:
            errors.append(
                f"SCR 训练无有效样本：scr.num_neg={cfg.scr.num_neg} 要求单内容至少覆盖 "
                f"{cfg.scr.num_neg + 1} 套训练风格，但当前单内容最高覆盖 {max_neg_any + 1} 套。"
            )
        elif scr_valid_samples < cfg.scr.train_batch_size:
            errors.append(
                f"SCR 有效样本数（{scr_valid_samples}）小于 scr.train_batch_size（{cfg.scr.train_batch_size}）。"
            )

    if cfg.phase1.enabled:
        p1_valid_samples = sum(
            len([c for c in index.style_to_items[s] if c in index.content_paths])
            for s in train_styles
            if s not in single_image_styles
        )
        if p1_valid_samples == 0:
            errors.append("Phase 1 扩散训练无有效样本，请检查 content 与 styles 目录文件名是否对齐。")

    if cfg.phase2.enabled:
        p2_valid_samples = sum(
            1
            for s in train_styles
            if s not in single_image_styles
            for c in index.style_to_items[s]
            if c in index.content_paths
            and train_content_coverage.get(c, 0) >= cfg.phase2.num_neg + 1
        )
        if p2_valid_samples == 0:
            errors.append(
                f"Phase 2 训练无有效样本：phase2.num_neg={cfg.phase2.num_neg} 要求单内容至少覆盖 "
                f"{cfg.phase2.num_neg + 1} 套训练风格，当前最高覆盖 {max_neg_any + 1} 套。"
            )
        elif p2_valid_samples < cfg.phase2.train_batch_size:
            errors.append(
                f"Phase 2 有效样本数（{p2_valid_samples}）小于 phase2.train_batch_size（{cfg.phase2.train_batch_size}）。"
            )

    # 扫描图片健康度、透明通道与尺寸分布
    all_image_paths: List[Path] = list(index.content_paths.values())
    for items in index.style_to_items.values():
        all_image_paths.extend(items.values())

    if not full_scan and len(all_image_paths) > max_image_checks:
        rng = random.Random(cfg.seed)
        check_paths = rng.sample(all_image_paths, max_image_checks)
    else:
        check_paths = all_image_paths

    corrupted: List[str] = []
    transparent_count = 0
    non_square_count = 0
    res_counter: Counter = Counter()

    for img_path in check_paths:
        try:
            with Image.open(img_path) as img:
                img.load()
                w, h = img.size
                res_counter[f"{w}x{h}"] += 1
                if w != h:
                    non_square_count += 1
                if has_transparency(img):
                    transparent_count += 1
        except Exception as exc:
            corrupted.append(f"{img_path} ({exc})")

    if corrupted:
        errors.append(
            f"检测到 {len(corrupted)} 张损坏或无法读取的图片（示例：{corrupted[0]}）。"
        )

    if non_square_count > 0 and cfg.data.resize_mode == "stretch":
        warnings.append(
            f"抽检图片中有 {non_square_count}/{len(check_paths)} 张非正方形，"
            "当前 data.resize_mode='stretch' 会直接拉伸变形；若为图标/道具，建议改用 data.resize_mode='pad'。"
        )

    if transparent_count > 0:
        warnings.append(
            f"检测到 {transparent_count}/{len(check_paths)} 张透明通道图像（RGBA/LA/P），"
            f"加载时将自动合成到背景色 RGB{list(cfg.data.background_color)}。"
        )

    top_res = [
        {"size": size, "count": count} for size, count in res_counter.most_common(5)
    ]

    return ValidationReport(
        ok=(len(errors) == 0),
        layout=index.layout,
        data_root=str(index.root),
        content_dir=str(index.content_dir) if index.content_dir else None,
        styles_dir=str(index.styles_dir),
        has_explicit_val=(explicit_val is not None),
        num_contents=len(index.content_paths),
        num_styles=len(index.style_ids),
        num_train_styles=len(train_styles),
        num_val_holdout_styles=len(holdout_styles),
        total_target_images=index.total_target_images,
        matrix_density=round(density, 4),
        orphan_target_count=len(index.orphan_targets),
        single_image_styles=single_image_styles,
        max_num_neg_100pct=max_neg_100,
        max_num_neg_90pct=max_neg_90,
        max_num_neg_any=max_neg_any,
        images_checked=len(check_paths),
        corrupted_images=corrupted,
        transparent_image_count=transparent_count,
        non_square_image_count=non_square_count,
        top_resolutions=top_res,
        errors=errors,
        warnings=warnings,
    )


def format_validation_report(report: ValidationReport) -> str:
    status = "PASS" if report.ok else "FAIL"
    res_str = ", ".join(
        f"{item['size']}({item['count']})" for item in report.top_resolutions
    ) or "N/A"
    lines = [
        "=" * 64,
        f"[数据体检报告] 状态: {status}",
        "=" * 64,
        f"  数据根目录      : {report.data_root}",
        f"  识别布局协议    : {report.layout}",
        f"  内容原型目录    : {report.content_dir or '(未使用)'}",
        f"  风格目标目录    : {report.styles_dir}",
        f"  显式验证集 val/ : {'是' if report.has_explicit_val else '否'}",
        f"  内容对象总数    : {report.num_contents}",
        f"  风格总数        : {report.num_styles}（训练 {report.num_train_styles} / 留出验证 {report.num_val_holdout_styles}）",
        f"  目标图像总数    : {report.total_target_images}（矩阵填充率 {report.matrix_density * 100:.1f}%）",
        f"  推荐最大 num_neg: 全覆盖={report.max_num_neg_100pct} | 90%覆盖={report.max_num_neg_90pct} | 理论上限={report.max_num_neg_any}",
        f"  图片健康抽检    : 共检 {report.images_checked} 张 | 损坏 {len(report.corrupted_images)} | 透明 {report.transparent_image_count} | 非正方形 {report.non_square_image_count}",
        f"  主要原始分辨率  : {res_str}",
    ]
    if report.warnings:
        lines.append("-" * 64)
        lines.append("  [WARN] 注意事项:")
        for w in report.warnings:
            lines.append(f"    - {w}")
    if report.errors:
        lines.append("-" * 64)
        lines.append("  [ERROR] 阻塞项（必须修复才能开始训练）:")
        for e in report.errors:
            lines.append(f"    - {e}")
    lines.append("=" * 64)
    return "\n".join(lines)
