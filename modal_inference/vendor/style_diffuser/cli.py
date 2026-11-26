from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence

import torch

from .config import SamplingConfig, load_config
from .data.validator import format_validation_report, validate_dataset
from .engine.infer import infer_from_paths
from .engine.logging_utils import find_latest_step_dir
from .engine.runner import run_pipeline
from .engine.train_flow import run_train_flow
from .engine.train_scr import run_train_scr


def _add_common_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="项目 YAML 配置文件路径。",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="覆盖 YAML 配置项（支持多级键，如 --set phase1.max_train_steps=100），可重复传参。",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="style-diffuser",
        description="PixelDiT-Flow v3：面向 2D 视觉资产的新一代像素空间流匹配风格复刻框架。",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # validate
    p_val = sub.add_parser("validate", help="对标准矩阵数据集执行数据体检并校验配置。")
    _add_common_config_args(p_val)
    p_val.add_argument(
        "--full-scan",
        action="store_true",
        help="扫描全部图片像素完整性（大数据集默认抽检 500 张）。",
    )
    p_val.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="可选：将体检结果保存为 JSON 文件。",
    )

    # train-scr
    p_scr = sub.add_parser("train-scr", help="执行轻量 SCR 2.0 跨内容风格对比模块预训练。")
    _add_common_config_args(p_scr)

    # train-p1
    p_p1 = sub.add_parser(
        "train-p1", help="执行 PixelDiT-Flow Phase 1（Rectified Flow + JiT + REPA）训练。"
    )
    _add_common_config_args(p_p1)

    # train-p2
    p_p2 = sub.add_parser(
        "train-p2",
        help="执行 Phase 2（SCR 2.0 风格对比精调 + Shortcut 1-4/8-10 步双档联合微调）。",
    )
    _add_common_config_args(p_p2)

    # run
    p_run = sub.add_parser(
        "run",
        help="一键串行执行 validate -> train-scr -> train-p1 -> train-p2（自动跳过已完成阶段）。",
    )
    _add_common_config_args(p_run)
    p_run.add_argument(
        "--force",
        action="store_true",
        help="忽略 stage_done.json，强制从头重跑所有已启用阶段。",
    )
    p_run.add_argument(
        "--skip-validate",
        action="store_true",
        help="跳过前置数据体检（不建议）。",
    )

    # infer
    p_inf = sub.add_parser(
        "infer",
        help="使用训练好的 PixelDiT-Flow 权重执行双档（1-4步极速 / 8-10步高画质）批量风格复刻推理。",
    )
    p_inf.add_argument(
        "--config",
        type=str,
        default=None,
        help="可选：提供项目 YAML；若省略 --ckpt-dir 则自动选用该项目最新权重。",
    )
    p_inf.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="配合 --config 使用的覆盖项。",
    )
    p_inf.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="模型权重目录（含 pixel_dit.pth / ema.pth / model_config.yaml）。",
    )
    p_inf.add_argument(
        "--style-image",
        action="append",
        required=True,
        help="风格参考图路径（可重复传入多次以启用 K-shot 多参考图融合）。",
    )
    p_inf.add_argument("--content-image", type=str, default=None, help="单张内容原型图路径。")
    p_inf.add_argument("--content-dir", type=str, default=None, help="内容原型图目录路径。")
    p_inf.add_argument("--output-dir", type=str, required=True, help="生成结果输出目录。")
    p_inf.add_argument("--batch-size", type=int, default=16, help="推理批大小（默认 16）。")
    p_inf.add_argument(
        "--mode",
        choices=("auto", "fast", "hd"),
        default=None,
        help="采样模式：fast(1-4步免CFG极速) / hd(8-10步Dual-CFG+APG) / auto(<=4步自动fast)。",
    )
    p_inf.add_argument(
        "--steps",
        type=int,
        default=None,
        help="采样步数（极速验证推荐 1-4，高画质交付推荐 8-10）。",
    )
    p_inf.add_argument(
        "--solver",
        choices=("euler", "heun", "heun_fast"),
        default=None,
        help="HD 模式 ODE 求解器（默认 heun_fast）。",
    )
    p_inf.add_argument(
        "--content-scale",
        type=float,
        default=None,
        help="独立骨架结构引导强度 w_c（默认 2.5）。",
    )
    p_inf.add_argument(
        "--style-scale",
        type=float,
        default=None,
        help="独立风格浓烈度引导强度 w_s（默认 3.5）。",
    )
    p_inf.add_argument(
        "--no-apg",
        action="store_true",
        help="关闭 APG 自适应投影引导（默认开启以防彩色过曝）。",
    )
    p_inf.add_argument(
        "--no-ema",
        action="store_true",
        help="不使用 ema.pth 平滑权重，改用原始训练权重推理。",
    )
    p_inf.add_argument("--seed", type=int, default=123, help="随机种子。")
    p_inf.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="推理设备（如 cuda:0 或 cpu）。",
    )
    p_inf.add_argument(
        "--no-comparison",
        action="store_true",
        help="不保存三联对照图 comparison.png。",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate":
        cfg = load_config(args.config, overrides=args.overrides)
        report = validate_dataset(cfg, full_scan=args.full_scan)
        print(format_validation_report(report))
        if args.json_out:
            report.save_json(args.json_out)
        return 0 if report.ok else 1

    if args.command == "train-scr":
        cfg = load_config(args.config, overrides=args.overrides)
        run_train_scr(cfg)
        return 0

    if args.command == "train-p1":
        cfg = load_config(args.config, overrides=args.overrides)
        run_train_flow(cfg, phase=1)
        return 0

    if args.command == "train-p2":
        cfg = load_config(args.config, overrides=args.overrides)
        run_train_flow(cfg, phase=2)
        return 0

    if args.command == "run":
        cfg = load_config(args.config, overrides=args.overrides)
        run_pipeline(
            config_path=args.config,
            cfg=cfg,
            overrides=args.overrides,
            force=args.force,
            skip_validate=args.skip_validate,
        )
        return 0

    if args.command == "infer":
        sampling_cfg = SamplingConfig()
        data_cfg = None
        ckpt_dir = args.ckpt_dir
        if args.config:
            cfg = load_config(args.config, overrides=args.overrides)
            sampling_cfg = cfg.sampling
            data_cfg = cfg.data
            if not ckpt_dir:
                resolved = find_latest_step_dir(cfg.phase2_dir) or find_latest_step_dir(
                    cfg.phase1_dir
                )
                if resolved is None:
                    raise SystemExit(
                        f"未在 '{cfg.phase2_dir}' 或 '{cfg.phase1_dir}' 找到权重，请显式传入 --ckpt-dir。"
                    )
                ckpt_dir = str(resolved)
        if not ckpt_dir:
            raise SystemExit("必须提供 --ckpt-dir 或 --config。")

        if args.mode is not None:
            sampling_cfg.mode = args.mode
        if args.steps is not None:
            sampling_cfg.num_inference_steps = args.steps
        if args.solver is not None:
            sampling_cfg.solver = args.solver
        if args.content_scale is not None:
            sampling_cfg.content_guidance_scale = args.content_scale
        if args.style_scale is not None:
            sampling_cfg.style_guidance_scale = args.style_scale
        if args.no_apg:
            sampling_cfg.use_apg = False
        if args.no_ema:
            sampling_cfg.use_ema = False

        style_paths = [Path(p) for p in args.style_image]
        infer_from_paths(
            ckpt_dir=Path(ckpt_dir),
            style_image_path=style_paths,
            output_dir=Path(args.output_dir),
            content_image_path=args.content_image,
            content_dir=args.content_dir,
            sampling_cfg=sampling_cfg,
            data_cfg=data_cfg,
            batch_size=args.batch_size,
            seed=args.seed,
            device=args.device,
            save_comparison=not args.no_comparison,
        )
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
