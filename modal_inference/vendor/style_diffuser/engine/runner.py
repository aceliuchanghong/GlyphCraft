from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

from ..config import ProjectConfig
from ..data.validator import format_validation_report, validate_dataset
from .logging_utils import find_latest_scr_ckpt, find_latest_step_dir, read_stage_done


def _build_stage_command(
    stage_cmd: str,
    config_path: str | Path,
    overrides: Sequence[str],
    use_accelerate: bool,
) -> List[str]:
    if use_accelerate:
        cmd = ["accelerate", "launch", "-m", "style_diffuser.cli", stage_cmd]
    else:
        cmd = [sys.executable, "-m", "style_diffuser.cli", stage_cmd]
    cmd.extend(["--config", str(config_path)])
    for item in overrides:
        cmd.extend(["--set", item])
    return cmd


def _run_subprocess(cmd: List[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(
            f"子阶段执行失败（exit code={result.returncode}）: {' '.join(cmd)}"
        )


def run_pipeline(
    config_path: str | Path,
    cfg: ProjectConfig,
    overrides: Optional[Sequence[str]] = None,
    force: bool = False,
    skip_validate: bool = False,
) -> Dict[str, Optional[str]]:
    """
    一键端到端编排器：
    validate -> train-scr -> train-p1 -> train-p2
    每个训练阶段以独立子进程运行（彻底释放显存），并依据 stage_done.json 自动跳过已完成阶段。
    """
    override_list = list(overrides) if overrides else []
    cfg.project_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据体检
    if not skip_validate:
        report = validate_dataset(cfg)
        print(format_validation_report(report), flush=True)
        report.save_json(cfg.project_dir / "validation_report.json")
        if not report.ok:
            raise SystemExit("数据体检未通过，已中止训练。请先修复上述 [ERROR] 项。")

    stages = [
        ("scr", "train-scr", cfg.scr.enabled, cfg.scr_dir),
        ("phase1", "train-p1", cfg.phase1.enabled, cfg.phase1_dir),
        ("phase2", "train-p2", cfg.phase2.enabled, cfg.phase2_dir),
    ]

    for stage_name, cli_cmd, enabled, stage_dir in stages:
        if not enabled:
            print(f"[Pipeline] 跳过已禁用阶段: {stage_name}", flush=True)
            continue

        done_info = read_stage_done(stage_dir)
        if done_info is not None and not force:
            print(
                f"[Pipeline] 检测到阶段 {stage_name} 已完成（step={done_info.get('global_step')}），"
                "自动跳过（如需重跑请加 --force）。",
                flush=True,
            )
            continue

        print(f"\n[Pipeline] 开始执行阶段: {stage_name} ...", flush=True)
        cmd = _build_stage_command(
            stage_cmd=cli_cmd,
            config_path=config_path,
            overrides=override_list,
            use_accelerate=cfg.runtime.use_accelerate,
        )
        _run_subprocess(cmd)

    # 汇总产物路径
    scr_ckpt = find_latest_scr_ckpt(cfg.scr_dir)
    p1_ckpt = find_latest_step_dir(cfg.phase1_dir)
    p2_ckpt = find_latest_step_dir(cfg.phase2_dir)
    final_ckpt = p2_ckpt or p1_ckpt

    summary = {
        "project_dir": str(cfg.project_dir),
        "scr_ckpt": str(scr_ckpt) if scr_ckpt else cfg.scr.pretrained_ckpt_path,
        "phase1_ckpt_dir": str(p1_ckpt) if p1_ckpt else None,
        "phase2_ckpt_dir": str(p2_ckpt) if p2_ckpt else None,
        "recommended_infer_ckpt_dir": str(final_ckpt) if final_ckpt else None,
    }

    print("\n" + "=" * 64, flush=True)
    print(f"[Pipeline 完成] 项目: {cfg.project_name}", flush=True)
    print(f"  输出根目录   : {summary['project_dir']}", flush=True)
    print(f"  SCR 权重     : {summary['scr_ckpt'] or '(无)'}", flush=True)
    print(f"  Phase 1 权重 : {summary['phase1_ckpt_dir'] or '(无)'}", flush=True)
    print(f"  Phase 2 权重 : {summary['phase2_ckpt_dir'] or '(无)'}", flush=True)
    if final_ckpt:
        print(
            "  快速推理命令 :\n"
            f"    style-diffuser infer --ckpt-dir \"{final_ckpt}\" "
            "--style-image <风格图> --content-image <内容图> "
            f"--output-dir \"{cfg.outputs_dir}\"",
            flush=True,
        )
    print("=" * 64, flush=True)
    return summary
