"""GlyphCraft Modal Inference & CPU TTF Repacking Service.

Exposes a FastAPI ASGI gateway (`web_api`) backed by:
  1. `FontInferenceEngine` (L4 GPU, `scaledown_window=120`):
     - `render_preview`: synchronous 4-step fast mode preview sentence PNG
     - `generate_full_job`: asynchronous 10-step HD 94-ASCII generation + TTF packing + R2 upload + Workers callback
  2. `repack_ttf_cpu` (CPU-only):
     - Re-vectorizes cached `glyphs.zip` into a new `.ttf` with updated `threshold`, `sidebearing`, `curve_mode`, and `family_name`
"""

from __future__ import annotations

import base64
from dataclasses import replace
import hmac
import io
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
from typing import Any, Dict, List, Literal, Optional
import zipfile

import modal

APP_NAME = "glyphcraft-inference"
WEIGHTS_MOUNT_PATH = "/weights"
HF_CACHE_DIR = "/cache/hf"
VENDOR_REMOTE_PATH = "/app/vendor"
VENDOR_LOCAL_PATH = Path(__file__).resolve().parent / "vendor"

app = modal.App(APP_NAME)
weights_volume = modal.Volume.from_name("glyphcraft-weights", create_if_missing=True)
app_secret = modal.Secret.from_name("glyphcraft-secrets")


def _precache_hf_backbone() -> None:
    """Download frozen DINOv2-small backbone during image build for instant cold starts."""
    from transformers import AutoImageProcessor, AutoModel

    model_id = "facebook/dinov2-small"
    AutoImageProcessor.from_pretrained(model_id)
    AutoModel.from_pretrained(model_id)


cpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi>=0.115.0",
        "pydantic>=2.10.0",
        "boto3>=1.35.0",
        "httpx>=0.28.0",
        "pillow>=11.0.0",
        "numpy>=2.0.0",
        "tqdm>=4.67.1",
        "vtracer>=0.6.11",
        "fonttools>=4.59.2",
    )
    .add_local_dir(str(VENDOR_LOCAL_PATH), remote_path=VENDOR_REMOTE_PATH)
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.6.0",
        "torchvision>=0.21.0",
        "transformers>=4.56.0",
        "diffusers>=0.35.1",
        "accelerate>=1.10.1",
        "kornia>=0.8.1",
        "pyyaml>=6.0.2",
        "pillow>=11.0.0",
        "numpy>=2.0.0",
        "tqdm>=4.67.1",
        "vtracer>=0.6.11",
        "fonttools>=4.59.2",
        "boto3>=1.35.0",
        "httpx>=0.28.0",
        "fastapi>=0.115.0",
        "pydantic>=2.10.0",
    )
    .env({"HF_HOME": HF_CACHE_DIR, "TOKENIZERS_PARALLELISM": "false"})
    .run_function(_precache_hf_backbone)
    .add_local_dir(str(VENDOR_LOCAL_PATH), remote_path=VENDOR_REMOTE_PATH)
)


def _ensure_vendor_on_path() -> None:
    for candidate in (VENDOR_REMOTE_PATH, str(VENDOR_LOCAL_PATH)):
        if candidate not in sys.path and os.path.isdir(candidate):
            sys.path.insert(0, candidate)


def _make_r2_client():
    import boto3
    from botocore.config import Config

    endpoint_url = os.environ.get("R2_ENDPOINT_URL")
    if not endpoint_url:
        account_id = os.environ["R2_ACCOUNT_ID"]
        endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4"),
    )


def _decode_style_png(raw_input: str | bytes, expected_size: int = 128):
    from PIL import Image

    if isinstance(raw_input, str):
        payload = raw_input.split(",", 1)[1] if "," in raw_input else raw_input
        raw_bytes = base64.b64decode(payload)
    else:
        raw_bytes = raw_input

    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(bg, rgba).convert("RGB")
        if composited.size != (expected_size, expected_size):
            composited = composited.resize(
                (expected_size, expected_size), Image.Resampling.BILINEAR
            )
        return composited


def _sanitize_family_name(name: str) -> str:
    cleaned = "".join(ch for ch in (name or "").strip() if ch.isalnum() or ch in " _-")
    cleaned = " ".join(cleaned.split())[:40]
    return cleaned or "CustomFont"


def _stack_rows(rows, gap: int = 12):
    from PIL import Image

    width = max(row.width for row in rows)
    height = sum(row.height for row in rows) + gap * (len(rows) - 1)
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height + gap
    return canvas


def _build_preview_sheet(glyph_map: Dict[str, Any], resolution: int, baseline_y: int):
    _ensure_vendor_on_path()
    from font_en.render_en import render_sentence_preview

    punct_line = "".join(chr(cp) for cp in range(33, 127) if not chr(cp).isalnum())
    rows = [
        render_sentence_preview(
            glyph_map,
            text="Hamburgefontsiv 0123456789",
            image_size=resolution,
            baseline_y=baseline_y,
        ),
        render_sentence_preview(
            glyph_map,
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            image_size=resolution,
            baseline_y=baseline_y,
        ),
        render_sentence_preview(
            glyph_map,
            text="abcdefghijklmnopqrstuvwxyz",
            image_size=resolution,
            baseline_y=baseline_y,
        ),
        render_sentence_preview(
            glyph_map,
            text="0123456789",
            image_size=resolution,
            baseline_y=baseline_y,
        ),
        render_sentence_preview(
            glyph_map,
            text=punct_line,
            image_size=resolution,
            baseline_y=baseline_y,
        ),
    ]
    return _stack_rows(rows)


def _post_callback(payload: Dict[str, Any]) -> None:
    import httpx

    callback_url = os.environ.get("WORKERS_CALLBACK_URL", "").strip()
    if not callback_url:
        return
    secret = os.environ["INTERNAL_API_SECRET"]
    headers = {
        "X-Internal-Secret": secret,
        "Content-Type": "application/json",
    }

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = httpx.post(callback_url, json=payload, headers=headers, timeout=20.0)
            resp.raise_for_status()
            return
        except Exception as exc:
            last_exc = exc
            time.sleep(1.5 * (2**attempt))

    if last_exc is not None:
        print(f"[modal-callback] Failed after retries: {last_exc}", flush=True)


@app.cls(
    image=gpu_image,
    gpu="L4",
    scaledown_window=120,
    timeout=300,
    volumes={WEIGHTS_MOUNT_PATH: weights_volume},
    secrets=[app_secret],
)
class FontInferenceEngine:
    @modal.enter()
    def startup(self) -> None:
        _ensure_vendor_on_path()
        import torch
        from font_en.charset import ASCII_94, glyph_stem
        from style_diffuser.config import SamplingConfig, load_config
        from style_diffuser.data.preprocess import load_rgb_image
        from style_diffuser.engine.infer import load_inference_pipeline

        subdir = os.environ.get("WEIGHTS_SUBDIR", "font_en_128").strip()
        ckpt_dir = Path(WEIGHTS_MOUNT_PATH) / subdir
        if not (ckpt_dir / "ema.pth").is_file() and not (ckpt_dir / "pixel_dit.pth").is_file():
            # Fallback if files were uploaded directly to volume root
            root_ckpt = Path(WEIGHTS_MOUNT_PATH)
            if (root_ckpt / "ema.pth").is_file() or (root_ckpt / "pixel_dit.pth").is_file():
                ckpt_dir = root_ckpt
            else:
                raise FileNotFoundError(
                    f"Model weights not found at {ckpt_dir} or {root_ckpt}. "
                    "Upload ema.pth, model_config.yaml, resolved_config.yaml, and content/ to the Modal Volume."
                )

        data_cfg = None
        sampling_cfg = SamplingConfig()
        bundled_cfg = ckpt_dir / "resolved_config.yaml"
        if bundled_cfg.is_file():
            cfg = load_config(bundled_cfg)
            data_cfg = cfg.data
            sampling_cfg = cfg.sampling

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        pipe, model_cfg, resolved_data_cfg = load_inference_pipeline(
            ckpt_dir=ckpt_dir,
            device=device,
            sampling_cfg=sampling_cfg,
            data_cfg=data_cfg,
        )

        content_dir = ckpt_dir / "content"
        if not content_dir.is_dir():
            raise FileNotFoundError(f"Missing content/ glyph templates in {ckpt_dir}")

        content_cache = {}
        for ch in ASCII_94:
            stem = glyph_stem(ch)
            img_path = content_dir / f"{stem}.png"
            if img_path.is_file():
                content_cache[stem] = load_rgb_image(
                    img_path, background_color=resolved_data_cfg.background_color
                )

        if len(content_cache) < 94:
            raise RuntimeError(
                f"Expected 94 ASCII content templates in {content_dir}, found {len(content_cache)}"
            )

        self.device = device
        self.pipe = pipe
        self.model_cfg = model_cfg
        self.data_cfg = resolved_data_cfg
        self.base_sampling_cfg = sampling_cfg
        self.content_cache = content_cache
        self.lock = threading.Lock()

    @modal.method()
    def render_preview(
        self,
        style_images_b64: List[str],
        text: str,
        seed: int = 123,
    ) -> bytes:
        _ensure_vendor_on_path()
        from font_en.charset import glyph_stem, load_ascii_charset, stem_to_char
        from font_en.render_en import DEFAULT_BASELINE_RATIO, render_sentence_preview
        from style_diffuser.engine.infer import infer_images

        chars = load_ascii_charset(chars=text)
        if len(chars) > 16:
            chars = chars[:16]

        res = self.data_cfg.resolution
        style_pils = [_decode_style_png(b64, expected_size=res) for b64 in style_images_b64[:3]]
        if not style_pils:
            raise ValueError("At least 1 style reference image is required.")

        content_items = [
            (glyph_stem(ch), self.content_cache[glyph_stem(ch)])
            for ch in chars
            if glyph_stem(ch) in self.content_cache
        ]

        with self.lock:
            sampling = replace(
                self.base_sampling_cfg,
                mode="fast",
                num_inference_steps=4,
            )
            self.pipe.sampling_cfg = sampling
            results = infer_images(
                pipe=self.pipe,
                model_cfg=self.model_cfg,
                data_cfg=self.data_cfg,
                sampling_cfg=sampling,
                style_image=style_pils,
                content_items=content_items,
                output_dir=None,
                batch_size=16,
                seed=seed,
                device=self.device,
                save_comparison=False,
            )

        glyph_map = {}
        for stem, pil_img in results:
            ch = stem_to_char(stem)
            if ch is not None:
                glyph_map[ch] = pil_img

        baseline_y = int(round(res * DEFAULT_BASELINE_RATIO))
        preview_img = render_sentence_preview(
            glyph_map,
            text=text,
            image_size=res,
            baseline_y=baseline_y,
        )
        buf = io.BytesIO()
        preview_img.save(buf, format="PNG")
        return buf.getvalue()

    @modal.method()
    def generate_full_job(
        self,
        job_id: str,
        draft_keys: List[str],
        family_name: str,
        curve_mode: str = "spline",
        threshold: int = 160,
        sidebearing: int = 55,
        seed: int = 123,
    ) -> Dict[str, Any]:
        _ensure_vendor_on_path()
        from font_en.charset import ASCII_94, glyph_stem, stem_to_char
        from font_en.render_en import DEFAULT_BASELINE_RATIO, DEFAULT_EM_RATIO
        from font_en.to_ttf import build_ttf_from_glyph_dir
        from style_diffuser.engine.infer import infer_images

        bucket = os.environ["R2_BUCKET_NAME"]
        s3 = _make_r2_client()
        safe_family = _sanitize_family_name(family_name)
        safe_curve_mode = "polygon" if curve_mode == "polygon" else "spline"
        safe_threshold = max(80, min(235, int(threshold)))
        safe_sidebearing = max(0, min(200, int(sidebearing)))

        try:
            _post_callback(
                {
                    "jobId": job_id,
                    "status": "running",
                    "progressMessage": "Generating all 94 ASCII glyphs (10-step HD flow)...",
                }
            )

            res = self.data_cfg.resolution
            style_pils = []
            for key in draft_keys[:3]:
                obj = s3.get_object(Bucket=bucket, Key=key)
                style_pils.append(_decode_style_png(obj["Body"].read(), expected_size=res))

            if not style_pils:
                raise ValueError("No draft reference images found in R2.")

            content_items = [
                (glyph_stem(ch), self.content_cache[glyph_stem(ch)])
                for ch in ASCII_94
                if glyph_stem(ch) in self.content_cache
            ]

            with tempfile.TemporaryDirectory() as tmpdir:
                work_dir = Path(tmpdir)
                glyphs_dir = work_dir / "glyphs"
                glyphs_dir.mkdir(parents=True, exist_ok=True)

                with self.lock:
                    sampling = replace(
                        self.base_sampling_cfg,
                        mode="hd",
                        num_inference_steps=10,
                    )
                    self.pipe.sampling_cfg = sampling
                    results = infer_images(
                        pipe=self.pipe,
                        model_cfg=self.model_cfg,
                        data_cfg=self.data_cfg,
                        sampling_cfg=sampling,
                        style_image=style_pils,
                        content_items=content_items,
                        output_dir=glyphs_dir,
                        batch_size=16,
                        seed=seed,
                        device=self.device,
                        save_comparison=False,
                    )

                _post_callback(
                    {
                        "jobId": job_id,
                        "status": "running",
                        "progressMessage": "Vectorizing 94 glyphs into TrueType (.ttf)...",
                    }
                )

                glyph_map = {}
                for stem, pil_img in results:
                    ch = stem_to_char(stem)
                    if ch is not None:
                        glyph_map[ch] = pil_img

                baseline_y = int(round(res * DEFAULT_BASELINE_RATIO))
                preview_sheet = _build_preview_sheet(glyph_map, res, baseline_y)
                preview_path = work_dir / "preview_sheet.png"
                preview_sheet.save(preview_path, format="PNG")

                zip_path = work_dir / "glyphs.zip"
                with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for ch in ASCII_94:
                        png_path = glyphs_dir / f"{glyph_stem(ch)}.png"
                        if png_path.is_file():
                            zf.write(png_path, arcname=png_path.name)

                ttf_path = work_dir / "font.ttf"
                build_ttf_from_glyph_dir(
                    input_dir=glyphs_dir,
                    output_ttf=ttf_path,
                    family_name=safe_family,
                    style_name="Regular",
                    baseline_ratio=DEFAULT_BASELINE_RATIO,
                    em_ratio=DEFAULT_EM_RATIO,
                    threshold=safe_threshold,
                    sidebearing=safe_sidebearing,
                    curve_mode=safe_curve_mode,
                )

                r2_ttf_key = f"jobs/{job_id}/font.ttf"
                r2_preview_key = f"jobs/{job_id}/preview_sheet.png"
                r2_glyphs_zip_key = f"jobs/{job_id}/glyphs.zip"

                with ttf_path.open("rb") as fh:
                    s3.put_object(
                        Bucket=bucket,
                        Key=r2_ttf_key,
                        Body=fh.read(),
                        ContentType="font/ttf",
                    )
                with preview_path.open("rb") as fh:
                    s3.put_object(
                        Bucket=bucket,
                        Key=r2_preview_key,
                        Body=fh.read(),
                        ContentType="image/png",
                    )
                with zip_path.open("rb") as fh:
                    s3.put_object(
                        Bucket=bucket,
                        Key=r2_glyphs_zip_key,
                        Body=fh.read(),
                        ContentType="application/zip",
                    )

            ttf_params = {
                "threshold": safe_threshold,
                "sidebearing": safe_sidebearing,
                "curve_mode": safe_curve_mode,
            }
            _post_callback(
                {
                    "jobId": job_id,
                    "status": "completed",
                    "progressMessage": "Font ready for preview & download",
                    "r2TtfKey": r2_ttf_key,
                    "r2PreviewKey": r2_preview_key,
                    "r2GlyphsZipKey": r2_glyphs_zip_key,
                    "ttfParams": ttf_params,
                }
            )
            return {
                "r2_ttf_key": r2_ttf_key,
                "r2_preview_key": r2_preview_key,
                "r2_glyphs_zip_key": r2_glyphs_zip_key,
                "ttf_params": ttf_params,
            }
        except Exception as exc:
            _post_callback(
                {
                    "jobId": job_id,
                    "status": "failed",
                    "progressMessage": "Generation failed",
                    "errorMessage": str(exc)[:400],
                }
            )
            raise


@app.function(
    image=cpu_image,
    cpu=2.0,
    memory=1024,
    timeout=60,
    secrets=[app_secret],
)
def repack_ttf_cpu(
    job_id: str,
    glyphs_zip_key: str,
    family_name: str,
    threshold: int = 160,
    sidebearing: int = 55,
    curve_mode: str = "spline",
) -> Dict[str, Any]:
    _ensure_vendor_on_path()
    from font_en.render_en import DEFAULT_BASELINE_RATIO, DEFAULT_EM_RATIO
    from font_en.to_ttf import build_ttf_from_glyph_dir

    bucket = os.environ["R2_BUCKET_NAME"]
    s3 = _make_r2_client()
    safe_family = _sanitize_family_name(family_name)
    safe_curve_mode = "polygon" if curve_mode == "polygon" else "spline"
    safe_threshold = max(80, min(235, int(threshold)))
    safe_sidebearing = max(0, min(200, int(sidebearing)))

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        glyphs_dir = work_dir / "glyphs"
        glyphs_dir.mkdir(parents=True, exist_ok=True)

        obj = s3.get_object(Bucket=bucket, Key=glyphs_zip_key)
        zip_bytes = obj["Body"].read()
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            for member in zf.namelist():
                filename = Path(member).name
                if filename.startswith("u") and filename.endswith(".png"):
                    (glyphs_dir / filename).write_bytes(zf.read(member))

        ttf_path = work_dir / "font_repacked.ttf"
        build_ttf_from_glyph_dir(
            input_dir=glyphs_dir,
            output_ttf=ttf_path,
            family_name=safe_family,
            style_name="Regular",
            baseline_ratio=DEFAULT_BASELINE_RATIO,
            em_ratio=DEFAULT_EM_RATIO,
            threshold=safe_threshold,
            sidebearing=safe_sidebearing,
            curve_mode=safe_curve_mode,
        )

        new_key = f"jobs/{job_id}/font_v{int(time.time())}.ttf"
        with ttf_path.open("rb") as fh:
            s3.put_object(
                Bucket=bucket,
                Key=new_key,
                Body=fh.read(),
                ContentType="font/ttf",
            )

    return {
        "r2_ttf_key": new_key,
        "ttf_params": {
            "threshold": safe_threshold,
            "sidebearing": safe_sidebearing,
            "curve_mode": safe_curve_mode,
        },
    }


@app.function(
    image=cpu_image,
    cpu=0.5,
    memory=512,
    scaledown_window=60,
    secrets=[app_secret],
)
@modal.asgi_app()
def web_api():
    from fastapi import FastAPI, Header, HTTPException
    from pydantic import BaseModel, Field

    api = FastAPI(title="GlyphCraft Inference Gateway")

    def _verify_secret(x_internal_secret: Optional[str]) -> None:
        expected = os.environ.get("INTERNAL_API_SECRET", "")
        if not expected or not x_internal_secret:
            raise HTTPException(status_code=401, detail="Missing internal secret")
        if not hmac.compare_digest(x_internal_secret, expected):
            raise HTTPException(status_code=401, detail="Invalid internal secret")

    class PreviewPayload(BaseModel):
        images: List[str] = Field(..., min_length=1, max_length=3)
        text: str = Field(..., min_length=1, max_length=16)
        seed: int = 123

    class GenerateFullPayload(BaseModel):
        job_id: str
        draft_keys: List[str] = Field(..., min_length=1, max_length=3)
        family_name: str = "CustomFont"
        curve_mode: Literal["spline", "polygon"] = "spline"
        threshold: int = 160
        sidebearing: int = 55
        seed: int = 123

    class RepackPayload(BaseModel):
        job_id: str
        glyphs_zip_key: str
        family_name: str = "CustomFont"
        threshold: int = 160
        sidebearing: int = 55
        curve_mode: Literal["spline", "polygon"] = "spline"

    @api.get("/health")
    def health():
        return {"status": "ok"}

    @api.post("/preview")
    def preview(
        payload: PreviewPayload,
        x_internal_secret: Optional[str] = Header(default=None),
    ):
        _verify_secret(x_internal_secret)
        png_bytes = FontInferenceEngine().render_preview.remote(
            style_images_b64=payload.images,
            text=payload.text,
            seed=payload.seed,
        )
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return {
            "preview_base64": encoded,
            "preview_data_url": f"data:image/png;base64,{encoded}",
        }

    @api.post("/generate-full", status_code=202)
    def generate_full(
        payload: GenerateFullPayload,
        x_internal_secret: Optional[str] = Header(default=None),
    ):
        _verify_secret(x_internal_secret)
        call = FontInferenceEngine().generate_full_job.spawn(
            job_id=payload.job_id,
            draft_keys=payload.draft_keys,
            family_name=payload.family_name,
            curve_mode=payload.curve_mode,
            threshold=payload.threshold,
            sidebearing=payload.sidebearing,
            seed=payload.seed,
        )
        return {"accepted": True, "call_id": call.object_id}

    @api.post("/repack-ttf")
    def repack_ttf(
        payload: RepackPayload,
        x_internal_secret: Optional[str] = Header(default=None),
    ):
        _verify_secret(x_internal_secret)
        return repack_ttf_cpu.remote(
            job_id=payload.job_id,
            glyphs_zip_key=payload.glyphs_zip_key,
            family_name=payload.family_name,
            threshold=payload.threshold,
            sidebearing=payload.sidebearing,
            curve_mode=payload.curve_mode,
        )

    return api
