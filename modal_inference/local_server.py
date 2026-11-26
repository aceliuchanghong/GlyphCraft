"""Local GPU/CPU FastAPI inference server for GlyphCraft development.

Runs the exact same PixelDiT-Flow inference pipeline as `modal_inference/app.py`
against the local weights in `local_weights/font_en_128/`, without needing Modal.

Usage:
    C:\\Users\\lawrence\\PycharmProjects\\style_diffuser\\.venv\\Scripts\\python.exe modal_inference/local_server.py
"""

from __future__ import annotations

import base64
from dataclasses import replace
import hmac
import io
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
import torch
import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_DIR = Path(__file__).resolve().parent / "vendor"
if str(VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(VENDOR_DIR))

from font_en.charset import ASCII_94, glyph_stem, load_ascii_charset, stem_to_char  # noqa: E402
from font_en.render_en import DEFAULT_BASELINE_RATIO, render_sentence_preview  # noqa: E402
from style_diffuser.config import SamplingConfig, load_config  # noqa: E402
from style_diffuser.data.preprocess import load_rgb_image  # noqa: E402
from style_diffuser.engine.infer import infer_images, load_inference_pipeline  # noqa: E402


DEFAULT_CKPT_DIR = REPO_ROOT / "local_weights" / "font_en_128"
INTERNAL_API_SECRET = os.environ.get("INTERNAL_API_SECRET", "local-dev-secret-123456")


def _decode_style_png(raw_input: str, expected_size: int = 128) -> Image.Image:
    payload = raw_input.split(",", 1)[1] if "," in raw_input else raw_input
    raw_bytes = base64.b64decode(payload)
    with Image.open(io.BytesIO(raw_bytes)) as img:
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        composited = Image.alpha_composite(bg, rgba).convert("RGB")
        if composited.size != (expected_size, expected_size):
            composited = composited.resize(
                (expected_size, expected_size), Image.Resampling.BILINEAR
            )
        return composited


class LocalFontEngine:
    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = ckpt_dir
        self.lock = threading.Lock()
        self._loaded = False
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.model_cfg = None
        self.data_cfg = None
        self.base_sampling_cfg = SamplingConfig()
        self.content_cache: Dict[str, Image.Image] = {}

    def load(self) -> None:
        if self._loaded:
            return
        with self.lock:
            if self._loaded:
                return
            t0 = time.perf_counter()
            if not (self.ckpt_dir / "ema.pth").is_file() and not (self.ckpt_dir / "pixel_dit.pth").is_file():
                raise FileNotFoundError(
                    f"Model weights not found in {self.ckpt_dir}. Expected ema.pth or pixel_dit.pth."
                )

            data_cfg = None
            sampling_cfg = SamplingConfig()
            bundled_cfg = self.ckpt_dir / "resolved_config.yaml"
            if bundled_cfg.is_file():
                cfg = load_config(bundled_cfg)
                data_cfg = cfg.data
                sampling_cfg = cfg.sampling

            print(
                f"[local-inference] Loading model from {self.ckpt_dir} on {self.device}...",
                flush=True,
            )
            pipe, model_cfg, resolved_data_cfg = load_inference_pipeline(
                ckpt_dir=self.ckpt_dir,
                device=self.device,
                sampling_cfg=sampling_cfg,
                data_cfg=data_cfg,
            )

            content_dir = self.ckpt_dir / "content"
            if not content_dir.is_dir():
                raise FileNotFoundError(
                    f"Missing content/ glyph templates in {self.ckpt_dir}"
                )

            content_cache: Dict[str, Image.Image] = {}
            for ch in ASCII_94:
                stem = glyph_stem(ch)
                img_path = content_dir / f"{stem}.png"
                if img_path.is_file():
                    content_cache[stem] = load_rgb_image(
                        img_path, background_color=resolved_data_cfg.background_color
                    )

            if len(content_cache) < 94:
                raise RuntimeError(
                    f"Expected 94 content templates in {content_dir}, found {len(content_cache)}"
                )

            self.pipe = pipe
            self.model_cfg = model_cfg
            self.data_cfg = resolved_data_cfg
            self.base_sampling_cfg = sampling_cfg
            self.content_cache = content_cache
            self._loaded = True
            print(
                f"[local-inference] Ready in {time.perf_counter() - t0:.2f}s ({self.device})",
                flush=True,
            )

    def render_preview(
        self,
        style_images_b64: List[str],
        text: str,
        seed: int = 123,
    ) -> bytes:
        self.load()
        assert self.pipe is not None and self.data_cfg is not None and self.model_cfg is not None

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


ckpt_dir = Path(os.environ.get("LOCAL_WEIGHTS_DIR", str(DEFAULT_CKPT_DIR)))
engine = LocalFontEngine(ckpt_dir=ckpt_dir)

app = FastAPI(title="GlyphCraft Local Inference Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PreviewPayload(BaseModel):
    images: List[str] = Field(..., min_length=1, max_length=3)
    text: str = Field(..., min_length=1, max_length=16)
    seed: int = Field(default_factory=lambda: random.randint(0, 2147483647))


def _verify_secret(x_internal_secret: Optional[str]) -> None:
    if not INTERNAL_API_SECRET:
        return
    if not x_internal_secret or not hmac.compare_digest(x_internal_secret, INTERNAL_API_SECRET):
        raise HTTPException(status_code=401, detail="Invalid internal secret")


@app.on_event("startup")
def _startup() -> None:
    engine.load()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": engine.device,
        "loaded": engine._loaded,
        "ckpt_dir": str(engine.ckpt_dir),
    }


@app.post("/preview")
def preview(
    payload: PreviewPayload,
    x_internal_secret: Optional[str] = Header(default=None),
):
    _verify_secret(x_internal_secret)
    png_bytes = engine.render_preview(
        style_images_b64=payload.images,
        text=payload.text,
        seed=payload.seed,
    )
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return {
        "preview_base64": encoded,
        "preview_data_url": f"data:image/png;base64,{encoded}",
    }


if __name__ == "__main__":
    port = int(os.environ.get("LOCAL_INFERENCE_PORT", "8000"))
    uvicorn.run(app, host="127.0.0.1", port=port)
