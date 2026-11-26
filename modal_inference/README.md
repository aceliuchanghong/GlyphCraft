# GlyphCraft Modal Inference Service

Standalone Modal deployment for PixelDiT-Flow (`font_en`) GPU inference and CPU TrueType vectorization.

## 1. Upload Model Weights to Modal Volume

Upload ONLY the runtime artifacts from `adapters/font_en/models/font_en_128_p2_step37500/`:

```bash
modal volume create glyphcraft-weights
modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/ema.pth /font_en_128/ema.pth
modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/model_config.yaml /font_en_128/model_config.yaml
modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/resolved_config.yaml /font_en_128/resolved_config.yaml
modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/content /font_en_128/content
```

## 2. Create `glyphcraft-secrets`

```bash
modal secret create glyphcraft-secrets \
  INTERNAL_API_SECRET="replace-with-strong-secret" \
  WORKERS_CALLBACK_URL="https://api.yourdomain.com/api/internal/modal-callback" \
  R2_ACCOUNT_ID="your-cloudflare-account-id" \
  R2_ENDPOINT_URL="https://your-cloudflare-account-id.r2.cloudflarestorage.com" \
  R2_BUCKET_NAME="glyphcraft-assets" \
  R2_ACCESS_KEY_ID="your-r2-access-key-id" \
  R2_SECRET_ACCESS_KEY="your-r2-secret-access-key" \
  WEIGHTS_SUBDIR="font_en_128"
```

## 3. Deploy or Serve Locally

```bash
modal deploy modal_inference/app.py
# Or for local ephemeral testing:
modal serve modal_inference/app.py
```
