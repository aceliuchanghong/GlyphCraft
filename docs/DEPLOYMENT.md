# GlyphCraft Deployment & Operations Guide

Complete setup, secret management, local development, and production deployment instructions for **GlyphCraft** (configurable brand name).

---

## 1. System Architecture

```text
Browser (React 19 + Vite on Cloudflare Pages)
  ├── /api/preview          ──▶ Cloudflare Workers (Hono)
  │                              ├── Turnstile siteverify
  │                              ├── Cloudflare KV (3/day/IP quota)
  │                              └── Modal /preview (L4 GPU, 4-step fast, 0 storage)
  │
  ├── /api/drafts           ──▶ Cloudflare Workers ──▶ R2 (drafts/{draftId}/ref_*.png)
  │
  ├── Google Sign-In        ──▶ Supabase Auth (OAuth)
  │
  ├── /api/checkout         ──▶ Cloudflare Workers ──▶ Creem Checkout Session
  │
  ├── Creem Webhook         ──▶ Cloudflare Workers (/api/webhooks/creem)
  │                              ├── Verify HMAC signature + idempotent orders row
  │                              ├── Insert font_jobs (status=queued)
  │                              └── Modal /generate-full (.spawn, returns 202)
  │                                     ├── 10-step HD 94-ASCII inference
  │                                     ├── to_ttf.py vectorization
  │                                     ├── Upload .ttf, preview_sheet.png, glyphs.zip to R2
  │                                     └── POST /api/internal/modal-callback
  │                                            ├── Update Supabase font_jobs
  │                                            └── Send email via Resend
  │
  └── /jobs/:id             ──▶ Cloudflare Workers (/api/jobs/:id & /repack)
                                 ├── 1-hour AWS SigV4 presigned R2 URLs
                                 └── Modal CPU /repack-ttf (fast re-vectorize)
```

---

## 2. Environment Variables & Secrets Checklist

### A. Frontend (`apps/web/.env.local` / Cloudflare Pages env vars)

| Variable | Description | Example / Dev Value |
| :--- | :--- | :--- |
| `VITE_BRAND_NAME` | Brand name shown across UI | `GlyphCraft` |
| `VITE_API_BASE_URL` | Cloudflare Workers API base URL | `http://127.0.0.1:8787` |
| `VITE_SUPABASE_URL` | Supabase project URL | `https://xxxx.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Supabase public anon key | `eyJ...` |
| `VITE_TURNSTILE_SITE_KEY` | Cloudflare Turnstile site key | `1x00000000000000000000AA` (dev test key) |
| `VITE_PRICE_DISPLAY` | Display price badge on pricing & checkout cards | `$9` |

### B. Cloudflare Workers (`apps/api/.dev.vars` locally, `wrangler secret put` in prod)

| Variable | Kind | Description |
| :--- | :--- | :--- |
| `ENVIRONMENT` | Var | `development` or `production` |
| `DEV_BYPASS_TURNSTILE` | Var | `true` ONLY for local dev if needed; `false` in prod |
| `BRAND_NAME` | Var | Brand name used in transactional email subjects & body |
| `FRONTEND_URL` | Var | Allowed CORS origin & redirect base (e.g. `https://glyphcraft.app`) |
| `FREE_PREVIEW_DAILY_LIMIT` | Var | Daily free preview limit per IP (default `3`) |
| `FONT_PRICE_CENTS` | Var | Fallback order amount in cents (`900`) |
| `INTERNAL_API_SECRET` | Secret | Shared secret (`openssl rand -hex 32`) with Modal |
| `MODAL_API_BASE_URL` | Secret/Var | Modal ASGI URL from `modal deploy` |
| `TURNSTILE_SECRET_KEY` | Secret | Cloudflare Turnstile secret key |
| `R2_ACCOUNT_ID` | Secret | Cloudflare Account ID (for SigV4 presign) |
| `R2_BUCKET_NAME` | Secret | R2 bucket name (`glyphcraft-assets`) |
| `R2_ACCESS_KEY_ID` | Secret | R2 S3 API token Access Key ID |
| `R2_SECRET_ACCESS_KEY` | Secret | R2 S3 API token Secret Access Key |
| `SUPABASE_URL` | Secret | Supabase project URL |
| `SUPABASE_ANON_KEY` | Secret | Supabase anon key (used for `/auth/v1/user` JWT validation) |
| `SUPABASE_SERVICE_ROLE_KEY` | Secret | Supabase service role key (bypasses RLS for server writes) |
| `CREEM_API_BASE` | Var | `https://test-api.creem.io` (sandbox) or `https://api.creem.io` (prod) |
| `CREEM_API_KEY` | Secret | Creem API key (`creem_...`) |
| `CREEM_PRODUCT_ID` | Secret | Creem one-time product ID (`prod_...`) |
| `CREEM_WEBHOOK_SECRET` | Secret | Creem webhook signing secret (`whsec_...`) |
| `RESEND_API_KEY` | Secret | Resend API key (`re_...`) |
| `RESEND_FROM_EMAIL` | Secret | Verified sender (e.g. `GlyphCraft <fonts@yourdomain.com>`) |

### C. Modal (`modal secret create glyphcraft-secrets ...`)

| Key | Description |
| :--- | :--- |
| `INTERNAL_API_SECRET` | Must exactly match Workers `INTERNAL_API_SECRET` |
| `WORKERS_CALLBACK_URL` | `https://api.yourdomain.com/api/internal/modal-callback` |
| `R2_ACCOUNT_ID` | Cloudflare Account ID |
| `R2_ENDPOINT_URL` | `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com` |
| `R2_BUCKET_NAME` | `glyphcraft-assets` |
| `R2_ACCESS_KEY_ID` | R2 S3 API token Access Key ID |
| `R2_SECRET_ACCESS_KEY` | R2 S3 API token Secret Access Key |
| `WEIGHTS_SUBDIR` | `font_en_128` |

---

## 3. Step-by-Step Third-Party Setup

### Step 1: Supabase (Auth + Postgres + RLS)
1. Create a new Supabase project.
2. Open **SQL Editor** and run [`supabase/migrations/001_init.sql`](../supabase/migrations/001_init.sql).
3. In **Authentication → Providers → Google**, enable Google OAuth and configure your Google Cloud OAuth Client ID/Secret.
4. In **Authentication → URL Configuration**:
   - Set **Site URL** to `https://yourdomain.com` (or `http://localhost:5173` during local dev).
   - Add Redirect URLs: `https://yourdomain.com/auth/callback` and `http://localhost:5173/auth/callback`.

### Step 2: Cloudflare R2 + KV + Turnstile
1. Create an R2 bucket named `glyphcraft-assets`.
2. **CRITICAL — Configure R2 Bucket CORS** so the browser can dynamically load signed `.ttf` URLs via `@font-face`:
   ```json
   [
     {
       "AllowedOrigins": [
         "https://yourdomain.com",
         "http://localhost:5173"
       ],
       "AllowedMethods": ["GET", "HEAD"],
       "AllowedHeaders": ["*"],
       "MaxAgeSeconds": 3600
     }
   ]
   ```
3. Under **R2 → Manage R2 API Tokens**, create an Object Read & Write token and save `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY`.
4. Create a KV namespace:
   ```bash
   npx wrangler kv namespace create RATE_LIMIT_KV
   ```
   Copy the returned `id` into `apps/api/wrangler.toml`.
5. In **Cloudflare Dashboard → Turnstile**, create a widget for your domain and copy the Site Key + Secret Key.

### Step 3: Modal Inference Service (`modal_inference/`)
1. Install and authenticate Modal CLI:
   ```bash
   pip install modal
   modal setup
   ```
2. Create the weights volume and upload ONLY the inference runtime assets from `style_diffuser/adapters/font_en/models/font_en_128_p2_step37500/`:
   ```bash
   modal volume create glyphcraft-weights
   modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/ema.pth /font_en_128/ema.pth
   modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/model_config.yaml /font_en_128/model_config.yaml
   modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/resolved_config.yaml /font_en_128/resolved_config.yaml
   modal volume put glyphcraft-weights path/to/font_en_128_p2_step37500/content /font_en_128/content
   ```
   > Note: Do **not** upload `pixel_dit.pth` if using EMA weights (`use_ema: true`), keeping the volume minimal.
3. Create the Modal secret:
   ```bash
   modal secret create glyphcraft-secrets \
     INTERNAL_API_SECRET="..." \
     WORKERS_CALLBACK_URL="https://api.yourdomain.com/api/internal/modal-callback" \
     R2_ACCOUNT_ID="..." \
     R2_ENDPOINT_URL="https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com" \
     R2_BUCKET_NAME="glyphcraft-assets" \
     R2_ACCESS_KEY_ID="..." \
     R2_SECRET_ACCESS_KEY="..." \
     WEIGHTS_SUBDIR="font_en_128"
   ```
4. Deploy Modal:
   ```bash
   modal deploy modal_inference/app.py
   ```
   Copy the printed `web_api` URL (`https://...modal.run`) into `MODAL_API_BASE_URL`.

### Step 4: Creem Payments
1. Create a One-Time Product in Creem (e.g. `$9.00 USD`).
2. Copy the Product ID (`prod_...`) and API Key (`creem_...`).
3. Add a Webhook endpoint pointing to:
   `https://api.yourdomain.com/api/webhooks/creem`
   Subscribe to `checkout.completed` and copy the `CREEM_WEBHOOK_SECRET`.

### Step 5: Resend Email
1. Verify your sending domain in Resend (SPF/DKIM DNS records).
2. Create an API key (`re_...`) and set `RESEND_FROM_EMAIL`.

---

## 4. Local Full-Stack Development

1. Install workspace dependencies:
   ```bash
   npm install
   ```
2. Create `apps/api/.dev.vars` and `apps/web/.env.local` using `.env.example` as a template.
3. Start the 3 local processes in separate terminals:
   ```bash
   # Terminal 1: Modal ephemeral ASGI + GPU/CPU workers
   modal serve modal_inference/app.py

   # Terminal 2: Cloudflare Workers API (http://127.0.0.1:8787)
   npm run dev:api

   # Terminal 3: Vite React frontend (http://localhost:5173)
   npm run dev:web
   ```
   > To test Creem webhooks locally, expose `127.0.0.1:8787` using `cloudflared tunnel --url http://127.0.0.1:8787` and point both `WORKERS_CALLBACK_URL` and Creem's test webhook to that tunnel URL.

---

## 5. Production Deployment

### Deploy Workers API (`apps/api`)
```bash
cd apps/api
npx wrangler secret put INTERNAL_API_SECRET
npx wrangler secret put MODAL_API_BASE_URL
npx wrangler secret put TURNSTILE_SECRET_KEY
npx wrangler secret put R2_ACCOUNT_ID
npx wrangler secret put R2_BUCKET_NAME
npx wrangler secret put R2_ACCESS_KEY_ID
npx wrangler secret put R2_SECRET_ACCESS_KEY
npx wrangler secret put SUPABASE_URL
npx wrangler secret put SUPABASE_ANON_KEY
npx wrangler secret put SUPABASE_SERVICE_ROLE_KEY
npx wrangler secret put CREEM_API_KEY
npx wrangler secret put CREEM_PRODUCT_ID
npx wrangler secret put CREEM_WEBHOOK_SECRET
npx wrangler secret put RESEND_API_KEY
npx wrangler secret put RESEND_FROM_EMAIL

npx wrangler deploy
```

### Deploy Frontend (`apps/web` on Cloudflare Pages)
- **Root directory**: repository root
- **Build command**: `npm run build` (or `npm --workspace @glyphcraft/web run build`)
- **Build output directory**: `apps/web/dist`
- **Environment variables**: set all `VITE_*` variables in Cloudflare Pages project settings.

---

## 6. Rebranding from GlyphCraft

The brand name is centrally configurable without touching business logic:
1. Update `DEFAULT_BRAND_CONFIG` in `packages/shared/src/brand.ts` (optional default).
2. Set `VITE_BRAND_NAME` in Cloudflare Pages (`apps/web`) and `BRAND_NAME` in `apps/api/wrangler.toml`.
3. Update `<title>` in `apps/web/index.html` if desired.
