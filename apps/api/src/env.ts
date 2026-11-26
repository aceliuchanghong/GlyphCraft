export interface Env {
  R2_BUCKET: R2Bucket;
  RATE_LIMIT_KV: KVNamespace;

  ENVIRONMENT?: string;
  DEV_BYPASS_TURNSTILE?: string;
  BRAND_NAME?: string;
  FRONTEND_URL: string;
  FREE_PREVIEW_DAILY_LIMIT?: string;
  FONT_PRICE_CENTS?: string;

  INTERNAL_API_SECRET: string;
  MODAL_API_BASE_URL: string;
  TURNSTILE_SECRET_KEY: string;

  R2_ACCOUNT_ID: string;
  R2_BUCKET_NAME: string;
  R2_ACCESS_KEY_ID: string;
  R2_SECRET_ACCESS_KEY: string;

  SUPABASE_URL: string;
  SUPABASE_ANON_KEY: string;
  SUPABASE_SERVICE_ROLE_KEY: string;

  CREEM_API_BASE: string;
  CREEM_API_KEY: string;
  CREEM_PRODUCT_ID: string;
  CREEM_WEBHOOK_SECRET: string;

  RESEND_API_KEY: string;
  RESEND_FROM_EMAIL: string;
}
