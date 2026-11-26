export type CurveMode = "spline" | "polygon";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export type OrderStatus = "pending" | "paid" | "refunded";

export interface TtfParams {
  threshold: number;
  sidebearing: number;
  curve_mode: CurveMode;
}

export interface FontJob {
  id: string;
  user_id: string;
  order_id: string | null;
  draft_id: string;
  family_name: string;
  status: JobStatus;
  progress_message: string;
  r2_ttf_key: string | null;
  r2_preview_key: string | null;
  r2_glyphs_zip_key: string | null;
  ttf_params: TtfParams;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface FontJobWithUrls extends FontJob {
  ttfUrl: string | null;
  ttfDownloadUrl: string | null;
  previewUrl: string | null;
}

export interface PreviewRequest {
  images: string[];
  text: string;
  turnstileToken: string;
}

export interface PreviewResponse {
  previewDataUrl: string;
  remainingQuota: number;
  limit: number;
}

export interface CreateDraftRequest {
  images: string[];
}

export interface CreateDraftResponse {
  draftId: string;
  refCount: number;
}

export interface CheckoutRequest {
  draftId: string;
  familyName: string;
  curveMode: CurveMode;
}

export interface CheckoutResponse {
  checkoutUrl: string;
  checkoutId: string;
}

export interface RepackRequest {
  familyName: string;
  threshold: number;
  sidebearing: number;
  curveMode: CurveMode;
}

export interface RepackResponse {
  job: FontJobWithUrls;
  ttfUrl: string;
  ttfDownloadUrl: string;
}

export interface ModalCallbackPayload {
  jobId: string;
  status: JobStatus;
  progressMessage?: string;
  r2TtfKey?: string | null;
  r2PreviewKey?: string | null;
  r2GlyphsZipKey?: string | null;
  ttfParams?: TtfParams;
  errorMessage?: string | null;
}

export interface ApiErrorResponse {
  error: string;
  code?: string;
  remainingQuota?: number;
}
