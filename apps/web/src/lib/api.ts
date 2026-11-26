import type {
  CheckoutRequest,
  CheckoutResponse,
  CreateDraftRequest,
  CreateDraftResponse,
  FontJobWithUrls,
  PreviewRequest,
  PreviewResponse,
  RepackRequest,
  RepackResponse,
} from "@glyphcraft/shared";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8787").replace(
  /\/$/,
  ""
);

export class ApiError extends Error {
  status: number;
  code?: string;
  remainingQuota?: number;

  constructor(
    message: string,
    status: number,
    code?: string,
    remainingQuota?: number
  ) {
    super(message);
    this.status = status;
    this.code = code;
    this.remainingQuota = remainingQuota;
  }
}

async function apiRequest<T>(
  path: string,
  options: RequestInit & { accessToken?: string } = {}
): Promise<T> {
  const { accessToken, headers, ...rest } = options;
  const reqHeaders = new Headers(headers);
  if (!reqHeaders.has("Content-Type") && rest.body) {
    reqHeaders.set("Content-Type", "application/json");
  }
  if (accessToken) {
    reqHeaders.set("Authorization", `Bearer ${accessToken}`);
  }

  const resp = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: reqHeaders,
  });

  const data = (await resp.json().catch(() => ({}))) as {
    error?: string;
    code?: string;
    remainingQuota?: number;
  };

  if (!resp.ok) {
    throw new ApiError(
      data.error || `Request failed (${resp.status})`,
      resp.status,
      data.code,
      data.remainingQuota
    );
  }

  return data as T;
}

export function generateFreePreview(
  payload: PreviewRequest
): Promise<PreviewResponse> {
  return apiRequest<PreviewResponse>("/api/preview", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function saveDraftToR2(
  payload: CreateDraftRequest
): Promise<CreateDraftResponse> {
  return apiRequest<CreateDraftResponse>("/api/drafts", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function createCheckoutSession(
  payload: CheckoutRequest,
  accessToken: string
): Promise<CheckoutResponse> {
  return apiRequest<CheckoutResponse>("/api/checkout", {
    method: "POST",
    body: JSON.stringify(payload),
    accessToken,
  });
}

export function listFontJobs(
  accessToken: string
): Promise<{ jobs: FontJobWithUrls[] }> {
  return apiRequest<{ jobs: FontJobWithUrls[] }>("/api/jobs", {
    method: "GET",
    accessToken,
  });
}

export function getFontJob(
  jobId: string,
  accessToken: string
): Promise<{ job: FontJobWithUrls }> {
  return apiRequest<{ job: FontJobWithUrls }>(
    `/api/jobs/${encodeURIComponent(jobId)}`,
    {
      method: "GET",
      accessToken,
    }
  );
}

export function repackFontJob(
  jobId: string,
  payload: RepackRequest,
  accessToken: string
): Promise<RepackResponse> {
  return apiRequest<RepackResponse>(
    `/api/jobs/${encodeURIComponent(jobId)}/repack`,
    {
      method: "POST",
      body: JSON.stringify(payload),
      accessToken,
    }
  );
}
