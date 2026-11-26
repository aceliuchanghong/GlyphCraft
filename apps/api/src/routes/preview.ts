import { Hono } from "hono";
import {
  DEFAULT_BRAND_CONFIG,
  validatePreviewText,
  type PreviewRequest,
  type PreviewResponse,
} from "@glyphcraft/shared";
import type { Env } from "../env";
import { consumePreviewQuota, refundPreviewQuota } from "../lib/rateLimit";
import { decodeBase64Png, getClientIp } from "../lib/security";
import { verifyTurnstileToken } from "../lib/turnstile";

export const previewRoute = new Hono<{ Bindings: Env }>();

previewRoute.post("/", async (c) => {
  let body: Partial<PreviewRequest>;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body." }, 400);
  }

  const images = Array.isArray(body.images) ? body.images : [];
  if (images.length < 1 || images.length > DEFAULT_BRAND_CONFIG.maxStyleRefs) {
    return c.json({ error: "Provide between 1 and 3 style reference PNGs." }, 400);
  }

  const textCheck = validatePreviewText(
    typeof body.text === "string" ? body.text : "",
    DEFAULT_BRAND_CONFIG.maxPreviewChars
  );
  if (!textCheck.valid) {
    return c.json({ error: textCheck.error ?? "Invalid preview text." }, 400);
  }

  const cleanedBase64s: string[] = [];
  try {
    for (const img of images) {
      const decoded = decodeBase64Png(img);
      cleanedBase64s.push(decoded.base64);
    }
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : "Invalid PNG image." },
      400
    );
  }

  const ip = getClientIp(c.req.raw);
  const turnstile = await verifyTurnstileToken(
    c.env,
    typeof body.turnstileToken === "string" ? body.turnstileToken : "",
    ip
  );
  if (!turnstile.success) {
    return c.json(
      {
        error: "Human verification failed. Please retry Turnstile.",
        code: "TURNSTILE_FAILED",
      },
      403
    );
  }

  const quota = await consumePreviewQuota(c.env, ip);
  c.header("X-Remaining-Quota", String(quota.remaining));
  c.header("X-Quota-Limit", String(quota.limit));

  if (!quota.allowed) {
    return c.json(
      {
        error: `Daily free preview limit (${quota.limit}/day) reached for this IP. Unlock a full 94-ASCII font or try again tomorrow.`,
        code: "QUOTA_EXCEEDED",
        remainingQuota: 0,
      },
      429
    );
  }

  const modalBase = c.env.MODAL_API_BASE_URL.replace(/\/$/, "");
  try {
    const modalResp = await fetch(`${modalBase}/preview`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Internal-Secret": c.env.INTERNAL_API_SECRET,
      },
      body: JSON.stringify({
        images: cleanedBase64s,
        text: textCheck.cleaned,
      }),
    });

    if (!modalResp.ok) {
      if (modalResp.status >= 500) {
        await refundPreviewQuota(c.env, quota.key);
      }
      const errText = await modalResp.text();
      return c.json(
        { error: `Inference service error (${modalResp.status}): ${errText.slice(0, 200)}` },
        502
      );
    }

    const modalData = (await modalResp.json()) as {
      preview_data_url?: string;
      preview_base64?: string;
    };

    const previewDataUrl =
      modalData.preview_data_url ??
      (modalData.preview_base64
        ? `data:image/png;base64,${modalData.preview_base64}`
        : "");

    if (!previewDataUrl) {
      await refundPreviewQuota(c.env, quota.key);
      return c.json({ error: "Inference returned empty preview image." }, 502);
    }

    const res: PreviewResponse = {
      previewDataUrl,
      remainingQuota: quota.remaining,
      limit: quota.limit,
    };
    return c.json(res, 200);
  } catch (err) {
    await refundPreviewQuota(c.env, quota.key);
    return c.json(
      {
        error:
          err instanceof Error
            ? `Failed to reach inference backend: ${err.message}`
            : "Failed to reach inference backend.",
      },
      502
    );
  }
});
