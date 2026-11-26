import { Hono } from "hono";
import { clampTtfParams, sanitizeFamilyName } from "@glyphcraft/shared";
import type { Env } from "../env";
import { verifyCreemWebhookSignature } from "../lib/creem";
import { SupabaseAdmin } from "../lib/supabase";

export const webhooksRoute = new Hono<{ Bindings: Env }>();

interface CreemWebhookPayload {
  eventType?: string;
  event_type?: string;
  type?: string;
  object?: Record<string, unknown>;
  data?: Record<string, unknown>;
}

webhooksRoute.post("/creem", async (c) => {
  const rawBody = await c.req.text();
  const signature =
    c.req.header("creem-signature") ?? c.req.header("x-creem-signature");

  const valid = await verifyCreemWebhookSignature(
    rawBody,
    signature,
    c.env.CREEM_WEBHOOK_SECRET
  );
  if (!valid) {
    return c.json({ error: "Invalid webhook signature." }, 401);
  }

  let payload: CreemWebhookPayload;
  try {
    payload = JSON.parse(rawBody) as CreemWebhookPayload;
  } catch {
    return c.json({ error: "Invalid JSON webhook payload." }, 400);
  }

  const eventType = (
    payload.eventType ??
    payload.event_type ??
    payload.type ??
    ""
  ).toLowerCase();

  // Only trigger font generation on completed checkouts
  if (eventType !== "checkout.completed") {
    return c.json({ ok: true, ignored: eventType || "unknown" }, 200);
  }

  const obj = (payload.object ?? payload.data ?? {}) as Record<string, unknown>;
  const orderObj = (obj.order as Record<string, unknown> | undefined) ?? {};
  const metadata = (obj.metadata as Record<string, unknown> | undefined) ?? {};

  const creemCheckoutId =
    typeof obj.id === "string"
      ? obj.id
      : typeof obj.checkout_id === "string"
        ? obj.checkout_id
        : null;
  const creemOrderId =
    typeof orderObj.id === "string"
      ? orderObj.id
      : typeof obj.order_id === "string"
        ? obj.order_id
        : creemCheckoutId ?? `evt_${crypto.randomUUID()}`;

  const userId = typeof metadata.user_id === "string" ? metadata.user_id.trim() : "";
  const draftId = typeof metadata.draft_id === "string" ? metadata.draft_id.trim() : "";
  const userEmail =
    typeof metadata.user_email === "string" && metadata.user_email.trim()
      ? metadata.user_email.trim()
      : null;

  if (!userId || !draftId) {
    return c.json({ error: "Missing user_id or draft_id in Creem metadata." }, 400);
  }

  const sb = new SupabaseAdmin(c.env);

  // Strict idempotency by creem_order_id
  const existingOrder = await sb.findOrderByCreemOrderId(creemOrderId);
  if (existingOrder) {
    const existingJob = await sb.findJobByOrderId(existingOrder.id);
    if (existingJob) {
      return c.json({ ok: true, idempotent: true, jobId: existingJob.id }, 200);
    }
  }

  const rawAmount = Number(
    orderObj.amount ?? obj.amount ?? c.env.FONT_PRICE_CENTS ?? "900"
  );
  const amountCents = Number.isFinite(rawAmount) ? Math.max(0, Math.round(rawAmount)) : 900;
  const currency =
    typeof orderObj.currency === "string"
      ? orderObj.currency
      : typeof obj.currency === "string"
        ? obj.currency
        : "USD";

  await sb.ensureProfile(userId, userEmail);

  const order =
    existingOrder ??
    (await sb.insertOrder({
      user_id: userId,
      creem_checkout_id: creemCheckoutId,
      creem_order_id: creemOrderId,
      amount_cents: amountCents,
      currency,
      status: "paid",
    }));

  const metaObj = await c.env.R2_BUCKET.get(`drafts/${draftId}/meta.json`);
  if (!metaObj) {
    return c.json({ error: `Draft ${draftId} metadata missing in R2.` }, 404);
  }
  const metaJson = (await metaObj.json()) as { keys?: string[] };
  const draftKeys = Array.isArray(metaJson.keys) ? metaJson.keys : [];
  if (draftKeys.length === 0) {
    return c.json({ error: `Draft ${draftId} contains no reference keys.` }, 400);
  }

  const familyName = sanitizeFamilyName(
    typeof metadata.family_name === "string" ? metadata.family_name : "CustomFont"
  );
  const ttfParams = clampTtfParams({
    curve_mode: metadata.curve_mode === "polygon" ? "polygon" : "spline",
  });

  const job = await sb.insertFontJob({
    user_id: userId,
    order_id: order.id,
    draft_id: draftId,
    family_name: familyName,
    status: "queued",
    progress_message: "Payment verified. Queuing GPU inference...",
    ttf_params: ttfParams,
  });

  const modalBase = c.env.MODAL_API_BASE_URL.replace(/\/$/, "");
  const spawnResp = await fetch(`${modalBase}/generate-full`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Internal-Secret": c.env.INTERNAL_API_SECRET,
    },
    body: JSON.stringify({
      job_id: job.id,
      draft_keys: draftKeys,
      family_name: familyName,
      curve_mode: ttfParams.curve_mode,
      threshold: ttfParams.threshold,
      sidebearing: ttfParams.sidebearing,
    }),
  });

  if (!spawnResp.ok) {
    const errText = await spawnResp.text();
    await sb.updateFontJob(job.id, {
      status: "failed",
      progress_message: "Failed to dispatch GPU job",
      error_message: `Modal spawn error (${spawnResp.status}): ${errText.slice(0, 200)}`,
    });
    return c.json({ error: "Failed to trigger Modal job." }, 502);
  }

  return c.json({ ok: true, jobId: job.id }, 200);
});
