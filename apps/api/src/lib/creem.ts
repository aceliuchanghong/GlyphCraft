import type { CurveMode } from "@glyphcraft/shared";
import type { Env } from "../env";
import { timingSafeEqualStr } from "./security";

export interface CreemCheckoutSessionResponse {
  id: string;
  checkout_url: string;
}

export async function createCreemCheckout(
  env: Env,
  params: {
    userId: string;
    userEmail: string | null;
    draftId: string;
    familyName: string;
    curveMode: CurveMode;
    requestId: string;
  }
): Promise<CreemCheckoutSessionResponse> {
  const apiBase = (env.CREEM_API_BASE || "https://api.creem.io").replace(/\/$/, "");
  const frontendUrl = env.FRONTEND_URL.replace(/\/$/, "");
  const successUrl = `${frontendUrl}/dashboard?checkout=success&draft_id=${encodeURIComponent(
    params.draftId
  )}`;

  const payload: Record<string, unknown> = {
    product_id: env.CREEM_PRODUCT_ID,
    request_id: params.requestId,
    success_url: successUrl,
    metadata: {
      user_id: params.userId,
      draft_id: params.draftId,
      family_name: params.familyName,
      curve_mode: params.curveMode,
      user_email: params.userEmail ?? "",
    },
  };

  if (params.userEmail) {
    payload.customer = { email: params.userEmail };
  }

  const resp = await fetch(`${apiBase}/v1/checkouts`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": env.CREEM_API_KEY,
    },
    body: JSON.stringify(payload),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Creem checkout API error (${resp.status}): ${errText}`);
  }

  const data = (await resp.json()) as Partial<CreemCheckoutSessionResponse>;
  if (!data.checkout_url || !data.id) {
    throw new Error("Creem checkout API returned incomplete response.");
  }

  return {
    id: data.id,
    checkout_url: data.checkout_url,
  };
}

export async function verifyCreemWebhookSignature(
  rawBody: string,
  signatureHeader: string | null | undefined,
  webhookSecret: string
): Promise<boolean> {
  if (!signatureHeader || !webhookSecret) {
    return false;
  }

  const enc = new TextEncoder();
  const key = await crypto.subtle.importKey(
    "raw",
    enc.encode(webhookSecret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"]
  );
  const sigBuffer = await crypto.subtle.sign("HMAC", key, enc.encode(rawBody));
  const computedHex = Array.from(new Uint8Array(sigBuffer), (b) =>
    b.toString(16).padStart(2, "0")
  ).join("");

  const received = signatureHeader.replace(/^sha256=/i, "").trim().toLowerCase();
  return timingSafeEqualStr(computedHex, received);
}
