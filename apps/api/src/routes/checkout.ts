import { Hono } from "hono";
import {
  sanitizeFamilyName,
  type CheckoutRequest,
  type CheckoutResponse,
  type CurveMode,
} from "@glyphcraft/shared";
import type { Env } from "../env";
import { createCreemCheckout } from "../lib/creem";
import { SupabaseAdmin, verifySupabaseUser } from "../lib/supabase";

export const checkoutRoute = new Hono<{ Bindings: Env }>();

checkoutRoute.post("/", async (c) => {
  const user = await verifySupabaseUser(c.req.raw, c.env);
  if (!user) {
    return c.json({ error: "Authentication required.", code: "UNAUTHORIZED" }, 401);
  }

  let body: Partial<CheckoutRequest>;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body." }, 400);
  }

  const draftId = typeof body.draftId === "string" ? body.draftId.trim() : "";
  if (!draftId || !/^[0-9a-f-]{8,64}$/i.test(draftId)) {
    return c.json({ error: "Invalid or missing draftId." }, 400);
  }

  const metaObj = await c.env.R2_BUCKET.get(`drafts/${draftId}/meta.json`);
  if (!metaObj) {
    return c.json(
      { error: "Draft not found or expired. Please re-save your glyphs." },
      404
    );
  }

  const familyName = sanitizeFamilyName(body.familyName);
  const curveMode: CurveMode = body.curveMode === "polygon" ? "polygon" : "spline";

  const sb = new SupabaseAdmin(c.env);
  await sb.ensureProfile(user.id, user.email);

  const requestId = `gc_${crypto.randomUUID()}`;
  const session = await createCreemCheckout(c.env, {
    userId: user.id,
    userEmail: user.email,
    draftId,
    familyName,
    curveMode,
    requestId,
  });

  const response: CheckoutResponse = {
    checkoutUrl: session.checkout_url,
    checkoutId: session.id,
  };
  return c.json(response, 200);
});
