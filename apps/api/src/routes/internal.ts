import { Hono } from "hono";
import type { ModalCallbackPayload } from "@glyphcraft/shared";
import type { Env } from "../env";
import { sendFontReadyEmail } from "../lib/resend";
import { timingSafeEqualStr } from "../lib/security";
import { SupabaseAdmin } from "../lib/supabase";

export const internalRoute = new Hono<{ Bindings: Env }>();

internalRoute.post("/modal-callback", async (c) => {
  const providedSecret = c.req.header("X-Internal-Secret") ?? "";
  if (
    !c.env.INTERNAL_API_SECRET ||
    !(await timingSafeEqualStr(providedSecret, c.env.INTERNAL_API_SECRET))
  ) {
    return c.json({ error: "Unauthorized internal callback." }, 401);
  }

  let body: ModalCallbackPayload;
  try {
    body = (await c.req.json()) as ModalCallbackPayload;
  } catch {
    return c.json({ error: "Invalid JSON payload." }, 400);
  }

  if (!body.jobId || !body.status) {
    return c.json({ error: "Missing jobId or status." }, 400);
  }

  const sb = new SupabaseAdmin(c.env);
  const patch: Parameters<SupabaseAdmin["updateFontJob"]>[1] = {
    status: body.status,
  };
  if (body.progressMessage !== undefined) patch.progress_message = body.progressMessage;
  if (body.r2TtfKey !== undefined) patch.r2_ttf_key = body.r2TtfKey;
  if (body.r2PreviewKey !== undefined) patch.r2_preview_key = body.r2PreviewKey;
  if (body.r2GlyphsZipKey !== undefined) patch.r2_glyphs_zip_key = body.r2GlyphsZipKey;
  if (body.ttfParams !== undefined) patch.ttf_params = body.ttfParams;
  if (body.errorMessage !== undefined) patch.error_message = body.errorMessage;

  const updated = await sb.updateFontJob(body.jobId, patch);
  if (!updated) {
    return c.json({ error: "Font job not found." }, 404);
  }

  if (body.status === "completed") {
    try {
      const email = await sb.getProfileEmail(updated.user_id);
      if (email) {
        await sendFontReadyEmail(c.env, {
          to: email,
          familyName: updated.family_name,
          jobId: updated.id,
        });
      }
    } catch (err) {
      console.error("[modal-callback] Resend email failed:", err);
    }
  }

  return c.json({ ok: true }, 200);
});
