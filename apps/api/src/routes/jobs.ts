import { Hono } from "hono";
import {
  clampTtfParams,
  sanitizeFamilyName,
  type FontJob,
  type FontJobWithUrls,
  type RepackRequest,
  type RepackResponse,
} from "@glyphcraft/shared";
import type { Env } from "../env";
import { createR2PresignedGetUrl } from "../lib/r2Presign";
import { SupabaseAdmin, verifySupabaseUser } from "../lib/supabase";

export const jobsRoute = new Hono<{ Bindings: Env }>();

async function attachPresignedUrls(
  env: Env,
  job: FontJob
): Promise<FontJobWithUrls> {
  if (job.status !== "completed" || !job.r2_ttf_key) {
    return {
      ...job,
      ttfUrl: null,
      ttfDownloadUrl: null,
      previewUrl: null,
    };
  }

  const safeSlug = job.family_name.replace(/[^A-Za-z0-9_-]+/g, "_") || "CustomFont";
  const [ttfUrl, ttfDownloadUrl, previewUrl] = await Promise.all([
    createR2PresignedGetUrl(env, job.r2_ttf_key, { expiresInSeconds: 3600 }),
    createR2PresignedGetUrl(env, job.r2_ttf_key, {
      expiresInSeconds: 3600,
      downloadFilename: `${safeSlug}.ttf`,
    }),
    job.r2_preview_key
      ? createR2PresignedGetUrl(env, job.r2_preview_key, { expiresInSeconds: 3600 })
      : Promise.resolve(null),
  ]);

  return {
    ...job,
    ttfUrl,
    ttfDownloadUrl,
    previewUrl,
  };
}

jobsRoute.get("/", async (c) => {
  const user = await verifySupabaseUser(c.req.raw, c.env);
  if (!user) {
    return c.json({ error: "Authentication required." }, 401);
  }

  const sb = new SupabaseAdmin(c.env);
  const jobs = await sb.listJobsForUser(user.id);
  const hydrated = await Promise.all(
    jobs.map((job) => attachPresignedUrls(c.env, job))
  );
  return c.json({ jobs: hydrated }, 200);
});

jobsRoute.get("/:id", async (c) => {
  const user = await verifySupabaseUser(c.req.raw, c.env);
  if (!user) {
    return c.json({ error: "Authentication required." }, 401);
  }

  const jobId = c.req.param("id");
  const sb = new SupabaseAdmin(c.env);
  const job = await sb.getJobForUser(jobId, user.id);
  if (!job) {
    return c.json({ error: "Font job not found." }, 404);
  }

  const hydrated = await attachPresignedUrls(c.env, job);
  return c.json({ job: hydrated }, 200);
});

jobsRoute.post("/:id/repack", async (c) => {
  const user = await verifySupabaseUser(c.req.raw, c.env);
  if (!user) {
    return c.json({ error: "Authentication required." }, 401);
  }

  const jobId = c.req.param("id");
  const sb = new SupabaseAdmin(c.env);
  const job = await sb.getJobForUser(jobId, user.id);
  if (!job) {
    return c.json({ error: "Font job not found." }, 404);
  }
  if (job.status !== "completed" || !job.r2_glyphs_zip_key) {
    return c.json(
      { error: "Font can only be repacked after full generation completes." },
      400
    );
  }

  let body: Partial<RepackRequest>;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body." }, 400);
  }

  const familyName = sanitizeFamilyName(body.familyName ?? job.family_name);
  const params = clampTtfParams({
    threshold: body.threshold ?? job.ttf_params.threshold,
    sidebearing: body.sidebearing ?? job.ttf_params.sidebearing,
    curveMode: body.curveMode ?? job.ttf_params.curve_mode,
  });

  const modalBase = c.env.MODAL_API_BASE_URL.replace(/\/$/, "");
  const repackResp = await fetch(`${modalBase}/repack-ttf`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "X-Internal-Secret": c.env.INTERNAL_API_SECRET,
    },
    body: JSON.stringify({
      job_id: job.id,
      glyphs_zip_key: job.r2_glyphs_zip_key,
      family_name: familyName,
      threshold: params.threshold,
      sidebearing: params.sidebearing,
      curve_mode: params.curve_mode,
    }),
  });

  if (!repackResp.ok) {
    const errText = await repackResp.text();
    return c.json(
      { error: `Repack service error (${repackResp.status}): ${errText.slice(0, 200)}` },
      502
    );
  }

  const modalData = (await repackResp.json()) as {
    r2_ttf_key: string;
    ttf_params: FontJob["ttf_params"];
  };

  const updated = await sb.updateFontJob(job.id, {
    family_name: familyName,
    r2_ttf_key: modalData.r2_ttf_key,
    ttf_params: modalData.ttf_params ?? params,
  });

  if (!updated) {
    return c.json({ error: "Failed to persist repacked font metadata." }, 500);
  }

  const hydrated = await attachPresignedUrls(c.env, updated);
  const res: RepackResponse = {
    job: hydrated,
    ttfUrl: hydrated.ttfUrl ?? "",
    ttfDownloadUrl: hydrated.ttfDownloadUrl ?? "",
  };
  return c.json(res, 200);
});
