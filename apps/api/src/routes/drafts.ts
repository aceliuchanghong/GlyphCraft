import { Hono } from "hono";
import {
  DEFAULT_BRAND_CONFIG,
  type CreateDraftRequest,
  type CreateDraftResponse,
} from "@glyphcraft/shared";
import type { Env } from "../env";
import { decodeBase64Png } from "../lib/security";

export const draftsRoute = new Hono<{ Bindings: Env }>();

draftsRoute.post("/", async (c) => {
  let body: Partial<CreateDraftRequest>;
  try {
    body = await c.req.json();
  } catch {
    return c.json({ error: "Invalid JSON body." }, 400);
  }

  const images = Array.isArray(body.images) ? body.images : [];
  if (images.length < 1 || images.length > DEFAULT_BRAND_CONFIG.maxStyleRefs) {
    return c.json({ error: "Provide between 1 and 3 reference PNGs." }, 400);
  }

  const decodedBytes: Uint8Array[] = [];
  try {
    for (const img of images) {
      decodedBytes.push(decodeBase64Png(img).bytes);
    }
  } catch (err) {
    return c.json(
      { error: err instanceof Error ? err.message : "Invalid PNG image." },
      400
    );
  }

  const draftId = crypto.randomUUID();
  const keys: string[] = [];

  await Promise.all(
    decodedBytes.map(async (bytes, idx) => {
      const key = `drafts/${draftId}/ref_${idx}.png`;
      keys.push(key);
      await c.env.R2_BUCKET.put(key, bytes, {
        httpMetadata: { contentType: "image/png" },
      });
    })
  );

  // Preserve sorted deterministic order in metadata
  const sortedKeys = decodedBytes.map((_, idx) => `drafts/${draftId}/ref_${idx}.png`);
  await c.env.R2_BUCKET.put(
    `drafts/${draftId}/meta.json`,
    JSON.stringify({
      draftId,
      keys: sortedKeys,
      createdAt: new Date().toISOString(),
    }),
    {
      httpMetadata: { contentType: "application/json" },
    }
  );

  const response: CreateDraftResponse = {
    draftId,
    refCount: sortedKeys.length,
  };
  return c.json(response, 201);
});
