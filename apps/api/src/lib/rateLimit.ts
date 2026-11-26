import type { Env } from "../env";

export interface QuotaCheckResult {
  allowed: boolean;
  remaining: number;
  limit: number;
  key: string;
}

function getDailyLimit(env: Env): number {
  const parsed = Number(env.FREE_PREVIEW_DAILY_LIMIT ?? "3");
  return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : 3;
}

function getDailyKey(ip: string): string {
  const today = new Date().toISOString().slice(0, 10);
  return `preview_quota:${today}:${ip}`;
}

export async function consumePreviewQuota(
  env: Env,
  ip: string
): Promise<QuotaCheckResult> {
  const limit = getDailyLimit(env);
  const key = getDailyKey(ip);
  const raw = await env.RATE_LIMIT_KV.get(key);
  const used = raw ? Math.max(0, Number.parseInt(raw, 10) || 0) : 0;

  if (used >= limit) {
    return {
      allowed: false,
      remaining: 0,
      limit,
      key,
    };
  }

  const nextUsed = used + 1;
  await env.RATE_LIMIT_KV.put(key, String(nextUsed), {
    expirationTtl: 60 * 60 * 26,
  });

  return {
    allowed: true,
    remaining: Math.max(0, limit - nextUsed),
    limit,
    key,
  };
}

export async function refundPreviewQuota(env: Env, key: string): Promise<void> {
  const raw = await env.RATE_LIMIT_KV.get(key);
  if (!raw) return;
  const used = Math.max(0, Number.parseInt(raw, 10) || 0);
  if (used <= 1) {
    await env.RATE_LIMIT_KV.delete(key);
  } else {
    await env.RATE_LIMIT_KV.put(key, String(used - 1), {
      expirationTtl: 60 * 60 * 26,
    });
  }
}
