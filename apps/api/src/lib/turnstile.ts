import type { Env } from "../env";

interface TurnstileVerifyResponse {
  success: boolean;
  "error-codes"?: string[];
}

export async function verifyTurnstileToken(
  env: Env,
  token: string,
  remoteIp: string
): Promise<{ success: boolean; errorCodes?: string[] }> {
  if (
    env.ENVIRONMENT === "development" &&
    env.DEV_BYPASS_TURNSTILE === "true"
  ) {
    return { success: true };
  }

  if (!token || !token.trim()) {
    return { success: false, errorCodes: ["missing-input-response"] };
  }

  const body = new URLSearchParams({
    secret: env.TURNSTILE_SECRET_KEY,
    response: token.trim(),
    remoteip: remoteIp,
  });

  const resp = await fetch(
    "https://challenges.cloudflare.com/turnstile/v0/siteverify",
    {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: body.toString(),
    }
  );

  if (!resp.ok) {
    return { success: false, errorCodes: [`http-${resp.status}`] };
  }

  const data = (await resp.json()) as TurnstileVerifyResponse;
  return {
    success: Boolean(data.success),
    errorCodes: data["error-codes"],
  };
}
