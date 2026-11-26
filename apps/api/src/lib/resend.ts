import { resolveBrandName } from "@glyphcraft/shared";
import type { Env } from "../env";

function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

export async function sendFontReadyEmail(
  env: Env,
  params: {
    to: string;
    familyName: string;
    jobId: string;
  }
): Promise<void> {
  if (!env.RESEND_API_KEY || !env.RESEND_FROM_EMAIL) {
    console.warn("[resend] Skipping email because RESEND_API_KEY or RESEND_FROM_EMAIL is unset.");
    return;
  }

  const brand = resolveBrandName(env.BRAND_NAME);
  const frontendUrl = env.FRONTEND_URL.replace(/\/$/, "");
  const jobUrl = `${frontendUrl}/jobs/${encodeURIComponent(params.jobId)}`;
  const safeFamily = escapeHtml(params.familyName);
  const safeBrand = escapeHtml(brand);

  const html = `
  <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 560px; margin: 0 auto; padding: 32px 24px; color: #111827; background: #fafaf8;">
    <div style="border: 1px solid #e5e7eb; background: #ffffff; border-radius: 16px; padding: 32px;">
      <p style="font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase; color: #d94625; margin: 0 0 12px 0; font-weight: 600;">${safeBrand}</p>
      <h1 style="font-size: 24px; line-height: 1.25; margin: 0 0 16px 0; font-weight: 700;">Your font "${safeFamily}" is ready.</h1>
      <p style="font-size: 15px; line-height: 1.6; color: #4b5563; margin: 0 0 24px 0;">
        All 94 printable English ASCII glyphs (A–Z, a–z, 0–9, and punctuation) have been generated in HD and packed into an installable TrueType (<code>.ttf</code>) font.
      </p>
      <a href="${jobUrl}" style="display: inline-block; background: #0a0a0b; color: #ffffff; text-decoration: none; font-weight: 600; font-size: 14px; padding: 14px 24px; border-radius: 10px;">
        Preview, Tune &amp; Download .TTF
      </a>
      <p style="font-size: 13px; line-height: 1.5; color: #6b7280; margin: 24px 0 0 0;">
        You can also fine-tune stroke thickness, letter spacing, and spline/polygon vectorization directly inside your ${safeBrand} studio page at no extra cost.
      </p>
    </div>
  </div>
  `.trim();

  const resp = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RESEND_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      from: env.RESEND_FROM_EMAIL,
      to: [params.to],
      subject: `[${brand}] Your "${params.familyName}" .ttf font is ready`,
      html,
    }),
  });

  if (!resp.ok) {
    const errText = await resp.text();
    throw new Error(`Resend email error (${resp.status}): ${errText}`);
  }
}
