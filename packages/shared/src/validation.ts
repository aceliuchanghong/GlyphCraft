import { DEFAULT_BRAND_CONFIG } from "./brand";
import type { CurveMode, TtfParams } from "./types";

/** Default preview phrase used when the input is left blank */
export const DEFAULT_PREVIEW_TEXT = "Hello World!";

/** Printable ASCII 0x21..0x7E plus space 0x20 */
export const PRINTABLE_ASCII_REGEX = /^[\x20-\x7E]+$/;

/** Complete 94 printable ASCII characters (0x21 '!' through 0x7E '~') */
export const ASCII_94: string = Array.from({ length: 94 }, (_, i) =>
  String.fromCharCode(33 + i)
).join("");

export interface PreviewTextValidation {
  valid: boolean;
  cleaned: string;
  uniqueCount: number;
  error?: string;
}

export function validatePreviewText(
  rawText: string,
  maxChars: number = DEFAULT_BRAND_CONFIG.maxPreviewChars
): PreviewTextValidation {
  const cleaned = rawText.trim() || DEFAULT_PREVIEW_TEXT;
  if (cleaned.length > maxChars) {
    return {
      valid: false,
      cleaned,
      uniqueCount: 0,
      error: `Preview text must be at most ${maxChars} ASCII characters.`,
    };
  }
  if (!PRINTABLE_ASCII_REGEX.test(cleaned)) {
    return {
      valid: false,
      cleaned,
      uniqueCount: 0,
      error: "Only standard English printable ASCII characters (A-Z, a-z, 0-9, punctuation) are supported.",
    };
  }

  const unique = new Set<string>();
  for (const ch of cleaned) {
    const cp = ch.charCodeAt(0);
    if (cp >= 33 && cp <= 126) {
      unique.add(ch);
    }
  }
  if (unique.size === 0) {
    return {
      valid: false,
      cleaned,
      uniqueCount: 0,
      error: "Please include at least one non-space character.",
    };
  }

  return {
    valid: true,
    cleaned,
    uniqueCount: unique.size,
  };
}

export function sanitizeFamilyName(name?: string | null): string {
  const raw = (name ?? "")
    .replace(/[^A-Za-z0-9 _-]/g, "")
    .replace(/\s+/g, " ")
    .trim()
    .slice(0, 40);
  return raw || "MyCustomFont";
}

export function clampTtfParams(input?: Partial<{
  threshold: number;
  sidebearing: number;
  curveMode: CurveMode;
  curve_mode: CurveMode;
}>): TtfParams {
  const rawThreshold = Number(input?.threshold ?? 160);
  const rawSidebearing = Number(input?.sidebearing ?? 55);
  const rawMode = input?.curveMode ?? input?.curve_mode ?? "spline";

  const threshold = Number.isFinite(rawThreshold)
    ? Math.max(80, Math.min(235, Math.round(rawThreshold)))
    : 160;
  const sidebearing = Number.isFinite(rawSidebearing)
    ? Math.max(0, Math.min(200, Math.round(rawSidebearing)))
    : 55;
  const curve_mode: CurveMode = rawMode === "polygon" ? "polygon" : "spline";

  return { threshold, sidebearing, curve_mode };
}
