export interface BrandConfig {
  name: string;
  tagline: string;
  supportEmail: string;
  priceCents: number;
  currency: string;
  freePreviewDailyLimit: number;
  maxPreviewChars: number;
  maxStyleRefs: number;
  canvasSize: number;
  baselineY: number;
  capLineY: number;
  xHeightY: number;
  descenderY: number;
}

export const DEFAULT_BRAND_CONFIG: BrandConfig = {
  name: "GlyphCraft",
  tagline: "Turn 1–3 letters into a complete English .ttf font.",
  supportEmail: "support@glyphcraft.app",
  priceCents: 900,
  currency: "USD",
  freePreviewDailyLimit: 3,
  maxPreviewChars: 16,
  maxStyleRefs: 3,
  canvasSize: 128,
  baselineY: 96,
  capLineY: 40,
  xHeightY: 56,
  descenderY: 118,
};

export function resolveBrandName(override?: string | null): string {
  const cleaned = override?.trim();
  return cleaned && cleaned.length > 0 ? cleaned : DEFAULT_BRAND_CONFIG.name;
}
