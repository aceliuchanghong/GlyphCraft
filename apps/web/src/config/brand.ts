import { DEFAULT_BRAND_CONFIG, resolveBrandName } from "@glyphcraft/shared";

export const BRAND_NAME = resolveBrandName(import.meta.env.VITE_BRAND_NAME);
export const BRAND_CONFIG = {
  ...DEFAULT_BRAND_CONFIG,
  name: BRAND_NAME,
  priceDisplay: import.meta.env.VITE_PRICE_DISPLAY || "$9",
};
