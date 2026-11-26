import { useCallback, useEffect, useState } from "react";
import { Turnstile } from "@marsidev/react-turnstile";
import {
  ArrowRight,
  CheckCircle2,
  Image as ImageIcon,
  LayoutGrid,
  Loader2,
  PenTool,
  Sparkles,
  Wand2,
} from "lucide-react";
import {
  DEFAULT_PREVIEW_TEXT,
  sanitizeFamilyName,
  validatePreviewText,
  type CurveMode,
} from "@glyphcraft/shared";
import { BRAND_CONFIG } from "../config/brand";
import { STUDIO_PRESETS, type StudioPreset } from "../config/samples";
import { useAuth } from "../hooks/useAuth";
import { useDraftStorage } from "../hooks/useDraftStorage";
import {
  ApiError,
  createCheckoutSession,
  generateFreePreview,
  saveDraftToR2,
} from "../lib/api";
import { GlyphCanvasPad } from "./GlyphCanvasPad";
import { ImageAlignUploader } from "./ImageAlignUploader";

interface StudioWorkbenchProps {
  onRequireAuth: () => void;
  requestedPreset?: { slug: string; seq: number } | null;
}

type InputTab = "preset" | "draw" | "upload";

const TURNSTILE_SITE_KEY =
  (import.meta.env.VITE_TURNSTILE_SITE_KEY as string | undefined) ||
  "1x00000000000000000000AA";

async function urlTo128PngDataUrl(url: string): Promise<string> {
  const resp = await fetch(url);
  if (!resp.ok) {
    throw new Error(`Failed to load starter style image (${resp.status}).`);
  }
  const blob = await resp.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to load preset image."));
    reader.readAsDataURL(blob);
  });
}

export function StudioWorkbench({
  onRequireAuth,
  requestedPreset,
}: StudioWorkbenchProps) {
  const { user, session } = useAuth();
  const { draft, setSlotImage, setAllSlots, updateFields } = useDraftStorage();

  const [tab, setTab] = useState<InputTab>("preset");
  const [loadingPresetSlug, setLoadingPresetSlug] = useState<string | null>(
    null
  );
  const [turnstileToken, setTurnstileToken] = useState<string>("");
  const [turnstileKey, setTurnstileKey] = useState(0);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
  const [previewStale, setPreviewStale] = useState(false);
  const [remainingQuota, setRemainingQuota] = useState<number>(
    BRAND_CONFIG.freePreviewDailyLimit
  );
  const [previewError, setPreviewError] = useState<string | null>(null);

  const [checkoutLoading, setCheckoutLoading] = useState(false);
  const [checkoutError, setCheckoutError] = useState<string | null>(null);

  const activeImages = draft.slots.filter((s): s is string => Boolean(s));
  const textCheck = validatePreviewText(
    draft.previewText,
    BRAND_CONFIG.maxPreviewChars
  );
  const sanitizedName = sanitizeFamilyName(draft.familyName);
  const showSanitizedHint =
    draft.familyName.trim().length > 0 &&
    draft.familyName.trim() !== sanitizedName;

  const handleSlotChange = useCallback(
    (index: 0 | 1 | 2, dataUrl: string | null) => {
      setPreviewError(null);
      if (previewImage) {
        setPreviewStale(true);
      }
      setSlotImage(index, dataUrl);
    },
    [setSlotImage, previewImage]
  );

  const handleApplyPreset = useCallback(
    async (preset: StudioPreset) => {
      setPreviewError(null);
      setCheckoutError(null);
      setLoadingPresetSlug(preset.slug);
      try {
        const loaded = await Promise.all(
          preset.glyphs.map((path) => urlTo128PngDataUrl(path))
        );
        setAllSlots([loaded[0], loaded[1], loaded[2]], {
          curveMode: preset.recommendedCurveMode,
          familyName: preset.name.replace(/\s+/g, ""),
          activePresetSlug: preset.slug,
        });
        if (previewImage) {
          setPreviewStale(true);
        }
      } catch (err) {
        setPreviewError(
          err instanceof Error
            ? err.message
            : "Could not load starter style letters."
        );
      } finally {
        setLoadingPresetSlug(null);
      }
    },
    [setAllSlots, previewImage]
  );

  useEffect(() => {
    if (!requestedPreset) return;
    const match = STUDIO_PRESETS.find((p) => p.slug === requestedPreset.slug);
    if (!match) return;
    setTab("preset");
    void handleApplyPreset(match);
  }, [requestedPreset]); // eslint-disable-line react-hooks/exhaustive-deps

  const handleGeneratePreview = async () => {
    setPreviewError(null);
    if (activeImages.length === 0) {
      setPreviewError(
        "Pick a style, handwrite, or upload at least 1 sample letter first."
      );
      return;
    }
    if (!textCheck.valid) {
      setPreviewError(textCheck.error ?? "Invalid preview phrase.");
      return;
    }
    if (remainingQuota <= 0) {
      setPreviewError(
        "Daily free preview limit reached. Try again tomorrow or export your full font."
      );
      return;
    }

    setPreviewLoading(true);
    try {
      const resp = await generateFreePreview({
        images: activeImages,
        text: textCheck.cleaned,
        turnstileToken: turnstileToken || "dev-token",
      });
      setPreviewImage(resp.previewDataUrl);
      setPreviewStale(false);
      setRemainingQuota(resp.remainingQuota);
    } catch (err) {
      if (err instanceof ApiError && err.remainingQuota !== undefined) {
        setRemainingQuota(err.remainingQuota);
      }
      setPreviewError(
        err instanceof Error ? err.message : "Preview generation failed."
      );
    } finally {
      setPreviewLoading(false);
      setTurnstileToken("");
      setTurnstileKey((k) => k + 1);
    }
  };

  const handleUnlockFullFont = async () => {
    setCheckoutError(null);
    if (activeImages.length === 0) {
      setCheckoutError(
        "Pick a style, handwrite, or upload at least 1 sample letter first."
      );
      return;
    }

    setCheckoutLoading(true);
    try {
      // Always persist the current draft in R2 first so nothing is lost across OAuth/checkout
      let draftId = draft.r2DraftId;
      if (!draftId) {
        const saved = await saveDraftToR2({ images: activeImages });
        draftId = saved.draftId;
        updateFields({ r2DraftId: draftId });
      }

      if (!user || !session?.access_token) {
        updateFields({ r2DraftId: draftId, pendingCheckout: true });
        setCheckoutLoading(false);
        onRequireAuth();
        return;
      }

      const checkout = await createCheckoutSession(
        {
          draftId,
          familyName: sanitizedName,
          curveMode: draft.curveMode,
        },
        session.access_token
      );
      updateFields({ pendingCheckout: false });
      window.location.href = checkout.checkoutUrl;
    } catch (err) {
      setCheckoutError(
        err instanceof Error ? err.message : "Could not start checkout."
      );
      setCheckoutLoading(false);
    }
  };

  return (
    <section
      id="studio"
      className="border-b border-black/10 bg-[#fafaf8] py-16 sm:py-24"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
            Interactive Studio
          </p>
          <h2 className="mt-2 font-display text-4xl tracking-tight text-[#0a0a0b] sm:text-5xl">
            Test your own style in seconds.
          </h2>
          <p className="mt-2.5 max-w-2xl text-sm leading-relaxed text-black/60">
            Pick a starter style, handwrite 1–3 letters, or upload a style
            sample to generate a free preview.
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* LEFT WORKBENCH: Input source + Preview Sentence + Turnstile */}
          <div className="space-y-5 rounded-2xl border border-black/15 bg-white p-5 shadow-sm lg:col-span-7 sm:p-6">
            {/* Ordered 3-tab Switcher: 1. Choose Style -> 2. Draw Letters -> 3. Upload Image */}
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="inline-flex flex-wrap rounded-xl border border-black/10 bg-[#fafaf8] p-1">
                <button
                  type="button"
                  onClick={() => setTab("preset")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                    tab === "preset"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <LayoutGrid className="h-3.5 w-3.5" />
                  Choose Style
                </button>
                <button
                  type="button"
                  onClick={() => setTab("draw")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                    tab === "draw"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <PenTool className="h-3.5 w-3.5" />
                  Handwrite
                </button>
                <button
                  type="button"
                  onClick={() => setTab("upload")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3.5 py-1.5 text-xs font-semibold transition ${
                    tab === "upload"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <ImageIcon className="h-3.5 w-3.5" />
                  Upload Style
                </button>
              </div>
            </div>

            {/* Active input mode */}
            {tab === "preset" && (
              <div className="space-y-3">
                <p className="text-xs text-black/60">
                  Pick a curated style to load 3 sample letters instantly, or
                  switch to Handwrite / Upload Style to use your own.
                </p>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {STUDIO_PRESETS.map((preset) => {
                    const isSelected = draft.activePresetSlug === preset.slug;
                    const isLoadingThis = loadingPresetSlug === preset.slug;
                    return (
                      <button
                        key={preset.slug}
                        type="button"
                        disabled={Boolean(loadingPresetSlug)}
                        onClick={() => void handleApplyPreset(preset)}
                        className={`flex flex-col justify-between rounded-xl border p-3.5 text-left transition disabled:cursor-wait ${
                          isSelected
                            ? "border-[#0a0a0b] bg-[#fafaf8] ring-1 ring-[#0a0a0b]"
                            : "border-black/15 bg-white hover:border-black/35 hover:bg-[#fafaf8]/60"
                        }`}
                      >
                        <div className="flex w-full items-start justify-between gap-2">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-semibold text-[#0a0a0b]">
                                {preset.name}
                              </span>
                              <span className="rounded-md bg-black/5 px-1.5 py-0.5 font-mono text-[10px] font-medium text-black/65">
                                {preset.tag}
                              </span>
                            </div>
                            <p className="mt-1 text-xs text-black/55">
                              {preset.description}
                            </p>
                          </div>

                          {isLoadingThis ? (
                            <Loader2 className="h-4 w-4 shrink-0 animate-spin text-[#d94625]" />
                          ) : (
                            isSelected && (
                              <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                            )
                          )}
                        </div>

                        <div className="mt-3 flex items-center gap-2">
                          {preset.glyphs.map((glyphSrc, idx) => (
                            <div
                              key={glyphSrc}
                              className="flex h-12 w-12 items-center justify-center overflow-hidden rounded-lg border border-black/10 bg-white p-1"
                            >
                              <img
                                src={glyphSrc}
                                alt={`${preset.name} sample ${idx + 1}`}
                                className="h-full w-full object-contain"
                              />
                            </div>
                          ))}
                        </div>
                      </button>
                    );
                  })}
                </div>

                {activeImages.length > 0 && (
                  <div className="flex flex-wrap items-center justify-between gap-2 rounded-xl border border-black/10 bg-[#fafaf8] px-3.5 py-2.5 text-xs">
                    <span className="text-black/65">
                      Want to tweak these letters?
                    </span>
                    <button
                      type="button"
                      onClick={() => setTab("draw")}
                      className="inline-flex items-center gap-1 font-semibold text-[#d94625] transition hover:underline"
                    >
                      Edit in Handwrite
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  </div>
                )}
              </div>
            )}

            {tab === "draw" && (
              <GlyphCanvasPad
                slots={draft.slots}
                onChangeSlot={handleSlotChange}
              />
            )}

            {tab === "upload" && (
              <ImageAlignUploader
                slots={draft.slots}
                onChangeSlot={handleSlotChange}
              />
            )}

            {/* Preview sentence input */}
            <div className="rounded-xl border border-black/10 bg-[#fafaf8] p-4">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="preview-text-input"
                  className="text-xs font-semibold text-[#0a0a0b]"
                >
                  Preview Text (max {BRAND_CONFIG.maxPreviewChars} English
                  characters)
                </label>
                <span
                  className={`font-mono text-[11px] ${
                    textCheck.valid ? "text-black/50" : "text-red-600"
                  }`}
                >
                  {draft.previewText.length}/{BRAND_CONFIG.maxPreviewChars}
                </span>
              </div>
              <input
                id="preview-text-input"
                type="text"
                maxLength={BRAND_CONFIG.maxPreviewChars}
                value={draft.previewText}
                onChange={(e) => {
                  setPreviewError(null);
                  if (previewImage) {
                    setPreviewStale(true);
                  }
                  updateFields({ previewText: e.target.value });
                }}
                placeholder={DEFAULT_PREVIEW_TEXT}
                className="mt-2 w-full rounded-lg border border-black/15 bg-white px-3.5 py-2.5 font-mono text-sm text-[#0a0a0b] outline-none focus:border-[#0a0a0b]"
              />
              {!textCheck.valid && (
                <p className="mt-1.5 text-xs text-red-600">{textCheck.error}</p>
              )}
            </div>

            {/* Turnstile + Generate Free Preview */}
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="min-h-[65px]">
                <Turnstile
                  key={turnstileKey}
                  siteKey={TURNSTILE_SITE_KEY}
                  onSuccess={(token) => setTurnstileToken(token)}
                  onExpire={() => setTurnstileToken("")}
                  options={{ theme: "light", size: "normal" }}
                />
              </div>

              <div className="flex flex-col items-start sm:items-end">
                <button
                  type="button"
                  disabled={
                    previewLoading ||
                    activeImages.length === 0 ||
                    !textCheck.valid ||
                    remainingQuota <= 0
                  }
                  onClick={() => void handleGeneratePreview()}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#0a0a0b] px-5 py-3 text-sm font-semibold text-white transition hover:bg-black/85 disabled:cursor-not-allowed disabled:opacity-45"
                >
                  {previewLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Generating preview...
                    </>
                  ) : remainingQuota <= 0 ? (
                    "Daily Free Limit Reached"
                  ) : (
                    <>
                      <Wand2 className="h-4 w-4 text-[#d94625]" />
                      Generate Free Preview
                    </>
                  )}
                </button>
                <span className="mt-1.5 font-mono text-[11px] text-black/45">
                  {activeImages.length === 0
                    ? "Pick a style or provide at least 1 letter first"
                    : `${remainingQuota}/${BRAND_CONFIG.freePreviewDailyLimit} free previews left today`}
                </span>
              </div>
            </div>

            {previewError && (
              <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">
                {previewError}
              </div>
            )}
          </div>

          {/* RIGHT WORKBENCH: Preview output + High-conversion Checkout Card */}
          <div className="flex flex-col gap-5 lg:col-span-5">
            {/* Live Preview Output */}
            <div className="rounded-2xl border border-black/15 bg-white p-5 shadow-sm">
              <div className="flex items-center justify-between gap-2">
                <span className="font-mono text-xs font-semibold uppercase tracking-wider text-black/60">
                  Instant Preview
                </span>
                {previewStale && previewImage && (
                  <span className="rounded-full border border-amber-300 bg-amber-50 px-2.5 py-0.5 font-mono text-[10px] font-medium text-amber-800">
                    Outdated — click Generate to refresh
                  </span>
                )}
              </div>

              {/* Active reference thumbnail strip (shown only when letters are loaded) */}
              {activeImages.length > 0 && (
                <div className="mt-3 flex items-center gap-2">
                  {activeImages.map((slotUrl, idx) => (
                    <div
                      key={idx}
                      className="flex h-9 w-9 items-center justify-center overflow-hidden rounded-lg border border-black/10 bg-white p-0.5"
                    >
                      <img
                        src={slotUrl}
                        alt={`Style reference ${idx + 1}`}
                        className="h-full w-full object-contain"
                      />
                    </div>
                  ))}
                  <span className="ml-1 text-[11px] text-black/45">
                    Style reference
                  </span>
                </div>
              )}

              <div className="relative mt-3 flex min-h-[140px] items-center justify-center overflow-hidden rounded-xl border border-dashed border-black/15 bg-[#fafaf8] p-4">
                {previewLoading ? (
                  <div className="flex flex-col items-center gap-2 text-xs text-black/55">
                    <Loader2 className="h-6 w-6 animate-spin text-[#d94625]" />
                    Generating preview...
                  </div>
                ) : previewImage ? (
                  <img
                    src={previewImage}
                    alt="Generated font preview sentence"
                    className={`max-h-36 w-auto object-contain transition-opacity ${
                      previewStale ? "opacity-65" : "opacity-100"
                    }`}
                  />
                ) : (
                  <p className="max-w-xs text-center text-xs leading-relaxed text-black/45">
                    Your generated phrase will appear here before you export the
                    full English{" "}
                    <code className="font-mono">.ttf</code> font.
                  </p>
                )}
              </div>

              <p className="mt-2.5 text-[11px] leading-relaxed text-black/50">
                Note: Each generation's results will differ slightly every time.
              </p>
            </div>

            {/* Conversion Card */}
            <div className="flex-1 rounded-2xl border border-[#0a0a0b] bg-[#0a0a0b] p-6 text-white shadow-xl">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="inline-flex items-center gap-1 rounded-full bg-[#d94625] px-2.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-white">
                    <Sparkles className="h-3 w-3" /> Full .TTF Export
                  </span>
                  <h3 className="mt-2.5 font-display text-3xl tracking-tight">
                    Generate Full English Font
                  </h3>
                </div>
                <div className="text-right">
                  <div className="font-display text-4xl">
                    {BRAND_CONFIG.priceDisplay}
                  </div>
                  <div className="font-mono text-[11px] text-white/55">
                    one-time
                  </div>
                </div>
              </div>

              <ul className="mt-4 space-y-2 text-xs text-white/80">
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                  Complete English character set (A–Z, a–z, 0–9 &amp;
                  punctuation)
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                  Installable TrueType <code className="font-mono">.ttf</code>{" "}
                  with commercial usage rights
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                  Free unlimited weight, spacing &amp; curve adjustments
                </li>
              </ul>

              <div className="mt-5 space-y-3 border-t border-white/15 pt-4">
                <label className="block text-xs">
                  <span className="font-medium text-white/75">
                    Font Family Name
                  </span>
                  <input
                    type="text"
                    maxLength={40}
                    value={draft.familyName}
                    onChange={(e) =>
                      updateFields({ familyName: e.target.value })
                    }
                    placeholder="MyCustomFont"
                    className="mt-1.5 w-full rounded-lg border border-white/15 bg-white/10 px-3 py-2 text-sm text-white placeholder:text-white/35 outline-none focus:border-white/50"
                  />
                  {showSanitizedHint && (
                    <span className="mt-1 block text-[11px] text-white/55">
                      Will export as{" "}
                      <code className="font-mono text-white/80">
                        {sanitizedName}.ttf
                      </code>{" "}
                      (English letters, numbers, spaces, - and _ only)
                    </span>
                  )}
                </label>

                <div className="text-xs">
                  <span className="font-medium text-white/75">
                    Outline Style
                  </span>
                  <div className="mt-1.5 grid grid-cols-2 gap-2">
                    {(
                      [
                        {
                          mode: "spline",
                          label: "Smooth Curves",
                          sub: "Handwriting, Serif & Sans",
                        },
                        {
                          mode: "polygon",
                          label: "Crisp Edges",
                          sub: "Pixel & Geometric",
                        },
                      ] as Array<{ mode: CurveMode; label: string; sub: string }>
                    ).map((opt) => (
                      <button
                        key={opt.mode}
                        type="button"
                        onClick={() => updateFields({ curveMode: opt.mode })}
                        className={`rounded-lg border p-2.5 text-left transition ${
                          draft.curveMode === opt.mode
                            ? "border-[#d94625] bg-[#d94625]/20 text-white"
                            : "border-white/15 bg-white/5 text-white/70 hover:border-white/30"
                        }`}
                      >
                        <div className="font-semibold">{opt.label}</div>
                        <div className="text-[10px] text-white/55">
                          {opt.sub}
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {checkoutError && (
                <div className="mt-3 rounded-lg border border-red-400/30 bg-red-500/20 px-3 py-2 text-xs text-red-100">
                  {checkoutError}
                </div>
              )}

              <button
                type="button"
                disabled={checkoutLoading || activeImages.length === 0}
                onClick={() => void handleUnlockFullFont()}
                className="mt-5 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[#d94625] px-5 py-3.5 text-sm font-semibold text-white transition hover:bg-[#c53c1d] disabled:cursor-not-allowed disabled:opacity-45"
              >
                {checkoutLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Preparing Checkout...
                  </>
                ) : (
                  <>
                    Export Full Font · {BRAND_CONFIG.priceDisplay}
                    <ArrowRight className="h-4 w-4" />
                  </>
                )}
              </button>

              {activeImages.length === 0 && (
                <p className="mt-2 text-center text-[11px] text-white/60">
                  Pick a style, handwrite, or upload at least 1 letter to enable
                  export.
                </p>
              )}

              <p className="mt-2.5 text-center font-mono text-[11px] text-white/50">
                {user
                  ? `Signed in as ${user.email} · Draft saved automatically`
                  : "Your draft is saved automatically."}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
