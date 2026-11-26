import { useState } from "react";
import { Turnstile } from "@marsidev/react-turnstile";
import {
  CheckCircle2,
  Image as ImageIcon,
  Loader2,
  Lock,
  PenTool,
  Sparkles,
  Type,
  Wand2,
} from "lucide-react";
import {
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
import { TtfRefUploader } from "./TtfRefUploader";

interface StudioWorkbenchProps {
  onRequireAuth: () => void;
}

type InputTab = "draw" | "upload" | "ttf";

const TURNSTILE_SITE_KEY =
  (import.meta.env.VITE_TURNSTILE_SITE_KEY as string | undefined) ||
  "1x00000000000000000000AA";

async function urlTo128PngDataUrl(url: string): Promise<string> {
  const resp = await fetch(url);
  const blob = await resp.blob();
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Failed to load preset image."));
    reader.readAsDataURL(blob);
  });
}

export function StudioWorkbench({ onRequireAuth }: StudioWorkbenchProps) {
  const { user, session } = useAuth();
  const { draft, setSlotImage, setAllSlots, updateFields } = useDraftStorage();

  const [tab, setTab] = useState<InputTab>("draw");
  const [turnstileToken, setTurnstileToken] = useState<string>("");
  const [turnstileKey, setTurnstileKey] = useState(0);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewImage, setPreviewImage] = useState<string | null>(null);
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

  const handleApplyPreset = async (preset: StudioPreset) => {
    setPreviewError(null);
    const loaded = await Promise.all(
      preset.glyphs.map((path) => urlTo128PngDataUrl(path))
    );
    setAllSlots([loaded[0], loaded[1], loaded[2]], {
      curveMode: preset.recommendedCurveMode,
      familyName: preset.name.replace(/\s+/g, ""),
    });
  };

  const handleGeneratePreview = async () => {
    setPreviewError(null);
    if (activeImages.length === 0) {
      setPreviewError("Draw or load at least 1 reference character first.");
      return;
    }
    if (!textCheck.valid) {
      setPreviewError(textCheck.error ?? "Invalid preview phrase.");
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
      setCheckoutError("Draw or upload at least 1 reference character first.");
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
          familyName: sanitizeFamilyName(draft.familyName),
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
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
              Interactive Studio
            </p>
            <h2 className="mt-2 font-display text-4xl tracking-tight text-[#0a0a0b] sm:text-5xl">
              Test your own style in seconds.
            </h2>
          </div>
          <p className="max-w-md text-sm text-black/60">
            Draw 1–3 letters, align a photo, or upload a reference{" "}
            <code className="font-mono">.ttf</code>. Instant preview is free (
            {remainingQuota}/{BRAND_CONFIG.freePreviewDailyLimit} left today).
          </p>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
          {/* LEFT WORKBENCH: Input slots + Preview Sentence + Turnstile */}
          <div className="space-y-5 rounded-2xl border border-black/15 bg-white p-5 shadow-sm lg:col-span-7 sm:p-6">
            {/* Mode Switcher + Presets */}
            <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="inline-flex rounded-xl border border-black/10 bg-[#fafaf8] p-1">
                <button
                  type="button"
                  onClick={() => setTab("draw")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    tab === "draw"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <PenTool className="h-3.5 w-3.5" />
                  Draw Pad
                </button>
                <button
                  type="button"
                  onClick={() => setTab("upload")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    tab === "upload"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <ImageIcon className="h-3.5 w-3.5" />
                  Align Image
                </button>
                <button
                  type="button"
                  onClick={() => setTab("ttf")}
                  className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                    tab === "ttf"
                      ? "bg-[#0a0a0b] text-white"
                      : "text-black/65 hover:text-[#0a0a0b]"
                  }`}
                >
                  <Type className="h-3.5 w-3.5" />
                  Ref .TTF
                </button>
              </div>

              {/* Quick presets */}
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="font-mono text-[11px] text-black/45">
                  Presets:
                </span>
                {STUDIO_PRESETS.map((preset) => (
                  <button
                    key={preset.slug}
                    type="button"
                    onClick={() => void handleApplyPreset(preset)}
                    className="rounded-md border border-black/10 bg-[#fafaf8] px-2 py-1 text-[11px] font-medium text-black/75 transition hover:border-black/30 hover:bg-white"
                  >
                    {preset.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Active input mode */}
            {tab === "draw" && (
              <GlyphCanvasPad
                slots={draft.slots}
                onChangeSlot={setSlotImage}
              />
            )}

            {tab === "upload" && (
              <ImageAlignUploader
                slots={draft.slots}
                onChangeSlot={setSlotImage}
              />
            )}

            {tab === "ttf" && (
              <TtfRefUploader
                onExtractSlots={(slots) => {
                  setAllSlots(slots);
                  setTab("draw");
                }}
              />
            )}

            {/* Preview sentence input */}
            <div className="rounded-xl border border-black/10 bg-[#fafaf8] p-4">
              <div className="flex items-center justify-between">
                <label
                  htmlFor="preview-text-input"
                  className="text-xs font-semibold text-[#0a0a0b]"
                >
                  Preview Phrase (max {BRAND_CONFIG.maxPreviewChars} ASCII
                  characters)
                </label>
                <span
                  className={`font-mono text-[11px] ${
                    textCheck.valid ? "text-black/50" : "text-red-600"
                  }`}
                >
                  {draft.previewText.length}/{BRAND_CONFIG.maxPreviewChars}
                  {textCheck.valid
                    ? ` · ${textCheck.uniqueCount} unique glyphs`
                    : ""}
                </span>
              </div>
              <input
                id="preview-text-input"
                type="text"
                maxLength={BRAND_CONFIG.maxPreviewChars}
                value={draft.previewText}
                onChange={(e) => updateFields({ previewText: e.target.value })}
                placeholder="Hello World!"
                className="mt-2 w-full rounded-lg border border-black/15 bg-white px-3.5 py-2.5 font-mono text-sm text-[#0a0a0b] outline-none focus:border-[#0a0a0b]"
              />
              {!textCheck.valid && (
                <p className="mt-1.5 text-xs text-red-600">{textCheck.error}</p>
              )}
            </div>

            {/* Turnstile + Generate Free Preview */}
            <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <div className="min-h-[65px]">
                <Turnstile
                  key={turnstileKey}
                  siteKey={TURNSTILE_SITE_KEY}
                  onSuccess={(token) => setTurnstileToken(token)}
                  onExpire={() => setTurnstileToken("")}
                  options={{ theme: "light", size: "normal" }}
                />
              </div>

              <button
                type="button"
                disabled={previewLoading || activeImages.length === 0 || !textCheck.valid}
                onClick={() => void handleGeneratePreview()}
                className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#0a0a0b] px-5 py-3 text-sm font-semibold text-white transition hover:bg-black/85 disabled:cursor-not-allowed disabled:opacity-45"
              >
                {previewLoading ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Synthesizing (4-step fast)...
                  </>
                ) : (
                  <>
                    <Wand2 className="h-4 w-4 text-[#d94625]" />
                    Generate Free Preview ({remainingQuota}/
                    {BRAND_CONFIG.freePreviewDailyLimit})
                  </>
                )}
              </button>
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
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-semibold uppercase tracking-wider text-black/60">
                  4-Step Instant Preview
                </span>
                <span className="font-mono text-[11px] text-black/45">
                  {activeImages.length}/3 reference slot
                  {activeImages.length === 1 ? "" : "s"} active
                </span>
              </div>

              <div className="mt-4 flex min-h-[140px] items-center justify-center overflow-hidden rounded-xl border border-dashed border-black/15 bg-[#fafaf8] p-4">
                {previewLoading ? (
                  <div className="flex flex-col items-center gap-2 text-xs text-black/55">
                    <Loader2 className="h-6 w-6 animate-spin text-[#d94625]" />
                    Running fast 4-step flow inference on GPU...
                  </div>
                ) : previewImage ? (
                  <img
                    src={previewImage}
                    alt="Generated font preview sentence"
                    className="max-h-36 w-auto object-contain"
                  />
                ) : (
                  <p className="max-w-xs text-center text-xs leading-relaxed text-black/45">
                    Your synthesized phrase will render here on a baseline guide
                    before you unlock the full 94-ASCII{" "}
                    <code className="font-mono">.ttf</code>.
                  </p>
                )}
              </div>
            </div>

            {/* Conversion Card */}
            <div className="flex-1 rounded-2xl border border-[#0a0a0b] bg-[#0a0a0b] p-6 text-white shadow-xl">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <span className="inline-flex items-center gap-1 rounded-full bg-[#d94625] px-2.5 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-wider text-white">
                    <Sparkles className="h-3 w-3" /> HD 10-Step Mode
                  </span>
                  <h3 className="mt-2.5 font-display text-3xl tracking-tight">
                    Unlock Full 94-ASCII Font
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
                  Complete 94 printable English ASCII glyphs (A–Z, a–z, 0–9,
                  punctuation)
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                  Installable TrueType <code className="font-mono">.ttf</code>{" "}
                  with commercial usage rights
                </li>
                <li className="flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4 shrink-0 text-[#d94625]" />
                  Free unlimited weight, spacing &amp; spline/polygon repacking
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
                </label>

                <div className="text-xs">
                  <span className="font-medium text-white/75">
                    Vectorization Mode
                  </span>
                  <div className="mt-1.5 grid grid-cols-2 gap-2">
                    {(
                      [
                        {
                          mode: "spline",
                          label: "Smooth Spline",
                          sub: "Handwriting / Serif / Sans",
                        },
                        {
                          mode: "polygon",
                          label: "Crisp Polygon",
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
                    <Lock className="h-4 w-4" />
                    Generate Full 94-ASCII Font (.ttf)
                  </>
                )}
              </button>
              <p className="mt-2.5 text-center font-mono text-[11px] text-white/50">
                {user
                  ? `Signed in as ${user.email} · Draft saved automatically`
                  : "Your drawn glyphs are saved automatically before Google sign-in"}
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
