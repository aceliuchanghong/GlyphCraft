import { Check, HelpCircle, Layers, Sliders, Sparkles } from "lucide-react";
import { BRAND_CONFIG, BRAND_NAME } from "../config/brand";
import { BeforeAfterSlider } from "../components/BeforeAfterSlider";
import { HeroPlayground } from "../components/HeroPlayground";
import { StudioWorkbench } from "../components/StudioWorkbench";

interface HomePageProps {
  onOpenAuth: () => void;
}

const FAQS = [
  {
    q: "Which characters are included in the generated .ttf font?",
    a: "Every generated font covers the complete 94 printable English ASCII character set (A–Z, a–z, 0–9, and 32 standard punctuation/symbol glyphs from 0x21 '!' to 0x7E '~', plus the space glyph). Non-English scripts (CJK, Cyrillic, Arabic, emoji) are not supported.",
  },
  {
    q: "How many reference letters do I need to draw or upload?",
    a: "You only need 1 to 3 reference glyphs. We recommend drawing H (uppercase cap height), g (lowercase curve + descender), and p (or another distinctive letter) aligned to the baseline guides.",
  },
  {
    q: "Can I adjust weight or spacing after generating the font?",
    a: "Yes. After your 94-ASCII font finishes generating, your result page lets you tweak stroke thickness (threshold), sidebearing spacing, font family name, and spline vs. polygon curve mode for free—repacking a fresh .ttf in seconds.",
  },
  {
    q: "Can I use the downloaded .ttf font commercially?",
    a: "Yes. You receive full commercial rights to use, embed, and distribute the .ttf files generated from your original drawings or licensed style references.",
  },
];

export function HomePage({ onOpenAuth }: HomePageProps) {
  return (
    <>
      <HeroPlayground />
      <BeforeAfterSlider />
      <StudioWorkbench onRequireAuth={onOpenAuth} />

      {/* Workflow + Pricing Section */}
      <section id="pricing" className="border-b border-black/10 bg-white py-16 sm:py-24">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="grid grid-cols-1 gap-10 lg:grid-cols-12 lg:items-center">
            <div className="lg:col-span-7">
              <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
                Foundry Architecture
              </p>
              <h2 className="mt-2 font-display text-4xl tracking-tight text-[#0a0a0b] sm:text-5xl">
                Built for real typography, not blurry raster grids.
              </h2>

              <div className="mt-8 grid grid-cols-1 gap-5 sm:grid-cols-3">
                <div className="rounded-2xl border border-black/10 bg-[#fafaf8] p-5">
                  <Layers className="h-5 w-5 text-[#d94625]" />
                  <h3 className="mt-3 text-sm font-semibold text-[#0a0a0b]">
                    Fixed Baseline Math
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-black/60">
                    Every glyph is locked to the exact{" "}
                    <code className="font-mono">y=96/128</code> baseline and
                    mapped cleanly to <code className="font-mono">y=0</code> in
                    1000-UPM TrueType space.
                  </p>
                </div>

                <div className="rounded-2xl border border-black/10 bg-[#fafaf8] p-5">
                  <Sparkles className="h-5 w-5 text-[#d94625]" />
                  <h3 className="mt-3 text-sm font-semibold text-[#0a0a0b]">
                    4-Step Try, 10-Step HD
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-black/60">
                    Test your handwriting in seconds with 4-step fast preview,
                    then generate all 94 glyphs in 10-step Dual-CFG HD quality.
                  </p>
                </div>

                <div className="rounded-2xl border border-black/10 bg-[#fafaf8] p-5">
                  <Sliders className="h-5 w-5 text-[#d94625]" />
                  <h3 className="mt-3 text-sm font-semibold text-[#0a0a0b]">
                    Non-Destructive Tuning
                  </h3>
                  <p className="mt-1.5 text-xs leading-relaxed text-black/60">
                    Re-vectorize your 94 glyphs anytime on CPU to refine stroke
                    weight, sidebearings, and spline/polygon contours.
                  </p>
                </div>
              </div>
            </div>

            <div className="lg:col-span-5">
              <div className="rounded-2xl border border-black/15 bg-[#fafaf8] p-7">
                <div className="flex items-baseline justify-between">
                  <div>
                    <p className="font-mono text-xs uppercase tracking-wider text-[#d94625]">
                      Pay Per Font
                    </p>
                    <h3 className="mt-1 font-display text-3xl text-[#0a0a0b]">
                      Full 94-ASCII Pack
                    </h3>
                  </div>
                  <div className="text-right">
                    <span className="font-display text-5xl text-[#0a0a0b]">
                      {BRAND_CONFIG.priceDisplay}
                    </span>
                    <span className="ml-1 font-mono text-xs text-black/55">
                      / font
                    </span>
                  </div>
                </div>

                <ul className="mt-6 space-y-2.5 text-sm text-black/75">
                  {[
                    "No subscription required—pay only when you love your preview",
                    "All 94 English ASCII characters + calibrated space glyph",
                    "Installable .TTF for macOS, Windows, Figma, Photoshop & Web",
                    "Unlimited free weight, spacing & curve repacks for each job",
                    "Email notification when your HD font finishes building",
                  ].map((item) => (
                    <li key={item} className="flex items-start gap-2.5">
                      <Check className="mt-0.5 h-4 w-4 shrink-0 text-[#d94625]" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>

                <a
                  href="#studio"
                  className="mt-7 inline-flex w-full items-center justify-center rounded-xl bg-[#0a0a0b] px-5 py-3 text-sm font-semibold text-white transition hover:bg-black/85"
                >
                  Create Your Font in {BRAND_NAME} Studio
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="bg-[#fafaf8] py-16 sm:py-20">
        <div className="mx-auto max-w-4xl px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2 font-mono text-xs uppercase tracking-widest text-[#d94625]">
            <HelpCircle className="h-3.5 w-3.5" />
            FAQ &amp; Scope
          </div>
          <h2 className="mt-2 font-display text-3xl text-[#0a0a0b] sm:text-4xl">
            Frequently asked questions
          </h2>

          <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
            {FAQS.map((faq) => (
              <div
                key={faq.q}
                className="rounded-2xl border border-black/10 bg-white p-5"
              >
                <h3 className="text-sm font-semibold text-[#0a0a0b]">
                  {faq.q}
                </h3>
                <p className="mt-2 text-xs leading-relaxed text-black/65">
                  {faq.a}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
