import { useState } from "react";
import { motion } from "motion/react";
import {
  ArrowDown,
  ChevronDown,
  ChevronUp,
  Download,
  Moon,
  Sun,
  Type,
} from "lucide-react";
import { ASCII_94 } from "@glyphcraft/shared";
import { SAMPLE_FONTS } from "../config/samples";

export function HeroPlayground() {
  const [activeId, setActiveId] = useState(SAMPLE_FONTS[0].id);
  const [sampleText, setSampleText] = useState(SAMPLE_FONTS[0].defaultText);
  const [fontSize, setFontSize] = useState(56);
  const [darkSpecimen, setDarkSpecimen] = useState(false);
  const [showAllAscii, setShowAllAscii] = useState(false);

  const activeFont =
    SAMPLE_FONTS.find((f) => f.id === activeId) ?? SAMPLE_FONTS[0];

  const handleSelectFont = (id: string) => {
    const next = SAMPLE_FONTS.find((f) => f.id === id);
    setActiveId(id);
    if (
      next &&
      SAMPLE_FONTS.some((f) => f.defaultText === sampleText)
    ) {
      setSampleText(next.defaultText);
    }
  };

  return (
    <section
      id="playground"
      className="relative overflow-hidden border-b border-black/10 bg-foundry-grid pb-16 pt-12 sm:pb-24 sm:pt-16"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-3xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="inline-flex items-center gap-2 rounded-full border border-black/10 bg-white px-3.5 py-1 text-xs font-medium text-black/70 shadow-xs"
          >
            <span className="h-2 w-2 rounded-full bg-[#d94625]" />
            PixelDiT-Flow Few-Shot Typography Engine
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.06 }}
            className="mt-6 font-display text-5xl leading-[1.03] tracking-tight text-[#0a0a0b] sm:text-7xl"
          >
            Draw 3 letters.{" "}
            <span className="italic text-[#d94625]">Own the entire</span>{" "}
            94-ASCII typeface.
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.12 }}
            className="mx-auto mt-5 max-w-2xl text-base leading-relaxed text-black/65 sm:text-lg"
          >
            Sketch 1–3 style glyphs on our baseline pad or upload a reference
            image. Our flow-matching diffusion model synthesizes every English
            uppercase, lowercase, digit, and symbol into an installable{" "}
            <code className="rounded bg-black/5 px-1.5 py-0.5 font-mono text-sm text-[#0a0a0b]">
              .ttf
            </code>{" "}
            font.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.18 }}
            className="mt-7 flex flex-wrap items-center justify-center gap-3"
          >
            <a
              href="#studio"
              className="inline-flex items-center gap-2 rounded-xl bg-[#0a0a0b] px-6 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-black/85"
            >
              Launch Free Studio
              <ArrowDown className="h-4 w-4 text-[#d94625]" />
            </a>
            <span className="font-mono text-xs text-black/55">
              3 free previews / day · No account needed to test
            </span>
          </motion.div>
        </div>

        {/* Live Interactive Type Specimen Card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.55, delay: 0.22 }}
          className="mt-12 rounded-2xl border border-black/15 bg-white shadow-xl shadow-black/[0.04]"
        >
          {/* Top toolbar: 5 real AI .ttf switchers */}
          <div className="flex flex-col gap-3 border-b border-black/10 p-4 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-1.5">
              {SAMPLE_FONTS.map((font) => {
                const active = font.id === activeFont.id;
                return (
                  <button
                    key={font.id}
                    type="button"
                    onClick={() => handleSelectFont(font.id)}
                    className={`relative rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      active
                        ? "text-white"
                        : "text-black/65 hover:bg-black/5 hover:text-[#0a0a0b]"
                    }`}
                  >
                    {active && (
                      <motion.span
                        layoutId="heroFontPill"
                        transition={{
                          type: "spring",
                          stiffness: 380,
                          damping: 30,
                        }}
                        className="absolute inset-0 rounded-lg bg-[#0a0a0b]"
                      />
                    )}
                    <span className="relative z-10">{font.label}</span>
                  </button>
                );
              })}
            </div>

            <div className="flex flex-wrap items-center gap-4 text-xs text-black/65">
              <label className="flex items-center gap-2">
                <Type className="h-3.5 w-3.5" />
                <span className="font-mono w-10">{fontSize}px</span>
                <input
                  type="range"
                  min={28}
                  max={92}
                  value={fontSize}
                  onChange={(e) => setFontSize(Number(e.target.value))}
                  className="h-1.5 w-24 accent-[#d94625]"
                />
              </label>

              <button
                type="button"
                onClick={() => setDarkSpecimen((v) => !v)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-black/10 px-2.5 py-1 font-medium text-black/70 transition hover:border-black/25"
              >
                {darkSpecimen ? (
                  <>
                    <Sun className="h-3.5 w-3.5" /> Paper
                  </>
                ) : (
                  <>
                    <Moon className="h-3.5 w-3.5" /> Ink
                  </>
                )}
              </button>

              <a
                href={activeFont.ttfUrl}
                download
                className="inline-flex items-center gap-1 rounded-lg border border-black/10 px-2.5 py-1 font-medium text-black/75 transition hover:border-black/30"
              >
                <Download className="h-3.5 w-3.5" />
                .TTF
              </a>
            </div>
          </div>

          {/* Interactive specimen textarea */}
          <div
            className={`relative transition-colors ${
              darkSpecimen
                ? "bg-[#0e0e11] text-[#fafaf8]"
                : "bg-white text-[#0a0a0b]"
            } rounded-b-2xl p-6 sm:p-10`}
          >
            <div className="mb-3 flex items-center justify-between text-[11px] font-mono uppercase tracking-wider opacity-50">
              <span>
                Live AI-Generated Specimen · {activeFont.category} (
                {activeFont.curveMode})
              </span>
              <span>Type below to test real .TTF glyphs</span>
            </div>

            <textarea
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              rows={2}
              spellCheck={false}
              aria-label="Interactive AI font specimen input"
              style={{
                fontFamily: `"${activeFont.fontFamily}", sans-serif`,
                fontSize: `${fontSize}px`,
                lineHeight: 1.18,
              }}
              className="w-full resize-none bg-transparent outline-none placeholder:opacity-30"
              placeholder="Type any English phrase..."
            />

            <div className="mt-4 flex flex-wrap items-center justify-between gap-2 border-t border-current/10 pt-3">
              <button
                type="button"
                onClick={() => setShowAllAscii((v) => !v)}
                className="inline-flex items-center gap-1.5 font-mono text-xs opacity-70 transition hover:opacity-100"
              >
                {showAllAscii ? (
                  <>
                    <ChevronUp className="h-3.5 w-3.5" /> Hide full 94-ASCII
                    character set
                  </>
                ) : (
                  <>
                    <ChevronDown className="h-3.5 w-3.5" /> Inspect all 94
                    generated ASCII glyphs
                  </>
                )}
              </button>
              <span className="font-mono text-[11px] opacity-50">
                Real 10-step HD outputs · Installable on macOS, Windows &amp;
                Figma
              </span>
            </div>

            {showAllAscii && (
              <div
                style={{
                  fontFamily: `"${activeFont.fontFamily}", sans-serif`,
                }}
                className="mt-4 break-all rounded-xl border border-current/10 bg-current/[0.03] p-4 text-2xl leading-relaxed tracking-wide sm:text-3xl"
              >
                {ASCII_94}
              </div>
            )}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
