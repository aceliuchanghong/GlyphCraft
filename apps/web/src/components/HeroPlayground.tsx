import { useState } from "react";
import { Download } from "lucide-react";
import { SAMPLE_FONTS } from "../config/samples";

interface HeroPlaygroundProps {
  onUseInStudio?: (presetSlug: string) => void;
}

export function HeroPlayground({ onUseInStudio }: HeroPlaygroundProps) {
  const [activeId, setActiveId] = useState(SAMPLE_FONTS[0].id);
  const [sampleText, setSampleText] = useState(SAMPLE_FONTS[0].defaultText);
  const [fontSize, setFontSize] = useState(44);

  const activeFont =
    SAMPLE_FONTS.find((f) => f.id === activeId) ?? SAMPLE_FONTS[0];
  const hasNonAscii = /[^\x20-\x7E\r\n]/.test(sampleText);

  const handleSelectFont = (id: string) => {
    const next = SAMPLE_FONTS.find((f) => f.id === id);
    setActiveId(id);
    if (next && SAMPLE_FONTS.some((f) => f.defaultText === sampleText)) {
      setSampleText(next.defaultText);
    }
  };

  return (
    <section
      id="examples"
      className="border-b border-black/10 bg-white py-14 sm:py-16"
    >
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h2 className="font-display text-3xl tracking-tight text-[#0a0a0b] sm:text-4xl">
              Try sample generated fonts
            </h2>
            <p className="mt-1 text-sm text-black/60">
              Switch between real AI-generated{" "}
              <code className="font-mono">.ttf</code> fonts and type anything
              below.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            {activeFont.presetSlug && onUseInStudio && (
              <button
                type="button"
                onClick={() => onUseInStudio(activeFont.presetSlug!)}
                className="inline-flex w-fit items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3 py-1.5 text-xs font-medium text-white transition hover:bg-black/85"
              >
                Try in Studio
              </button>
            )}
            <a
              href={activeFont.ttfUrl}
              download={activeFont.downloadFilename}
              className="inline-flex w-fit items-center gap-1.5 rounded-lg border border-black/15 px-3 py-1.5 text-xs font-medium text-[#0a0a0b] transition hover:border-black/35"
            >
              <Download className="h-3.5 w-3.5" />
              Download {activeFont.label} (.ttf)
            </a>
          </div>
        </div>

        <div className="mt-6 rounded-2xl border border-black/15 bg-[#fafaf8] shadow-xs">
          <div className="flex flex-wrap items-center justify-between gap-3 border-b border-black/10 p-3">
            <div className="flex flex-wrap items-center gap-1.5">
              {SAMPLE_FONTS.map((font) => {
                const active = font.id === activeFont.id;
                return (
                  <button
                    key={font.id}
                    type="button"
                    onClick={() => handleSelectFont(font.id)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition ${
                      active
                        ? "bg-[#0a0a0b] text-white"
                        : "text-black/65 hover:bg-black/5 hover:text-[#0a0a0b]"
                    }`}
                  >
                    {font.label}
                  </button>
                );
              })}
            </div>

            <div className="flex items-center gap-3 px-1">
              <span className="hidden font-mono text-[11px] text-black/45 sm:inline">
                {activeFont.category}
              </span>
              <label className="flex items-center gap-2 text-xs text-black/60">
                <span className="font-mono text-[11px]">{fontSize}px</span>
                <input
                  type="range"
                  min={24}
                  max={72}
                  value={fontSize}
                  onChange={(e) => setFontSize(Number(e.target.value))}
                  aria-label="Sample font size"
                  className="h-1.5 w-24 accent-[#0a0a0b]"
                />
              </label>
            </div>
          </div>

          <div className="rounded-b-2xl bg-white p-6 sm:p-8">
            <div className="mb-2 flex items-center justify-between">
              <span className="font-mono text-[11px] uppercase tracking-wider text-black/35">
                Type below to test
              </span>
              {sampleText !== activeFont.defaultText && (
                <button
                  type="button"
                  onClick={() => setSampleText(activeFont.defaultText)}
                  className="font-mono text-[11px] text-black/45 transition hover:text-[#0a0a0b]"
                >
                  Reset text
                </button>
              )}
            </div>
            <textarea
              value={sampleText}
              onChange={(e) => setSampleText(e.target.value)}
              rows={2}
              spellCheck={false}
              aria-label="Interactive font sample input"
              style={{
                fontFamily: `"${activeFont.fontFamily}", sans-serif`,
                fontSize: `${fontSize}px`,
                lineHeight: 1.35,
              }}
              className="min-h-[120px] w-full resize-none bg-transparent text-[#0a0a0b] outline-none placeholder:opacity-30"
              placeholder="Type any English phrase..."
            />
            {hasNonAscii && (
              <p className="mt-2 text-xs text-amber-700">
                Sample fonts cover English characters (A–Z, a–z, 0–9,
                punctuation) only.
              </p>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
