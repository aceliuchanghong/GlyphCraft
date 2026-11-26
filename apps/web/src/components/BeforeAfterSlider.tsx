import { useState } from "react";
import { MoveHorizontal } from "lucide-react";
import { SAMPLE_FONTS, STUDIO_PRESETS } from "../config/samples";

const SHOWCASE_ITEMS = [
  {
    id: "grenze",
    label: "Blackletter",
    preset: STUDIO_PRESETS[0],
    font: SAMPLE_FONTS[0],
  },
  {
    id: "cherry",
    label: "Cherry Swash",
    preset: STUDIO_PRESETS[1],
    font: SAMPLE_FONTS[1],
  },
  {
    id: "dotgothic",
    label: "DotGothic Pixel",
    preset: STUDIO_PRESETS[2],
    font: SAMPLE_FONTS[2],
  },
];

export function BeforeAfterSlider() {
  const [selectedId, setSelectedId] = useState(SHOWCASE_ITEMS[0].id);
  const [position, setPosition] = useState(48);

  const item =
    SHOWCASE_ITEMS.find((i) => i.id === selectedId) ?? SHOWCASE_ITEMS[0];

  return (
    <section className="border-b border-black/10 bg-white py-16 sm:py-20">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-end">
          <div>
            <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
              3-Shot Style Transfer
            </p>
            <h2 className="mt-2 font-display text-3xl tracking-tight text-[#0a0a0b] sm:text-4xl">
              From 3 reference glyphs to a full 94-character poster.
            </h2>
          </div>

          <div className="flex items-center gap-1.5 rounded-xl border border-black/10 bg-[#fafaf8] p-1">
            {SHOWCASE_ITEMS.map((entry) => (
              <button
                key={entry.id}
                type="button"
                onClick={() => setSelectedId(entry.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition ${
                  entry.id === item.id
                    ? "bg-[#0a0a0b] text-white"
                    : "text-black/60 hover:text-[#0a0a0b]"
                }`}
              >
                {entry.label}
              </button>
            ))}
          </div>
        </div>

        {/* Comparison Viewport */}
        <div className="relative mt-8 h-[340px] select-none overflow-hidden rounded-2xl border border-black/15 bg-[#fafaf8] sm:h-[400px]">
          {/* AFTER Layer (full width, right side visible) */}
          <div className="absolute inset-0 flex flex-col justify-between bg-[#0a0a0b] p-6 text-[#fafaf8] sm:p-10">
            <div className="flex items-center justify-between">
              <span className="rounded-full border border-white/15 bg-white/10 px-3 py-1 font-mono text-[11px] uppercase tracking-wider">
                After · Generated 94-ASCII .TTF
              </span>
              <span className="font-mono text-xs text-white/50">
                10-Step HD Flow + TrueType Vector
              </span>
            </div>

            <div
              style={{ fontFamily: `"${item.font.fontFamily}", sans-serif` }}
              className="space-y-3 text-right sm:space-y-4"
            >
              <p className="text-3xl leading-tight sm:text-5xl">
                Sphinx of black quartz, judge my vow!
              </p>
              <p className="text-xl tracking-wide text-white/80 sm:text-3xl">
                ABCDEFGHIJKLMNOPQRSTUVWXYZ
              </p>
              <p className="text-xl tracking-wide text-white/65 sm:text-2xl">
                abcdefghijklmnopqrstuvwxyz 0123456789 !?@#$%&amp;*
              </p>
            </div>
          </div>

          {/* BEFORE Layer (clipped by slider position) */}
          <div
            className="absolute inset-0 flex flex-col justify-between bg-[#fafaf8] p-6 text-[#0a0a0b] sm:p-10"
            style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
          >
            <div className="flex items-center justify-between">
              <span className="rounded-full border border-black/15 bg-white px-3 py-1 font-mono text-[11px] uppercase tracking-wider">
                Before · Only 3 Input PNGs (128×128)
              </span>
            </div>

            <div className="flex flex-wrap items-center gap-4 sm:gap-6">
              {item.preset.glyphs.map((src, idx) => (
                <div
                  key={src}
                  className="flex flex-col items-center rounded-xl border border-black/15 bg-white p-3 shadow-sm"
                >
                  <img
                    src={src}
                    alt={`Reference glyph ${idx + 1}`}
                    className="h-24 w-24 object-contain sm:h-32 sm:w-32"
                    draggable={false}
                  />
                  <span className="mt-2 font-mono text-[11px] text-black/50">
                    Ref #{idx + 1} · em_baseline
                  </span>
                </div>
              ))}
            </div>

            <p className="font-mono text-xs text-black/55">
              Drag the divider to compare input references against the full AI
              typeface →
            </p>
          </div>

          {/* Divider line & handle */}
          <div
            className="pointer-events-none absolute inset-y-0 w-0.5 bg-[#d94625]"
            style={{ left: `${position}%` }}
          >
            <div className="absolute left-1/2 top-1/2 flex h-10 w-10 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border-2 border-white bg-[#d94625] text-white shadow-lg">
              <MoveHorizontal className="h-4 w-4" />
            </div>
          </div>

          <input
            type="range"
            min={5}
            max={95}
            value={position}
            onChange={(e) => setPosition(Number(e.target.value))}
            aria-label="Before and after comparison slider"
            className="absolute inset-0 z-20 h-full w-full cursor-ew-resize opacity-0"
          />
        </div>
      </div>
    </section>
  );
}
