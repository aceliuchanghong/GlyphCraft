import { useState } from "react";
import { Loader2, SlidersHorizontal } from "lucide-react";
import {
  sanitizeFamilyName,
  type CurveMode,
  type TtfParams,
} from "@glyphcraft/shared";

interface FontRepackControlsProps {
  initialFamilyName: string;
  initialParams: TtfParams;
  repacking: boolean;
  onRepack: (params: {
    familyName: string;
    threshold: number;
    sidebearing: number;
    curveMode: CurveMode;
  }) => Promise<void>;
}

export function FontRepackControls({
  initialFamilyName,
  initialParams,
  repacking,
  onRepack,
}: FontRepackControlsProps) {
  const [familyName, setFamilyName] = useState(initialFamilyName);
  const [threshold, setThreshold] = useState(initialParams.threshold ?? 160);
  const [sidebearing, setSidebearing] = useState(
    initialParams.sidebearing ?? 55
  );
  const [curveMode, setCurveMode] = useState<CurveMode>(
    initialParams.curve_mode ?? "spline"
  );

  const handleSubmit = async () => {
    await onRepack({
      familyName: sanitizeFamilyName(familyName),
      threshold,
      sidebearing,
      curveMode,
    });
  };

  return (
    <div className="rounded-2xl border border-black/15 bg-white p-5 shadow-sm">
      <div className="flex items-center gap-2">
        <SlidersHorizontal className="h-4 w-4 text-[#d94625]" />
        <h3 className="text-sm font-semibold text-[#0a0a0b]">
          Fine-Tune &amp; Re-export .TTF (Free)
        </h3>
      </div>
      <p className="mt-1 text-xs text-black/55">
        Adjust stroke weight, letter spacing, and outline curves anytime for
        free.
      </p>

      <div className="mt-5 space-y-4">
        <label className="block text-xs">
          <span className="font-medium text-black/75">Font Family Name</span>
          <input
            type="text"
            maxLength={40}
            value={familyName}
            onChange={(e) => setFamilyName(e.target.value)}
            className="mt-1.5 w-full rounded-lg border border-black/15 bg-[#fafaf8] px-3 py-2 text-sm text-[#0a0a0b] outline-none focus:border-[#0a0a0b]"
          />
        </label>

        <label className="block text-xs">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-medium text-black/75">
              Stroke Weight
            </span>
            <span className="font-mono text-black/55">{threshold}</span>
          </div>
          <input
            type="range"
            min={100}
            max={220}
            value={threshold}
            onChange={(e) => setThreshold(Number(e.target.value))}
            className="h-1.5 w-full accent-[#0a0a0b]"
          />
          <div className="mt-1 flex justify-between font-mono text-[10px] text-black/40">
            <span>Thinner (100)</span>
            <span>Default (160)</span>
            <span>Heavier (220)</span>
          </div>
        </label>

        <label className="block text-xs">
          <div className="mb-1 flex items-center justify-between">
            <span className="font-medium text-black/75">
              Letter Spacing
            </span>
            <span className="font-mono text-black/55">{sidebearing}</span>
          </div>
          <input
            type="range"
            min={10}
            max={120}
            value={sidebearing}
            onChange={(e) => setSidebearing(Number(e.target.value))}
            className="h-1.5 w-full accent-[#0a0a0b]"
          />
          <div className="mt-1 flex justify-between font-mono text-[10px] text-black/40">
            <span>Tight (10)</span>
            <span>Default (55)</span>
            <span>Airy (120)</span>
          </div>
        </label>

        <div className="text-xs">
          <span className="font-medium text-black/75">Outline Style</span>
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
              ] as const
            ).map((opt) => (
              <button
                key={opt.mode}
                type="button"
                onClick={() => setCurveMode(opt.mode)}
                className={`rounded-lg border px-3 py-2 text-left transition ${
                  curveMode === opt.mode
                    ? "border-[#0a0a0b] bg-[#0a0a0b] text-white"
                    : "border-black/15 bg-[#fafaf8] text-black/70 hover:border-black/30"
                }`}
              >
                <div className="font-semibold">{opt.label}</div>
                <div className="text-[10px] opacity-70">{opt.sub}</div>
              </button>
            ))}
          </div>
        </div>

        <button
          type="button"
          disabled={repacking}
          onClick={() => void handleSubmit()}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[#d94625] px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-[#c53c1d] disabled:opacity-50"
        >
          {repacking ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Updating .TTF...
            </>
          ) : (
            "Apply & Update .TTF"
          )}
        </button>
      </div>
    </div>
  );
}
