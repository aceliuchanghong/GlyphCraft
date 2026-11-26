import { useState, type ChangeEvent } from "react";
import { Check, FileUp } from "lucide-react";
import { DEFAULT_BRAND_CONFIG } from "@glyphcraft/shared";

interface TtfRefUploaderProps {
  onExtractSlots: (slots: [string | null, string | null, string | null]) => void;
}

const SIZE = DEFAULT_BRAND_CONFIG.canvasSize; // 128
const BASELINE_Y = DEFAULT_BRAND_CONFIG.baselineY; // 96
const TARGET_CAP = SIZE * 0.44; // ~56px

function measureInkBoundsAtBaseline(
  fontFamily: string,
  char: string,
  fontSize: number
): { minX: number; maxX: number; minY: number; maxY: number } | null {
  const probeSize = 256;
  const baseline = 180;
  const anchorX = 96;
  const canvas = document.createElement("canvas");
  canvas.width = probeSize;
  canvas.height = probeSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, probeSize, probeSize);
  ctx.fillStyle = "#000000";
  ctx.font = `${fontSize}px "${fontFamily}"`;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(char, anchorX, baseline);

  const data = ctx.getImageData(0, 0, probeSize, probeSize).data;
  let minX = probeSize;
  let maxX = -1;
  let minY = probeSize;
  let maxY = -1;

  for (let y = 0; y < probeSize; y++) {
    for (let x = 0; x < probeSize; x++) {
      const idx = (y * probeSize + x) * 4;
      if (data[idx] < 240) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return null;
  return {
    minX: minX - anchorX,
    maxX: maxX - anchorX,
    minY: minY - baseline,
    maxY: maxY - baseline,
  };
}

function renderCharEmBaseline128(
  fontFamily: string,
  char: string,
  fontSize: number
): string | null {
  const scratchW = SIZE * 3;
  const scratchH = SIZE * 2;
  const padX = SIZE;
  const padY = SIZE / 2;
  const scratch = document.createElement("canvas");
  scratch.width = scratchW;
  scratch.height = scratchH;
  const ctx = scratch.getContext("2d");
  if (!ctx) return null;

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, scratchW, scratchH);
  ctx.fillStyle = "#000000";
  ctx.font = `${fontSize}px "${fontFamily}"`;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(char, padX, padY + BASELINE_Y);

  const data = ctx.getImageData(0, 0, scratchW, scratchH).data;
  let minX = scratchW;
  let maxX = -1;
  for (let y = 0; y < scratchH; y++) {
    for (let x = 0; x < scratchW; x++) {
      const idx = (y * scratchW + x) * 4;
      if (data[idx] < 245) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
      }
    }
  }
  if (maxX < 0) return null;

  const inkCenterX = (minX + maxX) / 2;
  const cropX = Math.max(
    0,
    Math.min(scratchW - SIZE, Math.round(inkCenterX - SIZE / 2))
  );

  const out = document.createElement("canvas");
  out.width = SIZE;
  out.height = SIZE;
  const octx = out.getContext("2d");
  if (!octx) return null;

  octx.fillStyle = "#ffffff";
  octx.fillRect(0, 0, SIZE, SIZE);
  octx.drawImage(scratch, cropX, padY, SIZE, SIZE, 0, 0, SIZE, SIZE);
  return out.toDataURL("image/png");
}

export function TtfRefUploader({ onExtractSlots }: TtfRefUploaderProps) {
  const [charsInput, setCharsInput] = useState("Hgp");
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);

  const handleUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setError(null);
    setStatus(null);
    setFileName(file.name);

    try {
      const buf = await file.arrayBuffer();
      const alias = `RefUpload_${Date.now()}`;
      const face = new FontFace(alias, buf);
      const loaded = await face.load();
      document.fonts.add(loaded);

      // Calibrate font size using uppercase H/E/M cap height (matches render_en.py)
      const probeSize = 100;
      const capHeights: number[] = [];
      for (const probeChar of ["H", "E", "M", "T"]) {
        const b = measureInkBoundsAtBaseline(alias, probeChar, probeSize);
        if (b && b.minY < 0) {
          capHeights.push(-b.minY);
        }
      }
      const refCap =
        capHeights.length > 0
          ? capHeights.sort((a, b) => a - b)[Math.floor(capHeights.length / 2)]
          : 72;
      const fontSize = Math.max(16, Math.round(probeSize * (TARGET_CAP / Math.max(18, refCap))));

      const chars = Array.from(charsInput.trim() || "Hgp").slice(0, 3);
      const rendered: [string | null, string | null, string | null] = [
        null,
        null,
        null,
      ];
      chars.forEach((ch, idx) => {
        if (idx < 3) {
          rendered[idx] = renderCharEmBaseline128(alias, ch, fontSize);
        }
      });

      document.fonts.delete(loaded);
      onExtractSlots(rendered);
      setStatus(
        `Extracted ${chars.join(", ")} at calibrated em_baseline (${fontSize}px) into slots.`
      );
    } catch (err) {
      setError(
        err instanceof Error
          ? err.message
          : "Could not parse font file. Upload a valid .ttf or .otf."
      );
    }
  };

  return (
    <div className="rounded-xl border border-black/10 bg-[#fafaf8] p-4 space-y-3">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-xs font-semibold text-[#0a0a0b]">
            Extract 1–3 Reference Glyphs from an Existing .TTF / .OTF
          </p>
          <p className="text-[11px] text-black/55">
            Automatically calibrated to <code className="font-mono">em_baseline</code>{" "}
            (y=96px baseline, ~56px cap height).
          </p>
        </div>

        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 text-xs text-black/70">
            Chars:
            <input
              type="text"
              maxLength={3}
              value={charsInput}
              onChange={(e) => setCharsInput(e.target.value)}
              className="w-16 rounded-lg border border-black/15 bg-white px-2 py-1 font-mono text-xs text-[#0a0a0b] outline-none focus:border-black/40"
            />
          </label>

          <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-black/85">
            <FileUp className="h-3.5 w-3.5" />
            {fileName ? "Replace .TTF" : "Upload .TTF"}
            <input
              type="file"
              accept=".ttf,.otf"
              onChange={(e) => void handleUpload(e)}
              className="hidden"
            />
          </label>
        </div>
      </div>

      {status && (
        <p className="inline-flex items-center gap-1.5 text-xs font-medium text-emerald-700">
          <Check className="h-3.5 w-3.5" /> {status}
        </p>
      )}
      {error && <p className="text-xs text-red-600">{error}</p>}
    </div>
  );
}
