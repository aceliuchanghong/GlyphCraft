import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Check, ImagePlus, RotateCcw, ZoomIn } from "lucide-react";
import { DEFAULT_BRAND_CONFIG } from "@glyphcraft/shared";

interface ImageAlignUploaderProps {
  slots: [string | null, string | null, string | null];
  onChangeSlot: (index: 0 | 1 | 2, dataUrl: string | null) => void;
}

const SIZE = DEFAULT_BRAND_CONFIG.canvasSize; // 128

export function ImageAlignUploader({
  slots,
  onChangeSlot,
}: ImageAlignUploaderProps) {
  const [activeSlot, setActiveSlot] = useState<0 | 1 | 2>(0);
  const [sourceImg, setSourceImg] = useState<HTMLImageElement | null>(null);
  const [scale, setScale] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const [threshold, setThreshold] = useState(175);
  const [invert, setInvert] = useState(false);
  const [applied, setApplied] = useState(false);

  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const dragRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(null);

  const renderAligned128 = (targetCanvas: HTMLCanvasElement) => {
    const ctx = targetCanvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, SIZE, SIZE);

    if (!sourceImg) return;

    const baseScale = Math.min(
      (SIZE * 0.72) / sourceImg.width,
      (SIZE * 0.72) / sourceImg.height
    );
    const drawW = sourceImg.width * baseScale * scale;
    const drawH = sourceImg.height * baseScale * scale;
    const dx = (SIZE - drawW) / 2 + offset.x;
    const dy = (SIZE - drawH) / 2 + offset.y;

    ctx.drawImage(sourceImg, dx, dy, drawW, drawH);

    // Threshold & de-shadow into pure white bg + black ink
    const imgData = ctx.getImageData(0, 0, SIZE, SIZE);
    const d = imgData.data;
    for (let i = 0; i < d.length; i += 4) {
      const lum = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
      const isInk = invert ? lum > threshold : lum < threshold;
      const val = isInk ? 0 : 255;
      d[i] = val;
      d[i + 1] = val;
      d[i + 2] = val;
      d[i + 3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  };

  useEffect(() => {
    if (previewCanvasRef.current) {
      renderAligned128(previewCanvasRef.current);
    }
  });

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        setSourceImg(img);
        setScale(1);
        setOffset({ x: 0, y: 0 });
        setApplied(false);
      };
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!sourceImg) return;
    e.currentTarget.setPointerCapture(e.pointerId);
    dragRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      origX: offset.x,
      origY: offset.y,
    };
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const factor = SIZE / rect.width;
    setOffset({
      x: dragRef.current.origX + (e.clientX - dragRef.current.startX) * factor,
      y: dragRef.current.origY + (e.clientY - dragRef.current.startY) * factor,
    });
    setApplied(false);
  };

  const handlePointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    try {
      e.currentTarget.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore
    }
  };

  const applyToSlot = () => {
    if (!previewCanvasRef.current || !sourceImg) return;
    renderAligned128(previewCanvasRef.current);
    const dataUrl = previewCanvasRef.current.toDataURL("image/png");
    onChangeSlot(activeSlot, dataUrl);
    setApplied(true);
  };

  return (
    <div className="grid grid-cols-1 gap-5 rounded-xl border border-black/10 bg-[#fafaf8] p-4 sm:grid-cols-[240px_1fr]">
      {/* Cropper frame */}
      <div className="flex flex-col items-center">
        <div className="relative h-56 w-56 overflow-hidden rounded-xl border border-black/20 bg-white">
          <canvas
            ref={previewCanvasRef}
            width={SIZE}
            height={SIZE}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            className="h-full w-full cursor-grab active:cursor-grabbing touch-none"
          />
          <svg
            viewBox="0 0 128 128"
            className="pointer-events-none absolute inset-0 h-full w-full"
          >
            <line
              x1="0"
              y1="40"
              x2="128"
              y2="40"
              stroke="#3b82f6"
              strokeOpacity="0.4"
              strokeWidth="0.8"
              strokeDasharray="2 2"
            />
            <line
              x1="0"
              y1="56"
              x2="128"
              y2="56"
              stroke="#64748b"
              strokeOpacity="0.35"
              strokeWidth="0.8"
              strokeDasharray="2 2"
            />
            <line
              x1="0"
              y1="96"
              x2="128"
              y2="96"
              stroke="#d94625"
              strokeOpacity="0.7"
              strokeWidth="1"
            />
          </svg>

          {!sourceImg && (
            <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center p-4 text-center text-xs text-black/45">
              <ImagePlus className="mb-1.5 h-6 w-6" />
              Upload a letter photo or screenshot, then drag to align its bottom
              to the red baseline.
            </div>
          )}
        </div>
        <span className="mt-2 font-mono text-[11px] text-black/50">
          Drag image to align · Outputs 128×128 B&amp;W PNG
        </span>
      </div>

      {/* Controls */}
      <div className="flex flex-col justify-between space-y-4">
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              {([0, 1, 2] as const).map((idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => {
                    setActiveSlot(idx);
                    setApplied(false);
                  }}
                  className={`rounded-lg px-2.5 py-1 font-mono text-xs font-medium transition ${
                    activeSlot === idx
                      ? "bg-[#0a0a0b] text-white"
                      : "border border-black/15 bg-white text-black/70"
                  }`}
                >
                  Slot {idx + 1} {slots[idx] ? "•" : ""}
                </button>
              ))}
            </div>

            <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-black/20 bg-white px-3 py-1.5 text-xs font-semibold text-[#0a0a0b] transition hover:border-black/40">
              <ImagePlus className="h-3.5 w-3.5" />
              Choose Image
              <input
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>
          </div>

          <label className="block text-xs">
            <div className="mb-1 flex items-center justify-between font-medium text-black/75">
              <span className="inline-flex items-center gap-1">
                <ZoomIn className="h-3.5 w-3.5" /> Zoom
              </span>
              <span className="font-mono text-black/55">{scale.toFixed(2)}×</span>
            </div>
            <input
              type="range"
              min={0.35}
              max={2.6}
              step={0.05}
              value={scale}
              onChange={(e) => {
                setScale(Number(e.target.value));
                setApplied(false);
              }}
              className="h-1.5 w-full accent-[#0a0a0b]"
            />
          </label>

          <label className="block text-xs">
            <div className="mb-1 flex items-center justify-between font-medium text-black/75">
              <span>Shadow Removal Threshold</span>
              <span className="font-mono text-black/55">{threshold}</span>
            </div>
            <input
              type="range"
              min={40}
              max={235}
              value={threshold}
              onChange={(e) => {
                setThreshold(Number(e.target.value));
                setApplied(false);
              }}
              className="h-1.5 w-full accent-[#d94625]"
            />
          </label>

          <div className="flex items-center justify-between text-xs">
            <label className="inline-flex items-center gap-2 font-medium text-black/75">
              <input
                type="checkbox"
                checked={invert}
                onChange={(e) => {
                  setInvert(e.target.checked);
                  setApplied(false);
                }}
                className="accent-[#0a0a0b]"
              />
              Invert light-on-dark image
            </label>

            <button
              type="button"
              onClick={() => {
                setScale(1);
                setOffset({ x: 0, y: 0 });
                setApplied(false);
              }}
              className="inline-flex items-center gap-1 text-black/55 hover:text-[#0a0a0b]"
            >
              <RotateCcw className="h-3 w-3" /> Reset crop
            </button>
          </div>
        </div>

        <button
          type="button"
          disabled={!sourceImg}
          onClick={applyToSlot}
          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[#0a0a0b] px-4 py-2.5 text-xs font-semibold text-white transition hover:bg-black/85 disabled:cursor-not-allowed disabled:opacity-40"
        >
          {applied ? (
            <>
              <Check className="h-4 w-4 text-emerald-400" />
              Saved to Slot {activeSlot + 1}
            </>
          ) : (
            `Apply Aligned Glyph to Slot ${activeSlot + 1}`
          )}
        </button>
      </div>
    </div>
  );
}
