import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { Check, ImagePlus, RotateCcw, X, ZoomIn } from "lucide-react";
import { DEFAULT_BRAND_CONFIG } from "@glyphcraft/shared";

interface ImageAlignUploaderProps {
  slots: [string | null, string | null, string | null];
  onChangeSlot: (index: 0 | 1 | 2, dataUrl: string | null) => void;
}

const SIZE = DEFAULT_BRAND_CONFIG.canvasSize; // 128
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png"]);

function isAllowedJpgOrPng(file: File): boolean {
  if (file.type) {
    return ALLOWED_MIME_TYPES.has(file.type.toLowerCase());
  }
  return /\.(jpe?g|png)$/i.test(file.name);
}

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
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const previewCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const dragRef = useRef<{
    startX: number;
    startY: number;
    origX: number;
    origY: number;
  } | null>(null);

  const renderAligned128 = useCallback(
    (targetCanvas: HTMLCanvasElement) => {
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
    },
    [sourceImg, scale, offset, threshold, invert]
  );

  useEffect(() => {
    if (previewCanvasRef.current) {
      renderAligned128(previewCanvasRef.current);
    }
  }, [renderAligned128]);

  const loadImageFile = (file: File) => {
    if (!isAllowedJpgOrPng(file)) {
      setUploadError("Only JPG and PNG images are supported.");
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setUploadError("Image must be 10 MB or smaller.");
      return;
    }

    setUploadError(null);
    const reader = new FileReader();
    reader.onerror = () => {
      setUploadError(
        "Could not read this image file. Please try another JPG or PNG."
      );
    };
    reader.onload = () => {
      const img = new Image();
      img.onerror = () => {
        setUploadError(
          "Could not read this image file. Please try another JPG or PNG."
        );
      };
      img.onload = () => {
        setSourceImg(img);
        setScale(1);
        setOffset({ x: 0, y: 0 });
        setApplied(false);
        setUploadError(null);
      };
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    loadImageFile(file);
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDraggingFile(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      loadImageFile(file);
    }
  };

  const handleCanvasClick = () => {
    if (!sourceImg) {
      fileInputRef.current?.click();
    }
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!sourceImg) return;
    if (e.pointerType === "mouse" && e.button !== 0) return;
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
    <div className="space-y-3 rounded-xl border border-black/10 bg-[#fafaf8] p-4">
      <div className="grid grid-cols-1 gap-5 sm:grid-cols-[240px_1fr]">
        {/* Cropper frame */}
        <div className="flex flex-col items-center">
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`relative h-56 w-56 overflow-hidden rounded-xl border bg-white transition ${
              isDraggingFile
                ? "border-[#d94625] ring-2 ring-[#d94625]/30"
                : "border-black/20"
            }`}
          >
            <canvas
              ref={previewCanvasRef}
              width={SIZE}
              height={SIZE}
              onClick={handleCanvasClick}
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerUp}
              className={`h-full w-full touch-none ${
                sourceImg
                  ? "cursor-grab active:cursor-grabbing"
                  : "cursor-pointer"
              }`}
            />
            {sourceImg && (
              <svg
                viewBox="0 0 128 128"
                className="pointer-events-none absolute inset-0 h-full w-full"
                aria-hidden="true"
              >
                {/* Cap height y=40 */}
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
                <text
                  x="3"
                  y="38"
                  fill="#3b82f6"
                  fillOpacity="0.65"
                  fontSize="5"
                  fontFamily="monospace"
                >
                  CAP
                </text>

                {/* x-height y=56 */}
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
                <text
                  x="3"
                  y="54"
                  fill="#64748b"
                  fillOpacity="0.65"
                  fontSize="5"
                  fontFamily="monospace"
                >
                  x-ht
                </text>

                {/* Baseline y=96 */}
                <line
                  x1="0"
                  y1="96"
                  x2="128"
                  y2="96"
                  stroke="#d94625"
                  strokeOpacity="0.7"
                  strokeWidth="1"
                />
                <text
                  x="3"
                  y="94"
                  fill="#d94625"
                  fillOpacity="0.85"
                  fontSize="5"
                  fontFamily="monospace"
                >
                  BASE
                </text>

                {/* Descender y=118 */}
                <line
                  x1="0"
                  y1="118"
                  x2="128"
                  y2="118"
                  stroke="#64748b"
                  strokeOpacity="0.25"
                  strokeWidth="0.6"
                  strokeDasharray="1.5 2"
                />
              </svg>
            )}

            {!sourceImg && (
              <div className="pointer-events-none absolute inset-0 flex flex-col items-center justify-center p-4 text-center text-xs text-black/50">
                <ImagePlus className="mb-2 h-6 w-6 text-black/40" />
                <span className="font-medium text-black/70">
                  Click or drop a style sample
                </span>
                <span className="mt-1 text-[11px] text-black/45">
                  Upload a letter photo or image, then align it to the BASE line.
                </span>
              </div>
            )}
          </div>
          <span className="mt-2 font-mono text-[11px] text-black/50">
            {sourceImg
              ? "Drag sample to align with guides"
              : "JPG or PNG, up to 10 MB"}
          </span>
        </div>

        {/* Controls */}
        <div className="flex flex-col justify-between space-y-4">
          <div className="space-y-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap items-center gap-1.5">
                {([0, 1, 2] as const).map((idx) => {
                  const slotVal = slots[idx];
                  const isActive = activeSlot === idx;
                  return (
                    <div
                      key={idx}
                      className={`inline-flex items-center gap-1.5 rounded-lg border px-2 py-1 transition ${
                        isActive
                          ? "border-[#0a0a0b] bg-[#0a0a0b] text-white"
                          : "border-black/15 bg-white text-black/75 hover:border-black/30"
                      }`}
                    >
                      <button
                        type="button"
                        onClick={() => {
                          setActiveSlot(idx);
                          setApplied(false);
                        }}
                        className="inline-flex items-center gap-1.5 font-mono text-xs font-medium"
                      >
                        {slotVal && (
                          <img
                            src={slotVal}
                            alt={`Letter ${idx + 1}`}
                            className="h-5 w-5 rounded border border-black/10 bg-white object-contain"
                          />
                        )}
                        <span>Letter {idx + 1}</span>
                      </button>
                      {slotVal && (
                        <button
                          type="button"
                          onClick={() => {
                            onChangeSlot(idx, null);
                            if (isActive) setApplied(false);
                          }}
                          title={`Clear Letter ${idx + 1}`}
                          aria-label={`Clear Letter ${idx + 1}`}
                          className={`rounded p-0.5 transition ${
                            isActive
                              ? "text-white/70 hover:bg-white/15 hover:text-white"
                              : "text-black/40 hover:bg-black/5 hover:text-red-600"
                          }`}
                        >
                          <X className="h-3 w-3" />
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>

              <label className="inline-flex cursor-pointer items-center gap-1.5 rounded-lg border border-black/20 bg-white px-3 py-1.5 text-xs font-semibold text-[#0a0a0b] transition hover:border-black/40">
                <ImagePlus className="h-3.5 w-3.5" />
                {sourceImg ? "Replace Sample" : "Upload Sample"}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".jpg,.jpeg,.png,image/jpeg,image/png"
                  onChange={handleFileChange}
                  className="hidden"
                />
              </label>
            </div>

            <label
              className={`block text-xs ${!sourceImg ? "opacity-50" : ""}`}
            >
              <div className="mb-1 flex items-center justify-between font-medium text-black/75">
                <span className="inline-flex items-center gap-1">
                  <ZoomIn className="h-3.5 w-3.5" /> Zoom
                </span>
                <span className="font-mono text-black/55">
                  {scale.toFixed(2)}×
                </span>
              </div>
              <input
                type="range"
                min={0.35}
                max={2.6}
                step={0.05}
                disabled={!sourceImg}
                value={scale}
                onChange={(e) => {
                  setScale(Number(e.target.value));
                  setApplied(false);
                }}
                className="h-1.5 w-full accent-[#0a0a0b] disabled:cursor-not-allowed"
              />
            </label>

            <label
              className={`block text-xs ${!sourceImg ? "opacity-50" : ""}`}
            >
              <div className="mb-1 flex items-center justify-between font-medium text-black/75">
                <span>Ink Contrast</span>
                <span className="font-mono text-black/55">{threshold}</span>
              </div>
              <input
                type="range"
                min={40}
                max={235}
                disabled={!sourceImg}
                value={threshold}
                onChange={(e) => {
                  setThreshold(Number(e.target.value));
                  setApplied(false);
                }}
                className="h-1.5 w-full accent-[#d94625] disabled:cursor-not-allowed"
              />
            </label>

            <div className="flex items-center justify-between text-xs">
              <label
                className={`inline-flex items-center gap-2 font-medium text-black/75 ${
                  !sourceImg ? "cursor-not-allowed opacity-50" : ""
                }`}
              >
                <input
                  type="checkbox"
                  disabled={!sourceImg}
                  checked={invert}
                  onChange={(e) => {
                    setInvert(e.target.checked);
                    setApplied(false);
                  }}
                  className="accent-[#0a0a0b] disabled:cursor-not-allowed"
                />
                Invert colors (light text on dark bg)
              </label>

              <button
                type="button"
                disabled={!sourceImg}
                onClick={() => {
                  setScale(1);
                  setOffset({ x: 0, y: 0 });
                  setApplied(false);
                }}
                className="inline-flex items-center gap-1 text-black/55 hover:text-[#0a0a0b] disabled:cursor-not-allowed disabled:opacity-40"
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
                Saved to Letter {activeSlot + 1}
              </>
            ) : (
              `Save to Letter ${activeSlot + 1}`
            )}
          </button>
        </div>
      </div>

      {uploadError && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
          {uploadError}
        </div>
      )}
    </div>
  );
}
