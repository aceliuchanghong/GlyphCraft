import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { RotateCcw, Trash2 } from "lucide-react";
import { DEFAULT_BRAND_CONFIG } from "@glyphcraft/shared";

interface GlyphCanvasPadProps {
  slots: [string | null, string | null, string | null];
  onChangeSlot: (index: 0 | 1 | 2, dataUrl: string | null) => void;
}

const SLOT_HINTS: Array<{ char: string; hint: string; required?: boolean }> = [
  { char: "H", hint: "Cap height (y=40..96)", required: true },
  { char: "g", hint: "Descender & curve" },
  { char: "p", hint: "Bowl & stem (optional)" },
];

const MODEL_SIZE = DEFAULT_BRAND_CONFIG.canvasSize; // 128
const DISPLAY_SCALE = 2; // 256x256 interactive drawing canvas -> crisp export to 128x128

interface SinglePadProps {
  index: 0 | 1 | 2;
  value: string | null;
  brushSize: number;
  onChange: (dataUrl: string | null) => void;
}

function SingleGlyphPad({ index, value, brushSize, onChange }: SinglePadProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const historyRef = useRef<ImageData[]>([]);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const lastLoadedValueRef = useRef<string | null>(null);

  const meta = SLOT_HINTS[index];

  const initWhiteCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  // Load external updates (presets, TTF uploads, or restored localStorage draft)
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    if (value === lastLoadedValueRef.current) return;

    if (!value) {
      initWhiteCanvas();
      historyRef.current = [];
      lastLoadedValueRef.current = null;
      return;
    }

    const img = new Image();
    img.onload = () => {
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      lastLoadedValueRef.current = value;
    };
    img.src = value;
  }, [value, initWhiteCanvas]);

  const export128Png = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = MODEL_SIZE;
    exportCanvas.height = MODEL_SIZE;
    const ectx = exportCanvas.getContext("2d");
    if (!ectx) return;

    ectx.fillStyle = "#ffffff";
    ectx.fillRect(0, 0, MODEL_SIZE, MODEL_SIZE);
    ectx.drawImage(canvas, 0, 0, MODEL_SIZE, MODEL_SIZE);

    // Check if there is any non-white ink
    const data = ectx.getImageData(0, 0, MODEL_SIZE, MODEL_SIZE).data;
    let hasInk = false;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] < 245 || data[i + 1] < 245 || data[i + 2] < 245) {
        hasInk = true;
        break;
      }
    }

    if (!hasInk) {
      lastLoadedValueRef.current = null;
      onChange(null);
      return;
    }

    const url = exportCanvas.toDataURL("image/png");
    lastLoadedValueRef.current = url;
    onChange(url);
  }, [onChange]);

  const getCoords = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current!;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * canvas.width,
      y: ((e.clientY - rect.top) / rect.height) * canvas.height,
    };
  };

  const handlePointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    canvas.setPointerCapture(e.pointerId);
    historyRef.current.push(
      ctx.getImageData(0, 0, canvas.width, canvas.height)
    );
    if (historyRef.current.length > 25) {
      historyRef.current.shift();
    }

    const pt = getCoords(e);
    drawingRef.current = true;
    lastPointRef.current = pt;

    const pressure = e.pressure > 0 && e.pointerType === "pen" ? e.pressure : 0.7;
    const radius = (brushSize * DISPLAY_SCALE * (0.65 + 0.55 * pressure)) / 2;

    ctx.fillStyle = "#000000";
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, radius, 0, Math.PI * 2);
    ctx.fill();
  };

  const handlePointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current || !lastPointRef.current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    const pt = getCoords(e);
    const pressure = e.pressure > 0 && e.pointerType === "pen" ? e.pressure : 0.7;
    const width = brushSize * DISPLAY_SCALE * (0.65 + 0.55 * pressure);

    ctx.strokeStyle = "#000000";
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(lastPointRef.current.x, lastPointRef.current.y);
    ctx.lineTo(pt.x, pt.y);
    ctx.stroke();

    lastPointRef.current = pt;
  };

  const handlePointerUp = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!drawingRef.current) return;
    drawingRef.current = false;
    lastPointRef.current = null;
    try {
      canvasRef.current?.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore if pointer capture already released
    }
    export128Png();
  };

  const handleUndo = () => {
    const canvas = canvasRef.current;
    const prev = historyRef.current.pop();
    if (!canvas || !prev) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    ctx.putImageData(prev, 0, 0);
    export128Png();
  };

  const handleClear = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;
    historyRef.current.push(
      ctx.getImageData(0, 0, canvas.width, canvas.height)
    );
    initWhiteCanvas();
    lastLoadedValueRef.current = null;
    onChange(null);
  };

  return (
    <div className="flex flex-col rounded-xl border border-black/15 bg-white p-3">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <span className="font-mono text-xs font-semibold text-[#0a0a0b]">
            Slot {index + 1}: {meta.char}
          </span>
          {meta.required && (
            <span className="ml-1.5 rounded bg-[#d94625]/10 px-1.5 py-0.5 font-mono text-[10px] font-medium text-[#d94625]">
              min 1
            </span>
          )}
          <p className="text-[11px] text-black/50">{meta.hint}</p>
        </div>

        <div className="flex items-center gap-1">
          <button
            type="button"
            onClick={handleUndo}
            title="Undo stroke"
            className="rounded-md border border-black/10 p-1.5 text-black/60 transition hover:bg-black/5 hover:text-[#0a0a0b]"
          >
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
          <button
            type="button"
            onClick={handleClear}
            title="Clear slot"
            className="rounded-md border border-black/10 p-1.5 text-black/60 transition hover:bg-red-50 hover:text-red-600"
          >
            <Trash2 className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {/* Canvas + Non-exported em_baseline overlay guides */}
      <div className="relative aspect-square w-full overflow-hidden rounded-lg border border-black/15 bg-white touch-none">
        <canvas
          ref={canvasRef}
          width={MODEL_SIZE * DISPLAY_SCALE}
          height={MODEL_SIZE * DISPLAY_SCALE}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
          className="h-full w-full cursor-crosshair"
        />

        {/* Four-line baseline guides matching y=40, y=56, y=96, y=118 out of 128 */}
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
            strokeOpacity="0.35"
            strokeWidth="0.7"
            strokeDasharray="2 2"
          />
          <text
            x="3"
            y="38"
            fill="#3b82f6"
            fillOpacity="0.6"
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
            strokeOpacity="0.3"
            strokeWidth="0.7"
            strokeDasharray="2 2"
          />
          <text
            x="3"
            y="54"
            fill="#64748b"
            fillOpacity="0.6"
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
            strokeOpacity="0.6"
            strokeWidth="0.9"
          />
          <text
            x="3"
            y="94"
            fill="#d94625"
            fillOpacity="0.8"
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
      </div>
    </div>
  );
}

export function GlyphCanvasPad({ slots, onChangeSlot }: GlyphCanvasPadProps) {
  const [brushSize, setBrushSize] = useState(7);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-black/10 bg-[#fafaf8] px-3.5 py-2.5 text-xs">
        <div className="flex items-center gap-3">
          <span className="font-medium text-black/75">Brush Size</span>
          <input
            type="range"
            min={3}
            max={14}
            value={brushSize}
            onChange={(e) => setBrushSize(Number(e.target.value))}
            className="h-1.5 w-28 accent-[#0a0a0b]"
          />
          <span className="font-mono text-[11px] text-black/55">
            {brushSize}px
          </span>
        </div>

        <span className="font-mono text-[11px] text-black/50">
          Align bottoms to red BASE line (y=96/128). Guides are never exported.
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        {([0, 1, 2] as const).map((idx) => (
          <SingleGlyphPad
            key={idx}
            index={idx}
            value={slots[idx]}
            brushSize={brushSize}
            onChange={(url) => onChangeSlot(idx, url)}
          />
        ))}
      </div>
    </div>
  );
}
