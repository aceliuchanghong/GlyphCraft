import type { CurveMode } from "@glyphcraft/shared";

export interface SampleFontSpec {
  id: string;
  label: string;
  category: string;
  fontFamily: string;
  ttfUrl: string;
  downloadFilename: string;
  curveMode: CurveMode;
  defaultText: string;
  presetSlug?: string;
}

export const SAMPLE_FONTS: SampleFontSpec[] = [
  {
    id: "velvet-swash",
    label: "Velvet Swash",
    category: "Editorial Swash Serif",
    fontFamily: "GC-VelvetSwash",
    ttfUrl: "/samples/velvet-swash.ttf",
    downloadFilename: "VelvetSwash.ttf",
    curveMode: "spline",
    defaultText: "Playful curls & crisp serifs!",
    presetSlug: "velvet-swash",
  },
  {
    id: "pixel-matrix",
    label: "Pixel Matrix",
    category: "Retro Pixel Grid",
    fontFamily: "GC-PixelMatrix",
    ttfUrl: "/samples/pixel-matrix.ttf",
    downloadFilename: "PixelMatrix.ttf",
    curveMode: "polygon",
    defaultText: "PIXEL GRID 0123456789",
    presetSlug: "pixel-matrix",
  },
  {
    id: "nordic-sans",
    label: "Nordic Sans",
    category: "Humanist Sans",
    fontFamily: "GC-NordicSans",
    ttfUrl: "/samples/nordic-sans.ttf",
    downloadFilename: "NordicSans.ttf",
    curveMode: "spline",
    defaultText: "Clean geometry, full ASCII.",
    presetSlug: "nordic-sans",
  },
  {
    id: "quill-serif",
    label: "Quill Serif",
    category: "Calligraphic Serif",
    fontFamily: "GC-QuillSerif",
    ttfUrl: "/samples/quill-serif.ttf",
    downloadFilename: "QuillSerif.ttf",
    curveMode: "spline",
    defaultText: "Sharp terminals & balance.",
  },
];

export interface StudioPreset {
  slug: string;
  name: string;
  tag: string;
  description: string;
  recommendedCurveMode: CurveMode;
  glyphs: [string, string, string];
}

export const STUDIO_PRESETS: StudioPreset[] = [
  {
    slug: "velvet-swash",
    name: "Velvet Swash",
    tag: "Swash Serif",
    description: "Expressive editorial serifs with playful curls",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/velvet-swash/H.png",
      "/presets/velvet-swash/F.png",
      "/presets/velvet-swash/y.png",
    ],
  },
  {
    slug: "marker-brush",
    name: "Marker Brush",
    tag: "Handwritten",
    description: "Bold hand-lettered felt-tip brush strokes",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/marker-brush/H.png",
      "/presets/marker-brush/F.png",
      "/presets/marker-brush/y.png",
    ],
  },
  {
    slug: "pixel-matrix",
    name: "Pixel Matrix",
    tag: "Pixel",
    description: "Crisp retro bitmap grid for games & UI",
    recommendedCurveMode: "polygon",
    glyphs: [
      "/presets/pixel-matrix/H.png",
      "/presets/pixel-matrix/F.png",
      "/presets/pixel-matrix/y.png",
    ],
  },
  {
    slug: "nordic-sans",
    name: "Nordic Sans",
    tag: "Clean Sans",
    description: "Balanced everyday humanist sans-serif",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/nordic-sans/H.png",
      "/presets/nordic-sans/F.png",
      "/presets/nordic-sans/y.png",
    ],
  },
];
