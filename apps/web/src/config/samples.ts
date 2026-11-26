import type { CurveMode } from "@glyphcraft/shared";

export interface SampleFontSpec {
  id: string;
  label: string;
  category: string;
  fontFamily: string;
  ttfUrl: string;
  curveMode: CurveMode;
  defaultText: string;
  presetSlug?: string;
}

export const SAMPLE_FONTS: SampleFontSpec[] = [
  {
    id: "grenze",
    label: "Grenze Gotisch",
    category: "Blackletter Display",
    fontFamily: "GC-GrenzeGotisch",
    ttfUrl: "/samples/grenzegotisch__light_w300_hd10.ttf",
    curveMode: "spline",
    defaultText: "Crafted from 3 strokes.",
    presetSlug: "grenzegotisch__light_w300",
  },
  {
    id: "cherry",
    label: "Cherry Swash",
    category: "Editorial Swash Serif",
    fontFamily: "GC-CherrySwash",
    ttfUrl: "/samples/cherryswash__bold_hd10.ttf",
    curveMode: "spline",
    defaultText: "Playful curls & crisp serifs!",
    presetSlug: "cherryswash__bold",
  },
  {
    id: "dotgothic",
    label: "DotGothic 16",
    category: "Pixel Matrix",
    fontFamily: "GC-DotGothic",
    ttfUrl: "/samples/dotgothic16__regular_hd10.ttf",
    curveMode: "polygon",
    defaultText: "PIXEL GRID 0123456789",
    presetSlug: "dotgothic16__regular",
  },
  {
    id: "ubuntu",
    label: "Ubuntu Sans",
    category: "Humanist Sans",
    fontFamily: "GC-Ubuntu",
    ttfUrl: "/samples/ubuntu__regular_hd10.ttf",
    curveMode: "spline",
    defaultText: "Clean geometry, full ASCII.",
    presetSlug: "ubuntu__regular",
  },
  {
    id: "abyssinica",
    label: "Abyssinica",
    category: "Calligraphic Serif",
    fontFamily: "GC-Abyssinica",
    ttfUrl: "/samples/abyssinicasil__regular_hd10.ttf",
    curveMode: "spline",
    defaultText: "Sharp terminals & balance.",
  },
];

export interface StudioPreset {
  slug: string;
  name: string;
  tag: string;
  recommendedCurveMode: CurveMode;
  glyphs: [string, string, string];
}

export const STUDIO_PRESETS: StudioPreset[] = [
  {
    slug: "grenzegotisch__light_w300",
    name: "Blackletter",
    tag: "Gothic",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/grenzegotisch__light_w300/H.png",
      "/presets/grenzegotisch__light_w300/F.png",
      "/presets/grenzegotisch__light_w300/y.png",
    ],
  },
  {
    slug: "cherryswash__bold",
    name: "Cherry Swash",
    tag: "Swash Serif",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/cherryswash__bold/H.png",
      "/presets/cherryswash__bold/F.png",
      "/presets/cherryswash__bold/y.png",
    ],
  },
  {
    slug: "dotgothic16__regular",
    name: "DotGothic",
    tag: "Pixel",
    recommendedCurveMode: "polygon",
    glyphs: [
      "/presets/dotgothic16__regular/H.png",
      "/presets/dotgothic16__regular/F.png",
      "/presets/dotgothic16__regular/y.png",
    ],
  },
  {
    slug: "permanentmarker__regular",
    name: "Marker Brush",
    tag: "Handwriting",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/permanentmarker__regular/H.png",
      "/presets/permanentmarker__regular/F.png",
      "/presets/permanentmarker__regular/y.png",
    ],
  },
  {
    slug: "ubuntu__regular",
    name: "Ubuntu Sans",
    tag: "Sans",
    recommendedCurveMode: "spline",
    glyphs: [
      "/presets/ubuntu__regular/H.png",
      "/presets/ubuntu__regular/F.png",
      "/presets/ubuntu__regular/y.png",
    ],
  },
];
