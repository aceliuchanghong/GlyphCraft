import { useEffect, useRef, useState } from "react";
import { ArrowDown } from "lucide-react";
import { HeroPlayground } from "../components/HeroPlayground";
import { StudioWorkbench } from "../components/StudioWorkbench";

interface HomePageProps {
  onOpenAuth: () => void;
}

const HERO_DEMOS = [
  {
    id: "quill",
    label: "Calligraphy Quill",
    src: "/video/feather-hero.mp4",
    poster: "/video/feather-poster.jpg",
    ariaLabel:
      "Hand writes H, g, p with a feather quill and the quill automatically finishes writing the alphabet in matching calligraphy style",
  },
  {
    id: "pen",
    label: "Modern Ink Pen",
    src: "/video/pen-hero.mp4",
    poster: "/video/pen-poster.jpg",
    ariaLabel:
      "Hand writes A, b, g with a matte-black pen and the pen automatically finishes writing the alphabet in the same bold style",
  },
] as const;

const FAQS = [
  {
    q: "What characters are included?",
    a: "Every generated .ttf font includes full standard English letters (A–Z, a–z), numbers (0–9), and common punctuation. Non-English scripts are not supported.",
  },
  {
    q: "How many sample letters do I need?",
    a: "Just 1 to 3 letters. Pick a starter style, handwrite a few letters (such as H, g, and y), or upload a style sample to give the AI enough cues for the full English font.",
  },
  {
    q: "Can I adjust thickness or spacing afterwards?",
    a: "Yes. After your font is generated, you can fine-tune stroke weight, letter spacing, and font name anytime for free.",
  },
  {
    q: "Can I use the .ttf font commercially?",
    a: "Yes. You get full commercial rights to install, embed, and use fonts generated from your own handwriting or licensed style samples.",
  },
];

export function HomePage({ onOpenAuth }: HomePageProps) {
  const [activeDemoIdx, setActiveDemoIdx] = useState(0);
  const [isHeroVisible, setIsHeroVisible] = useState(true);
  const [requestedPreset, setRequestedPreset] = useState<{
    slug: string;
    seq: number;
  } | null>(null);
  const heroMediaRef = useRef<HTMLDivElement | null>(null);
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([]);
  const prevDemoIdxRef = useRef<number>(0);

  const handleUsePresetInStudio = (slug: string) => {
    setRequestedPreset((prev) => ({ slug, seq: (prev?.seq ?? 0) + 1 }));
    document.getElementById("studio")?.scrollIntoView({ behavior: "smooth" });
  };

  const handleSelectDemo = (idx: number) => {
    if (idx === activeDemoIdx) {
      const activeVideo = videoRefs.current[idx];
      if (activeVideo) {
        activeVideo.currentTime = 0;
        const p = activeVideo.play();
        if (p) p.catch(() => {});
      }
      return;
    }
    setActiveDemoIdx(idx);
  };

  // Pause hero videos automatically when scrolled out of view to keep studio drawing & scrolling silky smooth.
  useEffect(() => {
    const container = heroMediaRef.current;
    if (!container || typeof IntersectionObserver === "undefined") return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        setIsHeroVisible(entry.isIntersecting);
      },
      { threshold: 0.15 }
    );
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  // Keep both <video> elements mounted and preloaded: switch via smooth opacity crossfade without DOM remounting.
  useEffect(() => {
    const switched = prevDemoIdxRef.current !== activeDemoIdx;
    prevDemoIdxRef.current = activeDemoIdx;

    videoRefs.current.forEach((video, idx) => {
      if (!video) return;
      if (idx === activeDemoIdx && isHeroVisible) {
        if (switched || video.ended) {
          video.currentTime = 0;
        }
        const playPromise = video.play();
        if (playPromise) {
          playPromise.catch(() => {
            // Ignore autoplay promise rejections in strict browser contexts.
          });
        }
      } else {
        video.pause();
      }
    });
  }, [activeDemoIdx, isHeroVisible]);

  return (
    <>
      {/* Clean, focused Hero with looping demo video */}
      <section className="border-b border-black/10 bg-foundry-grid py-12 sm:py-16">
        <div className="mx-auto grid max-w-7xl grid-cols-1 items-center gap-10 px-4 sm:px-6 lg:grid-cols-12 lg:px-8">
          <div className="text-center lg:col-span-6 lg:text-left">
            <h1 className="font-display text-[2.35rem] leading-[1.06] tracking-tight text-[#0a0a0b] sm:text-5xl xl:text-[3.75rem]">
              <span className="block">Start with 1–3 letters.</span>
              <span className="mt-1 block italic text-[#d94625]">
                Get a full English font.
              </span>
            </h1>

            <p className="mx-auto mt-4 max-w-xl text-base leading-relaxed text-black/65 lg:mx-0">
              Pick a starter style, handwrite a few letters, or upload a style
              sample. Preview for free, then export an installable{" "}
              <code className="rounded bg-black/5 px-1.5 py-0.5 font-mono text-sm text-[#0a0a0b]">
                .ttf
              </code>{" "}
              font.
            </p>

            <div className="mt-7 flex flex-wrap items-center justify-center gap-3 lg:justify-start">
              <a
                href="#studio"
                className="inline-flex items-center gap-2 rounded-xl bg-[#0a0a0b] px-6 py-3 text-sm font-semibold text-white shadow-xs transition hover:bg-black/85"
              >
                Start Creating
                <ArrowDown className="h-4 w-4 text-[#d94625]" />
              </a>
              <a
                href="#examples"
                className="inline-flex items-center rounded-xl border border-black/15 bg-white px-5 py-3 text-sm font-medium text-[#0a0a0b] transition hover:border-black/30"
              >
                Try Sample Fonts
              </a>
            </div>

            <p className="mt-5 font-mono text-xs text-black/45">
              Works on macOS · Windows · Figma · Photoshop · Canva
            </p>
          </div>

          <div className="lg:col-span-6">
            <div
              ref={heroMediaRef}
              className="relative aspect-video w-full overflow-hidden rounded-2xl border border-black/10 bg-[#f5f4f0] shadow-sm"
            >
              {HERO_DEMOS.map((demo, idx) => {
                const isActive = idx === activeDemoIdx;
                return (
                  <video
                    key={demo.id}
                    ref={(el) => {
                      videoRefs.current[idx] = el;
                    }}
                    src={demo.src}
                    poster={demo.poster}
                    autoPlay={idx === 0}
                    muted
                    playsInline
                    preload="auto"
                    onEnded={() => {
                      if (idx === activeDemoIdx) {
                        setActiveDemoIdx((prev) => (prev + 1) % HERO_DEMOS.length);
                      }
                    }}
                    aria-label={demo.ariaLabel}
                    aria-hidden={!isActive}
                    className={`absolute inset-0 block h-full w-full object-cover transition-opacity duration-500 will-change-[opacity] ${
                      isActive
                        ? "z-10 opacity-100"
                        : "pointer-events-none z-0 opacity-0"
                    }`}
                  />
                );
              })}
            </div>

            <div className="mt-3 flex items-center justify-center gap-2 lg:justify-end">
              {HERO_DEMOS.map((demo, idx) => {
                const isActive = idx === activeDemoIdx;
                return (
                  <button
                    key={demo.id}
                    type="button"
                    onClick={() => handleSelectDemo(idx)}
                    className={`rounded-full px-3 py-1 text-xs font-medium transition ${
                      isActive
                        ? "bg-[#0a0a0b] text-white"
                        : "border border-black/15 bg-white text-black/60 hover:border-black/30 hover:text-[#0a0a0b]"
                    }`}
                  >
                    {demo.label}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </section>

      {/* Core Interactive Studio right up front */}
      <StudioWorkbench
        onRequireAuth={onOpenAuth}
        requestedPreset={requestedPreset}
      />

      {/* Simple Sample Font Playground */}
      <HeroPlayground onUseInStudio={handleUsePresetInStudio} />

      {/* Compact FAQ */}
      <section className="bg-[#fafaf8] py-14">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <h2 className="font-display text-3xl text-[#0a0a0b]">
            Frequently asked questions
          </h2>
          <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2">
            {FAQS.map((faq) => (
              <div
                key={faq.q}
                className="rounded-xl border border-black/10 bg-white p-5"
              >
                <h3 className="text-sm font-semibold text-[#0a0a0b]">
                  {faq.q}
                </h3>
                <p className="mt-1.5 text-xs leading-relaxed text-black/65">
                  {faq.a}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>
    </>
  );
}
