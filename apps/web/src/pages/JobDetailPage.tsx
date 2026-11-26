import { useCallback, useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  Download,
  Loader2,
  Type,
} from "lucide-react";
import {
  ASCII_94,
  type CurveMode,
  type FontJobWithUrls,
} from "@glyphcraft/shared";
import { FontRepackControls } from "../components/FontRepackControls";
import { useAuth } from "../hooks/useAuth";
import { useDynamicFontFace } from "../hooks/useDynamicFontFace";
import { getFontJob, repackFontJob } from "../lib/api";

interface JobDetailPageProps {
  onOpenAuth: () => void;
}

export function JobDetailPage({ onOpenAuth }: JobDetailPageProps) {
  const { id } = useParams<{ id: string }>();
  const { user, session, loading: authLoading } = useAuth();

  const [job, setJob] = useState<FontJobWithUrls | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [repacking, setRepacking] = useState(false);
  const [repackNotice, setRepackNotice] = useState<string | null>(null);

  const [testPhrase, setTestPhrase] = useState(
    "The quick brown fox jumps over 0123456789!"
  );
  const [testSize, setTestSize] = useState(48);

  const fontRevisionKey = `${job?.id ?? "none"}_${job?.r2_ttf_key ?? ""}_${job?.updated_at ?? ""}`;
  const { loadedFamily, loading: fontLoading, error: fontError } =
    useDynamicFontFace(job?.ttfUrl ?? null, fontRevisionKey);

  const fetchJob = useCallback(async () => {
    if (!id || !session?.access_token) return;
    try {
      const data = await getFontJob(id, session.access_token);
      setJob(data.job);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load font job.");
    } finally {
      setLoading(false);
    }
  }, [id, session?.access_token]);

  useEffect(() => {
    if (authLoading) return;
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    void fetchJob();
  }, [authLoading, session?.access_token, fetchJob]);

  useEffect(() => {
    if (!job || (job.status !== "queued" && job.status !== "running")) return;
    const timer = window.setInterval(() => {
      void fetchJob();
    }, 3000);
    return () => window.clearInterval(timer);
  }, [job, fetchJob]);

  const handleRepack = async (params: {
    familyName: string;
    threshold: number;
    sidebearing: number;
    curveMode: CurveMode;
  }) => {
    if (!id || !session?.access_token) return;
    setRepacking(true);
    setRepackNotice(null);
    setError(null);
    try {
      const res = await repackFontJob(id, params, session.access_token);
      setJob(res.job);
      setRepackNotice(
        `Updated "${res.job.family_name}" — weight ${res.job.ttf_params.threshold}, spacing ${res.job.ttf_params.sidebearing}, ${
          res.job.ttf_params.curve_mode === "polygon"
            ? "Crisp Edges"
            : "Smooth Curves"
        }.`
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Repack failed.");
    } finally {
      setRepacking(false);
    }
  };

  if (authLoading || loading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-7 w-7 animate-spin text-[#d94625]" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="mx-auto max-w-md px-4 py-20 text-center">
        <div className="rounded-2xl border border-black/15 bg-white p-8">
          <h1 className="font-display text-3xl text-[#0a0a0b]">
            Sign in required
          </h1>
          <p className="mt-2 text-sm text-black/60">
            Please sign in with the Google account used to purchase this font.
          </p>
          <button
            type="button"
            onClick={onOpenAuth}
            className="mt-5 rounded-xl bg-[#0a0a0b] px-5 py-2.5 text-xs font-semibold text-white"
          >
            Continue with Google
          </button>
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="mx-auto max-w-lg px-4 py-16">
        <div className="rounded-2xl border border-red-200 bg-white p-6 text-center">
          <p className="text-sm text-red-600">{error || "Font job not found."}</p>
          <Link
            to="/dashboard"
            className="mt-4 inline-flex items-center gap-1.5 text-xs font-semibold text-[#0a0a0b]"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to My Fonts
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto min-h-[75vh] max-w-7xl px-4 py-10 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-1.5 font-mono text-xs text-black/55 hover:text-[#0a0a0b]"
          >
            <ArrowLeft className="h-3.5 w-3.5" /> Back to My Fonts
          </Link>
          <h1 className="mt-2 font-display text-4xl text-[#0a0a0b]">
            {job.family_name}
          </h1>
        </div>

        {job.status === "completed" && job.ttfDownloadUrl && (
          <a
            href={job.ttfDownloadUrl}
            className="inline-flex items-center justify-center gap-2 rounded-xl bg-[#0a0a0b] px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-black/85"
          >
            <Download className="h-4 w-4 text-[#d94625]" />
            Download {job.family_name}.ttf
          </a>
        )}
      </div>

      {error && (
        <div className="mt-5 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-xs text-red-700">
          {error}
        </div>
      )}

      {repackNotice && (
        <div className="mt-5 flex items-center gap-2 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-xs text-emerald-800">
          <CheckCircle2 className="h-4 w-4 shrink-0 text-emerald-600" />
          {repackNotice}
        </div>
      )}

      {/* Pending / Failed States */}
      {(job.status === "queued" || job.status === "running") && (
        <div className="mt-8 rounded-2xl border border-black/15 bg-white p-10 text-center shadow-sm">
          <Loader2 className="mx-auto h-9 w-9 animate-spin text-[#d94625]" />
          <h2 className="mt-4 font-display text-3xl text-[#0a0a0b]">
            Generating your English font...
          </h2>
          <p className="mt-2 font-mono text-xs text-black/60">
            {job.progress_message}
          </p>
          <p className="mt-4 text-xs text-black/45">
            This page updates automatically every 3 seconds.
          </p>
        </div>
      )}

      {job.status === "failed" && (
        <div className="mt-8 rounded-2xl border border-red-200 bg-red-50 p-8 text-center">
          <AlertCircle className="mx-auto h-8 w-8 text-red-600" />
          <h2 className="mt-3 font-display text-2xl text-red-950">
            Generation failed
          </h2>
          <p className="mt-2 font-mono text-xs text-red-700">
            {job.error_message || "Unexpected inference failure."}
          </p>
        </div>
      )}

      {/* Completed Workbench: Live @font-face Typing Tester + Repack Tuner */}
      {job.status === "completed" && (
        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-12">
          <div className="space-y-6 lg:col-span-8">
            {/* Interactive @font-face live tester */}
            <div className="rounded-2xl border border-black/15 bg-white p-6 shadow-sm">
              <div className="flex flex-wrap items-center justify-between gap-3 border-b border-black/10 pb-4">
                <div className="flex items-center gap-2">
                  <Type className="h-4 w-4 text-[#d94625]" />
                  <span className="font-mono text-xs font-semibold uppercase tracking-wider text-black/70">
                    Live Font Tester
                  </span>
                </div>

                <label className="flex items-center gap-2 text-xs text-black/65">
                  <span className="font-mono">{testSize}px</span>
                  <input
                    type="range"
                    min={24}
                    max={88}
                    value={testSize}
                    onChange={(e) => setTestSize(Number(e.target.value))}
                    className="h-1.5 w-28 accent-[#0a0a0b]"
                  />
                </label>
              </div>

              {fontLoading && (
                <div className="mt-4 flex items-center gap-2 text-xs text-black/55">
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Loading TrueType font into browser...
                </div>
              )}
              {fontError && (
                <p className="mt-3 text-xs text-amber-700">
                  Browser preview couldn&apos;t load cross-origin ({fontError}).
                  You can still download the .ttf file directly.
                </p>
              )}

              <textarea
                value={testPhrase}
                onChange={(e) => setTestPhrase(e.target.value)}
                rows={3}
                spellCheck={false}
                style={{
                  fontFamily: loadedFamily
                    ? `"${loadedFamily}", sans-serif`
                    : "sans-serif",
                  fontSize: `${testSize}px`,
                  lineHeight: 1.35,
                }}
                className="mt-4 w-full resize-none bg-transparent text-[#0a0a0b] outline-none"
              />
            </div>

            {/* Full character map */}
            <div className="rounded-2xl border border-black/15 bg-white p-6 shadow-sm">
              <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-black/60">
                Full English Character Map
              </h3>
              <div
                style={{
                  fontFamily: loadedFamily
                    ? `"${loadedFamily}", sans-serif`
                    : "sans-serif",
                }}
                className="mt-4 grid grid-cols-8 gap-2 sm:grid-cols-12"
              >
                {Array.from(ASCII_94).map((ch) => (
                  <div
                    key={ch}
                    title={`U+${ch.charCodeAt(0).toString(16).toUpperCase().padStart(4, "0")}`}
                    className="flex aspect-square items-center justify-center rounded-lg border border-black/10 bg-[#fafaf8] text-2xl text-[#0a0a0b]"
                  >
                    {ch}
                  </div>
                ))}
              </div>
            </div>

            {job.previewUrl && (
              <div className="rounded-2xl border border-black/15 bg-white p-6 shadow-sm">
                <h3 className="font-mono text-xs font-semibold uppercase tracking-wider text-black/60">
                  Generated Specimen Sheet
                </h3>
                <img
                  src={job.previewUrl}
                  alt={`${job.family_name} full specimen`}
                  className="mt-4 w-full rounded-xl border border-black/10 bg-white"
                />
              </div>
            )}
          </div>

          <div className="lg:col-span-4">
            <FontRepackControls
              key={job.updated_at}
              initialFamilyName={job.family_name}
              initialParams={job.ttf_params}
              repacking={repacking}
              onRepack={handleRepack}
            />
          </div>
        </div>
      )}
    </div>
  );
}
