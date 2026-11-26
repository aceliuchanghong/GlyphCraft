import { useCallback, useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowRight,
  CheckCircle2,
  Clock,
  Download,
  Loader2,
  Plus,
  RefreshCw,
} from "lucide-react";
import type { FontJobWithUrls } from "@glyphcraft/shared";
import { useAuth } from "../hooks/useAuth";
import { listFontJobs } from "../lib/api";

interface DashboardPageProps {
  onOpenAuth: () => void;
}

export function DashboardPage({ onOpenAuth }: DashboardPageProps) {
  const { user, session, loading: authLoading } = useAuth();
  const [searchParams] = useSearchParams();
  const checkoutSuccess = searchParams.get("checkout") === "success";

  const [jobs, setJobs] = useState<FontJobWithUrls[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchJobs = useCallback(async (manual = false) => {
    if (!session?.access_token) return;
    if (manual) setRefreshing(true);
    try {
      const data = await listFontJobs(session.access_token);
      setJobs(data.jobs);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load fonts.");
    } finally {
      setLoading(false);
      if (manual) setRefreshing(false);
    }
  }, [session?.access_token]);

  useEffect(() => {
    if (authLoading) return;
    if (!session?.access_token) {
      setLoading(false);
      return;
    }
    void fetchJobs();
  }, [authLoading, session?.access_token, fetchJobs]);

  // Poll automatically while any job is queued/running (or right after checkout)
  useEffect(() => {
    if (!session?.access_token) return;
    const hasPending =
      jobs.some((j) => j.status === "queued" || j.status === "running") ||
      (checkoutSuccess && jobs.length === 0);
    if (!hasPending) return;

    const timer = window.setInterval(() => {
      void fetchJobs();
    }, 3500);
    return () => window.clearInterval(timer);
  }, [jobs, checkoutSuccess, session?.access_token, fetchJobs]);

  if (authLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Loader2 className="h-6 w-6 animate-spin text-[#d94625]" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="mx-auto max-w-lg px-4 py-20 text-center">
        <div className="rounded-2xl border border-black/15 bg-white p-8 shadow-sm">
          <h1 className="font-display text-3xl text-[#0a0a0b]">
            Sign in to view your fonts
          </h1>
          <p className="mt-2 text-sm text-black/60">
            Access your generated English{" "}
            <code className="font-mono">.ttf</code> files, live specimen
            previews, and free parameter tuning controls.
          </p>
          <button
            type="button"
            onClick={onOpenAuth}
            className="mt-6 rounded-xl bg-[#0a0a0b] px-6 py-3 text-sm font-semibold text-white transition hover:bg-black/85"
          >
            Continue with Google
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mx-auto min-h-[70vh] max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
            Font Library
          </p>
          <h1 className="mt-1 font-display text-4xl text-[#0a0a0b]">
            My Generated Fonts
          </h1>
        </div>

        <div className="flex items-center gap-2.5">
          <button
            type="button"
            disabled={refreshing}
            onClick={() => void fetchJobs(true)}
            className="inline-flex items-center gap-1.5 rounded-xl border border-black/15 bg-white px-3.5 py-2 text-xs font-semibold text-[#0a0a0b] transition hover:border-black/35 disabled:opacity-60"
          >
            <RefreshCw
              className={`h-3.5 w-3.5 ${refreshing ? "animate-spin" : ""}`}
            />
            Refresh
          </button>
          <Link
            to="/#studio"
            className="inline-flex items-center gap-1.5 rounded-xl bg-[#0a0a0b] px-4 py-2 text-xs font-semibold text-white transition hover:bg-black/85"
          >
            <Plus className="h-3.5 w-3.5 text-[#d94625]" />
            New Font
          </Link>
        </div>
      </div>

      {checkoutSuccess && (
        <div className="mt-6 flex items-start gap-3 rounded-2xl border border-emerald-200 bg-emerald-50 p-4 text-sm text-emerald-900">
          <CheckCircle2 className="mt-0.5 h-5 w-5 shrink-0 text-emerald-600" />
          <div>
            <p className="font-semibold">
              Payment received! Your English font is being generated.
            </p>
            <p className="mt-0.5 text-xs text-emerald-800">
              Generation usually finishes in ~30–45 seconds. We will also email
              you at <span className="font-mono">{user.email}</span> as soon as
              your <code className="font-mono">.ttf</code> is ready.
            </p>
          </div>
        </div>
      )}

      {error && (
        <div className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-xs text-red-700">
          {error}
        </div>
      )}

      {loading ? (
        <div className="mt-12 flex items-center justify-center py-12">
          <Loader2 className="h-6 w-6 animate-spin text-[#d94625]" />
        </div>
      ) : jobs.length === 0 ? (
        <div className="mt-10 rounded-2xl border border-dashed border-black/20 bg-white p-12 text-center">
          <h2 className="font-display text-2xl text-[#0a0a0b]">
            No fonts generated yet
          </h2>
          <p className="mt-2 text-sm text-black/60">
            Head to the Studio, test your 1–3 letter style for free, and export
            your first English <code className="font-mono">.ttf</code> font.
          </p>
          <Link
            to="/#studio"
            className="mt-5 inline-flex items-center gap-2 rounded-xl bg-[#0a0a0b] px-5 py-2.5 text-xs font-semibold text-white"
          >
            Open Studio <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
      ) : (
        <div className="mt-8 grid grid-cols-1 gap-4 md:grid-cols-2">
          {jobs.map((job) => (
            <div
              key={job.id}
              className="flex flex-col justify-between rounded-2xl border border-black/15 bg-white p-5 shadow-xs transition hover:border-black/30"
            >
              <div>
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-lg font-semibold text-[#0a0a0b]">
                      {job.family_name}
                    </h2>
                    <p className="font-mono text-[11px] text-black/45">
                      {new Date(job.created_at).toLocaleString()} ·{" "}
                      {job.ttf_params.curve_mode === "polygon"
                        ? "Crisp Edges"
                        : "Smooth Curves"}
                    </p>
                  </div>

                  <StatusBadge status={job.status} />
                </div>

                <p className="mt-3 text-xs text-black/65">
                  {job.error_message || job.progress_message}
                </p>

                {job.previewUrl && (
                  <div className="mt-4 overflow-hidden rounded-xl border border-black/10 bg-[#fafaf8] p-2">
                    <img
                      src={job.previewUrl}
                      alt={`${job.family_name} preview sheet`}
                      className="max-h-32 w-full object-contain"
                    />
                  </div>
                )}
              </div>

              <div className="mt-5 flex items-center justify-between gap-2 border-t border-black/10 pt-4">
                <Link
                  to={`/jobs/${job.id}`}
                  className="inline-flex items-center gap-1.5 text-xs font-semibold text-[#0a0a0b] hover:text-[#d94625]"
                >
                  Open Live Tester &amp; Tuner
                  <ArrowRight className="h-3.5 w-3.5" />
                </Link>

                {job.status === "completed" && job.ttfDownloadUrl && (
                  <a
                    href={job.ttfDownloadUrl}
                    className="inline-flex items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-black/85"
                  >
                    <Download className="h-3.5 w-3.5" />
                    .TTF
                  </a>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: FontJobWithUrls["status"] }) {
  if (status === "completed") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2.5 py-1 font-mono text-[11px] font-medium text-emerald-700">
        <CheckCircle2 className="h-3 w-3" /> Completed
      </span>
    );
  }
  if (status === "failed") {
    return (
      <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2.5 py-1 font-mono text-[11px] font-medium text-red-700">
        <AlertCircle className="h-3 w-3" /> Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2.5 py-1 font-mono text-[11px] font-medium text-amber-800">
      <Clock className="h-3 w-3 animate-pulse" />{" "}
      {status === "running" ? "Generating" : "Queued"}
    </span>
  );
}
