import { useState } from "react";
import { Loader2, ShieldCheck, X } from "lucide-react";
import { BRAND_NAME } from "../config/brand";
import { useAuth } from "../hooks/useAuth";

interface AuthModalProps {
  open: boolean;
  onClose: () => void;
}

export function AuthModal({ open, onClose }: AuthModalProps) {
  const { signInWithGoogle, configured } = useAuth();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!open) return null;

  const handleGoogleSignIn = async () => {
    setError(null);
    setLoading(true);
    try {
      await signInWithGoogle("/auth/callback");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Failed to start Google sign-in."
      );
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 p-4 backdrop-blur-xs">
      <div className="relative w-full max-w-md rounded-2xl border border-black/15 bg-white p-6 shadow-2xl">
        <button
          type="button"
          onClick={onClose}
          aria-label="Close sign in dialog"
          className="absolute right-4 top-4 rounded-lg p-1.5 text-black/45 transition hover:bg-black/5 hover:text-[#0a0a0b]"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-[#0a0a0b] font-display text-2xl italic text-white">
          G
        </div>

        <h3 className="mt-4 font-display text-3xl text-[#0a0a0b]">
          Sign in to {BRAND_NAME}
        </h3>
        <p className="mt-2 text-sm leading-relaxed text-black/65">
          Sign in with Google to save your font library, receive download links by
          email, and continue to checkout.
        </p>

        <div className="mt-4 flex items-start gap-2.5 rounded-xl border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900">
          <ShieldCheck className="mt-0.5 h-4 w-4 shrink-0 text-emerald-600" />
          <span>
            Your current reference glyphs are preserved locally and in cloud
            draft storage across sign-in.
          </span>
        </div>

        {!configured && (
          <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3 text-xs text-amber-800">
            Supabase environment variables are not configured yet. Add{" "}
            <code className="font-mono">VITE_SUPABASE_URL</code> and{" "}
            <code className="font-mono">VITE_SUPABASE_ANON_KEY</code>.
          </div>
        )}

        {error && (
          <div className="mt-4 rounded-xl border border-red-200 bg-red-50 p-3 text-xs text-red-700">
            {error}
          </div>
        )}

        <button
          type="button"
          disabled={loading || !configured}
          onClick={() => void handleGoogleSignIn()}
          className="mt-5 inline-flex w-full items-center justify-center gap-2.5 rounded-xl bg-[#0a0a0b] px-4 py-3 text-sm font-semibold text-white transition hover:bg-black/85 disabled:cursor-not-allowed disabled:opacity-45"
        >
          {loading ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              Redirecting to Google...
            </>
          ) : (
            "Continue with Google"
          )}
        </button>
      </div>
    </div>
  );
}
