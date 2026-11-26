import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Loader2 } from "lucide-react";
import { sanitizeFamilyName } from "@glyphcraft/shared";
import { useAuth } from "../hooks/useAuth";
import { readStoredDraft, writeStoredDraft } from "../hooks/useDraftStorage";
import { createCheckoutSession, saveDraftToR2 } from "../lib/api";

export function AuthCallbackPage() {
  const { session, loading } = useAuth();
  const navigate = useNavigate();
  const [statusMessage, setStatusMessage] = useState(
    "Completing Google sign-in..."
  );
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (loading) return;

    if (!session?.access_token) {
      navigate("/#studio", { replace: true });
      return;
    }

    const draft = readStoredDraft();
    const activeImages = draft.slots.filter((s): s is string => Boolean(s));

    if (!draft.pendingCheckout || activeImages.length === 0) {
      navigate("/dashboard", { replace: true });
      return;
    }

    let cancelled = false;

    const resumeCheckout = async () => {
      try {
        setStatusMessage("Restoring your glyph draft & opening checkout...");
        let draftId = draft.r2DraftId;
        if (!draftId) {
          const saved = await saveDraftToR2({ images: activeImages });
          draftId = saved.draftId;
        }

        const checkout = await createCheckoutSession(
          {
            draftId,
            familyName: sanitizeFamilyName(draft.familyName),
            curveMode: draft.curveMode,
          },
          session.access_token
        );

        writeStoredDraft({
          ...draft,
          r2DraftId: draftId,
          pendingCheckout: false,
        });

        if (!cancelled) {
          window.location.href = checkout.checkoutUrl;
        }
      } catch (err) {
        writeStoredDraft({
          ...draft,
          pendingCheckout: false,
        });
        if (!cancelled) {
          setError(
            err instanceof Error
              ? err.message
              : "Could not resume checkout. Your glyph draft is still saved."
          );
        }
      }
    };

    void resumeCheckout();

    return () => {
      cancelled = true;
    };
  }, [loading, session, navigate]);

  return (
    <div className="mx-auto flex min-h-[60vh] max-w-md flex-col items-center justify-center px-4 text-center">
      {error ? (
        <div className="w-full rounded-2xl border border-red-200 bg-white p-6 shadow-sm">
          <h1 className="font-display text-2xl text-[#0a0a0b]">
            Checkout Redirect Interrupted
          </h1>
          <p className="mt-2 text-xs text-red-600">{error}</p>
          <button
            type="button"
            onClick={() => navigate("/#studio")}
            className="mt-5 w-full rounded-xl bg-[#0a0a0b] px-4 py-2.5 text-xs font-semibold text-white"
          >
            Return to Studio (Draft Preserved)
          </button>
        </div>
      ) : (
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-7 w-7 animate-spin text-[#d94625]" />
          <p className="text-sm font-medium text-[#0a0a0b]">{statusMessage}</p>
        </div>
      )}
    </div>
  );
}
