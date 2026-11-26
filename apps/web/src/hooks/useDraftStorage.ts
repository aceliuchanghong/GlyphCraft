import { useCallback, useEffect, useState } from "react";
import type { CurveMode } from "@glyphcraft/shared";

const STORAGE_KEY = "glyphcraft_studio_draft_v1";

export interface StudioDraftState {
  slots: [string | null, string | null, string | null];
  previewText: string;
  familyName: string;
  curveMode: CurveMode;
  r2DraftId: string | null;
  pendingCheckout: boolean;
}

const DEFAULT_DRAFT: StudioDraftState = {
  slots: [null, null, null],
  previewText: "Hello World!",
  familyName: "MyCustomFont",
  curveMode: "spline",
  r2DraftId: null,
  pendingCheckout: false,
};

export function readStoredDraft(): StudioDraftState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_DRAFT;
    const parsed = JSON.parse(raw) as Partial<StudioDraftState>;
    const slots: [string | null, string | null, string | null] = [
      Array.isArray(parsed.slots) ? parsed.slots[0] ?? null : null,
      Array.isArray(parsed.slots) ? parsed.slots[1] ?? null : null,
      Array.isArray(parsed.slots) ? parsed.slots[2] ?? null : null,
    ];
    return {
      slots,
      previewText:
        typeof parsed.previewText === "string"
          ? parsed.previewText
          : DEFAULT_DRAFT.previewText,
      familyName:
        typeof parsed.familyName === "string"
          ? parsed.familyName
          : DEFAULT_DRAFT.familyName,
      curveMode: parsed.curveMode === "polygon" ? "polygon" : "spline",
      r2DraftId: typeof parsed.r2DraftId === "string" ? parsed.r2DraftId : null,
      pendingCheckout: Boolean(parsed.pendingCheckout),
    };
  } catch {
    return DEFAULT_DRAFT;
  }
}

export function writeStoredDraft(next: StudioDraftState): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  } catch {
    // Ignore storage quota errors cleanly
  }
}

export function useDraftStorage() {
  const [draft, setDraft] = useState<StudioDraftState>(() => readStoredDraft());

  useEffect(() => {
    writeStoredDraft(draft);
  }, [draft]);

  const setSlotImage = useCallback((index: 0 | 1 | 2, dataUrl: string | null) => {
    setDraft((prev) => {
      const nextSlots: [string | null, string | null, string | null] = [...prev.slots];
      nextSlots[index] = dataUrl;
      return {
        ...prev,
        slots: nextSlots,
        r2DraftId: null,
      };
    });
  }, []);

  const setAllSlots = useCallback(
    (
      slots: [string | null, string | null, string | null],
      extra?: Partial<Pick<StudioDraftState, "curveMode" | "familyName">>
    ) => {
      setDraft((prev) => ({
        ...prev,
        ...extra,
        slots,
        r2DraftId: null,
      }));
    },
    []
  );

  const updateFields = useCallback(
    (patch: Partial<Omit<StudioDraftState, "slots">>) => {
      setDraft((prev) => ({ ...prev, ...patch }));
    },
    []
  );

  const clearPendingCheckout = useCallback(() => {
    setDraft((prev) => ({ ...prev, pendingCheckout: false }));
  }, []);

  return {
    draft,
    setSlotImage,
    setAllSlots,
    updateFields,
    clearPendingCheckout,
  };
}
