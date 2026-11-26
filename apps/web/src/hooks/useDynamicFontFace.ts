import { useEffect, useState } from "react";

export function useDynamicFontFace(fontUrl: string | null, aliasSeed: string) {
  const [loadedFamily, setLoadedFamily] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!fontUrl) {
      setLoadedFamily(null);
      setLoading(false);
      setError(null);
      return;
    }

    let cancelled = false;
    let activeFace: FontFace | null = null;
    const uniqueFamily = `GCJob_${aliasSeed.replace(/[^A-Za-z0-9]/g, "")}_${Date.now()}`;

    setLoading(true);
    setError(null);

    const face = new FontFace(uniqueFamily, `url("${fontUrl}")`);
    face
      .load()
      .then((loaded) => {
        if (cancelled) return;
        document.fonts.add(loaded);
        activeFace = loaded;
        setLoadedFamily(uniqueFamily);
        setLoading(false);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load .ttf font.");
        setLoading(false);
      });

    return () => {
      cancelled = true;
      if (activeFace) {
        document.fonts.delete(activeFace);
      }
    };
  }, [fontUrl, aliasSeed]);

  return { loadedFamily, loading, error };
}
