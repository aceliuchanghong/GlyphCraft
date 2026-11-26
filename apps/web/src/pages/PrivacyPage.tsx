import { BRAND_CONFIG, BRAND_NAME } from "../config/brand";

export function PrivacyPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-14 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
        Legal
      </p>
      <h1 className="mt-2 font-display text-4xl text-[#0a0a0b]">
        Privacy Policy
      </h1>
      <div className="mt-6 space-y-5 text-sm leading-relaxed text-black/75">
        <p>
          {BRAND_NAME} respects your privacy and minimizes data retention across
          both free previews and paid font generation.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          1. Free Studio Previews (Zero Storage)
        </h2>
        <p>
          Free preview requests are processed synchronously in memory and
          returned directly to your browser. We do not store preview images in
          databases or object storage. Only a daily IP rate-limit counter is
          retained temporarily in Cloudflare KV to prevent abuse.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          2. Account &amp; Paid Font Storage
        </h2>
        <p>
          When you sign in with Google via Supabase Auth and unlock a full
          94-ASCII font, we store your email address, order reference, draft
          glyphs, generated <code className="font-mono">.ttf</code>, and
          vectorizable glyph archive in Cloudflare R2 so you can re-tune and
          download your fonts anytime.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          3. Third-Party Processors
        </h2>
        <p>
          We rely on Cloudflare (hosting, Turnstile bot protection, R2 storage),
          Supabase (authentication &amp; database), Modal (GPU/CPU inference),
          Creem (checkout &amp; tax compliance), and Resend (delivery emails).
          For data deletion requests, email{" "}
          <a
            href={`mailto:${BRAND_CONFIG.supportEmail}`}
            className="underline"
          >
            {BRAND_CONFIG.supportEmail}
          </a>
          .
        </p>
      </div>
    </div>
  );
}
