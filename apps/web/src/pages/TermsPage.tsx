import { BRAND_CONFIG, BRAND_NAME } from "../config/brand";

export function TermsPage() {
  return (
    <div className="mx-auto max-w-3xl px-4 py-14 sm:px-6">
      <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
        Legal
      </p>
      <h1 className="mt-2 font-display text-4xl text-[#0a0a0b]">
        Terms of Service &amp; Font License
      </h1>
      <div className="mt-6 space-y-5 text-sm leading-relaxed text-black/75">
        <p>
          Welcome to {BRAND_NAME}. By generating or downloading font files
          through our platform, you agree to these terms.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          1. Supported Character Set (94 Printable English ASCII)
        </h2>
        <p>
          {BRAND_NAME} exclusively synthesizes the 94 printable English ASCII
          characters (<code className="font-mono">0x21 '!'</code> through{" "}
          <code className="font-mono">0x7E '~'</code>, covering A–Z, a–z, 0–9,
          and standard punctuation) plus the space glyph. Non-English scripts
          and extended Unicode blocks are not included.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          2. Reference Input Responsibility &amp; Commercial Rights
        </h2>
        <p>
          You represent that any hand-drawn strokes, images, or reference{" "}
          <code className="font-mono">.ttf</code> files you upload do not
          infringe third-party copyrights or trademarks. Subject to that
          condition, you receive a perpetual, worldwide, royalty-free commercial
          license to use, embed, and distribute the{" "}
          <code className="font-mono">.ttf</code> fonts generated for your
          account.
        </p>
        <h2 className="text-base font-semibold text-[#0a0a0b]">
          3. Payments &amp; Refunds
        </h2>
        <p>
          Payments are processed via Creem. Because GPU inference incurs
          immediate compute costs, completed font generations are generally
          non-refundable unless a technical failure prevents delivery of a
          valid <code className="font-mono">.ttf</code> file. Contact{" "}
          <a
            href={`mailto:${BRAND_CONFIG.supportEmail}`}
            className="underline"
          >
            {BRAND_CONFIG.supportEmail}
          </a>{" "}
          for billing support.
        </p>
      </div>
    </div>
  );
}
