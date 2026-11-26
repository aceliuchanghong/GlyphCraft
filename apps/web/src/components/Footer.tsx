import { Link } from "react-router-dom";
import { BRAND_CONFIG, BRAND_NAME } from "../config/brand";

export function Footer() {
  return (
    <footer className="border-t border-black/10 bg-white">
      <div className="mx-auto flex max-w-7xl flex-col gap-6 px-4 py-10 sm:flex-row sm:items-center sm:justify-between sm:px-6 lg:px-8">
        <div>
          <div className="flex items-center gap-2">
            <span className="flex h-6 w-6 items-center justify-center rounded bg-[#0a0a0b] font-display text-sm italic text-white">
              G
            </span>
            <span className="text-sm font-semibold text-[#0a0a0b]">
              {BRAND_NAME}
            </span>
          </div>
          <p className="mt-1.5 text-xs text-black/55">
            {BRAND_CONFIG.tagline}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-6 text-xs text-black/60">
          <Link to="/dashboard" className="hover:text-[#0a0a0b]">
            My Fonts
          </Link>
          <Link to="/terms" className="hover:text-[#0a0a0b]">
            Terms &amp; License
          </Link>
          <Link to="/privacy" className="hover:text-[#0a0a0b]">
            Privacy Policy
          </Link>
          <a
            href={`mailto:${BRAND_CONFIG.supportEmail}`}
            className="hover:text-[#0a0a0b]"
          >
            {BRAND_CONFIG.supportEmail}
          </a>
        </div>
      </div>
    </footer>
  );
}
