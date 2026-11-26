import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { LogOut, Menu, Sparkles, User, X } from "lucide-react";
import { BRAND_NAME } from "../config/brand";
import { useAuth } from "../hooks/useAuth";

interface NavbarProps {
  onOpenAuth: () => void;
}

export function Navbar({ onOpenAuth }: NavbarProps) {
  const { user, signOut } = useAuth();
  const location = useLocation();
  const isHome = location.pathname === "/";
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useEffect(() => {
    setMobileMenuOpen(false);
  }, [location.pathname, location.hash]);

  return (
    <header className="sticky top-0 z-40 border-b border-black/10 bg-[#fafaf8]/90 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
        <Link to="/" className="flex items-center gap-2.5">
          <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-[#0a0a0b] font-display text-xl italic text-white">
            G
          </span>
          <span className="text-base font-semibold tracking-tight text-[#0a0a0b]">
            {BRAND_NAME}
          </span>
        </Link>

        <nav className="hidden items-center gap-7 text-sm font-medium text-black/65 md:flex">
          {isHome ? (
            <a
              href="#studio"
              className="transition-colors hover:text-[#0a0a0b]"
            >
              Studio
            </a>
          ) : (
            <Link
              to="/#studio"
              className="transition-colors hover:text-[#0a0a0b]"
            >
              Studio
            </Link>
          )}
          {isHome ? (
            <a
              href="#examples"
              className="transition-colors hover:text-[#0a0a0b]"
            >
              Examples
            </a>
          ) : (
            <Link
              to="/#examples"
              className="transition-colors hover:text-[#0a0a0b]"
            >
              Examples
            </Link>
          )}
          <Link
            to="/dashboard"
            className="transition-colors hover:text-[#0a0a0b]"
          >
            My Fonts
          </Link>
        </nav>

        <div className="flex items-center gap-2">
          {user ? (
            <>
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 rounded-lg border border-black/15 bg-white px-3 py-1.5 text-xs font-medium text-[#0a0a0b] transition hover:border-black/30"
              >
                <User className="h-3.5 w-3.5 text-[#d94625]" />
                <span className="max-w-[110px] truncate sm:max-w-[140px]">
                  {user.email || "Dashboard"}
                </span>
              </Link>
              <button
                type="button"
                onClick={() => void signOut()}
                title="Sign out"
                aria-label="Sign out"
                className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-black/10 text-black/55 transition hover:border-black/25 hover:text-[#0a0a0b]"
              >
                <LogOut className="h-3.5 w-3.5" />
              </button>
            </>
          ) : (
            <button
              type="button"
              onClick={onOpenAuth}
              className="inline-flex items-center gap-1.5 rounded-lg border border-black/15 bg-white px-3.5 py-1.5 text-xs font-semibold text-[#0a0a0b] transition hover:border-black/35"
            >
              Sign In
            </button>
          )}
          {isHome ? (
            <a
              href="#studio"
              className="hidden items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-black/85 sm:inline-flex"
            >
              <Sparkles className="h-3.5 w-3.5 text-[#d94625]" />
              Try Free
            </a>
          ) : (
            <Link
              to="/#studio"
              className="hidden items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-black/85 sm:inline-flex"
            >
              <Sparkles className="h-3.5 w-3.5 text-[#d94625]" />
              Try Free
            </Link>
          )}

          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileMenuOpen}
            className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-black/10 bg-white text-[#0a0a0b] transition hover:border-black/25 md:hidden"
          >
            {mobileMenuOpen ? (
              <X className="h-4 w-4" />
            ) : (
              <Menu className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>

      {mobileMenuOpen && (
        <div className="border-t border-black/10 bg-[#fafaf8] px-4 py-3 md:hidden">
          <div className="flex flex-col gap-1 text-sm font-medium text-black/75">
            {isHome ? (
              <a
                href="#studio"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-lg px-3 py-2 hover:bg-black/5 hover:text-[#0a0a0b]"
              >
                Studio
              </a>
            ) : (
              <Link
                to="/#studio"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-lg px-3 py-2 hover:bg-black/5 hover:text-[#0a0a0b]"
              >
                Studio
              </Link>
            )}
            {isHome ? (
              <a
                href="#examples"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-lg px-3 py-2 hover:bg-black/5 hover:text-[#0a0a0b]"
              >
                Examples
              </a>
            ) : (
              <Link
                to="/#examples"
                onClick={() => setMobileMenuOpen(false)}
                className="rounded-lg px-3 py-2 hover:bg-black/5 hover:text-[#0a0a0b]"
              >
                Examples
              </Link>
            )}
            <Link
              to="/dashboard"
              onClick={() => setMobileMenuOpen(false)}
              className="rounded-lg px-3 py-2 hover:bg-black/5 hover:text-[#0a0a0b]"
            >
              My Fonts
            </Link>
          </div>
        </div>
      )}
    </header>
  );
}
