import { Link, useLocation } from "react-router-dom";
import { LogOut, Sparkles, User } from "lucide-react";
import { BRAND_NAME } from "../config/brand";
import { useAuth } from "../hooks/useAuth";

interface NavbarProps {
  onOpenAuth: () => void;
}

export function Navbar({ onOpenAuth }: NavbarProps) {
  const { user, signOut } = useAuth();
  const location = useLocation();
  const isHome = location.pathname === "/";

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
          <span className="hidden rounded-full border border-black/10 bg-white px-2 py-0.5 font-mono text-[10px] uppercase tracking-wider text-black/60 sm:inline-block">
            94-ASCII
          </span>
        </Link>

        <nav className="hidden items-center gap-7 text-sm font-medium text-black/65 md:flex">
          <a
            href={isHome ? "#playground" : "/#playground"}
            className="transition-colors hover:text-[#0a0a0b]"
          >
            Playground
          </a>
          <a
            href={isHome ? "#studio" : "/#studio"}
            className="transition-colors hover:text-[#0a0a0b]"
          >
            Studio
          </a>
          <a
            href={isHome ? "#pricing" : "/#pricing"}
            className="transition-colors hover:text-[#0a0a0b]"
          >
            Pricing
          </a>
          <Link
            to="/dashboard"
            className="transition-colors hover:text-[#0a0a0b]"
          >
            My Fonts
          </Link>
        </nav>

        <div className="flex items-center gap-2.5">
          {user ? (
            <>
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 rounded-lg border border-black/15 bg-white px-3 py-1.5 text-xs font-medium text-[#0a0a0b] transition hover:border-black/30"
              >
                <User className="h-3.5 w-3.5 text-[#d94625]" />
                <span className="max-w-[140px] truncate">
                  {user.email || "Dashboard"}
                </span>
              </Link>
              <button
                type="button"
                onClick={() => void signOut()}
                title="Sign out"
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
          <a
            href={isHome ? "#studio" : "/#studio"}
            className="inline-flex items-center gap-1.5 rounded-lg bg-[#0a0a0b] px-3.5 py-1.5 text-xs font-semibold text-white transition hover:bg-black/85"
          >
            <Sparkles className="h-3.5 w-3.5 text-[#d94625]" />
            Try Free
          </a>
        </div>
      </div>
    </header>
  );
}
