import { useEffect, useState } from "react";
import { BrowserRouter, Link, Route, Routes, useLocation } from "react-router-dom";
import { AuthModal } from "./components/AuthModal";
import { Footer } from "./components/Footer";
import { Navbar } from "./components/Navbar";
import { AuthProvider } from "./hooks/useAuth";
import { AuthCallbackPage } from "./pages/AuthCallbackPage";
import { DashboardPage } from "./pages/DashboardPage";
import { HomePage } from "./pages/HomePage";
import { JobDetailPage } from "./pages/JobDetailPage";
import { PrivacyPage } from "./pages/PrivacyPage";
import { TermsPage } from "./pages/TermsPage";

function ScrollToHash() {
  const { pathname, hash } = useLocation();

  useEffect(() => {
    if (!hash) return;
    const id = hash.slice(1);
    const timer = window.setTimeout(() => {
      const el = document.getElementById(id);
      if (el) {
        el.scrollIntoView({ behavior: "smooth" });
      }
    }, 50);
    return () => window.clearTimeout(timer);
  }, [pathname, hash]);

  return null;
}

export function App() {
  const [authModalOpen, setAuthModalOpen] = useState(false);

  return (
    <AuthProvider>
      <BrowserRouter>
        <ScrollToHash />
        <div className="flex min-h-screen flex-col bg-[#fafaf8] text-[#0a0a0b]">
          <Navbar onOpenAuth={() => setAuthModalOpen(true)} />
          <main className="flex-1">
            <Routes>
              <Route
                path="/"
                element={<HomePage onOpenAuth={() => setAuthModalOpen(true)} />}
              />
              <Route path="/auth/callback" element={<AuthCallbackPage />} />
              <Route
                path="/dashboard"
                element={
                  <DashboardPage onOpenAuth={() => setAuthModalOpen(true)} />
                }
              />
              <Route
                path="/jobs/:id"
                element={
                  <JobDetailPage onOpenAuth={() => setAuthModalOpen(true)} />
                }
              />
              <Route path="/terms" element={<TermsPage />} />
              <Route path="/privacy" element={<PrivacyPage />} />
              <Route
                path="*"
                element={
                  <div className="mx-auto max-w-md px-4 py-24 text-center">
                    <div className="rounded-2xl border border-black/15 bg-white p-8 shadow-sm">
                      <p className="font-mono text-xs uppercase tracking-widest text-[#d94625]">
                        404
                      </p>
                      <h1 className="mt-2 font-display text-3xl text-[#0a0a0b]">
                        Page not found
                      </h1>
                      <p className="mt-2 text-sm text-black/60">
                        The page you requested does not exist.
                      </p>
                      <Link
                        to="/"
                        className="mt-6 inline-flex items-center justify-center rounded-xl bg-[#0a0a0b] px-5 py-2.5 text-xs font-semibold text-white transition hover:bg-black/85"
                      >
                        Back to Home
                      </Link>
                    </div>
                  </div>
                }
              />
            </Routes>
          </main>
          <Footer />
          <AuthModal
            open={authModalOpen}
            onClose={() => setAuthModalOpen(false)}
          />
        </div>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
