import { useState } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
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

export function App() {
  const [authModalOpen, setAuthModalOpen] = useState(false);

  return (
    <AuthProvider>
      <BrowserRouter>
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
