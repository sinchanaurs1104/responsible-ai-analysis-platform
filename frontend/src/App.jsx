import { HashRouter, Routes, Route, Navigate } from "react-router-dom";
import { RunProvider } from "./api/RunContext";
import AppShell from "./components/AppShell";
import UploadPage from "./pages/UploadPage";
import ConfigurePage from "./pages/ConfigurePage";
import MitigatePage from "./pages/MitigatePage";
import VersionsPage from "./pages/VersionsPage";
import ComparePage from "./pages/ComparePage";

export default function App() {
  return (
    <RunProvider>
      <HashRouter>
        <AppShell>
          <Routes>
            <Route path="/" element={<Navigate to="/upload" replace />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/configure" element={<ConfigurePage />} />
            <Route path="/mitigate" element={<MitigatePage />} />
            <Route path="/versions" element={<VersionsPage />} />
            <Route path="/compare" element={<ComparePage />} />
          </Routes>
        </AppShell>
      </HashRouter>
    </RunProvider>
  );
}
