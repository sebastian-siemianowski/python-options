import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';
import ToastContainer from './components/ToastContainer';
import { ToastProvider } from './stores/toastStore';
import OverviewPage from './pages/OverviewPage';
import SignalsPage from './pages/SignalsPage';
import RiskPage from './pages/RiskPage';
import ChartsPage from './pages/ChartsPage';
import TuningPage from './pages/TuningPage';
import DataPage from './pages/DataPage';
import ArenaPage from './pages/ArenaPage';
import ServicesPage from './pages/ServicesPage';
import DiagnosticsPage from './pages/DiagnosticsPage';
import ProfitabilityPage from './pages/ProfitabilityPage';
import HeatmapPage from './pages/HeatmapPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2 * 60_000,     // 2 minutes - stale-while-revalidate
      gcTime: 10 * 60_000,       // 10 minutes garbage collection
      refetchOnWindowFocus: true, // Background refresh on tab return
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ToastProvider>
        <BrowserRouter>
          <Routes>
            <Route element={<Layout />}>
              <Route path="/" element={<OverviewPage />} />
              <Route path="/heatmap" element={<HeatmapPage />} />
              <Route path="/signals" element={<SignalsPage />} />
              <Route path="/risk" element={<RiskPage />} />
              <Route path="/charts" element={<ChartsPage />} />
              <Route path="/charts/:symbol" element={<ChartsPage />} />
              <Route path="/tuning" element={<TuningPage />} />
              <Route path="/data" element={<DataPage />} />
              <Route path="/arena" element={<ArenaPage />} />
              <Route path="/diagnostics" element={<DiagnosticsPage />} />
              <Route path="/diagnostics/profitability" element={<ProfitabilityPage />} />
              <Route path="/services" element={<ServicesPage />} />
            </Route>
          </Routes>
          <ToastContainer />
        </BrowserRouter>
      </ToastProvider>
    </QueryClientProvider>
  );
}
