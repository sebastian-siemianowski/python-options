import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient } from '@tanstack/react-query';
import { PersistQueryClientProvider } from '@tanstack/react-query-persist-client';
import { createSyncStoragePersister } from '@tanstack/query-sync-storage-persister';
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
import IndicatorsPage from './pages/IndicatorsPage';

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 2 * 60_000,     // 2 minutes - stale-while-revalidate
      gcTime: 24 * 60 * 60_000,  // 24h - keep in memory long enough to persist
      refetchOnWindowFocus: true, // Background refresh on tab return
      refetchOnMount: 'always',   // Re-fetch when component remounts (shows cached, refreshes in bg)
      retry: 1,
    },
  },
});

// Persist the query cache to localStorage so data is instantly available on
// page reload. Queries show cached data immediately, then refresh in the
// background when stale.
const persister = createSyncStoragePersister({
  storage: window.localStorage,
  key: 'python-options-query-cache',
  throttleTime: 1_000,
});

// Bump this when query response shapes change to invalidate stale caches
// across all users after a deploy.
const CACHE_BUSTER = 'v1';

export default function App() {
  return (
    <PersistQueryClientProvider
      client={queryClient}
      persistOptions={{
        persister,
        maxAge: 24 * 60 * 60_000, // 24 hours
        buster: CACHE_BUSTER,
        dehydrateOptions: {
          // Only persist successful queries to avoid caching error states.
          shouldDehydrateQuery: (q) => q.state.status === 'success',
        },
      }}
    >
      <ToastProvider>
        <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
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
              <Route path="/indicators" element={<IndicatorsPage />} />
              <Route path="/services" element={<ServicesPage />} />
            </Route>
          </Routes>
          <ToastContainer />
        </BrowserRouter>
      </ToastProvider>
    </PersistQueryClientProvider>
  );
}
