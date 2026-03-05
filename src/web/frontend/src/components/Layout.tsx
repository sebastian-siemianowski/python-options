import { Outlet, NavLink } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import {
  LayoutDashboard,
  Signal,
  ShieldAlert,
  LineChart,
  Settings,
  Database,
  Swords,
  Activity,
  HeartPulse,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/signals', label: 'Signals', icon: Signal },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/charts', label: 'Charts', icon: LineChart },
  { to: '/tuning', label: 'Tuning', icon: Settings },
  { to: '/data', label: 'Data', icon: Database },
  { to: '/arena', label: 'Arena', icon: Swords },
  { to: '/services', label: 'Services', icon: HeartPulse },
];

export default function Layout() {
  const healthQ = useQuery({
    queryKey: ['servicesHealth'],
    queryFn: api.servicesHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const allOk = healthQ.data
    ? healthQ.data.api.status === 'ok' && healthQ.data.signal_cache.status !== 'missing'
      && healthQ.data.price_data.status === 'ok'
    : true; // Assume ok while loading

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 flex-shrink-0 border-r border-[#2a2a4a] bg-[#0a0a1a] flex flex-col">
        {/* Logo */}
        <div className="px-4 py-5 border-b border-[#2a2a4a]">
          <div className="flex items-center gap-2">
            <Activity className="w-6 h-6 text-[#42A5F5]" />
            <div>
              <h1 className="text-sm font-bold gradient-text">Signal Engine</h1>
              <p className="text-[10px] text-[#64748b]">Bayesian Model Averaging</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-3 px-2 space-y-0.5">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all duration-150 ${
                  isActive
                    ? 'bg-[#16213e] text-[#42A5F5] font-medium'
                    : 'text-[#94a3b8] hover:bg-[#16213e]/50 hover:text-[#e2e8f0]'
                }`
              }
            >
              <Icon className="w-4 h-4 flex-shrink-0" />
              {label}
              {to === '/services' && (
                <span
                  className={`w-1.5 h-1.5 rounded-full ml-auto ${allOk ? 'bg-[#00E676]' : 'bg-[#FF1744]'} pulse-dot`}
                />
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-4 py-3 border-t border-[#2a2a4a] text-[10px] text-[#64748b]">
          <div className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#00E676] pulse-dot"></span>
            v5.30 • BMA + Kalman
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-[#0f0f23]">
        <div className="p-6 max-w-[1600px] mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
