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
  Stethoscope,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/signals', label: 'Signals', icon: Signal },
  { to: '/risk', label: 'Risk', icon: ShieldAlert },
  { to: '/charts', label: 'Charts', icon: LineChart },
  { to: '/tuning', label: 'Tuning', icon: Settings },
  { to: '/data', label: 'Data', icon: Database },
  { to: '/arena', label: 'Arena', icon: Swords },
  { to: '/diagnostics', label: 'Diagnostics', icon: Stethoscope },
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
    : true;

  return (
    <div className="flex h-screen overflow-hidden bg-[#060612]">
      {/* Sidebar */}
      <aside className="w-[220px] flex-shrink-0 border-r border-white/[0.04] bg-[#08081a]/90 backdrop-blur-xl flex flex-col">
        {/* Logo */}
        <div className="px-5 py-6">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#42A5F5]/20 to-[#7C4DFF]/20 flex items-center justify-center">
                <Activity className="w-5 h-5 text-[#42A5F5]" />
              </div>
              <span className={`absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full ring-2 ring-[#08081a] ${allOk ? 'bg-[#00E676]' : 'bg-[#FF1744]'} pulse-dot`} />
            </div>
            <div>
              <h1 className="text-[13px] font-semibold gradient-text tracking-tight leading-tight">Signal Engine</h1>
              <p className="text-[10px] text-[#475569] font-medium tracking-wider mt-0.5">BMA + Kalman v5.30</p>
            </div>
          </div>
        </div>

        <div className="divider-fade mx-4" />

        {/* Navigation */}
        <nav className="flex-1 py-4 px-3 space-y-0.5 overflow-y-auto">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === '/'}
              className={({ isActive }) =>
                `flex items-center gap-3 px-3 py-2 rounded-xl text-[13px] transition-all duration-200 ${
                  isActive
                    ? 'bg-white/[0.06] text-[#42A5F5] font-medium nav-active-indicator'
                    : 'text-[#64748b] hover:bg-white/[0.03] hover:text-[#94a3b8]'
                }`
              }
            >
              <Icon className="w-[18px] h-[18px] flex-shrink-0" />
              <span>{label}</span>
              {to === '/services' && (
                <span
                  className={`w-1.5 h-1.5 rounded-full ml-auto ${allOk ? 'bg-[#00E676]' : 'bg-[#FF1744]'} pulse-dot`}
                />
              )}
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="px-5 py-4">
          <div className="divider-fade mb-3" />
          <p className="text-[10px] text-[#3a3a5a] font-medium tracking-wide">Bayesian Model Averaging</p>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto bg-[#0a0a1a]">
        <div className="p-8 max-w-[1600px] mx-auto">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
