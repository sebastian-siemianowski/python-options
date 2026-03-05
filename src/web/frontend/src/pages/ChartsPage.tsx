import { useQuery } from '@tanstack/react-query';
import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { OHLCVBar } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries, type IChartApi } from 'lightweight-charts';

export default function ChartsPage() {
  const { symbol: paramSymbol } = useParams();
  const navigate = useNavigate();
  const [symbol, setSymbol] = useState(paramSymbol || '');
  const [search, setSearch] = useState('');

  const symbolsQ = useQuery({
    queryKey: ['chartSymbols'],
    queryFn: api.chartSymbols,
  });

  // Set default symbol
  useEffect(() => {
    if (!symbol && symbolsQ.data?.symbols?.length) {
      const defaultSym = symbolsQ.data.symbols.find((s) => s === 'SPY') || symbolsQ.data.symbols[0];
      setSymbol(defaultSym);
    }
  }, [symbol, symbolsQ.data]);

  const symbols = symbolsQ.data?.symbols || [];
  const filtered = symbols.filter((s) => s.toLowerCase().includes(search.toLowerCase()));

  return (
    <>
      <PageHeader title="Interactive Charts">
        {symbols.length} symbols available for charting
      </PageHeader>

      <div className="flex gap-4">
        {/* Symbol picker */}
        <div className="w-48 flex-shrink-0">
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search symbol..."
            className="w-full px-3 py-1.5 rounded-lg bg-[#1a1a2e] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5] mb-2"
          />
          <div className="glass-card overflow-y-auto max-h-[calc(100vh-220px)]">
            {filtered.slice(0, 50).map((s) => (
              <button
                key={s}
                onClick={() => { setSymbol(s); navigate(`/charts/${s}`); }}
                className={`w-full text-left px-3 py-1.5 text-xs font-medium transition hover:bg-[#16213e] ${
                  s === symbol ? 'bg-[#16213e] text-[#42A5F5]' : 'text-[#94a3b8]'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        {/* Chart area */}
        <div className="flex-1">
          {symbol ? (
            <ChartPanel symbol={symbol} />
          ) : (
            <div className="glass-card p-8 text-center text-[#64748b]">
              Select a symbol to view its chart
            </div>
          )}
        </div>
      </div>
    </>
  );
}

function ChartPanel({ symbol }: { symbol: string }) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<IChartApi | null>(null);

  const ohlcvQ = useQuery({
    queryKey: ['ohlcv', symbol],
    queryFn: () => api.chartOhlcv(symbol, 365),
    enabled: !!symbol,
  });

  const indQ = useQuery({
    queryKey: ['indicators', symbol],
    queryFn: () => api.chartIndicators(symbol, 365),
    enabled: !!symbol,
  });

  const forecastQ = useQuery({
    queryKey: ['forecast', symbol],
    queryFn: () => api.chartForecast(symbol),
    enabled: !!symbol,
  });

  useEffect(() => {
    if (!chartRef.current || !ohlcvQ.data?.data?.length) return;

    // Clean up previous chart
    if (chartInstance.current) {
      chartInstance.current.remove();
      chartInstance.current = null;
    }

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#1a1a2e' },
        textColor: '#94a3b8',
      },
      grid: {
        vertLines: { color: '#2a2a4a' },
        horzLines: { color: '#2a2a4a' },
      },
      width: chartRef.current.clientWidth,
      height: 450,
      crosshair: {
        vertLine: { color: '#42A5F5', width: 1, style: 2 },
        horzLine: { color: '#42A5F5', width: 1, style: 2 },
      },
    });

    chartInstance.current = chart;

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00E676',
      downColor: '#FF1744',
      borderUpColor: '#00E676',
      borderDownColor: '#FF1744',
      wickUpColor: '#00E676',
      wickDownColor: '#FF1744',
    });
    candleSeries.setData(ohlcvQ.data.data);

    // Volume
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.8, bottom: 0 },
    });
    volumeSeries.setData(
      ohlcvQ.data.data.map((d: OHLCVBar) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(0,230,118,0.3)' : 'rgba(255,23,68,0.3)',
      }))
    );

    // Indicators
    if (indQ.data?.indicators) {
      const ind = indQ.data.indicators;
      if (ind.sma20?.length) {
        const sma20 = chart.addSeries(LineSeries, { color: '#FFB300', lineWidth: 1 });
        sma20.setData(ind.sma20);
      }
      if (ind.sma50?.length) {
        const sma50 = chart.addSeries(LineSeries, { color: '#42A5F5', lineWidth: 1 });
        sma50.setData(ind.sma50);
      }
      if (ind.sma200?.length) {
        const sma200 = chart.addSeries(LineSeries, { color: '#AB47BC', lineWidth: 1 });
        sma200.setData(ind.sma200);
      }
      if (ind.bollinger) {
        if (ind.bollinger.upper?.length) {
          const bbU = chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 });
          bbU.setData(ind.bollinger.upper);
        }
        if (ind.bollinger.lower?.length) {
          const bbL = chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 });
          bbL.setData(ind.bollinger.lower);
        }
      }
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
      chartInstance.current = null;
    };
  }, [ohlcvQ.data, indQ.data]);

  if (ohlcvQ.isLoading) return <LoadingSpinner text={`Loading ${symbol}...`} />;
  if (ohlcvQ.error) return <div className="text-[#FF1744]">Failed to load chart for {symbol}</div>;

  const bars = ohlcvQ.data?.data;
  const lastBar = bars?.[bars.length - 1];
  const prevBar = bars?.[bars.length - 2];
  const change = lastBar && prevBar ? ((lastBar.close - prevBar.close) / prevBar.close * 100) : 0;

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-4 mb-3">
        <h2 className="text-xl font-bold text-[#e2e8f0]">{symbol}</h2>
        {lastBar && (
          <>
            <span className="text-lg font-medium text-[#e2e8f0]">${lastBar.close.toFixed(2)}</span>
            <span className={`text-sm font-medium ${change >= 0 ? 'text-[#00E676]' : 'text-[#FF1744]'}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
            </span>
          </>
        )}

        {/* Legend */}
        <div className="flex gap-3 ml-auto text-[10px]">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#FFB300] inline-block" />SMA20</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#42A5F5] inline-block" />SMA50</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#AB47BC] inline-block" />SMA200</span>
        </div>
      </div>

      {/* Chart */}
      <div className="glass-card p-2">
        <div ref={chartRef} />
      </div>

      {/* Forecast */}
      {forecastQ.data?.forecasts?.length ? (
        <div className="glass-card p-4 mt-4">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Multi-Horizon Forecast</h3>
          <div className="flex gap-3 flex-wrap">
            {forecastQ.data.forecasts.map((f) => {
              const color = f.signal_label === 'BUY' ? '#00E676' : f.signal_label === 'SELL' ? '#FF1744' : '#64748b';
              return (
                <div key={f.horizon_days} className="bg-[#0f0f23] rounded-lg px-3 py-2 text-center min-w-[80px]">
                  <p className="text-[10px] text-[#64748b]">{f.horizon_days}D</p>
                  <p className="text-sm font-bold" style={{ color }}>
                    {f.expected_return_pct >= 0 ? '+' : ''}{f.expected_return_pct.toFixed(1)}%
                  </p>
                  <p className="text-[10px]" style={{ color }}>
                    {f.signal_label} • {(f.probability_up * 100).toFixed(0)}%
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
