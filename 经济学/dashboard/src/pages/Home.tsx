import { useLoadData } from '../hooks/useLoadData';
import { useStore } from '../store';
import { useState, useMemo } from 'react';
import ChatPanel from '../components/ChatPanel';

export default function Home() {
  useLoadData();
  const data = useStore((s) => s.data);
  const loading = useStore((s) => s.loading);
  const error = useStore((s) => s.error);

  const [selected, setSelected] = useState('');

  // Build options sorted by name
  const options = useMemo(() => {
    if (!data) return [];
    return [...data.stocks].sort((a, b) => a.name.localeCompare(b.name, 'zh'));
  }, [data]);

  // Find selected stock
  const stock = useMemo(() => {
    if (!data || !selected) return null;
    return data.stocks.find((s) => s.sym === selected) ?? null;
  }, [data, selected]);

  // Vol series
  const volSeries = useMemo(() => {
    if (!data || !selected) return null;
    return data.vol_data[selected] ?? null;
  }, [data, selected]);

  // Compute stock rankings from pair data
  const rankings = useMemo(() => {
    if (!data) return [];
    const pairs = data.pairs;
    const stocks = data.stocks;
    const names = [...new Set([...pairs.map(p => p.target), ...pairs.map(p => p.source)])];
    const scores = names.map(n => {
      const t = pairs.filter(p => p.target === n && p.h1_xgb);
      const s = pairs.filter(p => p.source === n && p.h1_xgb);
      const t_avg = t.length ? t.reduce((a, p) => a + p.h1_xgb, 0) / t.length : 0;
      const s_avg = s.length ? s.reduce((a, p) => a + p.h1_xgb, 0) / s.length : 0;
      const weighted = t.length * t_avg + s.length * s_avg;
      const stk = stocks.find(x => x.name === n);
      return {
        name: n,
        sym: stk?.sym ?? '',
        label: stk?.label ?? '',
        score: Math.round(weighted),
        signalCount: t.length,
        predictCount: s.length,
        targetAvg: t_avg,
        sourceAvg: s_avg,
      };
    });
    scores.sort((a, b) => b.score - a.score || b.signalCount - a.signalCount);
    return scores.slice(0, 4);
  }, [data]);

  if (loading) return <Loading />;
  if (error) return <ErrorView msg={error} />;
  if (!data) return null;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      <div className="max-w-[1000px] mx-auto px-6 py-8">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-xl font-bold text-zinc-100 tracking-tight">
              五行命格 · 波动趋势
            </h1>
            <p className="text-xs text-zinc-500 mt-0.5">
              58只A股 · S₀命格预测系统
            </p>
          </div>
          <div className="text-right text-xs text-zinc-600">
            数据更新: {data.export_time}
          </div>
        </div>

        {/* Recommendation */}
        {rankings.length > 0 && (
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-3">
              <span className="text-xs text-zinc-600 font-medium">🏆 推荐关注</span>
              <div className="h-px flex-1 bg-zinc-800" />
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {rankings.map((r, i) => (
                <button
                  key={r.sym}
                  onClick={() => setSelected(r.sym)}
                  className={`relative text-left p-3 rounded-xl border transition-all duration-200 ${
                    selected === r.sym
                      ? 'border-amber-500/50 bg-amber-500/10'
                      : 'border-zinc-800 bg-zinc-900/50 hover:border-zinc-700 hover:bg-zinc-900'
                  }`}
                >
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <span className={`text-[10px] font-bold ${i === 0 ? 'text-amber-400' : i === 1 ? 'text-zinc-300' : i === 2 ? 'text-amber-600' : 'text-zinc-500'}`}>
                      #{i + 1}
                    </span>
                    <span className="text-xs text-zinc-200 font-medium truncate">{r.name}</span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-zinc-500">
                    <span className={`px-1.5 py-0.5 rounded ${r.label === '土旺缺火金' ? 'bg-amber-900/30 text-amber-400' : 'bg-zinc-800 text-zinc-500'}`}>
                      {r.label}
                    </span>
                    <span>{r.score}分</span>
                  </div>
                  <div className="text-[10px] text-zinc-600 mt-1">
                    {r.signalCount}源(均{r.targetAvg.toFixed(0)}%) · 预测{r.predictCount}只
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Dropdown */}
        <div className="mb-6">
          <label className="text-xs text-zinc-500 mb-1.5 block">选择股票</label>
          <select
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            className="w-full bg-zinc-900 border border-zinc-700 rounded-xl px-4 py-3 text-zinc-200 text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/40 focus:border-amber-500/60 transition-all appearance-none cursor-pointer"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' fill='none' viewBox='0 0 20 20'%3e%3cpath stroke='%2371717a' stroke-linecap='round' stroke-linejoin='round' stroke-width='1.5' d='M6 8l4 4 4-4'/%3e%3c/svg%3e")`,
              backgroundPosition: 'right 0.75rem center',
              backgroundRepeat: 'no-repeat',
              backgroundSize: '1.25rem',
            }}
          >
            <option value="">— 请选择 —</option>
            {data.groups
              .filter((g) => g.size >= 2)
              .map((g) => (
                <optgroup key={g.label} label={`${g.label} (${g.size}只)`}>
                  {g.members.map((name) => {
                    const s = data.stocks.find((x) => x.name === name);
                    if (!s) return null;
                    return (
                      <option key={s.sym} value={s.sym}>
                        {s.name} ({s.sym.replace('sh','沪').replace('sz','深')})
                      </option>
                    );
                  })}
                </optgroup>
              ))}
            {/* Singles group */}
            <optgroup label="独命格股">
              {data.stocks
                .filter((s) => {
                  const g = data.groups.find((x) => x.members.includes(s.name));
                  return g && g.size === 1;
                })
                .sort((a, b) => a.name.localeCompare(b.name, 'zh'))
                .map((s) => (
                  <option key={s.sym} value={s.sym}>
                    {s.name} ({s.sym.replace('sh','沪').replace('sz','深')})
                  </option>
                ))}
            </optgroup>
          </select>
        </div>

        {/* Analysis */}
        {stock && (
          <>
            <Analysis stock={stock} volSeries={volSeries} />
            <ChatPanel sym={stock.sym} stockName={stock.name} />
          </>
        )}
      </div>
    </div>
  );
}

function Analysis({ stock, volSeries }: { stock: any; volSeries: any }) {
  const data = useStore((s) => s.data);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<any>(null);
  const { sym, name, label, s0, s0_normalized, listing } = stock;

  // Group info
  const group = data?.groups.find((g) => g.members.includes(name));
  const mates = group ? group.members.filter((m) => m !== name) : [];

  // Pairs involving this stock
  const pairAsTarget = (data?.pairs ?? []).filter((p) => p.target === name);
  const pairAsSource = (data?.pairs ?? []).filter((p) => p.source === name);

  const WX = ['木', '火', '土', '金', '水'];
  const WX_COLORS: Record<string, string> = {
    '木': '#4ade80', '火': '#f87171', '土': '#fbbf24',
    '金': '#e2e8f0', '水': '#60a5fa',
  };

  const doPredict = () => {
    if (!data || !volSeries) return;
    setPredicting(true);
    // Simulate async to show loading state briefly
    setTimeout(() => {
      const result = computePrediction(name, sym, volSeries, pairAsTarget, data.vol_data);
      setPrediction(result);
      setPredicting(false);
    }, 400);
  };

  return (
    <div className="space-y-4 fade-in">
      {/* Stock info card */}
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-5">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-lg font-semibold text-zinc-100">{name}</h2>
            <div className="flex items-center gap-2 mt-1 text-xs text-zinc-500">
              <span>{sym.replace('sh', '上海').replace('sz', '深圳')}</span>
              <span className="w-1 h-1 rounded-full bg-zinc-700" />
              <span>上市: {listing}</span>
              <span className="w-1 h-1 rounded-full bg-zinc-700" />
              <span className="text-amber-400/80 font-medium">{label}</span>
            </div>
          </div>
          {/* Mini radar bars */}
          <div className="flex gap-1.5">
            {s0.map((v: number, i: number) => (
              <div key={i} className="flex flex-col items-center gap-0.5">
                <div className="text-[10px] text-zinc-500">{WX[i]}</div>
                <div
                  className="w-5 rounded-full transition-all"
                  style={{
                    height: `${Math.max(v * 16, 6)}px`,
                    backgroundColor: WX_COLORS[WX[i]],
                    opacity: 0.7,
                  }}
                />
                <div className="text-[10px] text-zinc-400 font-mono">{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Vol trend chart */}
      {volSeries && (
        <>
          <VolChart
            dates={volSeries.dates}
            values={volSeries.vol}
            name={name}
            prediction={prediction}
          />

          {/* Predict button */}
          <button
            onClick={doPredict}
            disabled={predicting}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-600/20 to-emerald-600/20 border border-amber-700/30 text-amber-300 text-sm font-medium hover:from-amber-600/30 hover:to-emerald-600/30 transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {predicting ? (
              <span className="flex items-center justify-center gap-2">
                <span className="w-3.5 h-3.5 rounded-full border-2 border-zinc-700 border-t-amber-400 animate-spin" />
                计算中...
              </span>
            ) : prediction ? (
              '重新预测'
            ) : (
              '🎯 生成趋势预测'
            )}
          </button>
        </>
      )}

      {/* Prediction result */}
      {prediction && <PredictionResult pred={prediction} />}

      {/* Prediction pairs */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PairCard
          title="作为标的 (被预测)"
          pairs={pairAsTarget}
          role="target"
          emptyText="暂无配对数据"
        />
        <PairCard
          title="作为信号源 (预测其他)"
          pairs={pairAsSource}
          role="source"
          emptyText="暂无配对数据"
        />
      </div>

      {/* Group mates */}
      {mates.length > 0 && (
        <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
          <h3 className="text-xs text-zinc-500 mb-2">
            同命格股票 ({label})
          </h3>
          <div className="flex flex-wrap gap-1.5">
            {mates.map((m) => (
              <span
                key={m}
                className="px-2.5 py-1 rounded-lg bg-zinc-800 text-zinc-300 text-xs"
              >
                {m}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/** Compute trend prediction from available vol + pair data */
function computePrediction(
  name: string,
  sym: string,
  volSeries: { dates: string[]; vol: number[] },
  pairs: any[],
  allVol: Record<string, { dates: string[]; vol: number[] }>,
) {
  const vol = volSeries.vol;
  const n = vol.length;
  const L = 20; // analysis window
  if (n < L + 5) return fallbackPrediction(vol);

  const recent = vol.slice(n - L);
  const last_val = recent[recent.length - 1];

  // ── 1. Mean-reversion baseline ──
  const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
  const reversion = 0.2; // pull 20% toward mean per step
  const baseline = last_val + (mean - last_val) * reversion;

  // ── 2. Linear trend over last 10 days ──
  const T = 10;
  const trendSlice = recent.slice(-T);
  const slope = linearSlope(trendSlice);
  const trend_weight = 0.25; // dampen trend influence
  const trend_component = slope * T * trend_weight;

  // ── 3. Signal source adjustment ──
  let signal_sum = 0;
  let weight_sum = 0;
  const signals: { name: string; weight: number; direction: number }[] = [];

  const goodPairs = pairs.filter((p) => p.h1_xgb != null && p.h1_xgb > 0);
  for (const p of goodPairs) {
    const srcVol = allVol[p.source_sym];
    if (!srcVol || srcVol.vol.length < 10) continue;

    const srcRecent = srcVol.vol.slice(-8);
    if (srcRecent.length < 5) continue;

    // Source direction: slope over last 8 days (relative)
    const srcSlope = linearSlope(srcRecent);
    const srcDir = srcSlope / (srcRecent[srcRecent.length - 1] || 0.001);
    const w = Math.max(0, p.h1_xgb);

    signal_sum += srcDir * w;
    weight_sum += w;
    signals.push({ name: p.source, weight: w, direction: srcDir });
  }

  // ── 4. Combined prediction ──
  let adj = 0;
  let confidence = '低';
  if (weight_sum > 0 && goodPairs.length >= 1) {
    const avg_signal = signal_sum / weight_sum;
    adj = avg_signal * 0.15; // signal dampening
  }
  confidence = goodPairs.length >= 2 ? '中' : goodPairs.length === 1 ? '低' : '无信号源';

  const pred_value = Math.max(0, baseline + trend_component + adj * last_val);
  const pct_change = ((pred_value - last_val) / (last_val || 0.001)) * 100;

  // ── 5. Classify direction (realistic thresholds for vol %) ──
  let direction: string;
  let arrow: string;
  let color: string;
  if (pct_change > 5) {
    direction = '上升 ↑';
    arrow = '↑';
    color = 'text-red-400';
  } else if (pct_change > 1.5) {
    direction = '微升 ↗';
    arrow = '↗';
    color = 'text-orange-400';
  } else if (pct_change > -1.5) {
    direction = '平稳 →';
    arrow = '→';
    color = 'text-zinc-400';
  } else if (pct_change > -5) {
    direction = '微降 ↘';
    arrow = '↘';
    color = 'text-blue-400';
  } else {
    direction = '下降 ↓';
    arrow = '↓';
    color = 'text-cyan-400';
  }

  return {
    direction,
    arrow,
    color,
    pred_change_pct: pct_change.toFixed(2),
    pred_value,
    last_value: last_val,
    confidence,
    signal_count: goodPairs.length,
    signals: signals.sort((a, b) => Math.abs(b.direction) - Math.abs(a.direction)).slice(0, 5),
    ar_direction: (slope / (last_val || 0.001) * 100).toFixed(2),
  };
}

/** Linear regression slope over array values */
function linearSlope(arr: number[]): number {
  const m = arr.length;
  if (m < 2) return 0;
  const xMean = (m - 1) / 2;
  let xy = 0, xx = 0;
  for (let i = 0; i < m; i++) {
    const dx = i - xMean;
    xy += dx * arr[i];
    xx += dx * dx;
  }
  return xx > 0 ? xy / xx : 0;
}

/** Fallback when vol data is too short */
function fallbackPrediction(vol: number[]) {
  const last_val = vol[vol.length - 1] || 0;
  return {
    direction: '数据不足',
    arrow: '?',
    color: 'text-zinc-500',
    pred_change_pct: '0.00',
    pred_value: last_val,
    last_value: last_val,
    confidence: '低',
    signal_count: 0,
    signals: [],
    ar_direction: '0.00',
  };
}

function PredictionResult({ pred }: { pred: any }) {
  return (
    <div className="bg-zinc-900/70 border border-zinc-700/50 rounded-xl p-5 fade-in">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-xs text-zinc-500 font-medium">趋势预测结果</h3>
        <span className="text-[10px] text-zinc-600">
          信号源: {pred.signal_count}个
        </span>
      </div>

      {/* Main prediction */}
      <div className="flex items-center gap-4 mb-4">
        <div className={`text-3xl font-bold ${pred.color}`}>
          {pred.arrow}
        </div>
        <div>
          <div className={`text-lg font-semibold ${pred.color}`}>
            {pred.direction}
          </div>
          <div className="text-xs text-zinc-500 mt-0.5">
            预计波动变化 <strong className="text-zinc-300">{pred.pred_change_pct}%</strong>
            {' · '}置信度: <strong className="text-zinc-300">{pred.confidence}</strong>
          </div>
        </div>
      </div>

      {/* Detail metrics */}
      <div className="grid grid-cols-3 gap-3 mb-3">
        <MetricBox label="当前波动" value={`${(pred.last_value * 100).toFixed(2)}%`} />
        <MetricBox label="预测波动" value={`${(pred.pred_value * 100).toFixed(2)}%`} />
        <MetricBox label="AR基准变化" value={`${pred.ar_direction}%`} />
      </div>

      {/* Signal source details */}
      {pred.signals.length > 0 && (
        <div>
          <div className="text-[10px] text-zinc-600 mb-1.5">信号贡献排名:</div>
          <div className="space-y-1">
            {pred.signals.map((s: any, i: number) => (
              <div
                key={s.name}
                className="flex items-center justify-between px-2.5 py-1 rounded bg-zinc-950/50 text-xs"
              >
                <div className="flex items-center gap-2">
                  <span className="text-zinc-600 w-3">#{i + 1}</span>
                  <span className="text-zinc-300">{s.name}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-zinc-600">权重 {s.weight.toFixed(0)}%</span>
                  <span className={s.direction > 0 ? 'text-red-400' : 'text-blue-400'}>
                    {(s.direction * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-zinc-950/50 rounded-lg px-3 py-2 text-center">
      <div className="text-xs font-mono text-zinc-300">{value}</div>
      <div className="text-[10px] text-zinc-600 mt-0.5">{label}</div>
    </div>
  );
}

function VolChart({ dates, values, name, prediction }: { dates: string[]; values: number[]; name: string; prediction?: any }) {
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  const range = maxVal - minVal || 1;

  // Sample to ~120 points for performance
  const step = Math.max(1, Math.floor(dates.length / 120));
  const sampled = dates
    .map((d, i) => ({ date: d, value: values[i], idx: i }))
    .filter((_, i) => i % step === 0);

  // Calculate statistics
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const recent = values.slice(-20);
  const recentMean = recent.reduce((a, b) => a + b, 0) / recent.length;

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-xs text-zinc-500">超额波动率趋势</h3>
        <div className="flex items-center gap-3 text-[10px] text-zinc-500">
          <span>均值 <strong className="text-zinc-300">{(mean * 100).toFixed(2)}%</strong></span>
          <span>近20日 <strong className="text-zinc-300">{(recentMean * 100).toFixed(2)}%</strong></span>
        </div>
      </div>
      <div className="relative h-48">
        {/* Y axis labels */}
        <div className="absolute left-0 top-0 bottom-6 flex flex-col justify-between text-[10px] text-zinc-600 w-10 text-right pr-2">
          <span>{(maxVal * 100).toFixed(1)}%</span>
          <span>{((maxVal + minVal) / 2 * 100).toFixed(1)}%</span>
          <span>{(minVal * 100).toFixed(1)}%</span>
        </div>
        {/* Chart area */}
        <div className="ml-12 mr-2 h-full relative">
          {/* Grid lines */}
          <div className="absolute inset-0 flex flex-col justify-between">
            {[0, 1, 2].map((i) => (
              <div key={i} className="border-t border-zinc-800/50 w-full" style={{ borderTopWidth: i === 1 ? '1px' : '0.5px' }} />
            ))}
          </div>
          {/* SVG line */}
          <svg
            viewBox={`0 0 ${sampled.length} 100`}
            preserveAspectRatio="none"
            className="absolute inset-0 w-full h-full"
          >
            <defs>
              <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="rgba(251, 191, 36, 0.3)" />
                <stop offset="100%" stopColor="rgba(251, 191, 36, 0)" />
              </linearGradient>
            </defs>
            <polygon
              fill="url(#volGrad)"
              points={`0,100 ${sampled
                .map((p, i) => `${i},${100 - ((p.value - minVal) / range) * 90}`)
                .join(' ')} ${sampled.length - 1},100`}
            />
            <polyline
              fill="none"
              stroke="#fbbf24"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              points={sampled
                .map((p, i) => `${i},${100 - ((p.value - minVal) / range) * 90}`)
                .join(' ')}
            />
            {prediction && (
              <>
                {/* Line extending to prediction */}
                <line
                  x1={sampled.length - 1}
                  y1={100 - ((values[values.length - 1] - minVal) / range) * 90}
                  x2={sampled.length + 5}
                  y2={100 - ((prediction.pred_value - minVal) / range) * 90}
                  stroke="#fbbf24"
                  strokeWidth="1"
                  strokeDasharray="3 2"
                  opacity="0.5"
                />
                {/* Prediction dot */}
                <circle
                  cx={sampled.length + 5}
                  cy={100 - ((prediction.pred_value - minVal) / range) * 90}
                  r="4"
                  fill="#f87171"
                  stroke="#09090b"
                  strokeWidth="1.5"
                />
              </>
            )}
          </svg>
        </div>
        {/* X axis */}
        <div className="ml-12 flex justify-between text-[10px] text-zinc-600 mt-1">
          <span>{dates[0]}</span>
          <span>{dates[Math.floor(dates.length / 2)]}</span>
          <span>{dates[dates.length - 1]}</span>
        </div>
      </div>
    </div>
  );
}

function PairCard({
  title,
  pairs,
  role,
  emptyText,
}: {
  title: string;
  pairs: any[];
  role: 'target' | 'source';
  emptyText: string;
}) {
  if (pairs.length === 0) {
    return (
      <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
        <h3 className="text-xs text-zinc-500 mb-3">{title}</h3>
        <p className="text-xs text-zinc-700">{emptyText}</p>
      </div>
    );
  }

  return (
    <div className="bg-zinc-900/50 border border-zinc-800 rounded-xl p-4">
      <h3 className="text-xs text-zinc-500 mb-3">{title}</h3>
      <div className="space-y-1.5">
        {pairs.map((p) => (
          <div
            key={`${p.target}-${p.source}`}
            className="flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-zinc-950/50"
          >
            <div className="flex items-center gap-2 text-xs">
              {role === 'target' ? (
                <>
                  <span className="text-zinc-300">{p.target}</span>
                  <span className="text-zinc-600">←</span>
                  <span className="text-zinc-500">{p.source}</span>
                </>
              ) : (
                <>
                  <span className="text-zinc-500">{p.target}</span>
                  <span className="text-zinc-600">→</span>
                  <span className="text-zinc-300">{p.source}</span>
                </>
              )}
            </div>
            <div className="text-xs font-mono">
              <span className="text-emerald-400 font-medium">
                +{p.h1_xgb?.toFixed(1) ?? '?'}%
              </span>
              <span className="text-zinc-600 ml-1.5">
                (R{'+'}{p.h1_ridge?.toFixed(1) ?? '?'})
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function Loading() {
  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
      <div className="text-center">
        <div className="w-6 h-6 rounded-full border-2 border-zinc-700 border-t-amber-400 animate-spin mx-auto mb-3" />
        <p className="text-zinc-500 text-sm">加载数据中...</p>
      </div>
    </div>
  );
}

function ErrorView({ msg }: { msg: string }) {
  return (
    <div className="min-h-screen bg-zinc-950 flex items-center justify-center">
      <div className="text-center">
        <p className="text-red-400 text-sm mb-1">数据加载失败</p>
        <p className="text-zinc-500 text-xs">{msg}</p>
      </div>
    </div>
  );
}
