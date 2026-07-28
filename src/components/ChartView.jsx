/**
 * src/components/ChartView.jsx
 * -------------------------------------------------------------------
 * Price chart with an RSI overlay, built on Recharts.
 *
 * Two stacked panels sharing one X axis, rather than a dual-Y-axis
 * single chart. RSI is bounded 0–100 and price is not, so plotting them
 * on twin axes makes the RSI line's slope visually meaningless. Split
 * panels keep both readable — which is how every real terminal draws it.
 * -------------------------------------------------------------------
 */

import {
  Area,
  AreaChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { formatPrice, formatTime, formatDateTime } from '../lib/helpers.js';

/* ------------------------------------------------------------------ */
/* Tooltip                                                             */
/* ------------------------------------------------------------------ */

function TerminalTooltip({ active, payload, symbol }) {
  if (!active || !payload?.length) return null;
  const point = payload[0].payload;

  return (
    <div className="panel px-3 py-2 shadow-lg">
      <p className="label mb-1.5">{formatDateTime(point.t)}</p>
      <dl className="space-y-0.5">
        <div className="flex justify-between gap-6">
          <dt className="text-micro uppercase text-ink-faint">Close</dt>
          <dd className="tnum text-tick text-ink">{formatPrice(point.close, symbol)}</dd>
        </div>
        {point.sma != null && (
          <div className="flex justify-between gap-6">
            <dt className="text-micro uppercase text-ink-faint">SMA</dt>
            <dd className="tnum text-tick text-amber">{formatPrice(point.sma, symbol)}</dd>
          </div>
        )}
        {point.rsi != null && (
          <div className="flex justify-between gap-6">
            <dt className="text-micro uppercase text-ink-faint">RSI</dt>
            <dd className="tnum text-tick text-ink">{point.rsi.toFixed(1)}</dd>
          </div>
        )}
      </dl>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Chart                                                               */
/* ------------------------------------------------------------------ */

export default function ChartView({ series, symbol, smaPeriod, entryWindow }) {
  if (!series?.length) {
    return (
      <div className="panel flex h-64 items-center justify-center">
        <p className="label">No series data</p>
      </div>
    );
  }

  const closes = series.map((d) => d.close);
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const pad = (max - min) * 0.08 || max * 0.01;

  // Colour the price area by net direction across the window.
  const isUp = closes[closes.length - 1] >= closes[0];
  const stroke = isUp ? '#3DD68C' : '#FF5C5C';
  const fillId = isUp ? 'fillUp' : 'fillDown';

  return (
    <div className="space-y-px">
      {/* ---------------- Price panel ---------------- */}
      <div className="panel px-2 pt-3 pb-1">
        <div className="mb-1 flex items-baseline justify-between px-2">
          <h3 className="label">Price · 5m · 24h</h3>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1.5 text-micro uppercase text-ink-faint">
              <span
                className="inline-block h-[2px] w-4"
                style={{ background: stroke }}
              />
              close
            </span>
            <span className="flex items-center gap-1.5 text-micro uppercase text-ink-faint">
              <span className="inline-block h-[2px] w-4 bg-amber" />
              sma {smaPeriod}
            </span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <defs>
              <linearGradient id="fillUp" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3DD68C" stopOpacity={0.22} />
                <stop offset="100%" stopColor="#3DD68C" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="fillDown" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#FF5C5C" stopOpacity={0.22} />
                <stop offset="100%" stopColor="#FF5C5C" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="2 4" vertical={false} />

            <XAxis
              dataKey="t"
              tickFormatter={formatTime}
              minTickGap={48}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[min - pad, max + pad]}
              tickFormatter={(v) => formatPrice(v, symbol)}
              width={72}
              orientation="right"
              axisLine={false}
              tickLine={false}
            />

            <Tooltip
              content={<TerminalTooltip symbol={symbol} />}
              cursor={{ stroke: '#2A3543', strokeWidth: 1 }}
            />

            {/* The historical best-entry moment, marked in amber. */}
            {entryWindow?.time && (
              <ReferenceLine
                x={entryWindow.time}
                stroke="#E8B339"
                strokeDasharray="3 3"
                strokeWidth={1}
                label={{
                  value: 'best entry',
                  position: 'insideTopLeft',
                  fill: '#E8B339',
                  fontSize: 9,
                  fontFamily: 'JetBrains Mono, monospace',
                }}
              />
            )}

            <Area
              type="monotone"
              dataKey="close"
              stroke={stroke}
              strokeWidth={1.5}
              fill={`url(#${fillId})`}
              dot={false}
              isAnimationActive={false}
            />
            <Area
              type="monotone"
              dataKey="sma"
              stroke="#E8B339"
              strokeWidth={1}
              strokeDasharray="4 3"
              fill="none"
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* ---------------- RSI panel ---------------- */}
      <div className="panel px-2 pt-3 pb-1">
        <div className="mb-1 flex items-baseline justify-between px-2">
          <h3 className="label">RSI · 14</h3>
          <div className="flex items-center gap-3">
            <span className="text-micro uppercase text-down">70 overbought</span>
            <span className="text-micro uppercase text-up">30 oversold</span>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={130}>
          <LineChart data={series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />

            {/* Shade the extreme zones rather than only drawing lines —
                it makes "how long has it been stretched" legible at a glance. */}
            <ReferenceArea y1={70} y2={100} fill="#FF5C5C" fillOpacity={0.07} />
            <ReferenceArea y1={0} y2={30} fill="#3DD68C" fillOpacity={0.07} />
            <ReferenceLine y={70} stroke="#7A2B2B" strokeDasharray="2 3" />
            <ReferenceLine y={50} stroke="#2A3543" />
            <ReferenceLine y={30} stroke="#1F6B47" strokeDasharray="2 3" />

            <XAxis
              dataKey="t"
              tickFormatter={formatTime}
              minTickGap={48}
              axisLine={false}
              tickLine={false}
            />
            <YAxis
              domain={[0, 100]}
              ticks={[0, 30, 50, 70, 100]}
              width={72}
              orientation="right"
              axisLine={false}
              tickLine={false}
            />

            <Tooltip
              content={<TerminalTooltip symbol={symbol} />}
              cursor={{ stroke: '#2A3543', strokeWidth: 1 }}
            />

            <Line
              type="monotone"
              dataKey="rsi"
              stroke="#E6EDF5"
              strokeWidth={1.25}
              dot={false}
              connectNulls
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
