/**
 * src/components/DetailView.jsx
 * -------------------------------------------------------------------
 * One instrument in depth: chart, indicator readouts, entry window, and
 * the prediction log.
 *
 * The prediction log is the honesty page. It shows every call the system
 * made and whether it was right — including when the hit rate is bad.
 * A signal tool that only shows its wins is worthless, so accuracy is
 * displayed plainly, and suppressed entirely until there are enough
 * graded samples to mean anything.
 * -------------------------------------------------------------------
 */

import { useCallback } from 'react';
import { Link, useParams } from 'react-router-dom';
import ChartView from './ChartView.jsx';
import ModelPanel from './ModelPanel.jsx';
import { fetchSymbol, usePolling } from '../lib/api.js';
import { isSymbolAllowed, withinHistoryWindow, planLimitsFor } from '../lib/plans.js';
import {
  formatPrice,
  formatPct,
  formatDateTime,
  formatTime,
  formatVolume,
  signalColor,
  signalGlyph,
  changeColor,
  rsiZone,
  timeAgo,
} from '../lib/helpers.js';

/* ------------------------------------------------------------------ */
/* Indicator readout grid                                              */
/* ------------------------------------------------------------------ */

function Readouts({ inst }) {
  const zone = rsiZone(inst.rsi);

  const cells = [
    { label: `SMA ${inst.smaPeriod}`, value: formatPrice(inst.sma, inst.symbol) },
    {
      label: 'VWAP',
      value: formatPrice(inst.vwap, inst.symbol),
      note: inst.vwapIsVolumeWeighted ? 'volume weighted' : 'unweighted (no FX volume)',
    },
    {
      label: 'RSI',
      value: inst.rsi == null ? '—' : inst.rsi.toFixed(1),
      note: zone.label,
      color: zone.color,
    },
    {
      label: 'Volatility',
      value: inst.volatility == null ? '—' : `${inst.volatility.toFixed(3)}%`,
      note: 'σ of log returns',
    },
    {
      label: 'BB upper',
      value: inst.bollinger ? formatPrice(inst.bollinger.upper, inst.symbol) : '—',
    },
    {
      label: 'BB lower',
      value: inst.bollinger ? formatPrice(inst.bollinger.lower, inst.symbol) : '—',
    },
    {
      label: 'Bandwidth',
      value: inst.bollinger ? `${(inst.bollinger.bandwidth * 100).toFixed(2)}%` : '—',
      note: inst.bollinger?.breakout
        ? `${inst.bollinger.breakout.toLowerCase()} break`
        : 'in channel',
    },
    {
      label: 'Rule fired',
      value: inst.rule,
      mono: true,
    },
  ];

  return (
    <div className="grid grid-cols-2 gap-px bg-hairline sm:grid-cols-4">
      {cells.map((c) => (
        <div key={c.label} className="bg-panel px-3 py-2.5">
          <p className="label">{c.label}</p>
          <p className={`tnum mt-0.5 text-sm ${c.color || 'text-ink'} ${c.mono ? 'text-tick' : ''}`}>
            {c.value}
          </p>
          {c.note && <p className="text-micro normal-case text-ink-faint">{c.note}</p>}
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Prediction log                                                      */
/* ------------------------------------------------------------------ */

function PredictionLog({ log, symbol, historyDays }) {
  const { rows = [], accuracy, graded = 0 } = log || {};

  return (
    <section className="panel edge-top">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-divider px-4 py-3">
        <div>
          <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
            Prediction log
          </h2>
          <p className="mt-0.5 text-micro normal-case text-ink-faint">
            Every call, graded 60 minutes later against the actual move
            {historyDays ? ` · last ${historyDays} day${historyDays === 1 ? '' : 's'} on this plan` : ''}.
          </p>
        </div>

        <div className="text-right">
          {accuracy == null ? (
            <>
              <p className="tnum text-readout leading-none text-ink-faint">—</p>
              <p className="text-micro uppercase text-ink-faint">
                {graded}/5 graded · need 5+
              </p>
            </>
          ) : (
            <>
              <p
                className={`tnum text-readout leading-none ${
                  accuracy >= 55 ? 'text-up' : accuracy >= 45 ? 'text-amber' : 'text-down'
                }`}
              >
                {accuracy}%
              </p>
              <p className="text-micro uppercase text-ink-faint">
                hit rate · {graded} graded
              </p>
            </>
          )}
        </div>
      </header>

      {/* Honest framing of what the hit rate is worth. */}
      {accuracy != null && (
        <p className="border-b border-hairline bg-void/40 px-4 py-2 text-micro normal-case leading-relaxed text-ink-muted">
          {accuracy < 50
            ? 'This is below a coin flip. The heuristic is not working for this instrument — treat its signals as noise.'
            : 'A hit rate over a small sample is not evidence of edge. Direction accuracy also ignores position size and how far price moved, so it says nothing about whether trading these calls would be profitable.'}
        </p>
      )}

      {rows.length === 0 ? (
        <div className="px-4 py-8 text-center">
          <p className="label">No predictions logged yet</p>
          <p className="mx-auto mt-2 max-w-sm text-tick normal-case text-ink-faint">
            The scheduled function writes one row per instrument every 5
            minutes. Rows appear here after the first cron run, and get graded
            an hour later.
          </p>
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="term-table">
            <thead>
              <tr>
                <th>Time</th>
                <th>Call</th>
                <th className="text-right">Conf</th>
                <th className="text-right">Entry</th>
                <th className="text-right">Resolved</th>
                <th className="text-right">Move</th>
                <th className="text-right">Result</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.id}>
                  <td className="text-ink-muted">{formatDateTime(r.created_at)}</td>
                  <td>
                    <span className={signalColor(r.predicted_direction)}>
                      {signalGlyph(r.predicted_direction)} {r.predicted_direction}
                    </span>
                  </td>
                  <td className="tnum text-right text-ink-muted">
                    {r.confidence == null ? '—' : `${r.confidence}%`}
                  </td>
                  <td className="tnum text-right">
                    {formatPrice(r.price_at_prediction, symbol)}
                  </td>
                  <td className="tnum text-right">
                    {r.price_at_resolution
                      ? formatPrice(r.price_at_resolution, symbol)
                      : '—'}
                  </td>
                  <td className={`tnum text-right ${changeColor(r.move_pct)}`}>
                    {r.move_pct == null ? '—' : formatPct(r.move_pct)}
                  </td>
                  <td className="text-right">
                    {r.actual_result === 'PENDING' ? (
                      <span className="text-ink-faint">pending</span>
                    ) : r.actual_result === 'CORRECT' ? (
                      <span className="text-up">correct</span>
                    ) : (
                      <span className="text-down">wrong</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Detail view                                                         */
/* ------------------------------------------------------------------ */

export default function DetailView({ strategy, plan = 'free' }) {
  const { symbol: rawSymbol } = useParams();
  const symbol = decodeURIComponent(rawSymbol || '');
  const symbolAllowed = isSymbolAllowed(plan, symbol);

  const fetcher = useCallback(
    (options) => fetchSymbol(symbol, strategy, options),
    [symbol, strategy]
  );

  // Skip fetching entirely for a symbol this plan can't see — no reason
  // to hit the API for a page we're about to replace with an upgrade
  // prompt. usePolling still needs a fetcher (it's called unconditionally
  // by the hook rules), so it gets a no-op instead.
  const { data, error, loading, isRefreshing, lastUpdated, refresh } = usePolling(
    symbolAllowed ? fetcher : async () => null,
    60000,
    [symbol, strategy, symbolAllowed]
  );

  if (!symbolAllowed) {
    return (
      <div className="panel border-amber/40 p-6 text-center">
        <h2 className="label-amber">{symbol} is on a higher plan</h2>
        <p className="mx-auto mt-2 max-w-sm text-sm normal-case text-ink-muted">
          Your current plan ({plan}) doesn't include this pair. Upgrade to unlock it.
        </p>
        <div className="mt-4 flex justify-center gap-2">
          <Link to="/settings" className="btn-amber">
            View plans
          </Link>
          <Link to="/" className="btn-ghost">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  if (loading && !data) {
    return (
      <div className="space-y-3">
        <div className="panel h-24 animate-pulse-tick" />
        <div className="panel h-72 animate-pulse-tick" />
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="panel border-amber/40 p-6">
        <h2 className="label-amber">Could not load {symbol}</h2>
        <p className="mt-2 text-sm normal-case text-ink-muted">{error}</p>
        <div className="mt-4 flex gap-2">
          <button type="button" onClick={refresh} className="btn-amber">
            Try again
          </button>
          <Link to="/" className="btn-ghost">
            Back to dashboard
          </Link>
        </div>
      </div>
    );
  }

  const inst = data?.instrument;
  if (!inst) return null;

  // Filter the log to the plan's history window, then recompute
  // accuracy/graded from THAT subset — the server's numbers cover its
  // full 50-row lookback regardless of plan, and showing a "72% · 20
  // graded" header over a 5-row table would be a confusing mismatch.
  const historyDays = planLimitsFor(plan).historyDays;
  const gatedLog = (() => {
    if (!data.predictionLog) return data.predictionLog;
    const rows = withinHistoryWindow(plan, data.predictionLog.rows || []);
    const graded = rows.filter((r) => r.actual_result && r.actual_result !== 'PENDING');
    const correct = graded.filter((r) => r.actual_result === 'CORRECT').length;
    return {
      ...data.predictionLog,
      rows,
      graded: graded.length,
      accuracy: graded.length >= 5 ? Math.round((correct / graded.length) * 100) : null,
    };
  })();

  return (
    <div className="space-y-3">
      {/* ---- Breadcrumb ---- */}
      <nav className="flex items-center gap-2 text-tick uppercase tracking-wider">
        <Link to="/" className="text-ink-faint transition-colors hover:text-amber">
          Dashboard
        </Link>
        <span className="text-ink-faint">/</span>
        <span className="text-ink">{inst.symbol}</span>
      </nav>

      {/* ---- Header block ---- */}
      <header className="panel edge-top flex flex-wrap items-end justify-between gap-6 px-4 py-4">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-sans text-2xl font-semibold text-ink">{inst.symbol}</h1>
            <span className="label border border-hairline px-1.5 py-0.5">
              {inst.assetClass}
            </span>
          </div>
          <p className="mt-0.5 text-tick text-ink-faint">{inst.name}</p>
          <p className="tnum mt-3 text-hero font-medium leading-none text-ink">
            {formatPrice(inst.price, inst.symbol)}
          </p>
          <p className={`tnum mt-1.5 text-sm ${changeColor(inst.change24h)}`}>
            {formatPct(inst.change24h)}{' '}
            <span className="text-ink-faint">over 24h</span>
          </p>
        </div>

        <div className="text-right">
          <p className="label">Current signal</p>
          <div
            className={`mt-1 flex items-center justify-end gap-2 ${signalColor(
              inst.signal
            )}`}
          >
            <span className="text-3xl leading-none" aria-hidden="true">
              {signalGlyph(inst.signal)}
            </span>
            <span className="font-sans text-2xl font-semibold">{inst.signal}</span>
          </div>
          <p className="tnum mt-1 text-tick text-ink-muted">
            {inst.confidence}% confidence · {data.strategy.label}
          </p>
          <p className="mt-2 text-micro uppercase text-ink-faint">
            {isRefreshing ? 'refreshing…' : `updated ${timeAgo(lastUpdated)}`}
          </p>
        </div>
      </header>

      {/* ---- Reasoning ---- */}
      {inst.reasons?.length > 0 && (
        <section className="panel px-4 py-3">
          <h2 className="label-amber">How this call was reached</h2>
          <ul className="mt-2 space-y-1.5">
            {inst.reasons.map((r, i) => (
              <li key={i} className="flex gap-2 text-tick normal-case leading-relaxed text-ink-muted">
                <span className="text-amber-dim">{String(i + 1).padStart(2, '0')}</span>
                <span>{r}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {/* ---- Chart ---- */}
      <ChartView
        series={inst.series}
        symbol={inst.symbol}
        smaPeriod={inst.smaPeriod}
        entryWindow={inst.entryWindow}
      />

      {/* ---- Indicators ---- */}
      <Readouts inst={inst} />

      {/* ---- Entry window ---- */}
      {inst.entryWindow && (
        <section className="panel edge-top px-4 py-3">
          <h2 className="label-amber">Best entry window</h2>
          <p className="tnum mt-1.5 text-sm text-ink">
            {formatTime(inst.entryWindow.time)} at{' '}
            {formatPrice(inst.entryWindow.price, inst.symbol)} · suggested stop{' '}
            {formatPrice(inst.entryWindow.stopLoss, inst.symbol)} (−
            {inst.entryWindow.stopLossPct}%)
          </p>
          <div className="mt-2 flex flex-wrap gap-x-5 gap-y-1">
            <span className="text-micro uppercase text-ink-faint">
              RSI at that point{' '}
              <span className="tnum text-ink-muted">
                {inst.entryWindow.rsi.toFixed(1)}
              </span>
            </span>
            <span className="text-micro uppercase text-ink-faint">
              Volume vs window mean{' '}
              <span className="tnum text-ink-muted">
                {inst.entryWindow.volumeRatio.toFixed(2)}×
              </span>
            </span>
            <span className="text-micro uppercase text-ink-faint">
              Quality{' '}
              <span className="text-ink-muted">{inst.entryWindow.quality}</span>
            </span>
          </div>
          <p className="mt-2.5 max-w-prose text-micro normal-case leading-relaxed text-ink-faint">
            This is the most oversold, highest-volume moment in the last{' '}
            {inst.entryWindow.lookbackHours} hours — a point in the past, not a
            forecast. It tells you where the recent window's best entry was, not
            that the same conditions will return.
          </p>
        </section>
      )}

      {/* ---- Model ---- */}
      <ModelPanel
        model={data.model}
        ml={inst.ml}
        combined={inst.combined}
        heuristicSignal={inst.signal}
      />

      {/* ---- Prediction log ---- */}
      <PredictionLog log={gatedLog} symbol={inst.symbol} historyDays={historyDays} />
    </div>
  );
}
