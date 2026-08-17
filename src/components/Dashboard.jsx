/**
 * src/components/Dashboard.jsx
 * -------------------------------------------------------------------
 * The grid. Polls /api/get-analysis every 60 seconds.
 *
 * Asset classes are grouped rather than mixed into one flat grid,
 * because a 2% move means something completely different in SOL than in
 * EUR/USD, and putting them side by side invites a false comparison.
 * -------------------------------------------------------------------
 */

import { useCallback, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import PredictionCard from './PredictionCard.jsx';
import { fetchDashboard, usePolling } from '../lib/api.js';
import { timeAgo, formatReopen } from '../lib/helpers.js';
import { allowedSymbolsFor } from '../lib/plans.js';

/* ------------------------------------------------------------------ */
/* Status bar                                                          */
/* ------------------------------------------------------------------ */

function StatusBar({ lastUpdated, isRefreshing, strategy, onRefresh, counts }) {
  return (
    <div className="panel edge-top relative overflow-hidden">
      {isRefreshing && <div className="scanline absolute inset-x-0 top-0 h-px" />}

      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 px-4 py-2.5">
        <div className="flex items-center gap-2">
          <span
            className={`h-1.5 w-1.5 rounded-full ${
              isRefreshing ? 'bg-amber animate-pulse-tick' : 'bg-up'
            }`}
            aria-hidden="true"
          />
          <span className="label">
            {isRefreshing ? 'Refreshing' : 'Live'} · 60s poll
          </span>
        </div>

        <div className="flex items-center gap-2">
          <span className="label">Updated</span>
          <span className="tnum text-tick text-ink-muted">{timeAgo(lastUpdated)}</span>
        </div>

        <div className="flex items-center gap-2">
          <span className="label">Strategy</span>
          <span className="text-tick uppercase tracking-wider text-amber">
            {strategy}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="tnum text-tick text-up">{counts.up} up</span>
          <span className="tnum text-tick text-down">{counts.down} down</span>
          <span className="tnum text-tick text-flat">{counts.neutral} flat</span>
        </div>

        <button type="button" onClick={onRefresh} className="btn-ghost ml-auto">
          Refresh now
        </button>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Skeleton                                                            */
/* ------------------------------------------------------------------ */

function CardSkeleton() {
  return (
    <div className="panel h-[340px] animate-pulse-tick">
      <div className="px-4 py-3">
        <div className="h-4 w-24 bg-hairline" />
        <div className="mt-3 h-8 w-32 bg-hairline" />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Group                                                               */
/* ------------------------------------------------------------------ */

function AssetGroup({ title, hint, instruments, watchlist, onToggleWatch, canWatch }) {
  if (instruments.length === 0) return null;

  return (
    <section>
      <div className="mb-3 flex items-baseline gap-3 border-b border-divider pb-2">
        <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
          {title}
        </h2>
        <span className="text-micro uppercase text-ink-faint">{hint}</span>
        <span className="tnum ml-auto text-tick text-ink-faint">
          {instruments.length}
        </span>
      </div>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
        {instruments.map((instrument) => (
          <PredictionCard
            key={instrument.symbol}
            instrument={instrument}
            isWatched={watchlist.includes(instrument.symbol)}
            onToggleWatch={onToggleWatch}
            canWatch={canWatch}
          />
        ))}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Asset class filter                                                  */
/* ------------------------------------------------------------------ */

const ASSET_CLASS_TABS = [
  { key: 'all', label: 'All' },
  { key: 'crypto', label: 'Crypto' },
  { key: 'forex', label: 'Forex' },
];

function AssetClassFilter({ active, onChange }) {
  return (
    <div className="segment max-w-sm">
      {ASSET_CLASS_TABS.map((tab) => (
        <button
          key={tab.key}
          type="button"
          onClick={() => onChange(tab.key)}
          aria-pressed={active === tab.key}
          className={`segment-item ${active === tab.key ? 'segment-item-active' : ''}`}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dashboard                                                           */
/* ------------------------------------------------------------------ */

export default function Dashboard({ strategy, watchlist, onToggleWatch, canWatch, plan = 'free' }) {
  const [assetClass, setAssetClass] = useState('all');

  const fetcher = useCallback(
    (options) => fetchDashboard(strategy, options),
    [strategy]
  );

  const { data, error, loading, isRefreshing, lastUpdated, refresh } = usePolling(
    fetcher,
    60000,
    [strategy]
  );

  const allInstruments = data?.instruments || [];

  // Plan gating happens client-side only — see src/lib/plans.js's
  // enforcement note. The API still returns every instrument; this
  // just decides what gets rendered.
  const allowed = useMemo(
    () => new Set(allowedSymbolsFor(plan, allInstruments.map((i) => i.symbol))),
    [plan, allInstruments]
  );
  const instruments = useMemo(
    () => allInstruments.filter((i) => allowed.has(i.symbol)),
    [allInstruments, allowed]
  );
  const lockedCount = allInstruments.length - instruments.length;

  const counts = useMemo(
    () => ({
      up: instruments.filter((i) => i.signal === 'UP').length,
      down: instruments.filter((i) => i.signal === 'DOWN').length,
      neutral: instruments.filter((i) => i.signal === 'NEUTRAL').length,
    }),
    [instruments]
  );

  const crypto = instruments.filter((i) => i.assetClass === 'crypto');
  const forex = instruments.filter((i) => i.assetClass === 'forex');
  const forexMarketClosed = forex.some((i) => i.marketClosed);

  // Split upstream failures: a closed weekend market isn't a "feed down"
  // and shouldn't read as one — it gets its own calm, non-amber notice.
  const downFailures = (data?.failures || []).filter((f) => !f.marketClosed);
  const closedFailures = (data?.failures || []).filter((f) => f.marketClosed);

  /* ---- First load ---- */
  if (loading && !data) {
    return (
      <div className="space-y-6">
        <div className="panel h-12 animate-pulse-tick" />
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-3">
          {Array.from({ length: 9 }, (_, i) => (
            <CardSkeleton key={i} />
          ))}
        </div>
      </div>
    );
  }

  /* ---- Hard failure: nothing to show at all ---- */
  if (error && !data) {
    return (
      <div className="panel border-amber/40 p-6">
        <h2 className="label-amber">Analysis service unreachable</h2>
        <p className="mt-2 max-w-prose text-sm normal-case text-ink-muted">
          {error}
        </p>
        <p className="mt-3 max-w-prose text-tick text-ink-faint">
          Check that the Netlify functions are deployed and that
          SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and TWELVE_DATA_API_KEY are
          set. The function logs in Netlify will name the missing one.
        </p>
        <button type="button" onClick={refresh} className="btn-amber mt-4">
          Try again
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StatusBar
        lastUpdated={lastUpdated}
        isRefreshing={isRefreshing}
        strategy={data?.strategy?.label || strategy}
        onRefresh={refresh}
        counts={counts}
      />

      {/* Soft failure: stale data still on screen, warn without blanking. */}
      {error && data && (
        <div className="panel border-amber/40 px-4 py-2.5">
          <p className="text-tick normal-case text-amber">
            Last refresh failed — showing data from {timeAgo(lastUpdated)}. {error}
          </p>
        </div>
      )}

      {/* Plan gate — how many pairs are hidden, and where to unlock them. */}
      {lockedCount > 0 && (
        <div className="panel border-amber/30 px-4 py-2.5">
          <p className="text-tick normal-case text-amber">
            {lockedCount} more pair{lockedCount === 1 ? '' : 's'} available on a higher
            plan.{' '}
            <Link to="/settings" className="underline underline-offset-2">
              Upgrade
            </Link>
          </p>
        </div>
      )}

      {/* Per-instrument upstream failures — actual problems only. */}
      {downFailures.length > 0 && (
        <div className="panel border-amber/30 px-4 py-2.5">
          <p className="label-amber mb-1">Some feeds are down</p>
          <ul className="space-y-0.5">
            {downFailures.map((f) => (
              <li key={f.symbol} className="text-micro normal-case text-ink-muted">
                <span className="text-amber">{f.symbol}</span> — {f.error}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Closed weekend market — expected, not a failure, so no amber. */}
      {closedFailures.length > 0 && (
        <div className="panel border-hairline px-4 py-2.5">
          <p className="label mb-1">Forex market closed</p>
          <p className="text-micro normal-case text-ink-faint">
            {closedFailures.length} pair{closedFailures.length === 1 ? '' : 's'} have no
            weekend data cached yet.{' '}
            {closedFailures[0].reopensAt &&
              `Reopens ${formatReopen(closedFailures[0].reopensAt)}.`}
          </p>
        </div>
      )}

      <AssetClassFilter active={assetClass} onChange={setAssetClass} />

      {(assetClass === 'all' || assetClass === 'crypto') && (
        <AssetGroup
          title="Crypto"
          hint="Binance · 5m candles"
          instruments={crypto}
          watchlist={watchlist}
          onToggleWatch={onToggleWatch}
          canWatch={canWatch}
        />
      )}

      {(assetClass === 'all' || assetClass === 'forex') && (
        <AssetGroup
          title="Forex"
          hint={
            forexMarketClosed
              ? 'Twelve Data · 5m candles · market closed for the weekend'
              : 'Twelve Data · 5m candles · no consolidated volume'
          }
          instruments={forex}
          watchlist={watchlist}
          onToggleWatch={onToggleWatch}
          canWatch={canWatch}
        />
      )}
    </div>
  );
}
