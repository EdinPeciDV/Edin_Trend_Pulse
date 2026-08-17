/**
 * src/components/PredictionCard.jsx
 * -------------------------------------------------------------------
 * One instrument, one card.
 *
 * The signature element is ConfidenceMeter: ten discrete blocks rather
 * than a smooth progress bar. A continuous bar implies a precision this
 * heuristic does not have. Ten blocks says "roughly 7 out of 10 of our
 * conditions agree", which is exactly what the number means — and it's
 * the same VU-meter vocabulary real trading desks use.
 * -------------------------------------------------------------------
 */

import { Link } from 'react-router-dom';
import {
  formatPrice,
  formatPct,
  formatTime,
  signalColor,
  signalSurface,
  signalGlyph,
  changeColor,
  rsiZone,
} from '../lib/helpers.js';

/* ------------------------------------------------------------------ */
/* Confidence meter — the signature                                    */
/* ------------------------------------------------------------------ */

function ConfidenceMeter({ confidence, signal }) {
  const filled = Math.round((confidence / 100) * 10);

  const litClass =
    signal === 'UP' ? 'bg-up' : signal === 'DOWN' ? 'bg-down' : 'bg-flat';

  return (
    <div className="flex items-center gap-2">
      <div
        className="flex gap-[3px]"
        role="meter"
        aria-valuenow={confidence}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Confidence ${confidence} percent`}
      >
        {Array.from({ length: 10 }, (_, i) => (
          <span
            key={i}
            className={`h-3 w-[6px] rounded-[1px] transition-colors duration-300 ${
              i < filled ? litClass : 'bg-hairline'
            }`}
          />
        ))}
      </div>
      <span className={`tnum text-tick font-medium ${signalColor(signal)}`}>
        {confidence}%
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Card                                                                */
/* ------------------------------------------------------------------ */

export default function PredictionCard({ instrument, isWatched, onToggleWatch, canWatch }) {
  const {
    symbol,
    name,
    assetClass,
    price,
    change24h,
    rsi,
    signal,
    confidence,
    entryWindow,
    reasons,
    bollinger,
    ml,
    combined,
    marketClosed,
  } = instrument;

  // Only surface the model on a card when it has actually earned it.
  // An unvalidated model gets no visual weight here — the detail page is
  // where the full comparison and its caveats live.
  const showMlBadge = ml?.available && ml.isValidated && combined?.agreement != null;

  const zone = rsiZone(rsi);

  return (
    <article
      className={`panel edge-top relative flex flex-col border ${signalSurface(
        signal
      )} animate-fade-up transition-colors duration-300`}
    >
      {/* ---- Header: ticker + watch toggle ---- */}
      <header className="flex items-start justify-between gap-3 px-4 pt-3 pb-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Link
              to={`/symbol/${encodeURIComponent(symbol)}`}
              className="font-sans text-base font-semibold text-ink hover:text-amber transition-colors"
            >
              {symbol}
            </Link>
            <span className="label border border-hairline px-1.5 py-0.5">
              {assetClass}
            </span>
            {marketClosed && (
              <span
                className="label border border-hairline px-1.5 py-0.5 text-ink-faint"
                title="Forex is closed for the weekend — showing the last known read."
              >
                closed
              </span>
            )}
          </div>
          <p className="mt-0.5 truncate text-tick text-ink-faint">{name}</p>
        </div>

        {canWatch && (
          <button
            type="button"
            onClick={() => onToggleWatch(symbol)}
            aria-pressed={isWatched}
            aria-label={isWatched ? `Unfollow ${symbol}` : `Follow ${symbol}`}
            className={`shrink-0 border px-1.5 py-0.5 text-tick transition-colors ${
              isWatched
                ? 'border-amber/50 text-amber bg-amber-glow'
                : 'border-hairline text-ink-faint hover:text-ink-muted hover:border-divider'
            }`}
          >
            {isWatched ? '★' : '☆'}
          </button>
        )}
      </header>

      {/* ---- Price + direction ---- */}
      <div className="flex items-end justify-between gap-4 px-4 pb-3">
        <div>
          <p className="tnum text-readout font-medium leading-none text-ink">
            {formatPrice(price, symbol)}
          </p>
          <p className={`tnum mt-1 text-tick ${changeColor(change24h)}`}>
            {formatPct(change24h)} <span className="text-ink-faint">24h</span>
          </p>
        </div>

        <div className="text-right">
          <div
            className={`flex items-center justify-end gap-1.5 ${signalColor(signal)}`}
          >
            <span className="text-xl leading-none" aria-hidden="true">
              {signalGlyph(signal)}
            </span>
            <span className="font-sans text-sm font-semibold tracking-wide">
              {signal}
            </span>
          </div>
          <div className="mt-2 flex justify-end">
            <ConfidenceMeter confidence={confidence} signal={signal} />
          </div>
          {showMlBadge && (
            <p
              className={`mt-1.5 text-micro uppercase ${
                combined.agreement ? 'text-up' : 'text-amber'
              }`}
              title={combined.note}
            >
              model {combined.agreement ? 'agrees' : 'disagrees'}
            </p>
          )}
        </div>
      </div>

      {/* ---- Indicator strip ---- */}
      <div className="grid grid-cols-3 border-t border-hairline">
        <div className="border-r border-hairline px-3 py-2">
          <p className="label">RSI</p>
          <p className={`tnum text-sm ${zone.color}`}>
            {rsi == null ? '—' : rsi.toFixed(1)}
          </p>
          <p className="text-micro text-ink-faint">{zone.label}</p>
        </div>
        <div className="border-r border-hairline px-3 py-2">
          <p className="label">Band %B</p>
          <p className="tnum text-sm text-ink">
            {bollinger ? (bollinger.percentB * 100).toFixed(0) : '—'}
          </p>
          <p className="text-micro text-ink-faint">
            {bollinger?.breakout
              ? `${bollinger.breakout.toLowerCase()} break`
              : 'in channel'}
          </p>
        </div>
        <div className="px-3 py-2">
          <p className="label">Vol</p>
          <p className="tnum text-sm text-ink">
            {bollinger ? `${(bollinger.bandwidth * 100).toFixed(2)}%` : '—'}
          </p>
          <p className="text-micro text-ink-faint">bandwidth</p>
        </div>
      </div>

      {/* ---- Best entry window ---- */}
      <div className="border-t border-hairline bg-void/40 px-4 py-2.5">
        <p className="label-amber">Best entry in last 12h</p>
        {entryWindow ? (
          <>
            <p className="tnum mt-1 text-tick text-ink">
              {formatTime(entryWindow.time)} @ {formatPrice(entryWindow.price, symbol)}
            </p>
            <p className="tnum text-micro text-ink-faint">
              stop {formatPrice(entryWindow.stopLoss, symbol)} (−
              {entryWindow.stopLossPct}%) · RSI {entryWindow.rsi.toFixed(0)} ·{' '}
              {entryWindow.quality}
            </p>
          </>
        ) : (
          <p className="mt-1 text-tick text-ink-faint">
            Not enough history in the window yet.
          </p>
        )}
      </div>

      {/* ---- Why ---- */}
      {reasons?.length > 0 && (
        <details className="group border-t border-hairline px-4 py-2">
          <summary className="label cursor-pointer list-none hover:text-ink-muted">
            Why this call
            <span className="ml-1 inline-block transition-transform group-open:rotate-90">
              ›
            </span>
          </summary>
          <ul className="mt-2 space-y-1">
            {reasons.map((r, i) => (
              <li key={i} className="flex gap-2 text-micro leading-relaxed text-ink-muted">
                <span className="text-amber-dim">·</span>
                <span className="normal-case tracking-normal">{r}</span>
              </li>
            ))}
          </ul>
        </details>
      )}

      {/* ---- Footer ---- */}
      <Link
        to={`/symbol/${encodeURIComponent(symbol)}`}
        className="mt-auto border-t border-hairline px-4 py-2 text-tick uppercase tracking-wider text-ink-faint transition-colors hover:bg-raised hover:text-amber"
      >
        Open chart →
      </Link>
    </article>
  );
}
