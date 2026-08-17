/**
 * src/lib/helpers.js
 * -------------------------------------------------------------------
 * The frontend's window onto the shared indicator engine.
 *
 * All the RSI / SMA / Bollinger / VWAP math lives in one place —
 * /shared/indicators.js — so the browser and the serverless functions
 * can never disagree about what an indicator means. This file just
 * re-exports it, plus a few display-only helpers that have no business
 * being on the server.
 * -------------------------------------------------------------------
 */

export {
  sma,
  smaSeries,
  rsi,
  rsiSeries,
  bollingerBands,
  vwap,
  realisedVolatility,
  computeSignal,
  findBestEntryWindow,
  formatPrice,
  isForexPair,
  formatTime,
  formatDateTime,
  STRATEGY_PROFILES,
  getProfile,
  resolveStopLossPct,
} from '@shared/indicators.js';

/* ------------------------------------------------------------------ */
/* Display-only helpers                                                */
/* ------------------------------------------------------------------ */

/** Signed percentage, always with an explicit + or −. */
export function formatPct(value, digits = 2) {
  if (value == null || Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  return `${sign}${Math.abs(value).toFixed(digits)}%`;
}

/** Tailwind text class for a direction. */
export function signalColor(signal) {
  if (signal === 'UP') return 'text-up';
  if (signal === 'DOWN') return 'text-down';
  return 'text-flat';
}

/** Tailwind classes for a signal's border + background treatment. */
export function signalSurface(signal) {
  if (signal === 'UP') return 'border-up/30 bg-up-glow';
  if (signal === 'DOWN') return 'border-down/30 bg-down-glow';
  return 'border-hairline bg-transparent';
}

/** Tailwind text class for a price change. */
export function changeColor(value) {
  if (value == null) return 'text-flat';
  if (value > 0) return 'text-up';
  if (value < 0) return 'text-down';
  return 'text-flat';
}

/** Arrow glyph for a direction. */
export function signalGlyph(signal) {
  if (signal === 'UP') return '▲';
  if (signal === 'DOWN') return '▼';
  return '■';
}

/** "3m ago" style relative time. */
export function timeAgo(ts) {
  if (!ts) return '—';
  const seconds = Math.floor((Date.now() - new Date(ts).getTime()) / 1000);
  if (seconds < 10) return 'now';
  if (seconds < 60) return `${seconds}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

/** "Sun 22:00 UTC" style — when a closed market reopens. */
export function formatReopen(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  const weekday = d.toLocaleDateString('en-US', { weekday: 'short', timeZone: 'UTC' });
  const hh = String(d.getUTCHours()).padStart(2, '0');
  const mm = String(d.getUTCMinutes()).padStart(2, '0');
  return `${weekday} ${hh}:${mm} UTC`;
}

/** Compact volume: 1.2M, 940K. */
export function formatVolume(value) {
  if (value == null || Number.isNaN(value)) return '—';
  if (value === 0) return 'n/a';
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return value.toFixed(2);
}

/**
 * How an RSI reading should be described in words.
 * Thresholds are the conventional 30 / 70 pair.
 */
export function rsiZone(value) {
  if (value == null) return { label: 'no data', color: 'text-ink-faint' };
  if (value >= 70) return { label: 'overbought', color: 'text-down' };
  if (value >= 60) return { label: 'strong', color: 'text-amber' };
  if (value > 40) return { label: 'neutral', color: 'text-ink-muted' };
  if (value > 30) return { label: 'weak', color: 'text-amber' };
  return { label: 'oversold', color: 'text-up' };
}
