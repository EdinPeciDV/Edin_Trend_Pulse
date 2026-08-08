/**
 * src/lib/plans.js
 * -------------------------------------------------------------------
 * Single source of truth for what each subscription tier unlocks.
 * Imported by Settings.jsx (billing panel), Dashboard.jsx (instrument
 * gating) and DetailView.jsx (history gating) so the three stay
 * consistent by construction, not by convention.
 *
 * ENFORCEMENT NOTE — read before assuming this is a security boundary.
 * This file gates what the UI RENDERS. /api/get-analysis has no concept
 * of plans and returns every instrument's full data to anyone who calls
 * it directly (curl, devtools, a second browser tab with a spoofed
 * plan) — that's a real, disclosed gap, not an oversight. Closing it
 * means threading the caller's plan into get-analysis.js (via a
 * Supabase access token in the request) and filtering server-side,
 * which is a materially bigger change than what was asked for here.
 * Treat this as a UX-layer gate today.
 * -------------------------------------------------------------------
 */

// Tiers are nested subsets by design: every symbol Free unlocks is also in
// Basic, and every symbol Basic unlocks is also in Pro (which gets all 20
// via symbols: null). Upgrading always ADDS pairs, never swaps them out.
const FREE_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'EUR/USD', 'USD/JPY', 'GBP/USD'];
const BASIC_SYMBOLS = [
  'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LINK/USDT',
  'EUR/USD', 'USD/JPY', 'GBP/USD', 'GBP/JPY', 'AUD/USD', 'USD/CAD', 'USD/CHF', 'EUR/JPY',
];

export const PLAN_LIMITS = {
  free: {
    key: 'free',
    label: 'Free',
    priceLabel: '$0',
    maxInstruments: FREE_SYMBOLS.length, // 3 crypto + 3 forex
    historyDays: 1,
    // null symbols = "every instrument"; an explicit list = only these.
    symbols: FREE_SYMBOLS,
    paddlePriceId: null,
  },
  basic: {
    key: 'basic',
    label: 'Basic',
    priceLabel: '$19/mo',
    maxInstruments: BASIC_SYMBOLS.length, // 8 crypto + 8 forex
    historyDays: 7,
    symbols: BASIC_SYMBOLS,
    paddlePriceId: import.meta.env.VITE_PADDLE_PRICE_ID_BASIC || null,
  },
  pro: {
    key: 'pro',
    label: 'Pro',
    priceLabel: '$49/mo',
    maxInstruments: 20, // every instrument — keep in sync with shared/marketSources.js INSTRUMENTS
    historyDays: 30,
    symbols: null,
    paddlePriceId: import.meta.env.VITE_PADDLE_PRICE_ID_PRO || null,
  },
};

export function planLimitsFor(planKey) {
  return PLAN_LIMITS[planKey] || PLAN_LIMITS.free;
}

/** Filter `allSymbols` (in whatever order the API returned them) down to what `planKey` may see. */
export function allowedSymbolsFor(planKey, allSymbols) {
  const limits = planLimitsFor(planKey);
  if (!limits.symbols) return allSymbols;
  const allowed = new Set(limits.symbols);
  return allSymbols.filter((s) => allowed.has(s));
}

export function isSymbolAllowed(planKey, symbol) {
  const limits = planLimitsFor(planKey);
  return !limits.symbols || limits.symbols.includes(symbol);
}

/** Trim a chronologically-descending list of rows (predictions, etc.) to `planKey`'s history window. */
export function withinHistoryWindow(planKey, rows, getDate = (r) => r.created_at) {
  const limits = planLimitsFor(planKey);
  const cutoff = Date.now() - limits.historyDays * 24 * 60 * 60 * 1000;
  return rows.filter((r) => {
    const t = new Date(getDate(r)).getTime();
    return Number.isFinite(t) && t >= cutoff;
  });
}
