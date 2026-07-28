/**
 * shared/marketSources.js
 * -------------------------------------------------------------------
 * Adapters that normalise two very different upstream APIs into one
 * candle shape:
 *
 *   { openTime: number(ms), open, high, low, close, volume }
 *
 * Crypto  -> Binance public REST  (no API key required)
 * Forex   -> Twelve Data          (free API key required)
 *
 * Only ever imported by serverless functions — never by the browser,
 * because the Twelve Data key must not reach the client.
 * -------------------------------------------------------------------
 */

/* ------------------------------------------------------------------ */
/* Instrument registry                                                 */
/* ------------------------------------------------------------------ */

export const INSTRUMENTS = [
  // --- Crypto (Binance) ---
  { symbol: 'BTC/USDT', kind: 'crypto', source: 'binance', sourceSymbol: 'BTCUSDT', name: 'Bitcoin' },
  { symbol: 'ETH/USDT', kind: 'crypto', source: 'binance', sourceSymbol: 'ETHUSDT', name: 'Ethereum' },
  { symbol: 'SOL/USDT', kind: 'crypto', source: 'binance', sourceSymbol: 'SOLUSDT', name: 'Solana' },
  // --- Forex (Twelve Data) ---
  { symbol: 'EUR/USD', kind: 'forex', source: 'twelvedata', sourceSymbol: 'EUR/USD', name: 'Euro / US Dollar' },
  { symbol: 'GBP/JPY', kind: 'forex', source: 'twelvedata', sourceSymbol: 'GBP/JPY', name: 'Pound / Yen' },
];

export function findInstrument(symbol) {
  return INSTRUMENTS.find((i) => i.symbol === symbol) || null;
}

/* ------------------------------------------------------------------ */
/* Binance                                                             */
/* ------------------------------------------------------------------ */

/**
 * Binance host. NOTE: api.binance.com returns HTTP 451 from US-based IPs,
 * and Netlify's build/runtime region may well be US. If you see 451s in
 * the function logs, set BINANCE_HOST=api.binance.us in Netlify and use
 * the *USD pairs (BTCUSD not BTCUSDT). Alternatives that work globally
 * include api.kraken.com and api.coinbase.com.
 */
const BINANCE_HOST = process.env.BINANCE_HOST || 'api.binance.com';

/**
 * Fetch klines (candles) from Binance.
 * @param {string} sourceSymbol e.g. 'BTCUSDT'
 * @param {string} interval     e.g. '5m'
 * @param {number} limit        max 1000
 */
export async function fetchBinanceCandles(sourceSymbol, interval = '5m', limit = 200) {
  const url =
    `https://${BINANCE_HOST}/api/v3/klines` +
    `?symbol=${encodeURIComponent(sourceSymbol)}` +
    `&interval=${interval}&limit=${limit}`;

  const res = await fetch(url, {
    headers: { Accept: 'application/json' },
  });

  if (!res.ok) {
    const body = await res.text().catch(() => '');
    if (res.status === 451) {
      throw new Error(
        `Binance returned 451 (region blocked) for ${sourceSymbol}. ` +
          `Set BINANCE_HOST=api.binance.us and use *USD pairs, or switch source.`
      );
    }
    throw new Error(`Binance ${res.status} for ${sourceSymbol}: ${body.slice(0, 200)}`);
  }

  const raw = await res.json();
  if (!Array.isArray(raw)) {
    throw new Error(`Binance returned an unexpected payload for ${sourceSymbol}`);
  }

  // Kline array layout:
  // [0] openTime [1] open [2] high [3] low [4] close [5] volume ...
  return raw.map((k) => ({
    openTime: Number(k[0]),
    open: Number(k[1]),
    high: Number(k[2]),
    low: Number(k[3]),
    close: Number(k[4]),
    volume: Number(k[5]),
  }));
}

/* ------------------------------------------------------------------ */
/* Twelve Data                                                         */
/* ------------------------------------------------------------------ */

/**
 * Fetch time series from Twelve Data.
 *
 * Free tier: 8 requests/minute, 800/day. Two FX pairs polled every
 * 5 minutes is 576 requests/day — inside the daily cap, but you have
 * no headroom for manual refreshes, so the ingest function serialises
 * FX requests and tolerates individual failures.
 */
export async function fetchTwelveDataCandles(sourceSymbol, interval = '5min', outputsize = 200) {
  const apiKey = process.env.TWELVE_DATA_API_KEY;
  if (!apiKey) {
    throw new Error(
      'TWELVE_DATA_API_KEY is not set — forex ingest cannot run. ' +
        'Add it in Netlify › Site configuration › Environment variables.'
    );
  }

  const url =
    'https://api.twelvedata.com/time_series' +
    `?symbol=${encodeURIComponent(sourceSymbol)}` +
    `&interval=${interval}&outputsize=${outputsize}` +
    `&order=ASC&apikey=${apiKey}`;

  const res = await fetch(url, { headers: { Accept: 'application/json' } });
  if (!res.ok) {
    throw new Error(`Twelve Data HTTP ${res.status} for ${sourceSymbol}`);
  }

  const json = await res.json();

  // Twelve Data signals errors in the body with status:'error', HTTP 200.
  if (json.status === 'error') {
    throw new Error(
      `Twelve Data error for ${sourceSymbol}: ${json.message || 'unknown'} ` +
        `(code ${json.code || '?'})`
    );
  }
  if (!Array.isArray(json.values)) {
    throw new Error(`Twelve Data returned no values for ${sourceSymbol}`);
  }

  return json.values.map((v) => ({
    openTime: new Date(v.datetime + 'Z').getTime(),
    open: Number(v.open),
    high: Number(v.high),
    low: Number(v.low),
    close: Number(v.close),
    // Spot FX has no consolidated volume; Twelve Data returns 0 or omits it.
    volume: Number(v.volume) || 0,
  }));
}

/* ------------------------------------------------------------------ */
/* Derivatives context (Phase 2.1) — Binance USDT-M futures, no auth   */
/* ------------------------------------------------------------------ */

/**
 * Highest-value-first per PREDICTION_SPEC.md Phase 2.1: funding rate,
 * open interest, top-trader positioning. All three are free, unauthed
 * Binance futures endpoints. Each is event-indexed (funding prints every
 * ~8h; the stats endpoints print on their own cadence), so they are
 * forward-filled onto the candle grid — a value is "current" until the
 * next print, not interpolated toward one that hasn't happened yet.
 */
const FUTURES_HOST = process.env.BINANCE_FUTURES_HOST || 'fapi.binance.com';

async function fetchFuturesJson(path) {
  const res = await fetch(`https://${FUTURES_HOST}${path}`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) {
    throw new Error(`Binance futures ${res.status} for ${path}`);
  }
  return res.json();
}

async function fetchFundingRateHistory(sourceSymbol, limit = 200) {
  const rows = await fetchFuturesJson(
    `/fapi/v1/fundingRate?symbol=${encodeURIComponent(sourceSymbol)}&limit=${limit}`
  );
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`no funding rate data for ${sourceSymbol}`);
  }
  return rows
    .map((r) => ({ time: Number(r.fundingTime), value: Number(r.fundingRate) }))
    .sort((a, b) => a.time - b.time);
}

async function fetchFuturesStatsSeries(sourceSymbol, path, valueKey, period = '5m', limit = 200) {
  const rows = await fetchFuturesJson(
    `${path}?symbol=${encodeURIComponent(sourceSymbol)}&period=${period}&limit=${limit}`
  );
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`no data from ${path} for ${sourceSymbol}`);
  }
  return rows
    .map((r) => ({ time: Number(r.timestamp), value: Number(r[valueKey]) }))
    .sort((a, b) => a.time - b.time);
}

/** Forward-fill event-indexed {time,value} points onto `targetTimes`. */
function alignToBars(events, targetTimes) {
  const out = new Array(targetTimes.length).fill(null);
  if (!events.length) return out;
  let j = -1;
  for (let i = 0; i < targetTimes.length; i++) {
    while (j + 1 < events.length && events[j + 1].time <= targetTimes[i]) j++;
    out[i] = j >= 0 ? events[j].value : null;
  }
  return out;
}

/**
 * Fetch and align funding rate / open interest / top-trader ratio onto
 * `candles`' own openTime grid, for the live-inference feature path
 * (mirrors ml/data.py's fetch_binance_derivatives(), used for training).
 *
 * Crypto only — returns null for forex, since spot FX has no futures
 * derivatives market. Best-effort per series: a failed upstream call
 * degrades that one series to an all-null array rather than throwing,
 * because a missing derivatives feed must not take down the dashboard.
 */
export async function fetchDerivativesContext(instrument, candles) {
  if (instrument.kind !== 'crypto' || !candles?.length) return null;

  const sourceSymbol = instrument.sourceSymbol; // e.g. 'BTCUSDT' — same base symbol on USDT-M futures
  const targetTimes = candles.map((c) => c.openTime);
  const n = targetTimes.length;

  const [funding, oi, topTrader] = await Promise.allSettled([
    fetchFundingRateHistory(sourceSymbol),
    fetchFuturesStatsSeries(sourceSymbol, '/futures/data/openInterestHist', 'sumOpenInterest'),
    fetchFuturesStatsSeries(sourceSymbol, '/futures/data/topLongShortPositionRatio', 'longShortRatio'),
  ]);

  const pick = (settled, label) => {
    if (settled.status !== 'fulfilled') {
      console.error(`[marketSources] ${label} failed for ${sourceSymbol}:`, settled.reason?.message);
      return new Array(n).fill(null);
    }
    return alignToBars(settled.value, targetTimes);
  };

  return {
    fundingRate: pick(funding, 'funding rate'),
    openInterest: pick(oi, 'open interest'),
    topTraderRatio: pick(topTrader, 'top-trader ratio'),
  };
}

/* ------------------------------------------------------------------ */
/* Unified entry point                                                 */
/* ------------------------------------------------------------------ */

/**
 * Fetch normalised candles for any registered instrument.
 * Always returns oldest-first.
 */
export async function fetchCandles(instrument, limit = 200) {
  let candles;
  if (instrument.source === 'binance') {
    candles = await fetchBinanceCandles(instrument.sourceSymbol, '5m', limit);
  } else if (instrument.source === 'twelvedata') {
    candles = await fetchTwelveDataCandles(instrument.sourceSymbol, '5min', limit);
  } else {
    throw new Error(`Unknown source "${instrument.source}"`);
  }

  return candles
    .filter((c) => Number.isFinite(c.close) && c.close > 0)
    .sort((a, b) => a.openTime - b.openTime);
}
