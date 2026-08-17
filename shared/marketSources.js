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
  // --- Crypto (Binance), USDT pairs ---
  {
    symbol: "BTC/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "BTCUSDT",
    name: "Bitcoin",
  },
  {
    symbol: "ETH/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "ETHUSDT",
    name: "Ethereum",
  },
  {
    symbol: "SOL/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "SOLUSDT",
    name: "Solana",
  },
  {
    symbol: "XRP/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "XRPUSDT",
    name: "XRP",
  },
  {
    symbol: "ADA/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "ADAUSDT",
    name: "Cardano",
  },
  {
    symbol: "DOGE/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "DOGEUSDT",
    name: "Dogecoin",
  },
  {
    symbol: "AVAX/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "AVAXUSDT",
    name: "Avalanche",
  },
  {
    symbol: "LINK/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "LINKUSDT",
    name: "Chainlink",
  },
  {
    symbol: "LTC/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "LTCUSDT",
    name: "Litecoin",
  },
  {
    symbol: "DOT/USDT",
    kind: "crypto",
    source: "binance",
    sourceSymbol: "DOTUSDT",
    name: "Polkadot",
  },
  // --- Forex (Twelve Data) ---
  {
    symbol: "EUR/USD",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "EUR/USD",
    name: "Euro / US Dollar",
  },
  {
    symbol: "GBP/JPY",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "GBP/JPY",
    name: "Pound / Yen",
  },
  {
    symbol: "USD/JPY",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "USD/JPY",
    name: "US Dollar / Yen",
  },
  {
    symbol: "GBP/USD",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "GBP/USD",
    name: "Pound / US Dollar",
  },
  {
    symbol: "AUD/USD",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "AUD/USD",
    name: "Australian Dollar / US Dollar",
  },
  {
    symbol: "USD/CAD",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "USD/CAD",
    name: "US Dollar / Canadian Dollar",
  },
  {
    symbol: "USD/CHF",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "USD/CHF",
    name: "US Dollar / Swiss Franc",
  },
  {
    symbol: "EUR/JPY",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "EUR/JPY",
    name: "Euro / Yen",
  },
  {
    symbol: "EUR/GBP",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "EUR/GBP",
    name: "Euro / Pound",
  },
  {
    symbol: "NZD/USD",
    kind: "forex",
    source: "twelvedata",
    sourceSymbol: "NZD/USD",
    name: "New Zealand Dollar / US Dollar",
  },
];

export function findInstrument(symbol) {
  return INSTRUMENTS.find((i) => i.symbol === symbol) || null;
}

/* ------------------------------------------------------------------ */
/* Forex market hours                                                  */
/* ------------------------------------------------------------------ */

/**
 * Spot FX trades 24/5: opens Sunday 22:00 UTC, closes Friday 22:00 UTC
 * (the conventional New York 17:00 ET close/open used industry-wide).
 * Crypto has no such window — this only applies to forex instruments.
 */
export function isForexMarketOpen(date = new Date()) {
  const day = date.getUTCDay(); // 0 Sun ... 6 Sat
  const hour = date.getUTCHours();
  if (day === 6) return false; // Saturday: closed all day
  if (day === 0 && hour < 22) return false; // Sunday before 22:00 UTC
  if (day === 5 && hour >= 22) return false; // Friday after 22:00 UTC
  return true;
}

/**
 * When the market next opens, or null if it's already open. Used to tell
 * the user "reopens Sun 22:00 UTC" instead of a bare "closed".
 */
export function nextForexOpenAt(date = new Date()) {
  if (isForexMarketOpen(date)) return null;
  const target = new Date(date);
  target.setUTCHours(22, 0, 0, 0);
  const daysUntilSunday = (7 - target.getUTCDay()) % 7;
  target.setUTCDate(target.getUTCDate() + daysUntilSunday);
  return target;
}

/* ------------------------------------------------------------------ */
/* Binance                                                             */
/* ------------------------------------------------------------------ */

/**
 * Binance host. api.binance.com returns HTTP 451 from US-based IPs, and
 * Netlify's runtime region is US — so production ingest hits 451 on every
 * crypto pair while local dev (non-US IP) works fine. api.binance.us
 * mirrors the same /api/v3/klines shape and accepts the same *USDT
 * symbols, so on a 451 we retry once against it before giving up.
 */
const BINANCE_HOST = process.env.BINANCE_HOST || "api.binance.com";
const BINANCE_FALLBACK_HOST = "api.binance.us";

function buildKlinesUrl(host, sourceSymbol, interval, limit) {
  return (
    `https://${host}/api/v3/klines` +
    `?symbol=${encodeURIComponent(sourceSymbol)}` +
    `&interval=${interval}&limit=${limit}`
  );
}

/**
 * Fetch klines (candles) from Binance.
 * @param {string} sourceSymbol e.g. 'BTCUSDT'
 * @param {string} interval     e.g. '5m'
 * @param {number} limit        max 1000
 */
export async function fetchBinanceCandles(
  sourceSymbol,
  interval = "5m",
  limit = 200,
) {
  let res = await fetch(
    buildKlinesUrl(BINANCE_HOST, sourceSymbol, interval, limit),
    {
      headers: { Accept: "application/json" },
    },
  );

  if (res.status === 451 && BINANCE_HOST !== BINANCE_FALLBACK_HOST) {
    res = await fetch(
      buildKlinesUrl(BINANCE_FALLBACK_HOST, sourceSymbol, interval, limit),
      {
        headers: { Accept: "application/json" },
      },
    );
  }

  if (!res.ok) {
    const body = await res.text().catch(() => "");
    if (res.status === 451) {
      throw new Error(
        `Binance returned 451 (region blocked) for ${sourceSymbol}, and the ` +
          `api.binance.us fallback also failed. ${body.slice(0, 200)}`,
      );
    }
    throw new Error(
      `Binance ${res.status} for ${sourceSymbol}: ${body.slice(0, 200)}`,
    );
  }

  const raw = await res.json();
  if (!Array.isArray(raw)) {
    throw new Error(
      `Binance returned an unexpected payload for ${sourceSymbol}`,
    );
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
 * Free tier: 8 requests/minute, 800 credits/day. Credits are charged per
 * SYMBOL, not per HTTP call — a batched request for 10 symbols still
 * costs 10 credits, but it's one request instead of ten, which is what
 * actually matters for the 8 req/min cap. With 10 FX pairs polled every
 * 20 minutes (72 ticks/day), that's 72 x 10 = 720 credits/day — inside
 * the daily cap, with ~80 credits of headroom for manual/local runs.
 */
function requireTwelveDataKey() {
  const apiKey = process.env.TWELVE_DATA_API_KEY;
  if (!apiKey) {
    throw new Error(
      "TWELVE_DATA_API_KEY is not set — forex ingest cannot run. " +
        "Add it in Netlify › Site configuration › Environment variables.",
    );
  }
  return apiKey;
}

/** Normalise one Twelve Data time_series response object into candles. */
function parseTwelveDataSeries(entry, sourceSymbol) {
  if (!entry) {
    throw new Error(`Twelve Data returned no data for ${sourceSymbol}`);
  }
  // Twelve Data signals errors in the body with status:'error', HTTP 200.
  if (entry.status === "error") {
    throw new Error(
      `Twelve Data error for ${sourceSymbol}: ${entry.message || "unknown"} ` +
        `(code ${entry.code || "?"})`,
    );
  }
  if (!Array.isArray(entry.values)) {
    throw new Error(`Twelve Data returned no values for ${sourceSymbol}`);
  }

  return entry.values.map((v) => ({
    openTime: new Date(v.datetime + "Z").getTime(),
    open: Number(v.open),
    high: Number(v.high),
    low: Number(v.low),
    close: Number(v.close),
    // Spot FX has no consolidated volume; Twelve Data returns 0 or omits it.
    volume: Number(v.volume) || 0,
  }));
}

/** Fetch time series for a single Twelve Data symbol. */
export async function fetchTwelveDataCandles(
  sourceSymbol,
  interval = "5min",
  outputsize = 200,
) {
  const apiKey = requireTwelveDataKey();

  const url =
    "https://api.twelvedata.com/time_series" +
    `?symbol=${encodeURIComponent(sourceSymbol)}` +
    `&interval=${interval}&outputsize=${outputsize}` +
    `&order=ASC&apikey=${apiKey}`;

  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`Twelve Data HTTP ${res.status} for ${sourceSymbol}`);
  }

  const json = await res.json();
  return parseTwelveDataSeries(json, sourceSymbol);
}

/**
 * Fetch time series for MULTIPLE Twelve Data instruments in a single HTTP
 * request (comma-separated `symbol=`), so N pairs cost N credits but only
 * ONE call against the 8 requests/minute cap.
 *
 * Twelve Data's batch shape only appears when more than one symbol is
 * requested: the top-level JSON is keyed by symbol, each value shaped
 * like the single-symbol response. With exactly one symbol it falls back
 * to the flat single-symbol shape, so that case is handled explicitly.
 *
 * Tolerates individual symbol failures — a bad/unsupported symbol in the
 * batch must not sink candles for the rest. Returns a map from each
 * instrument's registry `symbol` to either a candles array or an Error.
 */
export async function fetchTwelveDataBatchCandles(
  instruments,
  interval = "5min",
  outputsize = 200,
) {
  const out = {};
  if (!instruments || instruments.length === 0) return out;

  const apiKey = requireTwelveDataKey();
  const symbolList = instruments.map((i) => i.sourceSymbol).join(",");

  const url =
    "https://api.twelvedata.com/time_series" +
    `?symbol=${encodeURIComponent(symbolList)}` +
    `&interval=${interval}&outputsize=${outputsize}` +
    `&order=ASC&apikey=${apiKey}`;

  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    throw new Error(`Twelve Data HTTP ${res.status} for batch [${symbolList}]`);
  }
  const json = await res.json();

  if (instruments.length === 1) {
    const only = instruments[0];
    try {
      out[only.symbol] = parseTwelveDataSeries(json, only.sourceSymbol);
    } catch (err) {
      out[only.symbol] = err;
    }
    return out;
  }

  for (const instrument of instruments) {
    try {
      out[instrument.symbol] = parseTwelveDataSeries(
        json[instrument.sourceSymbol],
        instrument.sourceSymbol,
      );
    } catch (err) {
      out[instrument.symbol] = err;
    }
  }
  return out;
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
const FUTURES_HOST = process.env.BINANCE_FUTURES_HOST || "fapi.binance.com";

async function fetchFuturesJson(path) {
  const res = await fetch(`https://${FUTURES_HOST}${path}`, {
    headers: { Accept: "application/json" },
  });
  if (!res.ok) {
    throw new Error(`Binance futures ${res.status} for ${path}`);
  }
  return res.json();
}

async function fetchFundingRateHistory(sourceSymbol, limit = 200) {
  const rows = await fetchFuturesJson(
    `/fapi/v1/fundingRate?symbol=${encodeURIComponent(sourceSymbol)}&limit=${limit}`,
  );
  if (!Array.isArray(rows) || rows.length === 0) {
    throw new Error(`no funding rate data for ${sourceSymbol}`);
  }
  return rows
    .map((r) => ({ time: Number(r.fundingTime), value: Number(r.fundingRate) }))
    .sort((a, b) => a.time - b.time);
}

async function fetchFuturesStatsSeries(
  sourceSymbol,
  path,
  valueKey,
  period = "5m",
  limit = 200,
) {
  const rows = await fetchFuturesJson(
    `${path}?symbol=${encodeURIComponent(sourceSymbol)}&period=${period}&limit=${limit}`,
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
  if (instrument.kind !== "crypto" || !candles?.length) return null;

  const sourceSymbol = instrument.sourceSymbol; // e.g. 'BTCUSDT' — same base symbol on USDT-M futures
  const targetTimes = candles.map((c) => c.openTime);
  const n = targetTimes.length;

  const [funding, oi, topTrader] = await Promise.allSettled([
    fetchFundingRateHistory(sourceSymbol),
    fetchFuturesStatsSeries(
      sourceSymbol,
      "/futures/data/openInterestHist",
      "sumOpenInterest",
    ),
    fetchFuturesStatsSeries(
      sourceSymbol,
      "/futures/data/topLongShortPositionRatio",
      "longShortRatio",
    ),
  ]);

  const pick = (settled, label) => {
    if (settled.status !== "fulfilled") {
      console.error(
        `[marketSources] ${label} failed for ${sourceSymbol}:`,
        settled.reason?.message,
      );
      return new Array(n).fill(null);
    }
    return alignToBars(settled.value, targetTimes);
  };

  return {
    fundingRate: pick(funding, "funding rate"),
    openInterest: pick(oi, "open interest"),
    topTraderRatio: pick(topTrader, "top-trader ratio"),
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
  if (instrument.source === "binance") {
    candles = await fetchBinanceCandles(instrument.sourceSymbol, "5m", limit);
  } else if (instrument.source === "twelvedata") {
    candles = await fetchTwelveDataCandles(
      instrument.sourceSymbol,
      "5min",
      limit,
    );
  } else {
    throw new Error(`Unknown source "${instrument.source}"`);
  }

  return candles
    .filter((c) => Number.isFinite(c.close) && c.close > 0)
    .sort((a, b) => a.openTime - b.openTime);
}
