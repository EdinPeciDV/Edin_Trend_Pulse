/**
 * netlify/functions/fetch-market-data.js
 * -------------------------------------------------------------------
 * INGEST. For every instrument in the requested asset class:
 *   1. pull ~200 five-minute candles from the upstream API
 *   2. compute RSI, SMA, Bollinger Bands, VWAP, realised volatility
 *   3. derive the Up/Down/Neutral signal
 *   4. upsert one row into market_snapshots
 *
 * ?class=crypto|forex|all (default all) selects which instruments run.
 * ?group=0|1 (forex only, default 0) selects which HALF of the forex
 * registry runs — see FOREX_GROUP_SIZE below for why this exists.
 * schedule-task.js uses both to poll crypto every 5 minutes and forex
 * every 10, alternating groups so each forex symbol effectively refreshes
 * every 20 minutes — see that file for the full explanation.
 *
 * Crypto (Binance) is fetched serially, one call per instrument — no
 * meaningful rate limit there. Forex (Twelve Data) is fetched as ONE
 * batched call, but CAPPED to FOREX_GROUP_SIZE symbols per invocation:
 * Twelve Data's free tier is 8 API CREDITS/MINUTE (confirmed live — one
 * credit per symbol per call, batching does not reduce it), so a single
 * call for all 10 forex pairs 429s immediately regardless of batching.
 * This function is also the ONLY thing that ever calls Twelve Data —
 * get-analysis.js reads forex candles from the forex_candle_cache table
 * this function writes below, instead of calling Twelve Data itself.
 * That's not an optimisation, it's required: two independent uncoordinated
 * callers (this cron job and a 60s-polled dashboard) can't safely share
 * one 8-credit/minute budget without an occasional collision.
 *
 * Invoked by:
 *   - schedule-task.js on its cron tick (the normal path)
 *   - manually: GET /api/fetch-market-data (handy while developing;
 *     defaults to class=all, group=0 — see resolveForexGroup)
 *
 * Uses the SERVICE ROLE key, so it bypasses RLS. This key must never be
 * exposed to the browser — it lives only in Netlify env vars and is only
 * read inside functions.
 * -------------------------------------------------------------------
 */

import { createClient } from '@supabase/supabase-js';
import {
  INSTRUMENTS,
  fetchCandles,
  fetchTwelveDataBatchCandles,
} from '../../shared/marketSources.js';
import {
  sma,
  rsi,
  bollingerBands,
  vwap,
  realisedVolatility,
  computeSignal,
  getProfile,
} from '../../shared/indicators.js';

/* ---------------------------- CORS ---------------------------- */

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || '*',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
  'Content-Type': 'application/json',
};

function json(statusCode, body) {
  return { statusCode, headers: CORS_HEADERS, body: JSON.stringify(body) };
}

/* ------------------------ Supabase client --------------------- */

function getAdminClient() {
  const url = process.env.SUPABASE_URL;
  const serviceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

  if (!url || !serviceKey) {
    throw new Error(
      'SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must both be set. ' +
        'Add them in Netlify › Site configuration › Environment variables.'
    );
  }
  return createClient(url, serviceKey, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/* -------------------------- Asset class ------------------------ */

const VALID_CLASSES = ['crypto', 'forex', 'all'];

function resolveAssetClass(params) {
  const raw = (params?.class || 'all').toLowerCase();
  return VALID_CLASSES.includes(raw) ? raw : 'all';
}

function instrumentsForClass(assetClass) {
  if (assetClass === 'all') return INSTRUMENTS;
  return INSTRUMENTS.filter((i) => i.kind === assetClass);
}

// 10 forex symbols / 2 groups = 5 per call, comfortably under the
// 8-credits/minute cap even accounting for jitter. Never fetch more
// than one group in a single invocation — see the module comment.
const FOREX_GROUP_SIZE = 5;

function resolveForexGroup(params, forexInstruments) {
  if (forexInstruments.length === 0) return forexInstruments;
  const groups = [];
  for (let i = 0; i < forexInstruments.length; i += FOREX_GROUP_SIZE) {
    groups.push(forexInstruments.slice(i, i + FOREX_GROUP_SIZE));
  }
  const raw = Number(params?.group);
  const idx = Number.isInteger(raw) && groups[raw] ? raw : 0;
  return groups[idx];
}

/** Cache raw candles so get-analysis.js never has to call Twelve Data. */
async function cacheForexCandles(supabase, instrument, candles) {
  const { error } = await supabase.from('forex_candle_cache').upsert(
    { symbol: instrument.symbol, candles, updated_at: new Date().toISOString() },
    { onConflict: 'symbol' }
  );
  if (error) {
    console.error(
      `[fetch-market-data] candle cache write failed for ${instrument.symbol}:`,
      error.message
    );
  }
}

/* --------------------- Per-instrument work -------------------- */

/**
 * Build one snapshot row from already-fetched candles.
 * Snapshots are stored using the CONSERVATIVE profile so there is a
 * single canonical history; the aggressive view is recomputed on read
 * in get-analysis.js from the same candles.
 */
async function buildSnapshot(instrument, candles) {
  if (candles.length < 60) {
    throw new Error(
      `Only ${candles.length} candles for ${instrument.symbol} — need 60+ for a 50-period SMA.`
    );
  }

  const profile = getProfile('conservative');
  const closes = candles.map((c) => c.close);
  const price = closes[closes.length - 1];

  const smaValue = sma(closes, profile.smaPeriod);
  const rsiValue = rsi(closes, profile.rsiPeriod);
  const bb = bollingerBands(closes, profile.bbPeriod);
  const vwapResult = vwap(candles.slice(-Math.min(candles.length, 288))); // ~24h of 5m bars
  const volatility = realisedVolatility(closes.slice(-96));

  const { signal, confidence, rule, reasons } = computeSignal(
    { price, smaValue, rsiValue, bb, vwapValue: vwapResult?.value ?? null },
    profile
  );

  // Bucket the timestamp to the 5-minute boundary so re-runs inside the
  // same window update one row instead of inserting duplicates.
  const bucketMs = 5 * 60 * 1000;
  const bucket = new Date(Math.floor(Date.now() / bucketMs) * bucketMs);

  return {
    row: {
      symbol: instrument.symbol,
      asset_class: instrument.kind,
      price,
      rsi: rsiValue,
      sma_50: smaValue,
      bb_upper: bb?.upper ?? null,
      bb_lower: bb?.lower ?? null,
      bb_bandwidth: bb?.bandwidth ?? null,
      bb_breakout: bb?.breakout ?? null,
      vwap: vwapResult?.value ?? null,
      vwap_is_volume_weighted: vwapResult?.weighted ?? false,
      volatility,
      volume: candles[candles.length - 1].volume,
      signal,
      confidence,
      rule,
      reasons,
      bucket_at: bucket.toISOString(),
    },
    meta: { candleCount: candles.length },
  };
}

/* ---------------------------- Handler ------------------------- */

export async function handler(event) {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers: CORS_HEADERS, body: '' };
  }
  if (!['GET', 'POST'].includes(event.httpMethod)) {
    return json(405, { error: 'Method not allowed. Use GET or POST.' });
  }

  const startedAt = Date.now();
  let supabase;
  try {
    supabase = getAdminClient();
  } catch (err) {
    console.error('[fetch-market-data] config error:', err.message);
    return json(500, { error: err.message });
  }

  const assetClass = resolveAssetClass(event.queryStringParameters);
  const instruments = instrumentsForClass(assetClass);
  const cryptoInstruments = instruments.filter((i) => i.kind === 'crypto');
  const forexInstrumentsFull = instruments.filter((i) => i.kind === 'forex');
  const forexInstruments = resolveForexGroup(event.queryStringParameters, forexInstrumentsFull);

  const results = [];
  const rows = [];

  const recordSnapshot = async (instrument, candles) => {
    try {
      const { row, meta } = await buildSnapshot(instrument, candles);
      rows.push(row);
      results.push({
        symbol: instrument.symbol,
        ok: true,
        price: row.price,
        rsi: row.rsi == null ? null : Number(row.rsi.toFixed(2)),
        signal: row.signal,
        confidence: row.confidence,
        rule: row.rule,
        candles: meta.candleCount,
      });
    } catch (err) {
      // One bad instrument must not sink the whole run.
      console.error(`[fetch-market-data] ${instrument.symbol} failed:`, err.message);
      results.push({ symbol: instrument.symbol, ok: false, error: err.message });
    }
  };

  // Crypto: serial, one Binance call per instrument. No meaningful rate
  // limit on Binance's public REST, so this stays simple and fast.
  for (const instrument of cryptoInstruments) {
    try {
      const candles = await fetchCandles(instrument, 200);
      await recordSnapshot(instrument, candles);
    } catch (err) {
      console.error(`[fetch-market-data] ${instrument.symbol} failed:`, err.message);
      results.push({ symbol: instrument.symbol, ok: false, error: err.message });
    }
  }

  // Forex: ONE batched Twelve Data call, capped to FOREX_GROUP_SIZE
  // symbols (see resolveForexGroup) — never the full forex registry in
  // one call. Successful fetches are also cached to forex_candle_cache
  // so get-analysis.js can read them without calling Twelve Data itself.
  if (forexInstruments.length > 0) {
    let batch = {};
    try {
      batch = await fetchTwelveDataBatchCandles(forexInstruments, '5min', 300);
    } catch (err) {
      // The whole batch request failed (e.g. network, auth, HTTP error) —
      // every forex instrument in this run failed with it.
      console.error('[fetch-market-data] forex batch fetch failed:', err.message);
      for (const instrument of forexInstruments) {
        results.push({ symbol: instrument.symbol, ok: false, error: err.message });
      }
    }
    for (const instrument of forexInstruments) {
      const candlesOrErr = batch[instrument.symbol];
      if (candlesOrErr === undefined) continue; // already recorded above
      if (candlesOrErr instanceof Error) {
        console.error(`[fetch-market-data] ${instrument.symbol} failed:`, candlesOrErr.message);
        results.push({ symbol: instrument.symbol, ok: false, error: candlesOrErr.message });
        continue;
      }
      await recordSnapshot(instrument, candlesOrErr);
      await cacheForexCandles(supabase, instrument, candlesOrErr);
    }
  }

  if (rows.length === 0) {
    return json(502, {
      error: 'Every upstream fetch failed — nothing written.',
      results,
    });
  }

  // Upsert on (symbol, bucket_at) — see the unique index in migrations.sql.
  const { error: dbError } = await supabase
    .from('market_snapshots')
    .upsert(rows, { onConflict: 'symbol,bucket_at' });

  if (dbError) {
    console.error('[fetch-market-data] upsert failed:', dbError);
    return json(500, { error: `Supabase upsert failed: ${dbError.message}`, results });
  }

  return json(200, {
    ok: true,
    assetClass,
    written: rows.length,
    attempted: cryptoInstruments.length + forexInstruments.length,
    durationMs: Date.now() - startedAt,
    results,
  });
}
