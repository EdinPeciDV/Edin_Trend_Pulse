/**
 * netlify/functions/fetch-market-data.js
 * -------------------------------------------------------------------
 * INGEST. For every registered instrument:
 *   1. pull ~200 five-minute candles from the upstream API
 *   2. compute RSI, SMA, Bollinger Bands, VWAP, realised volatility
 *   3. derive the Up/Down/Neutral signal
 *   4. upsert one row into market_snapshots
 *
 * Invoked by:
 *   - schedule-task.js every 5 minutes (the normal path)
 *   - manually: GET /api/fetch-market-data (handy while developing)
 *
 * Uses the SERVICE ROLE key, so it bypasses RLS. This key must never be
 * exposed to the browser — it lives only in Netlify env vars and is only
 * read inside functions.
 * -------------------------------------------------------------------
 */

import { createClient } from '@supabase/supabase-js';
import { INSTRUMENTS, fetchCandles } from '../../shared/marketSources.js';
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

/* --------------------- Per-instrument work -------------------- */

/**
 * Build one snapshot row from live candles.
 * Snapshots are stored using the CONSERVATIVE profile so there is a
 * single canonical history; the aggressive view is recomputed on read
 * in get-analysis.js from the same candles.
 */
async function buildSnapshot(instrument) {
  const candles = await fetchCandles(instrument, 200);
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

  const results = [];
  const rows = [];

  // Serialised, not parallel: Twelve Data's free tier allows only
  // 8 requests/minute, and parallel bursts trip its rate limiter.
  for (const instrument of INSTRUMENTS) {
    try {
      const { row, meta } = await buildSnapshot(instrument);
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
    written: rows.length,
    attempted: INSTRUMENTS.length,
    durationMs: Date.now() - startedAt,
    results,
  });
}
