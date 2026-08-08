/**
 * netlify/functions/schedule-task.js
 * -------------------------------------------------------------------
 * The cron job. Fires every 5 minutes via Netlify Scheduled Functions
 * (declared both here with schedule() and in netlify.toml), but not
 * every tick does the same amount of work:
 *
 *   - CRYPTO ingests every tick (Binance has no meaningful rate limit).
 *   - FOREX ingests only on ticks that land on a 20-minute boundary
 *     (:00/:20/:40 UTC) — Twelve Data's free tier is 8 req/min and
 *     800 credits/day, and 10 FX pairs batched every 20 minutes is
 *     72 x 10 = 720 credits/day. Ingesting forex every 5 minutes like
 *     crypto would be 4x that and blow the daily cap by lunchtime.
 *
 * Three jobs per tick:
 *   1. INGEST  — invoke the same logic as fetch-market-data, scoped to
 *                whichever asset class(es) this tick covers
 *   2. LOG     — write the current signal into `predictions` so accuracy
 *                can be measured later, only for what was just ingested
 *   3. GRADE   — resolve predictions that are now old enough to score
 *
 * Scheduled functions cannot be triggered by HTTP in production; Netlify
 * invokes them internally. In local dev use:
 *     netlify functions:invoke schedule-task
 *
 * Grading rule: a prediction is graded once GRADE_AFTER_MINUTES have
 * passed. UP is correct if price rose by more than the dead band, DOWN
 * if it fell by more. NEUTRAL is correct if price stayed inside the band.
 * The dead band stops tiny noise from being scored as a win.
 *
 * Grading reads the current price from `latest_snapshots` (the ingest
 * step already wrote it) instead of live-fetching upstream. A graded
 * prediction is already >= GRADE_AFTER_MINUTES old, so being up to one
 * ingest cycle stale (<=5min crypto, <=20min forex) doesn't matter — and
 * it means grading spends zero extra Twelve Data credits, no matter how
 * many predictions are pending.
 * -------------------------------------------------------------------
 */

import { schedule } from '@netlify/functions';
import { createClient } from '@supabase/supabase-js';
import { handler as ingestHandler } from './fetch-market-data.js';
import { INSTRUMENTS } from '../../shared/marketSources.js';

const GRADE_AFTER_MINUTES = 60; // how long a prediction gets to play out
const DEAD_BAND_PCT = 0.15; // moves smaller than this count as "flat"
const FOREX_INGEST_INTERVAL_MINUTES = 20;

/** True on ticks that fall on a 20-minute UTC boundary: :00, :20, :40. */
function isForexIngestTick(now = new Date()) {
  return now.getUTCMinutes() % FOREX_INGEST_INTERVAL_MINUTES === 0;
}

function getAdminClient() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) {
    throw new Error('SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY are not configured.');
  }
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/* -------------------- 2. Log current signals ------------------- */

/**
 * Insert one prediction row per instrument from the snapshots the ingest
 * step just wrote. These are system-level predictions (user_id null);
 * per-user predictions are written by the frontend when a signed-in user
 * follows a symbol.
 *
 * `instruments` is scoped to whatever this tick actually ingested — not
 * the full registry. Forex only ingests every 20 minutes now, so logging
 * every 5-minute tick regardless would insert 4 duplicate PENDING rows
 * off the same stale snapshot for every real forex signal.
 */
async function logPredictions(supabase, instruments) {
  const logged = [];

  for (const instrument of instruments) {
    const { data, error } = await supabase
      .from('market_snapshots')
      .select('symbol, price, signal, confidence, rsi, created_at')
      .eq('symbol', instrument.symbol)
      .order('created_at', { ascending: false })
      .limit(1);

    if (error || !data || data.length === 0) continue;
    const snap = data[0];

    // Don't log low-conviction noise.
    if (snap.confidence < 40) continue;

    const { error: insertError } = await supabase.from('predictions').insert({
      user_id: null,
      symbol: snap.symbol,
      predicted_direction: snap.signal,
      confidence: snap.confidence,
      price_at_prediction: snap.price,
      rsi_at_prediction: snap.rsi,
      actual_result: 'PENDING',
    });

    if (insertError) {
      console.error(`[schedule-task] log failed for ${snap.symbol}:`, insertError.message);
    } else {
      logged.push({ symbol: snap.symbol, direction: snap.signal, confidence: snap.confidence });
    }
  }

  return logged;
}

/* ----------------------- 3. Grade old ones --------------------- */

async function gradePredictions(supabase) {
  const cutoff = new Date(Date.now() - GRADE_AFTER_MINUTES * 60 * 1000).toISOString();

  const { data: pending, error } = await supabase
    .from('predictions')
    .select('id, symbol, predicted_direction, price_at_prediction, created_at')
    .eq('actual_result', 'PENDING')
    .lt('created_at', cutoff)
    .limit(200);

  if (error) {
    console.error('[schedule-task] pending query failed:', error.message);
    return { graded: 0, error: error.message };
  }
  if (!pending || pending.length === 0) return { graded: 0 };

  // One Supabase read for every distinct symbol's latest snapshot — no
  // upstream API calls. See the module comment for why this is safe.
  const symbols = [...new Set(pending.map((p) => p.symbol))];
  const priceBySymbol = {};

  const { data: latest, error: latestError } = await supabase
    .from('latest_snapshots')
    .select('symbol, price')
    .in('symbol', symbols);

  if (latestError) {
    console.error('[schedule-task] latest_snapshots query failed:', latestError.message);
  } else {
    for (const row of latest || []) {
      priceBySymbol[row.symbol] = row.price;
    }
  }

  let graded = 0;

  for (const p of pending) {
    const now = priceBySymbol[p.symbol];
    if (now == null || !p.price_at_prediction) continue;

    const movePct = ((now - p.price_at_prediction) / p.price_at_prediction) * 100;

    let result;
    if (Math.abs(movePct) < DEAD_BAND_PCT) {
      // Price essentially didn't move.
      result = p.predicted_direction === 'NEUTRAL' ? 'CORRECT' : 'INCORRECT';
    } else if (movePct > 0) {
      result = p.predicted_direction === 'UP' ? 'CORRECT' : 'INCORRECT';
    } else {
      result = p.predicted_direction === 'DOWN' ? 'CORRECT' : 'INCORRECT';
    }

    const { error: updateError } = await supabase
      .from('predictions')
      .update({
        actual_result: result,
        price_at_resolution: now,
        move_pct: movePct,
        resolved_at: new Date().toISOString(),
      })
      .eq('id', p.id);

    if (updateError) {
      console.error(`[schedule-task] grade update failed for ${p.id}:`, updateError.message);
    } else {
      graded += 1;
    }
  }

  return { graded, considered: pending.length };
}

/* ---------------------------- Handler ------------------------- */

const run = async () => {
  const startedAt = Date.now();
  console.log('[schedule-task] tick start', new Date().toISOString());

  let supabase;
  try {
    supabase = getAdminClient();
  } catch (err) {
    console.error('[schedule-task] config error:', err.message);
    return { statusCode: 500, body: JSON.stringify({ error: err.message }) };
  }

  // 1. Ingest — crypto every tick, forex only on a 20-minute boundary.
  // Reuses the exact same code path as the HTTP endpoint, scoped by class.
  const forexTick = isForexIngestTick(new Date());
  const ingestedInstruments = INSTRUMENTS.filter((i) => i.kind === 'crypto');

  let cryptoIngest = null;
  try {
    const res = await ingestHandler({
      httpMethod: 'POST',
      queryStringParameters: { class: 'crypto' },
    });
    cryptoIngest = JSON.parse(res.body);
    console.log(`[schedule-task] crypto ingest wrote ${cryptoIngest.written ?? 0} rows`);
  } catch (err) {
    console.error('[schedule-task] crypto ingest threw:', err.message);
    cryptoIngest = { error: err.message };
  }

  let forexIngest = null;
  if (forexTick) {
    try {
      const res = await ingestHandler({
        httpMethod: 'POST',
        queryStringParameters: { class: 'forex' },
      });
      forexIngest = JSON.parse(res.body);
      console.log(`[schedule-task] forex ingest wrote ${forexIngest.written ?? 0} rows`);
      ingestedInstruments.push(...INSTRUMENTS.filter((i) => i.kind === 'forex'));
    } catch (err) {
      console.error('[schedule-task] forex ingest threw:', err.message);
      forexIngest = { error: err.message };
    }
  }

  // 2. Log the fresh signals — only for what was actually just ingested.
  let logged = [];
  try {
    logged = await logPredictions(supabase, ingestedInstruments);
    console.log(`[schedule-task] logged ${logged.length} predictions`);
  } catch (err) {
    console.error('[schedule-task] logging threw:', err.message);
  }

  // 3. Grade matured predictions.
  let grading = { graded: 0 };
  try {
    grading = await gradePredictions(supabase);
    console.log(`[schedule-task] graded ${grading.graded} predictions`);
  } catch (err) {
    console.error('[schedule-task] grading threw:', err.message);
  }

  const summary = {
    ok: true,
    ranAt: new Date().toISOString(),
    durationMs: Date.now() - startedAt,
    forexTick,
    crypto: cryptoIngest,
    forex: forexIngest,
    logged: logged.length,
    graded: grading.graded,
  };

  console.log('[schedule-task] tick done', JSON.stringify(summary));
  return { statusCode: 200, body: JSON.stringify(summary) };
};

// Netlify cron syntax is standard 5-field UTC cron.
export const handler = schedule('*/5 * * * *', run);
