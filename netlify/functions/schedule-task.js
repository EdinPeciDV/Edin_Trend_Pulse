/**
 * netlify/functions/schedule-task.js
 * -------------------------------------------------------------------
 * The cron job. Runs every 5 minutes via Netlify Scheduled Functions
 * (declared both here with schedule() and in netlify.toml).
 *
 * Three jobs per tick:
 *   1. INGEST  — invoke the same logic as fetch-market-data
 *   2. LOG     — write the current signal into `predictions` so accuracy
 *                can be measured later
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
 * -------------------------------------------------------------------
 */

import { schedule } from '@netlify/functions';
import { createClient } from '@supabase/supabase-js';
import { handler as ingestHandler } from './fetch-market-data.js';
import { INSTRUMENTS, fetchCandles } from '../../shared/marketSources.js';

const GRADE_AFTER_MINUTES = 60; // how long a prediction gets to play out
const DEAD_BAND_PCT = 0.15; // moves smaller than this count as "flat"

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
 */
async function logPredictions(supabase) {
  const logged = [];

  for (const instrument of INSTRUMENTS) {
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

  // One upstream fetch per distinct symbol, not per prediction.
  const symbols = [...new Set(pending.map((p) => p.symbol))];
  const priceBySymbol = {};

  for (const symbol of symbols) {
    const instrument = INSTRUMENTS.find((i) => i.symbol === symbol);
    if (!instrument) continue;
    try {
      const candles = await fetchCandles(instrument, 5);
      priceBySymbol[symbol] = candles[candles.length - 1].close;
    } catch (err) {
      console.error(`[schedule-task] grade fetch failed for ${symbol}:`, err.message);
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

  // 1. Ingest — reuse the exact same code path as the HTTP endpoint.
  let ingest = null;
  try {
    const res = await ingestHandler({ httpMethod: 'POST', queryStringParameters: {} });
    ingest = JSON.parse(res.body);
    console.log(`[schedule-task] ingest wrote ${ingest.written ?? 0} rows`);
  } catch (err) {
    console.error('[schedule-task] ingest threw:', err.message);
    ingest = { error: err.message };
  }

  // 2. Log the fresh signals.
  let logged = [];
  try {
    logged = await logPredictions(supabase);
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
    ingest,
    logged: logged.length,
    graded: grading.graded,
  };

  console.log('[schedule-task] tick done', JSON.stringify(summary));
  return { statusCode: 200, body: JSON.stringify(summary) };
};

// Netlify cron syntax is standard 5-field UTC cron.
export const handler = schedule('*/5 * * * *', run);
