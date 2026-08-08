/**
 * netlify/functions/get-analysis.js
 * -------------------------------------------------------------------
 * READ path. This is the only endpoint the frontend polls.
 *
 * GET /api/get-analysis
 *   ?strategy=aggressive|conservative   (default conservative)
 *   &symbol=BTC/USDT                    (optional — detail view)
 *   &history=1                           (include candles + prediction log)
 *
 * Two modes:
 *   Dashboard : one row per instrument, latest signal for the chosen strategy
 *   Detail    : as above plus the candle series, RSI series and the
 *               scored prediction log for a single symbol
 *
 * Why it recomputes rather than just reading the stored signal:
 * market_snapshots stores the CONSERVATIVE view. Switching to
 * Aggressive changes lookback lengths, so the signal has to be
 * recalculated from candles. Recomputing also means the dashboard is
 * never staler than the last upstream tick, even if the 5-minute cron
 * has just missed a beat.
 * -------------------------------------------------------------------
 */

import { createClient } from '@supabase/supabase-js';
import {
  INSTRUMENTS,
  findInstrument,
  fetchCandles,
  fetchDerivativesContext,
  fetchTwelveDataBatchCandles,
} from '../../shared/marketSources.js';
import rawModel from '../../shared/model.json';
import { loadModel, predictFromCandles, combineSignals } from '../../shared/mlModel.js';
import { alignCandlesByTime, buildCrossSectionalContext } from '../../shared/mlFeatures.js';
import {
  sma,
  rsi,
  rsiSeries,
  bollingerBands,
  vwap,
  realisedVolatility,
  computeSignal,
  findBestEntryWindow,
  getProfile,
  resolveStopLossPct,
} from '../../shared/indicators.js';

/* ---------------------------- CORS ---------------------------- */

const CORS_HEADERS = {
  'Access-Control-Allow-Origin': process.env.ALLOWED_ORIGIN || '*',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Content-Type': 'application/json',
  // Let Netlify's CDN absorb duplicate polls from multiple open tabs.
  'Cache-Control': 'public, max-age=30, stale-while-revalidate=60',
};

function json(statusCode, body, extraHeaders = {}) {
  return {
    statusCode,
    headers: { ...CORS_HEADERS, ...extraHeaders },
    body: JSON.stringify(body),
  };
}

function getAdminClient() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/* ------------------------- ML model ---------------------------- */

/**
 * The model is loaded once per cold start, not per request.
 *
 * A malformed or mismatched model must NOT take down the dashboard, so
 * a load failure is logged and the app falls back to the heuristic.
 * That is a real degradation, though, so it is surfaced in the response
 * as `mlError` rather than hidden.
 */
let ML_MODEL = null;
let ML_LOAD_ERROR = null;
try {
  ML_MODEL = loadModel(rawModel);
  if (ML_MODEL) {
    console.log(
      `[get-analysis] model loaded: trained on ${ML_MODEL.meta?.symbol}, ` +
        `verdict ${ML_MODEL.meta?.verdict}`
    );
  } else {
    console.log('[get-analysis] no model present — heuristic only');
  }
} catch (err) {
  ML_LOAD_ERROR = err.message;
  console.error('[get-analysis] model failed to load:', err.message);
}

// Set ML_SHOW_UNVALIDATED=true in Netlify to display a model that has
// not demonstrated an edge. Off by default: an unvalidated number on
// screen is worse than no number, because users will act on it.
const SHOW_UNVALIDATED = process.env.ML_SHOW_UNVALIDATED === 'true';

/* ----------------------- Cross-sectional + derivatives context ---- */

/**
 * Fetch every crypto instrument's candles ONCE per request (dashboard
 * mode) or once per detail request, shared across analyseInstrument()
 * calls — Phase 2.3's cross-sectional features need all pooled symbols'
 * candles, and fetching them independently per instrument would triple
 * the Binance calls for no benefit.
 */
async function fetchCryptoCandlesBundle(alreadyFetched = {}) {
  const cryptoInstruments = INSTRUMENTS.filter((i) => i.kind === 'crypto');
  const toFetch = cryptoInstruments.filter((i) => !alreadyFetched[i.symbol]);
  const settled = await Promise.allSettled(toFetch.map((i) => fetchCandles(i, 300)));

  const bySymbol = { ...alreadyFetched };
  settled.forEach((r, idx) => {
    if (r.status === 'fulfilled') bySymbol[toFetch[idx].symbol] = r.value;
    else console.error(`[get-analysis] sibling fetch failed for ${toFetch[idx].symbol}:`, r.reason?.message);
  });
  return bySymbol;
}

/**
 * Forex candles for the dashboard/detail read path, batched + cached.
 *
 * Binance has no meaningful rate limit, so crypto refetches live on every
 * request above. Twelve Data does not: it's 8 req/min and 800 credits/day,
 * and this function can be called on every /api/get-analysis poll (every
 * 60s per open browser tab). Fetching all 10 FX pairs individually per
 * request would burst well past 8 req/min on the first request alone.
 *
 * Two mitigations, both required:
 *   1. Always fetch via ONE batched Twelve Data call, same as ingest.
 *   2. Cache that batch in module scope for FOREX_CACHE_TTL_MS, aligned
 *      to the scheduled ingest's own 20-minute forex cadence (see
 *      schedule-task.js) — there's no fresher forex data to have anyway
 *      between ingest ticks.
 *
 * The cache is best-effort: a serverless cold start clears it, and under
 * concurrent cold instances each gets its own. It still turns the
 * worst case from "N individual calls per request" into "1 batched call
 * per warm instance per cache window," which is the difference between
 * blowing the daily credit cap in minutes versus staying inside it under
 * normal traffic. If Twelve Data's dashboard shows sustained throttling
 * despite this, the durable fix is to stop live-fetching forex on read
 * and serve it from persisted candles instead — a bigger change, flag it
 * rather than silently re-architecting this path.
 */
const FOREX_CACHE_TTL_MS = 20 * 60 * 1000;
let forexCandlesCache = { bySymbol: null, fetchedAt: 0 };

async function fetchForexCandlesBundle() {
  const now = Date.now();
  if (forexCandlesCache.bySymbol && now - forexCandlesCache.fetchedAt < FOREX_CACHE_TTL_MS) {
    return forexCandlesCache.bySymbol;
  }

  const forexInstruments = INSTRUMENTS.filter((i) => i.kind === 'forex');
  let batch = {};
  try {
    batch = await fetchTwelveDataBatchCandles(forexInstruments, '5min', 300);
  } catch (err) {
    // The whole batch request failed (network, auth, HTTP error). Fall
    // back to whatever's still in the cache rather than failing every
    // forex instrument in the response — stale forex data beats none.
    console.error('[get-analysis] forex batch fetch failed:', err.message);
    return forexCandlesCache.bySymbol || {};
  }

  const bySymbol = {};
  for (const instrument of forexInstruments) {
    const candlesOrErr = batch[instrument.symbol];
    if (candlesOrErr instanceof Error) {
      console.error(`[get-analysis] forex batch entry failed for ${instrument.symbol}:`, candlesOrErr.message);
      continue;
    }
    if (candlesOrErr) bySymbol[instrument.symbol] = candlesOrErr;
  }

  // Cache whatever we got, even if partial — a partial cache still beats
  // re-hitting Twelve Data for every symbol on the next request, and
  // failed symbols will retry naturally once the TTL expires.
  forexCandlesCache = { bySymbol, fetchedAt: now };
  return bySymbol;
}

/**
 * Build the (candles, context) pair the ML model actually predicts on.
 *
 * Cross-sectional alignment can trim a few trailing/leading rows where
 * symbols' timestamps don't intersect exactly, so the ML path uses its
 * own (possibly slightly shorter) aligned candle array — never the raw
 * `ownCandles` used for the heuristic/indicators/chart series, which
 * must stay exactly what they were before this feature existed.
 *
 * Forex and any failure along the way degrade gracefully to
 * (ownCandles, null context) — the same "context missing -> features
 * read 0" contract build_features()/buildFeatureMatrix() already define.
 */
async function buildMlInputs(instrument, ownCandles, cryptoCandlesBySymbol) {
  if (instrument.kind !== 'crypto' || !cryptoCandlesBySymbol) {
    return { mlCandles: ownCandles, mlContext: null };
  }
  try {
    const { bySymbol } = alignCandlesByTime(cryptoCandlesBySymbol);
    const alignedOwn = bySymbol[instrument.symbol];
    if (!alignedOwn || alignedOwn.length < 60) {
      return { mlCandles: ownCandles, mlContext: null };
    }

    const cross = buildCrossSectionalContext(bySymbol, instrument.symbol);
    let derivatives = null;
    try {
      derivatives = await fetchDerivativesContext(instrument, alignedOwn);
    } catch (err) {
      console.error(`[get-analysis] derivatives context failed for ${instrument.symbol}:`, err.message);
    }

    return {
      mlCandles: alignedOwn,
      mlContext: {
        basketRet: cross.basketRet,
        btcRet: cross.btcRet,
        fundingRate: derivatives?.fundingRate ?? null,
        openInterest: derivatives?.openInterest ?? null,
        topTraderRatio: derivatives?.topTraderRatio ?? null,
      },
    };
  } catch (err) {
    console.error(`[get-analysis] cross-sectional context failed for ${instrument.symbol}:`, err.message);
    return { mlCandles: ownCandles, mlContext: null };
  }
}

/* ----------------------- Analysis per symbol ------------------ */

async function analyseInstrument(
  instrument,
  profile,
  wantHistory,
  cryptoCandlesBySymbol = null,
  forexCandlesBySymbol = null
) {
  // Forex must never fall through to a live per-symbol Twelve Data call
  // here — that's exactly the individual-request burst
  // fetchForexCandlesBundle() exists to avoid. If the batch is missing
  // this symbol (a failed/partial fetch), surface that as a failure for
  // this instrument rather than silently falling back to a live call.
  if (instrument.kind === 'forex') {
    const candles = forexCandlesBySymbol?.[instrument.symbol];
    if (!candles) {
      throw new Error(`No forex candles available for ${instrument.symbol} this cycle.`);
    }
    return analyseFromCandles(instrument, profile, wantHistory, candles, null);
  }

  const candles = cryptoCandlesBySymbol?.[instrument.symbol] || (await fetchCandles(instrument, 300));
  return analyseFromCandles(instrument, profile, wantHistory, candles, cryptoCandlesBySymbol);
}

async function analyseFromCandles(instrument, profile, wantHistory, candles, cryptoCandlesBySymbol) {
  const closes = candles.map((c) => c.close);
  const price = closes[closes.length - 1];
  const prevClose = closes.length > 1 ? closes[closes.length - 2] : price;

  const smaValue = sma(closes, profile.smaPeriod);
  const rsiValue = rsi(closes, profile.rsiPeriod);
  const bb = bollingerBands(closes, profile.bbPeriod);
  const vwapResult = vwap(candles.slice(-288));
  const volatility = realisedVolatility(closes.slice(-96));

  const { signal, confidence, rule, reasons } = computeSignal(
    { price, smaValue, rsiValue, bb, vwapValue: vwapResult?.value ?? null },
    profile
  );

  const entryWindow = findBestEntryWindow(candles, {
    lookbackHours: 12,
    rsiPeriod: profile.rsiPeriod,
    // Stop distance depends on the asset class, not just the profile.
    stopLossPct: resolveStopLossPct(profile, instrument.kind),
  });

  /* ---- ML second opinion ----
   * The heuristic result above is never overwritten. The model produces
   * a separate opinion, and combineSignals() decides what to display —
   * defaulting to the heuristic whenever the model is unvalidated. Both
   * are returned so the UI can show the disagreement rather than hide it.
   */
  const heuristicResult = { signal, confidence, rule, reasons };
  let ml = null;
  if (ML_MODEL) {
    try {
      const { mlCandles, mlContext } = await buildMlInputs(instrument, candles, cryptoCandlesBySymbol);
      // Meta-labelling mode (Phase 0.3) predicts P(the heuristic's OWN
      // call is correct), so it needs that call passed in — it has no
      // direction of its own to offer.
      const primarySignal = ML_MODEL.meta?.mode === 'meta_label' ? signal : undefined;
      ml = predictFromCandles(ML_MODEL, mlCandles, { context: mlContext, primarySignal });
      if (ml?.available && !ml.isValidated && !SHOW_UNVALIDATED) {
        // Keep the prediction for transparency, but flag it clearly.
        ml.suppressed = true;
      }
    } catch (err) {
      console.error(`[get-analysis] inference failed for ${instrument.symbol}:`, err.message);
      ml = { available: false, reason: err.message };
    }
  }
  const combined = combineSignals(heuristicResult, ml);

  // 24h change, from ~288 five-minute bars back.
  const dayAgoIdx = Math.max(0, closes.length - 288);
  const dayAgoPrice = closes[dayAgoIdx];
  const change24h = dayAgoPrice ? ((price - dayAgoPrice) / dayAgoPrice) * 100 : 0;

  const payload = {
    symbol: instrument.symbol,
    name: instrument.name,
    assetClass: instrument.kind,
    price,
    prevClose,
    change24h,
    rsi: rsiValue,
    sma: smaValue,
    smaPeriod: profile.smaPeriod,
    vwap: vwapResult?.value ?? null,
    vwapIsVolumeWeighted: vwapResult?.weighted ?? false,
    volatility,
    bollinger: bb
      ? {
          upper: bb.upper,
          middle: bb.middle,
          lower: bb.lower,
          bandwidth: bb.bandwidth,
          percentB: bb.percentB,
          breakout: bb.breakout,
        }
      : null,
    // Heuristic verdict (unchanged, always present).
    signal,
    confidence,
    rule,
    reasons,
    // ML verdict and the combined display decision.
    ml,
    combined,
    entryWindow,
    lastTick: candles[candles.length - 1].openTime,
  };

  if (wantHistory) {
    // Trim to ~24h and thin to keep the JSON payload light.
    const recent = candles.slice(-288);
    const rsiVals = rsiSeries(
      candles.map((c) => c.close),
      profile.rsiPeriod
    ).slice(-288);
    const smaVals = recent.map((_, i) => {
      const abs = candles.length - 288 + i;
      return sma(closes.slice(0, abs + 1), profile.smaPeriod);
    });

    payload.series = recent.map((c, i) => ({
      t: c.openTime,
      close: c.close,
      high: c.high,
      low: c.low,
      volume: c.volume,
      rsi: rsiVals[i],
      sma: smaVals[i],
    }));
  }

  return payload;
}

/* ---------------------- Prediction log ------------------------ */

/**
 * Scored prediction history for a symbol — this is what the detail
 * view's accuracy table renders. Only rows that have already been
 * graded (actual_result set by schedule-task.js) count toward accuracy.
 */
async function getPredictionLog(supabase, symbol) {
  if (!supabase) return { rows: [], accuracy: null, graded: 0 };

  const { data, error } = await supabase
    .from('predictions')
    .select('id, symbol, predicted_direction, confidence, actual_result, price_at_prediction, price_at_resolution, created_at, resolved_at')
    .eq('symbol', symbol)
    .order('created_at', { ascending: false })
    .limit(50);

  if (error) {
    console.error('[get-analysis] prediction log query failed:', error.message);
    return { rows: [], accuracy: null, graded: 0, error: error.message };
  }

  const graded = (data || []).filter((r) => r.actual_result && r.actual_result !== 'PENDING');
  const correct = graded.filter((r) => r.actual_result === 'CORRECT').length;

  return {
    rows: data || [],
    graded: graded.length,
    accuracy: graded.length >= 5 ? Math.round((correct / graded.length) * 100) : null,
  };
}

/* ------------------------ Model metadata ----------------------- */

/**
 * What the frontend needs to describe the model honestly: what it was
 * trained on, what edge it actually demonstrated on held-out folds, and
 * whether that was enough to trust.
 */
function modelInfo() {
  if (ML_LOAD_ERROR) return { loaded: false, error: ML_LOAD_ERROR };
  if (!ML_MODEL) {
    return {
      loaded: false,
      reason: 'No model trained yet — running on the rule-based heuristic.',
    };
  }
  const wf = ML_MODEL.meta?.walk_forward || {};
  const calib = ML_MODEL.meta?.calibration || null;
  return {
    loaded: true,
    mode: ML_MODEL.meta?.mode === 'meta_label' ? 'meta_label' : 'direction',
    trainedOn: ML_MODEL.meta?.symbol ?? null,
    pooledWith: ML_MODEL.meta?.pooled_with ?? null,
    horizonBars: ML_MODEL.meta?.horizon_bars ?? null,
    tripleBarrier: ML_MODEL.meta?.triple_barrier ?? null,
    costBps: ML_MODEL.meta?.cost_bps ?? null,
    verdict: ML_MODEL.meta?.verdict ?? 'UNKNOWN',
    trainedRows: ML_MODEL.meta?.trained_rows ?? null,
    sampleUniqueness: ML_MODEL.meta?.sample_uniqueness ?? null,
    metaLabelThreshold: ML_MODEL.meta?.meta_label_threshold ?? null,
    // Measured on purged walk-forward folds — see ml/validation.py.
    walkForward: {
      accuracy: wf.accuracy_mean ?? null,
      baseRate: wf.base_rate_mean ?? null,
      edge: wf.edge_mean ?? null,
      edgeStd: wf.edge_std ?? null,
      foldsPositive: wf.folds_positive_edge ?? null,
      folds: wf.folds ?? null,
      netBps: wf.net_bps_mean ?? null,
      auc: wf.auc_mean ?? null,
    },
    // Phase 1.3 — Brier/ECE/MCE measured on a fresh held-out fold seen
    // by neither the base model nor the calibrator. Null if the model
    // was exported with --calibrate off.
    calibration: calib
      ? { method: calib.method, ece: calib.ece, mce: calib.mce }
      : null,
    showUnvalidated: SHOW_UNVALIDATED,
  };
}

/* ---------------------------- Handler ------------------------- */

export async function handler(event) {
  if (event.httpMethod === 'OPTIONS') {
    return { statusCode: 204, headers: CORS_HEADERS, body: '' };
  }
  if (event.httpMethod !== 'GET') {
    return json(405, { error: 'Method not allowed. Use GET.' });
  }

  const params = event.queryStringParameters || {};
  const profile = getProfile(params.strategy);
  const wantHistory = params.history === '1' || params.history === 'true';
  const supabase = getAdminClient();

  try {
    /* ---- Detail mode: one symbol, full depth ---- */
    if (params.symbol) {
      const instrument = findInstrument(params.symbol);
      if (!instrument) {
        return json(404, {
          error: `Unknown symbol "${params.symbol}".`,
          available: INSTRUMENTS.map((i) => i.symbol),
        });
      }

      // Cross-sectional context (Phase 2.3) needs the other pooled
      // crypto symbols' candles too, not just the requested one. Forex
      // always goes through the batched+cached bundle — never a live
      // single-symbol Twelve Data call from this read path.
      const cryptoCandlesBySymbol =
        instrument.kind === 'crypto' ? await fetchCryptoCandlesBundle() : null;
      const forexCandlesBySymbol =
        instrument.kind === 'forex' ? await fetchForexCandlesBundle() : null;

      const [analysis, log] = await Promise.all([
        analyseInstrument(instrument, profile, true, cryptoCandlesBySymbol, forexCandlesBySymbol),
        getPredictionLog(supabase, instrument.symbol),
      ]);

      return json(200, {
        ok: true,
        strategy: { name: params.strategy || 'conservative', ...profile },
        model: modelInfo(),
        generatedAt: new Date().toISOString(),
        instrument: analysis,
        predictionLog: log,
      });
    }

    /* ---- Dashboard mode: every instrument ---- */
    // Fetch every crypto instrument's candles once up front, shared by
    // every crypto instrument's cross-sectional context below (and by
    // its own analysis, avoiding a duplicate fetch of the same symbol).
    // Forex goes through the batched+cached bundle for the same reason —
    // 10 individual Twelve Data calls per dashboard poll would blow the
    // free tier's 8 req/min cap on the first request.
    const [cryptoCandlesBySymbol, forexCandlesBySymbol] = await Promise.all([
      fetchCryptoCandlesBundle(),
      fetchForexCandlesBundle(),
    ]);
    const settled = await Promise.allSettled(
      INSTRUMENTS.map((i) =>
        analyseInstrument(i, profile, wantHistory, cryptoCandlesBySymbol, forexCandlesBySymbol)
      )
    );

    const instruments = [];
    const failures = [];

    settled.forEach((r, idx) => {
      if (r.status === 'fulfilled') instruments.push(r.value);
      else {
        console.error(`[get-analysis] ${INSTRUMENTS[idx].symbol}:`, r.reason?.message);
        failures.push({
          symbol: INSTRUMENTS[idx].symbol,
          error: r.reason?.message || 'Unknown error',
        });
      }
    });

    if (instruments.length === 0) {
      return json(502, {
        error: 'No market data available from any source right now.',
        failures,
      });
    }

    return json(200, {
      ok: true,
      strategy: { name: params.strategy || 'conservative', ...profile },
      model: modelInfo(),
      generatedAt: new Date().toISOString(),
      instruments,
      failures,
    });
  } catch (err) {
    console.error('[get-analysis] unhandled:', err);
    return json(500, { error: err.message || 'Analysis failed.' });
  }
}
