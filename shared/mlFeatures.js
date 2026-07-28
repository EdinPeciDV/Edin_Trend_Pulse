/**
 * shared/mlFeatures.js
 * -------------------------------------------------------------------
 * JS twin of ml/features.py (feature computation only — labelling is a
 * training-time concern and has no JS twin).
 *
 * These two files MUST produce identical numbers. If they drift, the
 * model receives features in a different distribution from the ones it
 * was trained on and its output becomes noise — silently, with no error
 * anywhere. Nothing in the app would tell you.
 *
 * ml/parity.py exists to prevent exactly that: it runs both
 * implementations over the same candles (and the same context) and
 * fails if any feature differs by more than 1e-9.
 *
 * RULES FOR EDITING:
 *   1. Change a feature here -> change it in ml/features.py -> re-run
 *      `python3 ml/parity.py` -> retrain. All four steps, every time.
 *   2. Never reorder FEATURE_NAMES. Order is part of the model contract
 *      and model.json stores it; loadModel() asserts on mismatch.
 *   3. Only past bars. Feature at index i may read 0..i and no further.
 * -------------------------------------------------------------------
 */

/** Must match FEATURE_NAMES in ml/features.py, in order. */
export const FEATURE_NAMES = [
  'rsi_14_c',
  'rsi_9_c',
  'rsi_slope',
  'log_p_sma20',
  'log_p_sma50',
  'bb_pctb_c',
  'bb_bandwidth',
  'log_p_vwap',
  'ret_1',
  'ret_3',
  'ret_6',
  'ret_12',
  'vol_12',
  'vol_ratio',
  'atr_norm',
  'volume_z',
  'macd_hist_norm',
  'hour_sin',
  'hour_cos',
  // Phase 6: regime conditioning
  'adx_norm',
  // Phase 2.3: cross-sectional (pooled crypto training only)
  'rel_strength_basket',
  'corr_to_btc',
  // Phase 2.1: crypto derivatives context
  'funding_z',
  'oi_roc_z',
  'top_trader_z',
];

export const N_FEATURES = FEATURE_NAMES.length;

/** Features that read 0 unless `context` supplies the relevant series. */
export const CONTEXT_GATED_FEATURES = new Set([
  'rel_strength_basket',
  'corr_to_btc',
  'funding_z',
  'oi_roc_z',
  'top_trader_z',
]);

/* ------------------------------------------------------------------ */
/* Rolling primitives — all backward-looking                           */
/* ------------------------------------------------------------------ */

function rollingMean(x, w) {
  // Direct windowed sum rather than a running add/subtract. Slower
  // (O(n*w)) but exact: incremental sums accumulate different rounding
  // error than NumPy's cumsum, which broke feature parity at ~1e-9.
  // Windows here are <=576 over <=~20000 bars, so the cost is irrelevant.
  const n = x.length;
  const out = new Array(n).fill(NaN);
  if (n < w) return out;
  for (let i = w - 1; i < n; i++) {
    let sum = 0;
    for (let k = i - w + 1; k <= i; k++) sum += x[k];
    out[i] = sum / w;
  }
  return out;
}

function rollingStd(x, w) {
  // Two-pass (mean, then squared deviations) instead of the
  // sum-of-squares shortcut. The shortcut suffers catastrophic
  // cancellation when the mean is large relative to the spread — which
  // is exactly the case for price series — and that is what made the
  // Bollinger features disagree between Python and JS.
  const n = x.length;
  const out = new Array(n).fill(NaN);
  if (n < w) return out;
  for (let i = w - 1; i < n; i++) {
    let sum = 0;
    for (let k = i - w + 1; k <= i; k++) sum += x[k];
    const mean = sum / w;
    let acc = 0;
    for (let k = i - w + 1; k <= i; k++) {
      const d = x[k] - mean;
      acc += d * d;
    }
    out[i] = Math.sqrt(Math.max(acc / w, 0));
  }
  return out;
}

/** Backward rolling Pearson correlation over window w, two-pass. */
function rollingCorr(x, y, w) {
  const n = x.length;
  const out = new Array(n).fill(NaN);
  if (n < w) return out;
  for (let i = w - 1; i < n; i++) {
    let sx = 0;
    let sy = 0;
    for (let k = i - w + 1; k <= i; k++) {
      sx += x[k];
      sy += y[k];
    }
    const mx = sx / w;
    const my = sy / w;
    let cov = 0;
    let vx = 0;
    let vy = 0;
    for (let k = i - w + 1; k <= i; k++) {
      const dx = x[k] - mx;
      const dy = y[k] - my;
      cov += dx * dy;
      vx += dx * dx;
      vy += dy * dy;
    }
    cov /= w;
    const denom = Math.sqrt(vx / w) * Math.sqrt(vy / w);
    out[i] = denom > 0 ? cov / denom : 0;
  }
  return out;
}

/**
 * Backward rolling z-score, NaN/undefined collapsed to 0 (neutral) and
 * clipped to +/-`clip`. Missing or flat context series read as "nothing
 * unusual" rather than propagating into the model.
 */
function rollingZscore(x, w, clip = 5) {
  const mean = rollingMean(x, w);
  const sd = rollingStd(x, w);
  const n = x.length;
  const out = new Array(n).fill(0);
  for (let i = 0; i < n; i++) {
    let z = sd[i] > 0 ? (x[i] - mean[i]) / sd[i] : 0;
    if (!Number.isFinite(z)) z = 0;
    out[i] = Math.max(-clip, Math.min(clip, z));
  }
  return out;
}

/** Wilder RSI series — identical definition to wilder_rsi() in Python. */
function wilderRsiSeries(closes, period) {
  const n = closes.length;
  const out = new Array(n).fill(NaN);
  if (n < period + 1) return out;

  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const d = closes[i] - closes[i - 1];
    if (d > 0) gainSum += d;
    else if (d < 0) lossSum -= d;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;

  const rsiOf = (g, l) => (l === 0 ? 100 : g === 0 ? 0 : 100 - 100 / (1 + g / l));
  out[period] = rsiOf(avgGain, avgLoss);

  for (let i = period + 1; i < n; i++) {
    // NOTE: the delta for bar i is closes[i] - closes[i-1]. Using the
    // previous bar's delta here is an off-by-one that shifts the whole
    // RSI series by one bar — caught by ml/parity.py.
    const d = closes[i] - closes[i - 1];
    const g = d > 0 ? d : 0;
    const l = d < 0 ? -d : 0;
    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;
    out[i] = rsiOf(avgGain, avgLoss);
  }
  return out;
}

/** EMA with alpha = 2/(span+1), seeded on x[0] — matches Python _ema. */
function ema(x, span) {
  const n = x.length;
  const out = new Array(n).fill(NaN);
  if (n === 0) return out;
  const alpha = 2 / (span + 1);
  out[0] = x[0];
  for (let i = 1; i < n; i++) out[i] = alpha * x[i] + (1 - alpha) * out[i - 1];
  return out;
}

/** Wilder-smoothed ATR — matches Python atr(). */
function atrSeries(high, low, close, period) {
  const n = close.length;
  const out = new Array(n).fill(NaN);
  if (n < period + 1) return out;

  const tr = new Array(n);
  for (let i = 0; i < n; i++) {
    const pc = i === 0 ? close[0] : close[i - 1];
    tr[i] = Math.max(high[i] - low[i], Math.abs(high[i] - pc), Math.abs(low[i] - pc));
  }

  let a = 0;
  for (let i = 1; i <= period; i++) a += tr[i];
  a /= period;
  out[period] = a;

  for (let i = period + 1; i < n; i++) {
    a = (a * (period - 1) + tr[i]) / period;
    out[i] = a;
  }
  return out;
}

/**
 * Average Directional Index, Wilder-smoothed throughout — matches
 * Python adx() exactly, including its choice to smooth DM/TR as running
 * averages (not sums) so the whole file uses one smoothing idiom.
 */
function adxSeries(high, low, close, period) {
  const n = close.length;
  const out = new Array(n).fill(NaN);
  if (n < 2 * period + 1) return out;

  const tr = new Array(n);
  const plusDm = new Array(n);
  const minusDm = new Array(n);
  tr[0] = high[0] - low[0];
  plusDm[0] = 0;
  minusDm[0] = 0;
  for (let i = 1; i < n; i++) {
    const pc = close[i - 1];
    tr[i] = Math.max(high[i] - low[i], Math.abs(high[i] - pc), Math.abs(low[i] - pc));
    const upMove = high[i] - high[i - 1];
    const downMove = low[i - 1] - low[i];
    plusDm[i] = upMove > downMove && upMove > 0 ? upMove : 0;
    minusDm[i] = downMove > upMove && downMove > 0 ? downMove : 0;
  }

  const trAvg = new Array(n).fill(0);
  const pdmAvg = new Array(n).fill(0);
  const mdmAvg = new Array(n).fill(0);
  let trSum = 0;
  let pdmSum = 0;
  let mdmSum = 0;
  for (let i = 1; i <= period; i++) {
    trSum += tr[i];
    pdmSum += plusDm[i];
    mdmSum += minusDm[i];
  }
  trAvg[period] = trSum / period;
  pdmAvg[period] = pdmSum / period;
  mdmAvg[period] = mdmSum / period;
  for (let i = period + 1; i < n; i++) {
    trAvg[i] = (trAvg[i - 1] * (period - 1) + tr[i]) / period;
    pdmAvg[i] = (pdmAvg[i - 1] * (period - 1) + plusDm[i]) / period;
    mdmAvg[i] = (mdmAvg[i - 1] * (period - 1) + minusDm[i]) / period;
  }

  const dx = new Array(n).fill(NaN);
  for (let i = period; i < n; i++) {
    if (trAvg[i] > 0) {
      const plusDi = (100 * pdmAvg[i]) / trAvg[i];
      const minusDi = (100 * mdmAvg[i]) / trAvg[i];
      const s = plusDi + minusDi;
      dx[i] = s > 0 ? (100 * Math.abs(plusDi - minusDi)) / s : 0;
    }
  }

  const start = period * 2 - 1;
  if (start >= n) return out;
  let seedSum = 0;
  let seedOk = true;
  for (let i = period; i < period * 2; i++) {
    if (!Number.isFinite(dx[i])) {
      seedOk = false;
      break;
    }
    seedSum += dx[i];
  }
  if (!seedOk) return out;
  out[start] = seedSum / period;
  for (let i = start + 1; i < n; i++) {
    const dxi = Number.isFinite(dx[i]) ? dx[i] : 0;
    out[i] = (out[i - 1] * (period - 1) + dxi) / period;
  }
  return out;
}

/** Backward rolling VWAP with the FX no-volume fallback. */
function rollingVwap(high, low, close, volume, w) {
  const n = close.length;
  const typical = new Array(n);
  const pv = new Array(n);
  for (let i = 0; i < n; i++) {
    typical[i] = (high[i] + low[i] + close[i]) / 3;
    pv[i] = typical[i] * volume[i];
  }
  const pvMean = rollingMean(pv, w);
  const vMean = rollingMean(volume, w);
  const tpMean = rollingMean(typical, w);

  const out = new Array(n).fill(NaN);
  for (let i = 0; i < n; i++) {
    const pvSum = pvMean[i] * w;
    const vSum = vMean[i] * w;
    const weighted = pvSum / vSum;
    out[i] = vSum > 0 && Number.isFinite(weighted) ? weighted : tpMean[i];
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Feature matrix                                                      */
/* ------------------------------------------------------------------ */

/**
 * Build the feature matrix from candles.
 *
 * @param {Array<{openTime:number,open:number,high:number,low:number,close:number,volume:number}>} candles
 *        oldest first.
 * @param {{basketRet?:number[], btcRet?:number[], fundingRate?:number[],
 *          openInterest?:number[], topTraderRatio?:number[]}} [context]
 *        Optional, pre-aligned (same length as candles) raw series. Any
 *        key omitted degrades its feature(s) to a neutral 0 — see
 *        CONTEXT_GATED_FEATURES and ml/features.py's build_features()
 *        docstring for the full contract.
 * @returns {{rows: number[][], valid: boolean[], names: string[]}}
 */
export function buildFeatureMatrix(candles, context = null) {
  const n = candles.length;
  const close = candles.map((c) => c.close);
  const high = candles.map((c) => c.high);
  const low = candles.map((c) => c.low);
  const volume = candles.map((c) => Number(c.volume) || 0);
  const openTime = candles.map((c) => c.openTime);

  if (context) {
    for (const [key, arr] of Object.entries(context)) {
      if (arr != null && arr.length !== n) {
        throw new Error(
          `context.${key} has length ${arr.length}, candles have ${n} — ` +
            'context arrays must be pre-aligned bar-for-bar.'
        );
      }
    }
  }

  const logp = close.map((c) => Math.log(c));

  const laggedReturn = (k) => {
    const r = new Array(n).fill(NaN);
    for (let i = k; i < n; i++) r[i] = logp[i] - logp[i - k];
    return r;
  };
  const ret1 = laggedReturn(1);
  const ret3 = laggedReturn(3);
  const ret6 = laggedReturn(6);
  const ret12 = laggedReturn(12);

  // Python does np.nan_to_num(ret_1, nan=0.0) before the rolling std.
  const ret1Filled = ret1.map((v) => (Number.isFinite(v) ? v : 0));
  const vol12 = rollingStd(ret1Filled, 12);
  const vol48 = rollingStd(ret1Filled, 48);

  const sma20 = rollingMean(close, 20);
  const sma50 = rollingMean(close, 50);
  const bbSd = rollingStd(close, 20);
  const rsi14 = wilderRsiSeries(close, 14);
  const rsi9 = wilderRsiSeries(close, 9);
  const vwap = rollingVwap(high, low, close, volume, 48);
  const atr14 = atrSeries(high, low, close, 14);
  const adx14 = adxSeries(high, low, close, 14);

  const volMean = rollingMean(volume, 48);
  const volSd = rollingStd(volume, 48);

  const ema12 = ema(close, 12);
  const ema26 = ema(close, 26);
  const macd = new Array(n);
  for (let i = 0; i < n; i++) macd[i] = ema12[i] - ema26[i];
  const macdFilled = macd.map((v) => (Number.isFinite(v) ? v : 0));
  const macdSignal = ema(macdFilled, 9);

  // --- Cross-sectional (Phase 2.3) ---
  const basketRet = context?.basketRet ?? null;
  let relStrengthBasket;
  if (basketRet) {
    relStrengthBasket = new Array(n);
    for (let i = 0; i < n; i++) {
      const b = Number.isFinite(basketRet[i]) ? basketRet[i] : 0;
      relStrengthBasket[i] = ret1Filled[i] - b;
    }
  } else {
    relStrengthBasket = new Array(n).fill(0);
  }

  const btcRet = context?.btcRet ?? null;
  let corrToBtc;
  if (btcRet) {
    const btcRetFilled = btcRet.map((v) => (Number.isFinite(v) ? v : 0));
    const raw = rollingCorr(ret1Filled, btcRetFilled, 60);
    corrToBtc = raw.map((v) => (Number.isFinite(v) ? v : 0));
  } else {
    corrToBtc = new Array(n).fill(0);
  }

  // --- Crypto derivatives context (Phase 2.1) ---
  const fundingRate = context?.fundingRate ?? null;
  const fundingZ = fundingRate ? rollingZscore(fundingRate, 288) : new Array(n).fill(0);

  const openInterest = context?.openInterest ?? null;
  let oiRocZ;
  if (openInterest) {
    const oiRoc = new Array(n).fill(0);
    for (let i = 12; i < n; i++) {
      const ratio = openInterest[i] / openInterest[i - 12];
      oiRoc[i] = ratio > 0 && Number.isFinite(ratio) ? Math.log(ratio) : 0;
    }
    oiRocZ = rollingZscore(oiRoc, 48);
  } else {
    oiRocZ = new Array(n).fill(0);
  }

  const topTraderRatio = context?.topTraderRatio ?? null;
  const topTraderZ = topTraderRatio ? rollingZscore(topTraderRatio, 48) : new Array(n).fill(0);

  const rows = new Array(n);
  const valid = new Array(n);

  for (let i = 0; i < n; i++) {
    const bbMid = sma20[i];
    const bbUp = bbMid + 2 * bbSd[i];
    const bbLo = bbMid - 2 * bbSd[i];
    const rng = bbUp - bbLo;

    const pctB = rng > 0 ? (close[i] - bbLo) / rng : 0.5;
    const bandwidth = bbMid > 0 ? rng / bbMid : 0;

    const rsiSlope = i >= 3 ? (rsi14[i] - rsi14[i - 3]) / 100 : NaN;
    const volRatio = vol48[i] > 0 ? vol12[i] / vol48[i] : 1;

    let volumeZ = volSd[i] > 0 ? (volume[i] - volMean[i]) / volSd[i] : 0;
    if (!Number.isFinite(volumeZ)) volumeZ = 0;
    volumeZ = Math.max(-5, Math.min(5, volumeZ));

    const hours = ((openTime[i] / 3600000) % 24 + 24) % 24;

    const row = [
      rsi14[i] / 100 - 0.5,
      rsi9[i] / 100 - 0.5,
      rsiSlope,
      Math.log(close[i] / sma20[i]),
      Math.log(close[i] / sma50[i]),
      pctB - 0.5,
      bandwidth,
      Math.log(close[i] / vwap[i]),
      ret1[i],
      ret3[i],
      ret6[i],
      ret12[i],
      vol12[i],
      volRatio,
      atr14[i] / close[i],
      volumeZ,
      (macd[i] - macdSignal[i]) / close[i],
      Math.sin((2 * Math.PI * hours) / 24),
      Math.cos((2 * Math.PI * hours) / 24),
      adx14[i] / 100,
      relStrengthBasket[i],
      corrToBtc[i],
      fundingZ[i],
      oiRocZ[i],
      topTraderZ[i],
    ];

    rows[i] = row;
    valid[i] = row.every((v) => Number.isFinite(v));
  }

  return { rows, valid, names: FEATURE_NAMES };
}

/**
 * Features for the most recent bar only — the inference path.
 * @param {Array} candles
 * @param {object|null} [context] see buildFeatureMatrix()
 * @returns {number[]|null} null if the latest bar is still in warmup.
 */
export function latestFeatureVector(candles, context = null) {
  if (!Array.isArray(candles) || candles.length < 60) return null;
  const { rows, valid } = buildFeatureMatrix(candles, context);
  const i = rows.length - 1;
  return valid[i] ? rows[i] : null;
}

/* ------------------------------------------------------------------ */
/* Cross-sectional context (Phase 2.3) — JS twin of the corresponding  */
/* helpers in ml/data.py, used for live inference instead of training. */
/* ------------------------------------------------------------------ */

/**
 * Inner-join multiple { symbol: candles[] } series on openTime.
 * @param {Record<string, Array>} candlesBySymbol
 * @returns {{times: number[], bySymbol: Record<string, Array>}}
 */
export function alignCandlesByTime(candlesBySymbol) {
  const symbols = Object.keys(candlesBySymbol);
  let common = null;
  for (const sym of symbols) {
    const t = new Set(candlesBySymbol[sym].map((c) => c.openTime));
    common = common === null ? t : new Set([...common].filter((x) => t.has(x)));
  }
  const times = [...(common || [])].sort((a, b) => a - b);
  const timeSet = new Set(times);
  const bySymbol = {};
  for (const sym of symbols) {
    bySymbol[sym] = candlesBySymbol[sym]
      .filter((c) => timeSet.has(c.openTime))
      .sort((a, b) => a.openTime - b.openTime);
  }
  return { times, bySymbol };
}

/**
 * Build { basketRet, btcRet } for `selfSymbol` from a set of candle
 * series already aligned by alignCandlesByTime(). Mirrors
 * ml/data.py's build_cross_sectional_context().
 */
export function buildCrossSectionalContext(alignedBySymbol, selfSymbol) {
  const logRet1 = (candles) => {
    const n = candles.length;
    const out = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
      const r = Math.log(candles[i].close / candles[i - 1].close);
      out[i] = Number.isFinite(r) ? r : 0;
    }
    return out;
  };

  const symbols = Object.keys(alignedBySymbol);
  const rets = {};
  for (const sym of symbols) rets[sym] = logRet1(alignedBySymbol[sym]);

  const others = symbols.filter((s) => s !== selfSymbol);
  let basketRet = null;
  if (others.length) {
    const n = rets[others[0]].length;
    basketRet = new Array(n).fill(0);
    for (let i = 0; i < n; i++) {
      let sum = 0;
      for (const sym of others) sum += rets[sym][i];
      basketRet[i] = sum / others.length;
    }
  }

  const btcSymbol = symbols.find((s) => s.toUpperCase().startsWith('BTC/'));
  const btcRet = btcSymbol && btcSymbol !== selfSymbol ? rets[btcSymbol] : null;

  return { basketRet, btcRet };
}
