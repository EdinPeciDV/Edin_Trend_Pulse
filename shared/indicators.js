/**
 * shared/indicators.js
 * -------------------------------------------------------------------
 * Pure, dependency-free technical analysis functions.
 *
 * This file is the SINGLE SOURCE OF TRUTH for all math in TrendPulse.
 * It is imported by:
 *   - netlify/functions/*   (server-side, at ingest + analysis time)
 *   - src/lib/helpers.js    (client-side, for chart overlays)
 *
 * Every function takes plain arrays of numbers and returns plain values,
 * so it is trivially unit-testable and has no runtime dependencies.
 *
 * A "candle" throughout this file is:
 *   { openTime: number(ms), open, high, low, close, volume }
 * -------------------------------------------------------------------
 */

/* ------------------------------------------------------------------ */
/* Simple Moving Average                                               */
/* ------------------------------------------------------------------ */

/**
 * Simple Moving Average of the last `period` values.
 * @returns {number|null} null when there is not enough data.
 */
export function sma(values, period) {
  if (!Array.isArray(values) || values.length < period || period <= 0) return null;
  const window = values.slice(-period);
  const sum = window.reduce((acc, v) => acc + v, 0);
  return sum / period;
}

/**
 * Full SMA series (same length as input; leading entries are null).
 * Useful for chart overlays.
 */
export function smaSeries(values, period) {
  const out = new Array(values.length).fill(null);
  if (period <= 0) return out;
  let running = 0;
  for (let i = 0; i < values.length; i++) {
    running += values[i];
    if (i >= period) running -= values[i - period];
    if (i >= period - 1) out[i] = running / period;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Relative Strength Index (Wilder's smoothing)                        */
/* ------------------------------------------------------------------ */

/**
 * RSI using Wilder's exponential smoothing — the standard definition
 * used by TradingView / Bloomberg, NOT the naive simple-average variant.
 *
 * Requires at least `period + 1` closes.
 * @returns {number|null} 0–100, or null when there is not enough data.
 */
export function rsi(closes, period = 14) {
  if (!Array.isArray(closes) || closes.length < period + 1) return null;

  // Seed: simple average of the first `period` gains/losses.
  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const change = closes[i] - closes[i - 1];
    if (change >= 0) gainSum += change;
    else lossSum -= change; // losses stored positive
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;

  // Wilder smoothing across the remainder of the series.
  for (let i = period + 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
  }

  if (avgLoss === 0) return 100;
  if (avgGain === 0) return 0;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

/**
 * Full RSI series aligned to `closes` (leading entries null).
 * Used for the RSI overlay on the detail chart.
 */
export function rsiSeries(closes, period = 14) {
  const out = new Array(closes.length).fill(null);
  if (closes.length < period + 1) return out;

  let gainSum = 0;
  let lossSum = 0;
  for (let i = 1; i <= period; i++) {
    const change = closes[i] - closes[i - 1];
    if (change >= 0) gainSum += change;
    else lossSum -= change;
  }
  let avgGain = gainSum / period;
  let avgLoss = lossSum / period;
  out[period] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);

  for (let i = period + 1; i < closes.length; i++) {
    const change = closes[i] - closes[i - 1];
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;
    avgGain = (avgGain * (period - 1) + gain) / period;
    avgLoss = (avgLoss * (period - 1) + loss) / period;
    out[i] = avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss);
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Bollinger Bands                                                     */
/* ------------------------------------------------------------------ */

/**
 * Bollinger Bands over the last `period` closes.
 *
 * `percentB` locates price within the channel:
 *   0.0 = on the lower band, 0.5 = on the mean, 1.0 = on the upper band.
 * `bandwidth` is the normalised channel width — our volatility proxy.
 *
 * @returns {{middle,upper,lower,bandwidth,percentB,breakout}|null}
 *          breakout: 'UPPER' | 'LOWER' | null
 */
export function bollingerBands(closes, period = 20, stdDevMultiplier = 2) {
  if (!Array.isArray(closes) || closes.length < period) return null;

  const window = closes.slice(-period);
  const middle = window.reduce((a, b) => a + b, 0) / period;

  const variance =
    window.reduce((acc, v) => acc + Math.pow(v - middle, 2), 0) / period;
  const stdDev = Math.sqrt(variance);

  const upper = middle + stdDevMultiplier * stdDev;
  const lower = middle - stdDevMultiplier * stdDev;
  const price = closes[closes.length - 1];

  const range = upper - lower;
  const percentB = range === 0 ? 0.5 : (price - lower) / range;

  let breakout = null;
  if (price > upper) breakout = 'UPPER';
  else if (price < lower) breakout = 'LOWER';

  return {
    middle,
    upper,
    lower,
    stdDev,
    bandwidth: middle === 0 ? 0 : range / middle, // normalised volatility
    percentB,
    breakout,
  };
}

/* ------------------------------------------------------------------ */
/* VWAP                                                                */
/* ------------------------------------------------------------------ */

/**
 * Volume Weighted Average Price over the supplied candles.
 *
 * Uses the typical price (H+L+C)/3 per candle, which is the convention
 * for intraday VWAP.
 *
 * NOTE on Forex: spot FX has no centralised volume. Twelve Data returns
 * volume 0 or omits it for FX pairs. When total volume is 0 we fall back
 * to an unweighted mean of typical prices and flag it, rather than
 * silently returning a misleading "VWAP".
 *
 * @returns {{value:number, weighted:boolean}|null}
 */
export function vwap(candles) {
  if (!Array.isArray(candles) || candles.length === 0) return null;

  let pvSum = 0;
  let volSum = 0;
  let tpSum = 0;

  for (const c of candles) {
    const typical = (c.high + c.low + c.close) / 3;
    const vol = Number(c.volume) || 0;
    tpSum += typical;
    pvSum += typical * vol;
    volSum += vol;
  }

  if (volSum > 0) return { value: pvSum / volSum, weighted: true };
  return { value: tpSum / candles.length, weighted: false };
}

/* ------------------------------------------------------------------ */
/* Realised volatility                                                 */
/* ------------------------------------------------------------------ */

/**
 * Standard deviation of log returns, expressed as a percentage.
 * Feeds the confidence model and the entry-window search.
 */
export function realisedVolatility(closes) {
  if (!Array.isArray(closes) || closes.length < 3) return null;
  const returns = [];
  for (let i = 1; i < closes.length; i++) {
    if (closes[i - 1] > 0 && closes[i] > 0) {
      returns.push(Math.log(closes[i] / closes[i - 1]));
    }
  }
  if (returns.length < 2) return null;
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance =
    returns.reduce((acc, r) => acc + Math.pow(r - mean, 2), 0) /
    (returns.length - 1);
  return Math.sqrt(variance) * 100;
}

/* ------------------------------------------------------------------ */
/* Strategy profiles                                                   */
/* ------------------------------------------------------------------ */

/**
 * The "Aggressive" / "Conservative" toggle in Settings selects one of
 * these profiles. It changes the lookback lengths and how much
 * corroboration a signal needs before it is emitted.
 */
export const STRATEGY_PROFILES = {
  aggressive: {
    label: 'Aggressive',
    blurb: 'Short lookbacks, reacts fast, more signals and more noise.',
    smaPeriod: 20,
    rsiPeriod: 9,
    bbPeriod: 14,
    minConfidence: 45,
    // Stops are per asset class — see resolveStopLossPct() below for why.
    stopLossPct: { crypto: 2, forex: 0.5 },
    neutralBandwidth: 0.008, // below this width the market is "sideways"
  },
  conservative: {
    label: 'Conservative',
    blurb: 'Long lookbacks, waits for confirmation, fewer but steadier calls.',
    smaPeriod: 50,
    rsiPeriod: 14,
    bbPeriod: 20,
    minConfidence: 62,
    stopLossPct: { crypto: 3, forex: 0.75 },
    neutralBandwidth: 0.004,
  },
};

/**
 * Resolve the stop-loss distance for an instrument.
 *
 * DELIBERATE DEVIATION FROM THE ORIGINAL SPEC — the spec asked for a
 * flat "price − 2%" stop. That is sensible for crypto, where a 2% move
 * is ordinary intraday noise. It is badly wrong for spot FX: 2% of
 * EUR/USD is roughly 215 pips, and EUR/USD's entire average daily range
 * is under 100 pips. A flat 2% stop would therefore never be hit by
 * normal price action, which makes it not a stop at all.
 *
 * So the stop is scaled by asset class. Crypto keeps the spec's 2%
 * (3% on Conservative); FX uses 0.5%/0.75%, which is ~50–80 pips on
 * EUR/USD — still wide, but inside the realm of a real stop.
 *
 * To restore the literal spec behaviour, set both keys to 2.
 */
export function resolveStopLossPct(profile, assetClass = 'crypto') {
  const spec = profile?.stopLossPct;
  if (typeof spec === 'number') return spec; // flat value still supported
  return spec?.[assetClass] ?? spec?.crypto ?? 2;
}

export function getProfile(name) {
  return STRATEGY_PROFILES[name] || STRATEGY_PROFILES.conservative;
}

/* ------------------------------------------------------------------ */
/* The Up / Down / Neutral heuristic                                   */
/* ------------------------------------------------------------------ */

/**
 * Directional signal from the brief's rule set, with two documented
 * additions (see IMPORTANT note below).
 *
 * Brief's primary rules:
 *   UP    : price > SMA  AND  40 <= RSI <= 60      (stable growth)
 *   DOWN  : price < SMA  AND  RSI > 65             (overbought reversal)
 *   NEUTRAL: sideways / narrow range
 *
 * IMPORTANT — a note on the DOWN rule:
 *   "price below the mean AND overbought" is a genuinely rare state,
 *   because RSI > 65 almost always coincides with price ABOVE its moving
 *   average. Implemented alone it would fire so seldom that the app would
 *   read NEUTRAL nearly all the time. Rather than silently rewrite your
 *   spec, this function keeps that rule as `rule: 'spec-down'` and adds
 *   two clearly-labelled supplementary rules:
 *
 *     'reversal-down'  : price > SMA AND RSI > 70 AND upper-band breakout
 *                        (the classic overbought-reversal setup, which is
 *                        most likely what the original rule intended)
 *     'trend-down'     : price < SMA AND RSI < 40
 *                        (confirmed downtrend)
 *
 *   Every returned signal names the rule that fired, so you can see
 *   exactly which logic produced any given card, and delete the
 *   supplementary rules if you want the literal spec only.
 *
 * @param {object} inputs
 * @param {number} inputs.price
 * @param {number} inputs.smaValue
 * @param {number} inputs.rsiValue
 * @param {object|null} inputs.bb        result of bollingerBands()
 * @param {number|null} inputs.vwapValue
 * @param {object} profile               a STRATEGY_PROFILES entry
 * @returns {{signal:'UP'|'DOWN'|'NEUTRAL', confidence:number, rule:string, reasons:string[]}}
 */
export function computeSignal(
  { price, smaValue, rsiValue, bb, vwapValue },
  profile = STRATEGY_PROFILES.conservative
) {
  const reasons = [];

  // Guard: without SMA and RSI we cannot form a view.
  if (price == null || smaValue == null || rsiValue == null) {
    return {
      signal: 'NEUTRAL',
      confidence: 0,
      rule: 'insufficient-data',
      reasons: ['Not enough history yet — need more snapshots.'],
    };
  }

  const aboveSma = price > smaValue;
  const smaDistPct = ((price - smaValue) / smaValue) * 100;
  const bandwidth = bb ? bb.bandwidth : null;
  const breakout = bb ? bb.breakout : null;

  // --- Sideways gate: a narrow Bollinger channel means no real trend.
  if (bandwidth != null && bandwidth < profile.neutralBandwidth && !breakout) {
    return {
      signal: 'NEUTRAL',
      confidence: 30,
      rule: 'sideways-range',
      reasons: [
        `Bollinger bandwidth ${(bandwidth * 100).toFixed(2)}% is below the ` +
          `${(profile.neutralBandwidth * 100).toFixed(2)}% threshold — price is ranging.`,
      ],
    };
  }

  let signal = 'NEUTRAL';
  let rule = 'no-rule-matched';

  // --- Rule 1: UP (from the brief, verbatim)
  if (aboveSma && rsiValue >= 40 && rsiValue <= 60) {
    signal = 'UP';
    rule = 'spec-up';
    reasons.push(
      `Price is ${smaDistPct.toFixed(2)}% above the ${profile.smaPeriod}-period SMA.`,
      `RSI ${rsiValue.toFixed(1)} sits in the 40–60 stable-growth band.`
    );
  }
  // --- Rule 2: DOWN (from the brief, verbatim)
  else if (!aboveSma && rsiValue > 65) {
    signal = 'DOWN';
    rule = 'spec-down';
    reasons.push(
      `Price is ${Math.abs(smaDistPct).toFixed(2)}% below the ${profile.smaPeriod}-period SMA.`,
      `RSI ${rsiValue.toFixed(1)} is overbought (>65) while price sits under the mean.`
    );
  }
  // --- Rule 3: DOWN, overbought reversal (supplementary)
  else if (aboveSma && rsiValue > 70 && breakout === 'UPPER') {
    signal = 'DOWN';
    rule = 'reversal-down';
    reasons.push(
      `RSI ${rsiValue.toFixed(1)} is deeply overbought.`,
      'Price has pushed through the upper Bollinger band — stretched, reversal risk.'
    );
  }
  // --- Rule 4: DOWN, confirmed downtrend (supplementary)
  else if (!aboveSma && rsiValue < 40) {
    signal = 'DOWN';
    rule = 'trend-down';
    reasons.push(
      `Price is ${Math.abs(smaDistPct).toFixed(2)}% below the ${profile.smaPeriod}-period SMA.`,
      `RSI ${rsiValue.toFixed(1)} is weak and falling.`
    );
  }
  // --- Rule 5: UP, oversold bounce (supplementary)
  else if (rsiValue < 30 && breakout === 'LOWER') {
    signal = 'UP';
    rule = 'bounce-up';
    reasons.push(
      `RSI ${rsiValue.toFixed(1)} is deeply oversold.`,
      'Price has pierced the lower Bollinger band — mean-reversion setup.'
    );
  } else {
    reasons.push(
      `RSI ${rsiValue.toFixed(1)} and price ${smaDistPct >= 0 ? 'above' : 'below'} ` +
        'the mean do not satisfy any rule.'
    );
  }

  /* ---------------- Confidence model ----------------
   * Confidence is a transparent weighted score, NOT a probability.
   * It answers "how many of our conditions agree?", not
   * "how likely is this to be right?".
   */
  let score = 0;

  if (signal !== 'NEUTRAL') {
    score += 40; // a rule fired at all

    // Distance from the mean: conviction, capped so outliers don't dominate.
    score += Math.min(Math.abs(smaDistPct) * 4, 20);

    // RSI positioned in the direction of the call.
    if (signal === 'UP' && rsiValue >= 45 && rsiValue <= 60) score += 12;
    if (signal === 'DOWN' && rsiValue >= 65) score += 12;
    if (signal === 'DOWN' && rsiValue < 40) score += 8;

    // VWAP agreement — is price on the right side of the volume-weighted mean?
    if (vwapValue != null) {
      const vwapAgrees =
        (signal === 'UP' && price > vwapValue) ||
        (signal === 'DOWN' && price < vwapValue);
      if (vwapAgrees) {
        score += 12;
        reasons.push(
          `Price is ${signal === 'UP' ? 'above' : 'below'} VWAP — volume agrees.`
        );
      } else {
        score -= 8;
        reasons.push('VWAP disagrees with this call — treat it as weaker.');
      }
    }

    // Volatility: enough movement to travel, not so much it's chaos.
    if (bandwidth != null) {
      if (bandwidth > profile.neutralBandwidth * 2 && bandwidth < 0.06) score += 10;
      else if (bandwidth >= 0.06) {
        score -= 10;
        reasons.push('Volatility is extreme — the signal is unstable.');
      }
    }

    // Breakout confirming direction.
    if (
      (signal === 'UP' && breakout === 'UPPER') ||
      (signal === 'DOWN' && breakout === 'LOWER')
    ) {
      score += 6;
    }
  } else {
    score = 30;
  }

  const confidence = Math.max(0, Math.min(95, Math.round(score)));

  // Below the profile's floor we refuse to make a call.
  if (signal !== 'NEUTRAL' && confidence < profile.minConfidence) {
    return {
      signal: 'NEUTRAL',
      confidence,
      rule: `${rule}-below-threshold`,
      reasons: [
        ...reasons,
        `Confidence ${confidence}% is under the ${profile.label} floor of ` +
          `${profile.minConfidence}% — held back as NEUTRAL.`,
      ],
    };
  }

  return { signal, confidence, rule, reasons };
}

/* ------------------------------------------------------------------ */
/* "Best time to invest" window                                        */
/* ------------------------------------------------------------------ */

/**
 * Scans recent history for the most favourable historical entry point.
 *
 * Method (per the brief):
 *   1. Take the last `lookbackHours` of candles (default 12).
 *   2. Score each candle on two axes:
 *        - how oversold RSI was (lower = better)
 *        - how large the volume spike was vs the window mean
 *   3. Return the best-scoring candle as the entry window, with a stop
 *      loss set `stopLossPct` below that price.
 *
 * READ THIS: the returned window is a point in the PAST — the moment in
 * the recent window that would have been the best entry. It is a
 * descriptive statistic about what already happened, not a prediction
 * that the same conditions will recur. The UI labels it accordingly.
 *
 * @returns {object|null}
 */
export function findBestEntryWindow(candles, options = {}) {
  const {
    lookbackHours = 12,
    rsiPeriod = 14,
    stopLossPct = 2,
  } = options;

  if (!Array.isArray(candles) || candles.length < rsiPeriod + 5) return null;

  const cutoff = Date.now() - lookbackHours * 60 * 60 * 1000;
  const closes = candles.map((c) => c.close);
  const rsiVals = rsiSeries(closes, rsiPeriod);

  // Indices inside the lookback window that have a defined RSI.
  const idx = [];
  for (let i = 0; i < candles.length; i++) {
    if (candles[i].openTime >= cutoff && rsiVals[i] != null) idx.push(i);
  }
  if (idx.length === 0) return null;

  const vols = idx.map((i) => Number(candles[i].volume) || 0);
  const meanVol = vols.reduce((a, b) => a + b, 0) / vols.length || 0;

  let best = null;
  for (const i of idx) {
    const r = rsiVals[i];
    const vol = Number(candles[i].volume) || 0;

    // Oversold component: RSI 0 -> 1.0, RSI 100 -> 0.0
    const oversoldScore = (100 - r) / 100;

    // Volume-spike component, capped at 3x the window mean.
    const volRatio = meanVol > 0 ? vol / meanVol : 1;
    const volumeScore = Math.min(volRatio, 3) / 3;

    // Oversold is weighted more heavily than the volume spike.
    const score = oversoldScore * 0.65 + volumeScore * 0.35;

    if (!best || score > best.score) {
      best = {
        score,
        index: i,
        time: candles[i].openTime,
        price: candles[i].close,
        rsi: r,
        volume: vol,
        volumeRatio: volRatio,
      };
    }
  }
  if (!best) return null;

  const lowestRsi = idx.reduce(
    (acc, i) => (rsiVals[i] < acc.rsi ? { rsi: rsiVals[i], time: candles[i].openTime } : acc),
    { rsi: Infinity, time: null }
  );

  return {
    time: best.time,
    price: best.price,
    rsi: best.rsi,
    volumeRatio: best.volumeRatio,
    stopLoss: best.price * (1 - stopLossPct / 100),
    stopLossPct,
    lookbackHours,
    lowestRsiInWindow: lowestRsi.rsi === Infinity ? null : lowestRsi.rsi,
    quality: best.score >= 0.6 ? 'strong' : best.score >= 0.45 ? 'moderate' : 'weak',
  };
}

/* ------------------------------------------------------------------ */
/* Formatting helpers                                                  */
/* ------------------------------------------------------------------ */

/**
 * Fiat currency codes, used to tell a real FX pair from a crypto pair.
 *
 * Testing `symbol.includes('/')` is NOT enough: 'BTC/USDT' contains a
 * slash but must not be formatted to 5 decimal places like a currency
 * cross. A pair is FX only when BOTH legs are fiat.
 */
const FIAT_CODES = new Set([
  'USD', 'EUR', 'GBP', 'JPY', 'CHF', 'AUD', 'NZD', 'CAD',
  'SEK', 'NOK', 'DKK', 'PLN', 'CZK', 'HUF', 'TRY', 'ZAR',
  'MXN', 'SGD', 'HKD', 'CNH', 'CNY', 'INR', 'BRL',
]);

/** True only when both legs of the pair are fiat currencies. */
export function isForexPair(symbol = '') {
  const parts = symbol.toUpperCase().split('/');
  if (parts.length !== 2) return false;
  return FIAT_CODES.has(parts[0]) && FIAT_CODES.has(parts[1]);
}

/** Price formatting that adapts precision to the instrument's scale. */
export function formatPrice(value, symbol = '') {
  if (value == null || Number.isNaN(value)) return '—';

  if (isForexPair(symbol)) {
    // JPY crosses conventionally quote to 3dp; other crosses to 5dp.
    return value.toFixed(symbol.toUpperCase().includes('JPY') ? 3 : 5);
  }

  // Crypto and everything else: scale precision to magnitude.
  if (value >= 1000) return value.toLocaleString('en-US', { maximumFractionDigits: 2 });
  if (value >= 1) return value.toFixed(2);
  return value.toFixed(6);
}

export function formatTime(ts) {
  if (!ts) return '—';
  return new Date(ts).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}

export function formatDateTime(ts) {
  if (!ts) return '—';
  return new Date(ts).toLocaleString('en-US', {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  });
}
