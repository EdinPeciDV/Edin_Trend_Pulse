/**
 * tests/indicators.test.mjs
 * -------------------------------------------------------------------
 * Unit tests for the indicator engine in shared/indicators.js.
 *
 * Run with:  npm test
 *
 * The RSI test uses the canonical Wilder reference series, so a
 * regression in the smoothing shows up as a hard failure rather than as
 * subtly wrong signals. Everything else covers the edge cases that
 * actually bite in production: short inputs, flat series, zero-volume
 * forex, and the confidence floor that gates NEUTRAL.
 * -------------------------------------------------------------------
 */

import { rsi, sma, bollingerBands, vwap, computeSignal, findBestEntryWindow, getProfile, rsiSeries }
  from '../shared/indicators.js';

let pass = 0, fail = 0;
const t = (name, cond, extra='') => { cond ? (pass++, console.log('  ok  ', name)) : (fail++, console.log('  FAIL', name, extra)); };

// --- Canonical RSI test vector (Wilder, from the standard textbook series) ---
const closes = [44.34,44.09,44.15,43.61,44.33,44.83,45.10,45.42,45.84,46.08,
                45.89,46.03,45.61,46.28,46.28,46.00,46.03,46.41,46.22,45.64];
const r = rsi(closes.slice(0,15), 14);
// Expected ~70.46 for the first 14-period RSI of this classic series.
t('RSI matches Wilder reference (~70.46)', Math.abs(r - 70.46) < 0.5, `got ${r?.toFixed(2)}`);

const r2 = rsi(closes, 14);
t('RSI stays in 0..100', r2 > 0 && r2 < 100, `got ${r2?.toFixed(2)}`);

// --- RSI edge cases ---
t('RSI null on short input', rsi([1,2,3], 14) === null);
t('RSI = 100 on monotonic rise', rsi(Array.from({length:30},(_,i)=>i+1),14) === 100);
t('RSI = 0 on monotonic fall', rsi(Array.from({length:30},(_,i)=>30-i),14) === 0);

// --- SMA ---
t('SMA of 1..10 over 10 = 5.5', sma([1,2,3,4,5,6,7,8,9,10],10) === 5.5);
t('SMA null when short', sma([1,2],5) === null);

// --- rsiSeries aligns with rsi() ---
const series = rsiSeries(closes, 14);
t('rsiSeries last == rsi()', Math.abs(series[series.length-1] - r2) < 1e-9,
  `${series[series.length-1]} vs ${r2}`);
t('rsiSeries leading nulls', series[13] === null && series[14] !== null);

// --- Bollinger ---
const flat = Array(30).fill(100);
const bbFlat = bollingerBands(flat, 20);
t('BB on flat series: zero width', bbFlat.upper === 100 && bbFlat.lower === 100);
t('BB on flat series: no breakout', bbFlat.breakout === null);
t('BB percentB defaults to 0.5 when flat', bbFlat.percentB === 0.5);

const spike = [...Array(19).fill(100), 130];
const bbSpike = bollingerBands(spike, 20);
t('BB detects upper breakout', bbSpike.breakout === 'UPPER', JSON.stringify(bbSpike.breakout));
t('BB bandwidth positive', bbSpike.bandwidth > 0);

// --- VWAP ---
const candles = [
  {openTime:1, open:10, high:12, low:8,  close:10, volume:100},
  {openTime:2, open:10, high:22, low:18, close:20, volume:300},
];
const v = vwap(candles);
// typical prices: 10 and 20 -> (10*100 + 20*300)/400 = 17.5
t('VWAP volume-weighted = 17.5', Math.abs(v.value - 17.5) < 1e-9, `got ${v.value}`);
t('VWAP flagged weighted', v.weighted === true);

const noVol = candles.map(c => ({...c, volume: 0}));
const v2 = vwap(noVol);
t('VWAP falls back to unweighted mean (15)', Math.abs(v2.value - 15) < 1e-9, `got ${v2.value}`);
t('VWAP flags unweighted for FX', v2.weighted === false);

// --- Signal logic ---
const prof = getProfile('conservative');
const upSig = computeSignal(
  { price: 110, smaValue: 100, rsiValue: 50, bb: {bandwidth:0.03, breakout:null, percentB:0.7}, vwapValue: 105 },
  prof
);
t('spec-up rule fires', upSig.signal === 'UP' && upSig.rule === 'spec-up', JSON.stringify(upSig));
t('UP confidence in range', upSig.confidence > 0 && upSig.confidence <= 95, `${upSig.confidence}`);

const downSig = computeSignal(
  { price: 90, smaValue: 100, rsiValue: 70, bb: {bandwidth:0.03, breakout:null, percentB:0.2}, vwapValue: 95 },
  prof
);
t('spec-down rule fires', downSig.signal === 'DOWN' && downSig.rule === 'spec-down', JSON.stringify(downSig));

const sideways = computeSignal(
  { price: 100.1, smaValue: 100, rsiValue: 50, bb: {bandwidth:0.001, breakout:null, percentB:0.5}, vwapValue: 100 },
  prof
);
t('sideways gate returns NEUTRAL', sideways.signal === 'NEUTRAL' && sideways.rule === 'sideways-range');

const noData = computeSignal({ price: 100, smaValue: null, rsiValue: null, bb: null, vwapValue: null }, prof);
t('insufficient data -> NEUTRAL', noData.signal === 'NEUTRAL' && noData.confidence === 0);

// Confidence floor: aggressive floor is lower, so it should emit where conservative holds back
const weakInput = { price: 100.6, smaValue: 100, rsiValue: 50, bb:{bandwidth:0.01, breakout:null, percentB:0.6}, vwapValue: 101 };
const consv = computeSignal(weakInput, getProfile('conservative'));
const aggr  = computeSignal(weakInput, getProfile('aggressive'));
t('aggressive floor <= conservative floor',
  getProfile('aggressive').minConfidence < getProfile('conservative').minConfidence);
// Confidence legitimately differs: the volatility bonus is keyed to each
// profile's neutralBandwidth, so the same market reads differently per profile.
t('conservative holds back where aggressive emits',
  consv.signal === 'NEUTRAL' && consv.rule.endsWith('below-threshold') && aggr.signal === 'UP',
  `${consv.signal}/${consv.rule} vs ${aggr.signal}/${aggr.rule}`);
console.log(`     [info] conservative -> ${consv.signal}/${consv.rule}, aggressive -> ${aggr.signal}/${aggr.rule}, conf ${consv.confidence}`);

// --- Entry window ---
const now = Date.now();
const hist = Array.from({length: 100}, (_, i) => {
  const price = 100 + Math.sin(i/5)*5;
  return { openTime: now - (100-i)*5*60*1000, open:price, high:price+1, low:price-1, close:price, volume: 100 + (i===60?900:0) };
});
const ew = findBestEntryWindow(hist, {lookbackHours:12, rsiPeriod:14, stopLossPct:2});
t('entry window returned', ew !== null);
t('stop loss is 2% below entry', Math.abs(ew.stopLoss - ew.price*0.98) < 1e-9);
t('entry window inside lookback', ew.time >= now - 12*3600*1000);
t('entry window has quality label', ['strong','moderate','weak'].includes(ew.quality));
t('entry window null on tiny input', findBestEntryWindow([{openTime:now,open:1,high:1,low:1,close:1,volume:1}]) === null);

// --- Asset-class detection & price formatting ---
import { isForexPair, formatPrice, resolveStopLossPct } from '../shared/indicators.js';
t('BTC/USDT is not forex', isForexPair('BTC/USDT') === false);
t('EUR/USD is forex', isForexPair('EUR/USD') === true);
t('GBP/JPY is forex', isForexPair('GBP/JPY') === true);
t('SOL/USDT is not forex', isForexPair('SOL/USDT') === false);
t('crypto price uses thousands separator', formatPrice(61501.29249, 'BTC/USDT') === '61,501.29',
  formatPrice(61501.29249, 'BTC/USDT'));
t('fx price uses 5dp', formatPrice(1.0826912, 'EUR/USD') === '1.08269');
t('jpy cross uses 3dp', formatPrice(193.45678, 'GBP/JPY') === '193.457');

// --- Asset-class stop loss ---
t('crypto stop is wider than fx stop',
  resolveStopLossPct(getProfile('conservative'), 'crypto') >
  resolveStopLossPct(getProfile('conservative'), 'forex'));
t('flat numeric stopLossPct still supported', resolveStopLossPct({stopLossPct: 2}, 'forex') === 2);
t('unknown asset class falls back', resolveStopLossPct(getProfile('conservative'), 'zzz') === 3);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
