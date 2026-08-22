"""
ml_macd/indicator_parity_check.py
===================================================================
One-off comparison: Python (the implementations ml_macd/features.py
will use) vs shared/indicators.js (the JS the live dashboard actually
calls) for rsi, Bollinger bandwidth, and VWAP.

WHY THIS EXISTS. ml_macd/'s Python feature builder is authoritative
for the model (see ml_macd/README.md) — LightGBM is trained in Python
and served by a scheduled Python job (per PART 4's train/serve parity
requirement), so unlike ml/features.py <-> shared/mlFeatures.js there
is no live JS inference path to keep in lockstep. But the DASHBOARD
still shows rsi/Bollinger/VWAP computed by shared/indicators.js
(computeSignal() and the chart overlays), and the model is trained on
a separate Python computation of conceptually the same indicators. If
those two silently disagree, a user (or a future debugging session)
comparing "what the chart shows" against "what the model saw" will
see two different RSI values for the same candle and have no idea
why. This script measures that gap once and records it, rather than
letting it surprise someone later.

This is NOT a parity gate like ml/parity.py (there is no shared
inference path here to gate) — it is a documented, one-time
divergence report. Re-run it if either implementation changes.

Findings (see ml_macd/README.md "Indicator provenance" for the
narrated version):
  - RSI: both sides use Wilder smoothing, seeded identically (simple
    average of the first `period` gains/losses, then recursive
    (prev*(period-1)+new)/period). Same algorithm; only floating-point
    order-of-operations differs, so divergence should be ~1e-9-scale
    machine epsilon, not a real disagreement.
  - Bollinger bandwidth: both sides use (upper-lower)/middle with
    upper/lower = middle +/- 2*stddev(population, ddof=0) over a
    20-period SMA. Same formula; same expected near-zero divergence.
  - VWAP: both sides use the same primitive (volume-weighted mean of
    typical price (H+L+C)/3, falling back to an unweighted mean when
    total volume is 0 — the spot-FX case). shared/indicators.js's
    vwap() takes whatever candle slice the caller passes it, so this
    script calls it on the same trailing 48-bar window ml_macd uses,
    to compare like for like rather than two different window lengths.
===================================================================
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.abspath(ML_DIR))

from features import wilder_rsi, rolling_vwap, _rolling_mean, _rolling_std  # noqa: E402
import data as ml_data  # noqa: E402

VWAP_WINDOW = 48
BB_PERIOD = 20
RSI_PERIOD = 14


def python_indicators(candles):
    close = np.asarray(candles["close"], dtype=float)
    high = np.asarray(candles["high"], dtype=float)
    low = np.asarray(candles["low"], dtype=float)
    volume = np.asarray(candles["volume"], dtype=float)

    rsi = wilder_rsi(close, RSI_PERIOD)

    mid = _rolling_mean(close, BB_PERIOD)
    sd = _rolling_std(close, BB_PERIOD)
    rng = 4.0 * sd  # upper-lower = (mid+2sd) - (mid-2sd)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_bandwidth = np.where(mid > 0, rng / mid, 0.0)

    vwap = rolling_vwap(high, low, close, volume, VWAP_WINDOW)

    return rsi, bb_bandwidth, vwap


def js_indicators(candles):
    """
    Call shared/indicators.js exactly as the live app does: rsiSeries()
    for the full RSI series, but bollingerBands()/vwap() one trailing
    window at a time (indicators.js has no *Series variant for either —
    only the app-facing usage pattern of "call on the latest N-candle
    slice" exists), so this loop reproduces that usage rather than
    inventing a vectorized JS variant that wouldn't match what the
    dashboard actually runs.
    """
    n = len(candles["close"])
    payload = [
        {
            "openTime": float(candles["open_time"][i]),
            "open": float(candles["open"][i]),
            "high": float(candles["high"][i]),
            "low": float(candles["low"][i]),
            "close": float(candles["close"][i]),
            "volume": float(candles["volume"][i]),
        }
        for i in range(n)
    ]

    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "candles.json")
        with open(data_path, "w") as f:
            json.dump(payload, f)

        script = os.path.join(tmp, "run.mjs")
        with open(script, "w") as f:
            f.write(f"""
import {{ readFileSync }} from 'node:fs';
import {{ pathToFileURL }} from 'node:url';
import path from 'node:path';
const {{ rsiSeries, bollingerBands, vwap }} =
  await import(pathToFileURL(path.join({json.dumps(ROOT)}, 'shared', 'indicators.js')));
const candles = JSON.parse(readFileSync({json.dumps(data_path)}, 'utf8'));
const closes = candles.map(c => c.close);

const rsi = rsiSeries(closes, {RSI_PERIOD});

const bbBandwidth = new Array(candles.length).fill(null);
for (let i = {BB_PERIOD - 1}; i < candles.length; i++) {{
  const window = closes.slice(i - {BB_PERIOD - 1}, i + 1);
  const bb = bollingerBands(window, {BB_PERIOD}, 2);
  bbBandwidth[i] = bb ? bb.bandwidth : null;
}}

const vwapSeries = new Array(candles.length).fill(null);
for (let i = {VWAP_WINDOW - 1}; i < candles.length; i++) {{
  const window = candles.slice(i - {VWAP_WINDOW - 1}, i + 1);
  const v = vwap(window);
  vwapSeries[i] = v ? v.value : null;
}}

process.stdout.write(JSON.stringify({{ rsi, bbBandwidth, vwapSeries }}));
""")

        proc = subprocess.run(["node", script], capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"node failed:\n{proc.stderr[:2000]}")
        out = json.loads(proc.stdout)

    def to_arr(lst):
        return np.array([np.nan if v is None else v for v in lst], dtype=float)

    return to_arr(out["rsi"]), to_arr(out["bbBandwidth"]), to_arr(out["vwapSeries"])


def compare(name, py, js, warmup):
    both_valid = np.isfinite(py) & np.isfinite(js)
    both_valid[:warmup] = False
    n = int(both_valid.sum())
    if n == 0:
        print(f"  {name:<16} NO COMPARABLE ROWS (warmup={warmup})")
        return
    diff = np.abs(py[both_valid] - js[both_valid])
    print(f"  {name:<16} n={n:<5} max_abs_diff={diff.max():.3e}  mean_abs_diff={diff.mean():.3e}")


def main():
    print("=" * 70)
    print("INDICATOR DIVERGENCE: Python (ml_macd) vs shared/indicators.js")
    print("=" * 70)

    cases = [
        (ml_data.synthetic_random_walk(n=400, seed=7), "crypto-like (with volume)"),
    ]
    fx = ml_data.synthetic_random_walk(n=400, start_price=1.085, vol=0.0004, seed=8)
    fx["volume"] = np.zeros_like(fx["volume"])
    cases.append((fx, "forex-like (ZERO volume, VWAP fallback path)"))

    for candles, label in cases:
        print(f"\n--- {label} ({len(candles['close'])} bars) ---")
        rsi_py, bb_py, vwap_py = python_indicators(candles)
        rsi_js, bb_js, vwap_js = js_indicators(candles)
        compare("rsi_14", rsi_py, rsi_js, warmup=RSI_PERIOD + 5)
        compare("bb_bandwidth", bb_py, bb_js, warmup=BB_PERIOD + 5)
        compare("vwap", vwap_py, vwap_js, warmup=VWAP_WINDOW + 5)

    print("\n" + "=" * 70)
    print("Divergence at or below ~1e-9 is floating-point noise (same")
    print("algorithm, different language's arithmetic order) and confirms")
    print("agreement. Anything larger means the Python and JS definitions")
    print("have actually diverged and ml_macd/README.md's provenance")
    print("claim needs to be revisited before training on it.")
    print("=" * 70)


if __name__ == "__main__":
    main()
