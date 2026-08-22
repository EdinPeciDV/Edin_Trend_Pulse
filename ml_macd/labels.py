"""
ml_macd/labels.py
===================================================================
PART 1 labels: forward return over N bars, three classes (up/down/
flat) where flat = |return| < band * ATR. Grid, not a fixed choice:
N in {4, 8, 12, 24, 48}, band in {0.25, 0.5, 1.0, 1.5}.

Gap-aware for real, not just prototyped: every label's validity comes
from gap_handling.py::label_validity_by_timestamp() — a label whose
N-bar forward window crosses a gap/hole measures a longer real-world
horizon than N implies, and is EXCLUDED (never filled or
approximated), per that module's design and its own unit tests
(already passing — this file does not re-test that logic, only uses
it).

"Which (N, band) combos are most predictable" (PART 1) needs a
trained model and is explicitly deferred to training — this file
reports class balance and drop rate per grid cell, which needs no
model, not predictability.
===================================================================
"""

import os
import sys

import numpy as np

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml"))
ML_MACD_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
# Same fix as macd_features.py — see its comment for why the
# remove-then-reinsert is necessary, not just an idempotent check.
if ML_MACD_DIR in sys.path:
    sys.path.remove(ML_MACD_DIR)
sys.path.insert(0, ML_MACD_DIR)

from features import atr  # noqa: E402
from gap_handling import label_validity_by_timestamp  # noqa: E402

N_GRID = [4, 8, 12, 24, 48]
BAND_GRID = [0.25, 0.5, 1.0, 1.5]

TIMEFRAME_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}


def build_label(candles, horizon_bars, band, tolerance_bars=1, atr14=None):
    """
    Returns (label, valid, fwd_return):
      label       int array, -1 (down) / 0 (flat) / +1 (up). Only
                  meaningful where valid=True — an invalid row's label
                  value is a placeholder, never a real classification.
      valid       bool array — False for gap-corrupted windows (see
                  module docstring) AND for the trailing `horizon_bars`
                  rows with no future data to label at all.
      fwd_return  float array, log(close[i+N]/close[i]) — the raw
                  continuous return underlying the classification,
                  kept for anyone who wants regression instead of the
                  three-class split.

    `atr14` may be passed in precomputed (macd_features.py already
    computes it) to avoid recomputing per grid cell; computed here if
    omitted.
    """
    close = candles["close"]
    n = len(close)
    logp = np.log(close)

    fwd_return = np.full(n, np.nan)
    if n > horizon_bars:
        fwd_return[: n - horizon_bars] = logp[horizon_bars:] - logp[:-horizon_bars]

    if atr14 is None:
        atr14 = atr(candles["high"], candles["low"], close, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_frac = atr14 / close
    threshold = band * atr_frac

    label = np.zeros(n, dtype=int)
    label[fwd_return > threshold] = 1
    label[fwd_return < -threshold] = -1

    timeframe_s = TIMEFRAME_SECONDS[candles["timeframe"]]
    gap_valid = label_validity_by_timestamp(
        candles["open_time"].astype(np.float64), horizon_bars, timeframe_s, tolerance_bars)
    valid = gap_valid & np.isfinite(fwd_return) & np.isfinite(threshold)

    return label, valid, fwd_return


def build_label_grid(candles, n_grid=N_GRID, band_grid=BAND_GRID, tolerance_bars=1):
    """
    One (label, valid, fwd_return) triple per (N, band) cell. ATR is
    computed once and shared across the whole grid — labels differ
    only in horizon and threshold, not in the underlying volatility
    measure.
    """
    atr14 = atr(candles["high"], candles["low"], candles["close"], 14)
    grid = {}
    for N in n_grid:
        for band in band_grid:
            grid[(N, band)] = build_label(candles, N, band, tolerance_bars, atr14=atr14)
    return grid


def report_label_grid(candles, grid=None, symbol_label=""):
    """
    Class balance + drop rate per (N, band) cell — the matrix PART 1
    asks for, minus predictability (that needs a trained model).
    Prints a table, returns the same data as a dict.
    """
    if grid is None:
        grid = build_label_grid(candles)

    out = {}
    header = f"{'N':>4} {'band':>6} {'valid_n':>9} {'dropped%':>9} {'up%':>7} {'flat%':>7} {'down%':>7}"
    print(f"  {symbol_label}")
    print("  " + header)
    for (N, band), (label, valid, fwd_return) in sorted(grid.items()):
        n_total = int((~np.isnan(fwd_return)).sum())
        n_valid = int(valid.sum())
        dropped_pct = 100.0 * (1 - n_valid / n_total) if n_total else float("nan")
        if n_valid > 0:
            lv = label[valid]
            up_pct = 100.0 * (lv == 1).mean()
            flat_pct = 100.0 * (lv == 0).mean()
            down_pct = 100.0 * (lv == -1).mean()
        else:
            up_pct = flat_pct = down_pct = float("nan")
        out[(N, band)] = {
            "n_valid": n_valid, "dropped_pct": dropped_pct,
            "up_pct": up_pct, "flat_pct": flat_pct, "down_pct": down_pct,
        }
        print(f"  {N:>4} {band:>6.2f} {n_valid:>9} {dropped_pct:>8.3f}% "
             f"{up_pct:>6.1f}% {flat_pct:>6.1f}% {down_pct:>6.1f}%")
    return out


# ------------------------------------------------------------------ #
# Required test — label arithmetic itself, not just gap exclusion    #
# (gap_handling.py already tests the exclusion logic; this tests the #
# up/down/flat threshold math on a deliberately constructed series)  #
# ------------------------------------------------------------------ #

def _test_label_thresholds():
    """
    Flat ATR, flat starting price, three deliberately sized forward
    moves at N=4: one clearly above the up threshold, one clearly
    below the down threshold, one inside the flat band. Assert each
    classifies correctly.
    """
    n = 60
    close = np.full(n, 100.0)
    atr14 = np.full(n, 2.0)  # ATR/price = 0.02 fixed, band=1.0 -> threshold = 0.02 log-return
    band = 1.0
    horizon = 4

    # bar 10: big up move by bar 14 (log return ~0.05, threshold ~0.02 -> UP)
    close[14] = close[10] * np.exp(0.05)
    # bar 20: big down move by bar 24 (log return ~-0.05 -> DOWN)
    close[24] = close[20] * np.exp(-0.05)
    # bar 30: tiny move by bar 34 (log return ~0.005, inside +-0.02 band -> FLAT)
    close[34] = close[30] * np.exp(0.005)

    candles = {
        "close": close, "high": close, "low": close,
        "open_time": np.arange(n) * 3600, "timeframe": "1h",
    }
    label, valid, fwd_return = build_label(candles, horizon, band, atr14=atr14)

    assert valid[10] and label[10] == 1, f"expected UP at bar 10, got label={label[10]} valid={valid[10]}"
    assert valid[20] and label[20] == -1, f"expected DOWN at bar 20, got label={label[20]} valid={valid[20]}"
    assert valid[30] and label[30] == 0, f"expected FLAT at bar 30, got label={label[30]} valid={valid[30]}"

    print("PASS: label threshold test — UP/DOWN/FLAT classify correctly at the ATR-scaled boundary.")


if __name__ == "__main__":
    print("=" * 70)
    print("LABELS — required threshold test")
    print("=" * 70)
    _test_label_thresholds()

    print("\n" + "=" * 70)
    print("LABELS — real BTC/USDT 1h grid (PART 1's N x band matrix)")
    print("=" * 70)
    try:
        import data as ml_data  # noqa: E402
        candles = ml_data.load_candles("BTC/USDT", "1h", "crypto")
        report_label_grid(candles, symbol_label="BTC/USDT 1h")
    except Exception as e:
        print(f"  (skipped real-data section — {e})")
