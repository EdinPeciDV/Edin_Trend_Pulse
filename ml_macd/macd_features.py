"""
ml_macd/macd_features.py
===================================================================
Named macd_features.py, not features.py, DELIBERATELY: this file
imports from ml/features.py (the sibling module's primitives), which
requires both ml/ and ml_macd/ on sys.path at once — naming this file
features.py too would collide with that exact import and silently try
to import itself instead of the sibling module.

PHASE 2 (PART 1 features + PART 2 asset-class additions). Reuses
ml/features.py's primitives directly (wilder_rsi, atr, _ema,
rolling_vwap, _rolling_mean, _rolling_std) rather than reimplementing
them — those are already proven and parity-tested; ml_macd needs the
same math, applied to a MACD-centric feature set, not different math.

TWO RULES, same as ml/features.py and for the same reasons:
1. EVERY FEATURE IS STATIONARY — ratios, log returns, bounded
   oscillators, z-scores, one-hot flags. Never raw price.
2. EVERY FEATURE USES ONLY PAST DATA, THEN the whole matrix is
   shifted by one additional bar (PART 1: "All on closed candles,
   then shifted by 1 bar") — feature row i is computed from data
   available at bar i-1's close, one bar of margin beyond simply
   using closed candles, so a label anchored at bar i never shares
   even its own bar's close with the feature row predicting it.
   assert_no_lookahead() verifies this empirically, same
   truncate-and-recompute method as ml/features.py.

MULTI-TIMEFRAME (PART 1): build_features() REQUIRES htf_candles (the
4x-higher-timeframe series for the same symbol) — not optional, since
PART 1 treats multi-timeframe features as core, not an enhancement.

GAP-AWARE (PART 2, discovered mid-ingestion — see README.md section 8):
bars_missing_before / is_post_gap_bar come directly from
gap_handling.py, unmodified — this file does not recompute or
reimplement that logic.

NOT BUILT HERE, disclosed rather than faked:
- bars_until_session_close: no real FX/exchange holiday calendar
  exists in this project yet (PART 2 asks for one). Always NaN.
- Volume profile / TPO features (PART 3) — separate module, deferred.
===================================================================
"""

import os
import sys

import numpy as np

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml"))
ML_MACD_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
# ml_macd/'s own directory MUST win any name collision with ml/ (only
# "data" collides today — ml_macd/data.py vs ml/data.py) — a bare
# insert(0, ...) is not enough if some earlier-imported sibling module
# already inserted ML_DIR after this ran, so remove-then-reinsert makes
# this deterministic regardless of import order. Found the hard way:
# gap_handling.py used to unconditionally re-insert ML_DIR at position 0
# on every import, silently shadowing ml_macd/data.py with ml/data.py's
# unrelated same-named function for any file importing both.
if ML_MACD_DIR in sys.path:
    sys.path.remove(ML_MACD_DIR)
sys.path.insert(0, ML_MACD_DIR)

from features import wilder_rsi, atr, _ema, rolling_vwap, _rolling_mean, _rolling_std  # noqa: E402
from gap_handling import bars_missing_before, is_post_gap_bar  # noqa: E402

FEATURE_NAMES = [
    # --- MACD core ---
    "macd_norm_close", "signal_norm_close", "hist_norm_close",
    "macd_norm_atr", "signal_norm_atr", "hist_norm_atr",
    "macd_above_zero",
    # --- Crossover dynamics ---
    "bars_since_cross", "cross_direction", "price_move_since_cross_atr",
    # --- Gap dynamics (histogram) ---
    "hist_slope_1", "hist_slope_3", "hist_slope_5", "hist_slope_2nd_diff",
    "hist_ratio", "bars_since_hist_peak",
    # --- Multi-timeframe ---
    "htf_macd_above_zero", "htf_cross_direction", "htf_alignment",
    # --- Divergence ---
    "bearish_divergence", "bullish_divergence",
    # --- Order flow (crypto only — 0 for forex, never NaN) ---
    "taker_buy_ratio", "taker_buy_ratio_slope_5", "trades_per_unit_volume",
    # --- Context ---
    "rsi_14_c", "bb_width", "vwap_distance", "atr_pctile_200",
    "hour_sin", "hour_cos",
    "session_asia", "session_london", "session_ny",
    "dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun",
    # --- PART 2 asset-class / gap additions ---
    "gap_size_atr", "is_session_open_bar",
    "bars_missing_before", "is_post_gap_bar",
]
N_FEATURES = len(FEATURE_NAMES)

# Features that are 0 (never NaN) when the source doesn't provide the
# underlying data — crypto-only order flow, VWAP restricted to
# crypto/stock (PART 2's conflict-resolution binding decision: FX gets
# TPO/tick_volume profiles in PART 3, but VWAP itself stays gated).
CONTEXT_GATED_FEATURES = {
    "taker_buy_ratio", "taker_buy_ratio_slope_5", "trades_per_unit_volume",
    "vwap_distance",
}

DIVERGENCE_LOOKBACK = 20
ATR_PERCENTILE_WINDOW = 200


# ------------------------------------------------------------------ #
# Crossover / gap-dynamics helpers                                    #
# ------------------------------------------------------------------ #

def _cross_tracking(histogram):
    """
    A "cross" is histogram changing sign (MACD crossing its signal
    line). Returns:
      bars_since_cross    0 at the bar the cross happens, increasing
                          after; NaN before the first cross in the
                          series (nothing to measure from yet)
      cross_direction     +1 after an up-cross, -1 after a down-cross,
                          carried forward until the next cross; NaN
                          before the first cross
      cross_idx           bar index of the most recent cross, per bar
                          (int, -1 before the first cross) — internal,
                          used by hist_ratio/bars_since_hist_peak below
    """
    n = len(histogram)
    sign = np.sign(histogram)
    bars_since = np.full(n, np.nan)
    direction = np.full(n, np.nan)
    cross_idx = np.full(n, -1, dtype=int)

    last_cross = -1
    last_sign = sign[0] if n else 0.0
    last_direction = np.nan
    for i in range(n):
        if i > 0 and sign[i] != 0 and sign[i] != last_sign and last_sign != 0:
            last_cross = i
            last_direction = 1.0 if sign[i] > 0 else -1.0
        if sign[i] != 0:
            last_sign = sign[i]
        cross_idx[i] = last_cross
        if last_cross >= 0:
            bars_since[i] = i - last_cross
            direction[i] = last_direction
    return bars_since, direction, cross_idx


def _hist_ratio_and_peak(histogram, cross_idx):
    """
    hist_ratio = |histogram[i]| / max(|histogram|) since the last
    cross, bounded to [0, 1] by construction (current value can never
    exceed the running max it is itself part of).
    bars_since_hist_peak = bars since |histogram| last hit that
    running max within the current cross regime.
    """
    n = len(histogram)
    abs_hist = np.abs(histogram)
    ratio = np.full(n, np.nan)
    since_peak = np.full(n, np.nan)

    regime_start = None
    running_max = -np.inf
    peak_idx = None
    for i in range(n):
        if cross_idx[i] < 0:
            continue
        if cross_idx[i] != regime_start:
            regime_start = cross_idx[i]
            running_max = -np.inf
            peak_idx = regime_start
        if abs_hist[i] >= running_max:
            running_max = abs_hist[i]
            peak_idx = i
        ratio[i] = abs_hist[i] / running_max if running_max > 0 else 0.0
        since_peak[i] = i - peak_idx
    return ratio, since_peak


def _divergence(close, macd, window=DIVERGENCE_LOOKBACK):
    """
    Causal, explicitly-defined divergence (not a general swing
    detector): comparing bar i to the trailing `window` bars only.

      bearish_divergence = price makes a new `window`-bar high AND
                           macd does NOT make a new `window`-bar high
                           (momentum doesn't confirm the price high)
      bullish_divergence = price makes a new `window`-bar low AND
                           macd does NOT make a new `window`-bar low

    Both boolean (0.0/1.0), NaN during warmup (fewer than `window`
    prior bars available).
    """
    n = len(close)
    bearish = np.full(n, np.nan)
    bullish = np.full(n, np.nan)
    for i in range(window, n):
        price_win = close[i - window:i]
        macd_win = macd[i - window:i]
        price_new_high = close[i] > price_win.max()
        price_new_low = close[i] < price_win.min()
        macd_new_high = macd[i] > macd_win.max()
        macd_new_low = macd[i] < macd_win.min()
        bearish[i] = float(price_new_high and not macd_new_high)
        bullish[i] = float(price_new_low and not macd_new_low)
    return bearish, bullish


def _rolling_percentile_rank(x, window):
    """out[i] = fraction of x[i-window+1..i] that is <= x[i], in [0,1]."""
    n = len(x)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        win = x[i - window + 1:i + 1]
        valid = win[np.isfinite(win)]
        if len(valid) == 0 or not np.isfinite(x[i]):
            continue
        out[i] = float((valid <= x[i]).mean())
    return out


def _macd_core(close, high, low):
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(np.nan_to_num(macd, nan=0.0), 9)
    histogram = macd - signal
    atr14 = atr(high, low, close, 14)
    return macd, signal, histogram, atr14


def _htf_regime_aligned_to_entry(entry_open_time, htf_open_time, htf_macd_above_zero,
                                 htf_cross_direction, htf_timeframe_s):
    """
    As-of join: for each entry-TF bar, the higher-TF regime value is
    whichever HTF bar was the LAST ONE FULLY CLOSED as of the entry
    bar's own open_time — i.e. htf_open_time + htf_timeframe_s <=
    entry_open_time. Never the HTF bar that is still forming or that
    closes strictly after the entry bar opens; that would be reading
    the future from the entry bar's point of view.
    """
    n = len(entry_open_time)
    htf_close_time = htf_open_time + htf_timeframe_s
    idx = np.searchsorted(htf_close_time, entry_open_time, side="right") - 1

    out_regime = np.full(n, np.nan)
    out_direction = np.full(n, np.nan)
    valid = idx >= 0
    out_regime[valid] = htf_macd_above_zero[idx[valid]]
    out_direction[valid] = htf_cross_direction[idx[valid]]
    return out_regime, out_direction


TIMEFRAME_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}


# ------------------------------------------------------------------ #
# Main feature matrix                                                 #
# ------------------------------------------------------------------ #

def build_features(candles, htf_candles):
    """
    `candles` / `htf_candles`: dicts from data.py::load_candles() —
    entry timeframe and its 4x-higher timeframe, same symbol.

    Returns (X, valid_mask). X is (n_bars, N_FEATURES), already
    shifted by 1 bar (row i uses only data available at bar i-1's
    close). valid_mask is False for warmup / undefined rows — callers
    must drop those, never impute them.
    """
    open_time = candles["open_time"].astype(np.float64)
    open_ = candles["open"]
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]
    volume = candles["volume"]
    n_trades = candles["number_of_trades"]
    taker_base = candles["taker_buy_base_volume"]
    asset_class = candles["asset_class"]
    timeframe = candles["timeframe"]
    n = len(close)

    macd, signal, histogram, atr14 = _macd_core(close, high, low)
    with np.errstate(divide="ignore", invalid="ignore"):
        macd_norm_close = macd / close
        signal_norm_close = signal / close
        hist_norm_close = histogram / close
        macd_norm_atr = macd / atr14
        signal_norm_atr = signal / atr14
        hist_norm_atr = histogram / atr14
    macd_above_zero = (macd > 0).astype(float)

    bars_since_cross, cross_direction, cross_idx = _cross_tracking(histogram)
    with np.errstate(divide="ignore", invalid="ignore"):
        price_at_cross = np.where(cross_idx >= 0, close[np.clip(cross_idx, 0, n - 1)], np.nan)
        price_move_since_cross_atr = (close - price_at_cross) / atr14

    hist_slope_1 = np.full(n, np.nan)
    hist_slope_3 = np.full(n, np.nan)
    hist_slope_5 = np.full(n, np.nan)
    # STATIONARITY: slope is computed from hist_norm_atr, NOT the raw
    # histogram — raw histogram is in price units, so its slope would
    # scale with price level (huge on BTC at $90k, tiny on a $1 altcoin)
    # and violate this file's own no-raw-price rule. Found via a range
    # sanity check during testing (slopes up to ~1000 on raw histogram).
    hist_slope_1[1:] = hist_norm_atr[1:] - hist_norm_atr[:-1]
    hist_slope_3[3:] = hist_norm_atr[3:] - hist_norm_atr[:-3]
    hist_slope_5[5:] = hist_norm_atr[5:] - hist_norm_atr[:-5]
    hist_slope_2nd_diff = np.full(n, np.nan)
    hist_slope_2nd_diff[2:] = hist_slope_1[2:] - hist_slope_1[1:-1]

    hist_ratio, bars_since_hist_peak = _hist_ratio_and_peak(histogram, cross_idx)

    # --- Multi-timeframe ---
    htf_macd, htf_signal, htf_histogram, _ = _macd_core(
        htf_candles["close"], htf_candles["high"], htf_candles["low"])
    htf_macd_above_zero_series = (htf_macd > 0).astype(float)
    _, htf_cross_direction_series, _ = _cross_tracking(htf_histogram)
    htf_timeframe_s = TIMEFRAME_SECONDS[htf_candles["timeframe"]]
    htf_macd_above_zero, htf_cross_direction = _htf_regime_aligned_to_entry(
        open_time, htf_candles["open_time"].astype(np.float64),
        htf_macd_above_zero_series, htf_cross_direction_series, htf_timeframe_s)
    htf_alignment = (macd_above_zero == htf_macd_above_zero).astype(float)
    htf_alignment[np.isnan(htf_macd_above_zero)] = np.nan

    # --- Divergence ---
    bearish_divergence, bullish_divergence = _divergence(close, macd)

    # --- Order flow (crypto only) ---
    is_crypto = asset_class == "crypto"
    if is_crypto:
        with np.errstate(divide="ignore", invalid="ignore"):
            taker_buy_ratio = np.where(volume > 0, taker_base / volume, 0.0)
        taker_buy_ratio = np.nan_to_num(taker_buy_ratio, nan=0.0)
        taker_buy_ratio_slope_5 = np.full(n, 0.0)
        taker_buy_ratio_slope_5[5:] = taker_buy_ratio[5:] - taker_buy_ratio[:-5]
        with np.errstate(divide="ignore", invalid="ignore"):
            trades_per_unit_volume = np.where(volume > 0, n_trades / volume, 0.0)
        trades_per_unit_volume = np.nan_to_num(trades_per_unit_volume, nan=0.0)
    else:
        taker_buy_ratio = np.zeros(n)
        taker_buy_ratio_slope_5 = np.zeros(n)
        trades_per_unit_volume = np.zeros(n)

    # --- Context ---
    rsi14 = wilder_rsi(close, 14)
    rsi_14_c = rsi14 / 100.0 - 0.5

    bb_mid = _rolling_mean(close, 20)
    bb_sd = _rolling_std(close, 20)
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_width = np.where(bb_mid > 0, 4.0 * bb_sd / bb_mid, 0.0)

    if asset_class == "forex":
        # PART 2 conflict resolution (binding): VWAP stays gated to
        # crypto/stock. Never fabricate a fallback for FX here.
        vwap_distance = np.zeros(n)
    else:
        vwap = rolling_vwap(high, low, close, volume, 48)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap_distance = np.log(close / vwap)

    atr_pctile_200 = _rolling_percentile_rank(atr14, ATR_PERCENTILE_WINDOW)

    hours = (open_time / 3600.0) % 24.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)
    session_asia = ((hours >= 0) & (hours < 8)).astype(float)
    session_london = ((hours >= 8) & (hours < 16)).astype(float)
    session_ny = ((hours >= 13) & (hours < 21)).astype(float)

    dow = ((open_time / 86400.0).astype(np.int64) + 4) % 7  # unix epoch was a Thursday=3; +4 -> Mon=0
    dow_cols = {name: (dow == i).astype(float) for i, name in enumerate(
        ["dow_mon", "dow_tue", "dow_wed", "dow_thu", "dow_fri", "dow_sat", "dow_sun"])}

    # --- PART 2 additions ---
    prev_close = np.concatenate(([np.nan], close[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_size_atr = (open_ - prev_close) / atr14

    open_time_int = candles["open_time"]
    bmb = bars_missing_before(open_time_int, TIMEFRAME_SECONDS[timeframe]).astype(float)
    ipg = is_post_gap_bar(bmb).astype(float)
    is_session_open_bar = ipg.copy()

    cols = {
        "macd_norm_close": macd_norm_close, "signal_norm_close": signal_norm_close,
        "hist_norm_close": hist_norm_close,
        "macd_norm_atr": macd_norm_atr, "signal_norm_atr": signal_norm_atr,
        "hist_norm_atr": hist_norm_atr,
        "macd_above_zero": macd_above_zero,
        "bars_since_cross": bars_since_cross, "cross_direction": cross_direction,
        "price_move_since_cross_atr": price_move_since_cross_atr,
        "hist_slope_1": hist_slope_1, "hist_slope_3": hist_slope_3, "hist_slope_5": hist_slope_5,
        "hist_slope_2nd_diff": hist_slope_2nd_diff,
        "hist_ratio": hist_ratio, "bars_since_hist_peak": bars_since_hist_peak,
        "htf_macd_above_zero": htf_macd_above_zero, "htf_cross_direction": htf_cross_direction,
        "htf_alignment": htf_alignment,
        "bearish_divergence": bearish_divergence, "bullish_divergence": bullish_divergence,
        "taker_buy_ratio": taker_buy_ratio, "taker_buy_ratio_slope_5": taker_buy_ratio_slope_5,
        "trades_per_unit_volume": trades_per_unit_volume,
        "rsi_14_c": rsi_14_c, "bb_width": bb_width, "vwap_distance": vwap_distance,
        "atr_pctile_200": atr_pctile_200,
        "hour_sin": hour_sin, "hour_cos": hour_cos,
        "session_asia": session_asia, "session_london": session_london, "session_ny": session_ny,
        **dow_cols,
        "gap_size_atr": gap_size_atr, "is_session_open_bar": is_session_open_bar,
        "bars_missing_before": bmb, "is_post_gap_bar": ipg,
    }

    X_raw = np.column_stack([cols[name] for name in FEATURE_NAMES])

    # PART 1: "shifted by 1 bar" — row i becomes row i-1's values. Row 0
    # has nothing before it and is invalid by construction.
    X = np.full_like(X_raw, np.nan)
    X[1:] = X_raw[:-1]

    valid = np.isfinite(X).all(axis=1)

    # Storage requirement: float32, not float64, for every feature
    # column — indicator values don't need double precision, and it
    # halves memory outright. Cast AFTER computing `valid` from the
    # float64 matrix, not before: NaN-detection and the cross-tracking/
    # percentile logic upstream rely on float64 comparison precision,
    # and downcasting first could shift a borderline value across a
    # NaN/finite boundary that shouldn't move.
    X = X.astype(np.float32)
    return X, valid


# ------------------------------------------------------------------ #
# Look-ahead guard — same method as ml/features.py                    #
# ------------------------------------------------------------------ #

def assert_no_lookahead(candles, htf_candles, n_checks=6, tolerance=1e-6):
    """
    `tolerance=1e-6`, not ml/features.py's 1e-9 — build_features()
    returns float32 (storage requirement, see its final cast), which
    has ~7 significant digits; 1e-9 would risk a false failure from
    ordinary float32 rounding noise, not a real look-ahead leak.

    Truncate both the entry and higher-timeframe series at bar i,
    recompute, and compare bar i's feature row to the full-series
    computation. Any difference means some feature read data beyond
    bar i — including, critically, the multi-timeframe alignment,
    which is the highest-risk new surface for this kind of bug (an
    as-of join that accidentally reaches past its cutoff).

    Recursive primitives (EMA/ATR/RSI) legitimately differ near the
    start of a truncated series (different warmup) — checks only run
    well past that, matching ml/features.py's own convention.
    """
    X_full, _ = build_features(candles, htf_candles)
    n = len(candles["close"])
    warmup = 260  # ml_macd/README.md section 1's derived warmup_bars
    if n < warmup + 20:
        raise ValueError(f"need >{warmup + 20} bars to check look-ahead, got {n}")

    htf_timeframe_s = TIMEFRAME_SECONDS[htf_candles["timeframe"]]
    idxs = np.linspace(warmup, n - 2, n_checks).astype(int)
    failures = []

    for i in idxs:
        truncated = {k: (v[: i + 1] if isinstance(v, np.ndarray) else v) for k, v in candles.items()}
        cutoff_time = candles["open_time"][i]
        htf_keep = htf_candles["open_time"] + htf_timeframe_s <= cutoff_time + 1
        # Truncate the HTF series to only bars that could legitimately
        # be visible as of entry bar i — same cutoff the join itself uses.
        htf_truncated = {
            k: (v[htf_keep] if isinstance(v, np.ndarray) else v) for k, v in htf_candles.items()
        }
        if len(htf_truncated["close"]) < 30:
            continue  # not enough HTF history yet at this cutoff to check meaningfully

        X_trunc, _ = build_features(truncated, htf_truncated)
        diff = np.abs(X_full[i] - X_trunc[i])
        bad = np.where(np.nan_to_num(diff, nan=0.0) > tolerance)[0]
        for j in bad:
            failures.append((int(i), FEATURE_NAMES[j], float(diff[j])))

    if failures:
        lines = [f"    bar {i}: {name} differs by {d:.3e}" for i, name, d in failures[:12]]
        raise AssertionError(
            "LOOK-AHEAD DETECTED — these features read future data:\n"
            + "\n".join(lines)
        )
    return True
