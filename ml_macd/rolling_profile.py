"""
ml_macd/rolling_profile.py
===================================================================
PART 3 anchoring window: ROLLING profile. The higher-value window per
the decision recorded in README.md section 21 — always current
rather than reset-dependent, and behaves identically on 24/7 crypto
and gappy FX (unlike session profile, which is tied to a UTC-day
reset that means something different on each).

CAUSALITY, self-contained (same philosophy as session_profile.py):
for a profile attached to bar i, the underlying window is bars
STRICTLY BEFORE bar i — `[i-window, i-1]`, never including bar i
itself. This directly satisfies "must only summarize bars strictly
before the bar it's attached to" without depending on an external
shift-by-1 step elsewhere.

REFRESH CADENCE IS BOUNDED, a disclosed engineering choice, same
tradeoff already made for session-profile's naked-POC lookback
(README.md section 19): rebuilding a full O(window)-bar profile at
EVERY bar is prohibitively slow over a 78k-bar series. Profiles are
rebuilt every `refresh_every` bars (default `window // 4`) and
forward-filled between refreshes — staleness is bounded to at most
`refresh_every` bars, logged per profile so it's never silent.

Uses the SAME closed-bars-only, weight-resolution, and skip-vs-raise
NaN guard as session_profile.py (`volume_profile.py`'s shared
helpers) — not reimplemented.
===================================================================
"""

import os
import sys

import numpy as np

ML_MACD_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_MACD_DIR in sys.path:
    sys.path.remove(ML_MACD_DIR)
sys.path.insert(0, ML_MACD_DIR)

from volume_profile import (  # noqa: E402
    build_profile, resolve_weight_mode_array, check_weight_window, StructuralWeightGap,
)


def build_rolling_profiles(candles, weight_mode, atr14, window=100, refresh_every=None,
                           atr_bin_multiple=0.10, value_area_pct=0.70):
    """
    Returns (profiles, per_bar_profile_idx):
      profiles              list of build_profile() results, each also
                            carrying `built_at_idx` (the bar this
                            refresh happened at), `window_start_idx`,
                            `window_end_idx` (both inclusive — the
                            exact trailing window used, strictly
                            before `built_at_idx`).
      per_bar_profile_idx   int array, length n. per_bar_profile_idx[i]
                            is the index into `profiles` that bar i
                            should reference (the most recent profile
                            built at or before i, per the causal rule
                            above) — -1 where no profile exists yet
                            (fewer than `window` prior bars available).

    A profile is SKIPPED (not built, gap left in the forward-fill) at
    a refresh point whose window contains a structural weight gap
    (FX) — the previous profile keeps being forward-filled, exactly
    as if that refresh point had been staleness-bounded rather than
    refreshed. A genuine ingestion gap (crypto) still raises loudly,
    same as session_profile.py.
    """
    n = len(candles["close"])
    if refresh_every is None:
        refresh_every = max(1, window // 4)

    high, low = candles["high"], candles["low"]
    weight_full = resolve_weight_mode_array(candles, weight_mode)
    volume_is_proxy = candles.get("volume_is_proxy")

    profiles = []
    per_bar_profile_idx = np.full(n, -1, dtype=int)

    current_profile_idx = -1
    next_refresh_i = window  # earliest bar with `window` strictly-prior bars available

    for i in range(window, n):
        if i >= next_refresh_i:
            lo, hi = i - window, i - 1  # inclusive, strictly before i
            window_weight = weight_full[lo:hi + 1]
            atr_ref = atr14[hi]
            built = False
            if np.isfinite(atr_ref) and atr_ref > 0:
                try:
                    check_weight_window(window_weight, weight_mode, volume_is_proxy,
                                        context=f"rolling window ending bar {hi}")
                    profile = build_profile(
                        high[lo:hi + 1], low[lo:hi + 1], window_weight, atr_ref,
                        mode=weight_mode, profile_source=f"rolling_{weight_mode}",
                        atr_bin_multiple=atr_bin_multiple, value_area_pct=value_area_pct,
                        in_progress=False,  # a rolling window is never "in progress" the way a
                                            # session is — it's always exactly `window` CLOSED
                                            # bars, strictly before i, by construction.
                    )
                    profile["built_at_idx"] = i
                    profile["window_start_idx"] = lo
                    profile["window_end_idx"] = hi
                    profiles.append(profile)
                    current_profile_idx = len(profiles) - 1
                    built = True
                except StructuralWeightGap:
                    pass  # FX-like: skip this refresh, keep forward-filling the prior profile
            next_refresh_i = i + refresh_every
            if not built and atr_ref is not None and not (np.isfinite(atr_ref) and atr_ref > 0):
                next_refresh_i = i + 1  # ATR not warmed up — retry every bar, not every refresh_every

        per_bar_profile_idx[i] = current_profile_idx

    return profiles, per_bar_profile_idx


def extract_rolling_profile_features(candles, profiles, per_bar_profile_idx, atr14, mode):
    """
    Per-bar features, namespaced `rolling_{mode}_...`, referencing
    `per_bar_profile_idx[i]` — a profile built from bars strictly
    before i, by construction (see build_rolling_profiles()). No
    naked-POC features here: PART 3's naked-POC bookkeeping is
    explicitly a SESSION concept ("track prior-SESSION POCs") — this
    window doesn't have sessions to be "prior" to.
    """
    close = candles["close"]
    n = len(close)
    prefix = f"rolling_{mode}_"
    out = {
        prefix + "dist_to_poc_atr": np.full(n, np.nan),
        prefix + "dist_to_vah_atr": np.full(n, np.nan),
        prefix + "dist_to_val_atr": np.full(n, np.nan),
        prefix + "inside_value_area": np.full(n, np.nan),
        prefix + "dist_to_hvn_atr": np.full(n, np.nan),
        prefix + "dist_to_lvn_atr": np.full(n, np.nan),
        prefix + "dist_to_volume_edge_atr": np.full(n, np.nan),
        prefix + "poc_migration_atr": np.full(n, np.nan),
        prefix + "value_area_width_atr": np.full(n, np.nan),
    }

    for i in range(n):
        pidx = per_bar_profile_idx[i]
        if pidx < 0:
            continue
        a = atr14[i]
        if not np.isfinite(a) or a <= 0:
            continue
        p = profiles[pidx]

        out[prefix + "dist_to_poc_atr"][i] = (close[i] - p["poc"]) / a
        out[prefix + "dist_to_vah_atr"][i] = (close[i] - p["vah"]) / a
        out[prefix + "dist_to_val_atr"][i] = (close[i] - p["val"]) / a
        out[prefix + "inside_value_area"][i] = float(p["val"] <= close[i] <= p["vah"])
        out[prefix + "value_area_width_atr"][i] = (p["vah"] - p["val"]) / a

        if p["hvn"]:
            nearest = min(p["hvn"], key=lambda x: abs(x - close[i]))
            out[prefix + "dist_to_hvn_atr"][i] = (close[i] - nearest) / a
        if p["lvn"]:
            nearest = min(p["lvn"], key=lambda x: abs(x - close[i]))
            out[prefix + "dist_to_lvn_atr"][i] = (close[i] - nearest) / a
        if p["volume_edge"]:
            nearest = min(p["volume_edge"], key=lambda x: abs(x - close[i]))
            out[prefix + "dist_to_volume_edge_atr"][i] = (close[i] - nearest) / a

        if pidx > 0:
            prev = profiles[pidx - 1]
            out[prefix + "poc_migration_atr"][i] = (p["poc"] - prev["poc"]) / a

    return out


# ------------------------------------------------------------------ #
# Required leakage test — "must only summarize bars strictly before  #
# the bar it's attached to"                                          #
# ------------------------------------------------------------------ #

def assert_rolling_profile_strictly_prior(candles, weight_mode, atr14, window=100):
    """
    Two checks, both must hold:
      1. STRUCTURAL: every profile's own window_end_idx < built_at_idx
         <= every bar it's assigned to (per_bar_profile_idx). A
         profile built from bars ending at window_end_idx cannot be
         referenced by any bar <= window_end_idx.
      2. EMPIRICAL, same truncate-and-recompute method as every other
         look-ahead test in this project: rebuild the ENTIRE rolling
         pipeline on a series truncated at bar i, and confirm bar i's
         feature value matches the full-history computation exactly —
         proving the window never reached past bar i regardless of
         what data exists beyond it.
    """
    profiles, per_bar_profile_idx = build_rolling_profiles(candles, weight_mode, atr14, window=window)

    # 1. Structural check.
    for i, pidx in enumerate(per_bar_profile_idx):
        if pidx < 0:
            continue
        p = profiles[pidx]
        assert p["window_end_idx"] < i, (
            f"LEAK: bar {i} references a profile whose window ends at "
            f"{p['window_end_idx']}, not strictly before bar {i}"
        )
        assert p["built_at_idx"] <= i, (
            f"LEAK: bar {i} references a profile built at {p['built_at_idx']}, "
            f"which is AFTER bar {i}"
        )

    # 2. Empirical truncate-and-recompute.
    feats_full = extract_rolling_profile_features(candles, profiles, per_bar_profile_idx, atr14, weight_mode)
    n = len(candles["close"])
    check_idxs = np.linspace(window + 50, n - 2, 4).astype(int)
    for i in check_idxs:
        truncated = {k: (v[: i + 1] if isinstance(v, np.ndarray) else v) for k, v in candles.items()}
        atr14_trunc = atr14[: i + 1]
        p_trunc, idx_trunc = build_rolling_profiles(truncated, weight_mode, atr14_trunc, window=window)
        f_trunc = extract_rolling_profile_features(truncated, p_trunc, idx_trunc, atr14_trunc, weight_mode)
        for name in feats_full:
            full_val = feats_full[name][i]
            trunc_val = f_trunc[name][-1]
            full_finite, trunc_finite = np.isfinite(full_val), np.isfinite(trunc_val)
            assert full_finite == trunc_finite, (
                f"LEAK at bar {i}, {name}: finite/NaN mismatch (full={full_val}, trunc={trunc_val})"
            )
            if full_finite:
                assert abs(full_val - trunc_val) < 1e-6, (
                    f"LEAK at bar {i}, {name}: full={full_val} != trunc={trunc_val}"
                )

    return True


if __name__ == "__main__":
    import time

    # Same collision as every other file that needs both directories —
    # see macd_features.py's comment for the full story. ML_DIR must be
    # present for `from features import atr`, but ML_MACD_DIR must be
    # moved BACK to position 0 afterward, or `from data import
    # load_candles` below would silently resolve to ml/data.py instead
    # of ml_macd/data.py (caught here before it shipped, not after).
    ML_DIR = os.path.abspath(os.path.join(ML_MACD_DIR, "..", "ml"))
    if ML_DIR not in sys.path:
        sys.path.insert(0, ML_DIR)
    if ML_MACD_DIR in sys.path:
        sys.path.remove(ML_MACD_DIR)
    sys.path.insert(0, ML_MACD_DIR)

    from data import load_candles  # noqa: E402  (ml_macd/data.py)
    from features import atr  # noqa: E402  (ml/features.py)

    print("=" * 70)
    print("ROLLING PROFILE — required leakage test, real BTC/USDT data")
    print("=" * 70)
    candles = load_candles("BTC/USDT", "1h", "crypto")
    atr14 = atr(candles["high"], candles["low"], candles["close"], 14)
    t0 = time.time()
    assert_rolling_profile_strictly_prior(candles, "real_volume", atr14, window=200)
    print(f"PASS: rolling profile never references bars at or after the bar "
         f"it's attached to (structural + empirical truncate-and-recompute), "
         f"in {time.time() - t0:.1f}s")
