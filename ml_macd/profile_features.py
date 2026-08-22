"""
ml_macd/profile_features.py
===================================================================
Converts session_profile.py's per-SESSION profiles into per-BAR
features — the shape a model actually consumes. Every feature here
references the most recently CLOSED prior session's profile, never a
bar's own (possibly still-forming) session — the leakage boundary
session_profile.py's assert_no_current_session_leak() checks.

Features (namespaced `session_{mode}_...` per PART 3's "namespace
them clearly"): dist_to_poc/vah/val (signed, /ATR), inside_value_area,
dist_to_nearest_hvn/lvn/volume_edge (/ATR), poc_migration (/ATR),
value_area_width (/ATR), bars_since_poc_touch, naked_poc_distance
(/ATR), naked_poc_age.

NAKED-POC LOOKBACK IS BOUNDED, a disclosed engineering choice, not a
spec requirement: searching ALL naked POCs ever formed (thousands,
over a multi-year crypto history) for every single bar is O(n_bars x
n_naked_pocs) — tens of millions of comparisons, impractically slow
in pure Python. `naked_lookback_sessions` (default 60, ~2 months)
bounds the search to recently-formed naked levels, which is also the
economically sensible choice: an 8-year-old untouched POC is not a
price level traders are meaningfully watching today.
===================================================================
"""

import numpy as np

from session_profile import session_boundaries


def extract_session_profile_features(candles, profiles, naked_records, atr14,
                                      mode, naked_lookback_sessions=60):
    """
    Returns a dict of namespaced feature arrays (each length n, the
    number of bars in `candles`). NaN wherever no reference profile
    exists yet (before the first closed session) or ATR isn't warmed
    up — never fabricated.
    """
    close = candles["close"]
    n = len(close)
    open_time_s = candles["open_time"].astype(np.int64)
    session_ids = session_boundaries(open_time_s)

    closed_profiles = {p["session_id"]: p for p in profiles if not p["in_progress"]}
    sorted_sids = sorted(closed_profiles.keys())
    naked_by_session = {r["session_id"]: r for r in naked_records}

    prefix = f"session_{mode}_"
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
        prefix + "bars_since_poc_touch": np.full(n, np.nan),
        prefix + "naked_poc_distance_atr": np.full(n, np.nan),
        prefix + "naked_poc_age": np.full(n, np.nan),
    }

    # Sliding bounded window of "live" naked candidates, maintained as
    # we scan forward — avoids re-scanning the full naked_records list
    # per bar. A record enters when its session closes (formed_at_idx),
    # leaves when consumed or once naked_lookback_sessions have passed.
    active = []  # list of naked_records dicts currently in the window
    next_record_ptr = 0
    records_by_formed = sorted(naked_records, key=lambda r: r["formed_at_idx"])

    import bisect
    for i in range(n):
        sid = session_ids[i]
        pos = bisect.bisect_left(sorted_sids, sid)

        # Admit newly-formed naked records into the active window.
        while (next_record_ptr < len(records_by_formed)
              and records_by_formed[next_record_ptr]["formed_at_idx"] < i):
            active.append(records_by_formed[next_record_ptr])
            next_record_ptr += 1

        # Drop consumed-as-of-i or too-old-to-matter records.
        cutoff_sid = sorted_sids[max(0, pos - naked_lookback_sessions - 1)] if sorted_sids else None
        active = [
            r for r in active
            if (r["consumed_at_idx"] is None or r["consumed_at_idx"] > i)
            and (cutoff_sid is None or r["session_id"] >= cutoff_sid)
        ]

        if pos == 0:
            continue
        ref_sid = sorted_sids[pos - 1]
        ref_profile = closed_profiles[ref_sid]

        a = atr14[i]
        if not np.isfinite(a) or a <= 0:
            continue

        out[prefix + "dist_to_poc_atr"][i] = (close[i] - ref_profile["poc"]) / a
        out[prefix + "dist_to_vah_atr"][i] = (close[i] - ref_profile["vah"]) / a
        out[prefix + "dist_to_val_atr"][i] = (close[i] - ref_profile["val"]) / a
        out[prefix + "inside_value_area"][i] = float(
            ref_profile["val"] <= close[i] <= ref_profile["vah"])
        out[prefix + "value_area_width_atr"][i] = (ref_profile["vah"] - ref_profile["val"]) / a

        if ref_profile["hvn"]:
            nearest = min(ref_profile["hvn"], key=lambda p: abs(p - close[i]))
            out[prefix + "dist_to_hvn_atr"][i] = (close[i] - nearest) / a
        if ref_profile["lvn"]:
            nearest = min(ref_profile["lvn"], key=lambda p: abs(p - close[i]))
            out[prefix + "dist_to_lvn_atr"][i] = (close[i] - nearest) / a
        if ref_profile["volume_edge"]:
            nearest = min(ref_profile["volume_edge"], key=lambda p: abs(p - close[i]))
            out[prefix + "dist_to_volume_edge_atr"][i] = (close[i] - nearest) / a

        if pos >= 2:
            prev_profile = closed_profiles[sorted_sids[pos - 2]]
            out[prefix + "poc_migration_atr"][i] = (ref_profile["poc"] - prev_profile["poc"]) / a

        ref_naked = naked_by_session.get(ref_sid)
        if ref_naked is not None and ref_naked["consumed_at_idx"] is not None and ref_naked["consumed_at_idx"] <= i:
            out[prefix + "bars_since_poc_touch"][i] = i - ref_naked["consumed_at_idx"]

        if active:
            nearest = min(active, key=lambda r: abs(r["poc"] - close[i]))
            out[prefix + "naked_poc_distance_atr"][i] = (close[i] - nearest["poc"]) / a
            out[prefix + "naked_poc_age"][i] = i - nearest["formed_at_idx"]

    return out


def poc_agreement(profiles_tpo, profiles_volume, atr14):
    """
    PART 3 cross-mode agreement, crypto-only (README.md section 2 —
    FX has no volume POC to compare against, always NaN there):
    |poc_tpo - poc_volume| / ATR, per session where both profiles
    exist. Two uses per the brief: a data-quality diagnostic (report
    the distribution) and a direct model feature (fed in like any
    other). Returns {session_id: value}.
    """
    tpo_by_sid = {p["session_id"]: p for p in profiles_tpo if not p["in_progress"]}
    vol_by_sid = {p["session_id"]: p for p in profiles_volume if not p["in_progress"]}
    out = {}
    for sid in sorted(set(tpo_by_sid) & set(vol_by_sid)):
        end_idx = tpo_by_sid[sid]["end_idx"]
        a = atr14[end_idx]
        if not np.isfinite(a) or a <= 0:
            continue
        out[sid] = abs(tpo_by_sid[sid]["poc"] - vol_by_sid[sid]["poc"]) / a
    return out


# ------------------------------------------------------------------ #
# Required leakage test — the FULL pipeline, not just profile labels #
# ------------------------------------------------------------------ #

def assert_no_lookahead_profile_features(candles, atr_fn, mode="tpo",
                                         check_idxs=None, tolerance=1e-6):
    """
    PART 3's own explicit warning: "a session profile summarizes a
    whole session... this is the easiest place in this whole system
    to leak the future." session_profile.py's
    assert_no_current_session_leak() only checks that a profile isn't
    MISLABELED as closed — this checks the actual downstream feature
    values, truncate-and-recompute, same method as every other
    look-ahead test in this project.

    `atr_fn` — caller passes ml/features.py's `atr` (not imported here
    to avoid this file needing the cross-package sys.path dance every
    other file already documents once).
    """
    close_full = candles["close"]
    n = len(close_full)
    atr14_full = atr_fn(candles["high"], candles["low"], candles["close"], 14)

    def build_pipeline(c, a):
        from session_profile import build_session_profiles, track_naked_pocs
        profiles = build_session_profiles(c, mode, a)
        naked = track_naked_pocs(profiles, c["high"], c["low"])
        return extract_session_profile_features(c, profiles, naked, a, mode=mode)

    feats_full = build_pipeline(candles, atr14_full)

    if check_idxs is None:
        check_idxs = np.linspace(n // 3, n - 2, 4).astype(int)

    failures = []
    for i in check_idxs:
        truncated = {k: (v[: i + 1] if isinstance(v, np.ndarray) else v) for k, v in candles.items()}
        atr14_trunc = atr_fn(truncated["high"], truncated["low"], truncated["close"], 14)
        feats_trunc = build_pipeline(truncated, atr14_trunc)
        for name in feats_full:
            full_val = feats_full[name][i]
            trunc_val = feats_trunc[name][-1]
            full_finite, trunc_finite = np.isfinite(full_val), np.isfinite(trunc_val)
            if full_finite != trunc_finite:
                failures.append((int(i), name, "finite/NaN mismatch", full_val, trunc_val))
            elif full_finite and abs(full_val - trunc_val) > tolerance:
                failures.append((int(i), name, "value differs", full_val, trunc_val))

    if failures:
        lines = [f"    bar {i}: {name} ({reason}) full={fv} trunc={tv}"
                for i, name, reason, fv, tv in failures[:12]]
        raise AssertionError(
            "LOOK-AHEAD DETECTED in profile features:\n" + "\n".join(lines)
        )
    return True
