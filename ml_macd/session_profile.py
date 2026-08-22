"""
ml_macd/session_profile.py
===================================================================
PART 3 anchoring window: SESSION profile, built fully and correctly.
ROLLING and FIXED-RANGE anchoring windows are DESIGNED, not built —
disclosed here rather than faked (see README.md section 19).

Session boundary: UTC calendar day, for BOTH crypto and forex.
Correct by PART 3's own words for crypto ("crypto uses UTC day").
For FX, PART 3 asks for "the trading day boundary, respect the
holiday calendar" — NO real FX holiday calendar exists in this
project (already disclosed in macd_features.py for
`bars_until_session_close`) — UTC calendar day is used as the same
documented, honest approximation, not a silent stand-in.

NAKED POC bookkeeping (PART 3): a forward pass over sessions, in
chronological order — a prior session's POC is "naked" until price
trades back through it, then permanently consumed. Implemented as an
explicit stateful scan (`track_naked_pocs()`), unit-tested for the
exact property PART 3 calls out: a POC consumed at bar T must not
still read naked at bar T+1.

Every profile here is built from CLOSED bars only, and the CURRENT
(still-forming) session is marked `in_progress=True` — features never
attach to it (see `assert_no_current_session_leak()` below).
===================================================================
"""

import os
import sys

import numpy as np

ML_MACD_DIR = os.path.dirname(os.path.abspath(__file__))
if ML_MACD_DIR in sys.path:
    sys.path.remove(ML_MACD_DIR)
sys.path.insert(0, ML_MACD_DIR)

from volume_profile import build_profile  # noqa: E402

SECONDS_PER_DAY = 86400


def session_boundaries(open_time_s):
    """
    UTC calendar day index for every bar — the session id each bar
    belongs to. Two bars share a session iff they fall on the same
    UTC calendar day.
    """
    return (np.asarray(open_time_s, dtype=np.int64) // SECONDS_PER_DAY)


def build_session_profiles(candles, weight_mode, atr14, atr_bin_multiple=0.10,
                           value_area_pct=0.70, current_time_s=None):
    """
    One profile per UTC session (day) present in `candles`. `weight_mode`
    is "real_volume", "tick_volume", or "tpo" — resolved to the right
    weight array here (the one place mode selection happens for this
    anchoring window).

    `current_time_s` — wall-clock "now," defaults to the last bar's
    open_time + one bar (i.e. the caller is asking about live/current
    state). The session containing `current_time_s` is marked
    `in_progress=True` and should never be used as a training feature
    (PART 3: "exclude in-progress profiles from training features").

    Returns a list of dicts, one per session, chronologically ordered,
    each the `build_profile()` result PLUS `session_id`,
    `start_idx`/`end_idx` (bar indices into `candles`, inclusive).
    """
    open_time_s = candles["open_time"].astype(np.int64)
    high, low, close = candles["high"], candles["low"], candles["close"]
    n = len(close)

    if weight_mode == "real_volume":
        weight_full = candles["volume"]
    elif weight_mode == "tick_volume":
        weight_full = candles["number_of_trades"]
    elif weight_mode == "tpo":
        weight_full = np.ones(n)
    else:
        raise ValueError(f"unknown weight_mode {weight_mode!r}")

    if weight_mode in ("real_volume", "tick_volume") and np.isnan(weight_full).all():
        # Fail loud, not silent: this is exactly the FX case (README.md
        # section 2 — Twelve Data returns no volume/trade-count data for
        # FX at all). Calling real_volume/tick_volume mode there would
        # silently build a profile from all-NaN weight instead of
        # respecting the binding "FX is TPO-only" decision.
        raise ValueError(
            f"weight_mode={weight_mode!r} but every weight value is NaN — "
            f"this looks like FX data (volume_is_proxy={candles.get('volume_is_proxy')}), "
            f"which is TPO-only per README.md section 2. Use mode='tpo'."
        )

    session_ids = session_boundaries(open_time_s)
    if current_time_s is None:
        current_time_s = open_time_s[-1] + 1
    current_session_id = current_time_s // SECONDS_PER_DAY

    profiles = []
    unique_sessions = np.unique(session_ids)
    for sid in unique_sessions:
        mask = session_ids == sid
        idxs = np.where(mask)[0]
        start_idx, end_idx = int(idxs[0]), int(idxs[-1])
        atr_ref = atr14[end_idx]
        if not np.isfinite(atr_ref) or atr_ref <= 0:
            continue  # ATR not warmed up yet for this session — skip, don't fabricate

        profile = build_profile(
            high[mask], low[mask], weight_full[mask], atr_ref,
            mode=weight_mode, profile_source=f"session_{weight_mode}",
            atr_bin_multiple=atr_bin_multiple, value_area_pct=value_area_pct,
            in_progress=bool(sid == current_session_id),
        )
        profile["session_id"] = int(sid)
        profile["start_idx"] = start_idx
        profile["end_idx"] = end_idx
        profiles.append(profile)

    return profiles


# ------------------------------------------------------------------ #
# Naked POC bookkeeping                                               #
# ------------------------------------------------------------------ #

def track_naked_pocs(profiles, high, low):
    """
    Forward pass, chronological order (PART 3: "write it as a forward
    pass over history"). For each session's POC, once that session is
    closed (`in_progress=False`), it starts naked. On every subsequent
    bar, check whether price has traded through it — touch is defined
    against that bar's own [low, high] range (intrabar), not just
    close-to-close, since a POC price can be touched by a wick without
    the close itself crossing it. Once touched, `consumed_at_idx` is
    set permanently and it never reads naked again.

    Returns a list of dicts, one per (non-in-progress) session's POC:
      {session_id, poc, formed_at_idx, consumed_at_idx (None if still
       naked as of the end of the series)}
    """
    records = []
    for p in profiles:
        if p["in_progress"]:
            continue
        records.append({
            "session_id": p["session_id"], "poc": p["poc"],
            "formed_at_idx": p["end_idx"], "consumed_at_idx": None,
        })

    for rec in records:
        start = rec["formed_at_idx"] + 1
        for i in range(start, len(high)):
            if low[i] <= rec["poc"] <= high[i]:
                rec["consumed_at_idx"] = i
                break
    return records


# ------------------------------------------------------------------ #
# Required tests                                                      #
# ------------------------------------------------------------------ #

def _test_naked_poc_consumed_not_naked_next_bar():
    """
    PART 3's explicit requirement: "unit test that a POC consumed at
    bar T is not still flagged naked at bar T+1."
    """
    n = 20
    # Price sits well ABOVE the POC (100.0) for every bar except bar 10,
    # which dips down to touch it — the earlier version of this fixture
    # used [95,105] for every bar, which already spans 100.0 everywhere
    # and "touched" on the very first post-formation bar instead of at
    # the intended bar 10. Caught by the assertion below, not assumed.
    high = np.full(n, 110.0)
    low = np.full(n, 105.0)
    high[10], low[10] = 101.0, 99.0

    fake_profile = {
        "poc": 100.0, "session_id": 0, "end_idx": 5, "in_progress": False,
    }
    records = track_naked_pocs([fake_profile], high, low)
    rec = records[0]
    assert rec["consumed_at_idx"] == 10, f"expected consumption at bar 10, got {rec['consumed_at_idx']}"

    # is_naked_at(t) helper, mirroring how a feature builder would query this.
    def is_naked_at(rec, t):
        if t <= rec["formed_at_idx"]:
            return False  # doesn't exist yet
        if rec["consumed_at_idx"] is None:
            return True
        return t < rec["consumed_at_idx"]

    assert not is_naked_at(rec, 10), "POC touched AT bar 10 must not read naked at bar 10 itself"
    assert not is_naked_at(rec, 11), "POC consumed at bar 10 must NOT still be naked at bar 11"
    assert is_naked_at(rec, 9), "POC must still read naked the bar before it's touched"

    print("PASS: naked POC consumed at bar T is correctly not-naked at T and T+1.")


def assert_no_current_session_leak(profiles, current_time_s):
    """
    PART 3 leakage requirement: "a feature derived from the current
    session's completed profile must only be attached to bars AFTER
    that session closed." Asserts no profile marked in_progress=False
    actually still contains `current_time_s` within its own session
    boundary — i.e. every "closed" profile really is closed.
    """
    current_session_id = current_time_s // SECONDS_PER_DAY
    for p in profiles:
        if not p["in_progress"]:
            assert p["session_id"] != current_session_id, (
                f"LEAK: session {p['session_id']} marked closed but IS the current session"
            )
    return True


def _test_no_current_session_leak():
    n = 300
    open_time_s = np.arange(n) * 3600  # hourly bars, ~12.5 days of data
    close = 100 + np.cumsum(np.random.default_rng(0).normal(0, 0.1, n))
    high = close + 0.5
    low = close - 0.5
    candles = {
        "open_time": open_time_s, "high": high, "low": low, "close": close,
        "volume": np.ones(n), "number_of_trades": np.ones(n), "timeframe": "1h",
    }
    atr14 = np.full(n, 1.0)
    current_time_s = open_time_s[-1] + 3600
    profiles = build_session_profiles(candles, "tpo", atr14, current_time_s=current_time_s)
    assert_no_current_session_leak(profiles, current_time_s)
    assert any(p["in_progress"] for p in profiles), "the current session must be present and flagged in_progress"
    print("PASS: no closed profile is actually the current (in-progress) session.")


if __name__ == "__main__":
    print("=" * 70)
    print("SESSION PROFILE — required tests")
    print("=" * 70)
    _test_naked_poc_consumed_not_naked_next_bar()
    _test_no_current_session_leak()
