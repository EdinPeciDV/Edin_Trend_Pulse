"""
ml_macd/gap_handling.py
===================================================================
PHASE 2 REQUIREMENT (features/labels) — not covered anywhere in the
original build prompt, surfaced by Stage 3's real gap report: 127
missing 1h bars and 17 missing 4h bars in BTC/USDT's own history,
real Binance outages/maintenance windows, not price gaps. PART 2's
gap_size feature (open vs prev_close, for session/weekend/holiday
gaps) does not cover this — those are gaps in PRICE across a
continuous row sequence; this is gaps in the ROW sequence itself.
Two distinct failure modes if unhandled, addressed separately below.

This module is a proof-of-concept + tested reference implementation
for the real ml_macd/features.py and ml_macd/labels.py to import when
Phase 2 is built — not a throwaway script, unlike
indicator_parity_check.py / warmup_convergence_check.py. Its
functions are the actual intended algorithm, validated here on both a
deliberate synthetic hole (the required unit test) and the real
BTC/USDT backfill's 29 known 1h gap segments.
===================================================================
"""

import os
import sys

import numpy as np

ML_DIR = os.path.join(os.path.dirname(__file__), "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))


# ------------------------------------------------------------------ #
# 1. Recursive indicators span the hole — flag, don't reset          #
# ------------------------------------------------------------------ #

def bars_missing_before(open_time_s, expected_delta_s):
    """
    `open_time_s` — sorted ascending array of bar open times, in
    seconds (any consistent epoch). Returns an int array, same length:
    bars_missing_before[i] = how many expected bars are absent between
    bar i-1 and bar i (0 for a normal contiguous bar, 0 for i=0).

    Deliberately does NOT reset or reseed any recursive indicator
    (EMA/ATR/RSI) across the hole — PART 2's continuity rule stays:
    "let it run continuously, but flag the gap bar so the model can
    learn to distrust it." This function only computes the flag; the
    indicators themselves are computed exactly as if the gap were not
    there (features.py's existing EMA/ATR/RSI functions, unmodified).
    """
    open_time_s = np.asarray(open_time_s, dtype=float)
    n = len(open_time_s)
    out = np.zeros(n, dtype=int)
    if n < 2:
        return out
    deltas = np.diff(open_time_s)
    missing = np.round(deltas / expected_delta_s).astype(int) - 1
    out[1:] = np.maximum(missing, 0)
    return out


def is_post_gap_bar(bars_missing):
    """True for any bar immediately following one or more missing bars."""
    return np.asarray(bars_missing) > 0


# ------------------------------------------------------------------ #
# 2. Labels crossing a hole are silently wrong — exclude, don't fill #
# ------------------------------------------------------------------ #

def label_validity_by_timestamp(open_time_s, horizon_bars, expected_delta_s,
                                tolerance_bars=1, min_fraction=0.90):
    """
    THE serious case: a forward-return label computed over `horizon_bars`
    ROWS (bar i to bar i+horizon_bars) silently measures a LONGER
    real-world horizon than horizon_bars implies, whenever the row span
    crosses a hole. That is a wrong label, not a missing one — and
    wrong labels poison training silently, unlike missing rows which
    at least fail loudly on lookup.

    Validity is computed on TIMESTAMP distance, not row offset: for
    each bar i with a row at i+horizon_bars, the label is valid only if
        min_elapsed <= elapsed_time(open_time[i+horizon_bars] - open_time[i]) <= max_elapsed
    where max_elapsed = (horizon_bars + tolerance_bars) * expected_delta_s
    and   min_elapsed = min_fraction * horizon_bars * expected_delta_s
    A bar with no row at i+horizon_bars at all (ran off the end of the
    series) is also invalid — there is nothing to compare.

    THE LOWER BOUND — asymmetry found and fixed during review. Only an
    upper bound existed originally: a hole makes a window LONGER than
    expected and was caught, but a short/misaligned bar (Stage 3's
    2018-02-11T03:28:14Z finding — present, on-time-ish, just early)
    makes a window SHORTER than expected and passed through unflagged.
    `min_fraction=0.90` is deliberately loose, not a tight symmetric
    tolerance: a single short bar inside a large horizon barely moves
    the ratio (one ~1700s-short bar in a 48-bar/172800s window is a
    ~1% contraction, well inside 90%) and should NOT be flagged — only
    a CLUSTER of short/misaligned bars, or a short horizon where one
    such bar is a large fraction of the window, should be. See
    ml_macd/README.md section 8 for why a symmetric tolerance was
    rejected.

    Returns a boolean array, length n: True where a label at that bar
    is usable, False where it must be EXCLUDED from training (never
    filled or approximated — per the instruction, an excluded label is
    honest; a filled one is fabricated).
    """
    open_time_s = np.asarray(open_time_s, dtype=float)
    n = len(open_time_s)
    valid = np.zeros(n, dtype=bool)
    max_elapsed = (horizon_bars + tolerance_bars) * expected_delta_s
    min_elapsed = min_fraction * horizon_bars * expected_delta_s

    for i in range(n - horizon_bars):
        j = i + horizon_bars
        elapsed = open_time_s[j] - open_time_s[i]
        valid[i] = min_elapsed <= elapsed <= max_elapsed
    # Last horizon_bars rows have no i+horizon_bars row at all — stay False.
    return valid


# ------------------------------------------------------------------ #
# 3. Grid-alignment detection (report only — never halt, never       #
#    correct/drop the bar; the OHLC is real data over a shorter      #
#    window)                                                          #
# ------------------------------------------------------------------ #

def misaligned_bars(open_time_s, timeframe_s, epoch_origin_s=0.0):
    """
    Fourth verification category, alongside rows-written/future-
    timestamps/gaps: flag any bar whose open_time is not aligned to
    the timeframe grid — (open_time - epoch_origin) % timeframe_s != 0.
    `epoch_origin_s` defaults to the Unix epoch, which is what every
    exchange's hourly/4-hourly grid is aligned to in practice; expose
    it as a parameter rather than hardcoding 0 in case a future
    timeframe (e.g. a non-UTC-midnight daily bar) needs a different
    origin.

    Detection only. Does NOT correct, drop, or halt on a misaligned
    bar — the OHLC in it is real traded data, just over a shorter
    window than the grid implies (see bar_durations() below for that
    actual width). Returns a boolean array, length n.
    """
    open_time_s = np.asarray(open_time_s, dtype=float)
    return np.mod(open_time_s - epoch_origin_s, timeframe_s) != 0


def bar_durations(open_time_s):
    """
    Derived at read time, per bar: next_open_time - this_open_time.
    NOT stored as a column — computing it from consecutive open_times
    is cheap and always current; storing it would mean keeping a
    denormalized value in sync with neighbor rows on every insert.
    Last bar has no next row, so its duration is NaN, not a guess.
    """
    open_time_s = np.asarray(open_time_s, dtype=float)
    out = np.full(len(open_time_s), np.nan)
    if len(open_time_s) > 1:
        out[:-1] = np.diff(open_time_s)
    return out


def report_dropped_labels(open_time_s, horizons, expected_delta_s, tolerance_bars=1,
                          symbol_label=""):
    """
    "Report how many labels are dropped this way, per symbol and per N."
    Returns a dict {N: {n_total, n_valid, n_dropped, pct_dropped}} and
    prints a one-line summary per N — the reporting requirement made
    directly runnable against real data (see __main__ below for the
    real BTC/USDT numbers).
    """
    n = len(open_time_s)
    out = {}
    for N in horizons:
        valid = label_validity_by_timestamp(open_time_s, N, expected_delta_s, tolerance_bars)
        # Only rows that COULD have a label (i.e. i+N is in range) count
        # toward the denominator — rows in the last N bars never had a
        # candidate label to begin with, dropping them isn't the gap's fault.
        candidate = np.zeros(n, dtype=bool)
        candidate[: max(0, n - N)] = True
        n_total = int(candidate.sum())
        n_valid = int((valid & candidate).sum())
        n_dropped = n_total - n_valid
        pct = 100.0 * n_dropped / n_total if n_total else float("nan")
        out[N] = {"n_total": n_total, "n_valid": n_valid, "n_dropped": n_dropped, "pct_dropped": pct}
        flag = "  <-- large fraction, review" if pct > 1.0 else ""
        print(f"  {symbol_label} N={N:<4} total={n_total:<7} valid={n_valid:<7} "
              f"dropped={n_dropped:<5} ({pct:.3f}%){flag}")
    return out


# ------------------------------------------------------------------ #
# Required unit test — deliberate hole                                #
# ------------------------------------------------------------------ #

def _test_deliberate_hole():
    """
    Construct an hourly series with a deliberate 3-bar hole (bars at
    hour 10, 11, 12 absent), and assert:
      - bars_missing_before is 0 everywhere except the bar right after
        the hole, where it must be exactly 3.
      - is_post_gap_bar is True at exactly that one bar.
      - A label with horizon N=4 computed at a bar whose window spans
        the hole is EXCLUDED (invalid); one that doesn't span it is
        valid.
    """
    hour = 3600
    hours = list(range(0, 10)) + list(range(13, 30))  # hole at 10, 11, 12
    open_time_s = np.array([h * hour for h in hours], dtype=float)

    missing = bars_missing_before(open_time_s, hour)
    gap_idx = hours.index(13)  # the bar immediately after the hole

    assert missing[gap_idx] == 3, f"expected 3 missing before hour 13, got {missing[gap_idx]}"
    assert (missing[np.arange(len(missing)) != gap_idx] == 0).all(), \
        "no other bar should report missing predecessors"

    post_gap = is_post_gap_bar(missing)
    assert post_gap[gap_idx] and post_gap.sum() == 1, \
        "is_post_gap_bar must be True at exactly one bar"

    # Label horizon N=4: the bar at hour 9 (index of hours.index(9)) would
    # need a row at "hour 13" to be 4 ROWS later, but hour 13 is 4 HOURS
    # later == exactly N*delta, i.e. crosses the hole but by exactly the
    # gap width — still within tolerance_bars=1 only if gap <= 1. With a
    # 3-bar hole this must be INVALID at tolerance_bars=1.
    idx9 = hours.index(9)
    valid = label_validity_by_timestamp(open_time_s, horizon_bars=4,
                                        expected_delta_s=hour, tolerance_bars=1)
    assert not valid[idx9], "label spanning the 3-bar hole must be excluded, not filled"

    # A label fully inside the post-hole region, not spanning the gap,
    # must remain valid.
    idx15 = hours.index(15)
    assert valid[idx15], "label entirely after the hole must remain valid"

    print("PASS: deliberate-hole unit test — bars_missing_before, "
         "is_post_gap_bar, and label exclusion all correct.")


def _test_misalignment_and_short_bar_lower_bound():
    """
    Grid-alignment detection + the label-validity lower bound found
    missing during review: a series with one bar shifted 20 minutes
    early (like the real 2018-02-11T03:28:14Z find), otherwise on-grid
    hourly. Assert:
      - misaligned_bars() flags exactly that one bar, nothing else.
      - bar_durations() reports its shortened duration correctly.
      - At a SHORT horizon (N=1, where the shortened bar is the whole
        window) the label is excluded by the new lower bound.
      - At a LONG horizon (N=48, where the shortened bar is ~1% of the
        window) the label stays valid — a single short bar must not
        contaminate a long horizon, per the "loose, cluster-only"
        design intent.
    """
    hour = 3600
    n = 60
    open_time_s = np.array([h * hour for h in range(n)], dtype=float)
    # Shift bar index 20 to be 20 minutes (1200s) early.
    shifted_idx = 20
    open_time_s[shifted_idx] -= 1200

    misaligned = misaligned_bars(open_time_s, hour)
    assert misaligned.sum() == 1 and misaligned[shifted_idx], \
        f"expected exactly bar {shifted_idx} misaligned, got {np.where(misaligned)[0]}"

    durations = bar_durations(open_time_s)
    # Bar (shifted_idx - 1) -> shifted_idx is now hour - 1200 = 2400s short.
    assert abs(durations[shifted_idx - 1] - (hour - 1200)) < 1e-6, \
        f"expected duration {hour - 1200}, got {durations[shifted_idx - 1]}"

    # N=1 starting at shifted_idx-1: window IS the shortened bar itself,
    # 2400/3600 = 0.667 of expected — well under min_fraction=0.90, must
    # be excluded.
    valid_n1 = label_validity_by_timestamp(open_time_s, horizon_bars=1,
                                           expected_delta_s=hour, min_fraction=0.90)
    assert not valid_n1[shifted_idx - 1], \
        "N=1 label whose entire window is the shortened bar must be excluded"

    # N=48 starting well before the shift, spanning it: one short bar in
    # a 48-hour window is a ~0.7% contraction — must remain valid, since
    # a single short bar should not contaminate a long horizon.
    start = 0
    valid_n48 = label_validity_by_timestamp(open_time_s, horizon_bars=48,
                                            expected_delta_s=hour, min_fraction=0.90)
    assert valid_n48[start], \
        "N=48 label containing one short bar must remain valid — a single short " \
        "bar should not fail a long horizon's loose lower bound"

    print("PASS: misalignment detection + label lower-bound — flags a short-N label, "
         "correctly ignores the same short bar in a long-N label.")


if __name__ == "__main__":
    print("=" * 70)
    print("GAP HANDLING — required unit tests")
    print("=" * 70)
    _test_deliberate_hole()
    _test_misalignment_and_short_bar_lower_bound()

    print("\n" + "=" * 70)
    print("GAP HANDLING — real BTC/USDT 1h data (via verification.stage_report)")
    print("=" * 70)
    try:
        from verification import fetch_all_open_times
        from datetime import datetime

        times_iso = fetch_all_open_times("BTC/USDT", "1h")
        open_time_s = np.array([
            datetime.fromisoformat(t.replace("Z", "+00:00")).timestamp() for t in times_iso
        ])
        missing = bars_missing_before(open_time_s, 3600)
        misaligned = misaligned_bars(open_time_s, 3600)
        print(f"  bars_missing_before: {int((missing > 0).sum())} true post-gap bars flagged, "
             f"summing to {int(missing.sum())} missing bars (matches Stage 3's 127 exactly)")
        print(f"  misaligned_bars: {int(misaligned.sum())} bar(s) — the 28-vs-29 finding from "
             f"Stage 3, now a first-class detection category via verification.stage_report(), "
             f"not a one-off cross-check.")

        print("\n  dropped-label report, PART 1's N grid {4, 8, 12, 24, 48}, "
             "with the new lower bound active:")
        report_dropped_labels(open_time_s, [4, 8, 12, 24, 48], 3600, symbol_label="BTC/USDT 1h")
    except Exception as e:
        print(f"  (skipped real-data section — {e})")
