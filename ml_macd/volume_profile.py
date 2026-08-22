"""
ml_macd/volume_profile.py
===================================================================
PART 3: ONE profile engine, pluggable weighting mode — not three
separate implementations for real_volume/tick_volume/tpo.

    mode="real_volume"  -> weight = bar volume        (crypto)
    mode="tick_volume"  -> weight = number_of_trades   (proxy; NOT
                            used for FX in this project — see below)
    mode="tpo"          -> weight = 1 per bar (a time slice)

FX MODE ASSIGNMENT (README.md section 2, binding decision, confirmed
live via scripts/probe_twelvedata.py): Twelve Data returns NO volume
or trade-count data for FX at all — not degenerate, absent. FX profiles
are TPO-ONLY. `tick_volume` mode stays implemented (generic, usable
for crypto or a future CME-futures source per PART 4's deferred
extension point) but is never invoked for FX in this codebase.

BIN WIDTH is a function of ATR, not a fixed price step (PART 3): a
5-pip EUR/USD bin and a $5 BTC bin are not comparable. The ATR
multiple is configurable; the resolved bin width is logged per
profile (`resolved_bin_width` in every profile's return value) so it
is never a silent, undocumented number.

VOLUME/WEIGHT DISTRIBUTION WITHIN A BAR (PART 3, explicit
requirement): a bar spans multiple bins. Weight is distributed
PROPORTIONALLY TO OVERLAP LENGTH across every bin the bar's
[low, high] range touches — NOT dumped into the close's bin. This is
an approximation (real intra-bar trade distribution is unknown from
OHLCV alone) and is documented here, visibly, per the brief's own
instruction to make that visible rather than hide it.
===================================================================
"""

import numpy as np

DEFAULT_ATR_BIN_MULTIPLE = 0.10   # bin width = 10% of the reference ATR
DEFAULT_VALUE_AREA_PCT = 0.70


def _resolve_bin_width(atr_ref, atr_bin_multiple):
    """
    `atr_ref` — the ATR value used to size bins for this profile
    (typically the ATR at the profile's anchor/end bar). A profile
    spanning a volatility regime change uses ONE bin width throughout
    (resolved once, not re-sized bar-by-bar) — a profile with
    shifting bin widths mid-construction would make POC/VAH/VAL
    comparisons across time meaningless.
    """
    width = atr_bin_multiple * atr_ref
    if not np.isfinite(width) or width <= 0:
        raise ValueError(f"resolved bin width is invalid ({width}) — check atr_ref={atr_ref}")
    return float(width)


def build_profile(high, low, weight, atr_ref, mode, profile_source,
                  atr_bin_multiple=DEFAULT_ATR_BIN_MULTIPLE,
                  value_area_pct=DEFAULT_VALUE_AREA_PCT, in_progress=False):
    """
    Build one profile from a window of bars. `high`/`low`/`weight` are
    arrays for exactly the bars in this window (caller slices — this
    function has no notion of anchoring, see the *_profile() functions
    below for that).

    `weight` — already mode-resolved by the caller: volume for
    real_volume, number_of_trades for tick_volume, ones_like(high) for
    tpo. Keeping mode-selection at the call site (not inside this
    function) is what makes this ONE engine rather than three — the
    only thing that differs between modes is which array gets passed
    as `weight`.

    Returns a dict: bin_edges, bin_weights, resolved_bin_width, poc,
    vah, val, hvn (list of bin centers), lvn (list of bin centers),
    volume_edge (list of bin centers), total_weight, mode,
    profile_source, in_progress.

    `in_progress=True` for a still-forming (current, incomplete)
    session/window — PART 3: "the current session's profile is
    incomplete by definition... exclude in-progress profiles from
    training features." This function still builds it (useful for
    live display) but the flag travels with the result so callers can
    filter it out.
    """
    n = len(high)
    if n == 0:
        raise ValueError("cannot build a profile from zero bars")

    bin_width = _resolve_bin_width(atr_ref, atr_bin_multiple)
    price_low = float(np.min(low))
    price_high = float(np.max(high))
    if price_high <= price_low:
        price_high = price_low + bin_width

    n_bins = max(1, int(np.ceil((price_high - price_low) / bin_width)))
    bin_edges = price_low + bin_width * np.arange(n_bins + 1)
    bin_weights = np.zeros(n_bins)

    # Distribute each bar's weight proportionally to overlap length
    # across every bin its [low, high] range touches.
    for i in range(n):
        lo, hi = low[i], high[i]
        if hi <= lo:
            hi = lo + 1e-12  # degenerate zero-range bar — still gets its own bin
        bar_range = hi - lo
        first_bin = max(0, int((lo - price_low) // bin_width))
        last_bin = min(n_bins - 1, int((hi - price_low) // bin_width))
        for b in range(first_bin, last_bin + 1):
            b_lo = bin_edges[b]
            b_hi = bin_edges[b + 1]
            overlap = min(hi, b_hi) - max(lo, b_lo)
            if overlap > 0:
                bin_weights[b] += weight[i] * (overlap / bar_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    total_weight = float(bin_weights.sum())

    poc_bin = int(np.argmax(bin_weights))
    poc = float(bin_centers[poc_bin])

    vah, val = _value_area(bin_weights, bin_centers, poc_bin, value_area_pct)
    hvn, lvn = _local_extrema(bin_weights, bin_centers)
    volume_edge = _steepest_gradient_bins(bin_weights, bin_centers)

    return {
        "bin_edges": bin_edges, "bin_weights": bin_weights,
        "bin_centers": bin_centers, "resolved_bin_width": bin_width,
        "poc": poc, "vah": vah, "val": val,
        "hvn": hvn, "lvn": lvn, "volume_edge": volume_edge,
        "total_weight": total_weight, "mode": mode,
        "profile_source": profile_source, "in_progress": in_progress,
        "price_low": price_low, "price_high": price_high,
    }


def _value_area(bin_weights, bin_centers, poc_bin, value_area_pct):
    """
    Grow outward from POC, alternating to whichever neighbor bin (up
    or down) has more weight, until the accumulated weight reaches
    `value_area_pct` of the total. Standard value-area construction.
    """
    total = bin_weights.sum()
    if total <= 0:
        return float(bin_centers[poc_bin]), float(bin_centers[poc_bin])

    target = value_area_pct * total
    lo_bin = hi_bin = poc_bin
    accumulated = bin_weights[poc_bin]

    while accumulated < target and (lo_bin > 0 or hi_bin < len(bin_weights) - 1):
        next_lo_w = bin_weights[lo_bin - 1] if lo_bin > 0 else -1
        next_hi_w = bin_weights[hi_bin + 1] if hi_bin < len(bin_weights) - 1 else -1
        if next_hi_w >= next_lo_w:
            hi_bin += 1
            accumulated += bin_weights[hi_bin]
        else:
            lo_bin -= 1
            accumulated += bin_weights[lo_bin]

    return float(bin_centers[hi_bin]), float(bin_centers[lo_bin])


def _local_extrema(bin_weights, bin_centers, min_bins=3):
    """
    HVN (high volume nodes) — local peaks in the weight distribution.
    LVN (low volume nodes) — local troughs. Interior bins only (a
    profile's edge bin can't be a "local" anything without a neighbor
    on both sides). Returns lists of bin-center prices.
    """
    n = len(bin_weights)
    if n < min_bins:
        return [], []
    hvn, lvn = [], []
    for i in range(1, n - 1):
        if bin_weights[i] > bin_weights[i - 1] and bin_weights[i] > bin_weights[i + 1]:
            hvn.append(float(bin_centers[i]))
        elif bin_weights[i] < bin_weights[i - 1] and bin_weights[i] < bin_weights[i + 1]:
            lvn.append(float(bin_centers[i]))
    return hvn, lvn


def _steepest_gradient_bins(bin_weights, bin_centers, top_k=2):
    """
    volume_edge — bins where the weight gradient is steepest: the
    sharp drop-off from high to low weight, the reaction points PART 3
    describes. Returns the `top_k` bin-center prices with the largest
    |gradient|.
    """
    if len(bin_weights) < 2:
        return []
    grad = np.abs(np.diff(bin_weights))
    if grad.max() <= 0:
        return []
    idx = np.argsort(grad)[::-1][:top_k]
    # gradient at index i is between bin i and bin i+1 — report the
    # edge as the boundary price, not either bin's center.
    edges = (bin_centers[idx] + bin_centers[idx + 1]) / 2.0
    return sorted(float(e) for e in edges)
