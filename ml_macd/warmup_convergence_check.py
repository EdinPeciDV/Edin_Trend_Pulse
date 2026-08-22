"""
ml_macd/warmup_convergence_check.py
===================================================================
PART 4 correction #1 (WARMUP STATE): MACD's EMA is recursive with
unbounded memory. ml/features.py's `_ema()`, `wilder_rsi()`, and
`atr()` are all seeded on the FIRST bar of whatever series they are
given — feed them a short trailing window at serve time and a long
history at train time, and even identical code produces different
values at the same timestamp, because the recursion's "memory" from
bars before the window start is simply absent in the short-window
case. This is not a bug to fix; it is a property of the math that has
to be sized around.

This script answers: how many bars of history does the LIVE serving
job need to feed ml_macd/features.py's recursive indicators before,
seeded fresh, they converge to the same value a full-history
computation would give at the current bar?

Method: build one long synthetic series. For each candidate
`warmup_bars` W, compute each recursive indicator TWICE — once seeded
on the trailing W bars, once seeded on the trailing 4*W bars — and
diff their value at the identical final bar. If the two windows
(1x and 4x the candidate) already agree, doubling the window further
cannot change the answer materially (the recursion's own contraction
rate guarantees monotonically shrinking disagreement), so this
comparison is sufficient evidence of convergence without needing an
"infinite" reference series.

Run: python3 ml_macd/warmup_convergence_check.py
===================================================================
"""

import os
import sys

import numpy as np

ML_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "ml"))
if ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
# Idempotent insert, not unconditional — see indicator_parity_check.py's
# comment on the same line for why. This file's `import data` below
# deliberately means ml/data.py (its synthetic_random_walk()), not
# ml_macd/data.py — correct only because ml_macd/'s own directory is
# never added to sys.path here.

from features import _ema, wilder_rsi, atr  # noqa: E402
import data as ml_data  # noqa: E402

# Per-indicator tolerance for "converged". Chosen relative to each
# indicator's natural scale, not a single absolute number:
#   macd_hist_norm, atr_norm  — ratios of price, O(1e-3..1e-2) typical
#   rsi                        — 0-100 scale
TOLERANCES = {
    "ema26_norm": 1e-6,       # EMA26/price
    "macd_hist_norm": 1e-6,   # (ema12-ema26-signal_component)/price, see below
    "atr_norm": 1e-6,         # ATR14/price
    "rsi_14": 1e-3,           # 0-100 scale, Wilder-smoothed
}

CANDIDATES = [100, 150, 200, 260, 300, 400, 500, 650, 1000, 1500, 2000]


def indicators_at_end(close, high, low):
    """Compute the recursive indicators, seeded fresh on this exact
    window, and return their value at the FINAL bar."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(np.nan_to_num(macd, nan=0.0), 9)
    macd_hist_norm = (macd[-1] - signal[-1]) / close[-1]
    ema26_norm = ema26[-1] / close[-1]

    atr14 = atr(high, low, close, 14)
    atr_norm = atr14[-1] / close[-1] if np.isfinite(atr14[-1]) else np.nan

    rsi14 = wilder_rsi(close, 14)
    rsi_last = rsi14[-1] if np.isfinite(rsi14[-1]) else np.nan

    return {
        "ema26_norm": ema26_norm,
        "macd_hist_norm": macd_hist_norm,
        "atr_norm": atr_norm,
        "rsi_14": rsi_last,
    }


def run_for_series(close, high, low, label):
    print(f"\n--- {label} ---")
    n = len(close)
    header = f"{'W':>6}  " + "  ".join(f"{k:>16}" for k in TOLERANCES)
    print(header)

    converged_at = {}
    for w in CANDIDATES:
        w4 = 4 * w
        if w4 > n:
            print(f"{w:>6}  (skipped — need {w4} bars, series has {n})")
            continue

        v1 = indicators_at_end(close[-w:], high[-w:], low[-w:])
        v4 = indicators_at_end(close[-w4:], high[-w4:], low[-w4:])

        row = []
        all_ok = True
        for key, tol in TOLERANCES.items():
            diff = abs(v1[key] - v4[key])
            ok = diff <= tol
            all_ok &= ok
            row.append(f"{diff:>16.3e}{'*' if not ok else ' '}")
            if ok and key not in converged_at:
                converged_at[key] = w
        print(f"{w:>6}  " + "  ".join(row))

    print("  (* = exceeds tolerance; diff shown is |value@1x - value@4x| at the final bar)")
    print("  first W where each indicator converges:")
    for key in TOLERANCES:
        w = converged_at.get(key, "NOT REACHED within candidates")
        print(f"    {key:<16} {w}")
    return converged_at


def main():
    print("=" * 70)
    print("WARMUP CONVERGENCE CHECK — recursive indicators, seeded-window vs 4x")
    print("=" * 70)

    # Long crypto-like series (enough for the largest 4x candidate = 8000 bars).
    crypto = ml_data.synthetic_random_walk(n=9000, seed=42)
    run_for_series(crypto["close"], crypto["high"], crypto["low"], "crypto-like (5m bars)")

    # Higher-volatility case — convergence should be similar or faster,
    # since it's driven by the EMA decay constant, not price scale, but
    # worth confirming it's not scale-dependent.
    volatile = ml_data.synthetic_random_walk(n=9000, vol=0.02, seed=43)
    run_for_series(volatile["close"], volatile["high"], volatile["low"], "high-volatility (5m bars)")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("""
The floor stated in the brief — 10x the longest lookback, i.e. >=260
bars for EMA26 — is a reasonable rule of thumb but not, by itself,
evidence. Read the tables above: take the LARGEST "first W where
converged" across every indicator and every series tested, then round
up for margin (recursive EMA convergence is geometric, so a modest
safety margin costs little — going from converged-at-W to 2*W bars
of extra warmup is cheap; under-provisioning is what silently breaks
train/serve parity).

This script does not hardcode the final warmup_bars value — record
whatever the tables above actually show for THIS run's tolerances
in ml_macd/README.md, with the tolerance table alongside it, so the
number has a derivation attached rather than being a bare constant.
""")


if __name__ == "__main__":
    main()
