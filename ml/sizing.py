"""
ml/sizing.py
===================================================================
Turning an edge into a result (Phase 5 of PREDICTION_SPEC.md).

A real edge still loses money with wrong sizing. Four pieces, meant to
compose (see combined_position_size() at the bottom):

  1. size_from_probability()  — bet size from a CALIBRATED probability.
                                Only meaningful once Phase 1's
                                calibration is in place; sizing off a
                                raw, uncalibrated model score is sizing
                                off a number that isn't a probability.
  2. fractional_kelly()        — 0.25-0.5x Kelly. Full Kelly is optimal
                                only with KNOWN parameters; ours are
                                estimated with real error, and full Kelly
                                on estimated parameters routinely
                                produces catastrophic drawdowns.
  3. vol_target_size()         — scale inversely to forecast volatility
                                (ml.volatility) so risk per trade is
                                roughly constant regardless of regime.
  4. cost_model()               — fees + half-spread + slippage as a
                                function of size. Understating costs is
                                the most common way a backtest
                                manufactures an edge that vanishes live.

Every size returned here is a MULTIPLIER in [0, max_leverage], not a
dollar amount or share count — the caller applies it to whatever unit
their book uses.
===================================================================
"""

import math

import numpy as np


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# ------------------------------------------------------------------ #
# 1. Bet size from a calibrated probability                           #
# ------------------------------------------------------------------ #

def size_from_probability(p):
    """
    z = (p - 0.5) / sqrt(p(1-p)), mapped through the normal CDF.

    Naturally scales with confidence: p=0.50 -> size 0 (no edge, no
    bet), p->1 or p->0 -> size -> +/-1 (maximal conviction). Signed: a
    positive result means "size the primary/long side", negative means
    "size the opposite side" — the caller applies the sign to whichever
    direction the model/heuristic actually called.

    `p` MUST be a calibrated probability (Phase 1). A raw, uncalibrated
    classifier score fed into this formula produces a sizing decision
    that looks principled and isn't — the z-statistic assumes p really
    is P(win), not merely "high when the model is more confident".
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    z = (p - 0.5) / np.sqrt(p * (1 - p))
    signed = 2.0 * np.vectorize(_norm_cdf)(np.abs(z)) - 1.0
    return np.sign(p - 0.5) * signed


# ------------------------------------------------------------------ #
# 2. Fractional Kelly                                                 #
# ------------------------------------------------------------------ #

def fractional_kelly(p, payoff_ratio=2.0, fraction=0.375):
    """
    f* = p - (1-p)/b, the Kelly fraction for a bet that wins `b` units
    per unit risked with probability p and loses 1 unit with probability
    (1-p). `payoff_ratio` defaults to 2.0 to match this pipeline's
    triple-barrier k1/k2 = 2.0/1.0 reward:risk (Phase 0.1) — pass the
    ratio actually in force if you change those.

    `fraction` defaults to the middle of the spec's 0.25-0.5x range.
    Full Kelly (fraction=1.0) is optimal only with exactly-known p and
    payoff_ratio; both are estimated here with real error, and full
    Kelly on estimated parameters routinely produces catastrophic
    drawdowns the moment the estimate is even slightly optimistic.

    Returns a size in [0, fraction] — never negative. A negative full-
    Kelly value means "this side has no edge", clipped to 0 rather than
    interpreted as "bet the other way" (that is a separate, deliberate
    decision for the caller, not implied by the sizing formula).
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    full = p - (1 - p) / payoff_ratio
    return fraction * np.maximum(full, 0.0)


# ------------------------------------------------------------------ #
# 3. Volatility targeting                                             #
# ------------------------------------------------------------------ #

def vol_target_size(forecast_vol, target_vol, max_leverage=3.0):
    """
    size = target_vol / forecast_vol, capped at `max_leverage` and
    floored at 0. Pass a per-bar sigma_t (e.g. ml.volatility.fit_har_rv
    output) as `forecast_vol` and a constant risk budget as `target_vol`
    (same units — typically per-bar log-return std) so that a calm
    regime sizes UP and a violent one sizes DOWN, keeping expected risk
    per trade roughly constant across regimes.
    """
    forecast_vol = np.asarray(forecast_vol, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        size = np.where(forecast_vol > 0, target_vol / forecast_vol, 0.0)
    return np.clip(np.nan_to_num(size, nan=0.0), 0.0, max_leverage)


# ------------------------------------------------------------------ #
# 4. Cost model                                                       #
# ------------------------------------------------------------------ #

def cost_model(size, fee_bps=5.0, half_spread_bps=1.0, impact_bps_per_unit_size=3.0,
               sqrt_impact=True):
    """
    Round-trip cost, in bps, as a function of position size (size=1.0 ==
    "full", whatever unit the caller's `max_leverage` defines).

      fee_bps               taker/maker fee, paid TWICE (round trip)
      half_spread_bps        crossing the spread, paid TWICE
      impact_bps_per_unit    market impact — cost that GROWS with size,
                             because a bigger order moves the price
                             against itself. Understating this term is
                             the single most common way a backtest
                             manufactures an edge that isn't really
                             there (PREDICTION_SPEC.md invariant #6).

    `sqrt_impact=True` uses a square-root impact model (impact ~
    sqrt(size)), the standard microstructure assumption for how cost
    scales with order size relative to available liquidity; set False
    for a simpler linear model if you have reason to believe impact is
    linear in your instrument's depth.
    """
    size = np.clip(np.asarray(size, dtype=float), 0.0, None)
    impact = impact_bps_per_unit_size * (np.sqrt(size) if sqrt_impact else size)
    return 2 * fee_bps + 2 * half_spread_bps + impact


# ------------------------------------------------------------------ #
# Composition                                                         #
# ------------------------------------------------------------------ #

def combined_position_size(p, forecast_vol, target_vol, payoff_ratio=2.0,
                           kelly_fraction=0.375, max_leverage=3.0):
    """
    Combine probability-based fractional Kelly with volatility targeting
    into one final size multiplier: the smaller of the two, since either
    one alone can be wrong in a different direction (Kelly ignores
    regime, vol-targeting ignores edge quality) and taking the min is
    the conservative combination — never sized UP by one factor past
    what the other alone would allow.

    Returns a size in [0, max_leverage].
    """
    kelly_size = fractional_kelly(p, payoff_ratio=payoff_ratio, fraction=kelly_fraction)
    vol_size = vol_target_size(forecast_vol, target_vol, max_leverage=max_leverage)
    return np.minimum(kelly_size, vol_size)


def net_return_after_costs(gross_return, size, **cost_kwargs):
    """
    gross_return (fraction, e.g. 0.004 = 40bps) minus cost_model(size) in
    the same units. Convenience wrapper so callers don't hand-convert
    bps <-> fraction at every call site.
    """
    cost_bps = cost_model(size, **cost_kwargs)
    return np.asarray(gross_return, dtype=float) - cost_bps / 10000.0
