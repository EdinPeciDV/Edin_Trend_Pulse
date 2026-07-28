"""
ml/features.py
===================================================================
Feature engineering and labelling.

TWO RULES GOVERN FEATURES, and breaking either one silently destroys
the whole pipeline:

1. EVERY FEATURE MUST BE STATIONARY.
   Never feed raw price. A model given "BTC = 61,432" learns the price
   levels of its training window and collapses the moment price leaves
   that range. Every feature here is a ratio, a log return, a bounded
   oscillator, or a z-score — all scale-free, so a model trained on
   BTC at 30k still works at 90k.

2. EVERY FEATURE MUST USE ONLY PAST DATA.
   Feature at bar i may read bars 0..i and nothing beyond. This is
   enforced structurally: all rolling windows look backwards, and
   `assert_no_lookahead()` verifies it empirically by checking that
   recomputing features on a truncated series gives identical values.

LABELS are the opposite: they are ALLOWED to look forward, because a
label describes what happened after the bar, not what the model is
told about it. `assert_no_lookahead()` only ever checks build_features().

The JS twin of the feature code is shared/mlFeatures.js. ml/parity.py
verifies the two produce bit-comparable output. If you add a feature
here, add it there, or inference will silently read garbage.
===================================================================
"""

import numpy as np

# Feature order is part of the model contract. The exported model.json
# stores this list, and JS inference asserts against it. Never reorder
# without retraining. New features are appended at the end for the same
# reason — inserting in the middle silently reindexes every downstream
# coefficient.
FEATURE_NAMES = [
    "rsi_14_c",       # RSI(14) centred:      rsi/100 - 0.5
    "rsi_9_c",        # RSI(9) centred
    "rsi_slope",      # RSI(14) minus RSI(14) three bars ago, /100
    "log_p_sma20",    # log(price / SMA20)
    "log_p_sma50",    # log(price / SMA50)
    "bb_pctb_c",      # Bollinger %B centred: pctb - 0.5
    "bb_bandwidth",   # (upper-lower)/middle, already scale-free
    "log_p_vwap",     # log(price / VWAP)
    "ret_1",          # log return over 1 bar
    "ret_3",          # log return over 3 bars
    "ret_6",          # log return over 6 bars
    "ret_12",         # log return over 12 bars
    "vol_12",         # realised vol over 12 bars (sd of log returns)
    "vol_ratio",      # vol_12 / vol_48 — is volatility expanding?
    "atr_norm",       # ATR(14) / price
    "volume_z",       # volume vs 48-bar mean, z-scored (0 for spot FX)
    "macd_hist_norm", # MACD histogram / price
    "hour_sin",       # cyclical hour-of-day
    "hour_cos",
    # --- Phase 6: regime conditioning ---
    "adx_norm",           # ADX(14) / 100 — trend strength, always computed
    # --- Phase 2.3: cross-sectional (pooled crypto training only) ---
    "rel_strength_basket",  # ret_1 minus the sibling-basket's mean ret_1
    "corr_to_btc",           # 60-bar rolling corr of ret_1 to BTC's ret_1
    # --- Phase 2.1: crypto derivatives context (funding/OI/positioning) ---
    "funding_z",       # perp funding rate, z-scored
    "oi_roc_z",         # open-interest rate-of-change, z-scored
    "top_trader_z",     # top-trader long/short ratio, z-scored
]

N_FEATURES = len(FEATURE_NAMES)

# Features that default to a neutral 0 when the caller does not supply
# `context` (single-symbol training/inference, or an asset class the
# feature does not apply to — e.g. derivatives context for FX). Mirrors
# the existing volume_z-is-0-for-FX convention. Kept here so callers
# (train.py, parity.py) can reason about which features are context-gated
# without re-deriving it.
CONTEXT_GATED_FEATURES = {
    "rel_strength_basket", "corr_to_btc", "funding_z", "oi_roc_z", "top_trader_z",
}


# ------------------------------------------------------------------ #
# Primitive indicators (numpy, all backward-looking)                  #
# ------------------------------------------------------------------ #

def _windows(x, w):
    """Stride trick: a (n-w+1, w) view of every backward window."""
    return np.lib.stride_tricks.sliding_window_view(x, w)


def _rolling_mean(x, w):
    """
    Backward rolling mean. out[i] uses x[i-w+1..i]. NaN before warmup.

    Uses an explicit windowed mean rather than a cumsum difference.
    cumsum accumulates rounding error that differs from the JS
    implementation's arithmetic, which broke feature parity at the 1e-9
    level. Exactness across the two languages matters more here than
    the constant factor.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    out[w - 1:] = _windows(x, w).mean(axis=1)
    return out


def _rolling_std(x, w):
    """
    Backward rolling population std, two-pass.

    The sum-of-squares shortcut (E[x^2] - E[x]^2) loses precision badly
    when the mean dwarfs the spread, which is the normal case for price
    series. numpy's .std() is two-pass and matches the JS twin exactly.
    """
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) < w:
        return out
    out[w - 1:] = _windows(x, w).std(axis=1)
    return out


def _rolling_corr(x, y, w):
    """Backward rolling Pearson correlation over window w, two-pass."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    xw = _windows(x, w)
    yw = _windows(y, w)
    mx = xw.mean(axis=1, keepdims=True)
    my = yw.mean(axis=1, keepdims=True)
    dx = xw - mx
    dy = yw - my
    cov = (dx * dy).mean(axis=1)
    sx = np.sqrt((dx * dx).mean(axis=1))
    sy = np.sqrt((dy * dy).mean(axis=1))
    denom = sx * sy
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    out[w - 1:] = corr
    return out


def _rolling_zscore(x, w, clip=5.0):
    """
    Backward rolling z-score, NaN/undefined collapsed to 0 (neutral) and
    clipped to +/-`clip`. Used for every "raw external series -> feature"
    conversion (derivatives context), so a flat or missing series reads
    as "nothing unusual" rather than propagating NaN into the model.
    """
    x = np.asarray(x, dtype=float)
    mean = _rolling_mean(x, w)
    sd = _rolling_std(x, w)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(sd > 0, (x - mean) / sd, 0.0)
    z = np.clip(np.nan_to_num(z, nan=0.0), -clip, clip)
    return z


def wilder_rsi(closes, period=14):
    """
    RSI with Wilder smoothing — the same definition as
    shared/indicators.js. Returns a full series, NaN before warmup.
    """
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out

    delta = np.diff(closes)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = gain[:period].mean()
    avg_loss = loss[:period].mean()

    def rsi_of(g, l):
        if l == 0:
            return 100.0
        if g == 0:
            return 0.0
        return 100.0 - 100.0 / (1.0 + g / l)

    out[period] = rsi_of(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gain[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i - 1]) / period
        out[i] = rsi_of(avg_gain, avg_loss)
    return out


def _ema(x, span):
    """Standard EMA with alpha = 2/(span+1), seeded on the first value."""
    alpha = 2.0 / (span + 1.0)
    out = np.full(len(x), np.nan)
    if len(x) == 0:
        return out
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def atr(high, low, close, period=14):
    """Average True Range, Wilder-smoothed."""
    n = len(close)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    a = tr[1:period + 1].mean()
    out[period] = a
    for i in range(period + 1, n):
        a = (a * (period - 1) + tr[i]) / period
        out[i] = a
    return out


def adx(high, low, close, period=14):
    """
    Average Directional Index, Wilder-smoothed throughout — deliberately
    reusing the same "seed on the mean of the first `period` values, then
    recur as (prev*(period-1)+new)/period" pattern as wilder_rsi()/atr()
    above, rather than Wilder's classic sum-based DM smoothing. The ratio
    of two averages equals the ratio of the underlying sums, so this is
    numerically equivalent to the textbook formulation while keeping one
    smoothing idiom (and one parity risk) across the whole file.

    Trend-strength regime signal: near 0 = ranging/choppy, near 100 =
    strongly trending. A model that ignores directional features when
    ADX is low and leans on them when it's high is regime-conditioning
    itself; feeding ADX in as a feature lets a single linear model
    approximate that instead of training separate per-regime models.
    """
    n = len(close)
    out = np.full(n, np.nan)
    if n < 2 * period + 1:
        return out

    prev_close = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])

    up_move = np.empty(n)
    down_move = np.empty(n)
    up_move[0] = 0.0
    down_move[0] = 0.0
    up_move[1:] = high[1:] - high[:-1]
    down_move[1:] = low[:-1] - low[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0

    tr_avg = np.zeros(n)
    pdm_avg = np.zeros(n)
    mdm_avg = np.zeros(n)
    tr_avg[period] = tr[1:period + 1].mean()
    pdm_avg[period] = plus_dm[1:period + 1].mean()
    mdm_avg[period] = minus_dm[1:period + 1].mean()
    for i in range(period + 1, n):
        tr_avg[i] = (tr_avg[i - 1] * (period - 1) + tr[i]) / period
        pdm_avg[i] = (pdm_avg[i - 1] * (period - 1) + plus_dm[i]) / period
        mdm_avg[i] = (mdm_avg[i - 1] * (period - 1) + minus_dm[i]) / period

    dx = np.full(n, np.nan)
    for i in range(period, n):
        if tr_avg[i] > 0:
            plus_di = 100.0 * pdm_avg[i] / tr_avg[i]
            minus_di = 100.0 * mdm_avg[i] / tr_avg[i]
            s = plus_di + minus_di
            dx[i] = 100.0 * abs(plus_di - minus_di) / s if s > 0 else 0.0

    start = period * 2 - 1
    if start >= n:
        return out
    seed_window = dx[period:period * 2]
    if not np.isfinite(seed_window).all():
        return out
    out[start] = seed_window.mean()
    for i in range(start + 1, n):
        dxi = dx[i] if np.isfinite(dx[i]) else 0.0
        out[i] = (out[i - 1] * (period - 1) + dxi) / period
    return out


def rolling_vwap(high, low, close, volume, window=48):
    """
    Backward rolling VWAP over `window` bars.

    Spot FX reports zero volume, so when the window's volume sums to
    zero this falls back to an unweighted mean of typical prices —
    matching the fallback in shared/indicators.js. Any volume-derived
    feature is therefore meaningless for FX; `volume_z` goes to 0.
    """
    typical = (high + low + close) / 3.0
    pv = typical * volume
    pv_sum = _rolling_mean(pv, window) * window
    v_sum = _rolling_mean(volume, window) * window
    tp_mean = _rolling_mean(typical, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        weighted = pv_sum / v_sum
    return np.where((v_sum > 0) & np.isfinite(weighted), weighted, tp_mean)


# ------------------------------------------------------------------ #
# Feature matrix                                                      #
# ------------------------------------------------------------------ #

def build_features(candles, context=None):
    """
    Build the (n_bars, N_FEATURES) design matrix.

    `candles` is a dict of equal-length numpy arrays:
        open_time (ms), open, high, low, close, volume

    `context` is optional and enables the context-gated features listed
    in CONTEXT_GATED_FEATURES (cross-sectional + derivatives). When
    provided, every array in it must be the same length as `candles` and
    aligned bar-for-bar (the caller's job — see ml/data.py for training
    and shared/marketSources.js for live inference). Recognised keys,
    all optional and independently gated:

        basket_ret        mean log return of the OTHER pooled symbols
        btc_ret            BTC's own log return (omit when self IS BTC)
        funding_rate       raw perp funding rate, forward-filled to bars
        open_interest      raw open interest, forward-filled to bars
        top_trader_ratio   raw top-trader long/short ratio

    Any key omitted degrades its feature(s) to a neutral 0, exactly like
    volume_z already does for forex. This keeps build_features() callable
    identically for single-symbol, non-crypto, or context-free callers —
    the shape and column order of the design matrix never changes.

    Returns (X, valid_mask). Rows where any feature is NaN — the warmup
    region — are marked False in valid_mask and must be dropped by the
    caller rather than imputed. Imputing warmup rows invents data.
    """
    close = np.asarray(candles["close"], dtype=float)
    high = np.asarray(candles["high"], dtype=float)
    low = np.asarray(candles["low"], dtype=float)
    volume = np.asarray(candles["volume"], dtype=float)
    open_time = np.asarray(candles["open_time"], dtype=float)
    n = len(close)

    if context:
        for key, arr in context.items():
            if arr is not None and len(arr) != n:
                raise ValueError(
                    f"context['{key}'] has length {len(arr)}, candles have {n} — "
                    "context arrays must be pre-aligned bar-for-bar."
                )

    # Log price, the basis of every return feature.
    logp = np.log(close)

    def lagged_return(k):
        r = np.full(n, np.nan)
        if n > k:
            r[k:] = logp[k:] - logp[:-k]
        return r

    ret_1 = lagged_return(1)
    ret_3 = lagged_return(3)
    ret_6 = lagged_return(6)
    ret_12 = lagged_return(12)
    ret_1_filled = np.nan_to_num(ret_1, nan=0.0)

    # Realised volatility over two horizons.
    vol_12 = _rolling_std(ret_1_filled, 12)
    vol_48 = _rolling_std(ret_1_filled, 48)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_ratio = np.where(vol_48 > 0, vol_12 / vol_48, 1.0)

    # Moving averages.
    sma20 = _rolling_mean(close, 20)
    sma50 = _rolling_mean(close, 50)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p_sma20 = np.log(close / sma20)
        log_p_sma50 = np.log(close / sma50)

    # Bollinger.
    bb_mid = sma20
    bb_sd = _rolling_std(close, 20)
    bb_up = bb_mid + 2 * bb_sd
    bb_lo = bb_mid - 2 * bb_sd
    rng = bb_up - bb_lo
    with np.errstate(divide="ignore", invalid="ignore"):
        bb_pctb = np.where(rng > 0, (close - bb_lo) / rng, 0.5)
        bb_bandwidth = np.where(bb_mid > 0, rng / bb_mid, 0.0)

    # RSI.
    rsi14 = wilder_rsi(close, 14)
    rsi9 = wilder_rsi(close, 9)
    rsi_slope = np.full(n, np.nan)
    if n > 3:
        rsi_slope[3:] = (rsi14[3:] - rsi14[:-3]) / 100.0

    # VWAP.
    vwap = rolling_vwap(high, low, close, volume, 48)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p_vwap = np.log(close / vwap)

    # ATR.
    atr14 = atr(high, low, close, 14)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_norm = atr14 / close

    # ADX — regime / trend-strength (Phase 6), no context needed.
    adx14 = adx(high, low, close, 14)
    adx_norm = adx14 / 100.0

    # Volume z-score. Zero everywhere for FX (all volumes are 0).
    vol_mean = _rolling_mean(volume, 48)
    vol_sd = _rolling_std(volume, 48)
    with np.errstate(divide="ignore", invalid="ignore"):
        volume_z = np.where(vol_sd > 0, (volume - vol_mean) / vol_sd, 0.0)
    volume_z = np.clip(np.nan_to_num(volume_z, nan=0.0), -5, 5)

    # MACD histogram, normalised by price.
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    macd = ema12 - ema26
    signal = _ema(np.nan_to_num(macd, nan=0.0), 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        macd_hist_norm = (macd - signal) / close

    # Hour of day, cyclically encoded. FX has real session effects
    # (London/NY overlap); crypto is 24/7 so the model should learn to
    # ignore these, and if it leans on them hard that is a red flag.
    hours = (open_time / 3600000.0) % 24.0
    hour_sin = np.sin(2 * np.pi * hours / 24.0)
    hour_cos = np.cos(2 * np.pi * hours / 24.0)

    # --- Cross-sectional (Phase 2.3) ---
    basket_ret = context.get("basket_ret") if context else None
    if basket_ret is not None:
        rel_strength_basket = ret_1_filled - np.nan_to_num(np.asarray(basket_ret, dtype=float), nan=0.0)
    else:
        rel_strength_basket = np.zeros(n)

    btc_ret = context.get("btc_ret") if context else None
    if btc_ret is not None:
        btc_ret_filled = np.nan_to_num(np.asarray(btc_ret, dtype=float), nan=0.0)
        corr_to_btc = np.nan_to_num(_rolling_corr(ret_1_filled, btc_ret_filled, 60), nan=0.0)
    else:
        corr_to_btc = np.zeros(n)

    # --- Crypto derivatives context (Phase 2.1) ---
    # Windows are chosen in bar-units (5-minute bars): funding prints
    # every ~8h (96 bars), so a 288-bar (~24h / 3-cycle) window is the
    # shortest that reliably spans more than one distinct value. OI and
    # top-trader ratio update more often, so the standard 48-bar (4h)
    # stationarity window used elsewhere in this file is sufficient.
    funding_rate = context.get("funding_rate") if context else None
    funding_z = _rolling_zscore(funding_rate, 288) if funding_rate is not None else np.zeros(n)

    open_interest = context.get("open_interest") if context else None
    if open_interest is not None:
        oi = np.asarray(open_interest, dtype=float)
        oi_roc = np.full(n, np.nan)
        if n > 12:
            with np.errstate(divide="ignore", invalid="ignore"):
                oi_roc[12:] = np.log(oi[12:] / oi[:-12])
        oi_roc_z = _rolling_zscore(np.nan_to_num(oi_roc, nan=0.0), 48)
    else:
        oi_roc_z = np.zeros(n)

    top_trader_ratio = context.get("top_trader_ratio") if context else None
    top_trader_z = (
        _rolling_zscore(top_trader_ratio, 48) if top_trader_ratio is not None else np.zeros(n)
    )

    cols = {
        "rsi_14_c": rsi14 / 100.0 - 0.5,
        "rsi_9_c": rsi9 / 100.0 - 0.5,
        "rsi_slope": rsi_slope,
        "log_p_sma20": log_p_sma20,
        "log_p_sma50": log_p_sma50,
        "bb_pctb_c": bb_pctb - 0.5,
        "bb_bandwidth": bb_bandwidth,
        "log_p_vwap": log_p_vwap,
        "ret_1": ret_1,
        "ret_3": ret_3,
        "ret_6": ret_6,
        "ret_12": ret_12,
        "vol_12": vol_12,
        "vol_ratio": vol_ratio,
        "atr_norm": atr_norm,
        "volume_z": volume_z,
        "macd_hist_norm": macd_hist_norm,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "adx_norm": adx_norm,
        "rel_strength_basket": rel_strength_basket,
        "corr_to_btc": corr_to_btc,
        "funding_z": funding_z,
        "oi_roc_z": oi_roc_z,
        "top_trader_z": top_trader_z,
    }

    X = np.column_stack([cols[name] for name in FEATURE_NAMES])
    valid = np.isfinite(X).all(axis=1)
    return X, valid


# ------------------------------------------------------------------ #
# Labelling — triple-barrier (Phase 0.1)                              #
# ------------------------------------------------------------------ #

def _ewma_vol(ret_1_filled, span=20):
    """
    EWMA volatility of 1-bar log returns, used to scale triple-barrier
    distances so a barrier means the same thing in calm and violent
    markets. Reuses the file's EMA primitive on squared returns, seeded
    the same way as every other recursive series here.
    """
    var = _ema(ret_1_filled ** 2, span)
    return np.sqrt(np.maximum(var, 0.0))


def build_labels(close, k1=2.0, k2=1.0, vertical=24, ewma_span=20, sigma_override=None):
    """
    Triple-barrier labelling (López de Prado, AFML ch.3).

    For each bar i, set three barriers and label by whichever is
    touched first, scanning forward from i:

        upper (label +1)   logp[i] + k1 * sigma_t
        lower (label -1)   logp[i] - k2 * sigma_t
        vertical (label 0) i + `vertical` bars, if neither touched first

    sigma_t is an EWMA of recent 1-bar log returns as of bar i by
    default — using only information available at i, so barrier WIDTH
    never leaks the future even though the label ITSELF is necessarily
    forward-looking (that is what a label is). assert_no_lookahead()
    therefore only ever checks build_features(), never this function.

    `sigma_override` (Phase 3 integration point) — pass a pre-computed
    sigma_t series (length n) to use instead of the internal EWMA, e.g.
    ml.volatility.fit_har_rv(close)['sigma']. Using a HAR-RV/GARCH
    forecast here does not reintroduce look-ahead: sigma_t only sizes
    the barrier for label i, which is itself already allowed to see the
    future (that's what a label is) — this is a labelling PARAMETER,
    never a model-facing feature. Falls back to NaN wherever the
    override is NaN, same as EWMA warmup.

    The k1=2.0 / k2=1.0 default asymmetry encodes a 2:1 reward/risk: the
    label answers "would this trade have worked" rather than "did price
    tick up".

    Returns four arrays, each length n:
      label            +1 / 0 / -1
      touch_idx         bar index the barrier was touched (or the
                        vertical barrier's bar, on timeout)
      realised_return   log(close[touch] / close[i])
      labelable         False for bars too close to the end of the
                        series to know the answer (not enough future
                        data to say whether the vertical barrier would
                        have been reached) OR too early for sigma_t to
                        be defined yet (EWMA warmup). Callers must drop
                        these rows rather than treat label=0 as a real
                        timeout.
    """
    n = len(close)
    logp = np.log(np.asarray(close, dtype=float))

    if sigma_override is not None:
        sigma = np.asarray(sigma_override, dtype=float)
        if len(sigma) != n:
            raise ValueError(f"sigma_override has length {len(sigma)}, close has {n}")
    else:
        ret_1 = np.full(n, np.nan)
        if n > 1:
            ret_1[1:] = logp[1:] - logp[:-1]
        ret_1_filled = np.nan_to_num(ret_1, nan=0.0)
        sigma = _ewma_vol(ret_1_filled, span=ewma_span)

    label = np.zeros(n, dtype=int)
    touch_idx = np.arange(n, dtype=int)
    realised_return = np.full(n, np.nan)
    labelable = np.zeros(n, dtype=bool)

    for i in range(n):
        s = sigma[i]
        if not np.isfinite(s) or s <= 0:
            continue  # EWMA still in warmup

        end = min(i + vertical, n - 1)
        if end <= i:
            continue  # no future bars at all

        upper = logp[i] + k1 * s
        lower = logp[i] - k2 * s
        touched = False

        for j in range(i + 1, end + 1):
            if logp[j] >= upper:
                label[i] = 1
                touch_idx[i] = j
                realised_return[i] = logp[j] - logp[i]
                touched = True
                break
            if logp[j] <= lower:
                label[i] = -1
                touch_idx[i] = j
                realised_return[i] = logp[j] - logp[i]
                touched = True
                break

        if touched:
            labelable[i] = True
        elif end == i + vertical:
            # Genuinely exhausted the full vertical window without a
            # touch — a real timeout, not a truncated one.
            label[i] = 0
            touch_idx[i] = end
            realised_return[i] = logp[end] - logp[i]
            labelable[i] = True
        # else: ran out of series before reaching the vertical barrier
        # and never touched — unknowable, leave labelable[i] = False.

    return label, touch_idx, realised_return, labelable


# ------------------------------------------------------------------ #
# Look-ahead guard                                                    #
# ------------------------------------------------------------------ #

def assert_no_lookahead(candles, context=None, n_checks=6, tolerance=1e-9):
    """
    Empirically verify that no feature reads the future.

    Method: compute features on the full series, then recompute on the
    series truncated at bar i (candles AND context truncated in lockstep,
    since a context array that still reached past i would itself be a
    look-ahead leak). If any feature at bar i used data from beyond i,
    the two values will differ.

    This catches the single most common and most costly bug in
    financial ML — a centred rolling window, a full-series normalisation,
    a stray .shift(-1) — any of which produces a model that looks
    excellent in testing and is worthless live.

    NOTE ON EMA/ATR/ADX: these are recursive and seeded from the first
    bar, so a truncated series has a different warmup and values
    legitimately differ slightly near the start. The check therefore
    only inspects bars well past warmup, where the recursion has
    converged.
    """
    X_full, _ = build_features(candles, context=context)
    n = len(candles["close"])
    warmup = 120  # past all warmups; EMA/ADX recursion has converged
    if n < warmup + 20:
        raise ValueError(f"need >{warmup + 20} bars to check look-ahead, got {n}")

    idxs = np.linspace(warmup, n - 2, n_checks).astype(int)
    failures = []

    for i in idxs:
        truncated = {k: v[: i + 1] for k, v in candles.items()}
        trunc_context = (
            {k: (v[: i + 1] if v is not None else None) for k, v in context.items()}
            if context else None
        )
        X_trunc, _ = build_features(truncated, context=trunc_context)
        diff = np.abs(X_full[i] - X_trunc[i])
        bad = np.where(diff > tolerance)[0]
        for j in bad:
            failures.append((int(i), FEATURE_NAMES[j], float(diff[j])))

    if failures:
        lines = [f"    bar {i}: {name} differs by {d:.3e}" for i, name, d in failures[:12]]
        raise AssertionError(
            "LOOK-AHEAD DETECTED — these features read future data:\n"
            + "\n".join(lines)
            + "\n  Any model trained on these features is invalid."
        )
    return True
