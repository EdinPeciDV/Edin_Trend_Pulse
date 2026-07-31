"""
ml/volatility.py
===================================================================
Predict volatility, not just direction (Phase 3 of PREDICTION_SPEC.md).

Direction is near-unpredictable on 5-minute bars (see ml/README.md).
Volatility is genuinely forecastable — it clusters, and that is one of
the most robust findings in finance. This module is deliberately
independent of features.py/train.py: it operates directly on a close
series and produces a sigma_t (volatility) series plus the machinery to
evaluate forecast quality, for three downstream uses:

  1. Barrier placement   — ml.features.build_labels(..., sigma_override=)
                           can take this module's forecast instead of
                           its own internal EWMA.
  2. Position sizing      — ml.sizing.vol_target_size() takes a forecast
                           directly.
  3. Entry timing         — vol_expansion_signal() below.

Two models, in increasing order of complexity:
  HAR-RV     — regress realised vol on its own short/medium/long-window
               averages. No external dependency, hard to beat.
  GARCH(1,1) — via the optional `arch` package. Not installed by
               default (ml/requirements.txt stays numpy + scikit-learn);
               garch_forecast() raises a clear, actionable error if it's
               missing rather than degrading silently.

Evaluated with QLIKE, not MSE — MSE is dominated by the handful of
largest observations in any volatility series, which is precisely the
tail behaviour a vol forecast is judged on getting right.
===================================================================
"""

import numpy as np


# ------------------------------------------------------------------ #
# Realised variance primitives                                        #
# ------------------------------------------------------------------ #

def log_returns(close):
    close = np.asarray(close, dtype=float)
    r = np.full(len(close), np.nan)
    if len(close) > 1:
        r[1:] = np.log(close[1:]) - np.log(close[:-1])
    return r


def _rolling_sum(x, w):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, w)
    out[w - 1:] = windows.sum(axis=1)
    return out


def _rolling_mean(x, w):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.full(n, np.nan)
    if n < w:
        return out
    windows = np.lib.stride_tricks.sliding_window_view(x, w)
    out[w - 1:] = windows.mean(axis=1)
    return out


def realised_variance(close, window):
    """
    Backward-looking realised variance: sum of squared 1-bar log returns
    over the trailing `window` bars, ending at and including bar t.
    """
    r = np.nan_to_num(log_returns(close), nan=0.0)
    return _rolling_sum(r ** 2, window)


# ------------------------------------------------------------------ #
# HAR-RV (Corsi, 2009)                                                #
# ------------------------------------------------------------------ #
#
# Window choice: the textbook HAR-RV uses calendar daily/weekly/monthly
# realised variance. This pipeline trades 5-minute bars and typically
# has 5,000-15,000 bars of history (17-52 days, ml/README.md) — not
# enough for a literal 22-day "monthly" window in most runs. The
# defaults below keep the same SHORT << MEDIUM << LONG structure at a
# scale the data actually supports: 4h / 1 day / 5 days. Pass explicit
# windows if you have more history and want the classic cadence.

def har_rv_design(close, short=48, medium=288, long=1440):
    """
    Build (X, y) for the HAR-RV regression.

    X columns: [rv_short(t), rv_medium(t), rv_long(t)] — all backward-
    looking, using only bars up to and including t.
    y[t]: realised variance over the NEXT `short`-bar window (t, t+short]
    — necessarily forward-looking, because it is the regression TARGET,
    not a feature. (This module has no JS twin and is never used to
    build model-facing features directly; assert_no_lookahead() in
    features.py is unaffected by anything here.)

    Returns (X, y, valid_mask) — valid_mask excludes both the long-window
    warmup and the last `short` rows, whose forward target does not exist.
    """
    close = np.asarray(close, dtype=float)
    n = len(close)
    rv_short = realised_variance(close, short)
    rv_medium = _rolling_mean(rv_short, medium // short * short if medium >= short else medium)
    # rv_medium/rv_long are rolling means of the SHORT realised-variance
    # series (an average of `medium/short` daily-scale RVs), matching
    # the classic HAR-RV construction rather than recomputing variance
    # from raw returns over the longer window.
    rv_medium = _rolling_mean(rv_short, max(2, medium // short))
    rv_long = _rolling_mean(rv_short, max(2, long // short))

    y = np.full(n, np.nan)
    if n > short:
        y[:-short] = rv_short[short:]

    X = np.column_stack([rv_short, rv_medium, rv_long])
    valid = np.isfinite(X).all(axis=1) & np.isfinite(y)
    return X, y, valid


def fit_har_rv(close, short=48, medium=288, long=1440, holdout_frac=0.2):
    """
    Fit HAR-RV by OLS (numpy lstsq — no sklearn dependency needed for a
    3-feature linear regression) on the first (1 - holdout_frac) of
    valid rows, chronologically; evaluate QLIKE on the remainder. This
    keeps the same "never evaluate on data used to fit" discipline as
    the rest of the pipeline, at a scale appropriate for a diagnostic
    regression rather than the full purged walk-forward harness.

    Returns a dict: {coef, intercept, qlike, sigma, windows}. `sigma` is
    scaled to a PER-BAR standard deviation (fitted/forecast realised
    variance divided by `short` before the square root), matching the
    units build_labels()'s own internal EWMA sigma_t uses — NOT the raw
    sqrt(fitted realised variance), which is a `short`-bar CUMULATIVE
    volatility and would be ~sqrt(short)x too large to use directly as
    sigma_override (barriers would come out that many times too wide,
    and almost every label would time out instead of touching a barrier).
    """
    X, y, valid = har_rv_design(close, short, medium, long)
    idx = np.where(valid)[0]
    if len(idx) < 50:
        raise ValueError(
            f"only {len(idx)} valid rows for HAR-RV with windows "
            f"({short},{medium},{long}) — need more history or smaller windows."
        )

    split = int(len(idx) * (1 - holdout_frac))
    train_idx, test_idx = idx[:split], idx[split:]

    A = np.column_stack([np.ones(len(train_idx)), X[train_idx]])
    coef, *_ = np.linalg.lstsq(A, y[train_idx], rcond=None)
    intercept, betas = coef[0], coef[1:]

    def predict(rows):
        return intercept + X[rows] @ betas

    pred_test = np.maximum(predict(test_idx), 1e-12)
    qlike = qlike_loss(y[test_idx], pred_test) if len(test_idx) else float("nan")

    fitted_all = intercept + X @ betas
    per_bar_variance = np.nan_to_num(fitted_all, nan=0.0) / short
    sigma = np.sqrt(np.maximum(per_bar_variance, 1e-14))
    sigma[~valid] = np.nan

    return {
        "intercept": float(intercept),
        "coef": {"short": float(betas[0]), "medium": float(betas[1]), "long": float(betas[2])},
        "qlike": qlike,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "sigma": sigma,
        "windows": {"short": short, "medium": medium, "long": long},
    }


# ------------------------------------------------------------------ #
# GARCH(1,1) — optional, via the `arch` package                       #
# ------------------------------------------------------------------ #

def garch_available():
    try:
        import arch  # noqa: F401
        return True
    except ImportError:
        return False


def fit_garch(close, holdout_frac=0.2):
    """
    GARCH(1,1) via the `arch` package. Not a hard dependency of this
    pipeline (ml/requirements.txt stays numpy + scikit-learn) — install
    with `pip install arch` to use this; HAR-RV works without it.

    Evaluated the same way as fit_har_rv(): fit on the chronologically
    first (1 - holdout_frac) of the series, QLIKE on the remainder,
    using the model's own one-step-ahead conditional-variance path
    (`res.conditional_volatility`) rather than a rolling re-fit — a
    full rolling re-fit is the more rigorous evaluation but is minutes,
    not seconds, per symbol; this is the fast, honest-enough version.
    """
    if not garch_available():
        raise RuntimeError(
            "GARCH(1,1) needs the optional 'arch' package: pip install arch. "
            "HAR-RV (fit_har_rv) needs no extra dependency and is the default."
        )
    from arch import arch_model

    close = np.asarray(close, dtype=float)
    r = np.nan_to_num(log_returns(close), nan=0.0) * 100.0  # arch prefers %-scale returns
    n = len(r)
    split = int(n * (1 - holdout_frac))

    am = arch_model(r[:split], vol="Garch", p=1, q=1, dist="normal", mean="Zero")
    res = am.fit(disp="off")

    # One-step-ahead forecasts walked forward across the holdout, refit
    # is skipped for speed (see docstring) — parameters are held fixed
    # and only the conditional variance recursion is rolled forward.
    forecasts = res.forecast(horizon=1, start=split - 1, reindex=False)
    pred_var_pct2 = forecasts.variance.values[:, 0]  # in (%-return)^2 units
    pred_var = pred_var_pct2 / (100.0 ** 2)

    actual_var = r[split:] ** 2 / (100.0 ** 2)
    m = min(len(pred_var) - 1, len(actual_var))  # forecast[i] predicts r[split+i]
    qlike = qlike_loss(actual_var[:m], np.maximum(pred_var[1:1 + m], 1e-14))

    sigma = np.full(n, np.nan)
    sigma[:split] = res.conditional_volatility / 100.0
    if m > 0:
        sigma[split:split + m] = np.sqrt(np.maximum(pred_var[1:1 + m], 1e-14))

    return {"qlike": qlike, "n_train": split, "n_test": m, "sigma": sigma, "result": res}


# ------------------------------------------------------------------ #
# QLIKE loss                                                           #
# ------------------------------------------------------------------ #

def qlike_loss(y_true_var, y_pred_var):
    """
    QLIKE(y, yhat) = mean( y/yhat - log(y/yhat) - 1 ), minimised at
    y == yhat everywhere. Unlike MSE, QLIKE penalises under- and
    over-prediction asymmetrically in a way that matches how a
    volatility forecast is actually used (risk sizing, barrier width) —
    and is not dominated by the handful of largest realised-variance
    spikes the way squared error is.
    """
    y = np.asarray(y_true_var, dtype=float)
    yhat = np.asarray(y_pred_var, dtype=float)
    mask = np.isfinite(y) & np.isfinite(yhat) & (y > 0) & (yhat > 0)
    if mask.sum() == 0:
        return float("nan")
    ratio = y[mask] / yhat[mask]
    return float(np.mean(ratio - np.log(ratio) - 1))


# ------------------------------------------------------------------ #
# Downstream uses                                                     #
# ------------------------------------------------------------------ #

def vol_expansion_signal(sigma_series, lookback=48, low_pctile=30, recent_slope_bars=6):
    """
    Phase 3's "entry timing" use: flag bars where recent volatility is
    LOW relative to its own trailing history AND trending upward — the
    setup the spec calls "the best risk/reward": calm enough that a
    barrier/position sized off current vol isn't already stopped out by
    noise, with an expansion under way that a directional bet benefits
    from.

    "Low" — sigma[i] is at or below the `low_pctile` percentile of the
    trailing `lookback` window.
    "Expanding" — the mean of the last `recent_slope_bars` values is
    higher than the mean of the `recent_slope_bars` before that, i.e.
    the forecast itself is trending up, not just currently small.

    Returns a boolean array, True where both conditions hold.
    """
    sigma = np.asarray(sigma_series, dtype=float)
    n = len(sigma)
    out = np.zeros(n, dtype=bool)
    k = recent_slope_bars
    for i in range(lookback, n):
        if not np.isfinite(sigma[i]) or i < 2 * k:
            continue
        window = sigma[i - lookback:i]
        window = window[np.isfinite(window)]
        if len(window) < lookback // 2:
            continue
        low_thresh = np.percentile(window, low_pctile)
        is_low = sigma[i] <= low_thresh

        recent = sigma[i - k + 1:i + 1]
        prior = sigma[i - 2 * k + 1:i - k + 1]
        if not (np.isfinite(recent).all() and np.isfinite(prior).all()):
            continue
        is_expanding = recent.mean() > prior.mean()

        out[i] = bool(is_low and is_expanding)
    return out
