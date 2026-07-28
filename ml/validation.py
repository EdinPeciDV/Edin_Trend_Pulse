"""
ml/validation.py
===================================================================
Validation, baselines and metrics.

This is the most important file in the ML pipeline. A model is only as
credible as the procedure that evaluated it, and the standard sklearn
workflow — train_test_split, cross_val_score — is catastrophically
wrong for this problem in two independent ways:

  LEAK 1: RANDOM SHUFFLING.
    Shuffling puts future bars in the training set and past bars in the
    test set. The model learns from tomorrow to predict yesterday.
    This alone can turn a 50% coin flip into a reported 70%+.
    FIX: walk-forward splits. Train strictly before test, always.

  LEAK 2: OVERLAPPING LABELS.
    With horizon=12, the label at bar i covers bars i..i+12, and the
    label at bar i+1 covers i+1..i+13. Consecutive samples share 11 of
    12 bars of outcome. Adjacent train and test samples are therefore
    near-duplicates, so the model gets graded on data it effectively
    trained on. This is subtle, invisible in the code, and inflates
    scores by several points.
    FIX: purge `horizon` bars between train and test, then embargo a
    further gap. (López de Prado, "Advances in Financial ML", ch. 7.)

Every number this file reports is measured under both fixes.

THE HEADLINE METRIC IS `edge`, NOT `accuracy`.
    A model scoring 53% on a market that rose 53% of the time has
    learned nothing. edge = accuracy − majority-class base rate. Only
    edge above zero, sustained across folds, is evidence of anything.
===================================================================
"""

import math

import numpy as np


# ------------------------------------------------------------------ #
# Purged walk-forward splitter                                        #
# ------------------------------------------------------------------ #

def purged_walk_forward(n_samples, n_splits=5, horizon=12, embargo=6,
                        min_train=500, expanding=True):
    """
    Yield (train_idx, test_idx) with a purge+embargo gap between them.

    Layout of one fold (expanding=True):

        |<--------- train --------->|  gap  |<-- test -->|
        0                          t       t+g          t+g+len

    where gap = horizon + embargo. The purge (`horizon`) removes train
    samples whose forward-looking label would reach into the test
    window; the embargo adds margin for slow-decaying autocorrelation.

    expanding=True grows the training set each fold (more data, but
    mixes old regimes in). expanding=False uses a fixed-width rolling
    window (adapts to regime change, less data). Both are reported by
    train.py because they answer different questions.
    """
    gap = horizon + embargo
    usable = n_samples - gap
    if usable <= min_train + n_splits:
        raise ValueError(
            f"Not enough samples for {n_splits} folds: have {n_samples}, "
            f"need > {min_train + n_splits + gap}. Collect more history."
        )

    test_size = (usable - min_train) // n_splits
    if test_size < 20:
        raise ValueError(
            f"Test folds would only be {test_size} samples. Reduce n_splits "
            f"or collect more history."
        )

    for k in range(n_splits):
        train_end = min_train + k * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n_samples)
        if test_end - test_start < 20:
            break

        if expanding:
            train_idx = np.arange(0, train_end)
        else:
            train_idx = np.arange(max(0, train_end - min_train), train_end)

        yield train_idx, np.arange(test_start, test_end)


# ------------------------------------------------------------------ #
# Baselines                                                           #
# ------------------------------------------------------------------ #

def baseline_predictions(y_train, y_test, rng):
    """
    The bar any model must clear. If a model cannot beat these, it has
    found nothing and should not be deployed.

      always_up  : predict 1 every time. Exposes the drift in the data —
                   in a rising market this alone can score 55%+.
      majority   : predict whichever class dominated training.
      random     : coin flip, matched to the training class balance.
    """
    n = len(y_test)
    maj = int(round(y_train.mean())) if len(y_train) else 1
    p_up = y_train.mean() if len(y_train) else 0.5
    return {
        "always_up": np.ones(n, dtype=int),
        "majority": np.full(n, maj, dtype=int),
        "random": (rng.random(n) < p_up).astype(int),
    }


def heuristic_predictions(X, feature_names):
    """
    The existing rule-based signal, as a baseline.

    This is the comparison that actually decides whether the ML is worth
    shipping: does it beat the hand-written heuristic already running in
    shared/indicators.js? If not, the simpler system wins on every
    axis — it needs no training data, no retraining, and it can explain
    itself.

    Reconstructed from features rather than raw prices:
      log_p_sma50 > 0        <=> price above the 50-period SMA
      rsi_14_c in [-0.1, .1] <=> RSI between 40 and 60

    Returns 1 (up), 0 (down), or -1 (no opinion / NEUTRAL).
    """
    idx = {name: i for i, name in enumerate(feature_names)}
    above_sma = X[:, idx["log_p_sma50"]] > 0
    rsi = (X[:, idx["rsi_14_c"]] + 0.5) * 100.0
    bw = X[:, idx["bb_bandwidth"]]

    pred = np.full(len(X), -1, dtype=int)
    # spec-up: above SMA and RSI in the stable band
    pred[above_sma & (rsi >= 40) & (rsi <= 60)] = 1
    # spec-down: below SMA and overbought
    pred[(~above_sma) & (rsi > 65)] = 0
    # trend-down supplement: below SMA and weak
    pred[(~above_sma) & (rsi < 40)] = 0
    # sideways gate overrides everything: no opinion
    pred[bw < 0.004] = -1
    return pred


# ------------------------------------------------------------------ #
# Metrics                                                             #
# ------------------------------------------------------------------ #

def classification_metrics(y_true, y_pred, y_prob=None, fwd_ret=None,
                           cost_bps=0.0):
    """
    Metrics for one fold.

    Returns a dict containing, among others:

      accuracy    fraction correct
      base_rate   majority class share in y_true — the score to beat
      edge        accuracy - base_rate  <-- THE HEADLINE NUMBER
      auc         ranking quality, robust to class imbalance
      net_bps     realised return per trade after costs, in basis
                  points. This is the only metric denominated in money.
                  A positive edge with negative net_bps means the model
                  is right slightly more often but on smaller moves than
                  the ones it gets wrong — a losing strategy.
      brier       calibration error of the probabilities (lower better)
      top_decile_acc
                  accuracy on the 10% of predictions the model was most
                  confident about. If this is not clearly above overall
                  accuracy, the confidence score is meaningless and must
                  not be shown to users as though it ranks anything.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    out = {"n": int(n)}
    if n == 0:
        return out

    correct = (y_pred == y_true)
    out["accuracy"] = float(correct.mean())

    p_up = float(y_true.mean())
    out["up_rate"] = p_up
    out["base_rate"] = float(max(p_up, 1 - p_up))
    out["edge"] = out["accuracy"] - out["base_rate"]

    # Per-class recall, to catch a model that just predicts one class.
    for cls, name in ((1, "up"), (0, "down")):
        mask = y_true == cls
        out[f"recall_{name}"] = float(correct[mask].mean()) if mask.any() else float("nan")
    out["pred_up_rate"] = float((y_pred == 1).mean())

    # AUC via rank statistic (no sklearn dependency needed here).
    if y_prob is not None:
        y_prob = np.asarray(y_prob, dtype=float)
        pos, neg = y_prob[y_true == 1], y_prob[y_true == 0]
        if len(pos) and len(neg):
            ranks = np.argsort(np.argsort(y_prob)) + 1
            r_pos = ranks[y_true == 1].sum()
            out["auc"] = float(
                (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
            )
        else:
            out["auc"] = float("nan")

        out["brier"] = float(np.mean((y_prob - y_true) ** 2))

        # Does confidence actually rank anything?
        conf = np.abs(y_prob - 0.5)
        if n >= 20:
            k = max(1, n // 10)
            top = np.argsort(conf)[-k:]
            out["top_decile_acc"] = float(correct[top].mean())
            out["top_decile_n"] = int(k)

    # Money. Trade in the predicted direction, pay costs both ways.
    if fwd_ret is not None:
        fwd_ret = np.asarray(fwd_ret, dtype=float)
        direction = np.where(y_pred == 1, 1.0, -1.0)
        gross = direction * fwd_ret
        net = gross - (cost_bps / 10000.0)
        out["gross_bps"] = float(np.nanmean(gross) * 10000)
        out["net_bps"] = float(np.nanmean(net) * 10000)
        out["hit_rate_money"] = float(np.nanmean(net > 0))
        sd = np.nanstd(net)
        # Per-trade Sharpe proxy, not annualised.
        out["sharpe_per_trade"] = float(np.nanmean(net) / sd) if sd > 0 else float("nan")

    return out


def aggregate_folds(fold_metrics):
    """
    Mean and standard deviation across folds.

    The SD matters as much as the mean. An edge of +2% with an SD of 6%
    across five folds is noise; +2% with an SD of 0.5% is a signal worth
    investigating. `folds_positive` counts how many folds had edge > 0 —
    with 5 folds, anything under 4 is not a pattern.
    """
    if not fold_metrics:
        return {}
    keys = set()
    for m in fold_metrics:
        keys |= set(m.keys())

    agg = {}
    for k in sorted(keys):
        vals = [m[k] for m in fold_metrics if k in m and np.isfinite(m.get(k, np.nan))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

    edges = [m.get("edge", np.nan) for m in fold_metrics]
    edges = [e for e in edges if np.isfinite(e)]
    agg["folds"] = len(fold_metrics)
    agg["folds_positive_edge"] = int(sum(1 for e in edges if e > 0))
    return agg


# ------------------------------------------------------------------ #
# Verdict                                                             #
# ------------------------------------------------------------------ #

def verdict(agg, n_folds):
    """
    Turn the aggregate metrics into a plain-language recommendation.

    Deliberately hard to pass. The default answer is "no edge", because
    for this problem that is very nearly always the correct answer, and
    a pipeline that flatters its own model is worse than no pipeline.

    Returns (verdict_code, human_readable_explanation).
    """
    edge = agg.get("edge_mean", float("nan"))
    edge_sd = agg.get("edge_std", float("nan"))
    pos = agg.get("folds_positive_edge", 0)
    net = agg.get("net_bps_mean", float("nan"))
    auc = agg.get("auc_mean", float("nan"))

    if not np.isfinite(edge):
        return "UNKNOWN", "Not enough valid folds to judge."

    consistent = pos >= max(2, int(np.ceil(n_folds * 0.8)))
    significant = np.isfinite(edge_sd) and edge_sd > 0 and edge > 2 * edge_sd
    profitable = np.isfinite(net) and net > 0

    if edge <= 0:
        return "NO_EDGE", (
            f"Mean edge {edge*100:+.2f}pp — the model does not beat simply "
            f"predicting the majority class. This is the expected and normal "
            f"result for 5-minute direction prediction. Do not trade it."
        )
    if not consistent:
        return "NOISE", (
            f"Edge {edge*100:+.2f}pp but positive in only {pos}/{n_folds} folds "
            f"(SD {edge_sd*100:.2f}pp). That is fold-to-fold luck, not a pattern. "
            f"Do not trade it."
        )
    if not significant:
        return "WEAK", (
            f"Edge {edge*100:+.2f}pp, consistent across {pos}/{n_folds} folds, but "
            f"smaller than 2x its own variance (SD {edge_sd*100:.2f}pp). Suggestive "
            f"at best. Keep collecting data; do not size positions on this."
        )
    if not profitable:
        return "EDGE_NO_PROFIT", (
            f"Edge {edge*100:+.2f}pp is real and consistent, but net return after "
            f"costs is {net:+.2f} bps per trade — negative. The model is right more "
            f"often on smaller moves than the ones it gets wrong. Directionally "
            f"informative, not tradeable."
        )
    return "PROMISING", (
        f"Edge {edge*100:+.2f}pp (SD {edge_sd*100:.2f}pp), positive in {pos}/{n_folds} "
        f"folds, AUC {auc:.3f}, net {net:+.2f} bps/trade after costs. This cleared "
        f"every check in this harness. That still is not proof: validate on a fresh "
        f"out-of-sample period you have never touched, then paper-trade it before "
        f"risking money."
    )


# ------------------------------------------------------------------ #
# Phase 1 — calibration                                                #
# ------------------------------------------------------------------ #
#
# Calibration and discrimination are separate properties (see
# PREDICTION_SPEC.md 1.1). A model that always predicts 52% on a market
# that rises 52% of the time is perfectly calibrated and useless — it
# has calibration but no discrimination (resolution == 0). The Brier
# decomposition below is the one diagnostic that separates the two.

def brier_decomposition(y_true, y_prob, n_bins=10):
    """
    Murphy (1973) decomposition: brier = reliability - resolution + uncertainty.

      reliability   calibration error — how far mean-predicted is from
                    observed frequency within each bin. Want -> 0.
      resolution    genuine discriminative power — how far each bin's
                    observed frequency is from the overall base rate.
                    Want -> as large as possible. THIS is the number
                    that says whether the model found anything; a model
                    can have reliability == 0 and resolution == 0 at the
                    same time (always predicts the base rate) and that
                    is calibrated but worthless.
      uncertainty   base-rate variance, p(1-p). A property of the data,
                    not the model — the ceiling resolution is chasing.

    Bins are equal-width on [0, 1], the standard ECE convention (NOT
    equal-count), so an empty bin contributes 0 to every term.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    if n == 0:
        return {"reliability": float("nan"), "resolution": float("nan"),
                "uncertainty": float("nan"), "brier": float("nan")}

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    obar = y_true.mean()
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        pk = y_prob[mask].mean()
        ok = y_true[mask].mean()
        reliability += nk * (pk - ok) ** 2
        resolution += nk * (ok - obar) ** 2
    reliability /= n
    resolution /= n
    uncertainty = obar * (1 - obar)

    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        # Sanity identity: brier ≈ reliability - resolution + uncertainty.
        # Exact only in the equal-count-bin limit; equal-width bins used
        # here leave a small, expected residual.
    }


def reliability_table(y_true, y_prob, n_bins=10):
    """
    Predicted vs observed per bucket — the single most informative
    diagnostic in this file (PREDICTION_SPEC.md 1.3). Also what
    ModelPanel.jsx renders for the bucket the live prediction falls in:
    "of the last N times the model said 65-70%, it was right X%."

    Returns a list of dicts, one per non-empty bin, in ascending order:
      {lo, hi, n, mean_predicted, observed_freq}
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    if n == 0:
        return []
    bin_idx = np.clip(np.digitize(y_prob, edges[1:-1], right=True), 0, n_bins - 1)

    table = []
    for k in range(n_bins):
        mask = bin_idx == k
        nk = int(mask.sum())
        if nk == 0:
            continue
        table.append({
            "lo": float(edges[k]),
            "hi": float(edges[k + 1]),
            "n": nk,
            "mean_predicted": float(y_prob[mask].mean()),
            "observed_freq": float(y_true[mask].mean()),
        })
    return table


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    ECE = sum_k (n_k/N) * |mean_predicted_k - observed_freq_k|.
    Acceptance target in the spec: ECE < 0.05.
    """
    table = reliability_table(y_true, y_prob, n_bins)
    n = len(y_true)
    if n == 0 or not table:
        return float("nan")
    return float(sum(
        (row["n"] / n) * abs(row["mean_predicted"] - row["observed_freq"])
        for row in table
    ))


def maximum_calibration_error(y_true, y_prob, n_bins=10):
    """
    MCE = worst single bucket's |mean_predicted - observed_freq|.
    Catches a model that is well calibrated on average but badly wrong
    at the confident extremes — exactly the predictions a user would act
    on, and exactly what a plain average (ECE) can hide.
    """
    table = reliability_table(y_true, y_prob, n_bins)
    if not table:
        return float("nan")
    return float(max(abs(row["mean_predicted"] - row["observed_freq"]) for row in table))


def calibrate_probabilities(base_model, X_calib, y_calib, method="sigmoid"):
    """
    Fit a calibrator (Platt/sigmoid or isotonic) on a held-out purged
    fold the base model never trained on. cv='prefit' is required here —
    fitting on training data produces a calibration curve that looks
    perfect and is meaningless (PREDICTION_SPEC.md 1.2).

    `method='sigmoid'` (2 parameters) is the default: safer with limited
    calibration data. `method='isotonic'` is more flexible but needs
    roughly 1000+ calibration samples or it overfits — the caller
    decides based on how much held-out data it has.

    Returns a fitted CalibratedClassifierCV wrapping base_model.
    """
    from sklearn.calibration import CalibratedClassifierCV

    try:
        # sklearn >=1.6: cv='prefit' was removed in favour of explicitly
        # freezing the already-fitted estimator so CalibratedClassifierCV
        # cannot re-fit it internally.
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    except ImportError:
        # sklearn <1.6
        calibrator = CalibratedClassifierCV(base_model, method=method, cv="prefit")
    calibrator.fit(X_calib, y_calib)
    return calibrator


def compare_calibration_methods(base_model, X_calib, y_calib):
    """
    Fit both Platt and isotonic on the same held-out fold, score both on
    it with Brier/ECE/MCE, and return (best_method, results_by_method).
    "Best" = lowest Brier score, the spec's tie-breaker of choice since
    it captures both calibration and discrimination in one number.
    """
    results = {}
    for method in ("sigmoid", "isotonic"):
        try:
            calib = calibrate_probabilities(base_model, X_calib, y_calib, method=method)
            prob = calib.predict_proba(X_calib)[:, 1]
            results[method] = {
                "model": calib,
                **brier_decomposition(y_calib, prob),
                "ece": expected_calibration_error(y_calib, prob),
                "mce": maximum_calibration_error(y_calib, prob),
            }
        except Exception as e:  # isotonic can fail to fit on tiny folds
            results[method] = {"error": str(e)}

    scored = {k: v for k, v in results.items() if "brier" in v}
    best = min(scored, key=lambda k: scored[k]["brier"]) if scored else None
    return best, results


# ------------------------------------------------------------------ #
# Phase 4.1 — Combinatorial Purged Cross-Validation                   #
# ------------------------------------------------------------------ #

def combinatorial_purged_cv(n_samples, n_groups=6, n_test_groups=2,
                            horizon=12, embargo=6):
    """
    CPCV (López de Prado, AFML ch.12). Split the series into `n_groups`
    contiguous time blocks, then yield one (train_idx, test_idx) fold for
    every C(n_groups, n_test_groups) way of choosing which blocks are
    test blocks. Train always excludes the test blocks AND purges
    `horizon + embargo` bars on both sides of every train/test boundary.

    Where a single walk-forward split gives one performance number,
    this gives a DISTRIBUTION across many overlapping-but-distinct
    train/test partitions of the same data — evaluate.py's caller can
    then report a confidence interval on the edge instead of a point
    estimate ("Sharpe 1.2 +/- 0.9" instead of "Sharpe 1.2").

    NOTE: unlike purged_walk_forward, test blocks here are not
    necessarily chronologically after their train data — some test
    blocks are embedded inside the training span. That is intentional
    for CPCV (it's what makes "combinatorial" meaningful), but it means
    this must never be the ONLY validation used — pair it with the plain
    walk-forward result, which is the more conservative, causally
    ordered one.
    """
    from itertools import combinations

    if n_test_groups >= n_groups:
        raise ValueError("n_test_groups must be < n_groups")

    bounds = np.linspace(0, n_samples, n_groups + 1).astype(int)
    blocks = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]
    gap = horizon + embargo

    for test_group_ids in combinations(range(n_groups), n_test_groups):
        test_idx = np.concatenate([blocks[g] for g in test_group_ids])
        test_group_ids_set = set(test_group_ids)

        train_blocks = []
        for g in range(n_groups):
            if g in test_group_ids_set:
                continue
            block = blocks[g]
            # Purge: drop train bars within `gap` of any test block's
            # start or end, on either side.
            keep = np.ones(len(block), dtype=bool)
            for tg in test_group_ids:
                t_start, t_end = bounds[tg], bounds[tg + 1] - 1
                near = (block >= t_start - gap) & (block <= t_end + gap)
                keep &= ~near
            train_blocks.append(block[keep])

        train_idx = np.concatenate(train_blocks) if train_blocks else np.array([], dtype=int)
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue
        yield np.sort(train_idx), np.sort(test_idx)


# ------------------------------------------------------------------ #
# Statistics helpers — self-contained so the pipeline does not gain a #
# scipy dependency for two functions (ml/requirements.txt stays       #
# numpy + scikit-learn only).                                          #
# ------------------------------------------------------------------ #

def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p):
    """
    Inverse standard normal CDF — Acklam's rational approximation
    (accurate to ~1.15e-9), the standard scipy-free substitute for
    `scipy.stats.norm.ppf`.
    """
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    p_low = 0.02425
    p_high = 1 - p_low

    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def _skew_kurtosis(returns):
    """Sample skewness and (non-excess) kurtosis, population moments."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return 0.0, 3.0
    mu = r.mean()
    sd = r.std()
    if sd <= 0:
        return 0.0, 3.0
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess; normal == 3
    return skew, kurt


# ------------------------------------------------------------------ #
# Phase 4.2 — Deflated Sharpe Ratio                                    #
# ------------------------------------------------------------------ #

def sharpe_ratio(returns):
    """Per-observation Sharpe (not annualised — caller scales if needed)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std())


def expected_max_sharpe(n_trials, trial_sharpes):
    """
    E[max Sharpe over `n_trials` independent trials of zero true skill]
    (Bailey & López de Prado, 2014). Under the null of no edge, this
    grows with n_trials — the more configurations you try and keep the
    best of, the better that best one looks by chance alone, even though
    nothing changed about the underlying strategy.

    `trial_sharpes` — the actually-observed Sharpe ratios of every
    configuration tried (including ones you rejected). Their standard
    deviation estimates the variance of the SR estimator itself, which
    the extreme-value formula needs. Log every configuration you test —
    without this, DSR cannot be computed honestly.
    """
    if n_trials <= 1 or len(trial_sharpes) < 2:
        return 0.0
    sigma_sr = float(np.std(trial_sharpes, ddof=1))
    if sigma_sr == 0:
        return 0.0
    gamma = 0.5772156649015329  # Euler-Mascheroni constant
    e = math.e
    term = (1 - gamma) * _norm_ppf(1 - 1.0 / n_trials) + gamma * _norm_ppf(1 - 1.0 / (n_trials * e))
    return sigma_sr * term


def deflated_sharpe_ratio(observed_sharpe, n_obs, n_trials, trial_sharpes,
                          returns=None, skew=0.0, kurtosis=3.0):
    """
    DSR — the probability that the observed Sharpe ratio is genuinely
    positive AFTER correcting for how many configurations were tried,
    the length of the series, and the skew/kurtosis of returns
    (Bailey & López de Prado, "The Deflated Sharpe Ratio").

    If you try 50 configurations and keep the best, the best will look
    good by chance alone — DSR is the number that says whether it still
    looks good after that correction. DSR > 0.95 is the common bar for
    "probably real"; this function returns the raw probability, not a
    verdict, so the caller can set its own bar.

    `returns`, if given, overrides `skew`/`kurtosis` by computing them
    directly from the trade-level returns — more honest than assuming
    normality, since return distributions in this pipeline are fat-tailed.
    """
    if returns is not None:
        skew, kurtosis = _skew_kurtosis(returns)

    sr0 = expected_max_sharpe(n_trials, trial_sharpes)

    denom = 1 - skew * observed_sharpe + ((kurtosis - 1) / 4.0) * observed_sharpe ** 2
    if denom <= 0 or n_obs <= 1:
        return {"dsr": float("nan"), "sr0": sr0, "sigma_sr": float("nan")}
    sigma_sr_hat = math.sqrt(denom / (n_obs - 1))
    if sigma_sr_hat <= 0:
        return {"dsr": float("nan"), "sr0": sr0, "sigma_sr": sigma_sr_hat}

    z = (observed_sharpe - sr0) / sigma_sr_hat
    return {"dsr": float(_norm_cdf(z)), "sr0": float(sr0), "sigma_sr": float(sigma_sr_hat)}


# ------------------------------------------------------------------ #
# Phase 4.3 — Probability of Backtest Overfitting (via CSCV)          #
# ------------------------------------------------------------------ #

def probability_of_backtest_overfitting(performance_matrix):
    """
    PBO via Combinatorially Symmetric Cross-Validation (Bailey, Borwein,
    López de Prado & Zhu, 2015).

    `performance_matrix` — (n_configs, n_periods) array of a performance
    statistic (edge, or Sharpe) for every configuration you evaluated, on
    every one of `n_periods` comparable sub-periods (e.g. one column per
    walk-forward fold). n_periods must be even and >= 4; an odd count is
    trimmed by dropping the oldest period.

    Method: split the periods in half every possible way (C(S, S/2)
    combinations). For each split, find the configuration with the best
    mean performance on one half (in-sample, IS); look up where THAT
    configuration ranks on the other half (out-of-sample, OOS). Convert
    the OOS rank to a logit lambda = ln(r / (1-r)) where r is the
    relative rank in (0, 1). PBO = fraction of splits where lambda <= 0,
    i.e. the IS-best configuration performed at or below the OOS median —
    pure noise-fitting, not a real edge that happened to look best.

    Above ~0.5 means the selection process (trying many configurations
    and keeping the best) is indistinguishable from picking at random.
    """
    from itertools import combinations

    perf = np.asarray(performance_matrix, dtype=float)
    n_configs, n_periods = perf.shape
    if n_periods % 2 != 0:
        perf = perf[:, 1:]
        n_periods -= 1
    if n_periods < 4 or n_configs < 2:
        return {"pbo": float("nan"), "n_splits": 0, "logits": []}

    half = n_periods // 2
    logits = []

    for is_periods in combinations(range(n_periods), half):
        is_periods = list(is_periods)
        oos_periods = [p for p in range(n_periods) if p not in is_periods]

        is_perf = perf[:, is_periods].mean(axis=1)
        oos_perf = perf[:, oos_periods].mean(axis=1)

        best_config = int(np.argmax(is_perf))
        # Rank of the IS-best config's OOS performance among all configs,
        # 1 = worst, n_configs = best.
        order = np.argsort(oos_perf)
        rank = int(np.where(order == best_config)[0][0]) + 1
        r = rank / (n_configs + 1)
        r = min(max(r, 1e-6), 1 - 1e-6)
        logits.append(math.log(r / (1 - r)))

    logits = np.array(logits)
    pbo = float(np.mean(logits <= 0))
    return {"pbo": pbo, "n_splits": len(logits), "logits": logits.tolist()}
