"""
ml/train.py
===================================================================
Train, validate honestly, and export a model for JS inference.

    python3 ml/train.py --symbol BTC/USDT
    python3 ml/train.py --synthetic random      # expect NO_EDGE
    python3 ml/train.py --synthetic signal      # expect PROMISING
    python3 ml/train.py --symbol BTC/USDT --export

Phase flags (PREDICTION_SPEC.md):
    --derivatives            fetch+use funding/OI/top-trader context (2.1)
    --pool ETH/USDT,SOL/USDT pool --symbol with siblings, cross-sectional
                              features + 3x the training rows (2.3)
    --meta-label              train primary(heuristic)+secondary(P(correct))
                              instead of raw direction (0.3)
    --calibrate sigmoid|isotonic|both|off   Platt/isotonic calibration,
                              reported via Brier/ECE/MCE/reliability (1.2, 1.3)
    --vol-model har|garch|off  use ml/volatility.py's forecast as the
                              triple-barrier sigma_t instead of the
                              built-in EWMA (Phase 3 -> 0.1 integration)
    --cpcv                    also report Combinatorial Purged CV (4.1)
    --log-trial                append this run's Sharpe to
                              ml/cache/trials.jsonl and report the
                              Deflated Sharpe Ratio against every trial
                              ever logged (4.2)

Models compared (all against the baselines in validation.py):
  logreg  — L2 logistic regression on standardised features. Linear,
            interpretable, hard to overfit, and inference in JS is a
            single dot product. The default.
  gbm     — histogram gradient boosting. Captures interactions, but with
            noisy labels it usually overfits; included so the comparison
            is honest rather than assumed.

Only the winner by mean walk-forward edge is exported, and export is
BLOCKED unless the verdict clears a minimum bar — a model with no
measured edge must not silently reach production.
===================================================================
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from features import (FEATURE_NAMES, N_FEATURES, build_features,
                      build_labels, assert_no_lookahead)
from weights import sample_weights, average_uniqueness
from validation import (purged_walk_forward, combinatorial_purged_cv,
                        baseline_predictions, heuristic_predictions,
                        classification_metrics, aggregate_folds, verdict,
                        compare_calibration_methods, reliability_table,
                        expected_calibration_error, maximum_calibration_error,
                        brier_decomposition, sharpe_ratio, deflated_sharpe_ratio,
                        probability_of_backtest_overfitting)
import meta_labeling
import volatility as vol_mod
import data as data_mod

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


# Round-trip cost assumptions, in basis points. These drive the net-return
# metric, so getting them roughly right matters more than any hyperparameter.
#   crypto: ~5bps taker fee each way + slippage
#   forex:  ~1.5bps typical retail EUR/USD spread
DEFAULT_COST_BPS = {"crypto": 12.0, "forex": 3.0}

TRIALS_LOG = os.path.join(os.path.dirname(__file__), "cache", "trials.jsonl")


def asset_class_of(symbol):
    fiat = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
    p = symbol.upper().split("/")
    return "forex" if len(p) == 2 and p[0] in fiat and p[1] in fiat else "crypto"


# ------------------------------------------------------------------ #
# Dataset assembly (Phase 0.1, 0.2)                                   #
# ------------------------------------------------------------------ #

def build_dataset(candles, k1=2.0, k2=1.0, vertical=24, ewma_span=20,
                  context=None, check_lookahead=True, sigma_override=None,
                  time_decay_min=0.5):
    """
    candles -> a dict with two views of the labelled data:

      'binary'  — decisive triple-barrier touches only (label != 0),
                  the direction-classification training set. Timeouts
                  are dropped: they are genuinely ambiguous ("the trade
                  neither clearly worked nor clearly failed"), the same
                  role the old cost-band dead zone used to play.
      'full'    — every labelable row including timeouts, with sample
                  weights computed over the FULL set (a timeout label
                  still occupies bars and must count toward its
                  neighbours' concurrency). Used by meta-labelling,
                  which scores "was the primary directionally right in
                  sign" — a question timeouts can still answer.

    Sample weights (uniqueness x |return| x time-decay, Phase 0.2) are
    always computed on the full labelable set so concurrency accounts
    for every open label, then subset for the binary view.
    """
    if check_lookahead:
        assert_no_lookahead(candles, context=context)

    X, valid = build_features(candles, context=context)
    label, touch_idx, realised_return, labelable = build_labels(
        candles["close"], k1=k1, k2=k2, vertical=vertical,
        ewma_span=ewma_span, sigma_override=sigma_override,
    )

    n_bars = len(X)
    keep = valid & labelable
    orig_idx = np.where(keep)[0]

    X_k = X[keep]
    label_k = label[keep]
    touch_k = touch_idx[keep]
    ret_k = realised_return[keep]

    w_full = sample_weights(n_bars, orig_idx, touch_k, ret_k, time_decay_min=time_decay_min)
    uniq = average_uniqueness(n_bars, orig_idx, touch_k)

    decisive = label_k != 0
    n_timeout = int((~decisive).sum())

    print(f"  rows: {n_bars} total -> {int(keep.sum())} labelable "
          f"({int((~valid).sum())} warmup, {int(valid.sum() - keep.sum())} label-warmup)")
    print(f"  triple-barrier: {int((label_k==1).sum())} upper, "
          f"{int((label_k==-1).sum())} lower, {n_timeout} timeout (dropped from direction set)")
    print(f"  average sample uniqueness: {uniq:.3f} "
          f"(1.0 = no overlap, ~1/{vertical} = fully overlapping)")

    return {
        "n_bars": n_bars,
        "full": {
            "X": X_k, "label": label_k, "touch_idx": touch_k,
            "realised_return": ret_k, "orig_idx": orig_idx, "weight": w_full,
        },
        "binary": {
            "X": X_k[decisive],
            "y": (label_k[decisive] == 1).astype(int),
            "realised_return": ret_k[decisive],
            "weight": w_full[decisive],
            "orig_idx": orig_idx[decisive],
        },
        "uniqueness": uniq,
    }


# ------------------------------------------------------------------ #
# Models                                                              #
# ------------------------------------------------------------------ #

def make_model(kind):
    if kind == "logreg":
        return LogisticRegression(
            C=0.1,               # strong regularisation: the data is noisy
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        )
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_depth=3,         # shallow on purpose
            max_iter=150,
            learning_rate=0.05,
            l2_regularization=1.0,
            min_samples_leaf=40,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        )
    raise ValueError(f"unknown model kind: {kind}")


class Standardiser:
    """
    Fit mean/std on TRAIN ONLY.

    Fitting the scaler on the full dataset is a real and common leak:
    test-set statistics bleed into the training transform. Doing it per
    fold is the only correct option. The final exported scaler is refit
    on all data, which is fine because at that point there is no test
    set left to contaminate.
    """

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        return self

    def transform(self, X):
        return (X - self.mean_) / self.scale_


# ------------------------------------------------------------------ #
# Walk-forward evaluation                                             #
# ------------------------------------------------------------------ #

def evaluate(X, y, fwd, weight=None, kinds=("logreg", "gbm"), n_splits=5,
            horizon=24, embargo=6, cost_bps=0.0, expanding=True, seed=0):
    rng = np.random.default_rng(seed)
    per_model = {k: [] for k in kinds}
    per_baseline = {}
    heur_folds = []
    n_folds = 0

    for train_idx, test_idx in purged_walk_forward(
        len(X), n_splits=n_splits, horizon=horizon, embargo=embargo,
        expanding=expanding,
    ):
        n_folds += 1
        Xtr, ytr = X[train_idx], y[train_idx]
        wtr = weight[train_idx] if weight is not None else None
        Xte, yte, fte = X[test_idx], y[test_idx], fwd[test_idx]

        # Degenerate fold: only one class present in train.
        if len(np.unique(ytr)) < 2:
            print(f"  fold {n_folds}: skipped (single class in train)")
            continue

        scaler = Standardiser().fit(Xtr)
        Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

        for kind in kinds:
            model = make_model(kind)
            # Trees do not need scaling; linear models do.
            a, b = (Xtr_s, Xte_s) if kind == "logreg" else (Xtr, Xte)
            model.fit(a, ytr, sample_weight=wtr)
            prob = model.predict_proba(b)[:, 1]
            pred = (prob >= 0.5).astype(int)
            per_model[kind].append(
                classification_metrics(yte, pred, prob, fte, cost_bps)
            )

        for name, pred in baseline_predictions(ytr, yte, rng).items():
            per_baseline.setdefault(name, []).append(
                classification_metrics(yte, pred, None, fte, cost_bps)
            )

        # Heuristic, scored only where it expressed an opinion.
        hp = heuristic_predictions(Xte, FEATURE_NAMES)
        op = hp >= 0
        if op.sum() >= 20:
            m = classification_metrics(yte[op], hp[op], None, fte[op], cost_bps)
            m["coverage"] = float(op.mean())
            heur_folds.append(m)

    return {
        "n_folds": n_folds,
        "models": {k: aggregate_folds(v) for k, v in per_model.items()},
        "baselines": {k: aggregate_folds(v) for k, v in per_baseline.items()},
        "heuristic": aggregate_folds(heur_folds),
        "model_folds": per_model,
    }


# ------------------------------------------------------------------ #
# Reporting                                                           #
# ------------------------------------------------------------------ #

def report(results, label=""):
    print("\n" + "=" * 74)
    print(f"WALK-FORWARD RESULTS{(' — ' + label) if label else ''}"
          f"   ({results['n_folds']} folds, purged + embargoed)")
    print("=" * 74)
    print(f"{'model':<12} {'acc':>7} {'base':>7} {'EDGE':>8} {'±sd':>7} "
          f"{'+folds':>7} {'auc':>6} {'net bps':>9}")
    print("-" * 74)

    def row(name, agg):
        if not agg:
            print(f"{name:<12} {'—':>7}")
            return
        acc = agg.get("accuracy_mean", float("nan"))
        base = agg.get("base_rate_mean", float("nan"))
        edge = agg.get("edge_mean", float("nan"))
        sd = agg.get("edge_std", float("nan"))
        pf = agg.get("folds_positive_edge", 0)
        nf = agg.get("folds", 0)
        auc = agg.get("auc_mean", float("nan"))
        net = agg.get("net_bps_mean", float("nan"))
        print(f"{name:<12} {acc*100:>6.2f}% {base*100:>6.2f}% "
              f"{edge*100:>+7.2f}pp {sd*100:>6.2f} {pf:>3}/{nf:<3} "
              f"{auc:>6.3f} {net:>+9.2f}")

    for k, agg in results["models"].items():
        row(k, agg)
    print("-" * 74)
    for k, agg in results["baselines"].items():
        row(k, agg)
    if results["heuristic"]:
        row("heuristic", results["heuristic"])
        cov = results["heuristic"].get("coverage_mean")
        if cov is not None:
            print(f"{'':12} (rule-based signal, expressed an opinion on "
                  f"{cov*100:.0f}% of bars)")
    print("-" * 74)

    best, best_edge = None, -np.inf
    for k, agg in results["models"].items():
        e = agg.get("edge_mean", -np.inf)
        if np.isfinite(e) and e > best_edge:
            best, best_edge = k, e

    if best:
        code, text = verdict(results["models"][best], results["n_folds"])
        print(f"\nBEST MODEL: {best}")
        print(f"VERDICT:    {code}")
        # Wrap the explanation to a readable width.
        words, line = text.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > 70:
                print(f"            {line}")
                line = w
            else:
                line = f"{line} {w}".strip()
        if line:
            print(f"            {line}")
    print("=" * 74 + "\n")
    return best


def report_cpcv(X, y, weight, kinds, n_groups, n_test_groups, vertical, embargo, cost_bps):
    """Phase 4.1 — a distribution of edge across CPCV's combinatorial splits."""
    print("\n" + "-" * 74)
    print(f"COMBINATORIAL PURGED CV  ({n_groups} groups, choose {n_test_groups} as test)")
    print("-" * 74)
    per_model = {k: [] for k in kinds}
    n_splits = 0
    for train_idx, test_idx in combinatorial_purged_cv(
        len(X), n_groups=n_groups, n_test_groups=n_test_groups,
        horizon=vertical, embargo=embargo,
    ):
        ytr = y[train_idx]
        if len(np.unique(ytr)) < 2 or len(test_idx) < 20:
            continue
        n_splits += 1
        scaler = Standardiser().fit(X[train_idx])
        wtr = weight[train_idx] if weight is not None else None
        for kind in kinds:
            model = make_model(kind)
            a = scaler.transform(X[train_idx]) if kind == "logreg" else X[train_idx]
            b = scaler.transform(X[test_idx]) if kind == "logreg" else X[test_idx]
            model.fit(a, ytr, sample_weight=wtr)
            prob = model.predict_proba(b)[:, 1]
            pred = (prob >= 0.5).astype(int)
            per_model[kind].append(classification_metrics(y[test_idx], pred, prob))

    for kind in kinds:
        edges = [m["edge"] for m in per_model[kind] if np.isfinite(m.get("edge", np.nan))]
        if not edges:
            print(f"{kind:<12} no valid splits")
            continue
        edges = np.array(edges)
        lo, hi = np.percentile(edges, [5, 95])
        print(f"{kind:<12} edge {edges.mean()*100:+.2f}pp  "
              f"90% range [{lo*100:+.2f}, {hi*100:+.2f}]pp  over {len(edges)}/{n_splits} splits")
    print("-" * 74)
    return per_model


def report_dsr(results, kinds, cost_bps, log_trial, symbol_label):
    """
    Phase 4.2 — Deflated Sharpe Ratio. Every fold's per-trade Sharpe
    proxy (already computed by classification_metrics as
    'sharpe_per_trade') is pooled into one observed Sharpe per model
    kind, then compared against every trial ever logged to
    ml/cache/trials.jsonl (across all runs of this script, any symbol
    or config) — this is the only honest trial count: DSR corrects for
    how many configurations were tried in total, not just this run's.
    """
    os.makedirs(os.path.dirname(TRIALS_LOG), exist_ok=True)
    history = []
    if os.path.exists(TRIALS_LOG):
        with open(TRIALS_LOG) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    print("\n" + "-" * 74)
    print("DEFLATED SHARPE RATIO")
    print("-" * 74)

    new_records = []
    for kind in kinds:
        folds = results["model_folds"].get(kind, [])
        sharpes = [m["sharpe_per_trade"] for m in folds if np.isfinite(m.get("sharpe_per_trade", np.nan))]
        n_obs = sum(m.get("n", 0) for m in folds)
        if not sharpes:
            print(f"{kind:<12} no valid folds")
            continue
        observed = float(np.mean(sharpes))
        record = {"symbol": symbol_label, "kind": kind, "sharpe": observed,
                  "n_obs": int(n_obs), "ts": time.time()}
        new_records.append(record)

        all_sharpes = [h["sharpe"] for h in history if np.isfinite(h.get("sharpe", float("nan")))]
        all_sharpes.append(observed)
        n_trials = len(all_sharpes)

        dsr = deflated_sharpe_ratio(
            observed_sharpe=observed, n_obs=max(n_obs, 2), n_trials=n_trials,
            trial_sharpes=all_sharpes,
        )
        print(f"{kind:<12} observed Sharpe {observed:+.4f}  over {n_trials} logged trial(s)  "
              f"-> DSR {dsr['dsr']*100:.1f}% probability of genuine (non-chance) edge")
        if n_trials < 5:
            print(f"{'':12} (few trials logged so far — DSR is not very informative yet; "
                  f"use --log-trial on every run to build up an honest trial count)")

    if log_trial and new_records:
        with open(TRIALS_LOG, "a") as f:
            for r in new_records:
                f.write(json.dumps(r) + "\n")
        print(f"{'':12} logged {len(new_records)} trial(s) to {TRIALS_LOG}")
    print("-" * 74)


def report_pbo(results, kinds):
    """Phase 4.3 — PBO across the model kinds actually compared this run."""
    folds_by_kind = [results["model_folds"].get(k, []) for k in kinds]
    n_folds = min((len(f) for f in folds_by_kind), default=0)
    if len(kinds) < 2 or n_folds < 4:
        print("\n(PBO skipped — needs >=2 model kinds and >=4 comparable folds)")
        return None

    perf = np.array([[f["edge"] for f in folds[:n_folds]] for folds in folds_by_kind])
    result = probability_of_backtest_overfitting(perf)
    print("\n" + "-" * 74)
    print(f"PROBABILITY OF BACKTEST OVERFITTING (CSCV, {result['n_splits']} splits)")
    print(f"  PBO = {result['pbo']*100:.1f}% "
          f"(above ~50% means picking the 'best' of {{{', '.join(kinds)}}} here is noise-fitting)")
    print("-" * 74)
    return result


def report_calibration(X, y, vertical, embargo, methods):
    """
    Phase 1.2/1.3 — proper three-way split: base model fit on TRAIN,
    calibrator fit on CALIB (a purged fold the base model never saw),
    Brier/ECE/MCE/reliability reported on TEST (seen by neither).
    """
    n = len(X)
    gap = vertical + embargo
    test_size = max(int(n * 0.15), 50)
    calib_size = max(int(n * 0.15), 50)
    test_start = n - test_size
    calib_end = test_start - gap
    calib_start = calib_end - calib_size
    train_end = calib_start - gap

    if train_end < 300 or calib_start < 0:
        print("\n(calibration report skipped — not enough rows for a clean "
              "train/calib/test split; collect more history)")
        return None

    train_idx = np.arange(0, train_end)
    calib_idx = np.arange(calib_start, calib_end)
    test_idx = np.arange(test_start, n)

    if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[calib_idx])) < 2:
        print("\n(calibration report skipped — a split had only one class)")
        return None

    scaler = Standardiser().fit(X[train_idx])
    base = make_model("logreg").fit(scaler.transform(X[train_idx]), y[train_idx])

    print("\n" + "-" * 74)
    print("CALIBRATION  (Platt vs isotonic, fit on a held-out purged fold)")
    print("-" * 74)

    if methods == "both":
        best_method, calib_results = compare_calibration_methods(
            base, scaler.transform(X[calib_idx]), y[calib_idx]
        )
    else:
        best_method = methods
        try:
            from validation import calibrate_probabilities
            m = calibrate_probabilities(base, scaler.transform(X[calib_idx]), y[calib_idx], method=methods)
            prob = m.predict_proba(scaler.transform(X[calib_idx]))[:, 1]
            calib_results = {methods: {
                "model": m, **brier_decomposition(y[calib_idx], prob),
                "ece": expected_calibration_error(y[calib_idx], prob),
                "mce": maximum_calibration_error(y[calib_idx], prob),
            }}
        except Exception as e:
            print(f"  calibration failed: {e}")
            return None

    if best_method is None or "model" not in calib_results.get(best_method, {}):
        print("  calibration failed for every method attempted")
        return None

    for method, res in calib_results.items():
        if "brier" not in res:
            print(f"  {method:<10} FAILED: {res.get('error')}")
            continue
        marker = " <- selected" if method == best_method else ""
        print(f"  {method:<10} brier {res['brier']:.4f}  reliability {res['reliability']:.4f}  "
              f"resolution {res['resolution']:.4f}  ece {res['ece']:.4f}  mce {res['mce']:.4f}{marker}")

    calibrator = calib_results[best_method]["model"]
    test_prob = calibrator.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    test_ece = expected_calibration_error(y[test_idx], test_prob)
    test_mce = maximum_calibration_error(y[test_idx], test_prob)
    table = reliability_table(y[test_idx], test_prob)

    print(f"\n  On a FRESH held-out test fold (seen by neither base model nor calibrator):")
    print(f"  ECE {test_ece:.4f}  (target < 0.05)   MCE {test_mce:.4f}")
    print(f"  {'bucket':<14} {'n':>6} {'mean pred':>10} {'observed':>10}")
    for row_ in table:
        print(f"  {row_['lo']*100:4.0f}-{row_['hi']*100:3.0f}%   {row_['n']:>6} "
              f"{row_['mean_predicted']*100:>9.1f}% {row_['observed_freq']*100:>9.1f}%")

    return {
        "method": best_method,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "ece": test_ece,
        "mce": test_mce,
        "reliability_table": table,
    }


def report_meta_labeling(X, feature_names, realised_return, weight, vertical, embargo, cost_bps):
    """Phase 0.3."""
    print("\n" + "-" * 74)
    print("META-LABELLING  (primary = rule-based heuristic, secondary = P(correct))")
    print("-" * 74)
    result = meta_labeling.evaluate_meta_labeling(
        X, feature_names, realised_return, sample_weight=weight,
        horizon=vertical, embargo=embargo, cost_bps=cost_bps,
    )
    if not result.get("folds"):
        print("  not enough data for a meta-labelling fold")
        return None

    print(f"  secondary model: precision {result['precision_mean']:.3f}  "
          f"recall {result['recall_mean']:.3f}  F1 {result['f1_mean']:.3f}")
    print(f"  strategy return, ALL primary signals:            "
          f"{result['unfiltered_bps_mean']:+.2f} bps/trade")
    print(f"  strategy return, filtered by P(correct)>={result['threshold']}: "
          f"{result['filtered_bps_mean']:+.2f} bps/trade "
          f"(traded {result['coverage_mean']*100:.0f}% of opinions)")
    improved = (
        np.isfinite(result['filtered_bps_mean']) and np.isfinite(result['unfiltered_bps_mean'])
        and result['filtered_bps_mean'] > result['unfiltered_bps_mean']
    )
    print(f"  filter {'IMPROVES' if improved else 'does not improve'} on trading every signal")
    print("-" * 74)
    return result


# ------------------------------------------------------------------ #
# Export                                                              #
# ------------------------------------------------------------------ #

ALLOW_EXPORT = {"PROMISING", "WEAK", "EDGE_NO_PROFIT"}


def export_model(kind, X, y, weight, out_path, meta, calibration=None, force=False):
    """
    Refit on all data and write model.json for JS inference.

    Only logistic regression is exportable: inference becomes a dot
    product, which shared/mlModel.js implements exactly, so the JS and
    Python predictions agree to floating-point precision. Serialising a
    boosted-tree ensemble to JS is possible but the tree-traversal
    reimplementation is a fresh source of silent divergence.

    `calibration`, if given (report_calibration()'s return value), is
    embedded in meta so shared/mlModel.js can expose the reliability
    table for whichever bucket the live prediction falls into (Phase 1.4).
    Calibration itself is NOT applied to the exported coef/intercept —
    Platt scaling is just an affine transform of the logit, so exporting
    the RAW logistic regression plus the reliability table (rather than
    trying to serialise a second calibration model) keeps JS inference a
    single dot product while still letting the UI show honestly-measured
    reliability for the bucket the current prediction lands in.
    """
    if kind != "logreg":
        print(f"  [export] {kind} is not exportable to JS. Re-run with "
              f"--model logreg to export a deployable model.")
        return None

    scaler = Standardiser().fit(X)
    model = make_model("logreg").fit(scaler.transform(X), y, sample_weight=weight)

    payload = {
        "schema_version": 2,
        "kind": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "meta": meta,
    }
    if calibration:
        payload["meta"]["calibration"] = {
            "method": calibration["method"],
            "ece": calibration["ece"],
            "mce": calibration["mce"],
            "reliability_table": calibration["reliability_table"],
        }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [export] wrote {out_path}")

    # Feature weights, largest first — the audit trail. If `hour_sin`
    # dominates on a 24/7 crypto pair, the model has latched onto a
    # spurious pattern and should not be trusted.
    order = np.argsort(-np.abs(model.coef_[0]))
    print("\n  Feature weights (standardised, |largest| first):")
    for i in order[:12]:
        print(f"    {FEATURE_NAMES[i]:<22} {model.coef_[0][i]:+.4f}")
    return payload


def export_meta_model(X, feature_names, realised_return, weight, out_path, meta,
                      threshold=0.55, force=False):
    """
    Export the meta-labelling secondary model. Same JSON schema as
    export_model() (still a plain logistic regression -> single JS dot
    product), but meta.mode='meta_label' tells shared/mlModel.js the
    output is P(primary heuristic is right), not P(up) — the UI gates
    on it rather than reading it as a direction.
    """
    model, mean, scale, primary_all, meta_y, has_opinion = meta_labeling.fit_meta_model(
        X, feature_names, realised_return, sample_weight=weight
    )

    payload = {
        "schema_version": 2,
        "kind": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "n_features": N_FEATURES,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "meta": {**meta, "mode": "meta_label", "meta_label_threshold": threshold},
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  [export] wrote {out_path} (meta-labelling mode)")
    return payload


# ------------------------------------------------------------------ #
# Pooled / cross-sectional training (Phase 2.3)                       #
# ------------------------------------------------------------------ #

def build_pooled_dataset(symbols, k1, k2, vertical, ewma_span, use_derivatives,
                         check_lookahead=True):
    """
    Load `symbols`, align them on a shared time grid, and build a
    per-symbol dataset with cross-sectional (+ optionally derivatives)
    context. Returns {symbol: build_dataset() result}.
    """
    candles_by_symbol = {s: data_mod.load_candles(s) for s in symbols}
    _, aligned = data_mod.align_candles_by_time(candles_by_symbol)

    out = {}
    for sym in symbols:
        ctx = data_mod.build_cross_sectional_context(aligned, sym)
        if use_derivatives and asset_class_of(sym) == "crypto":
            deriv = data_mod.load_derivatives_context(sym, aligned[sym]["open_time"])
            ctx.update(deriv)
        print(f"\n  [{sym}] building features + labels ...")
        out[sym] = build_dataset(
            aligned[sym], k1=k1, k2=k2, vertical=vertical, ewma_span=ewma_span,
            context=ctx, check_lookahead=check_lookahead,
        )
    return out


def evaluate_pooled(per_symbol, kinds, n_splits, vertical, embargo, cost_bps, seed=0):
    """
    Walk-forward per symbol (correct purge/embargo semantics require a
    single symbol's own chronology), but each fold's TRAIN set is that
    symbol's own purged train rows UNION the full labelled history of
    every OTHER pooled symbol. Cross-symbol rows carry no temporal
    leakage risk into a given symbol's test window — they are a
    different instrument, not future information about this one — so
    only the symbol's own timeline needs purging.
    """
    rng = np.random.default_rng(seed)
    per_model = {k: [] for k in kinds}
    per_baseline = {}
    heur_folds = []
    n_folds = 0

    symbols = list(per_symbol.keys())
    for sym in symbols:
        ds = per_symbol[sym]["binary"]
        X, y, w, fwd = ds["X"], ds["y"], ds["weight"], ds["realised_return"]
        others_X = np.concatenate([per_symbol[s]["binary"]["X"] for s in symbols if s != sym]) \
            if len(symbols) > 1 else np.empty((0, X.shape[1]))
        others_y = np.concatenate([per_symbol[s]["binary"]["y"] for s in symbols if s != sym]) \
            if len(symbols) > 1 else np.empty((0,), dtype=int)
        others_w = np.concatenate([per_symbol[s]["binary"]["weight"] for s in symbols if s != sym]) \
            if len(symbols) > 1 else np.empty((0,))

        for train_idx, test_idx in purged_walk_forward(
            len(X), n_splits=n_splits, horizon=vertical, embargo=embargo,
        ):
            Xtr = np.concatenate([X[train_idx], others_X])
            ytr = np.concatenate([y[train_idx], others_y])
            wtr = np.concatenate([w[train_idx], others_w])
            Xte, yte, fte = X[test_idx], y[test_idx], fwd[test_idx]

            if len(np.unique(ytr)) < 2:
                continue
            n_folds += 1

            scaler = Standardiser().fit(Xtr)
            Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

            for kind in kinds:
                model = make_model(kind)
                a, b = (Xtr_s, Xte_s) if kind == "logreg" else (Xtr, Xte)
                model.fit(a, ytr, sample_weight=wtr)
                prob = model.predict_proba(b)[:, 1]
                pred = (prob >= 0.5).astype(int)
                per_model[kind].append(classification_metrics(yte, pred, prob, fte, cost_bps))

            for name, pred in baseline_predictions(ytr, yte, rng).items():
                per_baseline.setdefault(name, []).append(
                    classification_metrics(yte, pred, None, fte, cost_bps)
                )

            hp = heuristic_predictions(Xte, FEATURE_NAMES)
            op = hp >= 0
            if op.sum() >= 20:
                m = classification_metrics(yte[op], hp[op], None, fte[op], cost_bps)
                m["coverage"] = float(op.mean())
                heur_folds.append(m)

    return {
        "n_folds": n_folds,
        "models": {k: aggregate_folds(v) for k, v in per_model.items()},
        "baselines": {k: aggregate_folds(v) for k, v in per_baseline.items()},
        "heuristic": aggregate_folds(heur_folds),
        "model_folds": per_model,
    }


# ------------------------------------------------------------------ #
# Main                                                                #
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Train and honestly validate a TrendPulse model.")
    ap.add_argument("--symbol", default=None, help="e.g. BTC/USDT or EUR/USD")
    ap.add_argument("--synthetic", choices=["random", "signal"], default=None)
    ap.add_argument("--horizon", type=int, default=24,
                    help="triple-barrier vertical, bars ahead (24 = 2h on 5min bars)")
    ap.add_argument("--k1", type=float, default=2.0, help="upper barrier, x EWMA sigma")
    ap.add_argument("--k2", type=float, default=1.0, help="lower barrier, x EWMA sigma")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--embargo", type=int, default=6)
    ap.add_argument("--cost-bps", type=float, default=None)
    ap.add_argument("--model", choices=["logreg", "gbm", "both"], default="both")
    ap.add_argument("--rolling", action="store_true", help="rolling instead of expanding window")
    ap.add_argument("--derivatives", action="store_true",
                    help="fetch+use funding/OI/top-trader context (crypto only, Phase 2.1)")
    ap.add_argument("--pool", default=None,
                    help="comma-separated sibling symbols to pool with --symbol (Phase 2.3), "
                         "e.g. --symbol BTC/USDT --pool ETH/USDT,SOL/USDT")
    ap.add_argument("--meta-label", action="store_true",
                    help="report + optionally export the primary/secondary meta-labelling pair (Phase 0.3)")
    ap.add_argument("--meta-threshold", type=float, default=0.55)
    ap.add_argument("--calibrate", choices=["sigmoid", "isotonic", "both", "off"], default="both",
                    help="Phase 1.2/1.3 calibration report")
    ap.add_argument("--vol-model", choices=["har", "garch", "off"], default="off",
                    help="use ml/volatility.py's forecast as triple-barrier sigma_t (Phase 3)")
    ap.add_argument("--cpcv", action="store_true", help="also report CPCV (Phase 4.1)")
    ap.add_argument("--log-trial", action="store_true",
                    help="log this run's Sharpe for DSR (Phase 4.2)")
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--force-export", action="store_true",
                    help="export even with no measured edge (not recommended)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not args.symbol and not args.synthetic:
        ap.error("pass --symbol or --synthetic")
    if args.pool and not args.symbol:
        ap.error("--pool requires --symbol (the pool's primary/exported symbol)")

    vertical = args.horizon

    # --- data + labels
    if args.synthetic:
        label = f"synthetic:{args.synthetic}"
        print(f"Loading {label} ...")
        candles = (data_mod.synthetic_random_walk(n=6000, drift=0.00004)
                   if args.synthetic == "random"
                   else data_mod.synthetic_with_signal(n=6000))
        cost = args.cost_bps if args.cost_bps is not None else 0.0
        print(f"  {len(candles['close'])} bars loaded")

        sigma_override = None
        if args.vol_model != "off":
            sigma_override = _vol_forecast(candles["close"], args.vol_model)

        print("\nBuilding features + triple-barrier labels (with look-ahead assertion) ...")
        ds = build_dataset(candles, k1=args.k1, k2=args.k2, vertical=vertical,
                           sigma_override=sigma_override)
        pooled = {label: ds}

    else:
        label = args.symbol
        klass = asset_class_of(args.symbol)
        cost = args.cost_bps if args.cost_bps is not None else DEFAULT_COST_BPS[klass]
        print(f"Loading {label} ({klass}) ... round-trip cost assumption: {cost:.1f} bps")

        symbols = [args.symbol] + (args.pool.split(",") if args.pool else [])
        symbols = [s.strip() for s in symbols if s.strip()]

        if len(symbols) > 1:
            print(f"  pooling: {', '.join(symbols)}")
            pooled = build_pooled_dataset(
                symbols, args.k1, args.k2, vertical, ewma_span=20,
                use_derivatives=args.derivatives,
            )
        else:
            candles = data_mod.load_candles(args.symbol)
            print(f"  {len(candles['close'])} bars loaded")
            context = None
            if args.derivatives and klass == "crypto":
                context = data_mod.load_derivatives_context(args.symbol, candles["open_time"])

            sigma_override = None
            if args.vol_model != "off":
                sigma_override = _vol_forecast(candles["close"], args.vol_model)

            print("\nBuilding features + triple-barrier labels (with look-ahead assertion) ...")
            ds = build_dataset(candles, k1=args.k1, k2=args.k2, vertical=vertical,
                               context=context, sigma_override=sigma_override)
            pooled = {args.symbol: ds}

    # --- evaluate
    kinds = ("logreg", "gbm") if args.model == "both" else (args.model,)
    print(f"\nEvaluating {', '.join(kinds)} ...")

    if len(pooled) > 1:
        results = evaluate_pooled(pooled, kinds, args.splits, vertical, args.embargo, cost)
        # Training matrices for downstream (calibration/export/CPCV/meta) —
        # the primary symbol's own binary set plus siblings, same union
        # evaluate_pooled() trains each fold on.
        primary = args.symbol
        X_all = np.concatenate([pooled[s]["binary"]["X"] for s in pooled])
        y_all = np.concatenate([pooled[s]["binary"]["y"] for s in pooled])
        w_all = np.concatenate([pooled[s]["binary"]["weight"] for s in pooled])
        ds_primary = pooled[primary]
    else:
        (primary, ds_primary), = pooled.items()
        b = ds_primary["binary"]
        X_all, y_all, w_all = b["X"], b["y"], b["weight"]
        results = evaluate(
            X_all, y_all, b["realised_return"], weight=w_all, kinds=kinds,
            n_splits=args.splits, horizon=vertical, embargo=args.embargo,
            cost_bps=cost, expanding=not args.rolling,
        )

    best = report(results, label)

    if args.cpcv:
        report_cpcv(X_all, y_all, w_all, kinds, n_groups=6, n_test_groups=2,
                   vertical=vertical, embargo=args.embargo, cost_bps=cost)

    report_dsr(results, kinds, cost, args.log_trial, label)
    report_pbo(results, kinds)

    calibration = None
    if args.calibrate != "off" and best == "logreg":
        calibration = report_calibration(X_all, y_all, vertical, args.embargo, args.calibrate)
    elif args.calibrate != "off":
        print(f"\n(calibration report skipped — best model is '{best}', "
              f"calibration is only wired up for logreg here)")

    if args.meta_label:
        full = ds_primary["full"]
        report_meta_labeling(full["X"], FEATURE_NAMES, full["realised_return"],
                            full["weight"], vertical, args.embargo, cost)

    # --- export
    if args.export and best:
        code, _ = verdict(results["models"][best], results["n_folds"])
        if code not in ALLOW_EXPORT and not args.force_export:
            print(f"EXPORT BLOCKED: verdict is {code}. This model has no measured "
                  f"edge, so shipping it would put a number on screen that means "
                  f"nothing.\nUse --force-export to override (and set "
                  f"ML_SHOW_UNVALIDATED=true in Netlify to display it as unvalidated).")
        else:
            out = args.out or os.path.join(
                os.path.dirname(__file__), "..", "shared", "model.json")
            meta = {
                "symbol": label,
                "pooled_with": [s for s in pooled if s != primary] or None,
                "horizon_bars": vertical,
                "triple_barrier": {"k1": args.k1, "k2": args.k2, "vertical": vertical},
                "cost_bps": cost,
                "verdict": code,
                "sample_uniqueness": ds_primary["uniqueness"],
                "walk_forward": {
                    k: v for k, v in results["models"][best].items()
                    if k.endswith(("_mean", "_std")) or k.startswith("folds")
                },
                "trained_rows": int(len(X_all)),
            }

            if args.meta_label:
                full = ds_primary["full"]
                export_meta_model(
                    full["X"], FEATURE_NAMES, full["realised_return"], full["weight"],
                    os.path.abspath(out), meta, threshold=args.meta_threshold,
                    force=args.force_export,
                )
            else:
                export_model(best, X_all, y_all, w_all, os.path.abspath(out), meta,
                            calibration=calibration, force=args.force_export)

    return 0


def _vol_forecast(close, kind):
    """Phase 3 -> Phase 0.1 integration: HAR-RV or GARCH sigma_t series."""
    print(f"  fitting {kind.upper()} volatility model for barrier placement ...")
    try:
        if kind == "har":
            fit = vol_mod.fit_har_rv(close)
        else:
            fit = vol_mod.fit_garch(close)
        print(f"  {kind.upper()} QLIKE (holdout): {fit['qlike']:.4f}")
        return fit["sigma"]
    except Exception as e:
        print(f"  [vol-model] {kind} failed ({e}) — falling back to the built-in EWMA")
        return None


if __name__ == "__main__":
    sys.exit(main())
