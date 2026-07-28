"""
ml/meta_labeling.py
===================================================================
Meta-labelling (Phase 0.3 of PREDICTION_SPEC.md) — "the highest-leverage
change in Phase 0".

Instead of asking a model to predict direction (close to random, per
ml/README.md), reuse the existing rule-based heuristic
(validation.heuristic_predictions — the JS twin is computeSignal() in
shared/indicators.js) as a PRIMARY model that already picks a side, and
train a SECONDARY model to answer a narrower, better-posed question:

    P(the primary call is correct)

Trade only when that probability clears a threshold. This works better
because:
  - "will this specific setup work" has a clearer signal than "which way
    will the market go".
  - the base rate is informative (it's however often the heuristic is
    right, not ~50%).
  - precision and recall can be tuned independently of direction.
  - the secondary model never has to be right about direction on its
    own — only to recognise when the primary rule is in a bad spot.

The secondary target is 1 when the primary's implied direction agrees
with the SIGN of the realised return at the triple-barrier touch (Phase
0.1), restricted to bars where the primary actually expressed an
opinion (heuristic_predictions returns -1 for "no opinion", and those
bars carry no meta-label at all — there is nothing to be right or wrong
about).
===================================================================
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from validation import heuristic_predictions, purged_walk_forward  # noqa: E402


def meta_target(primary, realised_return):
    """
    Build the secondary target from the primary's calls and the
    triple-barrier realised return.

    Returns (meta_y, has_opinion):
      meta_y       1 where the primary's direction agreed with the sign
                   of the realised return, 0 otherwise. Undefined (0,
                   but see has_opinion) where the primary had no view.
      has_opinion  boolean mask — only these rows carry a real
                   meta-label and should be used for training/scoring.
    """
    primary = np.asarray(primary)
    realised_return = np.asarray(realised_return, dtype=float)
    has_opinion = primary >= 0

    meta_y = np.zeros(len(primary), dtype=int)
    meta_y[has_opinion & (primary == 1) & (realised_return > 0)] = 1
    meta_y[has_opinion & (primary == 0) & (realised_return < 0)] = 1
    return meta_y, has_opinion


def precision_recall_f1(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = float("nan")
    return precision, recall, f1


def _standardise_fit(X):
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-12] = 1.0
    return mean, scale


def strategy_return_bps(primary, realised_return, cost_bps, trade_mask=None):
    """
    Mean net return (bps) of following `primary`'s direction on every
    bar it has trade_mask True (defaults to every bar with an opinion),
    paying `cost_bps` round-trip on each trade. Used to compare "trade
    every primary signal" against "trade only when the secondary model
    clears its threshold" — the acceptance criterion the spec asks for.
    """
    primary = np.asarray(primary)
    realised_return = np.asarray(realised_return, dtype=float)
    has_opinion = primary >= 0
    if trade_mask is None:
        trade_mask = has_opinion
    else:
        trade_mask = trade_mask & has_opinion

    if trade_mask.sum() == 0:
        return float("nan"), 0

    direction = np.where(primary[trade_mask] == 1, 1.0, -1.0)
    gross = direction * realised_return[trade_mask]
    net = gross - (cost_bps / 10000.0)
    return float(np.nanmean(net) * 10000), int(trade_mask.sum())


def evaluate_meta_labeling(X, feature_names, realised_return, sample_weight=None,
                           n_splits=5, horizon=24, embargo=6, cost_bps=0.0,
                           threshold=0.55, C=0.2):
    """
    Purged walk-forward evaluation of the primary/secondary pair.

    For each fold: compute the primary's calls on the fold from
    `feature_names` alone (it's a fixed rule, not fitted), train the
    secondary logistic regression on TRAIN rows where the primary had an
    opinion, score it on TEST rows with the same restriction.

    Returns a dict with aggregated precision/recall/F1 for the secondary
    model, and the strategy's realised return (bps) with and without the
    P(correct) >= `threshold` filter — the acceptance criteria the spec
    asks for.
    """
    from sklearn.linear_model import LogisticRegression

    primary_all = heuristic_predictions(X, feature_names)

    fold_metrics = []
    n_folds = 0

    for train_idx, test_idx in purged_walk_forward(
        len(X), n_splits=n_splits, horizon=horizon, embargo=embargo,
    ):
        n_folds += 1
        meta_y, has_opinion = meta_target(primary_all, realised_return)

        tr_mask = has_opinion[train_idx]
        te_mask = has_opinion[test_idx]
        if tr_mask.sum() < 50 or te_mask.sum() < 20:
            continue

        Xtr = X[train_idx][tr_mask]
        ytr = meta_y[train_idx][tr_mask]
        wtr = sample_weight[train_idx][tr_mask] if sample_weight is not None else None
        Xte = X[test_idx][te_mask]
        yte = meta_y[test_idx][te_mask]

        if len(np.unique(ytr)) < 2:
            continue

        mean, scale = _standardise_fit(Xtr)
        Xtr_s = (Xtr - mean) / scale
        Xte_s = (Xte - mean) / scale

        model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
        model.fit(Xtr_s, ytr, sample_weight=wtr)
        p_correct = model.predict_proba(Xte_s)[:, 1]
        pred = (p_correct >= 0.5).astype(int)

        precision, recall, f1 = precision_recall_f1(yte, pred)

        primary_test = primary_all[test_idx][te_mask]
        ret_test = realised_return[test_idx][te_mask]
        unfiltered_bps, unfiltered_n = strategy_return_bps(primary_test, ret_test, cost_bps)
        filtered_mask = p_correct >= threshold
        filtered_bps, filtered_n = strategy_return_bps(
            primary_test, ret_test, cost_bps, trade_mask=filtered_mask
        )

        fold_metrics.append({
            "precision": precision, "recall": recall, "f1": f1,
            "unfiltered_bps": unfiltered_bps, "unfiltered_n": unfiltered_n,
            "filtered_bps": filtered_bps, "filtered_n": filtered_n,
            "coverage": float(filtered_mask.mean()) if len(filtered_mask) else float("nan"),
        })

    if not fold_metrics:
        return {"n_folds": n_folds, "folds": []}

    def mean_of(key):
        vals = [m[key] for m in fold_metrics if np.isfinite(m.get(key, np.nan))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "n_folds": n_folds,
        "folds": fold_metrics,
        "precision_mean": mean_of("precision"),
        "recall_mean": mean_of("recall"),
        "f1_mean": mean_of("f1"),
        "unfiltered_bps_mean": mean_of("unfiltered_bps"),
        "filtered_bps_mean": mean_of("filtered_bps"),
        "coverage_mean": mean_of("coverage"),
        "threshold": threshold,
    }


def fit_meta_model(X, feature_names, realised_return, sample_weight=None, C=0.2):
    """
    Refit the secondary model on ALL data (no held-out split) — the
    final export step, mirroring train.py's export_model() for the
    direction model. Returns (model, mean, scale, primary_all, meta_y,
    has_opinion) so the caller can both export and report training
    coverage.
    """
    from sklearn.linear_model import LogisticRegression

    primary_all = heuristic_predictions(X, feature_names)
    meta_y, has_opinion = meta_target(primary_all, realised_return)

    Xf = X[has_opinion]
    yf = meta_y[has_opinion]
    wf = sample_weight[has_opinion] if sample_weight is not None else None

    mean, scale = _standardise_fit(Xf)
    model = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")
    model.fit((Xf - mean) / scale, yf, sample_weight=wf)

    return model, mean, scale, primary_all, meta_y, has_opinion
