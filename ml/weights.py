"""
ml/weights.py
===================================================================
Sample weights by uniqueness (Phase 0.2 of PREDICTION_SPEC.md).

Overlapping labels break the IID assumption every classifier makes.
With a 24-bar vertical barrier, up to 24 consecutive samples share
outcome information — the model sees ~24x less independent data than
the row count suggests, and every significance test is correspondingly
wrong (López de Prado, AFML ch.4).

Three multiplicative components, combined into one `sample_weight`:

  1. UNIQUENESS   — mean(1/c_t) over the bars a label spans, where c_t
                     is how many concurrently-open labels touch bar t.
                     A label that overlaps nothing gets weight 1; one
                     that overlaps 23 others gets close to 1/24.
  2. RETURN ATTR  — |realised return| at touch. A sample that moved 4%
                     carries more information than one that moved 0.05%.
  3. TIME DECAY   — linear decay from 1.0 (newest) to ~0.5 (oldest).
                     Markets are non-stationary; old regimes should not
                     vote as loudly as recent ones.

All three are computed from information available at or before the
label's own touch time, so none of this introduces look-ahead.
===================================================================
"""

import numpy as np


def concurrency(n_bars, start_idx, end_idx):
    """
    Count how many labels are "open" (span this bar) at every bar.

    `start_idx[i]` / `end_idx[i]` are the bar indices a label spans,
    inclusive. Returns an (n_bars,) array c_t = number of labels whose
    [start, end] interval contains bar t.

    Implemented as a difference array so it is O(n_bars + n_labels)
    rather than O(n_bars * n_labels).
    """
    start_idx = np.asarray(start_idx, dtype=np.int64)
    end_idx = np.asarray(end_idx, dtype=np.int64)
    diff = np.zeros(n_bars + 1, dtype=np.float64)
    for s, e in zip(start_idx, end_idx):
        s = max(0, s)
        e = min(n_bars - 1, e)
        if e < s:
            continue
        diff[s] += 1.0
        diff[e + 1] -= 1.0
    return np.cumsum(diff[:n_bars])


def uniqueness_weights(n_bars, start_idx, end_idx):
    """
    Average uniqueness 1/c_t over each label's own span.

    Returns an (n_labels,) array, one weight per label, NOT normalised
    (the caller usually wants to combine this with other weight
    components before normalising to a mean of 1).
    """
    c = concurrency(n_bars, start_idx, end_idx)
    c = np.maximum(c, 1.0)  # a label's own bar is always "open"
    inv_c = 1.0 / c

    start_idx = np.asarray(start_idx, dtype=np.int64)
    end_idx = np.asarray(end_idx, dtype=np.int64)
    out = np.empty(len(start_idx), dtype=np.float64)
    for i, (s, e) in enumerate(zip(start_idx, end_idx)):
        s = max(0, s)
        e = min(n_bars - 1, e)
        if e < s:
            out[i] = 1.0
            continue
        out[i] = inv_c[s:e + 1].mean()
    return out


def time_decay_weights(n_labels, min_weight=0.5):
    """
    Linear decay from 1.0 (most recent / last row) to `min_weight`
    (oldest / first row). Labels are assumed ordered oldest-first, the
    convention used everywhere else in this pipeline.
    """
    if n_labels <= 1:
        return np.ones(n_labels)
    return np.linspace(min_weight, 1.0, n_labels)


def sample_weights(n_bars, start_idx, touch_idx, realised_return,
                   time_decay_min=0.5):
    """
    Combine uniqueness x |return| x time-decay into one sample_weight
    array, mean-normalised to 1.0 so it does not silently rescale the
    classifier's regularisation strength.

    `start_idx`   — bar index each label starts at (label i's own row).
    `touch_idx`   — bar index each label's barrier was touched (its end).
    `realised_return` — realised log return at touch, same length.

    Expect measured edge to drop once this is wired into .fit() — that
    drop is the correction of an illusion (rows that looked independent
    were mostly duplicates), not a regression.
    """
    start_idx = np.asarray(start_idx, dtype=np.int64)
    touch_idx = np.asarray(touch_idx, dtype=np.int64)
    realised_return = np.asarray(realised_return, dtype=np.float64)
    n = len(start_idx)
    if n == 0:
        return np.zeros(0)

    uniq = uniqueness_weights(n_bars, start_idx, touch_idx)
    ret_attr = np.abs(realised_return)
    ret_attr = np.nan_to_num(ret_attr, nan=0.0)
    # Guard against an all-zero return vector (e.g. synthetic edge cases)
    # collapsing every weight to zero.
    if ret_attr.sum() <= 0:
        ret_attr = np.ones(n)

    decay = time_decay_weights(n, min_weight=time_decay_min)

    w = uniq * ret_attr * decay
    w = np.clip(w, 0, None)
    mean_w = w.mean()
    if mean_w <= 0 or not np.isfinite(mean_w):
        return np.ones(n)
    return w / mean_w


def average_uniqueness(n_bars, start_idx, touch_idx):
    """
    Single summary number: mean sample uniqueness across all labels.
    ~1.0 means labels barely overlap (close to IID); ~1/horizon means
    heavy overlap (the model is seeing far fewer independent samples
    than the row count suggests). Useful to print once per training run.
    """
    u = uniqueness_weights(n_bars, start_idx, touch_idx)
    return float(u.mean()) if len(u) else float('nan')
