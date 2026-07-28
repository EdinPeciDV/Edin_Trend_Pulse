/**
 * shared/mlModel.js
 * -------------------------------------------------------------------
 * Pure-JS inference for the exported logistic regression, plus the
 * logic for combining the ML opinion with the rule-based heuristic.
 *
 * Why logistic regression and not something bigger: inference is a
 * standardise-then-dot-product, which is ~15 lines and provably
 * identical to what sklearn computes. There is no tree traversal or
 * matrix library to reimplement, so JS and Python cannot silently
 * disagree. On noisy 5-minute labels a regularised linear model also
 * tends to match boosted trees anyway — train.py prints the comparison
 * so you can check that claim on your own data rather than trust it.
 *
 * If no model.json is present, everything here degrades to null and the
 * app runs on the heuristic alone. A missing model must never break the
 * dashboard.
 * -------------------------------------------------------------------
 */

import { FEATURE_NAMES, latestFeatureVector } from './mlFeatures.js';

/* ------------------------------------------------------------------ */
/* Loading                                                             */
/* ------------------------------------------------------------------ */

/**
 * Validate and wrap a parsed model.json.
 *
 * The feature-name check is the important part: if the model was
 * trained on a different feature set or order than mlFeatures.js now
 * produces, inference would read the wrong number into every
 * coefficient. That must be a loud failure, not a silent one.
 */
export function loadModel(json) {
  if (!json || typeof json !== 'object') return null;

  // A placeholder model.json ships in the repo so that the static import
  // in the serverless functions always resolves, even before you have
  // trained anything. 'absent' means "no model yet" and is not an error:
  // the app runs on the heuristic alone.
  if (json.kind === 'absent') return null;

  if (json.kind !== 'logistic_regression') {
    throw new Error(
      `model.json has kind "${json.kind}" — only logistic_regression can be ` +
        `run in JS. Re-export with: python3 ml/train.py --model logreg --export`
    );
  }

  const names = json.feature_names;
  if (!Array.isArray(names) || names.length !== FEATURE_NAMES.length) {
    throw new Error(
      `model.json expects ${names?.length} features but mlFeatures.js produces ` +
        `${FEATURE_NAMES.length}. Retrain after changing the feature set.`
    );
  }
  for (let i = 0; i < names.length; i++) {
    if (names[i] !== FEATURE_NAMES[i]) {
      throw new Error(
        `Feature order mismatch at index ${i}: model.json has "${names[i]}", ` +
          `mlFeatures.js has "${FEATURE_NAMES[i]}". Retrain — do not reorder features.`
      );
    }
  }
  if (
    json.coef?.length !== names.length ||
    json.mean?.length !== names.length ||
    json.scale?.length !== names.length
  ) {
    throw new Error('model.json is malformed: coef/mean/scale length mismatch.');
  }

  return json;
}

/* ------------------------------------------------------------------ */
/* Inference                                                           */
/* ------------------------------------------------------------------ */

function sigmoid(z) {
  // Numerically stable for large |z|.
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const e = Math.exp(z);
  return e / (1 + e);
}

/**
 * P(up) for a single feature vector.
 * Mirrors sklearn exactly: standardise with the training mean/scale,
 * then apply the linear decision function.
 */
export function predictProba(model, features) {
  if (!model || !Array.isArray(features)) return null;
  if (features.length !== model.coef.length) return null;

  let z = model.intercept;
  for (let i = 0; i < features.length; i++) {
    if (!Number.isFinite(features[i])) return null;
    z += ((features[i] - model.mean[i]) / model.scale[i]) * model.coef[i];
  }
  return sigmoid(z);
}

/**
 * Look up the reliability-table bucket a given probability falls into
 * (Phase 1.4). model.meta.calibration.reliability_table is written by
 * ml/train.py's report_calibration() — predicted-vs-observed on a fresh
 * held-out fold the calibrator never saw either. Returns null if the
 * model was exported without a calibration report (--calibrate off) or
 * `prob` doesn't land in any recorded bucket.
 */
export function reliabilityForProb(model, prob) {
  const table = model?.meta?.calibration?.reliability_table;
  if (!Array.isArray(table) || !Number.isFinite(prob)) return null;
  const row = table.find((r) => prob >= r.lo && (prob < r.hi || (r.hi === 1 && prob <= 1)));
  return row || null;
}

/**
 * Run the model over candles and return a structured opinion.
 *
 * Two modes, read from model.meta.mode (default 'direction'):
 *
 *   'direction'   the model predicts P(up) directly. `confidence` is
 *                 distance from the 50% decision boundary, mapped to
 *                 0-100 — how far from indifferent the model is, NOT a
 *                 probability of being right (that's what the
 *                 reliability table is for). train.py reports
 *                 `top_decile_acc` so `confidenceIsMeaningful` can say
 *                 whether that number ranks anything at all.
 *
 *   'meta_label'  (Phase 0.3) the model predicts P(the rule-based
 *                 heuristic's call is correct), not a direction. Needs
 *                 `primarySignal` — the heuristic's own UP/DOWN/NEUTRAL
 *                 for this bar — to know what it's gating. Trades only
 *                 when P(correct) clears model.meta.meta_label_threshold;
 *                 below it, the displayed signal degrades to NEUTRAL
 *                 ("the setup doesn't look like one where the rule is
 *                 reliable right now"), not a direction of its own.
 *
 * @param {object} model
 * @param {Array} candles
 * @param {{context?: object, primarySignal?: string}} [opts]
 */
export function predictFromCandles(model, candles, opts = {}) {
  if (!model) return null;
  const { context = null, primarySignal = null } = opts;

  const features = latestFeatureVector(candles, context);
  if (!features) {
    return {
      available: false,
      reason: 'Not enough history for the model (needs 60+ bars past warmup).',
    };
  }

  const prob = predictProba(model, features);
  if (prob == null) {
    return { available: false, reason: 'Feature vector was invalid.' };
  }

  const mode = model.meta?.mode === 'meta_label' ? 'meta_label' : 'direction';
  const wf = model.meta?.walk_forward || {};
  const edge = wf.edge_mean;
  const topDecile = wf.top_decile_acc_mean;
  const accuracy = wf.accuracy_mean;
  const isValidated = Number.isFinite(edge) && edge > 0 &&
    ['PROMISING', 'WEAK', 'EDGE_NO_PROFIT'].includes(model.meta?.verdict);

  const common = {
    horizonBars: model.meta?.horizon_bars ?? null,
    verdict: model.meta?.verdict ?? 'UNKNOWN',
    trainedOn: model.meta?.symbol ?? null,
    measuredEdge: Number.isFinite(edge) ? edge : null,
    measuredAccuracy: Number.isFinite(accuracy) ? accuracy : null,
    isValidated,
    reliability: reliabilityForProb(model, prob),
  };

  if (mode === 'meta_label') {
    if (!primarySignal || primarySignal === 'NEUTRAL') {
      return {
        available: false,
        reason: 'Meta-labelling model has nothing to gate — the rule-based heuristic has no opinion on this bar.',
      };
    }
    const threshold = model.meta?.meta_label_threshold ?? 0.55;
    const meetsThreshold = prob >= threshold;
    return {
      available: true,
      mode,
      signal: meetsThreshold ? primarySignal : 'NEUTRAL',
      primarySignal,
      pCorrect: prob,
      meetsThreshold,
      threshold,
      confidence: Math.round(prob * 100),
      confidenceIsMeaningful: true, // pCorrect IS the calibrated quantity by construction
      ...common,
    };
  }

  const signal = prob >= 0.5 ? 'UP' : 'DOWN';
  const confidence = Math.round(Math.abs(prob - 0.5) * 200);

  return {
    available: true,
    mode,
    signal,
    probUp: prob,
    confidence,
    // Is the confidence number worth showing? Only if high-confidence
    // predictions were measurably more accurate than average ones.
    confidenceIsMeaningful:
      Number.isFinite(topDecile) && Number.isFinite(accuracy)
        ? topDecile > accuracy + 0.02
        : false,
    ...common,
  };
}

/* ------------------------------------------------------------------ */
/* Combining ML with the heuristic                                     */
/* ------------------------------------------------------------------ */

/**
 * Combine the two opinions into one displayed view.
 *
 * The rule is deliberately conservative, and the reasoning matters:
 *
 *   - If the model has NOT demonstrated an edge on held-out data, the
 *     heuristic decides. An unvalidated model is not evidence, and
 *     letting it override a rule you can read and reason about would be
 *     a downgrade dressed up as an upgrade.
 *   - If the model IS validated and they AGREE, confidence rises — but
 *     only modestly. Two correlated methods built from the same
 *     indicators agreeing is weaker evidence than it feels like.
 *   - If the model IS validated and they DISAGREE, the output is
 *     NEUTRAL. Disagreement is real information: it means the setup is
 *     ambiguous. Silently picking a side would throw that away.
 *
 * Nothing here is averaged. Averaging two directional calls produces a
 * number that looks decisive while meaning nothing.
 */
export function combineSignals(heuristic, ml) {
  const base = {
    signal: heuristic.signal,
    confidence: heuristic.confidence,
    source: 'heuristic',
    agreement: null,
    mlAvailable: Boolean(ml?.available),
  };

  if (!ml?.available) {
    return { ...base, note: 'Model not loaded — rule-based signal only.' };
  }

  if (!ml.isValidated) {
    return {
      ...base,
      agreement: ml.signal === heuristic.signal,
      note:
        `Model shown for comparison only: it has no demonstrated edge on ` +
        `held-out data (verdict ${ml.verdict}). The rule-based signal decides.`,
    };
  }

  // Validated model from here on.
  if (heuristic.signal === 'NEUTRAL') {
    return {
      signal: ml.signal,
      confidence: Math.min(ml.confidence, 70),
      source: 'ml',
      agreement: null,
      mlAvailable: true,
      note: 'Rules had no opinion; the validated model supplied this one.',
    };
  }

  if (ml.signal === heuristic.signal) {
    return {
      signal: heuristic.signal,
      confidence: Math.min(95, Math.round(heuristic.confidence + 8)),
      source: 'both',
      agreement: true,
      mlAvailable: true,
      note: 'Rules and model agree. Both derive from the same indicators, so treat this as one confirmation, not two.',
    };
  }

  return {
    signal: 'NEUTRAL',
    confidence: 35,
    source: 'conflict',
    agreement: false,
    mlAvailable: true,
    note:
      `Rules say ${heuristic.signal}, model says ${ml.signal}. Held as NEUTRAL — ` +
      `a genuine disagreement means the setup is ambiguous.`,
  };
}
