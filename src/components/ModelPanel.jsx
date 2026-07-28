/**
 * src/components/ModelPanel.jsx
 * -------------------------------------------------------------------
 * Shows the model's opinion beside the rule-based one.
 *
 * The design brief for this component is unusual: its job is to make
 * the model's measured performance impossible to ignore, not to make
 * the model look good. So the largest number on the panel is the
 * held-out EDGE, not the confidence. A user who glances at this should
 * come away knowing whether the model has demonstrated anything.
 *
 * Three states:
 *   no model      -> say so plainly, explain how to train one
 *   unvalidated   -> show the prediction struck through, with the reason
 *   validated     -> show it, with the measured edge alongside
 * -------------------------------------------------------------------
 */

import { signalColor, signalGlyph } from '../lib/helpers.js';

/* ------------------------------------------------------------------ */
/* Verdict styling                                                     */
/* ------------------------------------------------------------------ */

const VERDICT_STYLE = {
  PROMISING: { color: 'text-up', label: 'edge demonstrated' },
  WEAK: { color: 'text-amber', label: 'suggestive only' },
  EDGE_NO_PROFIT: { color: 'text-amber', label: 'edge but unprofitable' },
  NOISE: { color: 'text-down', label: 'fold-to-fold noise' },
  NO_EDGE: { color: 'text-down', label: 'no edge' },
  UNTRAINED: { color: 'text-ink-faint', label: 'not trained' },
  UNKNOWN: { color: 'text-ink-faint', label: 'unknown' },
};

function pp(value) {
  if (value == null || Number.isNaN(value)) return '—';
  const sign = value > 0 ? '+' : value < 0 ? '−' : '';
  return `${sign}${Math.abs(value * 100).toFixed(2)}pp`;
}

function pct(value, digits = 1) {
  return value == null || Number.isNaN(value) ? '—' : `${(value * 100).toFixed(digits)}%`;
}

/* ------------------------------------------------------------------ */
/* Panel                                                               */
/* ------------------------------------------------------------------ */

export default function ModelPanel({ model, ml, combined, heuristicSignal }) {
  /* ---- No model at all ---- */
  if (!model?.loaded) {
    return (
      <section className="panel px-4 py-3">
        <h2 className="label">Model</h2>
        <p className="mt-1.5 max-w-prose text-tick normal-case leading-relaxed text-ink-muted">
          {model?.error
            ? `Failed to load: ${model.error}`
            : 'No model trained yet — every signal on screen comes from the rule-based heuristic.'}
        </p>
        <p className="mt-2 text-micro normal-case text-ink-faint">
          Train one with{' '}
          <code className="text-amber">python3 ml/train.py --symbol BTC/USDT --export</code>.
          The pipeline will refuse to export a model that shows no edge.
        </p>
      </section>
    );
  }

  const wf = model.walkForward || {};
  const style = VERDICT_STYLE[model.verdict] || VERDICT_STYLE.UNKNOWN;
  const validated = ml?.isValidated;

  return (
    <section className="panel edge-top">
      <header className="flex flex-wrap items-baseline justify-between gap-3 border-b border-divider px-4 py-3">
        <div>
          <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
            Model {model.mode === 'meta_label' && <span className="text-ink-faint">· meta-labelling</span>}
          </h2>
          <p className="mt-0.5 text-micro normal-case text-ink-faint">
            Logistic regression · trained on {model.trainedOn || 'unknown'}
            {model.pooledWith?.length ? ` + ${model.pooledWith.join(', ')} (pooled)` : ''} ·{' '}
            {model.horizonBars ? `${model.horizonBars * 5}min vertical barrier` : 'unknown horizon'}
            {model.trainedRows ? ` · ${model.trainedRows.toLocaleString()} rows` : ''}
          </p>
          {model.mode === 'meta_label' && (
            <p className="mt-0.5 text-micro normal-case text-ink-faint">
              Predicts P(the rule-based signal is right), not direction — trades only when that
              clears {pct(model.metaLabelThreshold, 0)}.
            </p>
          )}
        </div>

        {/* The headline is the measured edge, deliberately — not confidence. */}
        <div className="text-right">
          <p className="label">Held-out edge</p>
          <p
            className={`tnum text-readout leading-none ${
              wf.edge > 0 ? 'text-up' : 'text-down'
            }`}
          >
            {pp(wf.edge)}
          </p>
          <p className={`text-micro uppercase ${style.color}`}>{style.label}</p>
        </div>
      </header>

      {/* ---- Measured performance ---- */}
      <div className="grid grid-cols-2 gap-px bg-hairline sm:grid-cols-4">
        {[
          { label: 'Accuracy', value: wf.accuracy != null ? `${(wf.accuracy * 100).toFixed(2)}%` : '—' },
          { label: 'Base rate', value: wf.baseRate != null ? `${(wf.baseRate * 100).toFixed(2)}%` : '—', note: 'score to beat' },
          {
            label: 'Folds positive',
            value: wf.folds ? `${wf.foldsPositive}/${wf.folds}` : '—',
            note: wf.edgeStd != null ? `±${(wf.edgeStd * 100).toFixed(2)}pp` : null,
          },
          {
            label: 'Net / trade',
            value: wf.netBps != null ? `${wf.netBps > 0 ? '+' : '−'}${Math.abs(wf.netBps).toFixed(1)}bps` : '—',
            note: model.costBps ? `after ${model.costBps}bps costs` : null,
            color: wf.netBps > 0 ? 'text-up' : 'text-down',
          },
        ].map((c) => (
          <div key={c.label} className="bg-panel px-3 py-2.5">
            <p className="label">{c.label}</p>
            <p className={`tnum mt-0.5 text-sm ${c.color || 'text-ink'}`}>{c.value}</p>
            {c.note && <p className="text-micro normal-case text-ink-faint">{c.note}</p>}
          </div>
        ))}
      </div>

      {/* ---- Calibration (Phase 1.3) ---- */}
      {model.calibration && (
        <div className="grid grid-cols-2 gap-px border-t border-hairline bg-hairline sm:grid-cols-2">
          {[
            { label: 'Calibration method', value: model.calibration.method || '—' },
            {
              label: 'ECE / MCE',
              value: `${pct(model.calibration.ece, 2)} / ${pct(model.calibration.mce, 2)}`,
              note: 'held-out fold · target ECE < 5%',
            },
          ].map((c) => (
            <div key={c.label} className="bg-panel px-3 py-2.5">
              <p className="label">{c.label}</p>
              <p className="tnum mt-0.5 text-sm text-ink">{c.value}</p>
              {c.note && <p className="text-micro normal-case text-ink-faint">{c.note}</p>}
            </div>
          ))}
        </div>
      )}

      {/* ---- Head-to-head ---- */}
      {ml?.available && (
        <div className="border-t border-hairline px-4 py-3">
          <p className="label mb-2">This bar</p>
          <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
            <div>
              <p className="text-micro uppercase text-ink-faint">Rules say</p>
              <p className={`font-sans text-sm font-semibold ${signalColor(heuristicSignal)}`}>
                {signalGlyph(heuristicSignal)} {heuristicSignal}
              </p>
            </div>

            <div>
              <p className="text-micro uppercase text-ink-faint">
                {ml.mode === 'meta_label' ? 'Model gates on' : 'Model says'}
              </p>
              <p
                className={`font-sans text-sm font-semibold ${signalColor(ml.signal)} ${
                  validated ? '' : 'line-through opacity-50'
                }`}
              >
                {signalGlyph(ml.signal)} {ml.signal}
              </p>
              <p className="tnum text-micro text-ink-faint">
                {ml.mode === 'meta_label'
                  ? `P(rule is right) ${(ml.pCorrect * 100).toFixed(1)}% ${
                      ml.meetsThreshold ? '≥' : '<'
                    } ${(ml.threshold * 100).toFixed(0)}% threshold`
                  : `P(up) ${(ml.probUp * 100).toFixed(1)}%`}
              </p>
            </div>

            <div>
              <p className="text-micro uppercase text-ink-faint">Displayed</p>
              <p className={`font-sans text-sm font-semibold ${signalColor(combined?.signal)}`}>
                {signalGlyph(combined?.signal)} {combined?.signal}
              </p>
              <p className="text-micro uppercase text-ink-faint">
                via {combined?.source}
              </p>
            </div>

            {combined?.agreement != null && (
              <div
                className={`border px-2 py-1 text-micro uppercase ${
                  combined.agreement
                    ? 'border-up/40 text-up'
                    : 'border-amber/40 text-amber'
                }`}
              >
                {combined.agreement ? 'agree' : 'disagree'}
              </div>
            )}
          </div>

          {combined?.note && (
            <p className="mt-3 max-w-prose text-micro normal-case leading-relaxed text-ink-muted">
              {combined.note}
            </p>
          )}

          {/* ---- Reliability for THIS prediction's bucket (Phase 1.4) ----
           * The checkable sentence: not "68% confident" in the abstract,
           * but "of the last N times it said this, it was right X% of
           * the time" — measured on a held-out fold, not asserted. */}
          {ml.reliability && (
            <div className="mt-3 border border-hairline bg-void/40 px-3 py-2.5">
              <p className="label">Reliability of this bucket</p>
              <p className="mt-1 text-tick normal-case leading-relaxed text-ink-muted">
                Of the last <strong className="text-ink">{ml.reliability.n}</strong> held-out
                predictions where the model said{' '}
                <strong className="text-ink">
                  {(ml.reliability.lo * 100).toFixed(0)}–{(ml.reliability.hi * 100).toFixed(0)}%
                </strong>
                , it was right{' '}
                <strong className="text-ink">
                  {(ml.reliability.observed_freq * 100).toFixed(1)}%
                </strong>{' '}
                of the time (predicted average was{' '}
                {(ml.reliability.mean_predicted * 100).toFixed(1)}%).
              </p>
            </div>
          )}
        </div>
      )}

      {/* ---- Confidence caveat ---- */}
      {ml?.available && !ml.confidenceIsMeaningful && (
        <p className="border-t border-hairline bg-void/40 px-4 py-2 text-micro normal-case leading-relaxed text-ink-muted">
          The model's high-confidence predictions were <strong>not</strong> measurably
          more accurate than its average ones on held-out data, so its confidence
          number does not rank anything. It is hidden here rather than shown as
          though it were informative.
        </p>
      )}

      {/* ---- Unvalidated warning ---- */}
      {ml?.available && !validated && (
        <p className="border-t border-amber/30 bg-amber-glow px-4 py-2.5 text-micro normal-case leading-relaxed text-amber">
          This model did not demonstrate an edge on held-out data (verdict{' '}
          {model.verdict}). Its prediction is shown struck through, for comparison
          only, and does not influence the displayed signal. That is the correct
          outcome for most attempts at this problem, not a bug.
        </p>
      )}
    </section>
  );
}
