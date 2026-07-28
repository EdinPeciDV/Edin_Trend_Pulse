"""
ml/parity.py
===================================================================
Verify that shared/mlFeatures.js and ml/features.py compute the same
numbers — including the context-gated features (Phase 2.1 derivatives,
Phase 2.3 cross-sectional) — and that shared/mlModel.js reproduces
sklearn's predictions.

WHY THIS EXISTS. The model is trained in Python and executed in
JavaScript. If the two feature implementations drift by even a small
amount, the model receives inputs from a different distribution than it
was fitted on. There is no exception, no crash, no log line — the
predictions just quietly become noise while the UI keeps showing a
confident-looking percentage.

This test is the only thing that catches that. Run it after ANY change
to either features file:

    python3 ml/parity.py

Exits non-zero on any mismatch, so it can gate CI.
===================================================================
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from features import FEATURE_NAMES, build_features, build_labels
import data as data_mod

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TOLERANCE = 1e-9


def run_js_features(candles, context=None):
    """Invoke the JS implementation via node and return its matrix."""
    payload = [
        {
            "openTime": float(candles["open_time"][i]),
            "open": float(candles["open"][i]),
            "high": float(candles["high"][i]),
            "low": float(candles["low"][i]),
            "close": float(candles["close"][i]),
            "volume": float(candles["volume"][i]),
        }
        for i in range(len(candles["close"]))
    ]

    # Python's context uses snake_case keys (basket_ret, btc_ret,
    # funding_rate, ...); JS's uses camelCase (basketRet, btcRet,
    # fundingRate, ...). Translate here so both sides can be driven from
    # the same Python-side test fixtures.
    key_map = {
        "basket_ret": "basketRet", "btc_ret": "btcRet",
        "funding_rate": "fundingRate", "open_interest": "openInterest",
        "top_trader_ratio": "topTraderRatio",
    }
    js_context = None
    if context:
        js_context = {
            key_map[k]: (v.tolist() if v is not None else None)
            for k, v in context.items() if k in key_map
        }

    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, "candles.json")
        with open(data_path, "w") as f:
            json.dump(payload, f)
        ctx_path = os.path.join(tmp, "context.json")
        with open(ctx_path, "w") as f:
            json.dump(js_context, f)

        script = os.path.join(tmp, "run.mjs")
        with open(script, "w") as f:
            f.write(f"""
import {{ readFileSync }} from 'node:fs';
import {{ pathToFileURL }} from 'node:url';
import path from 'node:path';
const {{ buildFeatureMatrix }} = await import(pathToFileURL(path.join({json.dumps(ROOT)}, 'shared', 'mlFeatures.js')));
const candles = JSON.parse(readFileSync({json.dumps(data_path)}, 'utf8'));
const context = JSON.parse(readFileSync({json.dumps(ctx_path)}, 'utf8'));
const {{ rows, valid, names }} = buildFeatureMatrix(candles, context);
process.stdout.write(JSON.stringify({{ rows, valid, names }}));
""")

        proc = subprocess.run(
            ["node", script], capture_output=True, text=True, cwd=ROOT
        )
        if proc.returncode != 0:
            raise RuntimeError(f"node failed:\n{proc.stderr[:2000]}")
        return json.loads(proc.stdout)


def run_js_inference(model_path, feature_rows):
    """Invoke shared/mlModel.js predictProba on given feature rows."""
    with tempfile.TemporaryDirectory() as tmp:
        rows_path = os.path.join(tmp, "rows.json")
        with open(rows_path, "w") as f:
            json.dump(feature_rows, f)

        script = os.path.join(tmp, "infer.mjs")
        with open(script, "w") as f:
            f.write(f"""
import {{ readFileSync }} from 'node:fs';
import {{ pathToFileURL }} from 'node:url';
import path from 'node:path';
const {{ loadModel, predictProba }} = await import(pathToFileURL(path.join({json.dumps(ROOT)}, 'shared', 'mlModel.js')));
const model = loadModel(JSON.parse(readFileSync({json.dumps(model_path)}, 'utf8')));
const rows = JSON.parse(readFileSync({json.dumps(rows_path)}, 'utf8'));
process.stdout.write(JSON.stringify(rows.map(r => predictProba(model, r))));
""")

        proc = subprocess.run(["node", script], capture_output=True, text=True, cwd=ROOT)
        if proc.returncode != 0:
            raise RuntimeError(f"node inference failed:\n{proc.stderr[:2000]}")
        return json.loads(proc.stdout)


def compare_features(candles, label, context=None):
    print(f"\n--- {label} ({len(candles['close'])} bars) ---")

    X_py, valid_py = build_features(candles, context=context)
    js = run_js_features(candles, context=context)
    X_js = np.array(js["rows"], dtype=float)
    valid_js = np.array(js["valid"], dtype=bool)

    if js["names"] != FEATURE_NAMES:
        print("  FAIL: feature name lists differ")
        print(f"    python: {FEATURE_NAMES}")
        print(f"    js    : {js['names']}")
        return False

    if X_js.shape != X_py.shape:
        print(f"  FAIL: shape mismatch python {X_py.shape} vs js {X_js.shape}")
        return False

    # Compare only rows both consider valid; warmup NaNs are expected.
    both = valid_py & valid_js
    if valid_py.sum() != valid_js.sum():
        print(f"  WARN: valid-row counts differ — python {valid_py.sum()}, "
              f"js {valid_js.sum()}")

    if both.sum() == 0:
        print("  FAIL: no jointly-valid rows to compare")
        return False

    diff = np.abs(X_py[both] - X_js[both])
    worst_per_feature = diff.max(axis=0)
    ok = True

    print(f"  comparing {int(both.sum())} valid rows x {len(FEATURE_NAMES)} features")
    for j, name in enumerate(FEATURE_NAMES):
        w = worst_per_feature[j]
        if w > TOLERANCE:
            print(f"    MISMATCH {name:<22} max abs diff {w:.3e}")
            ok = False

    if ok:
        print(f"  PASS  max abs diff across all features: "
              f"{worst_per_feature.max():.3e} (tolerance {TOLERANCE:.0e})")
    return ok


def compare_inference(candles, label):
    """Train a throwaway model, then check JS reproduces sklearn's output."""
    from sklearn.linear_model import LogisticRegression

    print(f"\n--- inference parity: {label} ---")

    X, valid = build_features(candles)
    tb_label, _, _, labelable = build_labels(candles["close"], vertical=24)
    keep = valid & labelable & (tb_label != 0)
    Xk, yk = X[keep], (tb_label[keep] == 1).astype(int)
    if len(np.unique(yk)) < 2 or len(Xk) < 200:
        print("  SKIP: not enough labelled data")
        return True

    mean, scale = Xk.mean(axis=0), Xk.std(axis=0)
    scale[scale < 1e-12] = 1.0
    model = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced")
    model.fit((Xk - mean) / scale, yk)

    payload = {
        "schema_version": 2,
        "kind": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "meta": {"symbol": label, "verdict": "PARITY_TEST"},
    }

    with tempfile.TemporaryDirectory() as tmp:
        mp = os.path.join(tmp, "model.json")
        with open(mp, "w") as f:
            json.dump(payload, f)

        sample = Xk[-200:]
        py_prob = model.predict_proba((sample - mean) / scale)[:, 1]
        js_prob = np.array(run_js_inference(mp, sample.tolist()), dtype=float)

    if np.isnan(js_prob).any():
        print(f"  FAIL: js returned null for {int(np.isnan(js_prob).sum())} rows")
        return False

    d = np.abs(py_prob - js_prob).max()
    if d > 1e-12:
        print(f"  FAIL: max prob diff {d:.3e}")
        return False
    print(f"  PASS  sklearn vs JS max prob diff: {d:.3e} over {len(sample)} rows")
    return True


def _synthetic_context(n, seed=0):
    """Deterministic fake derivatives + cross-sectional series to drive
    the context-gated feature paths through parity, without a network
    call. Values are dimensionally plausible, not realistic — parity
    only cares that both languages transform them identically."""
    rng = np.random.default_rng(seed)
    return {
        "basket_ret": rng.normal(0, 0.001, n),
        "btc_ret": rng.normal(0, 0.001, n),
        "funding_rate": rng.normal(0.0001, 0.00005, n),
        "open_interest": np.abs(rng.normal(1_000_000, 50_000, n)),
        "top_trader_ratio": np.abs(rng.normal(1.5, 0.3, n)),
    }


def main():
    print("=" * 70)
    print("FEATURE PARITY: ml/features.py  <->  shared/mlFeatures.js")
    print("=" * 70)

    cases = [
        (data_mod.synthetic_random_walk(n=1200, seed=1), "random walk, with volume"),
        (data_mod.synthetic_with_signal(n=1200, seed=2), "mean-reverting, with volume"),
    ]

    # A zero-volume case, to exercise the FX fallback paths in both
    # implementations (VWAP unweighted, volume_z forced to 0).
    fx = data_mod.synthetic_random_walk(n=1200, start_price=1.085, vol=0.0004, seed=3)
    fx["volume"] = np.zeros_like(fx["volume"])
    cases.append((fx, "forex-like, ZERO volume"))

    ok = True
    for candles, label in cases:
        ok &= compare_features(candles, label)

    # Context-gated features (Phase 2.1 derivatives, Phase 2.3
    # cross-sectional) — exercised separately since they are 0 unless a
    # context is supplied, and the highest-risk new parity surface.
    ctx_candles, _ = cases[0]
    context = _synthetic_context(len(ctx_candles["close"]), seed=11)
    ok &= compare_features(ctx_candles, "with derivatives + cross-sectional context",
                          context=context)

    # Partial context — only derivatives, no cross-sectional — to catch
    # a bug where one branch's "context missing -> 0" fallback breaks
    # when SOME keys are present and others are not.
    partial = {"funding_rate": context["funding_rate"],
              "open_interest": context["open_interest"]}
    ok &= compare_features(ctx_candles, "with PARTIAL context (derivatives only)",
                          context=partial)

    for candles, label in cases[:1]:
        ok &= compare_inference(candles, label)

    print("\n" + "=" * 70)
    if ok:
        print("ALL PARITY CHECKS PASSED — Python and JS agree.")
    else:
        print("PARITY FAILED. Do not deploy: JS inference would receive")
        print("features from a different distribution than training used.")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
