
# The ML pipeline

> **Read this first:** for 5-minute crypto and forex direction, the
> overwhelmingly likely result is **no exploitable edge**. That is not a
> defect in this pipeline — it is the finding. This pipeline is built to
> tell you that clearly rather than hand you a number that looks good and
> isn't.

This pipeline implements `PREDICTION_SPEC.md` end to end: triple-barrier
labelling, sample weighting, meta-labelling, calibration, crypto
derivatives + cross-sectional features, volatility forecasting, CPCV/DSR/
PBO, and position sizing. The order below is the spec's own "Order of
work" — do the labels and calibration first, architecture last, because
that is where the wins (and the honest negative results) actually are.

## Set expectations before you start

Published attempts at short-horizon retail price prediction from OHLCV
data cluster around 50–53% directional accuracy, which after spreads and
fees is a losing strategy. Papers reporting 70%+ on this task almost
always contain one of the leaks described below.

So the honest framing of what you are about to do: you are running a
well-instrumented experiment that will probably return a negative result,
and the value is in trusting that answer. If a real edge existed in
14-period RSI on BTC 5-minute bars, it would have been arbitraged away
long before it reached a free API.

## Quick start

```bash
pip install -r ml/requirements.txt

# Confirm the harness is honest before trusting it on real data
python3 ml/train.py --synthetic random    # expect NO_EDGE
python3 ml/train.py --synthetic signal    # expect PROMISING
python3 ml/leakage_demo.py                # why purging matters

# Then real data
python3 ml/train.py --symbol BTC/USDT
python3 ml/train.py --symbol BTC/USDT --model logreg --export

# With the Phase 2/3 enhancements
python3 ml/train.py --symbol BTC/USDT --derivatives --pool ETH/USDT,SOL/USDT \
    --meta-label --calibrate both --cpcv --log-trial --export

python3 ml/parity.py                      # after ANY change to features
```

## Files

| File | Role |
|---|---|
| `features.py` | Feature engineering, triple-barrier labelling, look-ahead assertion |
| `weights.py` | Sample weights by uniqueness x return-attribution x time-decay |
| `meta_labeling.py` | Primary (heuristic) + secondary (P(correct)) pair |
| `volatility.py` | HAR-RV / GARCH(1,1) volatility forecasting, QLIKE loss |
| `sizing.py` | Bet sizing: calibrated-probability sizing, fractional Kelly, vol targeting, cost model |
| `validation.py` | Purged walk-forward + CPCV, baselines, metrics, calibration, DSR, PBO, verdict |
| `data.py` | Binance / Twelve Data loaders, derivatives fetch, cross-sectional alignment, synthetic generators |
| `train.py` | Orchestration: labels, weights, calibration, meta-labelling, pooling, DSR/CPCV/PBO reporting, JSON export |
| `parity.py` | Verifies Python and JS compute identical features, including context-gated ones |
| `leakage_demo.py` | Shows naive validation reporting a fake edge |

JS side: `shared/mlFeatures.js` (feature twin), `shared/mlModel.js`
(inference + reliability lookup + signal combination), `shared/model.json`
(exported weights), `shared/marketSources.js` (`fetchDerivativesContext()`).

## The invariants this pipeline is built to protect

1. **No feature reads the future.** `assert_no_lookahead()` recomputes
   every feature on a truncated series and fails if any value changes.
   Labels are the deliberate exception — a label is allowed to look
   forward, because that's what it's *for*. Triple-barrier's `sigma_t`
   barrier width is a labelling parameter, not a feature, so a HAR-RV/
   GARCH forecast fit on the whole series can size it without that being
   a leak (see `build_labels(sigma_override=...)`).
2. **Python and JS features agree to 1e-9.** `parity.py` gates it,
   including the context-gated features (derivatives, cross-sectional).
3. **Purge + embargo between train and test, always** — plain walk-forward
   for the headline number, CPCV for a distribution around it.
4. **The random-walk self-test must keep reporting no edge.**
   `python3 ml/train.py --synthetic random` must print `VERDICT: NO_EDGE`.
   If it ever reports a positive edge, the harness is leaking and every
   number in the project is void.
5. **Calibrators fit only on data the base model never saw** — a dedicated
   train / calib / test split, purged on both boundaries.
6. **Costs are never optimistic** — `cost_model()` in `sizing.py` grows
   with size (square-root impact), not flat.

## Reading the walk-forward output

```
model            acc    base     EDGE     ±sd  +folds    auc   net bps
logreg        51.20%  52.40%   -1.20pp   1.80   2/5    0.503     -1.40
```

- **acc** — raw accuracy. On its own this number is meaningless.
- **base** — the majority class share among decisive triple-barrier
  touches (timeouts are dropped from the direction-classification set).
- **EDGE** — `acc − base`. **This is the only headline that matters.**
- **±sd** — standard deviation of edge across folds. If SD exceeds the
  edge, you have noise.
- **+folds** — how many folds had a positive edge. With 5 folds, fewer
  than 4 is not a pattern.
- **auc** — ranking quality. 0.5 is random.
- **net bps** — return per trade after costs. The only metric in money.

Always compare against the `heuristic` row. If the ML does not beat the
hand-written rules in `shared/indicators.js`, ship the rules.

## Verdicts

| Verdict | Meaning | Exportable |
|---|---|---|
| `NO_EDGE` | Does not beat the base rate | No |
| `NOISE` | Positive edge in too few folds | No |
| `WEAK` | Consistent but within its own variance | Yes, with warnings |
| `EDGE_NO_PROFIT` | Real edge, negative after costs | Yes, with warnings |
| `PROMISING` | Cleared every check in the harness | Yes |

`train.py --export` refuses to write `model.json` for `NO_EDGE` or
`NOISE`. `--force-export` overrides it, and then the UI labels the model
unvalidated and ignores it when deciding what to display unless you also
set `ML_SHOW_UNVALIDATED=true`.

---

## Phase 0 — Labels

### Triple-barrier labelling

`build_labels()` sets three barriers per bar and labels by whichever is
touched first: upper (`+k1 x sigma_t`, default `k1=2.0`), lower
(`-k2 x sigma_t`, default `k2=1.0`), vertical (`--horizon`, default 24
bars). `sigma_t` is an EWMA of recent returns, so barrier width scales
with volatility. Timeouts (label 0) are dropped from the direction-
classification set — they're genuinely ambiguous, the same role the old
fixed-horizon cost band used to play.

```
python3 ml/train.py --symbol BTC/USDT --k1 2.0 --k2 1.0 --horizon 24
```

### Sample weights

`weights.py` computes uniqueness (mean `1/c_t` over the bars a label
spans), multiplies in `|realised return|` and a linear time-decay to
~0.5 at the oldest sample, and normalises to mean 1.0. Passed as
`sample_weight` to every `.fit()` call automatically — no flag needed.
**Expect measured edge to drop** once this is wired in; that drop is the
correction of an illusion (overlapping labels inflating the effective
sample size), not a regression.

### Meta-labelling

```
python3 ml/train.py --symbol BTC/USDT --meta-label
```

Reports precision/recall/F1 of the secondary model (predicting whether
the rule-based heuristic's call was right) plus the strategy's return
with and without the `P(correct) >= threshold` filter
(`--meta-threshold`, default 0.55). Combine with `--export` to ship the
meta-labelling model instead of the direction model — `shared/mlModel.js`
reads `model.meta.mode` and gates the heuristic's own call rather than
inventing a direction of its own.

---

## Phase 1 — Calibration

```
python3 ml/train.py --symbol BTC/USDT --calibrate both   # sigmoid vs isotonic, picks by Brier
```

Fits the base model on TRAIN, the calibrator on a purged CALIB fold the
base model never saw, and reports Brier decomposition (reliability /
resolution / uncertainty), ECE, MCE, and a full reliability table on a
third, disjoint TEST fold. Acceptance target: ECE < 0.05.

The exported `model.json` embeds the winning method's reliability table
under `meta.calibration`. `shared/mlModel.js`'s `reliabilityForProb()`
looks up the bucket the *current* prediction falls into, and
`ModelPanel.jsx` renders it as the checkable sentence the spec asks for:
"of the last N times the model said 65–70%, it was right X%."

Note what is and isn't exported: Platt/isotonic calibration is not
re-serialised into `model.json` (that would need a second model in JS).
Instead the raw logistic regression ships as-is (still a single JS dot
product) alongside the honestly-measured reliability table — the UI
shows real held-out reliability without JS needing to run a second model.

---

## Phase 2 — Data the price series doesn't contain

### Crypto derivatives (`--derivatives`)

```
python3 ml/train.py --symbol BTC/USDT --derivatives
```

Fetches funding rate, open interest, and top-trader long/short ratio from
Binance USDT-M futures (`ml/data.py::fetch_binance_derivatives`, JS twin
`shared/marketSources.js::fetchDerivativesContext()`), forward-fills them
onto the 5-minute bar grid, and feeds z-scored/rate-of-change versions in
as `funding_z`, `oi_roc_z`, `top_trader_z`. Live inference
(`netlify/functions/get-analysis.js`) fetches the same context for crypto
instruments automatically — no flag needed there, it's on whenever the
exported model has non-zero weight on those features and the fetch
succeeds; a failed fetch degrades those three features to 0, same as FX.

### Forex

Deliberately **not implemented**: interest-rate differentials, COT
positioning, and economic-calendar surprise all need a paid or
unreliable-free upstream, and shipping a shaky integration would be worse
than being honest that FX gets fewer features than crypto here. `volume_z`
and all three derivatives features are identically 0 for every FX pair —
check the exported feature weights for exactly that before trusting an
FX model.

### Cross-sectional (`--pool`)

```
python3 ml/train.py --symbol BTC/USDT --pool ETH/USDT,SOL/USDT
```

Pools BTC/ETH/SOL into one training set (3x the rows) and adds
`rel_strength_basket` (this symbol's return minus the other pooled
symbols' mean return) and `corr_to_btc` (60-bar rolling correlation to
BTC). Walk-forward purging stays correct per-symbol: each fold's test
window is purged against *that symbol's own* history only, then trained
on that symbol's purged train rows **union** every other pooled symbol's
full history — a different instrument at the same timestamp isn't a
temporal leak, it's a contemporaneous feature. Live inference pools the
same way, fetching all three crypto instruments' candles once per request
and sharing them.

---

## Phase 3 — Volatility (`ml/volatility.py`)

```python
from volatility import fit_har_rv, fit_garch
fit = fit_har_rv(close)           # no extra dependency
fit = fit_garch(close)            # needs `pip install arch`
```

Both fit on a chronological train split and report QLIKE (not MSE — MSE
is dominated by the handful of largest observations) on a held-out tail.
`--vol-model har|garch` swaps the forecast in as the triple-barrier
`sigma_t` (`build_labels(sigma_override=...)`) instead of the built-in
EWMA — this does not reintroduce look-ahead, because barrier width is a
labelling parameter, not a model-facing feature (see invariant #1).
`vol_expansion_signal()` flags low-vol-about-to-expand bars, the setup
the spec calls the best risk/reward — available for further use, not
currently wired into the live dashboard signal.

---

## Phase 4 — Validation that survives contact with reality

- **`--cpcv`** — Combinatorial Purged CV: splits into 6 groups, evaluates
  all `C(6,2)=15` test-group combinations, reports a 90% range around the
  edge instead of one point estimate.
- **`--log-trial`** — appends this run's per-fold Sharpe to
  `ml/cache/trials.jsonl` (gitignored) and reports the Deflated Sharpe
  Ratio against *every trial ever logged*, correcting for how many
  configurations you've tried in total. Few logged trials -> DSR isn't
  informative yet; log every run you take seriously.
- **PBO** — reported automatically whenever `--model both` is compared,
  via CSCV over the folds already collected. Above ~0.5 means picking the
  better of logreg/gbm here is noise-fitting, not a real difference.
- **One-shot holdout** — not automated (by design: automating it would
  make it trivial to look twice, which defeats the point). Reserve the
  most recent 20% of a symbol's history yourself, don't touch it during
  development, and evaluate exactly once when you believe you're done.

---

## Phase 5 — Turning an edge into a result (`ml/sizing.py`)

```python
from sizing import combined_position_size, net_return_after_costs
size = combined_position_size(p_calibrated, forecast_vol, target_vol)
net = net_return_after_costs(gross_return, size)
```

`size_from_probability()` (z-statistic through the normal CDF),
`fractional_kelly()` (0.25–0.5x, never full Kelly on estimated
parameters), `vol_target_size()` (size inversely to forecast vol), and
`cost_model()` (fees + half-spread + square-root market impact). Not
wired into the live dashboard — TrendPulse displays signals, not managed
positions — but available for anyone taking this past paper-trading.

---

## How the app uses the model

`combineSignals()` in `shared/mlModel.js`:

- **Model not validated** → heuristic decides. An unvalidated model is
  not evidence.
- **Validated and agrees** → confidence rises by 8 points, not more. Both
  methods are built from the same indicators, so agreement is one
  confirmation rather than two.
- **Validated and disagrees** → **NEUTRAL**. Disagreement means the setup
  is ambiguous, and that is information worth keeping rather than
  resolving by picking a side.
- **Heuristic NEUTRAL, model has a view** → model's view, capped at 70%.

In `meta_label` mode, `ml.signal` is already the heuristic's own call
(gated to NEUTRAL when `P(correct)` misses the threshold), so this same
agree/disagree machinery works unmodified: a gated-out call reads as the
model "disagreeing" with the heuristic, which correctly collapses to
NEUTRAL.

Nothing here is averaged. Averaging two directional calls produces a
number that looks decisive and means nothing.

## Feature reference

25 features, all stationary — ratios, log returns, bounded oscillators,
z-scores. Raw price is never a feature.

| Feature | Notes |
|---|---|
| `rsi_14_c`, `rsi_9_c` | RSI/100 − 0.5 |
| `rsi_slope` | 3-bar RSI change |
| `log_p_sma20`, `log_p_sma50` | log(price/SMA) |
| `bb_pctb_c`, `bb_bandwidth` | Bollinger position and width |
| `log_p_vwap` | log(price/VWAP) |
| `ret_1/3/6/12` | Log returns |
| `vol_12`, `vol_ratio` | Realised vol and its expansion |
| `atr_norm` | ATR/price |
| `volume_z` | **Always 0 for spot forex** |
| `macd_hist_norm` | MACD histogram / price |
| `hour_sin`, `hour_cos` | Cyclical hour — real for FX sessions, spurious for 24/7 crypto |
| `adx_norm` | ADX(14)/100 — trend-strength regime signal (Phase 6) |
| `rel_strength_basket` | Return vs the pooled-symbol basket — **0 unless `--pool`** |
| `corr_to_btc` | 60-bar rolling corr to BTC — **0 unless `--pool`, and 0 for BTC itself** |
| `funding_z` | Perp funding rate, z-scored — **0 unless `--derivatives` (crypto only)** |
| `oi_roc_z` | Open-interest ROC, z-scored — **0 unless `--derivatives`** |
| `top_trader_z` | Top-trader long/short ratio, z-scored — **0 unless `--derivatives`** |

Read the exported feature weights with suspicion on three axes: `hour_*`
dominating on a 24/7 crypto pair, `volume_z` carrying weight on FX, or
any context-gated feature carrying weight on a model trained *without*
that context (it should be exactly 0 and contribute nothing).

## Costs

`DEFAULT_COST_BPS` in `train.py`: crypto 12bps, forex 3bps round trip.
These set the triple-barrier scale implicitly (via `k1`/`k2` x `sigma_t`)
and the net-return metric explicitly. Set them to your real costs.
Understating them is the most common way a backtest manufactures an edge
that evaporates the moment you trade — `sizing.py`'s `cost_model()` grows
with position size for the same reason.
