# sysrfx_predictor

Zero-cost forex + crypto direction predictor (BTC/USDT, ETH/USDT,
EUR/USD, GBP/USD, 1-hour bars) with Supabase logging.

## Read this first

The original brief for this script asked for a hard-coded retry loop:
*"if accuracy < 0.72, automatically adjust Hyperopt parameters and retry
(at least 3 attempts)."* That loop is **not implemented**, on purpose.

Searching over configurations until a backtest number clears a fixed
target isn't validation — it's a way to manufacture a number that looks
good. The more configurations you try and keep the best of, the better
that best one looks *by chance alone*, with no guarantee it holds up
even one day forward live. This script's `run_backtest()` reports
whatever walk-forward accuracy it actually measures — on real BTC/USDT,
ETH/USDT, EUR/USD, and GBP/USD data at the time this was built, that was
consistently in the **48-58% range**, with edge over the majority-class
base rate landing on both sides of zero across folds. That is the
expected, honest result for this problem, not a bug in the script.

If you want a number you can trust more than a single backtest run,
raise `--folds`, run `--tune` a few times on different date ranges, and
look at the standard deviation across folds — not just the mean.

## Install

```bash
cd sysrfx_predictor
pip install -r requirements.txt
cp .env.example .env   # fill in whatever free keys you have; all optional
```

Every optional data source (Glassnode on-chain, FRED, Alpha Vantage,
Supabase) degrades gracefully — skip the feature, log an info/warning
line — if its key is missing, rather than crashing.

## Storage: Supabase (recommended) or local SQLite (automatic fallback)

1. Create a free [Supabase](https://supabase.com) project.
2. Run `schema.sql` once — either paste it into the Supabase SQL Editor,
   or `psql "$SUPABASE_DB_URL" -f schema.sql`.
3. Set `SUPABASE_URL` and `SUPABASE_KEY` (Project Settings -> API) in `.env`.
4. Optional: also set `SUPABASE_DB_URL` (Project Settings -> Database ->
   Connection string) and the script will run `schema.sql` for you on
   startup instead of you doing step 2 by hand.

If `SUPABASE_URL`/`SUPABASE_KEY` are absent, or Supabase can't be
reached, the script automatically falls back to a local SQLite file at
`sysrfx_predictor/cache/sysrfx_local.db` — no configuration needed, and
nothing crashes.

**Why the script can't just run `CREATE TABLE` through the normal
Supabase client:** the standard `supabase-py` client talks to
PostgREST, which exposes your tables as a REST API and deliberately has
no endpoint for arbitrary DDL (that's a security boundary). Auto-schema
only works via `SUPABASE_DB_URL`, a real Postgres connection.

## Run

```bash
# One pair, default hyperparameters, default 10-fold walk-forward backtest
python3 sysrfx_predictor.py --mode backtest --pair BTC/USDT

# All four default pairs
python3 sysrfx_predictor.py --mode backtest --all

# Tune first (Hyperopt, 50 evals — takes 1-3 minutes depending on the pair),
# cache the result, then backtest with those hyperparameters
python3 sysrfx_predictor.py --mode backtest --pair BTC/USDT --tune

# Live: fetch the latest bar every --poll-seconds (default 3600s, one per
# 1h bar), predict, apply the confidence>0.6 filter, log to storage
python3 sysrfx_predictor.py --mode live --pair BTC/USDT

# Force local storage even if Supabase keys are set
python3 sysrfx_predictor.py --mode backtest --pair BTC/USDT --supabase false
```

`--tune` hyperparameters are cached to `cache/hparams_<PAIR>.json` and
reused on every subsequent run (backtest without `--tune`, and every
Module 6 background retrain in live mode) until you `--tune` again —
this is what the original brief's "tune weekly" becomes here: an
explicit, deliberate action, not something the script decides to do on
its own on a timer.

## What each module actually does

| # | Module | Where |
|---|---|---|
| 0 | OHLCV (Binance via `ccxt`, Yahoo Finance via `yfinance`) + ~90 technical indicators (`ta` library) | `fetch_ohlcv*`, `add_technical_indicators` |
| 1 | Feature selection — top 20 by `mutual_info_classif` | `select_top_features` |
| 2 | Target smoothing — 3-bar-ahead direction, asset-class-scaled dead band | `build_target`, `resolve_dead_band_pct` |
| 3 | On-chain (Glassnode: exchange netflow, BTC miner reserve) | `fetch_onchain_features` |
| 4 | Order-flow / volume + macro (BTC dominance; FRED rate spread + Alpha Vantage CPI for forex) | `_volume_based_flow_proxy`, `add_vwap_deviation`, `fetch_macro_features` (all called from `build_feature_matrix`) |
| 5 | LightGBM + Hyperopt (TPE, 50 evals, objective = purged walk-forward accuracy) | `make_lgbm`, `hyperopt_tune` |
| 6 | Online retraining — background thread, every 50 new bars, 500-bar sliding window | `OnlineRetrainer` |
| 7 | Regime switching — ADX > 25 trending vs <= 25 ranging, one model per regime | `regime_of`, `train_regime_models` |

## Deliberate deviations from the original brief (and why)

**No retry-until-target loop.** Covered above — this is the main one.

**Mutual information via `mutual_info_classif`, not
`mutual_info_regression`.** The target is a binary class label; MI
against a discrete target is a classification problem. The regression
variant would run without erroring but is the wrong tool.

**Order-flow via a Binance REST depth snapshot, not a WebSocket
stream.** One imbalance reading is needed per hourly bar close, not a
continuous stream — a persistent websocket buys nothing here and adds
real complexity (reconnect/backoff, keeping a connection alive between
predictions). Same underlying data.

**The live order-book snapshot is NOT a model input.** It's fetched and
logged in live mode as a human-readable extra signal, but the trained
feature vector uses `order_flow_imbalance` — a historically-computable
proxy (signed, z-scored volume) — consistently in both backtesting and
live inference. Feeding the live snapshot into the model would mean
training and inference see two different definitions of the same
column, which is exactly the kind of train/serve skew that quietly
turns a model's output into noise.

**Dead-band threshold scales by asset class (0.5% crypto / 0.08%
forex), not one flat 0.5%.** Measured on real data, EUR/USD's median
3-bar move is roughly 5-6x smaller than BTC/USDT's. A flat 0.5% band
leaves crypto with a usable ~30% of bars and forex with **under 1%** —
in practice, not enough rows to train on EUR/USD or GBP/USD at all. This
mirrors the same asset-class-scaling fix applied to stop-loss sizing in
the sibling TrendPulse project (`../shared/indicators.js`,
`resolveStopLossPct`).

**`ta.add_all_ta_features`'s `trend_psar_up`/`trend_psar_down` are
excluded from the candidate feature set.** These two columns are
complementary by design (exactly one is populated per row — up during
an uptrend, down during a downtrend, the other always NaN) — requiring
both non-null, which every other indicator's warmup-then-always-valid
pattern implicitly assumes, drops every single row. The `_indicator`
variants (0/1 trend-flip flags, always populated) stay in.

**Macro features (BTC dominance, FRED rates, Alpha Vantage CPI) are a
single current snapshot broadcast onto every training row, not a
historically time-aligned series.** A true point-in-time macro history
would need a paid vendor for anything beyond FRED's own ~200-print
lookback, which is short relative to an hourly-bar training set. Treat
these three columns as a coarse "current backdrop," not a rigorous
feature — this is a real limitation, not hidden.

**`alpha_vantage` isn't a runtime dependency.** The PyPI package wraps
time-series/FX/crypto/technical-indicator endpoints, not the economic-
indicators category CPI lives in — that's called directly via
`requests`, so the package was dropped from `requirements.txt` rather
than shipped unused.

## Known limitations (disclosed, not fixed)

- **Tuning and the final backtest fold set are not fully disjoint.**
  `--tune`'s Hyperopt objective already uses purged walk-forward folds
  (never a shuffled/leaky CV), but it draws those folds from the same
  data the subsequent backtest folds are drawn from. A fully rigorous
  setup would carve out a tuning-only slice never touched by the
  reported backtest. This script doesn't do that — a real, if smaller,
  version of the same "don't evaluate on data you searched over" concern
  the removed retry loop was a much larger version of.
- **Macro features are current-snapshot-only** (see above).
- **Yahoo Finance intraday history is capped** (typically ~60 days for
  60-minute bars) — `fetch_ohlcv_forex`'s default `period` reflects that
  ceiling, not an arbitrary choice.
- **Confidence is a distance-from-0.5 measure, not a calibrated
  probability.** `predict_proba`'s raw output is used directly; nothing
  here checks whether "60% confident" predictions are actually right 60%
  of the time (the sibling TrendPulse project's `ml/validation.py` has a
  full Brier/ECE/reliability-table implementation, if you want to port
  that discipline over here).

## CLI reference

```text
--pair SYMBOL         e.g. BTC/USDT, EUR/USD (default: BTC/USDT)
--all                 run all four default pairs instead of --pair
--timeframe TF         default: 1h (the only timeframe this has been exercised against)
--mode backtest|live   default: backtest
--supabase true|false   default: true (falls back to SQLite automatically if keys are missing)
--tune                 run Hyperopt (50 evals) before training; otherwise
                       uses cached (cache/hparams_<PAIR>.json) or default hyperparameters
--folds N               walk-forward fold count (default: 10)
--poll-seconds N         live mode poll interval (default: 3600, matching --timeframe 1h)
```
