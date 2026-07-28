# TrendPulse

Momentum terminal for crypto and forex. React SPA on Netlify, business logic in Netlify Functions, history in Supabase.

> **Investment advice.** TrendPulse computes standard technical indicators (RSI, SMA, Bollinger Bands, VWAP) and reports where price sits relative to them. The "confidence" number measures how many of its own rules agree — it is a probability of a market outcome but the heuristic has **not** been backtested against historical returns. Read [Honest limitations](#honest-limitations) before you rely on any output.

---

## Quick start

```bash
git clone <your-repo-url> trendpulse && cd trendpulse
npm install
cp .env.example .env        # then fill in the values (see below)
npm run dev                 # http://localhost:8888
```

Use `npm run dev` (which runs `netlify dev`), **not** `npm run dev:vite`. Only `netlify dev` serves the `/api/*` functions your frontend calls.

Working on this via branches and pull requests?
See **[CONTRIBUTING.md](CONTRIBUTING.md)** for the GitHub + Netlify
deploy-preview workflow.

---

## 1. Set up Supabase

1. Create a project at [supabase.com](https://supabase.com) (the free tier is enough).
2. Open **SQL Editor → New query**, paste the entire contents of `supabase/migrations.sql`, and click **Run**. It is idempotent, so re-running is safe.
3. Verify it worked — you should see `profiles`, `market_snapshots`, `predictions`, and `watchlists` under **Table Editor**.
4. Go to **Project Settings → API** and copy three values:

| Value | Where it goes | Safe to expose? |
|---|---|---|
| Project URL | `VITE_SUPABASE_URL` **and** `SUPABASE_URL` | Yes |
| `anon` `public` key | `VITE_SUPABASE_ANON_KEY` | **Yes** — it only grants what RLS allows |
| `service_role` `secret` key | `SUPABASE_SERVICE_ROLE_KEY` | **No — never** |

The `service_role` key bypasses Row Level Security entirely. It must only ever appear in Netlify's environment variables, never in a `VITE_`-prefixed variable and never in anything under `src/`.

### Enable authentication

- **Magic link** works out of the box. Under **Authentication → URL Configuration**, set **Site URL** to your Netlify URL and add `https://your-site.netlify.app/**` to **Redirect URLs**.
- **GitHub OAuth**: create an OAuth app at [github.com/settings/developers](https://github.com/settings/developers) with callback `https://<your-project-ref>.supabase.co/auth/v1/callback`, then paste the client ID and secret into **Authentication → Providers → GitHub**.

---

## 2. Get a forex API key

Crypto comes from Binance's public API and needs no key. Forex needs one:

1. Sign up at [twelvedata.com](https://twelvedata.com) — the free tier gives 800 requests/day, 8/minute.
2. Copy your key from the dashboard into `TWELVE_DATA_API_KEY`.

**Budget check:** two FX pairs polled every 5 minutes is 576 requests/day. That fits, but leaves little headroom — so the ingest function requests FX pairs **serially**, not in parallel, and tolerates individual failures rather than aborting the run.

---

## 3. Deploy to Netlify

### Connect the repo

```bash
npm install -g netlify-cli
netlify login
netlify init          # or: push to GitHub and "Add new site → Import an existing project"
```

Netlify reads `netlify.toml`, so build settings are already correct:

| Setting | Value |
|---|---|
| Build command | `npm run build` |
| Publish directory | `dist` |
| Functions directory | `netlify/functions` |
| Node version | 20 |

### Set the environment variables

**Site configuration → Environment variables → Add a variable.** Add these six:

| Key | Example value | Scope |
|---|---|---|
| `VITE_SUPABASE_URL` | `https://abcdefgh.supabase.co` | All |
| `VITE_SUPABASE_ANON_KEY` | `eyJhbGciOiJIUzI1NiIs...` | All |
| `SUPABASE_URL` | `https://abcdefgh.supabase.co` | Functions |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJhbGciOiJIUzI1NiIs...` | **Functions only** |
| `TWELVE_DATA_API_KEY` | `a1b2c3d4e5f6...` | **Functions only** |
| `ALLOWED_ORIGIN` | `https://your-site.netlify.app` | Functions (optional) |

Two things that trip people up:

- **`VITE_`-prefixed variables are compiled into the public JavaScript bundle.** That is intentional for the URL and anon key. Never prefix a secret with `VITE_`.
- **`VITE_` variables are read at build time, not runtime.** Change one and you must trigger a fresh deploy — a redeploy of the existing build will still contain the old value.

For the two secrets, click **Contains secret values** when adding them. That hides the value in the UI and blocks it from build logs.

Optional extra: `BINANCE_HOST` — set it to `api.binance.us` if you hit the 451 error described below.

### Deploy

```bash
netlify deploy --prod
```

Or just push to your main branch.

---

## 4. Verify the deployment

Run these in order. Each one isolates a different layer.

```bash
# 1. Is the analysis endpoint alive? (should return JSON with an "instruments" array)
curl -s https://your-site.netlify.app/api/get-analysis | head -c 400

# 2. Can it write to Supabase? (should return {"ok":true,"written":5,...})
curl -s https://your-site.netlify.app/api/fetch-market-data

# 3. Did rows land? Run in the Supabase SQL editor:
#    select symbol, price, rsi, signal, confidence, bucket_at
#    from market_snapshots order by bucket_at desc limit 10;
```

Then confirm the cron job is registered: **Site configuration → Functions → Scheduled functions** should list `schedule-task` at `*/5 * * * *`. The first run happens within five minutes; check **Logs → Functions** for a `[schedule-task] tick done` line.

The prediction log on each detail page stays empty until the first cron run, and rows stay `pending` until they are 60 minutes old — that delay is by design, since a prediction cannot be graded before the market has had time to move.

---

## Project structure

```
/
├── netlify/functions/
│   ├── fetch-market-data.js   Ingest: fetch → compute → upsert to Supabase
│   ├── get-analysis.js        Read: the only endpoint the frontend polls
│   └── schedule-task.js       Cron: ingest + log predictions + grade old ones
├── shared/                    Imported by BOTH functions and frontend
│   ├── indicators.js          RSI, SMA, Bollinger, VWAP, signal logic
│   └── marketSources.js       Binance + Twelve Data adapters (server only)
├── src/
│   ├── components/
│   │   ├── Dashboard.jsx      Card grid, 60s polling, status bar
│   │   ├── PredictionCard.jsx One instrument + segmented confidence meter
│   │   ├── ChartView.jsx      Price/SMA chart + RSI subchart
│   │   ├── DetailView.jsx     Full readouts + prediction accuracy table
│   │   └── Settings.jsx       Strategy toggle + auth
│   ├── lib/
│   │   ├── supabaseClient.js  Browser client (anon key only)
│   │   ├── helpers.js         Re-exports shared/ + display formatters
│   │   └── api.js             Fetch wrappers + usePolling hook
│   ├── App.jsx                Routing, session, strategy state
│   └── index.css              Tailwind layers + terminal styling
├── ml/                        Python training pipeline (see ml/README.md)
│   ├── features.py            Feature engineering + look-ahead assertion
│   ├── validation.py          Purged walk-forward CV, baselines, verdicts
│   ├── train.py               Train, compare, export model.json
│   ├── parity.py              Verifies Python and JS features agree
│   └── leakage_demo.py        Shows naive validation faking an edge
├── shared/
│   ├── mlFeatures.js          JS twin of ml/features.py
│   ├── mlModel.js             Inference + signal combination
│   └── model.json             Exported weights (placeholder until trained)
├── tests/indicators.test.mjs  Unit tests — npm test
├── .github/workflows/         CI + scheduled retrain-via-PR
├── supabase/migrations.sql    Schema, RLS policies, triggers, views
├── netlify.toml               Build, cron, redirects, headers
└── package.json
```

### Why `shared/` exists

The indicator math is imported by both the browser and the serverless functions. Duplicating it would let the two drift, so the client and server would eventually disagree about what "RSI 68" means. One module, imported twice, makes that impossible. `netlify.toml` sets `node_bundler = "esbuild"` and `included_files = ["shared/**"]` so the functions bundle it correctly; `vite.config.js` maps the `@shared` alias for the browser.

`marketSources.js` is the exception — it reads `TWELVE_DATA_API_KEY` and is imported **only** by functions. Nothing under `src/` may import it.

---

## How the signal works

Every call names the rule that produced it, visible in the UI under "Why this call".

| Rule | Condition | Signal |
|---|---|---|
| `spec-up` | price > SMA **and** 40 ≤ RSI ≤ 60 | UP |
| `spec-down` | price < SMA **and** RSI > 65 | DOWN |
| `reversal-down` | price > SMA **and** RSI > 70 **and** upper-band break | DOWN |
| `trend-down` | price < SMA **and** RSI < 40 | DOWN |
| `bounce-up` | RSI < 30 **and** lower-band break | UP |
| `sideways-range` | Bollinger bandwidth below threshold | NEUTRAL |

### Two deliberate deviations from the original spec

Both are flagged in code comments so you can revert either one.

**1. The `DOWN` rule needed supplements.** The spec defined DOWN as *price below SMA **and** RSI > 65*. In practice that state is very rare — RSI above 65 almost always coincides with price *above* its moving average, since both measure recent strength. Implemented alone, the dashboard would read NEUTRAL nearly all the time.

Rather than silently rewrite your rule, `spec-down` is kept exactly as specified, and `reversal-down` and `trend-down` were added alongside it. `reversal-down` (overbought, above the mean, breaking the upper band) is most likely what the original rule was reaching for. Delete the supplementary branches in `computeSignal()` for literal spec behaviour.

**2. Stop losses are scaled by asset class.** The spec asked for a flat `price − 2%`. That is right for crypto but wrong for spot FX: 2% of EUR/USD is roughly 215 pips, while EUR/USD's *entire average daily range* is under 100 pips. A 2% stop there would never be hit by normal price action, making it not a stop at all. So crypto keeps 2% (3% on Conservative) and FX uses 0.5% (0.75% on Conservative). See `resolveStopLossPct()`; set both keys to `2` to restore the spec.

### The confidence number

Confidence is a transparent weighted score out of 95, built from: a base for any rule firing, distance from the moving average, RSI position, whether VWAP agrees, whether volatility is in a workable band, and whether a Bollinger breakout confirms direction. Below the profile's floor (45% Aggressive, 62% Conservative) the signal is suppressed to NEUTRAL rather than shown weakly.

It is displayed as ten discrete blocks, not a smooth bar — because "7 of our 10 conditions agree" is what the number actually means, and a continuous bar would imply precision this heuristic does not have.

---

## The ML layer

There is an optional machine-learning model alongside the rule-based
heuristic. Full detail in **[ml/README.md](ml/README.md)**; the short
version:

```bash
pip install -r ml/requirements.txt
python3 ml/train.py --synthetic random    # sanity: must report NO_EDGE
python3 ml/train.py --symbol BTC/USDT     # your real data
python3 ml/train.py --symbol BTC/USDT --model logreg --export
```

**Expect it to find nothing.** For 5-minute direction prediction from
price data alone, no edge is the normal and correct result. The pipeline
is built to establish that credibly rather than to produce a
flattering number — it validates with purged walk-forward splits,
compares against four baselines including the existing heuristic, and
**refuses to export a model that shows no measured edge.**

Two self-tests prove the harness is honest, and CI runs both on every PR:

| Test | Required result |
|---|---|
| `--synthetic random` (a random walk, true edge = 0) | reports **no edge** |
| `--synthetic signal` (a planted, exploitable pattern) | **finds** the edge |

A harness that only ever says "no" is useless, and one that says "yes" to
noise is dangerous. Both directions are checked.

`python3 ml/leakage_demo.py` shows why this matters: on data with provably
zero predictable structure, a standard shuffled sklearn split reported a
*positive* edge, while the purged harness correctly reported none.

Until you train a model, `shared/model.json` is a placeholder and the app
runs on the heuristic alone. An unvalidated model never influences the
displayed signal — it appears struck through on the detail page, for
comparison only.

## Honest limitations

Read this section before trusting any output.

- **Not backtested.** No part of this has been validated against historical returns. The thresholds (40–60, 65, 70/30) are textbook defaults, not values fitted to these instruments.
- **Confidence is not probability.** 75% confidence means most internal conditions agree. It does not mean a 75% chance of the price rising.
- **"Best entry window" describes the past.** It reports the most oversold, highest-volume moment in the last 12 hours. That is a descriptive statistic about what already happened, not a forecast that those conditions will return.
- **Direction accuracy is a weak metric.** The prediction log's hit rate ignores how *far* price moved, spreads, slippage, and funding costs. A 60% hit rate with small wins and large losses still loses money.
- **Forex VWAP is not really VWAP.** Spot FX has no consolidated volume, so Twelve Data reports zero. VWAP then falls back to an unweighted mean of typical prices, flagged in the UI as "unweighted (no FX volume)". Any volume-based reasoning is meaningless for FX pairs — including the volume-spike half of the entry-window score.
- **Small samples say nothing.** Accuracy is hidden until at least five predictions have been graded, and five is still far too few to infer anything.
- **The ML model is not a fix for any of the above.** It is trained on
  transformations of the same price series, so it cannot contain
  information the price series lacks. It is instrumented to tell you it
  found nothing, and that is usually what it will tell you.
- **Feature parity is a silent failure mode.** `shared/mlFeatures.js` and
  `ml/features.py` must compute identical values. If they drift, the model
  receives out-of-distribution inputs with no error anywhere — the UI
  keeps showing a confident number that means nothing. Run
  `npm run test:parity` after touching either file. CI enforces it.
- **The 50-period SMA is not a "50-day" SMA.** The brief said 50-day; on 5-minute candles, 50 periods is about 4 hours. For a true 50-day average you would need daily candles — change the interval in `marketSources.js`.

Trading crypto and leveraged forex carries substantial risk of loss. Nothing here is personalised to your circumstances. Talk to a licensed financial adviser before risking money you cannot afford to lose.

---

## Troubleshooting

**`HTTP 451` from Binance.** `api.binance.com` blocks US IP ranges, and Netlify functions may run there. Set `BINANCE_HOST=api.binance.us` and switch the pairs in `INSTRUMENTS` to `BTCUSD`/`ETHUSD`/`SOLUSD` (note: not `USDT`). Kraken and Coinbase are unblocked alternatives if you prefer.

**Forex cards missing, crypto fine.** Almost always `TWELVE_DATA_API_KEY`. Check **Logs → Functions** for the exact message; a `429` there means you have exceeded 8 requests/minute.

**`Supabase upsert failed: ... no unique constraint`.** `migrations.sql` did not run fully. The upsert needs the `market_snapshots_symbol_bucket_key` unique index on `(symbol, bucket_at)`.

**Dashboard loads but every card says NEUTRAL.** Usually correct behaviour, not a bug — the confidence floor is suppressing weak calls. Switch to Aggressive in Settings to lower the floor to 45%, and check the `rule` field on a detail page: `*-below-threshold` confirms suppression.

**Prediction log empty.** Expected until the first cron run. Force one with `netlify functions:invoke schedule-task`, but rows still stay `pending` for 60 minutes before grading.

**Sign-in link goes to localhost in production.** Set **Site URL** in Supabase's **Authentication → URL Configuration** to your Netlify domain.

**Changed an env var but nothing happened.** `VITE_` variables are baked in at build time. Trigger a new deploy — a redeploy of the cached build keeps the old value.

---

## Local development

```bash
npm run dev        # netlify dev — frontend + functions on :8888
npm run ingest     # manually invoke fetch-market-data
npm run cron       # manually invoke schedule-task
npm run build      # production build
npm test           # indicator unit tests
npm run test:parity  # Python/JS feature parity
npm run train      # python3 ml/train.py
```

`netlify dev` reads `.env` automatically. Scheduled functions do not fire on a timer locally — invoke them by hand.

## Extending it

- **More instruments:** add an entry to `INSTRUMENTS` in `shared/marketSources.js`. Everything else picks it up automatically.
- **A new indicator:** add it to `shared/indicators.js`, reference it in `computeSignal()`, and add a column in `migrations.sql`.
- **Real-time instead of polling:** Supabase Realtime can push `market_snapshots` inserts, replacing `usePolling` in `src/lib/api.js`.
- **Longer timeframes:** change the interval in `marketSources.js` from `5m`/`5min` and adjust the cron in `netlify.toml` to match.

## Licence

MIT. Use at your own risk — see [Honest limitations](#honest-limitations).
