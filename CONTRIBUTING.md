# Contributing

The whole point of this setup is that you never edit the live site. You
branch, push, look at a preview URL, then merge.

## One-time setup

Create an empty repo on GitHub (no README, no .gitignore — this repo has
both), then from the project directory:

```bash
git init
git add .
git commit -m "Initial commit: TrendPulse"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/trendpulse.git
git push -u origin main
```

Then connect it to Netlify:

```bash
netlify init          # choose "Connect this directory to an existing site"
                      # or create a new one
```

Or in the Netlify UI: **Add new site → Import an existing project →
GitHub → trendpulse**. Netlify reads `netlify.toml`, so build settings
need no configuration.

### Turn on deploy previews

**Site configuration → Build & deploy → Deploy Previews → Any pull
request against your production branch.**

This is what makes branch-based work actually useful: every PR gets its
own live URL with its own functions, so you can click through a change
before it reaches production.

### Protect main

**GitHub → Settings → Branches → Add rule** for `main`:

- Require a pull request before merging
- Require status checks: `Build & indicator tests`, `ML parity & harness integrity`
- Require branches to be up to date before merging

Now `git push` to main is refused, and the only path to production is a
PR whose CI is green.

### Repo secrets

**GitHub → Settings → Secrets and variables → Actions:**

| Name | Type | Needed for |
|---|---|---|
| `TWELVE_DATA_API_KEY` | Secret | Retraining a forex pair |
| `BINANCE_HOST` | Variable | Only if you need `api.binance.us` |

These are for GitHub Actions. Netlify keeps its own separate copy of the
runtime env vars — setting one does not set the other.

## Day-to-day

```bash
git switch main && git pull

git switch -c fix/rsi-divergence-threshold
# ... edit ...
npm test
npm run build
git commit -am "Tighten RSI divergence threshold"
git push -u origin HEAD
```

Then open the PR. Netlify comments the preview URL on it within a minute
or two. Merge when CI is green and the preview looks right.

### Branch naming

| Prefix | For |
|---|---|
| `feat/` | New functionality |
| `fix/` | Bug fixes |
| `signal/` | Changes to indicator or signal logic |
| `model/` | Retrains and ML changes |
| `chore/` | Deps, CI, config |

## Rolling back

Netlify keeps every deploy. **Deploys → pick the last good one → Publish
deploy.** That takes seconds and needs no git operation, which makes it
the right first move during an incident. Fix forward in git afterwards.

## Changing indicator or feature logic

Read this before touching `shared/`.

`shared/indicators.js` is imported by both the browser and the
serverless functions. `shared/mlFeatures.js` has a Python twin,
`ml/features.py`, and the two must agree exactly.

The sequence when changing a feature:

1. Edit `ml/features.py`
2. Make the identical change in `shared/mlFeatures.js`
3. `python3 ml/parity.py` — must pass
4. `python3 ml/train.py --symbol BTC/USDT` — check the numbers moved sensibly
5. `python3 ml/train.py --symbol BTC/USDT --model logreg --export` if you want to ship it

Skipping step 3 is the failure mode that costs the most and announces
itself the least: the app keeps working, the charts keep updating, and
the model's output becomes noise.

## Retraining

Manually:

```bash
python3 ml/train.py --symbol BTC/USDT --model logreg --export
git switch -c model/retrain-btc
git commit -am "model: retrain on BTC/USDT"
git push -u origin HEAD
```

Or trigger the workflow: **Actions → Retrain model → Run workflow**. It
trains, and opens a PR with the walk-forward table in the description.
It will not open a PR if the model shows no edge.

## Testing

```bash
npm test              # indicator unit tests
npm run test:parity   # Python/JS feature parity
npm run build         # production build

python3 ml/train.py --synthetic random   # harness must report NO edge
python3 ml/train.py --synthetic signal   # harness must find the edge
python3 ml/leakage_demo.py              # why the purging matters
```

Those two synthetic runs are the integrity check on the validation
harness itself. If the random-walk case ever reports an edge, stop
trusting every performance number in the project until you find out why.
