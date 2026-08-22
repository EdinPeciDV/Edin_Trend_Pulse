# ml_macd/

MACD-centric multiclass (up/down/flat) directional predictor. Sibling
to `ml/` and `sysrfx_predictor/`, not a replacement for either —
imports `ml/validation.py` as a library for purged walk-forward, CPCV,
calibration, and DSR/PBO rather than reimplementing them.

This file currently records the four pre-build decisions resolved
before any ingestion/feature code was written. Sections will grow as
each build phase lands (features, training, serving, monitoring).

## 1. Warmup state — why a short trailing window at serve time is not free

MACD's EMA (and Wilder-smoothed RSI/ATR) are recursive with unbounded
memory: `_ema()`, `wilder_rsi()`, `atr()` in `ml/features.py` are all
seeded on the *first* bar of whatever series they're given. Feed the
live serving job a short trailing window and the training pipeline a
long history, and identical code produces different values at the
same timestamp — not because of a bug, but because the recursion's
memory of bars before the window start is genuinely absent.

**`warmup_bars = 260`**, derived empirically, not assumed:
`ml_macd/warmup_convergence_check.py` computes `ema26_norm`,
`macd_hist_norm`, `atr_norm`, and `rsi_14` twice per candidate `W` —
once seeded on the trailing `W` bars, once on `4*W` — and diffs the
value at the identical final bar. Across a crypto-like and a
high-volatility synthetic series (9000 bars each, seeds 42/43):

| Indicator | First W where converged |
|---|---|
| `ema26_norm` | 150 |
| `macd_hist_norm` | 150 |
| `atr_norm` | 100–150 |
| `rsi_14` | 150 |

All four converge (diff below the indicator's own tolerance, see the
script's `TOLERANCES` dict) by `W=150`, and are at exact float64
equality by `W=500`. The brief's own floor — 10x the longest lookback,
i.e. `>=260` for EMA26 — lands with a comfortable ~1.7x margin past
the empirically observed convergence point, so **260 bars is the
adopted `warmup_bars` for entry-timeframe indicators** (15m/1h). The
4x-higher timeframe used for multi-timeframe features (1h/4h) needs
its own 260 bars *at that timeframe's granularity* — i.e. the serving
job must fetch enough higher-TF history independently, not just derive
it from the entry-TF window.

Not yet covered by this derivation: rolling volume-profile windows
(PART 3) have their own warmup question once their window sizes are
chosen — deferred to that phase, not assumed to inherit 260.

**Outstanding, committed for Phase 4 (serving)**: the actual parity
test — run `serve.py`'s feature assembly over a historical window and
assert exact equality against the stored `macd_features` rows for
those timestamps — needs `serve.py` and `macd_features` to exist
first. `ml_macd/warmup_convergence_check.py` is the standalone
evidence for the warmup value in the meantime; the Phase-4 test
verifies the *actual* serving code path uses it correctly (also
catching dtype/timezone/NaN-handling drift the convergence check
alone wouldn't).

## 2. Source pinning for FX — Twelve Data, confirmed via live probe

`scripts/probe_twelvedata.py` (throwaway, not part of this module —
delete once this section is trusted) called the real API on
**2026-08-21** and found:

| Question | Finding |
|---|---|
| Volume field for FX | **Absent entirely** — not `0`, not degenerate, the `volume` key is simply not present in any bar, across EUR/USD, GBP/USD, USD/JPY, EUR/GBP, at every depth and timeframe tried |
| 15m depth | >= 799 days (2.2 years) reached in 12 calls, free tier, without exhausting available history |
| 1h depth | >= 1,676 days (4.6 years) reached in 6 calls, free tier, without exhausting available history |
| Credits used | 25 of 800/day free-tier budget |

**Decision: Twelve Data stays the single pinned FX source for both
backfill and live** (depth comfortably clears the 2yr/15m and 4yr/1h
minimums on the free tier alone — no Grow upgrade needed on that
axis). The consequence, decided explicitly rather than worked around:

- **FX profile mode is TPO-only.** PART 3's `fx -> tick_volume` mode
  assignment is void for this source — there is no tick/volume signal
  to bucket by. The mode enum stays `("real_volume", "tick_volume",
  "tpo")` unchanged; only FX's assignment within it changes to `tpo`
  alone. `tick_volume` remains valid for crypto and stays the
  documented extension point for a future CME FX-futures source
  (PART 4 deferred section) — that override is now not just an
  optional enhancement but **the only route to a real volume profile
  on FX at all**, since spot aggregators structurally don't have one.
- **No fabricated FX volume rows.** `macd_candles.volume` (and
  `number_of_trades`/`taker_buy_*`) are `NULL` for every forex row —
  never zero-filled. An all-zero `tick_volume` profile would be worse
  than an absent one: it would look computed rather than admit it
  isn't available. See the migration's column comments in
  `supabase/migrations.sql` section 9.
- **`poc_agreement` is crypto-only.** Computing `|poc_tpo -
  poc_volume|` needs both a TPO and a volume POC to exist for the same
  profile; FX only ever has the TPO one. Recorded as `NULL` for FX
  (not dropped — the column stays present for schema consistency), and
  PART 4's reporting must render it as "not applicable" rather than as
  a missing/failed value when it filters or aggregates by asset class.
- **PART 3 ablation splits by asset class, not one pooled table**:
  - crypto: (a) MACD, (b) MACD+TPO, (c) MACD+TPO+volume
  - forex: (a) MACD, (b) MACD+TPO — no (c); the report states plainly
    that this is a source limitation (no FX volume data exists to
    ablate), not a result the model produced.

## 3. `ml/validation.py` — additive multiclass extension

Added, without modifying any existing function: `classification_metrics_multiclass()`,
`coverage_precision_curve()`, `thresholds_for_coverage()`,
`purged_walk_forward_by_timestamp()`, `assert_no_cross_symbol_leak()`.
Calibration needed no change — `calibrate_probabilities()`'s
`CalibratedClassifierCV` already dispatches to one-vs-rest for >2
classes. The binary Brier/ECE/MCE/reliability-table functions are
reused unmodified, called once per class as one-vs-rest, rather than
forked into multiclass copies.

Before/after regression check (existing test suite for `ml/`; `sysrfx_predictor/`
was excluded — see below):

| Check | Before | After |
|---|---|---|
| `python3 ml/train.py --synthetic random` | `VERDICT: NO_EDGE` | `VERDICT: NO_EDGE` |
| `python3 ml/train.py --synthetic signal` | `VERDICT: PROMISING` | `VERDICT: PROMISING` |
| `python3 ml/leakage_demo.py` | leaky methods overstate edge by ~2.5pp vs. purged walk-forward | same qualitative result (exact pp figures vary run-to-run — the script uses an unseeded RNG — but the ordering and ~2.5pp gap are consistent) |
| `python3 -c "import ml.validation"` | — | imports cleanly |

New functions smoke-tested directly (cross-symbol purge assertion,
multiclass metrics, coverage-precision curve, threshold-for-coverage
inversion) — all passed; see conversation for the test script and
output.

`sysrfx_predictor/sysrfx_predictor.py` was **not** re-run before/after:
it has zero import of `ml/validation.py` (grepped — the only match is
a comment on line 818 referencing it descriptively), so no code path
connects the two and a before/after comparison would be comparing a
file to itself.

## 4. Indicator provenance — Python vs. `shared/indicators.js`

**Python is authoritative for the model.** The live dashboard's
rule-based heuristic and chart overlays run `shared/indicators.js`;
`ml_macd`'s model is trained and served entirely in Python (see PART 4
serving decision — a scheduled job, not a JS inference path), so there
is no shared code path between the two the way `ml/features.py` and
`shared/mlFeatures.js` share one via `ml/parity.py`. That means the
dashboard and the model **can** show slightly different numbers for
"the same" indicator, and that's expected, not a bug — this section
exists so a future debugging session finds the answer already written
down instead of re-discovering it.

**RSI: Wilder smoothing on both sides**, seeded identically (simple
average of the first `period` gains/losses, then recursive
`(prev*(period-1)+new)/period`) — `shared/indicators.js::rsi()` and
`ml/features.py::wilder_rsi()` (which `ml_macd` reuses, not
reimplements) are the same algorithm.

One-off comparison (`ml_macd/indicator_parity_check.py`, synthetic
400-bar crypto-like and zero-volume forex-like series):

| Indicator | max abs diff (crypto) | max abs diff (forex, zero-vol) |
|---|---|---|
| `rsi_14` | `0.000e+00` | `0.000e+00` |
| `bb_bandwidth` | `1.266e-16` | `2.056e-16` |
| `vwap` (48-bar window) | `4.366e-11` | `8.882e-16` |

All three are at or near float64 machine epsilon — genuine agreement,
not coincidental closeness. `bb_bandwidth` uses the same formula on
both sides: `(upper-lower)/middle` over a 20-period SMA with
`upper/lower = middle +/- 2*population_stddev`. `vwap` uses the same
primitive on both sides (volume-weighted mean of typical price
`(H+L+C)/3`, falling back to an unweighted mean when total volume is 0
— the spot-FX case) — compared here on the same trailing 48-bar
window `ml_macd` will use, since `shared/indicators.js::vwap()` takes
whatever slice its caller passes rather than owning a fixed window
itself.

MACD itself has no JS counterpart to compare against —
`shared/indicators.js` does not implement MACD at all (only
`ml/features.py`'s buried `macd_hist_norm` feature and
`sysrfx_predictor`'s generic `ta` library output do), so PART 4's
train/serve-disagreement concern doesn't apply to MACD specifically,
only to the three indicators named above.

## 5. Step 1 — Ingestion (PART 1 deliverable 1)

Files: `ml_macd/providers.py` (`BinanceProvider`, `TwelveDataProvider`
— one narrow `fetch_candles`-shaped interface each), `ml_macd/data.py`
(closed-candle guard, Supabase upsert, backfill/live CLI). Migration:
`supabase/migrations.sql` section 9, `public.macd_candles`.

**Design decisions:**
- Supabase writes go through raw PostgREST REST calls (`urllib`, zero
  new dependency) rather than `supabase-py`, matching `ml/`'s
  deliberately minimal-dependency philosophy and this project's own
  established direct-REST pattern for service-role writes.
- `drop_forming_candle()` is the single guard enforcing "never store a
  forming candle," applied uniformly after fetch regardless of source,
  rather than trusting each provider to only ever return closed bars.
- Binance's REST API and `data.binance.vision` bulk archives are
  treated as ONE `source` value (`binance_spot`) — both deliver the
  same exchange's own trade-level data, just via different mechanisms
  (recent-window REST vs. bulk monthly archive), so combining them for
  one symbol's history is not the PART 3 source-mixing violation that
  switching FX vendors would be.

**Bug found and fixed during testing**: `data.binance.vision`'s
archives silently changed timestamp precision from milliseconds to
**microseconds** for recent months (confirmed empirically — the
2024-01 archive is ms-scale, the 2026-07 archive is µs-scale; nothing
in the archive or its filename says which). Un-normalized, every
archive-sourced bar from an affected month read as a date ~700 years
in the future, and `drop_forming_candle()` correctly-but-silently
filtered every single row out as "not yet closed" — a backfill that
would have appeared to succeed (no error, no rows written, easy to
miss) rather than loudly fail. Fixed in
`providers.py::_normalize_to_ms()`: anything above `10**14` is treated
as microseconds and divided down. Verified against both a definitely-ms
month (2024-01: 744 hourly bars) and a definitely-µs month (2026-07:
744 hourly bars, correctly parsed after the fix, 0 before it).

**Exact switchover date, confirmed by probing `BTCUSDT-1h` monthly
archives directly (not inferred, not documented anywhere by Binance
that this project found)**: the **2024-12** archive is millisecond,
the **2025-01** archive is microsecond — the switch happens at the
January 2025 archive, cleanly at a month boundary (not mixed within
one month, at least for this pair/timeframe). Checked
`2024-12, 2025-01..2025-06`, all six 2025 months confirmed
microsecond. Treat any archive from `2025-01` onward as microsecond
and anything through `2024-12` as millisecond for BTCUSDT 1h; other
symbols/timeframes were not individually re-checked, so verify before
assuming the same exact month if `_normalize_to_ms()`'s magnitude
-based detection is ever removed in favor of a hardcoded date.

**Verified end-to-end** (dry-run except the last row, which was
deliberately run for real to confirm the failure mode):
- `data.binance.vision` backfill, BTC/USDT, 1h + 4h, single month —
  744 / 186 bars, all fields populated including `number_of_trades`
  and `taker_buy_*`.
- Binance REST live increment, BTC/USDT, 1h + 4h — 9 bars each
  (10-bar lookback minus the still-forming candle, as expected).
- Twelve Data live increment, EUR/USD, 1h + 4h — 9 bars each,
  `volume`/`number_of_trades`/`taker_buy_*` correctly `None` throughout.
- A real (non-dry-run) upsert attempt against the actual Supabase
  project correctly failed with PostgREST `PGRST205` — "Could not
  find the table 'public.macd_candles'" — because **the migration has
  not been applied yet**. This is expected: PostgREST has no DDL
  endpoint by design (same constraint `sysrfx_predictor/README.md`
  documents for its own schema), so `supabase/migrations.sql` section
  9 must be pasted into the Supabase SQL editor before any real
  backfill can write data. No backfill has been run for real yet —
  everything above is fetch-and-validate only.

**Post-review corrections** (§9 of the migration, reviewed before
applying):
- `asset_class` uses `'forex'`, not `'fx'` — matches
  `market_snapshots`' existing convention. An earlier draft of this
  build's brief said `'fx'`; the schema and `ml_macd/data.py` (which
  never actually used `'fx'` — confirmed by grep before this note was
  written) both use `'forex'`. `ASSET_CLASS_CRYPTO`/`_FOREX`/`_STOCK`
  constants added to `data.py` so the value is typed once, not
  scattered as string literals.
- `timeframe` now has `check (timeframe in ('15m', '1h', '4h'))` —
  without it, `'1h'` vs `'1H'` vs `'60m'` would each pass the unique
  index and silently create separate, disagreeing series. No `'1d'`:
  nothing in `ml_macd/data.py` fetches or writes it, so it's left out
  rather than pre-approved for a future code path that may never
  exist. `ALLOWED_TIMEFRAMES` in `data.py` mirrors this exactly and is
  validated in `bars_to_rows()` before any row is built, not just at
  the database boundary.
- `open_time` now has `check (open_time > '2010-01-01' and open_time
  < '2100-01-01')` — a second line of defense against a repeat of the
  microsecond-vs-millisecond bug above, for future code paths the
  Python-side fix doesn't cover. Static bounds only: `now()` is not
  `IMMUTABLE` and cannot appear in a `CHECK` constraint.
- Confirmed, not assumed: `upsert_candles()` sends both `apikey` and
  `Authorization: Bearer` as `SUPABASE_SERVICE_ROLE_KEY` (grepped the
  actual request headers in `data.py`), so `macd_candles`'
  no-insert-policy RLS setup is the intended path, not a blocker — the
  earlier `PGRST205` really was table-not-found, not an auth failure
  the table's absence happened to mask.

## 6. Stage 2 — backfill guards (no production writes in this stage)

Before any multi-file backfill (Stage 3+): Stage 1 wrote 744 rows in
one month, where a silent failure would have been obvious in the row
count alone. A full-history backfill spans hundreds of archive files —
if a subset silently yields nothing, it won't show in one summary
line, it'll show up weeks later as a hole in a feature. Four guards,
all in `ml_macd/providers.py` / `ml_macd/data.py`:

1. **Post-write assertion per archive file** — `BackfillIntegrityError`
   (`data.py`), raised and the whole run halted if a file parsed
   non-empty but wrote zero rows. A past month's archive that parses
   successfully should always yield at least one closed candle;
   `drop_forming_candle()` can only ever drop the single bar nearest
   "now," never a whole past month. Demonstrated live: monkeypatched
   `backfill_from_vision_files()` to yield a file with 5 parsed rows,
   all timestamped by the same bug pattern that was fixed above (so
   `drop_forming_candle()` filters all 5 out) — `BackfillIntegrityError`
   fired exactly as intended, naming the file and the resume point,
   with zero network calls (the empty row list short-circuits
   `upsert_candles()` before it would ever reach Supabase).
2. **Parse-time sanity bound** — `providers.py::_normalize_and_validate_ms()`,
   rejects anything outside `[2010-01-01, now+1day]` at parse time,
   naming the file/context and the offending raw value. The DB CHECK
   on `macd_candles.open_time` (migration section 9) is the *second*
   line of defence, not the first — by the time a bad value reaches
   Postgres it has already silently corrupted whatever in-process
   logic ran before the write.
3. **Detected-unit logging per file** — every archive file's ms-vs-µs
   detection is now surfaced in the per-file log line (`unit=ms` /
   `unit=us`), not silently handled. Re-running the Stage 1 range
   extended to span the boundary (`2024-12`..`2025-01`) confirms it:
   `[BTCUSDT-1h-2024-12.zip] unit=ms parsed=744 closed=744` then
   `[BTCUSDT-1h-2025-01.zip] unit=us parsed=744 closed=744` — the
   detector correctly flips exactly at the already-documented boundary,
   with no rows lost on either side.
4. **Per-file progress log** — filename, detected unit(s), rows
   parsed, rows written, one line per archive file (see the two lines
   above).

`backfill_from_vision()` was restructured into
`backfill_from_vision_files()`, a per-file generator, since guard #1
requires writing and verifying after each file rather than
accumulating a multi-year list before any write happens at all.

**Bug found and fixed while building guard #2**: the out-of-bounds
error path itself crashed — formatting an absurdly-out-of-range
timestamp via `datetime.fromtimestamp(...).isoformat()` raised
`OSError: [Errno 22] Invalid argument` (platform-dependent; hit on
this Windows dev environment) instead of the intended `ValueError`,
which would have hidden the actual problem behind an unrelated
crash — the second time in this build that the code meant to catch a
bad timestamp needed its own bug fixed first. Wrapped in
try/except, falling back to a plain "unrepresentable as a date"
message when the value is too extreme to format at all. Re-verified
both an extreme value and a realistic near-zero-epoch mixup after the
fix — both now raise the intended, readable `ValueError`.

## 7. Stage 3 — BTC/USDT full depth (real writes)

Full available history, both 1h and 4h (the entry+4x-higher pairing
firing deliberately — see `ENTRY_TO_HIGHER_TF` in `data.py`), earliest
published archive (`2017-08`) through `2026-07`.

**A second real bug, caught mid-run by the run itself**: the first
attempt crashed with `HTTPError 409` —
`duplicate key value violates unique constraint "macd_candles_symbol_tf_open_key"` —
at exactly the `2025-01` file, the one Stage 1 had already written.
Root cause: `Prefer: resolution=merge-duplicates` alone does not tell
PostgREST which constraint to build the `ON CONFLICT` clause against;
without an explicit `on_conflict` query parameter it only targets the
table's primary key (`id`, never colliding on insert), so it silently
fell through to a bare `INSERT` and Postgres raised the raw duplicate-
key error itself. Fixed by adding
`?on_conflict=symbol,timeframe,open_time` to the upsert URL in
`upsert_candles()`. **No incorrect rows were written under the broken
behavior** — every file before the crash was a genuine first-time
insert (no conflict existed to mishandle), and the one batch that did
conflict failed atomically (PostgREST batches are one SQL statement;
a 409 means zero rows from that batch committed) — the failure mode
was "crash before writing," never "write something wrong." Re-verified
directly: upserting a known-good row back into the exact conflicting
slot (`2025-01-01T00:00:00Z`) succeeded cleanly post-fix with no value
change.

Resumed via `--resume` (now proven live, not just smoke-tested):
```
[resume] BTC/USDT 1h: latest recorded row at 2025-01-31T23:00:00+00:00 -> resuming from 2025-01
[resume] BTC/USDT 4h: no existing rows, falling back to --start 2017-08
```

**Results, queried fresh from Supabase (not from ingestion output):**

| | 1h | 4h |
|---|---|---|
| Total rows | 78,373 | 19,608 |
| `min(open_time)` | 2017-08-17T04:00:00Z | 2017-08-17T04:00:00Z |
| `max(open_time)` | 2026-07-31T23:00:00Z | 2026-07-31T20:00:00Z |
| Future-timestamp count | 0 | 0 |
| True gap segments / missing bars | 28 / 127 | 9 / 17 |
| Unit switch (`2025-01`) | both sides confirmed processed | both sides confirmed processed |

**Row-count reconciliation — the actual completeness proof**:
expected slots between `min` and `max` inclusive, computed
independently as `(max - min) / interval + 1`:
`1h: 78,500 = 78,373 actual + 127 missing` (exact) and
`4h: 19,625 = 19,608 actual + 17 missing` (exact). Both gap sets
cluster almost entirely in 2017–2021 (Binance's early, less mature
years) and are near-nonexistent after 2021 — consistent with genuine
exchange outages/maintenance, not a parsing artifact. Full per-bar gap
timestamps recorded in conversation, not duplicated here.

**A third finding, from cross-checking the gap count itself — corrected
after building proper grid-alignment detection (§8)**: the original
report here said "one non-hour-aligned bar." That was **wrong,
caught by the project's own tooling**: an ad-hoc `delta != 3600
between neighbors` check only catches the LAST bar of a run of
misaligned bars, because bars shifted by the same offset still show a
normal delta *to each other* — only the boundary where the run starts
or ends produces an unusual neighbor-to-neighbor delta. Once
`misaligned_bars()` (§8) checked absolute grid alignment
(`open_time % timeframe_s`) instead of neighbor deltas, the true
extent came out: **BTC/USDT 1h has one contiguous 43-bar run**,
`2018-02-09T09:28:14Z` through `2018-02-11T03:28:14Z`, every bar
offset exactly 1694s (28m14s) from the hourly grid. It sits
immediately after the already-recorded 33-bar gap
(`2018-02-08T01:00`→`2018-02-09T08:00`) and resumes cleanly on-grid at
`2018-02-11T04:00:00Z`. **4h has zero misaligned bars** for the same
period — whatever caused this didn't affect that timeframe's native
candle generation. The 127-missing-bars total is unaffected either
way (misalignment and missing-ness are independent categories). This
correction is the reason `misaligned_bars()` now exists as a proper,
reusable detection function (`ml_macd/gap_handling.py`, wired into
every stage report via `ml_macd/verification.py`) rather than an
inline one-off check.

**Table size — estimated, not measured**: no `SUPABASE_DB_URL` is
configured and PostgREST exposes no `pg_total_relation_size` RPC, so
there is no way to query this exactly through the paths this project
currently has. Rough estimate from row/column composition (heap + the
two indexes on `macd_candles`): 97,981 total rows (both timeframes) ≈
**23–28 MB**. Free tier is 500MB, so — at BTC/USDT's rate, the oldest
and most complete symbol this project will backfill (other symbols'
shorter listing histories should cost less, making this a
conservative/pessimistic ceiling) — that implies roughly **20
symbol-pairs** (one symbol, both its entry and 4x-higher timeframes)
before hitting the free tier. **Flag before Stage 4**: check this
against the real remaining-crypto symbol count before backfilling
them, and re-derive from Supabase's own Database Size report (the
authoritative source) rather than this estimate once more than a
couple of symbols are in.

## 8. Phase 2 requirement — missing-bar handling (not covered by the original prompt)

Surfaced directly by Stage 3's real gap report: 127 missing 1h bars,
17 missing 4h bars, real Binance outages — absent ROWS, not
overnight/weekend price gaps. PART 2's `gap_size` feature (open vs.
prev_close) does not cover this; naively, consecutive rows across a
hole look adjacent to any code that isn't specifically checking for
it, which breaks two separate things. Prototyped and unit-tested in
`ml_macd/gap_handling.py` — a reference implementation for the real
`ml_macd/features.py`/`labels.py` to import, not a throwaway, unlike
`indicator_parity_check.py`/`warmup_convergence_check.py`.

**1. Recursive indicators span the hole.** EMA26/MACD/ATR/RSI run
straight across an absence without error, producing wrong values for
roughly the following warmup period. Per PART 2's continuity rule,
the EMA is NOT reset across the hole — `bars_missing_before()` and
`is_post_gap_bar()` only compute a flag, alongside indicators computed
exactly as if the gap weren't there, so the model can learn to
distrust affected bars (or they can be excluded from training) without
changing how any indicator itself is computed.

**2. Labels crossing a hole are silently wrong, not missing.** The
serious case: a forward return over `N` ROWS spans a longer real-world
horizon than `N` implies whenever the row span crosses a hole — a
corrupted label, not an absent one, and corrupted labels poison
training silently. `label_validity_by_timestamp()` computes validity
on TIMESTAMP distance, not row offset: invalid (excluded, never
filled or approximated) whenever the actual elapsed time from bar `i`
to bar `i+N` exceeds `N * timeframe + 1 bar's worth` (`tolerance_bars`,
configurable) — an upper bound.

**The asymmetry, found during review and fixed**: only that upper
bound existed originally. A hole makes a window LONGER than expected
and was caught; a short or misaligned bar makes a window SHORTER than
expected and passed through unflagged. Added a lower bound,
`min_elapsed = min_fraction * N * timeframe` with `min_fraction=0.90`
default — deliberately loose, not a tight symmetric tolerance: a
single short bar inside a long horizon barely moves the ratio (one
~1700s-short bar in a 48-bar/172800s window is a ~1% contraction,
nowhere near 90%) and must NOT be flagged; only a short horizon where
one such bar is a large fraction of the window, or a genuine cluster,
should be. Verified both directions in one test (below).

**3. Grid misalignment — detection now, feature treatment deferred.**
`misaligned_bars()` flags any bar where `open_time % timeframe_s != 0`
— absolute grid position, not neighbor-to-neighbor delta (the
distinction that mattered: see §7's correction, where a delta-only
check missed 42 of 43 real misaligned bars because bars shifted by the
same offset still look normally-spaced *to each other*).
`bar_durations()` derives each bar's actual width
(`next_open_time - open_time`) at read time rather than storing it —
cheap to compute, always current, no denormalized column to keep in
sync. Neither function corrects, drops, or halts on a misaligned bar;
the OHLC in it is real traded data over a shorter window, reported
like a gap, not treated as an error. Wired into every stage report via
the new `ml_macd/verification.py::stage_report()` — one canonical
function all future stages use, replacing the ad-hoc per-stage query
scripts Stages 1–3 each wrote separately.

**Deferred, open item**: feature-level handling of short/misaligned
bars (ATR distorted by a truncated range; a volume-profile bucket
under-weighting a short bar's real contribution) is not built. Not
worth it for one 43-bar cluster out of 78,373 rows in one symbol.
Revisit once Stage 4 (remaining crypto) and Stage 5 (forex) give a
real prevalence count across more history and more symbols/vendors.

**Required unit tests — both passed**:
- Deliberate synthetic 3-bar hole: `bars_missing_before` is exactly 3
  at the one affected bar and 0 elsewhere, `is_post_gap_bar` is True
  at exactly that bar, a horizon-4 label spanning the hole is excluded
  while one entirely after it remains valid.
- Deliberate synthetic short bar (one bar shifted 20 minutes early):
  `misaligned_bars` flags exactly that bar, `bar_durations` reports
  its shortened width correctly, an N=1 label whose entire window IS
  the shortened bar is excluded by the new lower bound, and an N=48
  label merely *containing* that one short bar stays valid — proving
  the lower bound is loose enough not to contaminate a long horizon
  over a single short bar.

**Real-data validation, BTC/USDT 1h, PART 1's N grid, with the lower
bound active**:

| N | total candidate labels | valid | dropped | % dropped |
|---|---|---|---|---|
| 4 | 78,369 | 78,289 | 80 | 0.102% |
| 8 | 78,365 | 78,213 | 152 | 0.194% |
| 12 | 78,361 | 78,133 | 228 | 0.291% |
| 24 | 78,349 | 77,893 | 456 | 0.582% |
| 48 | 78,325 | 77,413 | 912 | **1.164%** |

Only N=4 changed versus the pre-lower-bound numbers (76→80 dropped) —
the 43-bar misaligned cluster is large enough, relative to a 4-bar
window, to push a handful of short-horizon labels below 90% of
expected elapsed time; at N=8 and above the same cluster is too small
a fraction of the window to matter, exactly the intended "loose,
cluster/short-horizon-only" behavior. `misaligned_bars` on the real
series confirms the corrected §7 finding: 43 bars flagged on 1h
(one contiguous run), 0 on 4h. Drop rate still grows roughly linearly
with N; N=48 still crosses the reporting function's 1% flag threshold.

**Not yet run**: forex (Stage 5) — paused pending go-ahead. Stage 4
(below) is complete.

## 9. Stage 4 — remaining crypto symbols (real writes)

All 9 remaining crypto symbols from `shared/marketSources.js`'s
10-pair registry (BTC/USDT was Stage 3): ETH, SOL, XRP, ADA, DOGE,
AVAX, LINK, LTC, DOT — all `/USDT`, both entry and 4x-higher
timeframe each, from each symbol's real earliest-available archive
month (probed directly, not guessed — see table below) through
`2026-07`.

**Pre-flight projection** (per §7's flag): probed each symbol's
earliest archive individually rather than assuming BTC's worst-case
size for all nine — most are listed years after BTC/ETH, so their
real history is much shorter. Projected 1,604 files, ~732,270 rows,
~182MB — comfortably within headroom (~207MB of 500MB projected,
not "approaching" the ceiling), so proceeded without an extra pause.

**A second real duplicate-row bug, caught mid-run**: crashed with
`HTTPError 500` — `ON CONFLICT DO UPDATE command cannot affect row a
second time` (Postgres `21000`) — partway through `AVAX/USDT` 4h.
Different failure mode from Stage 3's `on_conflict` bug: this time the
unique-index target was correctly specified, but a SINGLE archive
file's own rows contained a genuine duplicate — `AVAXUSDT-4h-2026-05.zip`
has the exact same `(open_time, OHLCV, ...)` row appearing twice,
confirmed by fetching and inspecting the raw CSV directly. Postgres
cannot apply `ON CONFLICT DO UPDATE` twice to the same target row
within one `INSERT` statement, so two identical rows for the same key
in one batch crashes the whole write even though nothing is wrong with
the table or the conflict target. Fixed with
`providers.py::_dedupe_by_open_time()` — keeps the first occurrence,
drops the rest, and reports every drop (never silent), distinguishing
identical duplicates (safe, just redundant — this case) from
duplicates with **differing** OHLCV values for the same timestamp
(would print as a `CONFLICT`, none found here). Verified directly
against the broken file (187 raw rows → 1 duplicate dropped → 186
correct rows) before resuming.

Resumed via `resolve_backfill_start()` per symbol/timeframe (DB state
checked directly before resuming, not inferred from the crash log):
ETH/SOL/XRP/ADA/DOGE and AVAX/USDT 1h were already complete through
`2026-07` (harmlessly re-overlapped their last month); AVAX/USDT 4h
picked up from `2026-04` (where the crash stopped it); LINK/LTC/DOT
started fresh. Completed cleanly — no further duplicate-row or other
errors on the second pass, including across the three fresh symbols.

**Results — queried fresh via the canonical `verification.stage_report()`,
not from ingestion output (all `future_count = 0`):**

| Symbol | TF | Rows | min open_time | Gap segs / missing | Misaligned |
|---|---|---|---|---|---|
| ETH/USDT | 1h | 78,373 | 2017-08-17 | 28 / 127 | **43** (Feb 2018 incident) |
| ETH/USDT | 4h | 19,608 | 2017-08-17 | 9 / 17 | 0 |
| SOL/USDT | 1h | 52,319 | 2020-08-11 | 10 / 19 | 0 |
| SOL/USDT | 4h | 13,085 | 2020-08-11 | 0 / 0 | 0 |
| XRP/USDT | 1h | 72,169 | 2018-05-04 | 25 / 87 | 0 |
| XRP/USDT | 4h | 18,055 | 2018-05-04 | 7 / 9 | 0 |
| ADA/USDT | 1h | 72,581 | 2018-04-17 | 25 / 87 | 0 |
| ADA/USDT | 4h | 18,158 | 2018-04-17 | 7 / 9 | 0 |
| DOGE/USDT | 1h | 61,961 | 2019-07-05 | 18 / 43 | 0 |
| DOGE/USDT | 4h | 15,499 | 2019-07-05 | 2 / 2 | 0 |
| AVAX/USDT | 1h | 51,311 | 2020-09-22 | 10 / 19 | 0 |
| AVAX/USDT | 4h | 12,833 | 2020-09-22 | 0 / 0 | 0 |
| LINK/USDT | 1h | 66,027 | 2019-01-16 | 20 / 59 | 0 |
| LINK/USDT | 4h | 16,517 | 2019-01-16 | 4 / 5 | 0 |
| LTC/USDT | 1h | 75,549 | 2017-12-13 | 27 / 120 | **43** (same incident) |
| LTC/USDT | 4h | 18,902 | 2017-12-13 | 8 / 16 | 0 |
| DOT/USDT | 1h | 52,134 | 2020-08-18 | 10 / 19 | 0 |
| DOT/USDT | 4h | 13,039 | 2020-08-18 | 0 / 0 | 0 |

**Total: 728,120 rows.**

**The Feb 2018 grid-shift finding generalizes cleanly, not
mysteriously**: ETH/USDT and LTC/USDT are the only two Stage-4 symbols
whose history reaches back before Feb 2018 (both listed 2017); every
other symbol's earliest archive starts AFTER the incident window
(XRP: 2018-05, ADA: 2018-04, the rest 2019+), so they simply have no
overlapping data to be affected. LTC's 43-bar run is the identical
window (`2018-02-09T09:28` → `2018-02-11T03:28`) at a ~2-second
different offset (`:16` vs. ETH's `:14`) — consistent with one dated,
multi-symbol Binance-side event, not a per-symbol data artifact. No
other symbol shows any misalignment at all.

**Table size, updated**: 826,101 total crypto rows now (97,981 from
Stage 3 + 728,120 from Stage 4) ≈ **205 MB estimated** (still an
estimate, not a measurement — see §7's caveat) of the 500MB free tier,
leaving ~295MB headroom before Stage 5 (forex — expected much smaller,
given Twelve Data's shorter confirmed FX history).

**Not yet run**: forex (Stage 5) — paused pending go-ahead.

## 10. Stage 4 verification — reconciliation, contamination check, corrected formula

Before Stage 5, two gaps in §9's report were closed.

**Per-symbol reconciliation, all 18 pairs — the completeness proof
generalized from §7**: NOT the literal
`written + missing + duplicates = expected` originally proposed —
that overcounts, because a duplicate is an extra copy of an
already-expected slot, not an additional one. Applying it literally
would have flagged `AVAX/USDT` 4h as a false mismatch
(`12,833 + 0 + 1 = 12,834 ≠ 12,833`). What actually holds, and does
so exactly for all 18 pairs: `written + missing_bars = expected_slots`
(expected computed independently from `(max−min)/interval + 1`), with
`duplicates_dropped` reported as its own fact rather than folded into
the equation. `AVAX/USDT` 4h is the clean proof this split is correct:
`written(12,833) + missing(0) = expected(12,833)` exactly, with the 1
dropped duplicate accounted for separately (it was an extra raw copy
of a slot already counted). Comprehensive grep across every backfill
log confirms `AVAX/USDT` 4h is the ONLY duplicate-row event across all
of Stage 3 and Stage 4 — every other pair: `duplicates_dropped = 0`.
Gap-segment/missing-bar counts per pair are recorded in §9's table;
this reconciliation is the independent cross-check that they're real
and complete, not just self-consistent.

**BTC/ETH/LTC contamination check — verified, not assumed.** ETH's
row/gap/misalignment counts match BTC's exactly (`78,373`/`19,608`,
28 gap segments, 127 missing bars, both hit by the Feb 2018 incident);
suspicious enough on its own to warrant a real check rather than a
plausibility argument. Spot-checked close prices directly:
`2017-08-17T04:00:00Z` — BTC `$4,308.83`, ETH `$301.61` (both match
known historical values for that date); `2020-01-01T00:00:00Z` — BTC
`$7,177.02`, ETH `$128.87`, LTC `$41.28` (all distinct, all
plausible). **Genuinely distinct series, not a copy.** BTC/ETH's
exact-matching counts are explained, not coincidental-and-unexplained:
they are Binance's two longest-listed, most liquid pairs, so they
share essentially every exchange-wide outage window — including Feb
2018's. LTC's overall counts differ from both (listed 4 months later)
and only shares the Feb 2018 misalignment specifically because its own
history happens to reach back that far too, at a distinguishable
2-second offset (`:16` vs. ETH's `:14`).

**Table size — still unverified against Supabase's actual reported
number.** No `SUPABASE_DB_URL` and no size-reporting RPC means there
is no path from this codebase to the dashboard's real figure; the
~205MB estimate in §9 stands unconfirmed. Check Database → Database
Size directly if this matters before adding more symbols later.

**Universal write-path dedup added** (`data.py::upsert_candles()`):
`providers.py`'s `_dedupe_by_open_time()` only covers the crypto
archive-file path, where the AVAX bug was found. `upsert_candles()` is
the actual choke point every write — crypto archive, crypto REST
live-increment, AND forex — passes through, so it now dedupes there
too as defense-in-depth, rather than assuming forex (untested at full
scale before Stage 5) or any future source is equally clean. Reports
(never silently drops) anything actually removed.

## 11. Stage 5 preparation — real depth probe, not an estimate

Continued the original Twelve Data probe's walkback (scripts/probe_twelvedata.py's
partial results) to find EUR/USD's TRUE earliest available data,
rather than project from a partial depth check:

| Timeframe | True earliest | Calls to confirm |
|---|---|---|
| 1h | `2020-01-30` | 9 (6 original + 3 continuation) |
| 15m | `2020-01-30` | 34 (12 original + 22 continuation) |
| 4h | `2020-01-30` | 3 |

**Identical floor across all three timeframes** — one underlying feed
with a fixed coverage start, not three independently-limited series.
FX real depth is ~6.5 years (2020-01-30 to present), materially
shorter than crypto's ~9 years for BTC/ETH, ~2 years for the youngest
crypto symbols (SOL/AVAX/DOT). Total call cost for full-depth backfill
of both configured symbols (`EUR/USD`, `GBP/USD`), all three
timeframes: **~110 calls** (well under the 800/day cap, single-day
completion) — the multi-day-pause threshold this check existed to
catch was never close to being hit.

**Decisions for Stage 5** (all three noted directly, not re-derived
later): take BOTH FX timeframe pairings (15m+1h AND 1h+4h — 3
timeframes total, ~55 calls/symbol) so the entry-timeframe choice
stays open at modelling time instead of costing a second backfill
round; `macd_candles.timeframe`'s `CHECK (in ('15m','1h','4h'))`
already allows all three, no migration change needed. Two symbols
only (`EUR/USD`, `GBP/USD`) — not the full 10-pair
`shared/marketSources.js` registry — since correlated majors add rows
without adding independent information for PART 2's cross-symbol
purge, and two is enough to prove the FX path end to end.

## 12. Known limitations, recorded before they bite

**FX has no working `--resume` — a real asymmetry with crypto, not
an oversight to gloss over.** `backfill_crypto()` writes incrementally,
one archive file at a time, with `resolve_backfill_start()` deriving a
resume point from the DB. `backfill_forex()` fetches the ENTIRE
requested range into memory first and writes ONCE at the very end —
so a mid-run failure discards every successful call before it, and a
retry re-spends those credits from scratch rather than picking up
where it left off. This is exactly what happened on Stage 5's first
attempt: the crash was on the very first underlying page call, so
nothing had accumulated yet and nothing was lost — but at a larger
scale (more symbols, deeper history, or a failure later in a long
pagination run) the same asymmetry would waste real, possibly
significant, Twelve Data credit budget. **Not blocking at Stage 5's
~55 calls/symbol scale. The fix**: `backfill_forex()` should write
incrementally too — per page (matching Twelve Data's natural
`outputsize=5000` chunking, the FX equivalent of crypto's per-archive-
file granularity) rather than accumulating the whole range in memory,
so `resolve_backfill_start()`/`--resume` work identically for both
providers. **Revisit before any deeper FX backfill or additional FX
pairs** — the current scope doesn't justify the refactor yet, a larger
one would.

**Cross-process rate-limit pacing has no shared state — noted as an
option, not built.** `TwelveDataProvider`'s `min_spacing_s` pacing
(§7's 429 fix added retry-with-backoff on top of it) only has
visibility into calls made by its OWN process. Running multiple
separate scripts against the same Twelve Data key in quick succession
(exactly what happened here: probe scripts, then the Stage 5 driver,
each a fresh process) can still burst past the per-minute credit cap
even though each process individually paces itself correctly — there
is no cross-process memory of "when was the last call." A cheap fix,
if this becomes a recurring problem rather than a one-off: a shared
timestamp file (e.g. `ml_macd/cache/.twelvedata_last_call`) that every
process reads before its first call and writes after every call,
turning the pacing into a crude but effective mutex-by-timestamp
across however many processes touch the key. Not built now — noted as
the option to reach for if this happens again, not a problem serious
enough to solve preemptively for a two-symbol backfill.

**Table size — real number in, estimate corrected.** Checked directly
against the Supabase dashboard (not derived): **384MB of 500MB**, for
crypto alone (Stage 5's FX rows had not written yet at the time of
this check — `backfill_forex()`'s write-once-at-the-end design, see
above, means zero FX rows existed in the table at this point). That
is ~87% higher than §9's ~205MB estimate from row/column composition
— the per-row byte estimate there was too low (likely underestimating
index overhead, page fill factor, or `text` column storage), not a
sign of a data problem; row counts, gap reconciliation, and price
spot-checks all independently confirmed the data itself is correct
(§10). Estimates in this file from here on should be treated as
order-of-magnitude only, not budget-planning numbers — check the
dashboard directly for anything that matters. At 384MB, headroom
before the 500MB cap is **~116MB**, still enough for Stage 5's ~110
forex API calls' worth of rows (a small fraction of crypto's volume)
plus `macd_predictions`/`macd_model_runs`, but materially tighter than
the ~295MB this file previously assumed — no more free crypto symbols
without a real check first.

## 13. Phase 2 storage design (recorded ahead of the build)

Not yet built — Phase 2 (features/labels) hasn't started; ingestion
(Stage 5) is still finishing. Recorded now so the design is fixed
before the first line of `ml_macd/features.py` is written, per the
same "decide before you build" pattern as every other section here.

**No default persistence.** Features/labels are computed per symbol,
in memory, at training time, then released — NOT cached to Supabase or
disk by default. They're fully deterministic from candles + a
config_hash, so caching buys little while the feature set is still
being iterated on (every change invalidates the cache anyway, making
it dead weight more often than it's a speedup). Process ONE symbol at
a time, compute, use, release — keeps peak RAM bounded regardless of
how many symbols the pipeline eventually covers. **All feature columns
use `float32`, not `float64`** — indicator values don't need double
precision, and it halves memory outright. An **optional** `--cache-dir`
flag will write Parquet when a stable feature set makes caching worth
it — off by default, and the directory will be gitignored.

**Cloudflare R2, when persistence IS wanted.** 10GB free permanently,
zero egress, S3-compatible — ~20x Supabase's free tier with no local
disk footprint. The storage layer will be written behind one interface
with three backends — `memory` (default), `local_parquet`, and
`r2` (via `boto3`/`pyarrow` S3FS) — same call site, swapped by config,
not by code change. **R2 itself is not implemented yet** — only the
interface shape is fixed now, specifically so wiring in the real
backend later is a config change, not a refactor. R2 credentials go in
`.env` only, never committed, matching every other credential in this
project.

**Supabase's role stays fixed**: `macd_candles` (already backfilled —
Stages 1-5 — do not touch), `macd_predictions`, `macd_model_runs`.
Nothing else lands there — features/labels are memory-or-R2 territory,
not Supabase's, per the 384MB-of-500MB reality above.

**Committed measurement, not a guess**: after Phase 2's feature
builder exists, report peak RAM per symbol, wall-clock to compute
features for one symbol, and total wall-clock across all symbols (the
brief says 11; the actual registry is 10 crypto + 2 FX = 12 — will
verify and report the real count when Phase 2 runs, not silently
reconcile it here ahead of time). If recompute time turns out
annoying at that measured number, that's what justifies building the
`--cache-dir`/R2 path for real — not before.

## 14. Stage 5 — forex full depth (real writes)

`EUR/USD` and `GBP/USD`, all three timeframes (15m, 1h, 4h — both
pairings, per §11's decision), `2020-01-30` (the real confirmed floor,
identical across all three timeframes) through present.

**A 429 crash on the very first call, root-caused and fixed**: the
first attempt died immediately with `HTTPError 429` from Twelve Data.
Root cause: `TwelveDataProvider`'s pacing is per-PROCESS only, and
several separate probe scripts had hit the same API key minutes
earlier — each individually paced itself correctly, but nothing
tracked elapsed time *across* those processes, so the combined
real-world call rate burst past the per-minute credit cap. Confirmed
zero rows had been written before the crash (`backfill_forex()` writes
once at the very end — see §12's known-limitation writeup), fixed
`_call()` to catch `429` and back off 65s with retry (up to 5
attempts) instead of crashing, and restarted clean. One retry fired
during the successful run (`GBP/USD 15min`, attempt 1/5) and resolved
itself.

**Duplicate rows — different mechanism from Stage 3/4's, but the same
universal dedup (§10) caught it automatically**: every single
symbol/timeframe pull had internal duplicates — 33 for every 15m
series, 8 for every 1h series, 2 for every 4h series, identical counts
across both symbols for the same timeframe. Not sporadic: `33 = 34
pages − 1`, `8 = 9 pages − 1`, `2 = 3 pages − 1` — Twelve Data's
walk-backward pagination includes the boundary bar in both the page
that ends there and the page that starts there, so every page
transition produces exactly one duplicate row. Structurally different
from the AVAX archive-internal duplicate, but caught by the same
`upsert_candles()` dedup without any FX-specific code — confirming
that defense-in-depth was the right call, not overcaution.

**A second real finding — not an error, a vendor grid convention**:
`misaligned_bars()` flagged 5,761/10,852 EUR/USD 4h bars (53%) and
5,770/10,869 GBP/USD 4h bars, and ZERO on 15m/1h for both symbols.
Investigated rather than reported as-is: hour-of-day sampling shows
January 2021 4h bars grid on `{00,04,08,12,16,20}` UTC, July 2021 grids
on `{01,05,09,13,17,21}` UTC — **Twelve Data's FX 4h candle grid
shifts by exactly 1 hour with US DST**, evidently anchored to a fixed
session time in a DST-observing timezone (New York) rather than a
fixed UTC origin. `misaligned_bars()`'s fixed-origin (`epoch_origin_s=0`)
assumption is simply the wrong tool for this series — this is a
deliberate, consistent vendor convention, not a data-quality problem,
and is directly relevant to PART 2's session/holiday-calendar features
later (the 4h grid itself is DST-aware; naive UTC-hour features would
need to account for this). Not fixed here — recorded as a fact
`ml_macd/features.py` will need when it gets to FX session features.
A small residual (~1-150 count) of other hours in the distribution are
the DST-transition weeks themselves — not individually investigated
further.

**Reconciliation, all 6 pairs** — `written + missing = expected`:

| Symbol | TF | Written | Missing | Dup | Expected | Balance |
|---|---|---|---|---|---|---|
| EUR/USD | 15m | 169,087 | 60,942 | 33 | 230,029 | ✓ |
| EUR/USD | 1h | 42,415 | 15,092 | 8 | 57,507 | ✓ |
| EUR/USD | 4h | 10,852 | 3,559 | 2 | 14,377 | off by 34 |
| GBP/USD | 15m | 169,226 | 60,782 | 33 | 230,008 | ✓ |
| GBP/USD | 1h | 42,474 | 15,028 | 8 | 57,502 | ✓ |
| GBP/USD | 4h | 10,869 | 3,540 | 2 | 14,375 | off by 34 |

**The two 4h "mismatches" are a measurement artifact, not lost data** —
direct consequence of the DST finding above: both `expected_slots`
(`(max−min)/interval + 1`) and `bars_missing_before()` assume a
uniform fixed interval, which is false for a DST-shifting grid. Every
DST transition inserts a real ±1-hour jump the fixed-interval math
can't represent correctly. Quantified rather than dismissed: the US
observed **13 DST transitions** in the `2020-01-30`→present window;
34 bars ÷ 13 ≈ **2.6 bars per transition**, consistent with small edge
effects at each transition, not a sign of lost or corrupted data. A
DST-aware reconciliation formula would resolve this exactly, but isn't
built — noted as a follow-up alongside the DST-grid finding itself,
not a blocker.

**Total FX rows written: 444,923.** 15m/1h gap counts (`gap_segments`:
EUR/USD 523/349, GBP/USD 551/357) are notably higher as a fraction of
total bars than crypto's — expected for spot FX, which has real
weekend/holiday closures crypto doesn't; not investigated bar-by-bar
here since PART 2's session/holiday-calendar work is the right place
to characterize FX's closure pattern properly, not this ingestion
report.

**Volume verified NULL, not zero, per §9's design intent** — checked
directly, not assumed: zero non-null `volume` rows found across all 6
symbol/timeframe pairs (sampled up to 5 per pair, all returned empty).
Also spot-checked `number_of_trades`, `taker_buy_base_volume`,
`taker_buy_quote_volume` across all forex rows — zero non-null in
every case, confirming the whole Binance-specific column group is
correctly NULL for FX, not just `volume` alone.

**Fold feasibility, PART 1's largest label horizon (N=48), purge+embargo
(embargo=24, `ml/validation.py` defaults otherwise)** — the check
that decides whether FX is a real second asset class or just a
sanity check:

| Series | Bars | 5-fold | 10-fold | 20-fold | CPCV(6,2) |
|---|---|---|---|---|---|
| EUR/USD 15m | 169,087 | ✓ (~33.7k/fold) | — | — | — |
| EUR/USD 1h | 42,415 | ✓ (~8.4k/fold) | — | — | — |
| EUR/USD 4h | 10,852 | ✓ (~2.1k/fold) | ✓ (~1.0k/fold) | ✓ (~514/fold) | 15 folds |
| GBP/USD 15m | 169,226 | ✓ (~33.7k/fold) | — | — | — |
| GBP/USD 1h | 42,474 | ✓ (~8.4k/fold) | — | — | — |
| GBP/USD 4h | 10,869 | ✓ (~2.1k/fold) | ✓ (~1.0k/fold) | ✓ (~514/fold) | 15 folds |

**FX is a real second asset class, not just a sanity check.** Even the
tightest series (4h, ~10,850 bars) comfortably clears the standard
5-fold walk-forward with thousands of test bars per fold, and still
supports 20 splits or 15 CPCV combinatorial folds with hundreds of
bars per fold — nowhere close to infeasible. BTC/USDT checked for
context at the same settings: 1h and 4h both also clear 5-fold
cleanly, confirming the FX numbers aren't being graded on an unfairly
easy curve.

## 15. Laptop-independence — model artifacts, working state, rebuild story

Checked directly rather than assumed, per the requirement that
everything be reproducible from off-machine state. Two real gaps
found; both are being fixed in this same pass, not deferred.

### Model artifact storage (design — Phase 4/training hasn't started yet)

A fitted LightGBM model + its calibrator represents real, expensive
work (hours of training against a specific config) that recomputing
features does not — so it gets the same three-backend interface as
§13's feature cache (`memory` doesn't apply here; a model that's never
saved isn't useful), extended with:

- **Key naming**: `models/{symbol}/{config_hash}_{timestamp}.{ext}` —
  config_hash and timestamp both in the key itself, not just in
  metadata, so a model file is traceable to the exact run that
  produced it even if separated from its DB row.
- **`macd_model_runs`** (Supabase — not yet created; this table is a
  Phase 4 deliverable, added to `supabase/migrations.sql` when
  training exists) records the artifact key alongside every metric
  already specified elsewhere in this file (config hash, seeds, data
  snapshot boundaries, per-fold thresholds, coverage-precision curve,
  ablations) — so a model is locatable from the DB alone, and the DB
  row is meaningless without the artifact it points to. Neither is
  sufficient by itself; both together are the actual record.
- **Backends**: `local` (default — `ml_macd/cache/models/`, gitignored,
  see below) and `r2` (same interface, swapped by config, not
  implemented yet — same "interface now, backend later" approach as
  §13's feature cache).

### Working state — `.gitignore`, verified not assumed

Ran `git status` and `git add -n` directly rather than trust that
things were probably fine. Found: **`ml_macd/` and `scripts/` were
entirely untracked — nothing from this whole build (README, source,
everything) had ever been committed.** That's the actual
laptop-independence gap this requirement exists to catch, not a
hypothetical one — fixed here, but committing is still a decision for
you to make (see below), not something done unilaterally.

Also found two data-dump files not covered by the existing `*.log`
pattern (`ml_macd/stage3_gap_report.json`, `scripts/probe_twelvedata.out`)
— added explicit `.gitignore` entries for both, plus
`ml_macd/cache/` ahead of Phase 2/4 actually creating it. Added
`ml_macd/requirements.txt` too, which didn't exist — grepped every
import across every `ml_macd/*.py` file rather than guess: the
ingestion path (`providers.py`/`data.py`/`verification.py`/
`gap_handling.py`) needs only `numpy` beyond stdlib. The two
diagnostic scripts (`indicator_parity_check.py`,
`warmup_convergence_check.py`) additionally need `../ml/requirements.txt`
(scikit-learn) and a working `node` on `PATH` — neither is required
to backfill or verify candles, only to re-run those two specific
proofs.

Verified the fix with `git add -n ml_macd/ scripts/ .gitignore`
(dry-run, nothing actually staged): exactly source + `README.md` +
`requirements.txt` + the three small historical driver scripts
(`stage4_driver.py`, `stage4_resume.py`, `stage5_driver.py`) +
`scripts/probe_twelvedata.py` would be added — every log file, JSON
dump, and `__pycache__` correctly excluded.

**Not done: nothing is actually committed yet.** Fixing `.gitignore`
doesn't commit anything by itself — that's a real, user-facing action
this file's own operating conventions require asking about first, not
assuming. Say the word and it happens; until then, everything above is
still only protected by this one local checkout.

### What lives where

```
Supabase   -> candles (macd_candles), predictions (macd_predictions,
              not yet created), model run metadata (macd_model_runs,
              not yet created)
git        -> all code and documentation (once actually committed —
              see above)
R2 / local -> model artifacts, optional feature cache (--cache-dir)
nowhere    -> features and labels — recomputed on demand, every time,
              by design (section 13)
```

### Rebuild from scratch on a new machine — honest about what works today

```
git clone <repo>
pip install -r ml_macd/requirements.txt
cp .env.example .env   # fill in SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY,
                        # TWELVE_DATA_API_KEY, BINANCE_HOST (optional)
```

Then, only if starting against a FRESH Supabase project (an existing
one already has `macd_candles` populated and needs nothing further):
paste `supabase/migrations.sql` into the Supabase SQL editor, then
re-run the Stage 1-5 backfills (`ml_macd/data.py` for crypto/forex
live+backfill modes, or the three small driver scripts for the exact
full-depth runs already documented in sections 7/9/14). This is real
wall-clock time and API calls, not instant — but cheap, per the actual
measured costs in this file (crypto backfill is effectively free via
`data.binance.vision`; FX full depth for both configured symbols was
~110 Twelve Data calls total, single-day even on the free tier).

**Where this story currently stops, said plainly rather than
overclaimed**: candles are the only thing that rebuilds today.
Features, labels, training, serving, and model artifacts are all
still design (sections 13/15) — none of that code exists yet. A
"rebuild from scratch" right now gets you back to exactly where
Stage 5 left off (a fully populated `macd_candles`), not a trained
model — because nothing past ingestion has been built.

## 16. Phase 2 — features and labels

Built and tested against real data. Stopping here, before training,
per the original plan.

`ml_macd/macd_features.py` — the full PART 1 feature set (44 columns):
MACD core (normalized by both close and ATR, `macd_above_zero`
regime flag), crossover dynamics (`bars_since_cross`,
`cross_direction`, ATR-normalized price move since cross), histogram
gap dynamics (1/3/5-bar slope + second difference, `hist_ratio`
bounded to [0,1] since last cross, `bars_since_hist_peak`),
multi-timeframe (HTF regime + cross direction, as-of aligned to the
last CLOSED higher-timeframe bar, `htf_alignment`), divergence
(explicitly defined as a trailing-window price-extreme-without-
momentum-confirmation flag, not a general swing detector), order flow
(crypto-only, exactly 0 — never NaN — for forex), and context (Wilder
RSI, Bollinger width, VWAP distance gated to non-forex per PART 2's
binding conflict resolution, ATR percentile rank over 200 bars,
cyclical hour, one-hot session/day-of-week). PART 2 additions:
`gap_size_atr`, `is_session_open_bar` (proxied from
`bars_missing_before` — no real FX holiday calendar exists in this
project, disclosed rather than faked), and `bars_missing_before`/
`is_post_gap_bar` passed straight through from `gap_handling.py`,
not recomputed. `bars_until_session_close` is NOT built — same
missing-holiday-calendar reason — always NaN, disclosed rather than
silently omitted from the column list.

`ml_macd/labels.py` — the N×band grid (`{4,8,12,24,48} x
{0.25,0.5,1.0,1.5}`), gap-aware for real: every label's validity comes
directly from `gap_handling.py::label_validity_by_timestamp()`, not
reimplemented. Reports class balance and drop rate per grid cell —
"which combos are most predictable" (PART 1) needs a trained model
and stays deferred; this reports what needs no model.

### Three real bugs found while testing against real data

**1. A stationarity violation in `hist_slope_1/3/5`/`hist_slope_2nd_diff`** —
caught by a plain range sanity check (values up to ~1000, not the
expected roughly-unit scale). Root cause: computed from the raw,
unnormalized MACD histogram (price units), not `hist_norm_atr` — on
BTC this silently encodes price level into a feature the file's own
opening rule explicitly forbids ("never feed raw price"). A model
using this feature would not transfer from BTC at $4k to BTC at $90k.
Fixed by computing the slope from `hist_norm_atr` instead; re-verified
range is now consistently near [-1, 1].

**2. A module-name collision that silently loaded the wrong data —
found and fixed twice, the second time more subtle.** First:
naming the new feature file `ml_macd/features.py` collided with
`ml/features.py`'s module name (this file legitimately needs both on
`sys.path` at once, since it imports the sibling module's primitives)
— renamed to `macd_features.py` before ever testing, with the reason
recorded in its own docstring. Second, worse because it passed an
initial smoke test before failing: `ml_macd/data.py` ALSO collides
with `ml/data.py`, and `gap_handling.py`'s own unconditional
`sys.path.insert(0, ML_DIR)` — which ran every time gap_handling.py
was imported, including as a dependency of `labels.py` — silently
re-pushed `ml/` ahead of `ml_macd/` on `sys.path` mid-way through
`labels.py`'s own execution. The result: `import data as ml_data`
inside `labels.py`'s self-test resolved to `ml/data.py`'s unrelated
`load_candles()` (a local `.npz`-cache loader) instead of
`ml_macd/data.py`'s Supabase-backed one — surfaced as a bare
`TypeError` deep inside `ml/data.py`, not an obvious naming bug.
**Root cause turned out to be dead code**: `gap_handling.py` doesn't
actually import anything from `ml/` at all — the `sys.path` lines were
unnecessary and simply deleted. Fixed more broadly too, since the
same class of bug could recur: every file that inserts `ML_DIR` now
does so idempotently, and every file that needs `ml_macd/`'s own
directory on `sys.path` now removes-then-reinserts it at position 0
as the last step, so the final order is deterministic regardless of
which file gets imported first. Verified with a direct stress test:
importing `labels` then `data` fresh in the same process now
correctly resolves to `ml_macd/data.py`, and `macd_features` +
`labels` imported together (the realistic real-usage pattern) both
resolve correctly too.

**3. The ctypes-based RAM measurement itself returned 0.0 on the
first attempt** (see below) — `GetProcessMemoryInfo` "succeeded" while
leaving every field zero, because `ctypes.windll.psapi`'s default
argtypes/restype misinterpret pointer/return types on 64-bit Windows.
Fixed by setting them explicitly.

### Verification against real data

- **No-lookahead** (`assert_no_lookahead()`, same truncate-and-recompute
  method as `ml/features.py`): passes on BTC/USDT (crypto) and
  EUR/USD (forex) — the two paths that exercise materially different
  code (order-flow/VWAP gating). Tolerance relaxed to `1e-6` from
  `ml/`'s `1e-9`: `build_features()` returns `float32` (storage
  requirement below), which only has ~7 significant digits, and `1e-9`
  would risk flagging ordinary rounding noise as a leak.
- **Multi-timeframe alignment**, spot-checked directly against a
  manual as-of computation at 8 bars spanning the full BTC/USDT
  history (accounting for the shift-by-1 — `X[i]` reflects bar `i-1`'s
  raw computation, so the manual check used bar `i-1`'s own
  `open_time`, not bar `i`'s): 8/8 correct.
- **Label threshold test**: a deliberately constructed series with a
  clear up move, clear down move, and a move inside the flat band,
  each at a known ATR — all three classify correctly at the
  ATR-scaled boundary.
- `gap_handling.py`'s own tests (already passing, not re-tested here)
  cover the gap-exclusion logic `labels.py` calls into.

### Measured, not guessed — RAM and wall-clock

Per the storage requirement: measured directly (native Windows
`GetProcessMemoryInfo` via `ctypes`, no new dependency added — see
bug #3 above), one fresh subprocess per symbol so peak working set is
a true per-symbol number, not cumulative across a batch:

| Symbol | Bars | Peak RSS (MB) | Load (s) | Features (s) | Labels (s) | Total (s) |
|---|---|---|---|---|---|---|
| BTC/USDT | 78,373 | 190.8 | 38.8 | 5.5 | 2.1 | 46.4 |
| ETH/USDT | 78,373 | 191.6 | 39.6 | 5.5 | 2.2 | 47.3 |
| SOL/USDT | 52,319 | 141.1 | 27.2 | 3.8 | 2.3 | 33.2 |
| XRP/USDT | 72,169 | 178.6 | 37.5 | 2.6 | 1.1 | 41.2 |
| ADA/USDT | 72,581 | 179.1 | 36.5 | 4.4 | 1.8 | 42.7 |
| DOGE/USDT | 61,961 | 158.4 | 33.5 | 3.9 | 1.5 | 38.9 |
| AVAX/USDT | 51,311 | 136.9 | 28.2 | 3.0 | 1.2 | 32.4 |
| LINK/USDT | 66,027 | 166.1 | 34.9 | 2.6 | 1.2 | 38.7 |
| LTC/USDT | 75,549 | 184.5 | 39.0 | 4.6 | 1.9 | 45.5 |
| DOT/USDT | 52,134 | 140.7 | 27.1 | 3.2 | 1.3 | 31.5 |
| EUR/USD | 42,415 | 115.3 | 20.8 | 2.5 | 1.1 | 24.4 |
| GBP/USD | 42,474 | 115.6 | 21.7 | 2.4 | 1.1 | 25.2 |

**Peak RAM: 115–192MB per symbol** (both entry + HTF series loaded,
44-column float32 feature matrix, full label grid) — well within
"bounded," never approaching a concern on any machine this would
realistically run on. **Total wall-clock, all 12 symbols, sequential
one-process-each: ~447s (~7.5 minutes).**

**The number that actually matters for the caching decision**: load
time (Supabase network round-trip, averaging ~33s/symbol) dominates
total time by roughly 6:1 over compute (`features` + `labels`
combined average ~5.3s/symbol). **A feature cache would mostly be
caching against network latency, not recomputation cost** — a
materially different justification than "recompute is slow," worth
weighing directly against `--cache-dir`'s own cost (staleness risk
while the feature set is still being iterated on, per section 13)
when that decision actually comes up. At ~5.3s of real compute per
symbol, recompute is not remotely "annoying" yet.

## 17. Stationarity audit — every one of the 44 features, checked

Prompted directly by bug #1 in section 16 (the histogram-slope bug):
a range check alone only catches a stationarity violation that
produces an ABSURD number. A feature that silently encodes price or
volume level but happens to land in a plausible-looking range would
pass every range check ever written and still teach a model "what
asset is this" instead of momentum — invisible to the
coverage-precision curve, which would just quietly cap below its true
ceiling with no error anywhere.

**Method**: for every feature, classify it by construction (raw /
normalized / bounded / gated), then check empirically — not just
reason about it — using real backfilled data at three genuinely
different price/volume scales: BTC/USDT (~$2.9k–126k), EUR/USD
(~0.95–1.23), and DOGE/USDT (~$0.001–0.74, with volume in the
BILLIONS of DOGE/hour vs. BTC's thousands of BTC/hour — the volume
comparison needed its own third asset, since context-gated
crypto-only features are correctly 0 on FX by design and a crypto-vs-FX
comparison alone can't test them).

| # | Feature | Design | Normalized by | Cross-asset check |
|---|---|---|---|---|
| 1 | `macd_norm_close` | normalized | close | PASS — BTC/DOGE ranges comparable despite 100,000x price gap; EUR narrower reflects FX's genuinely lower volatility, not a leak |
| 2 | `signal_norm_close` | normalized | close | PASS (same reasoning as #1) |
| 3 | `hist_norm_close` | normalized | close | PASS |
| 4 | `macd_norm_atr` | normalized | ATR | PASS — ranges closely comparable across all three (~±3) |
| 5 | `signal_norm_atr` | normalized | ATR | PASS |
| 6 | `hist_norm_atr` | normalized | ATR | PASS |
| 7 | `macd_above_zero` | bounded | {0,1} | PASS (boolean) |
| 8 | `bars_since_cross` | bar count | — | PASS — not price-related by construction; comparable ranges (0–65 to 0–83) confirm no hidden price dependence |
| 9 | `cross_direction` | bounded | {-1,0,1} | PASS |
| 10 | `price_move_since_cross_atr` | normalized | ATR | PASS — comparable ranges (~±11–13) |
| 11 | `hist_slope_1` | normalized | ATR (via hist_norm_atr) | **FIXED** — was raw histogram (bug #1); now comparable ranges (~±0.5) |
| 12 | `hist_slope_3` | normalized | ATR | FIXED (same root cause as #11) |
| 13 | `hist_slope_5` | normalized | ATR | FIXED |
| 14 | `hist_slope_2nd_diff` | normalized | ATR | FIXED |
| 15 | `hist_ratio` | bounded | [0,1] by construction (ratio to its own running max) | PASS — provably bounded, not just empirically |
| 16 | `bars_since_hist_peak` | bar count | — | PASS — comparable ranges (0–46 to 0–57) |
| 17 | `htf_macd_above_zero` | bounded | {0,1} | PASS |
| 18 | `htf_cross_direction` | bounded | {-1,0,1} | PASS |
| 19 | `htf_alignment` | bounded | {0,1} | PASS |
| 20 | `bearish_divergence` | bounded | {0,1} — a flag, not a magnitude | PASS — divergence is deliberately boolean in this design (see section 16), so there is no raw magnitude to leak |
| 21 | `bullish_divergence` | bounded | {0,1} | PASS |
| 22 | `taker_buy_ratio` | ratio of same-unit quantities | taker_buy_base / volume (units cancel) | PASS — structurally scale-free; 0 for forex by design (context-gated) |
| 23 | `taker_buy_ratio_slope_5` | ratio-of-ratio | — | PASS |
| 24 | `trades_per_unit_volume_z` | **was raw, now z-scored** | rolling 48-bar mean/std (own history) | **FAILED, FIXED** — see below, the one real finding |
| 25 | `rsi_14_c` | bounded | [-0.5, 0.5] | PASS |
| 26 | `bb_width` | normalized | close (via SMA mid) | PASS — DOGE's much wider range (max 2.64 vs. BTC's 0.57) reflects genuine extreme volatility (DOGE's real pump history), not unit-encoding; structurally a coefficient-of-variation, provably scale-free with respect to price level |
| 27 | `vwap_distance` | log-ratio | close/vwap | PASS — scale-free by construction; 0 for forex (context-gated, PART 2 binding decision) |
| 28 | `atr_pctile_200` | bounded | [0,1], percentile rank | PASS |
| 29 | `hour_sin` | bounded | [-1,1] | PASS |
| 30 | `hour_cos` | bounded | [-1,1] | PASS |
| 31–33 | `session_asia/london/ny` | bounded | {0,1} | PASS |
| 34–40 | `dow_mon`…`dow_sun` | bounded | {0,1} | PASS |
| 41 | `gap_size_atr` | normalized | ATR | PASS — EUR's wider range (up to ±5 vs. crypto's ±1) reflects real FX weekend-gap risk crypto structurally doesn't have, not a leak |
| 42 | `is_session_open_bar` | bounded | {0,1}, proxy | PASS |
| 43 | `bars_missing_before` | bar count | — | PASS — different max values across assets (EUR 70 vs. DOGE 8) reflect genuine differences in how often each vendor/asset has gaps, not price-encoding |
| 44 | `is_post_gap_bar` | bounded | {0,1} | PASS |

**The one real failure: `trades_per_unit_volume_z` (was `trades_per_unit_volume`)**.
Empirically, before the fix: BTC max `719.5` vs. DOGE max `0.00386` —
a **~186,000x** gap on the same nominal feature. Root cause: the raw
ratio `number_of_trades / volume` is not scale-free across symbols —
`volume` is denominated in base-asset units, which differ by orders
of magnitude (BTC: thousands of BTC/hour; DOGE: billions of
DOGE/hour), so the raw ratio silently encoded which asset it was,
not a market-behavior signal. This is precisely the failure mode the
requirement warned about: a plausible-looking number (not visibly
absurd like the ~1000-scale histogram bug) that would have sailed
through any range check.

**Fix**: rolling 48-bar z-score against the symbol's OWN history —
same window and pattern as `ml/features.py`'s existing `volume_z`
(clipped to ±5, undefined-or-zero-std collapses to 0, matching that
file's established convention rather than inventing a new one).
Column renamed `trades_per_unit_volume_z` so the transformation is
visible in the name, not just a code comment. Re-verified directly:
BTC and DOGE both now land in `[-5, 5]` with near-identical means
(`0.024` vs. `0.027`) — genuinely comparable distributions, not just
a coincidentally-matching range. Re-ran the full test suite after the
fix: no-lookahead still passes on both BTC/USDT and EUR/USD, and the
label threshold test is unaffected (it doesn't touch this feature).

**Nothing else required a fix.** The wide range DIFFERENCES noted
above (DOGE's `bb_width`, EUR's `gap_size_atr`) are real market
structure — different assets and asset classes genuinely have
different volatility/gap characteristics — not price or volume level
leaking through a normalization that should have cancelled it. The
distinguishing test throughout: does the SAME feature, on the SAME
asset, look reasonable at wildly different price levels (BTC at
~$3k vs. ~$126k, or DOGE at ~$0.001 vs. ~$0.74) — not whether every
asset produces an identical distribution, which would be the wrong
bar (real volatility differences are information, not noise).

**Two implications of the fix, recorded so they aren't rediscovered:**

1. **Warmup — confirmed, not assumed.** `trades_per_unit_volume_z`
   adds its own 48-bar rolling warmup, separate from the EMA/ATR/RSI
   recursive warmup section 1 derived `warmup_bars=260` for. Tested
   directly rather than reasoning "260 >> 48 so it's probably fine":
   built features from the full BTC/USDT history, then again from
   only the trailing 260 bars (simulating a live-serving buffer sized
   at `warmup_bars`), and compared this feature's value at the final
   bar — **exact match, `diff = 0.0`**, not just within tolerance.
   This is actually a stronger guarantee than the EMA-based features
   get: a plain rolling mean/std (unlike a recursive EMA) has ZERO
   approximation error once enough prior bars exist — 48 bars is a
   hard, exact requirement, not an asymptotic convergence question,
   and 260 clears it with 212 bars to spare. `warmup_bars=260` fully
   absorbs this feature with no change needed.
2. **Semantic note — read this before using the feature.**
   Z-scoring against the symbol's OWN trailing 48 bars means
   `trades_per_unit_volume_z` encodes "trade intensity relative to
   THIS symbol's recent regime," not an absolute level and not a
   cross-asset-comparable one the way the raw ratio's NAME might
   suggest. That's the correct choice for stationarity (section 17's
   whole point), but it is a genuinely different meaning than the raw
   ratio had — a value of `+2` means "unusually high for this symbol
   lately," not "high in some universal sense." Don't read it as the
   latter later.

## 18. Label construction — same units-don't-cancel trap, checked

The same class of bug that broke `trades_per_unit_volume` could
equally have broken the LABEL itself — a $500 BTC move and a 0.005
EUR/USD move can both be "one ATR," but as raw price deltas they are
wildly different absolute numbers; a model trained on absolute deltas
would learn asset identity through the target, not just the features,
which no amount of feature-side stationarity work would catch.

Checked directly in `ml_macd/labels.py::build_label()`:
- **Forward return** (line 71): `fwd_return = logp[i+N] - logp[i]` —
  `logp = np.log(close)`, so this is `log(close[i+N] / close[i])`, a
  LOG-RETURN. Scale-free by construction: a log-return doesn't care
  whether `close` is 1.08 or 91,000.
- **Threshold** (lines 76–77): `atr_frac = atr14 / close`;
  `threshold = band * atr_frac` — ATR expressed as a FRACTION of
  price, not raw ATR. Also scale-free.
- The classification itself (`fwd_return > threshold` /
  `fwd_return < -threshold`) therefore compares two already-scale-free
  quantities. **No raw price delta anywhere in label construction** —
  confirmed by reading the actual code, not inferred from intent.
