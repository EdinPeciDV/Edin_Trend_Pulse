"""
ml_macd/data.py
===================================================================
Ingestion + backfill orchestration (PART 1 deliverable 1). Wraps
ml_macd/providers.py's BinanceProvider/TwelveDataProvider with:

  - the closed-candle-only guard ("never store a forming candle")
  - idempotent Supabase upsert into public.macd_candles
    (supabase/migrations.sql section 9 — MUST be applied in the
    Supabase SQL editor before this can write anything real; see
    upsert_candles()'s docstring)
  - a small CLI for backfill and incremental ingest

Supabase writes go through raw PostgREST REST calls (urllib, no new
dependency) rather than the supabase-py client — matching ml/'s
deliberately minimal-dependency philosophy (ml/requirements.txt is
numpy + scikit-learn only) and this project's own established
direct-REST pattern for service-role writes.

TIMEFRAME PAIRING (PART 1): each symbol is ingested at its entry
timeframe AND at 4x it, for the multi-timeframe features:
  15m entry -> 1h higher
  1h entry  -> 4h higher
===================================================================
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

from providers import (
    BinanceProvider, TwelveDataProvider, load_env_var,
)

# Match supabase/migrations.sql section 9's check constraints exactly —
# these are the source of truth on both sides, kept as one named
# constant each rather than typed as string literals more than once.
ASSET_CLASS_CRYPTO = "crypto"
ASSET_CLASS_FOREX = "forex"   # NOT 'fx' — matches market_snapshots'
                              # existing convention. An earlier draft
                              # of this build's brief said 'fx'; the
                              # schema (and this code) uses 'forex'.
ASSET_CLASS_STOCK = "stock"
ALLOWED_ASSET_CLASSES = (ASSET_CLASS_CRYPTO, ASSET_CLASS_FOREX, ASSET_CLASS_STOCK)

# No '1d': nothing in this file fetches or writes it. Widen this in the
# same change that adds that code path, not ahead of it — see the
# column comment on macd_candles.timeframe in the migration.
ALLOWED_TIMEFRAMES = ("15m", "1h", "4h")

ENTRY_TO_HIGHER_TF = {"15m": "1h", "1h": "4h"}
TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000, "4h": 4 * 60 * 60 * 1000,
}

CRYPTO_SYMBOLS = {  # ml_macd symbol -> Binance REST/vision symbol
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
}
FOREX_SYMBOLS = {  # ml_macd symbol -> Twelve Data symbol (identical here, kept explicit)
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
}

SUPABASE_URL = load_env_var("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = load_env_var("SUPABASE_SERVICE_ROLE_KEY")


def now_ms():
    return int(datetime.now(tz=timezone.utc).timestamp() * 1000)


def drop_forming_candle(bars, timeframe):
    """
    The single most important guard in this file: a candle is only
    usable once it has CLOSED. `bars` is sorted ascending by
    open_time_ms; drop the trailing bar(s) whose close time
    (open_time + timeframe duration) is still in the future.
    """
    duration = TIMEFRAME_MS[timeframe]
    cutoff = now_ms()
    return [b for b in bars if b["open_time_ms"] + duration <= cutoff]


def to_iso(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bars_to_rows(bars, symbol, asset_class, timeframe, source):
    if asset_class not in ALLOWED_ASSET_CLASSES:
        raise ValueError(f"asset_class {asset_class!r} not in {ALLOWED_ASSET_CLASSES}")
    if timeframe not in ALLOWED_TIMEFRAMES:
        raise ValueError(f"timeframe {timeframe!r} not in {ALLOWED_TIMEFRAMES}")

    rows = []
    for b in bars:
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "timeframe": timeframe,
            "open_time": to_iso(b["open_time_ms"]),
            "open": b["open"], "high": b["high"], "low": b["low"], "close": b["close"],
            "volume": b["volume"],
            "number_of_trades": b["number_of_trades"],
            "taker_buy_base_volume": b["taker_buy_base_volume"],
            "taker_buy_quote_volume": b["taker_buy_quote_volume"],
            "volume_is_proxy": asset_class == ASSET_CLASS_FOREX,
            "adjusted": True if asset_class == ASSET_CLASS_STOCK else None,
            "source": source,
        })
    return rows


def query_max_open_time(symbol, timeframe, asset_class):
    """
    For --resume: the latest recorded open_time already in
    macd_candles for (symbol, timeframe), or None if there are no rows
    yet (fresh symbol — caller falls back to an explicit --start).
    asset_class is included in the filter purely so this can never
    accidentally match a differently-classed row for a symbol string
    that happened to collide (defensive; the unique index doesn't
    include asset_class, so this guards a query, not data integrity).
    """
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set (.env)")

    import urllib.parse as _urlparse
    params = _urlparse.urlencode({
        "symbol": f"eq.{symbol}", "timeframe": f"eq.{timeframe}",
        "asset_class": f"eq.{asset_class}",
        "select": "open_time", "order": "open_time.desc", "limit": "1",
    })
    url = f"{SUPABASE_URL}/rest/v1/macd_candles?{params}"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    })
    with urllib.request.urlopen(req, timeout=30) as resp:
        rows = json.loads(resp.read())
    if not rows:
        return None
    return datetime.fromisoformat(rows[0]["open_time"].replace("Z", "+00:00"))


def resolve_backfill_start(symbol, timeframe, asset_class, requested_start, resume):
    """
    STAGE 3 addition: --resume derives the start point from the DB
    instead of a human re-typing a month off an error message.

    Deliberately resumes from the SAME month as the last recorded row,
    not the month after it — upserts are idempotent on
    (symbol, timeframe, open_time), so re-fetching one month costs
    nothing but a few redundant (harmless) writes, while resuming from
    "the following month" risks a permanent gap if the prior run
    stopped mid-month (e.g. a BackfillIntegrityError partway through a
    file) rather than cleanly at a month boundary. Overlap is cheap;
    a silent seam gap is not.

    Prints the derived resume point before returning — "before writing
    anything" per the brief.
    """
    if not resume:
        return requested_start

    latest = query_max_open_time(symbol, timeframe, asset_class)
    if latest is None:
        print(f"[resume] {symbol} {timeframe}: no existing rows, "
              f"falling back to --start {requested_start}")
        return requested_start

    resume_yyyymm = latest.strftime("%Y-%m")
    print(f"[resume] {symbol} {timeframe}: latest recorded row at "
          f"{latest.isoformat()} -> resuming from {resume_yyyymm} "
          f"(re-fetching that month deliberately, for seam overlap)")
    return resume_yyyymm


def upsert_candles(rows, batch_size=1000):
    """
    Idempotent upsert into public.macd_candles via PostgREST, on the
    (symbol, timeframe, open_time) unique index (migration section 9).

    BUG FOUND DURING STAGE 3 (real backfill, not caught by any earlier
    dry-run or smoke test — every prior real write happened to target
    previously-empty rows, so a genuine conflict was never actually
    exercised until this run revisited January 2025, which Stage 1 had
    already written): `Prefer: resolution=merge-duplicates` alone does
    NOT tell PostgREST which unique constraint to build the
    `ON CONFLICT` clause against — without an explicit `on_conflict`
    query parameter, PostgREST's upsert only targets the table's
    PRIMARY KEY (`id`, a bigserial that can never collide on insert),
    so it silently falls through to a bare INSERT and the database
    itself raises a raw, uncaught `23505 duplicate key` the instant it
    hits the (symbol, timeframe, open_time) unique index instead.
    `on_conflict=symbol,timeframe,open_time` below is what actually
    makes this idempotent, not the `Prefer` header alone.

    REQUIRES supabase/migrations.sql to have been applied in the
    Supabase SQL editor first — this table does not exist until then,
    and every call here will fail with a clear 404/undefined-table
    error until it has been. This function does not apply the
    migration itself; PostgREST has no DDL endpoint by design (same
    reason sysrfx_predictor/README.md documents for its own schema).
    """
    if not rows:
        return 0
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not set (.env)")

    # DEFENSE IN DEPTH, Stage 5: providers.py's _dedupe_by_open_time()
    # only covers the crypto archive-file path (where the AVAX/USDT
    # duplicate-row bug was found and fixed). This table is the true
    # choke point every write path passes through — crypto archive,
    # crypto REST live-increment, and forex — so dedupe here too,
    # rather than assume every future/other source is equally clean.
    # Keeps the first occurrence per (symbol, timeframe, open_time);
    # reports (never silently drops) anything actually removed.
    seen = {}
    deduped = []
    n_dropped = 0
    for row in rows:
        key = (row["symbol"], row["timeframe"], row["open_time"])
        if key in seen:
            n_dropped += 1
            continue
        seen[key] = True
        deduped.append(row)
    if n_dropped:
        print(f"  [upsert_candles] dropped {n_dropped} duplicate (symbol,timeframe,open_time) "
             f"row(s) from this write batch before sending")
    rows = deduped

    url = f"{SUPABASE_URL}/rest/v1/macd_candles?on_conflict=symbol,timeframe,open_time"
    written = 0
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        body = json.dumps(batch).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, method="POST",
            headers={
                "apikey": SUPABASE_SERVICE_ROLE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "resolution=merge-duplicates,return=minimal",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp.read()
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Supabase upsert failed ({e.code}): {e.read()[:500]}") from e
        written += len(batch)
        time.sleep(0.1)
    return written


# ------------------------------------------------------------------ #
# Backfill / ingest orchestration                                     #
# ------------------------------------------------------------------ #

class BackfillIntegrityError(RuntimeError):
    """
    STAGE 2 GUARD: raised (and the whole run halted) when an archive
    file parsed non-empty but contributed zero written rows. A past
    month's archive that parses successfully should ALWAYS yield at
    least one closed candle — drop_forming_candle() can only ever drop
    the single bar nearest "now", never an entire past month. Zero
    written rows from a non-empty parse means something upstream
    silently ate the data (exactly what the undetected microsecond
    bug did), and continuing the run would keep doing it file after
    file with nothing in the summary to show for it.
    """


def backfill_crypto(ml_symbol, binance_symbol, timeframe, start_yyyymm, end_yyyymm, dry_run=False):
    """
    Per-archive-file backfill: fetch, validate, write, and verify ONE
    month at a time (via BinanceProvider.backfill_from_vision_files(),
    a generator — see its docstring for why per-file granularity is
    the point). Halts immediately via BackfillIntegrityError if any
    file's parse-then-write pipeline silently drops all its rows.
    """
    provider = BinanceProvider()
    print(f"[backfill] {ml_symbol} {timeframe} via data.binance.vision {start_yyyymm}..{end_yyyymm}")

    total_written = 0
    for file_result in provider.backfill_from_vision_files(binance_symbol, timeframe, start_yyyymm, end_yyyymm):
        ym, filename, status = file_result["ym"], file_result["filename"], file_result["status"]

        if status == "not_published":
            print(f"  [{filename}] not published, skipping")
            continue

        rows_parsed = file_result["rows_parsed"]
        units = ",".join(sorted(file_result["detected_units"])) or "n/a"
        bars = drop_forming_candle(file_result["bars"], timeframe)
        rows = bars_to_rows(bars, ml_symbol, ASSET_CLASS_CRYPTO, timeframe, provider.source)

        if dry_run:
            print(f"  [{filename}] unit={units} parsed={rows_parsed} closed={len(rows)} DRY RUN, not writing")
            total_written += len(rows)
            continue

        written = upsert_candles(rows)
        print(f"  [{filename}] unit={units} parsed={rows_parsed} written={written}")

        if rows_parsed > 0 and written == 0:
            raise BackfillIntegrityError(
                f"{filename}: parsed {rows_parsed} rows but wrote 0 — halting run. "
                f"Every row before this file is already committed to Supabase; "
                f"fix the cause and resume from {ym}, do not re-run from the start."
            )
        total_written += written

    print(f"[backfill] {ml_symbol} {timeframe}: {total_written} rows total")
    return total_written


def backfill_forex(ml_symbol, td_symbol, timeframe, start_ms, dry_run=False):
    provider = TwelveDataProvider()
    print(f"[backfill] {ml_symbol} {timeframe} via Twelve Data from {to_iso(start_ms)}")
    bars = provider.fetch_candles(td_symbol, timeframe, start_ms=start_ms)
    bars = drop_forming_candle(bars, timeframe)
    rows = bars_to_rows(bars, ml_symbol, ASSET_CLASS_FOREX, timeframe, provider.source)
    print(f"[backfill] {ml_symbol} {timeframe}: {len(rows)} closed candles ready")
    if dry_run:
        print(f"[backfill] DRY RUN — not writing. Sample row: {rows[0] if rows else None}")
        return len(rows)
    written = upsert_candles(rows)
    print(f"[backfill] {ml_symbol} {timeframe}: upserted {written} rows")
    return written


def ingest_latest_crypto(ml_symbol, binance_symbol, timeframe, lookback_bars=10, dry_run=False):
    provider = BinanceProvider()
    start_ms = now_ms() - lookback_bars * TIMEFRAME_MS[timeframe]
    bars = provider.fetch_klines(binance_symbol, timeframe, start_ms=start_ms)
    bars = drop_forming_candle(bars, timeframe)
    rows = bars_to_rows(bars, ml_symbol, ASSET_CLASS_CRYPTO, timeframe, provider.source)
    if dry_run:
        print(f"[live] {ml_symbol} {timeframe}: DRY RUN, {len(rows)} rows")
        return len(rows)
    return upsert_candles(rows)


def ingest_latest_forex(ml_symbol, td_symbol, timeframe, lookback_bars=10, dry_run=False):
    provider = TwelveDataProvider()
    start_ms = now_ms() - lookback_bars * TIMEFRAME_MS[timeframe]
    bars = provider.fetch_candles(td_symbol, timeframe, start_ms=start_ms)
    bars = drop_forming_candle(bars, timeframe)
    rows = bars_to_rows(bars, ml_symbol, ASSET_CLASS_FOREX, timeframe, provider.source)
    if dry_run:
        print(f"[live] {ml_symbol} {timeframe}: DRY RUN, {len(rows)} rows")
        return len(rows)
    return upsert_candles(rows)


def main():
    ap = argparse.ArgumentParser(description="ml_macd candle ingestion")
    ap.add_argument("--mode", choices=["backfill", "live"], default="live")
    ap.add_argument("--asset-class", choices=[ASSET_CLASS_CRYPTO, ASSET_CLASS_FOREX], required=True)
    ap.add_argument("--symbol", required=True, help="e.g. BTC/USDT or EUR/USD")
    ap.add_argument("--timeframe", choices=["15m", "1h"], default="1h",
                    help="entry timeframe; the 4x higher timeframe is fetched automatically")
    ap.add_argument("--start", help="backfill start: YYYY-MM for crypto, ISO date for forex")
    ap.add_argument("--end", help="backfill end: YYYY-MM for crypto (default: current month)")
    ap.add_argument("--dry-run", action="store_true", help="fetch and print, do not write to Supabase")
    ap.add_argument("--resume", action="store_true",
                    help="derive backfill start from max(open_time) already in macd_candles "
                         "for this (symbol, timeframe), instead of --start. Explicit flag, "
                         "not a silent default, so a full re-run is always still available "
                         "by simply omitting it.")
    args = ap.parse_args()

    timeframes = [args.timeframe, ENTRY_TO_HIGHER_TF[args.timeframe]]

    for tf in timeframes:
        if args.mode == "backfill":
            if args.asset_class == ASSET_CLASS_CRYPTO:
                binance_symbol = CRYPTO_SYMBOLS.get(args.symbol, args.symbol.replace("/", ""))
                end = args.end or datetime.now(tz=timezone.utc).strftime("%Y-%m")
                start = resolve_backfill_start(args.symbol, tf, ASSET_CLASS_CRYPTO, args.start, args.resume)
                if not start:
                    print("ERROR: --start YYYY-MM required for crypto backfill "
                         "(or --resume against a symbol/timeframe with existing rows)", file=sys.stderr)
                    sys.exit(1)
                backfill_crypto(args.symbol, binance_symbol, tf, start, end, dry_run=args.dry_run)
            else:
                td_symbol = FOREX_SYMBOLS.get(args.symbol, args.symbol)
                if not args.start:
                    print("ERROR: --start ISO date required for forex backfill", file=sys.stderr)
                    sys.exit(1)
                start_dt = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
                start_ms = int(start_dt.timestamp() * 1000)
                backfill_forex(args.symbol, td_symbol, tf, start_ms, dry_run=args.dry_run)
        else:
            if args.asset_class == ASSET_CLASS_CRYPTO:
                binance_symbol = CRYPTO_SYMBOLS.get(args.symbol, args.symbol.replace("/", ""))
                n = ingest_latest_crypto(args.symbol, binance_symbol, tf, dry_run=args.dry_run)
            else:
                td_symbol = FOREX_SYMBOLS.get(args.symbol, args.symbol)
                n = ingest_latest_forex(args.symbol, td_symbol, tf, dry_run=args.dry_run)
            print(f"[live] {args.symbol} {tf}: {n} rows")


if __name__ == "__main__":
    main()
