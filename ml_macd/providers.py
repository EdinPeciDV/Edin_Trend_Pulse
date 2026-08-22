"""
ml_macd/providers.py
===================================================================
Candle providers, behind one narrow interface: fetch_candles(symbol,
timeframe, start_ms=None, end_ms=None) -> list[bar dict], where a bar
dict always has these keys (value None where the source doesn't have
it — never fabricated):

    open_time_ms, open, high, low, close, volume,
    number_of_trades, taker_buy_base_volume, taker_buy_quote_volume

Two providers:

  BinanceProvider    — crypto. Two delivery mechanisms for the SAME
                        underlying exchange data: REST /api/v3/klines
                        (recent history, live increments) and
                        data.binance.vision monthly archives (bulk
                        backfill). Both report source="binance_spot" —
                        mixing them is not a source-pinning violation
                        (PART 3 correction #2), because it's one
                        vendor's own trade-level ground truth either
                        way, not two different measurements of the
                        same thing.

  TwelveDataProvider — forex. The ONLY FX source, pinned per PART 3
                        correction #2 (never mix vendors within one
                        symbol's history). Confirmed live via
                        scripts/probe_twelvedata.py (2026-08-21):
                        depth is good (2+ years on 15m, 4+ years on
                        1h, free tier) but volume is NEVER returned
                        for FX — number_of_trades/taker_buy_* and
                        volume itself are always None for every bar
                        this provider returns. Do not "fix" that here
                        by substituting a proxy; ml_macd/README.md and
                        the macd_candles migration both document why.
===================================================================
"""

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(ROOT, ".env")


def load_env_var(name, path=ENV_PATH, default=None):
    if not os.path.exists(path):
        return os.environ.get(name, default)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name, default)


# ------------------------------------------------------------------ #
# Binance — crypto                                                    #
# ------------------------------------------------------------------ #

BINANCE_HOST = os.environ.get("BINANCE_HOST", "api.binance.com")
BINANCE_FALLBACK_HOST = "api.binance.us"

# Interval strings Binance's REST API and data.binance.vision both use.
BINANCE_INTERVALS = {"15m": "15m", "1h": "1h", "4h": "4h"}


# Millisecond epoch stays below 10**14 until the year 5138; microsecond
# epoch is already above 10**15 today. Safe threshold between the two
# for any date this project will ever see (used both to detect which
# unit a row is in, and — after normalization — as part of the sanity
# bound below).
_US_MS_THRESHOLD = 10 ** 14

# STAGE 2 GUARD (parse-time sanity bound). The DB CHECK on macd_candles
# (open_time between 2010 and 2100) is the SECOND line of defence, not
# the first — by the time a bad timestamp reaches Postgres it has
# already silently corrupted whatever in-process logic ran before the
# write (e.g. drop_forming_candle() filtering every row of a whole
# archive file out as "not yet closed", which is exactly what the
# undetected microsecond bug did before this guard existed). Reject
# here, at parse time, naming the offending file and raw value.
_MIN_VALID_MS = 1262304000000  # 2010-01-01T00:00:00Z


def _normalize_and_validate_ms(raw_ts, context):
    """
    Returns (open_time_ms, detected_unit) for one raw timestamp value,
    or raises ValueError naming `context` (the archive filename or API
    call the caller is inside) and the offending raw value.

    `detected_unit` is 'ms' or 'us', purely for the caller to log —
    STAGE 2 GUARD: "log the detected unit per archive file, so a future
    format change appears in the logs rather than as missing data."
    """
    ts = int(raw_ts)
    if ts > _US_MS_THRESHOLD:
        ms, unit = ts // 1000, "us"
    else:
        ms, unit = ts, "ms"

    upper_bound_ms = int((datetime.now(timezone.utc).timestamp() + 86400) * 1000)
    if not (_MIN_VALID_MS <= ms <= upper_bound_ms):
        # Formatting `ms` as a date can itself raise (OSError on
        # Windows/some platforms for sufficiently absurd values) —
        # exactly the kind of masking failure this guard exists to
        # avoid, so the fallback below must never let that hide the
        # actual ValueError this function is trying to raise.
        try:
            as_date = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()
        except (OSError, OverflowError, ValueError):
            as_date = "(unrepresentable as a date — value is too extreme to format)"
        raise ValueError(
            f"TIMESTAMP OUT OF BOUNDS in {context}: raw value {raw_ts!r} "
            f"normalized to {ms} ms ({as_date}), "
            f"outside [2010-01-01, now+1day]. Refusing to parse this row — "
            f"this is exactly the failure mode that produced a silent, "
            f"all-rows-filtered backfill before this guard existed."
        )
    return ms, unit


def _binance_kline_row_to_bar(row, context):
    ms, unit = _normalize_and_validate_ms(row[0], context)
    return {
        "open_time_ms": ms,
        "_detected_unit": unit,
        "open": float(row[1]),
        "high": float(row[2]),
        "low": float(row[3]),
        "close": float(row[4]),
        "volume": float(row[5]),
        "number_of_trades": int(row[8]),
        "taker_buy_base_volume": float(row[9]),
        "taker_buy_quote_volume": float(row[10]),
    }


class BinanceProvider:
    source = "binance_spot"

    def _get(self, path, host=None):
        host = host or BINANCE_HOST
        url = f"https://{host}{path}"
        req = urllib.request.Request(url, headers={"User-Agent": "trendpulse-ml_macd/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def fetch_klines(self, symbol, timeframe, start_ms=None, end_ms=None, limit=1000):
        """
        REST pagination. `symbol` e.g. 'BTCUSDT'. Returns closed AND the
        possibly-still-forming final candle — caller (data.py) is
        responsible for dropping any bar not yet closed.
        """
        interval = BINANCE_INTERVALS[timeframe]
        out = []
        cursor = start_ms

        while True:
            params = {"symbol": symbol, "interval": interval, "limit": limit}
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            path = "/api/v3/klines?" + urllib.parse.urlencode(params)

            status, body = self._get(path)
            if status == 451:
                status, body = self._get(path, host=BINANCE_FALLBACK_HOST)
            if status != 200:
                raise RuntimeError(f"Binance klines failed ({status}): {body[:300]}")

            rows = json.loads(body)
            if not rows:
                break
            context = f"REST klines {symbol} {interval}"
            bars = [_binance_kline_row_to_bar(r, context) for r in rows]
            out.extend(bars)

            if len(rows) < limit:
                break
            cursor = bars[-1]["open_time_ms"] + 1
            if end_ms is not None and cursor >= end_ms:
                break
            time.sleep(0.25)  # polite pacing, well under Binance's weight limits

        return out

    def backfill_from_vision_files(self, symbol, timeframe, start_yyyymm, end_yyyymm):
        """
        Bulk monthly archive backfill from data.binance.vision, ONE
        ARCHIVE FILE AT A TIME (a generator, not a single accumulated
        list) — STAGE 2 GUARD: a full-history backfill spans hundreds
        of archive files, and if a subset silently yields nothing it
        will not show up in one final summary line, only weeks later
        as a hole in a feature. Per-file granularity is what lets the
        caller (ml_macd/data.py) write and verify after each file
        instead of after the whole multi-year run.

        `start_yyyymm`/`end_yyyymm` are "YYYY-MM" strings, inclusive.
        Skips (yielding a result with status='not_published', not an
        exception) any month that 404s — a pair that didn't exist yet
        that month, or hasn't had that month's archive published.

        Yields one dict per month attempted:
          {
            ym, filename, status ('ok' | 'not_published'),
            detected_units (set of 'ms'/'us' seen in this file —
              normally exactly one; more than one is itself suspicious
              and worth the caller's attention, not raised here since
              STAGE 2 wants per-row validation to fail loudly on its
              own, not this generator second-guessing it),
            rows_parsed (int), bars (list, only when status='ok'),
          }
        """
        interval = BINANCE_INTERVALS[timeframe]
        months = _month_range(start_yyyymm, end_yyyymm)

        for ym in months:
            filename = f"{symbol}-{interval}-{ym}.zip"
            path = f"/data/spot/monthly/klines/{symbol}/{interval}/{filename}"
            status, body = self._get(path, host="data.binance.vision")

            if status == 404:
                yield {"ym": ym, "filename": filename, "status": "not_published",
                      "detected_units": set(), "rows_parsed": 0, "bars": []}
                continue
            if status != 200:
                raise RuntimeError(f"data.binance.vision failed ({status}) for {ym}: {body[:300]}")

            context = f"{filename}"
            bars = []
            detected_units = set()
            with zipfile.ZipFile(io.BytesIO(body)) as zf:
                names = zf.namelist()
                csv_name = next((n for n in names if n.endswith(".csv")), names[0])
                with zf.open(csv_name) as f:
                    for line in io.TextIOWrapper(f, encoding="utf-8"):
                        cols = line.strip().split(",")
                        if not cols or not cols[0].isdigit():
                            continue  # header row, if the archive ever has one
                        bar = _binance_kline_row_to_bar(cols, context)
                        detected_units.add(bar.pop("_detected_unit"))
                        bars.append(bar)

            bars, dup_report = _dedupe_by_open_time(bars, filename)
            if dup_report["n_duplicates"]:
                kind = "identical" if not dup_report["conflicting"] else "CONFLICTING VALUES"
                print(f"  [{filename}] {dup_report['n_duplicates']} duplicate open_time row(s) "
                     f"found within this archive file ({kind}), kept first occurrence")
                if dup_report["conflicting"]:
                    for ts_ms, a, b in dup_report["conflicting"]:
                        print(f"    CONFLICT at open_time_ms={ts_ms}: kept {a} vs dropped {b}")

            yield {"ym": ym, "filename": filename, "status": "ok",
                  "detected_units": detected_units, "rows_parsed": len(bars),
                  "duplicates_dropped": dup_report["n_duplicates"], "bars": bars}
            time.sleep(0.25)


def _dedupe_by_open_time(bars, filename):
    """
    STAGE 4 FIX: a real Binance archive (AVAXUSDT-4h-2026-05.zip,
    discovered live) contained an exact duplicate row — same
    open_time, same OHLCV — appearing twice. PostgREST's upsert
    cannot apply ON CONFLICT DO UPDATE twice to the same target row
    within one INSERT statement (Postgres error 21000: "ON CONFLICT
    DO UPDATE command cannot affect row a second time"), so a
    duplicate WITHIN one file's own batch crashes the write even
    though the (symbol, timeframe, open_time) unique index is
    perfectly correct and nothing is wrong with the target table.

    Keeps the FIRST occurrence of each open_time_ms, drops the rest.
    Reports (does not silently drop) every duplicate found, and
    distinguishes identical duplicates (safe, just redundant) from
    duplicates with DIFFERING OHLCV values for the same timestamp
    (worth a human's attention — this function still resolves it by
    keeping the first occurrence, but flags it as a conflict rather
    than a harmless repeat).
    """
    seen = {}
    out = []
    n_duplicates = 0
    conflicting = []
    for bar in bars:
        key = bar["open_time_ms"]
        if key not in seen:
            seen[key] = bar
            out.append(bar)
            continue
        n_duplicates += 1
        first = seen[key]
        comparable_keys = ("open", "high", "low", "close", "volume",
                          "number_of_trades", "taker_buy_base_volume", "taker_buy_quote_volume")
        if any(first[k] != bar[k] for k in comparable_keys):
            conflicting.append((key, {k: first[k] for k in comparable_keys},
                               {k: bar[k] for k in comparable_keys}))
    return out, {"n_duplicates": n_duplicates, "conflicting": conflicting}


def _month_range(start_yyyymm, end_yyyymm):
    sy, sm = (int(x) for x in start_yyyymm.split("-"))
    ey, em = (int(x) for x in end_yyyymm.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


# ------------------------------------------------------------------ #
# Twelve Data — forex (the ONLY pinned FX source, see module docstring)
# ------------------------------------------------------------------ #

TWELVE_DATA_BASE = "https://api.twelvedata.com/time_series"
TWELVE_DATA_INTERVALS = {"15m": "15min", "1h": "1h", "4h": "4h"}


class TwelveDataProvider:
    source = "twelvedata_fx"

    def __init__(self, api_key=None, min_spacing_s=8.0):
        self.api_key = api_key or load_env_var("TWELVE_DATA_API_KEY")
        if not self.api_key:
            raise RuntimeError("TWELVE_DATA_API_KEY not set (.env or environment)")
        self.min_spacing_s = min_spacing_s
        self._last_call = None

    def _pace(self):
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            if elapsed < self.min_spacing_s:
                time.sleep(self.min_spacing_s - elapsed)
        self._last_call = time.monotonic()

    def _call(self, symbol, interval, outputsize=5000, end_date=None, max_retries=5,
              backoff_s=65.0):
        """
        `min_spacing_s` pacing is per-PROCESS only — it has no visibility
        into other processes hitting the same API key (e.g. a separate
        probe script run minutes earlier). Found live during Stage 5:
        the driver's very first call 429'd, because an unrelated probe
        script's last call landed seconds before this process started
        and had no shared state to pace against. Fixed here, not by
        trying to coordinate cross-process state (a lock file would
        still race), but by making 429 non-fatal: back off a full
        window (`backoff_s`, default 65s — safely past any 60s credit
        window) and retry, up to `max_retries` times, rather than
        crash the whole multi-hour backfill over one contended minute.
        """
        params = {
            "symbol": symbol, "interval": interval, "outputsize": outputsize,
            "apikey": self.api_key, "timezone": "UTC", "order": "DESC",
        }
        if end_date:
            params["end_date"] = end_date
        url = TWELVE_DATA_BASE + "?" + urllib.parse.urlencode(params)

        for attempt in range(1, max_retries + 1):
            self._pace()
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code == 429 and attempt < max_retries:
                    print(f"  [twelvedata] 429 rate-limited on {symbol} {interval} "
                         f"(attempt {attempt}/{max_retries}) — backing off {backoff_s:.0f}s")
                    time.sleep(backoff_s)
                    continue
                raise
            if isinstance(data, dict) and data.get("status") == "error":
                raise RuntimeError(f"Twelve Data error for {symbol} {interval}: {data}")
            return data

    def fetch_candles(self, symbol, timeframe, start_ms=None, end_ms=None, max_bars=None):
        """
        `symbol` e.g. 'EUR/USD'. Walks backwards in outputsize=5000
        chunks from `end_ms` (or now) until reaching `start_ms` or
        max_bars, matching the pattern proven in
        scripts/probe_twelvedata.py. volume/number_of_trades/
        taker_buy_* are always None — see module docstring.
        """
        interval = TWELVE_DATA_INTERVALS[timeframe]
        end_date = (
            datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            if end_ms else None
        )
        out = []

        while True:
            data = self._call(symbol, interval, outputsize=5000, end_date=end_date)
            values = data.get("values") or []
            if not values:
                break

            for v in values:
                dt = datetime.fromisoformat(v["datetime"].replace(" ", "T")).replace(tzinfo=timezone.utc)
                ms = int(dt.timestamp() * 1000)
                if start_ms is not None and ms < start_ms:
                    continue
                out.append({
                    "open_time_ms": ms,
                    "open": float(v["open"]), "high": float(v["high"]),
                    "low": float(v["low"]), "close": float(v["close"]),
                    "volume": None, "number_of_trades": None,
                    "taker_buy_base_volume": None, "taker_buy_quote_volume": None,
                })

            earliest_ms = out[-1]["open_time_ms"] if out else None
            if len(values) < 5000:
                break
            if start_ms is not None and earliest_ms is not None and earliest_ms <= start_ms:
                break
            if max_bars is not None and len(out) >= max_bars:
                break

            oldest_dt = datetime.fromisoformat(values[-1]["datetime"].replace(" ", "T"))
            end_date = oldest_dt.strftime("%Y-%m-%d %H:%M:%S")

        out.sort(key=lambda b: b["open_time_ms"])
        return out
