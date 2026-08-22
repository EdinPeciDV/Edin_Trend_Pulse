"""
scripts/probe_twelvedata.py — THROWAWAY, not part of ml_macd/ or any
production pipeline.

Answers four questions about what Twelve Data's FX data ACTUALLY is
(not its marketing copy) before ml_macd/data.py commits to it as the
single pinned FX provider for both backfill and live:

  1. Is a volume field returned for FX at all, and does it vary
     plausibly by session?
  2. How far back does real history go, for 15m and 1h, on the free
     tier?
  3. Is volume present at the OLDEST available data, or does the
     provider backfill price further than volume?
  4. Is coverage confirmed for EUR/USD, GBP/USD, USD/JPY, and one
     cross (EUR/GBP)?

Delete this file once ml_macd/data.py's FX provider interface is
written and these answers are recorded in ml_macd/README.md.

Usage: python3 scripts/probe_twelvedata.py
"""

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ENV_PATH = os.path.join(ROOT, ".env")
BASE = "https://api.twelvedata.com/time_series"

# Free tier: 8 credits/min. Space calls comfortably past that floor.
MIN_SPACING_S = 8.0
MAX_TOTAL_CALLS = 60  # hard ceiling so this probe can't accidentally
                       # burn a meaningful chunk of the 800/day cap

_calls_made = 0
_credits_log = []


def load_env_var(name, path=ENV_PATH):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


API_KEY = load_env_var("TWELVE_DATA_API_KEY")
if not API_KEY:
    print("TWELVE_DATA_API_KEY not found in .env — aborting probe.")
    sys.exit(1)


def call(symbol, interval, outputsize=5000, end_date=None, note=""):
    global _calls_made
    if _calls_made >= MAX_TOTAL_CALLS:
        raise RuntimeError(f"Hit MAX_TOTAL_CALLS={MAX_TOTAL_CALLS} safety ceiling.")
    params = {
        "symbol": symbol, "interval": interval, "outputsize": outputsize,
        "apikey": API_KEY, "timezone": "UTC", "order": "DESC",
    }
    if end_date:
        params["end_date"] = end_date
    url = BASE + "?" + urllib.parse.urlencode(params)

    if _calls_made > 0:
        time.sleep(MIN_SPACING_S)
    _calls_made += 1
    _credits_log.append(f"[{_calls_made}] {symbol} {interval} outputsize={outputsize} "
                        f"end_date={end_date} {note}")

    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return {"status": "error", "http_error": e.code, "message": body}
    return data


def bars_of(data):
    """Normalise a successful time_series response to a list of bar dicts,
    or [] on error/empty."""
    if not isinstance(data, dict) or data.get("status") == "error":
        return []
    return data.get("values") or []


def volume_report(bars, label):
    if not bars:
        print(f"    {label}: NO BARS")
        return
    vols = []
    for b in bars:
        v = b.get("volume")
        try:
            vols.append(float(v) if v is not None else None
                       )
        except (TypeError, ValueError):
            vols.append(None)

    present = [v for v in vols if v is not None]
    n = len(bars)
    n_present = len(present)
    n_nonzero = sum(1 for v in present if v not in (0, 0.0))
    distinct = len(set(present))

    print(f"    {label}: {n} bars, volume field present in {n_present}/{n} "
          f"({'MISSING FIELD ENTIRELY' if n_present == 0 else 'ok'})")
    if n_present > 0:
        print(f"      non-zero: {n_nonzero}/{n_present}   distinct values: {distinct}"
              f"   min={min(present):.4g} max={max(present):.4g} mean={sum(present)/n_present:.4g}")
        if distinct <= 2:
            print("      *** DEGENERATE: volume is effectively constant (<=2 distinct values) ***")


def session_variation_report(bars, label):
    """Bucket by UTC hour, report mean volume per hour to see if it
    plausibly tracks London/NY/Asia session activity."""
    buckets = defaultdict(list)
    for b in bars:
        try:
            v = float(b.get("volume")) if b.get("volume") is not None else None
        except (TypeError, ValueError):
            v = None
        if v is None:
            continue
        try:
            dt = datetime.fromisoformat(b["datetime"].replace(" ", "T")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        buckets[dt.hour].append(v)

    if not buckets:
        print(f"    {label}: no volume data to bucket by session")
        return
    print(f"    {label}: mean volume by UTC hour (London ~07-16, NY ~12-21, Asia ~23-08)")
    row = []
    for h in range(24):
        vs = buckets.get(h)
        row.append(f"{h:02d}:{(sum(vs)/len(vs)):.3g}" if vs else f"{h:02d}:--")
    print("      " + "  ".join(row))


def earliest_datetime(bars):
    if not bars:
        return None
    # order=DESC, so the last element is the earliest.
    return bars[-1].get("datetime")


def walk_back_depth(symbol, interval, target_days, max_calls, label):
    """Walk backwards in outputsize=5000 chunks until either target_days
    is comfortably cleared, the API stops returning full pages (start of
    history), or max_calls is spent."""
    print(f"\n  --- depth walkback: {symbol} {interval} (target >= {target_days} days) ---")
    end_date = None
    earliest = None
    calls_used = 0
    all_bars_seen = 0
    volume_missing_anywhere = False

    while calls_used < max_calls:
        data = call(symbol, interval, outputsize=5000, end_date=end_date,
                   note=f"depth-walk {label}")
        if not isinstance(data, dict) or data.get("status") == "error":
            print(f"    call {calls_used+1}: ERROR — {json.dumps(data)[:300]}")
            break
        bars = bars_of(data)
        calls_used += 1
        if not bars:
            print(f"    call {calls_used}: 0 bars returned — history exhausted here.")
            break

        all_bars_seen += len(bars)
        this_earliest = earliest_datetime(bars)
        vols = [b.get("volume") for b in bars]
        if any(v in (None, "0", 0) for v in vols):
            volume_missing_anywhere = True

        print(f"    call {calls_used}: {len(bars)} bars, earliest={this_earliest}, "
              f"latest={bars[0].get('datetime')}, "
              f"volume_sample={vols[:3]}")

        if len(bars) < 5000:
            print(f"    got fewer than requested (5000) — likely hit start of available history.")
            earliest = this_earliest
            break

        earliest = this_earliest
        end_date = this_earliest  # next page ends where this one left off

        if earliest:
            try:
                earliest_dt = datetime.fromisoformat(earliest.replace(" ", "T"))
                days_covered = (datetime.utcnow() - earliest_dt).days
                if days_covered >= target_days:
                    print(f"    reached {days_covered} days of history — target cleared, stopping walkback early.")
                    break
            except Exception:
                pass

    return {
        "symbol": symbol, "interval": interval, "earliest": earliest,
        "calls_used": calls_used, "total_bars_seen": all_bars_seen,
        "volume_missing_anywhere_in_walk": volume_missing_anywhere,
    }


def main():
    print("=" * 78)
    print("TWELVE DATA FX PROBE — real API, not documentation")
    print("=" * 78)

    print("\n[1] VOLUME AVAILABILITY (recent data, EUR/USD)")
    recent_15m = bars_of(call("EUR/USD", "15min", outputsize=500, note="recent volume check"))
    volume_report(recent_15m, "EUR/USD 15min (last ~5 days)")
    session_variation_report(recent_15m, "EUR/USD 15min")

    recent_1h = bars_of(call("EUR/USD", "1h", outputsize=500, note="recent volume check"))
    volume_report(recent_1h, "EUR/USD 1h (last ~20 days)")

    print("\n[2] HISTORICAL DEPTH (walk backwards)")
    depth_15m = walk_back_depth("EUR/USD", "15min", target_days=730, max_calls=16, label="15m")
    depth_1h = walk_back_depth("EUR/USD", "1h", target_days=1460, max_calls=10, label="1h")

    print("\n[3] VOLUME AT DEPTH (oldest ~500 bars we could reach)")
    if depth_15m["earliest"]:
        oldest_15m = bars_of(call("EUR/USD", "15min", outputsize=500,
                                  end_date=depth_15m["earliest"], note="oldest-chunk volume check"))
        volume_report(oldest_15m, f"EUR/USD 15min near earliest reached ({depth_15m['earliest']})")
    if depth_1h["earliest"]:
        oldest_1h = bars_of(call("EUR/USD", "1h", outputsize=500,
                                 end_date=depth_1h["earliest"], note="oldest-chunk volume check"))
        volume_report(oldest_1h, f"EUR/USD 1h near earliest reached ({depth_1h['earliest']})")

    print("\n[4] PAIR COVERAGE (1h, small sample each)")
    for sym in ["GBP/USD", "USD/JPY", "EUR/GBP"]:
        data = call(sym, "1h", outputsize=50, note="pair coverage check")
        bars = bars_of(data)
        if not bars:
            print(f"    {sym}: NOT AVAILABLE — {json.dumps(data)[:300]}")
        else:
            volume_report(bars, sym)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Total API calls used by this probe: {_calls_made} (of {MAX_TOTAL_CALLS} ceiling, "
          f"800/day free-tier budget)")
    print(f"15min depth reached: earliest={depth_15m['earliest']} over {depth_15m['calls_used']} calls, "
          f"{depth_15m['total_bars_seen']} bars seen, "
          f"volume gaps seen in walk: {depth_15m['volume_missing_anywhere_in_walk']}")
    print(f"1h depth reached:    earliest={depth_1h['earliest']} over {depth_1h['calls_used']} calls, "
          f"{depth_1h['total_bars_seen']} bars seen, "
          f"volume gaps seen in walk: {depth_1h['volume_missing_anywhere_in_walk']}")
    print("\nCall log:")
    for line in _credits_log:
        print(f"  {line}")


if __name__ == "__main__":
    main()
