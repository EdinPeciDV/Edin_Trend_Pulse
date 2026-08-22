"""
ml_macd/verification.py
===================================================================
Canonical stage-report generator, used by every backfill stage from
Stage 4 onward instead of ad-hoc inline scripts. Queries Supabase
fresh each time (never trusts ingestion-time print output) and
reports FOUR categories, not three: rows written, future-timestamp
count, gaps (missing bars), and — added after Stage 3 turned up a
real misaligned bar — grid-misalignment. Never halts on anything;
this module only reports.
===================================================================
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))

from gap_handling import bars_missing_before, misaligned_bars, bar_durations  # noqa: E402
from providers import load_env_var  # noqa: E402

SUPABASE_URL = load_env_var("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = load_env_var("SUPABASE_SERVICE_ROLE_KEY")

TIMEFRAME_SECONDS = {"15m": 900, "1h": 3600, "4h": 14400}


def fetch_all_open_times(symbol, timeframe, asset_class=None):
    """Every open_time for (symbol, timeframe), paginated past PostgREST's
    server-side row cap, ordered ascending. Optionally filtered by
    asset_class too (defensive, matches query_max_open_time's same guard)."""
    out, offset = [], 0
    while True:
        params = {
            "symbol": f"eq.{symbol}", "timeframe": f"eq.{timeframe}",
            "select": "open_time", "order": "open_time.asc", "offset": str(offset),
        }
        if asset_class:
            params["asset_class"] = f"eq.{asset_class}"
        url = f"{SUPABASE_URL}/rest/v1/macd_candles?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        })
        with urllib.request.urlopen(req, timeout=60) as resp:
            rows = json.loads(resp.read())
        if not rows:
            break
        out.extend(r["open_time"] for r in rows)
        offset += len(rows)
    return out


def stage_report(symbol, timeframe, asset_class=None, print_report=True):
    """
    Query Supabase fresh and return the four-category report:
      n, min_open_time, max_open_time, future_count,
      gap_segments (list of (start_iso, end_iso, n_missing)),
      total_missing_bars,
      misaligned (list of (open_time_iso, duration_seconds)),
    Never raises, never halts on findings — pure reporting.
    """
    timeframe_s = TIMEFRAME_SECONDS[timeframe]
    times_iso = fetch_all_open_times(symbol, timeframe, asset_class)
    times = [datetime.fromisoformat(t.replace("Z", "+00:00")) for t in times_iso]
    n = len(times)

    result = {
        "symbol": symbol, "timeframe": timeframe, "n": n,
        "min_open_time": times[0].isoformat() if times else None,
        "max_open_time": times[-1].isoformat() if times else None,
        "future_count": 0, "gap_segments": [], "total_missing_bars": 0,
        "misaligned": [],
    }
    if n == 0:
        if print_report:
            print(f"[report] {symbol} {timeframe}: no rows")
        return result

    now = datetime.now(tz=timezone.utc)
    result["future_count"] = sum(1 for t in times if t > now)

    open_time_s = [t.timestamp() for t in times]
    missing = bars_missing_before(open_time_s, timeframe_s)
    for i in range(1, n):
        if missing[i] > 0:
            result["gap_segments"].append((times[i - 1].isoformat(), times[i].isoformat(), int(missing[i])))
    result["total_missing_bars"] = int(sum(missing))

    misaligned_mask = misaligned_bars(open_time_s, timeframe_s)
    durations = bar_durations(open_time_s)
    for i in range(n):
        if misaligned_mask[i]:
            dur = durations[i]
            result["misaligned"].append((times[i].isoformat(), None if str(dur) == "nan" else float(dur)))

    if print_report:
        print(f"[report] {symbol} {timeframe}: n={n} min={result['min_open_time']} "
             f"max={result['max_open_time']} future={result['future_count']} "
             f"gap_segments={len(result['gap_segments'])} missing_bars={result['total_missing_bars']} "
             f"misaligned={len(result['misaligned'])}")
        for ts, dur in result["misaligned"]:
            dur_str = f"{dur:.0f}s" if dur is not None else "n/a (last row)"
            print(f"  MISALIGNED: {ts}  bar_duration={dur_str} (expected {timeframe_s}s)")

    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ml_macd stage report")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", required=True, choices=list(TIMEFRAME_SECONDS))
    ap.add_argument("--asset-class")
    args = ap.parse_args()
    stage_report(args.symbol, args.timeframe, args.asset_class)
