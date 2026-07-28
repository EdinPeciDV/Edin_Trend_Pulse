"""
ml/data.py
===================================================================
Data loading and synthetic generators.

Three sources:
  1. synthetic_random_walk  — a market with NO predictable structure.
     Used to prove the validation harness reports "no edge" when there
     genuinely is none. If it ever reports an edge here, the harness is
     leaking and every other result it produces is void.

  2. synthetic_with_signal  — a market with a deliberately planted,
     exploitable pattern. Used to prove the harness can still detect a
     real edge when one exists, i.e. that it is not simply pessimistic.

  3. load_candles           — real 5-minute candles from Binance /
     Twelve Data, cached to disk so repeated training runs do not
     re-hit the rate limits.

Together, 1 and 2 form a two-sided test of the harness: it must say no
when the answer is no, and yes when the answer is yes.
===================================================================
"""

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


# ------------------------------------------------------------------ #
# Candle helpers                                                      #
# ------------------------------------------------------------------ #

def _candles_from_closes(closes, start_ms=None, interval_ms=5 * 60 * 1000,
                         volume=None, seed=0):
    """
    Wrap a close series into a full OHLCV candle dict, synthesising
    plausible highs/lows around each close.
    """
    rng = np.random.default_rng(seed)
    n = len(closes)
    if start_ms is None:
        start_ms = int(time.time() * 1000) - n * interval_ms

    closes = np.asarray(closes, dtype=float)
    opens = np.concatenate(([closes[0]], closes[:-1]))
    noise = np.abs(rng.normal(0, 0.0008, n)) * closes
    highs = np.maximum(opens, closes) + noise
    lows = np.minimum(opens, closes) - noise
    if volume is None:
        volume = rng.lognormal(mean=3.0, sigma=0.5, size=n)

    return {
        "open_time": start_ms + np.arange(n) * interval_ms,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": np.asarray(volume, dtype=float),
    }


# ------------------------------------------------------------------ #
# 1. Random walk — no predictable structure                           #
# ------------------------------------------------------------------ #

def synthetic_random_walk(n=6000, start_price=60000.0, vol=0.0015,
                          drift=0.0, seed=42):
    """
    A geometric random walk. Future returns are independent of all past
    information, so the theoretically best achievable accuracy is the
    base rate, and true edge is exactly zero.

    `drift` is deliberately available: a positive drift makes the
    always_up baseline score above 50%, which is precisely the trap that
    makes naive accuracy numbers look impressive. The harness must
    report edge near zero even when raw accuracy is well above 50%.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    closes = start_price * np.exp(np.cumsum(returns))
    return _candles_from_closes(closes, seed=seed)


# ------------------------------------------------------------------ #
# 2. Planted signal — a real, detectable edge                         #
# ------------------------------------------------------------------ #

def synthetic_with_signal(n=6000, start_price=60000.0, vol=0.0015,
                          strength=0.55, seed=7):
    """
    A random walk with one exploitable rule planted in it: mean
    reversion conditional on a short-term momentum extreme.

    Construction: the next return is nudged against the sign of the
    trailing 12-bar return, with the nudge sized as `strength` x vol.
    That makes `ret_12` genuinely predictive of the forward return, so a
    competent model SHOULD find it and the harness SHOULD report an edge.

    This is not how real markets behave. It exists only to confirm the
    harness is capable of saying yes.
    """
    rng = np.random.default_rng(seed)
    logp = np.empty(n)
    logp[0] = np.log(start_price)
    window = 12

    for i in range(1, n):
        shock = rng.normal(0, vol)
        if i > window:
            trailing = logp[i - 1] - logp[i - 1 - window]
            # Push against the trailing move: mean reversion.
            nudge = -np.sign(trailing) * strength * vol
        else:
            nudge = 0.0
        logp[i] = logp[i - 1] + shock + nudge

    return _candles_from_closes(np.exp(logp), seed=seed)


# ------------------------------------------------------------------ #
# 3. Real market data                                                 #
# ------------------------------------------------------------------ #

def _http_json(url, timeout=25):
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "User-Agent": "trendpulse-ml/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_binance(symbol="BTCUSDT", interval="5m", limit=1000, pages=5,
                  host=None):
    """
    Fetch up to `pages` x `limit` candles from Binance, walking backwards
    with the endTime parameter. 5 pages x 1000 = 5000 bars ~= 17 days of
    5-minute data.

    Set BINANCE_HOST=api.binance.us if api.binance.com returns HTTP 451
    from your IP.
    """
    host = host or os.environ.get("BINANCE_HOST", "api.binance.com")
    all_rows = []
    end_time = None

    for _ in range(pages):
        url = (f"https://{host}/api/v3/klines?symbol={symbol}"
               f"&interval={interval}&limit={limit}")
        if end_time is not None:
            url += f"&endTime={end_time}"
        rows = _http_json(url)
        if not rows:
            break
        all_rows = rows + all_rows
        end_time = int(rows[0][0]) - 1
        time.sleep(0.25)  # be polite to the public endpoint

    if not all_rows:
        raise RuntimeError(f"Binance returned no data for {symbol}")

    # De-duplicate by open time and sort ascending.
    seen = {}
    for r in all_rows:
        seen[int(r[0])] = r
    rows = [seen[k] for k in sorted(seen)]

    arr = np.array(
        [[float(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])]
         for r in rows]
    )
    return {
        "open_time": arr[:, 0],
        "open": arr[:, 1],
        "high": arr[:, 2],
        "low": arr[:, 3],
        "close": arr[:, 4],
        "volume": arr[:, 5],
    }


def fetch_twelvedata(symbol="EUR/USD", interval="5min", outputsize=5000,
                     api_key=None):
    """
    Fetch forex candles from Twelve Data. The free tier caps outputsize
    at 5000 per request and 8 requests/minute.

    Remember: spot FX volume is reported as 0, so every volume-derived
    feature is inert for these pairs.
    """
    api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "TWELVE_DATA_API_KEY is not set. export it, or put it in .env "
            "and run via `set -a; . .env; set +a`."
        )

    url = ("https://api.twelvedata.com/time_series?"
           + urllib.parse.urlencode({
               "symbol": symbol, "interval": interval,
               "outputsize": outputsize, "order": "ASC", "apikey": api_key,
           }))
    js = _http_json(url)
    if js.get("status") == "error":
        raise RuntimeError(f"Twelve Data error: {js.get('message')}")
    values = js.get("values") or []
    if not values:
        raise RuntimeError(f"Twelve Data returned no values for {symbol}")

    def ts(s):
        return time.mktime(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) * 1000

    arr = np.array([[ts(v["datetime"]), float(v["open"]), float(v["high"]),
                     float(v["low"]), float(v["close"]),
                     float(v.get("volume") or 0)] for v in values])
    arr = arr[np.argsort(arr[:, 0])]
    return {
        "open_time": arr[:, 0], "open": arr[:, 1], "high": arr[:, 2],
        "low": arr[:, 3], "close": arr[:, 4], "volume": arr[:, 5],
    }


def load_candles(symbol, use_cache=True, max_age_hours=6):
    """
    Load candles for a symbol, routing to the right upstream and caching
    to ml/cache/<symbol>.npz.

    Crypto symbols contain USDT/USD without a fiat pair; forex symbols
    are FIAT/FIAT. Mirrors isForexPair() in shared/indicators.js.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")
    path = os.path.join(CACHE_DIR, f"{safe}.npz")

    if use_cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h < max_age_hours:
            z = np.load(path)
            print(f"  [cache] {symbol}: {len(z['close'])} bars, {age_h:.1f}h old")
            return {k: z[k] for k in z.files}

    fiat = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}
    parts = symbol.upper().split("/")
    is_fx = len(parts) == 2 and parts[0] in fiat and parts[1] in fiat

    print(f"  [fetch] {symbol} ({'forex' if is_fx else 'crypto'}) ...")
    if is_fx:
        candles = fetch_twelvedata(symbol)
    else:
        candles = fetch_binance(symbol.replace("/", ""))

    np.savez_compressed(path, **candles)
    print(f"  [fetch] {symbol}: {len(candles['close'])} bars cached")
    return candles


# ------------------------------------------------------------------ #
# 4. Crypto derivatives context (Phase 2.1)                           #
# ------------------------------------------------------------------ #
#
# Three free, no-auth Binance USDT-M futures endpoints, chosen per the
# spec as the highest-value-per-effort of the eight listed: funding
# rate, open interest, and top-trader positioning. All three are
# EVENT-indexed (funding prints every ~8h; the stats endpoints print on
# their own cadence and Binance only retains ~30 days of history), so
# they must be forward-filled onto the 5-minute bar grid before they can
# become per-bar features — a stale-until-the-next-print value is the
# correct alignment, not an interpolated one.

FUTURES_HOST = os.environ.get("BINANCE_FUTURES_HOST", "fapi.binance.com")


def _align_to_bars(event_times_ms, values, target_times_ms):
    """
    Forward-fill event-indexed (time, value) pairs onto `target_times_ms`.

    out[i] = the most recent value at or before target_times_ms[i], NaN
    if no event has happened yet by that bar (leading warmup — the
    feature layer's rolling z-score already collapses NaN to a neutral
    0, matching how volume_z degrades for FX).
    """
    event_times_ms = np.asarray(event_times_ms, dtype=float)
    values = np.asarray(values, dtype=float)
    target_times_ms = np.asarray(target_times_ms, dtype=float)
    if len(event_times_ms) == 0:
        return np.full(len(target_times_ms), np.nan)

    order = np.argsort(event_times_ms)
    et = event_times_ms[order]
    v = values[order]
    idx = np.searchsorted(et, target_times_ms, side="right") - 1
    out = np.full(len(target_times_ms), np.nan)
    ok = idx >= 0
    out[ok] = v[idx[ok]]
    return out


def fetch_binance_funding_rate(source_symbol, host=None, limit=1000, pages=2):
    """Funding rate history (~8h cadence). Two pages of 1000 covers ~2 years."""
    host = host or FUTURES_HOST
    all_rows = []
    end_time = None
    for _ in range(pages):
        url = (f"https://{host}/fapi/v1/fundingRate?symbol={source_symbol}"
               f"&limit={limit}")
        if end_time is not None:
            url += f"&endTime={end_time}"
        rows = _http_json(url)
        if not rows:
            break
        all_rows = rows + all_rows
        end_time = int(rows[0]["fundingTime"]) - 1
        time.sleep(0.2)
    if not all_rows:
        raise RuntimeError(f"no funding rate data for {source_symbol}")
    seen = {int(r["fundingTime"]): float(r["fundingRate"]) for r in all_rows}
    times = np.array(sorted(seen))
    return times, np.array([seen[t] for t in times])


def fetch_binance_stats_series(source_symbol, endpoint, value_key,
                               period="5m", limit=500, host=None):
    """
    Shared fetcher for the Binance "futures data" stats endpoints
    (openInterestHist, topLongShortPositionRatio, ...): all take the
    same {symbol, period, limit} shape and return a `timestamp` field
    plus one value field. Binance only retains ~30 days for these, and
    they do not reliably paginate further back, so this is a best-effort
    single-page fetch of the most recent `limit` records.
    """
    host = host or FUTURES_HOST
    url = (f"https://{host}{endpoint}?symbol={source_symbol}"
           f"&period={period}&limit={limit}")
    rows = _http_json(url)
    if not rows:
        raise RuntimeError(f"no data from {endpoint} for {source_symbol}")
    times = np.array([float(r["timestamp"]) for r in rows])
    values = np.array([float(r[value_key]) for r in rows])
    order = np.argsort(times)
    return times[order], values[order]


def fetch_binance_derivatives(source_symbol, target_open_times, host=None):
    """
    Fetch and align all three derivatives series onto `target_open_times`
    (the candle grid being trained/inferred on). Best-effort per series —
    a single failed upstream call degrades that series to None (which
    build_features() treats as "no context for this feature", not a hard
    failure) rather than aborting the whole fetch.

    Returns {'funding_rate': array|None, 'open_interest': array|None,
             'top_trader_ratio': array|None}.
    """
    out = {}

    try:
        t, v = fetch_binance_funding_rate(source_symbol, host=host)
        out["funding_rate"] = _align_to_bars(t, v, target_open_times)
    except Exception as e:
        print(f"  [derivatives] funding rate failed for {source_symbol}: {e}")
        out["funding_rate"] = None

    try:
        t, v = fetch_binance_stats_series(
            source_symbol, "/futures/data/openInterestHist", "sumOpenInterest", host=host
        )
        out["open_interest"] = _align_to_bars(t, v, target_open_times)
    except Exception as e:
        print(f"  [derivatives] open interest failed for {source_symbol}: {e}")
        out["open_interest"] = None

    try:
        t, v = fetch_binance_stats_series(
            source_symbol, "/futures/data/topLongShortPositionRatio",
            "longShortRatio", host=host
        )
        out["top_trader_ratio"] = _align_to_bars(t, v, target_open_times)
    except Exception as e:
        print(f"  [derivatives] top-trader ratio failed for {source_symbol}: {e}")
        out["top_trader_ratio"] = None

    return out


def load_derivatives_context(symbol, target_open_times, use_cache=True, max_age_hours=6):
    """
    Cached wrapper around fetch_binance_derivatives(), analogous to
    load_candles(). Cache key includes the candle grid's length + first/
    last timestamp so a context built for a different training window
    is never silently reused for one it doesn't align with.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = symbol.replace("/", "_")
    n = len(target_open_times)
    grid_tag = f"{n}_{int(target_open_times[0])}_{int(target_open_times[-1])}" if n else "empty"
    path = os.path.join(CACHE_DIR, f"{safe}_deriv_{grid_tag}.npz")

    if use_cache and os.path.exists(path):
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h < max_age_hours:
            z = np.load(path)
            print(f"  [cache] {symbol} derivatives: {age_h:.1f}h old")
            return {k: z[k] for k in z.files}

    source_symbol = symbol.replace("/", "")
    print(f"  [fetch] {symbol} derivatives (funding/OI/top-trader) ...")
    ctx = fetch_binance_derivatives(source_symbol, target_open_times)
    # Only cache fully-successful fetches — a partial one (some series
    # None because an endpoint hiccuped) should retry next run rather
    # than freeze a gap into the cache.
    if all(v is not None for v in ctx.values()):
        np.savez_compressed(path, **ctx)
    return ctx


# ------------------------------------------------------------------ #
# 5. Cross-sectional alignment (Phase 2.3)                            #
# ------------------------------------------------------------------ #

def align_candles_by_time(candles_by_symbol):
    """
    Inner-join multiple candle dicts on open_time.

    Crypto instruments share the same Binance 5-minute grid, so this is
    normally a no-op past leading/trailing gaps — but pairs can miss a
    tick independently, and the intersection guarantees every symbol's
    row i refers to the exact same wall-clock bar, which cross-sectional
    features (rel_strength_basket, corr_to_btc) require.

    Returns (open_times, {symbol: candles_dict_restricted_to_common_times}).
    """
    common = None
    for c in candles_by_symbol.values():
        t = set(np.asarray(c["open_time"]).tolist())
        common = t if common is None else (common & t)
    if not common:
        raise ValueError("no overlapping timestamps across the pooled symbols")

    common_sorted = np.array(sorted(common))
    out = {}
    for symbol, c in candles_by_symbol.items():
        t = np.asarray(c["open_time"])
        idx = np.searchsorted(t, common_sorted)
        out[symbol] = {k: np.asarray(v)[idx] for k, v in c.items()}
    return common_sorted, out


def build_cross_sectional_context(aligned_candles_by_symbol, self_symbol):
    """
    For `self_symbol`, build the basket_ret / btc_ret context arrays
    build_features() expects, from a set of symbols already aligned by
    align_candles_by_time().

    basket_ret[i] = mean 1-bar log return of every OTHER pooled symbol
                    at bar i (equal-weighted).
    btc_ret[i]    = BTC's own 1-bar log return at bar i, omitted (None)
                    when self_symbol IS the BTC pair — corr(x, x) = 1 is
                    not informative and would just be a constant feature.
    """
    def log_ret1(closes):
        c = np.asarray(closes, dtype=float)
        r = np.full(len(c), np.nan)
        if len(c) > 1:
            r[1:] = np.log(c[1:]) - np.log(c[:-1])
        return np.nan_to_num(r, nan=0.0)

    rets = {sym: log_ret1(c["close"]) for sym, c in aligned_candles_by_symbol.items()}

    others = [sym for sym in rets if sym != self_symbol]
    basket_ret = (
        np.mean([rets[sym] for sym in others], axis=0) if others else None
    )

    btc_symbol = next((sym for sym in rets if sym.upper().startswith("BTC/")), None)
    btc_ret = rets[btc_symbol] if btc_symbol and btc_symbol != self_symbol else None

    return {"basket_ret": basket_ret, "btc_ret": btc_ret}
