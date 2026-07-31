#!/usr/bin/env python3
"""
sysrfx_predictor.py
===================================================================
Zero-cost forex + crypto direction predictor, with Supabase logging.

    python3 sysrfx_predictor.py --mode backtest --pair BTC/USDT
    python3 sysrfx_predictor.py --mode backtest --all --tune
    python3 sysrfx_predictor.py --mode live --pair BTC/USDT

Install: `pip install -r requirements.txt`. Copy `.env.example` to `.env`
and fill in whatever free API keys you have — every optional data source
(on-chain, FRED, Alpha Vantage, Supabase) degrades gracefully to
"feature skipped" / "local SQLite fallback" if its keys are absent,
rather than crashing.

READ THIS BEFORE TRUSTING THE ACCURACY NUMBER THIS SCRIPT PRINTS
------------------------------------------------------------------
Walk-forward accuracy is reported as MEASURED, not targeted. There is
deliberately no "if accuracy < target, retune and retry" loop here.
Searching over hyperparameters until a backtest number clears a fixed
target isn't validation — it's a way to manufacture a number that looks
good, and the more configurations you try, the more likely you are to
find one that clears any given bar on THIS data by chance alone, with
no guarantee it generalises even one day forward. If a run reports 54%,
that IS the answer. Report it; don't keep searching until you like it
better.

Modules (see PREDICTION notes inline; module numbers match the design
this was specified against):
  0 — OHLCV + ~90 technical indicators (via the `ta` library)
  1 — feature selection (mutual_info_classif, top 20)
  2 — target smoothing (3-bar-ahead direction, 0.5% dead band)
  3 — on-chain features (Glassnode, crypto only, needs a free key)
  4 — order-flow / volume features + macro context
  5 — LightGBM + Hyperopt tuning (explicit, via --tune — not automatic)
  6 — online retraining (background thread, every 50 new bars, 500-bar window)
  7 — regime switching (ADX > 25 trending vs <= 25 ranging, separate models)
===================================================================
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    from dotenv import load_dotenv
    # Scoped to THIS script's own directory, not python-dotenv's default
    # upward search — this script is meant to be a self-contained,
    # standalone project (its own requirements.txt, schema.sql, cache/),
    # and an upward search would happily find and load an unrelated
    # parent project's .env if this ever sits inside another repo (e.g.
    # a SUPABASE_URL for a completely different Supabase project/schema).
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass

# ------------------------------------------------------------------ #
# Optional dependencies — each isolated so a missing package disables
# exactly the one feature that needs it, not the whole script.
# ------------------------------------------------------------------ #
try:
    import ccxt
except ImportError:
    ccxt = None
try:
    import yfinance as yf
except ImportError:
    yf = None
try:
    import ta as ta_lib
except ImportError:
    ta_lib = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import hyperopt as _hyperopt_mod  # noqa: F401 — presence check only, used lazily
    HYPEROPT_AVAILABLE = True
except ImportError:
    HYPEROPT_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sysrfx")
logging.getLogger("hyperopt").setLevel(logging.WARNING)

# ------------------------------------------------------------------ #
# Constants                                                            #
# ------------------------------------------------------------------ #

DEFAULT_PAIRS = ["BTC/USDT", "ETH/USDT", "EUR/USD", "GBP/USD"]
FIAT_CODES = {"USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"}

HORIZON_BARS = 3            # Module 2 — predict 3 bars ahead
DEAD_BAND_PCT = {"crypto": 0.5, "forex": 0.08}
# Module 2 — ignore forward moves smaller than this (%). DELIBERATE
# DEVIATION FROM A FLAT THRESHOLD: the original brief specified one flat
# 0.5% dead band for every pair. Measured on real data, 3-bar-ahead
# moves on EUR/USD and GBP/USD have a MEDIAN size of ~0.06%, roughly 5-6x
# smaller than BTC/USDT's ~0.31% — a flat 0.5% band leaves crypto with
# ~30% of bars usable but leaves forex with under 1% (effectively zero
# usable rows, i.e. the script can't train on FX at all). 0.08% for
# forex gets back to a comparable ~35-40% retention. Same asset-class-
# scaling problem, same fix shape, as ../shared/indicators.js's
# resolveStopLossPct() in the sibling TrendPulse project.
CONFIDENCE_THRESHOLD = 0.6    # only emit a live trade signal above this
ADX_REGIME_THRESHOLD = 25      # Module 7
RETRAIN_EVERY_N_BARS = 50       # Module 6
RETRAIN_WINDOW_BARS = 500        # Module 6
HYPEROPT_MAX_EVALS = 50
TOP_K_FEATURES = 20
WALKFORWARD_FOLDS = 10
EMBARGO_BARS = 2              # purge/embargo gap around each fold boundary

FEATURE_EXCLUDE = {
    "timestamp", "open", "high", "low", "close", "volume",
    # `ta`'s Parabolic SAR emits psar_up/psar_down as a COMPLEMENTARY
    # pair by design — exactly one of the two is populated on any given
    # row (up during an uptrend, down during a downtrend), the other is
    # NaN, always, not as a warmup artifact. Requiring both non-null
    # would drop every row in the dataset. The companion
    # *_indicator columns (0/1 trend-flip flags, always populated) stay
    # in the candidate set — they carry the same information without
    # the structural-NaN problem. Both raw psar columns are also plain
    # price levels anyway (not stationary), so excluding them costs
    # nothing a differenced/ratio feature doesn't already cover.
    "trend_psar_up", "trend_psar_down",
}
# Raw price/volume levels are excluded from the candidate feature set on
# purpose — they aren't stationary (a model trained on them just
# memorises the price range it saw), and OHLCV is already summarised by
# the ~90 stationary indicators `ta` derives from it.

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)


def is_crypto(pair):
    """A pair is forex only when BOTH legs are fiat; everything else is
    treated as crypto (BTC/USDT, ETH/USDT, ...). Mirrors the same check
    used throughout ../shared/indicators.js in the sibling TrendPulse
    project — 'quote == USD' alone is NOT enough, since USDT/BUSD pairs
    also end in a dollar-shaped code."""
    parts = pair.upper().split("/")
    if len(parts) != 2:
        return True
    return not (parts[0] in FIAT_CODES and parts[1] in FIAT_CODES)


def to_yf_symbol(pair):
    base, quote = pair.upper().split("/")
    return f"{base}{quote}=X"


# ==================================================================== #
# Storage layer — Supabase (primary) with local SQLite fallback         #
# ==================================================================== #

class Storage:
    def ensure_schema(self):
        raise NotImplementedError

    def save_bars(self, pair, timeframe, df):
        raise NotImplementedError

    def load_recent_bars(self, pair, timeframe, n):
        raise NotImplementedError

    def save_prediction(self, pair, timestamp, direction, confidence, regime, horizon_bars):
        raise NotImplementedError

    def save_model_metadata(self, pair, regime, n_train_rows, wf_acc, wf_acc_std,
                            base_rate, conf_filtered_acc, hyperparams, feature_importances):
        raise NotImplementedError


class SQLiteStore(Storage):
    """Local fallback — always available, zero configuration."""

    def __init__(self, path=None):
        self.path = str(path or (CACHE_DIR / "sysrfx_local.db"))
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.lock = threading.Lock()
        self.ensure_schema()

    def ensure_schema(self):
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bars (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL, timeframe TEXT NOT NULL, timestamp TEXT NOT NULL,
                    open REAL, high REAL, low REAL, close REAL, volume REAL, features TEXT,
                    UNIQUE(pair, timeframe, timestamp)
                );
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pair TEXT NOT NULL, timestamp TEXT NOT NULL, direction INTEGER,
                    confidence REAL, regime TEXT, horizon_bars INTEGER,
                    actual_result INTEGER, resolved_at TEXT
                );
                CREATE TABLE IF NOT EXISTS model_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trained_at TEXT, pair TEXT, regime TEXT, n_train_rows INTEGER,
                    walk_forward_accuracy REAL, walk_forward_accuracy_std REAL, base_rate REAL,
                    confidence_filtered_accuracy REAL, hyperparams TEXT, feature_importances TEXT
                );
                """
            )

    def save_bars(self, pair, timeframe, df):
        rows = [
            (pair, timeframe, str(r.timestamp), float(r.open), float(r.high),
             float(r.low), float(r.close), float(r.volume), None)
            for r in df.itertuples(index=False)
        ]
        try:
            with self.lock, self.conn:
                self.conn.executemany(
                    "INSERT OR IGNORE INTO bars "
                    "(pair, timeframe, timestamp, open, high, low, close, volume, features) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    rows,
                )
        except Exception as e:
            log.warning(f"[storage/sqlite] save_bars failed: {e}")

    def load_recent_bars(self, pair, timeframe, n):
        try:
            with self.lock:
                cur = self.conn.execute(
                    "SELECT timestamp, open, high, low, close, volume FROM bars "
                    "WHERE pair=? AND timeframe=? ORDER BY timestamp DESC LIMIT ?",
                    (pair, timeframe, n),
                )
                rows = cur.fetchall()
            if not rows:
                return None
            df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            log.warning(f"[storage/sqlite] load_recent_bars failed: {e}")
            return None

    def save_prediction(self, pair, timestamp, direction, confidence, regime, horizon_bars):
        try:
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT INTO predictions (pair, timestamp, direction, confidence, regime, horizon_bars) "
                    "VALUES (?,?,?,?,?,?)",
                    (pair, str(timestamp), int(direction), float(confidence), regime, int(horizon_bars)),
                )
        except Exception as e:
            log.warning(f"[storage/sqlite] save_prediction failed: {e}")

    def save_model_metadata(self, pair, regime, n_train_rows, wf_acc, wf_acc_std,
                            base_rate, conf_filtered_acc, hyperparams, feature_importances):
        try:
            with self.lock, self.conn:
                self.conn.execute(
                    "INSERT INTO model_metadata (trained_at, pair, regime, n_train_rows, "
                    "walk_forward_accuracy, walk_forward_accuracy_std, base_rate, "
                    "confidence_filtered_accuracy, hyperparams, feature_importances) "
                    "VALUES (datetime('now'),?,?,?,?,?,?,?,?,?)",
                    (pair, regime, int(n_train_rows), float(wf_acc), float(wf_acc_std),
                     float(base_rate),
                     float(conf_filtered_acc) if conf_filtered_acc is not None else None,
                     json.dumps(hyperparams, default=str), json.dumps(feature_importances)),
                )
        except Exception as e:
            log.warning(f"[storage/sqlite] save_model_metadata failed: {e}")


class SupabaseStore(Storage):
    """Primary backend. Falls back to SQLite at the call site (get_storage())
    if the client can't even be constructed — every method here still
    guards its own calls too, since a network hiccup mid-run shouldn't
    crash a live loop that's otherwise working fine."""

    def __init__(self, url, key, db_url=None):
        from supabase import create_client
        self.client = create_client(url, key)
        self.db_url = db_url
        self.ensure_schema()

    def ensure_schema(self):
        if not self.db_url:
            log.info(
                "[storage/supabase] SUPABASE_DB_URL not set — assuming schema.sql "
                "has already been run (SQL Editor or `psql`). See README."
            )
            return
        try:
            import psycopg2
            sql = (Path(__file__).parent / "schema.sql").read_text()
            with psycopg2.connect(self.db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute(sql)
                conn.commit()
            log.info("[storage/supabase] schema ensured via direct Postgres connection")
        except Exception as e:
            log.warning(
                f"[storage/supabase] could not auto-create schema ({e}) — "
                f"run schema.sql manually in the Supabase SQL editor"
            )

    def save_bars(self, pair, timeframe, df):
        records = [
            {
                "pair": pair, "timeframe": timeframe, "timestamp": r.timestamp.isoformat(),
                "open": float(r.open), "high": float(r.high), "low": float(r.low),
                "close": float(r.close), "volume": float(r.volume),
            }
            for r in df.itertuples(index=False)
        ]
        try:
            for i in range(0, len(records), 500):  # PostgREST payload limit headroom
                self.client.table("bars").upsert(
                    records[i:i + 500], on_conflict="pair,timeframe,timestamp"
                ).execute()
        except Exception as e:
            log.warning(f"[storage/supabase] save_bars failed: {e}")

    def load_recent_bars(self, pair, timeframe, n):
        try:
            res = (
                self.client.table("bars").select("timestamp,open,high,low,close,volume")
                .eq("pair", pair).eq("timeframe", timeframe)
                .order("timestamp", desc=True).limit(n).execute()
            )
            if not res.data:
                return None
            df = pd.DataFrame(res.data)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df.sort_values("timestamp").reset_index(drop=True)
        except Exception as e:
            log.warning(f"[storage/supabase] load_recent_bars failed: {e}")
            return None

    def save_prediction(self, pair, timestamp, direction, confidence, regime, horizon_bars):
        try:
            self.client.table("predictions").insert({
                "pair": pair,
                "timestamp": pd.Timestamp(timestamp).isoformat(),
                "direction": int(direction), "confidence": float(confidence),
                "regime": regime, "horizon_bars": int(horizon_bars),
            }).execute()
        except Exception as e:
            log.warning(f"[storage/supabase] save_prediction failed: {e}")

    def save_model_metadata(self, pair, regime, n_train_rows, wf_acc, wf_acc_std,
                            base_rate, conf_filtered_acc, hyperparams, feature_importances):
        try:
            self.client.table("model_metadata").insert({
                "pair": pair, "regime": regime, "n_train_rows": int(n_train_rows),
                "walk_forward_accuracy": float(wf_acc), "walk_forward_accuracy_std": float(wf_acc_std),
                "base_rate": float(base_rate),
                "confidence_filtered_accuracy": float(conf_filtered_acc) if conf_filtered_acc is not None else None,
                "hyperparams": json.loads(json.dumps(hyperparams, default=str)),
                "feature_importances": feature_importances,
            }).execute()
        except Exception as e:
            log.warning(f"[storage/supabase] save_model_metadata failed: {e}")


def get_storage(use_supabase=True):
    if use_supabase:
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            try:
                store = SupabaseStore(url, key, db_url=os.environ.get("SUPABASE_DB_URL"))
                log.info("[storage] using Supabase")
                return store
            except Exception as e:
                log.warning(f"[storage] Supabase init failed ({e}) — falling back to local SQLite")
        else:
            log.warning("[storage] SUPABASE_URL / SUPABASE_KEY not set — falling back to local SQLite")
    store = SQLiteStore()
    log.info(f"[storage] using local SQLite fallback ({store.path})")
    return store


# ==================================================================== #
# Data fetch — OHLCV, on-chain, macro, order-flow                       #
# ==================================================================== #

def fetch_ohlcv_crypto(pair, timeframe="1h", limit=1000):
    if ccxt is None:
        raise RuntimeError("ccxt is not installed — pip install ccxt")
    exchange = ccxt.binance({"enableRateLimit": True})
    raw = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=min(limit, 1000))
    if not raw:
        raise RuntimeError(f"Binance returned no OHLCV for {pair}")
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df


def fetch_ohlcv_forex(pair, timeframe="1h", period="60d"):
    if yf is None:
        raise RuntimeError("yfinance is not installed — pip install yfinance")
    interval = "60m" if timeframe in ("1h", "60m") else timeframe
    symbol = to_yf_symbol(pair)
    raw = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
    if raw is None or raw.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    ts_col = "Datetime" if "Datetime" in raw.columns else "Date"
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(raw[ts_col], utc=True),
        "open": raw["Open"].astype(float),
        "high": raw["High"].astype(float),
        "low": raw["Low"].astype(float),
        "close": raw["Close"].astype(float),
        "volume": raw["Volume"].astype(float) if "Volume" in raw.columns else 0.0,
    })
    return df.dropna(subset=["close"]).reset_index(drop=True)


def fetch_ohlcv(pair, timeframe="1h", limit=1000):
    """Unified entry point — routes to Binance (crypto) or Yahoo Finance (forex)."""
    if is_crypto(pair):
        return fetch_ohlcv_crypto(pair, timeframe=timeframe, limit=limit)
    period = "60d" if timeframe in ("1h", "60m") else "1y"
    return fetch_ohlcv_forex(pair, timeframe=timeframe, period=period)


def fetch_glassnode_metric(asset, metric, api_key):
    """Glassnode free tier: one (timestamp, value) series per call."""
    if not api_key:
        return None
    try:
        r = requests.get(
            f"https://api.glassnode.com/v1/metrics/{metric}",
            params={"a": asset, "api_key": api_key, "i": "24h"}, timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        if not data:
            return None
        col = metric.split("/")[-1]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["t"], unit="s", utc=True)
        df = df.rename(columns={"v": col})
        return df[["timestamp", col]]
    except Exception as e:
        log.warning(f"[on-chain] Glassnode fetch failed for {metric}: {e}")
        return None


def fetch_onchain_features(pair):
    """Module 3 — exchange netflow (BTC/ETH) + miner reserve (BTC only)."""
    if not is_crypto(pair):
        return None
    base = pair.split("/")[0].lower()
    if base not in ("btc", "eth"):
        return None
    api_key = os.environ.get("GLASSNODE_API_KEY")
    if not api_key:
        log.info("[on-chain] GLASSNODE_API_KEY not set — Module 3 (on-chain) skipped")
        return None
    netflow = fetch_glassnode_metric(base, "transactions/transfers_volume_exchanges_net", api_key)
    miner_reserve = fetch_glassnode_metric("btc", "mining/reserve", api_key) if base == "btc" else None
    frames = [f for f in (netflow, miner_reserve) if f is not None]
    if not frames:
        return None
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on="timestamp", how="outer")
    return out.sort_values("timestamp").reset_index(drop=True)


def fetch_btc_dominance():
    """Free, no key — CoinGecko global market data."""
    try:
        r = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
        r.raise_for_status()
        return float(r.json()["data"]["market_cap_percentage"]["btc"])
    except Exception as e:
        log.warning(f"[macro] CoinGecko BTC dominance fetch failed: {e}")
        return None


def fetch_fred_series(series_id, api_key, limit=200):
    if not api_key:
        return None
    try:
        from fredapi import Fred
        s = Fred(api_key=api_key).get_series(series_id).tail(limit)
        return float(s.iloc[-1]) if len(s) else None
    except Exception as e:
        log.warning(f"[macro] FRED fetch failed for {series_id}: {e}")
        return None


def fetch_alpha_vantage_cpi(api_key):
    """
    US CPI (monthly), Alpha Vantage's free 'Economic Indicators' REST
    category. NOTE: this endpoint isn't wrapped by the `alpha_vantage`
    PyPI package (which covers time series / FX / crypto / technical
    indicators, not economic indicators) — called directly via
    `requests` instead, so that package isn't a real dependency of this
    script despite being listed in the original brief's source table.
    """
    if not api_key:
        return None
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={"function": "CPI", "interval": "monthly", "apikey": api_key}, timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data")
        return float(data[0]["value"]) if data else None
    except Exception as e:
        log.warning(f"[macro] Alpha Vantage CPI fetch failed: {e}")
        return None


def fetch_macro_features(pair):
    """
    Module 4 (macro half). BTC dominance for crypto; US/ECB rate spread
    (FRED) + US CPI (Alpha Vantage) for forex. NOTE: all of these are a
    single CURRENT snapshot, not a history — broadcast onto every
    training row (see build_feature_matrix()). This is a real, disclosed
    simplification: a true historical macro series would need a paid
    data vendor for anything beyond the last ~200 daily FRED prints,
    which is short relative to an hourly-bar training set. Treat these
    columns as a coarse "current backdrop" signal, not a rigorously
    time-aligned feature.
    """
    if is_crypto(pair):
        dom = fetch_btc_dominance()
        return {"btc_dominance": dom} if dom is not None else None

    out = {}
    fred_key = os.environ.get("FRED_API_KEY")
    if fred_key:
        us = fetch_fred_series("DFF", fred_key)           # effective Fed funds rate
        ecb = fetch_fred_series("ECBDFR", fred_key)         # ECB deposit facility rate
        if us is not None:
            out["us_rate"] = us
            if ecb is not None:
                out["rate_diff"] = us - ecb
    else:
        log.info("[macro] FRED_API_KEY not set — interest-rate feature skipped")

    av_key = os.environ.get("ALPHA_VANTAGE_KEY")
    if av_key:
        cpi = fetch_alpha_vantage_cpi(av_key)
        if cpi is not None:
            out["us_cpi"] = cpi
    else:
        log.info("[macro] ALPHA_VANTAGE_KEY not set — CPI feature skipped")

    return out or None


def fetch_order_book_imbalance(pair, depth=20):
    """
    Live-only diagnostic (crypto). DESIGN NOTE: the original brief asked
    for a Binance WebSocket depth20 STREAM; this uses the equivalent REST
    snapshot (`/api/v3/depth`) instead. A persistent websocket buys
    nothing here — we only need one imbalance reading per hourly bar
    close, not a continuous stream — and adds real complexity (reconnect/
    backoff handling, keeping a long-lived connection alive between
    predictions). Same underlying data, much simpler.

    NOT fed into the trained feature vector (see order_flow_imbalance
    in add_order_flow_features): a live order-book snapshot has no
    historical equivalent to train against, so using it as a model
    INPUT would mean train and inference see two different definitions
    of the same column — exactly the kind of train/serve skew that
    quietly turns a model's output into noise. It's logged and shown
    alongside the prediction as a human-readable extra signal instead.
    """
    try:
        symbol = pair.replace("/", "")
        r = requests.get(
            "https://api.binance.com/api/v3/depth",
            params={"symbol": symbol, "limit": depth}, timeout=10,
        )
        r.raise_for_status()
        book = r.json()
        bid_vol = sum(float(b[1]) for b in book["bids"])
        ask_vol = sum(float(a[1]) for a in book["asks"])
        total = bid_vol + ask_vol
        return (bid_vol - ask_vol) / total if total > 0 else 0.0
    except Exception as e:
        log.warning(f"[order-flow] Binance depth snapshot failed for {pair}: {e}")
        return None


# ==================================================================== #
# Feature engineering                                                   #
# ==================================================================== #

def add_technical_indicators(df):
    """Module 0 — ~90 indicators (RSI, MACD, Bollinger, ATR, SMA/EMA,
    Stochastic, ADX, OBV, MFI, ...) via the `ta` library rather than
    hand-rolled, so the indicator math itself is a well-tested dependency
    and not a fresh source of bugs."""
    if ta_lib is None:
        raise RuntimeError("the `ta` package is required — pip install ta")
    out = df.copy()
    out = ta_lib.add_all_ta_features(
        out, open="open", high="high", low="low", close="close", volume="volume", fillna=False,
    )
    return out


def add_seasonality_features(df):
    ts = pd.to_datetime(df["timestamp"], utc=True)
    hours = ts.dt.hour + ts.dt.minute / 60.0
    dow = ts.dt.dayofweek
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def _volume_based_flow_proxy(df):
    """
    A historically-computable stand-in for order-flow imbalance: bar
    direction times bar volume, rolling z-scored. This is what BOTH the
    backtest and live inference train/predict on for every bar — the
    live order-book snapshot (fetch_order_book_imbalance) is a separate,
    display-only diagnostic, never mixed into this column (see that
    function's docstring for why that mixing would be a real bug).
    """
    direction = np.sign(df["close"].diff().fillna(0))
    signed_vol = direction * df["volume"].fillna(0)
    roll_mean = signed_vol.rolling(20, min_periods=5).mean()
    roll_std = signed_vol.rolling(20, min_periods=5).std().replace(0, np.nan)
    z = ((signed_vol - roll_mean) / roll_std).fillna(0)
    return np.clip(z, -5, 5)


def add_vwap_deviation(df):
    """Forex leg of Module 4: tick-volume Z-score + VWAP deviation, both
    computed directly from the OHLCV bars (yfinance's own volume field —
    "tick volume", not a real traded-size volume, for spot FX)."""
    df = df.copy()
    typical = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)
    pv = typical * vol
    cum_pv = pv.rolling(20, min_periods=5).sum()
    cum_v = vol.rolling(20, min_periods=5).sum().replace(0, np.nan)
    vwap = (cum_pv / cum_v).bfill().ffill()
    df["vwap_deviation"] = ((df["close"] - vwap) / vwap.replace(0, np.nan)).fillna(0)
    vmean = vol.rolling(20, min_periods=5).mean()
    vstd = vol.rolling(20, min_periods=5).std().replace(0, np.nan)
    df["volume_zscore"] = ((vol - vmean) / vstd).fillna(0)
    return df


def merge_asof_feature(df, feature_df, cols):
    """Forward-fill an irregularly-sampled feature series onto `df`'s bar grid."""
    if feature_df is None or feature_df.empty:
        out = df.copy()
        for c in cols:
            out[c] = 0.0
        return out
    left = df.sort_values("timestamp")
    right = feature_df[["timestamp"] + cols].sort_values("timestamp")
    merged = pd.merge_asof(left, right, on="timestamp", direction="backward")
    for c in cols:
        merged[c] = merged[c].ffill().fillna(0.0)
    return merged


def build_feature_matrix(df, pair):
    """Modules 0 + 3 + 4 combined. NaNs from indicator warmup are left in
    place — the caller drops them (never impute a warmup row)."""
    out = add_technical_indicators(df)
    out = add_seasonality_features(out)
    out = out.assign(order_flow_imbalance=_volume_based_flow_proxy(out))
    if not is_crypto(pair):
        out = add_vwap_deviation(out)

    onchain = fetch_onchain_features(pair)
    if onchain is not None:
        onchain_cols = [c for c in onchain.columns if c != "timestamp"]
        out = merge_asof_feature(out, onchain, onchain_cols)

    macro = fetch_macro_features(pair)
    if macro:
        for k, v in macro.items():
            if v is not None:
                out[k] = v  # single current snapshot — see fetch_macro_features() docstring

    return out


def regime_of(df):
    """Module 7. `ta.add_all_ta_features` names the ADX column
    'trend_adx' — resolved by suffix match rather than a hard-coded name
    so a `ta` version bump that changes naming degrades to a safe default
    (everything 'ranging') instead of crashing."""
    adx_col = next((c for c in df.columns if c.lower() in ("trend_adx", "adx")), None)
    if adx_col is None:
        return pd.Series("ranging", index=df.index)
    return pd.Series(
        np.where(df[adx_col] > ADX_REGIME_THRESHOLD, "trending", "ranging"), index=df.index
    )


def resolve_dead_band_pct(pair):
    return DEAD_BAND_PCT["crypto"] if is_crypto(pair) else DEAD_BAND_PCT["forex"]


def build_target(df, pair, horizon=HORIZON_BARS, dead_band_pct=None):
    """
    Module 2 — direction `horizon` bars ahead, dropping forward moves
    smaller than `dead_band_pct`% as noise. This label is legitimately
    forward-looking (that's what a label is); only FEATURES must never
    see the future, and build_feature_matrix() above never does.
    """
    if dead_band_pct is None:
        dead_band_pct = resolve_dead_band_pct(pair)
    close = df["close"].to_numpy(dtype=float)
    n = len(close)
    fwd = np.full(n, np.nan)
    if n > horizon:
        fwd[:-horizon] = (close[horizon:] - close[:-horizon]) / close[:-horizon] * 100.0
    y = np.full(n, np.nan)
    y[fwd > dead_band_pct] = 1
    y[fwd < -dead_band_pct] = 0
    return y, fwd


def select_top_features(X, y, k=TOP_K_FEATURES):
    """
    Module 1. Uses sklearn's `mutual_info_classif`, NOT
    `mutual_info_regression` as the original brief named: the target
    here is a binary class label (UP/DOWN), and mutual information
    against a discrete target is a classification problem — regression
    MI would run without erroring but is the wrong statistical tool for
    this target. Deliberate, disclosed substitution.
    """
    from sklearn.feature_selection import mutual_info_classif
    mi = mutual_info_classif(X, y, discrete_features=False, random_state=0)
    order = np.argsort(-mi)
    top_idx = order[: min(k, len(order))]
    return list(top_idx), {X.columns[i]: float(mi[i]) for i in top_idx}


def prepare_dataset(pair, raw_ohlcv):
    """Build (X, y, fwd_return, regime) with warmup/dead-band/NaN rows dropped."""
    feats = build_feature_matrix(raw_ohlcv, pair)
    y, fwd = build_target(feats, pair)
    feats = feats.assign(__target=y, __fwd=fwd, __regime=regime_of(feats).to_numpy())

    candidate_cols = [c for c in feats.columns if c not in FEATURE_EXCLUDE and not c.startswith("__")]
    numeric = feats.replace([np.inf, -np.inf], np.nan)
    candidate_cols = [c for c in candidate_cols if numeric[c].notna().any()]

    keep = numeric[candidate_cols].notna().all(axis=1) & numeric["__target"].notna()
    clean = feats.loc[keep].reset_index(drop=True)

    X = clean[candidate_cols]
    y = clean["__target"].astype(int)
    fwd_ret = clean["__fwd"]
    regime = clean["__regime"]
    ts = clean["timestamp"] if "timestamp" in clean else None
    return X, y, fwd_ret, regime, ts


# ==================================================================== #
# Model — LightGBM + Hyperopt (Module 5), regime split (Module 7)       #
# ==================================================================== #

def make_lgbm(params=None):
    if lgb is None:
        raise RuntimeError("lightgbm is required — pip install lightgbm")
    default = dict(
        n_estimators=200, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_samples=30, class_weight="balanced", verbosity=-1, random_state=0,
    )
    if params:
        default.update(params)
    return lgb.LGBMClassifier(**default)


def purged_splits(n, n_folds=WALKFORWARD_FOLDS, horizon=HORIZON_BARS, embargo=EMBARGO_BARS, min_train=100):
    """
    Expanding-window walk-forward with a purge/embargo gap around every
    train/test boundary — a plain chronological split still leaks
    through overlapping labels near the cut (bar i's label and bar i+1's
    label share `horizon-1` bars of outcome). Same discipline the rest
    of this repository's ML pipeline (../ml/validation.py) applies.
    """
    gap = horizon + embargo
    usable = n - gap
    if usable <= min_train + n_folds:
        raise ValueError(f"not enough rows ({n}) for {n_folds} purged folds")
    test_size = (usable - min_train) // n_folds
    if test_size < 5:
        raise ValueError("folds would be too small — reduce n_folds or collect more history")
    for k in range(n_folds):
        train_end = min_train + k * test_size
        test_start = train_end + gap
        test_end = min(test_start + test_size, n)
        if test_end - test_start < 5:
            break
        yield np.arange(0, train_end), np.arange(test_start, test_end)


def hyperopt_tune(X, y, max_evals=HYPEROPT_MAX_EVALS):
    """
    Module 5 tuning. The objective is mean accuracy across PURGED
    walk-forward folds of the TRAINING data only — never a shuffled CV,
    which would hand Hyperopt an inflated, leaky signal to chase and
    produce a materially worse live model while looking better in the
    log. This is a real search (up to `max_evals` distinct configurations
    tried), which is exactly why its result should never be treated as
    the final accuracy number — see run_backtest()'s separate, later
    evaluation on folds the tuning process never touched with this
    specific parameter set... in this simplified script tuning and the
    final backtest do share data (there's no held-out tuning-only slice),
    which is a real, disclosed limitation — see README "Limitations".
    """
    if not HYPEROPT_AVAILABLE:
        log.warning("[tune] hyperopt not installed — using default LightGBM hyperparameters")
        return {}
    if len(np.unique(y)) < 2:
        log.warning("[tune] only one class present — skipping tuning")
        return {}

    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

    space = {
        "max_depth": hp.choice("max_depth", [3, 4, 5, 6]),
        "num_leaves": hp.choice("num_leaves", [7, 15, 31, 63]),
        "learning_rate": hp.loguniform("learning_rate", np.log(0.01), np.log(0.2)),
        "n_estimators": hp.choice("n_estimators", [100, 200, 300]),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_lambda": hp.loguniform("reg_lambda", np.log(0.1), np.log(10)),
    }
    choices = {
        "max_depth": [3, 4, 5, 6], "num_leaves": [7, 15, 31, 63], "n_estimators": [100, 200, 300],
    }

    def objective(params):
        accs = []
        try:
            for train_idx, test_idx in purged_splits(len(X), n_folds=5):
                ytr = y.iloc[train_idx]
                if len(np.unique(ytr)) < 2:
                    continue
                model = make_lgbm(params)
                model.fit(X.iloc[train_idx], ytr)
                pred = model.predict(X.iloc[test_idx])
                accs.append(float(np.mean(pred == y.iloc[test_idx].to_numpy())))
        except ValueError:
            pass
        if not accs:
            return {"loss": 1.0, "status": STATUS_OK}
        return {"loss": -float(np.mean(accs)), "status": STATUS_OK}

    trials = Trials()
    best = fmin(objective, space, algo=tpe.suggest, max_evals=max_evals, trials=trials,
               show_progressbar=False, verbose=False)

    return {
        "max_depth": choices["max_depth"][best["max_depth"]],
        "num_leaves": choices["num_leaves"][best["num_leaves"]],
        "n_estimators": choices["n_estimators"][best["n_estimators"]],
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "reg_lambda": float(best["reg_lambda"]),
    }


def hyperparam_cache_path(pair):
    return CACHE_DIR / f"hparams_{pair.replace('/', '_')}.json"


def load_cached_hyperparams(pair):
    path = hyperparam_cache_path(pair)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_hyperparams(pair, params):
    hyperparam_cache_path(pair).write_text(json.dumps(params, indent=2))


def train_regime_models(pair, df, params=None):
    """
    Modules 1 + 5 + 7 combined: select the top-K features, split by
    regime, train one LightGBM classifier per regime. `params` (from a
    prior --tune run, or {} for LightGBM's own defaults) is applied to
    both regimes' models uniformly — tuning itself is explicit and
    happens once via --tune, not re-run inside this function, so it is
    NOT re-triggered on every Module 6 retrain cycle.
    """
    X_all, y_all, _fwd, regime_all, _ts = prepare_dataset(pair, df)
    models, feature_cols, importances = {}, {}, {}
    for regime in ("trending", "ranging"):
        mask = (regime_all == regime).to_numpy()
        Xr = X_all[mask].reset_index(drop=True)
        yr = y_all[mask].reset_index(drop=True)
        if len(yr) < 60 or len(np.unique(yr)) < 2:
            log.info(f"[train] {pair}/{regime}: not enough rows ({len(yr)}) — skipped")
            continue
        top_idx, _mi = select_top_features(Xr, yr, k=min(TOP_K_FEATURES, Xr.shape[1]))
        cols = list(Xr.columns[top_idx])
        model = make_lgbm(params)
        model.fit(Xr[cols], yr)
        models[regime] = model
        feature_cols[regime] = cols
        importances[regime] = {c: float(v) for c, v in zip(cols, model.feature_importances_)}
    return {"models": models, "feature_cols": feature_cols, "feature_importances": importances}


# ==================================================================== #
# Module 6 — online retraining (background thread)                      #
# ==================================================================== #

class OnlineRetrainer:
    """
    Retrains on a `RETRAIN_WINDOW_BARS`-bar sliding window every
    `RETRAIN_EVERY_N_BARS` new bars, on a background thread so a retrain
    never blocks a prediction. `get_model()` reads through a lock so a
    prediction never sees a half-updated model mid-swap.
    """

    def __init__(self, pair, params=None):
        self.pair = pair
        self.params = params or {}
        self.lock = threading.Lock()
        self.models = {}
        self.feature_cols = {}
        self.bars_seen = 0
        self._retraining = False

    def seed(self, initial):
        with self.lock:
            self.models = initial["models"]
            self.feature_cols = initial["feature_cols"]

    def note_new_bar(self):
        self.bars_seen += 1

    def maybe_retrain(self, full_df):
        if self._retraining:
            return
        if self.models and self.bars_seen < RETRAIN_EVERY_N_BARS:
            return
        self._retraining = True
        threading.Thread(target=self._retrain, args=(full_df.copy(),), daemon=True).start()

    def _retrain(self, full_df):
        try:
            window = full_df.tail(RETRAIN_WINDOW_BARS).reset_index(drop=True)
            result = train_regime_models(self.pair, window, params=self.params)
            if result["models"]:
                with self.lock:
                    self.models = result["models"]
                    self.feature_cols = result["feature_cols"]
                self.bars_seen = 0
                log.info(f"[retrain] {self.pair}: background retrain complete "
                        f"({len(window)}-bar sliding window, regimes={list(result['models'])})")
        except Exception as e:
            log.error(f"[retrain] {self.pair}: background retrain failed: {e}")
        finally:
            self._retraining = False

    def get_model(self, regime):
        with self.lock:
            return self.models.get(regime), self.feature_cols.get(regime)


# ==================================================================== #
# Honest walk-forward backtest (no target-chasing)                      #
# ==================================================================== #

def run_backtest(pair, storage, n_folds=WALKFORWARD_FOLDS, tune=False):
    log.info(f"=== Walk-forward backtest: {pair} ===")
    raw = fetch_ohlcv(pair, timeframe="1h")
    log.info(f"  {len(raw)} bars fetched")
    storage.save_bars(pair, "1h", raw)

    X_all, y_all, fwd_all, regime_all, _ts = prepare_dataset(pair, raw)
    log.info(f"  {len(X_all)} usable rows after warmup + dead-band filtering "
             f"(dropped {len(raw) - len(X_all)})")
    if len(X_all) < 150:
        log.error(f"  not enough usable rows to backtest {pair} — need more history")
        return None

    tuned_params = {}
    if tune:
        if len(np.unique(y_all)) < 2:
            log.warning("  only one class in the full dataset — skipping --tune")
        else:
            top_idx, _mi = select_top_features(X_all, y_all, k=min(TOP_K_FEATURES, X_all.shape[1]))
            cols = list(X_all.columns[top_idx])
            log.info(f"  tuning hyperparameters ({HYPEROPT_MAX_EVALS} evals; this can take a while)...")
            tuned_params = hyperopt_tune(X_all[cols], y_all)
            save_hyperparams(pair, tuned_params)
            log.info(f"  tuned hyperparameters (reused across both regimes): {tuned_params}")
    else:
        tuned_params = load_cached_hyperparams(pair)
        if tuned_params:
            log.info(f"  using cached hyperparameters from a previous --tune run: {tuned_params}")

    fold_accs, fold_base_rates, fold_conf_accs, fold_coverage = [], [], [], []
    per_regime_accs = {"trending": [], "ranging": []}

    for fold_i, (train_idx, test_idx) in enumerate(purged_splits(len(X_all), n_folds=n_folds)):
        ytr_full, yte_full = y_all.iloc[train_idx], y_all.iloc[test_idx]
        if len(np.unique(ytr_full)) < 2:
            continue

        fold_preds = np.full(len(test_idx), np.nan)
        fold_probs = np.full(len(test_idx), np.nan)

        for regime in ("trending", "ranging"):
            rmask_tr = (regime_all.iloc[train_idx] == regime).to_numpy()
            rmask_te = (regime_all.iloc[test_idx] == regime).to_numpy()
            if rmask_tr.sum() < 60 or rmask_te.sum() == 0:
                continue
            ytr_r = ytr_full[rmask_tr]
            if len(np.unique(ytr_r)) < 2:
                continue

            Xtr_r = X_all.iloc[train_idx][rmask_tr]
            top_idx, _mi = select_top_features(Xtr_r, ytr_r, k=min(TOP_K_FEATURES, Xtr_r.shape[1]))
            cols = list(Xtr_r.columns[top_idx])

            model = make_lgbm(tuned_params)
            model.fit(Xtr_r[cols], ytr_r)

            Xte_r = X_all.iloc[test_idx][rmask_te][cols]
            proba = model.predict_proba(Xte_r)[:, 1]
            fold_probs[rmask_te] = proba
            fold_preds[rmask_te] = (proba >= 0.5).astype(int)

            r_correct = fold_preds[rmask_te] == yte_full.to_numpy()[rmask_te]
            per_regime_accs[regime].append(float(np.mean(r_correct)))

        scored = ~np.isnan(fold_preds)
        if scored.sum() == 0:
            continue
        yte_arr = yte_full.to_numpy()
        acc = float(np.mean(fold_preds[scored] == yte_arr[scored]))
        base_rate = float(max(yte_full.mean(), 1 - yte_full.mean()))
        conf_mask = scored & (np.abs(fold_probs - 0.5) >= (CONFIDENCE_THRESHOLD - 0.5))
        conf_acc = (float(np.mean(fold_preds[conf_mask] == yte_arr[conf_mask]))
                   if conf_mask.sum() > 0 else float("nan"))

        fold_accs.append(acc)
        fold_base_rates.append(base_rate)
        fold_conf_accs.append(conf_acc)
        fold_coverage.append(float(conf_mask.sum() / max(scored.sum(), 1)))

        conf_txt = (f"| confidence>{CONFIDENCE_THRESHOLD} acc {conf_acc*100:.2f}% "
                   f"on {fold_coverage[-1]*100:.0f}% of bars" if np.isfinite(conf_acc) else "")
        log.info(f"  fold {fold_i+1}: acc {acc*100:.2f}% (base {base_rate*100:.2f}%) {conf_txt}")

    if not fold_accs:
        log.error(f"  no valid folds for {pair} — nothing to report")
        return None

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    mean_base = float(np.mean(fold_base_rates))
    finite_conf = [a for a in fold_conf_accs if np.isfinite(a)]
    mean_conf_acc = float(np.mean(finite_conf)) if finite_conf else None

    log.info("-" * 64)
    log.info(f"WALK-FORWARD ACCURACY ({pair}): {mean_acc:.4f}  (+/- {std_acc:.4f}, {len(fold_accs)} folds)")
    log.info(f"  base rate (majority class): {mean_base:.4f}  ->  edge {mean_acc - mean_base:+.4f}")
    if mean_conf_acc is not None:
        log.info(f"  confidence>{CONFIDENCE_THRESHOLD} filtered accuracy: {mean_conf_acc:.4f}")
    for regime, accs in per_regime_accs.items():
        if accs:
            log.info(f"  {regime}: mean acc {np.mean(accs):.4f} over {len(accs)} fold-appearances")
    log.info("-" * 64)
    log.info(
        "This is a MEASURED result, not a targeted one — no retry-to-target loop ran "
        "against it. See the module docstring at the top of this file for why that "
        "matters."
    )

    final = train_regime_models(pair, raw, params=tuned_params)
    for regime, model in final["models"].items():
        storage.save_model_metadata(
            pair=pair, regime=regime, n_train_rows=len(X_all),
            wf_acc=mean_acc, wf_acc_std=std_acc, base_rate=mean_base,
            conf_filtered_acc=mean_conf_acc, hyperparams=model.get_params(),
            feature_importances=final["feature_importances"].get(regime, {}),
        )

    return {"accuracy": mean_acc, "std": std_acc, "base_rate": mean_base,
           "confidence_filtered_accuracy": mean_conf_acc, "final": final, "params": tuned_params}


# ==================================================================== #
# Live mode                                                             #
# ==================================================================== #

def predict_latest(pair, storage, retrainer):
    raw = fetch_ohlcv(pair, timeframe="1h", limit=max(RETRAIN_WINDOW_BARS, 600))
    retrainer.note_new_bar()
    retrainer.maybe_retrain(raw)

    feats = build_feature_matrix(raw, pair)
    regime = regime_of(feats).iloc[-1]
    model, cols = retrainer.get_model(regime)
    if model is None or not cols:
        log.warning(f"[live] {pair}: no trained model yet for regime={regime}")
        return None

    latest = feats.iloc[[-1]][cols].replace([np.inf, -np.inf], np.nan)
    if latest.isna().any(axis=None):
        log.warning(f"[live] {pair}: latest bar has missing indicator values (warmup) — skipping")
        return None

    proba_up = float(model.predict_proba(latest)[0, 1])
    direction = 1 if proba_up >= 0.5 else 0
    confidence = abs(proba_up - 0.5) * 2  # 0..1

    live_imbalance = fetch_order_book_imbalance(pair) if is_crypto(pair) else None
    ts = pd.to_datetime(raw["timestamp"].iloc[-1])

    extra = f"  live_book_imbalance={live_imbalance:.3f}" if live_imbalance is not None else ""
    log.info(f"[live] {pair} @ {ts}  regime={regime}  P(up)={proba_up:.3f}  "
            f"confidence={confidence:.3f}{extra}")

    if confidence < CONFIDENCE_THRESHOLD:
        log.info(f"[live] {pair}: confidence below {CONFIDENCE_THRESHOLD} — no signal emitted")
    else:
        log.info(f"[live] {pair}: SIGNAL {'UP' if direction else 'DOWN'} (confidence {confidence:.3f})")

    storage.save_prediction(pair, ts, direction, confidence, regime, HORIZON_BARS)
    return {"direction": direction, "confidence": confidence, "regime": regime, "proba_up": proba_up}


def run_live(pair, storage, poll_seconds=None):
    log.info(f"=== Live mode: {pair} (confidence threshold {CONFIDENCE_THRESHOLD}) ===")
    params = load_cached_hyperparams(pair)
    if not params:
        log.info(f"[live] no cached hyperparameters for {pair} — using LightGBM defaults "
                f"(run `--mode backtest --tune` first for tuned ones)")

    retrainer = OnlineRetrainer(pair, params=params)
    raw = fetch_ohlcv(pair, timeframe="1h", limit=max(RETRAIN_WINDOW_BARS, 600))
    initial = train_regime_models(pair, raw.tail(RETRAIN_WINDOW_BARS), params=params)
    if not initial["models"]:
        log.error(f"[live] {pair}: could not train an initial model — aborting live mode for this pair")
        return
    retrainer.seed(initial)
    log.info(f"[live] {pair}: initial models trained for regimes: {list(initial['models'])}")

    interval = poll_seconds or 3600
    while True:
        try:
            predict_latest(pair, storage, retrainer)
        except Exception as e:
            log.error(f"[live] {pair}: cycle failed: {e}")
        time.sleep(interval)


# ==================================================================== #
# CLI                                                                   #
# ==================================================================== #

def main():
    ap = argparse.ArgumentParser(description="Zero-cost forex + crypto direction predictor.")
    ap.add_argument("--pair", default="BTC/USDT")
    ap.add_argument("--all", action="store_true", help=f"run for all of {DEFAULT_PAIRS}")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--mode", choices=["backtest", "live"], default="backtest")
    ap.add_argument("--supabase", default="true", choices=["true", "false"])
    ap.add_argument("--tune", action="store_true",
                    help=f"run Hyperopt ({HYPEROPT_MAX_EVALS} evals) before training; "
                        f"without this flag, cached (or default) hyperparameters are used")
    ap.add_argument("--folds", type=int, default=WALKFORWARD_FOLDS)
    ap.add_argument("--poll-seconds", type=int, default=None,
                    help="live mode poll interval; default matches --timeframe (3600s for 1h)")
    args = ap.parse_args()

    storage = get_storage(use_supabase=(args.supabase == "true"))
    pairs = DEFAULT_PAIRS if args.all else [args.pair]

    if args.mode == "backtest":
        results = {}
        for pair in pairs:
            try:
                results[pair] = run_backtest(pair, storage, n_folds=args.folds, tune=args.tune)
            except Exception as e:
                log.error(f"[backtest] {pair} failed: {e}")
                results[pair] = None

        log.info("=== Summary ===")
        for pair, r in results.items():
            if r:
                log.info(f"  {pair}: accuracy {r['accuracy']*100:.2f}%  "
                        f"(base {r['base_rate']*100:.2f}%, edge {(r['accuracy']-r['base_rate'])*100:+.2f}pp)")
            else:
                log.info(f"  {pair}: failed / insufficient data")
        return 0

    # live mode
    if len(pairs) == 1:
        run_live(pairs[0], storage, poll_seconds=args.poll_seconds)
    else:
        threads = [
            threading.Thread(target=run_live, args=(pair, storage),
                            kwargs={"poll_seconds": args.poll_seconds}, daemon=True)
            for pair in pairs
        ]
        for t in threads:
            t.start()
        try:
            while any(t.is_alive() for t in threads):
                time.sleep(1)
        except KeyboardInterrupt:
            log.info("[live] stopped by user")
    return 0


if __name__ == "__main__":
    sys.exit(main())
