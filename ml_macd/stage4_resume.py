"""
Stage 4 resume — throwaway. Uses resolve_backfill_start() (DB-derived
resume, --resume equivalent) per symbol/timeframe so already-complete
symbols just harmlessly re-fetch their last month, AVAX/USDT 4h picks
up from where the duplicate-row crash stopped it, and LINK/LTC/DOT
start fresh.
"""
import sys
sys.path.insert(0, "ml_macd")
from data import backfill_crypto, resolve_backfill_start, ENTRY_TO_HIGHER_TF, ASSET_CLASS_CRYPTO

SYMBOLS = {
    "ETH/USDT": ("ETHUSDT", "2017-08"),
    "SOL/USDT": ("SOLUSDT", "2020-08"),
    "XRP/USDT": ("XRPUSDT", "2018-05"),
    "ADA/USDT": ("ADAUSDT", "2018-04"),
    "DOGE/USDT": ("DOGEUSDT", "2019-07"),
    "AVAX/USDT": ("AVAXUSDT", "2020-09"),
    "LINK/USDT": ("LINKUSDT", "2019-01"),
    "LTC/USDT": ("LTCUSDT", "2017-12"),
    "DOT/USDT": ("DOTUSDT", "2020-08"),
}
END = "2026-07"

for ml_symbol, (binance_symbol, earliest) in SYMBOLS.items():
    for tf in ["1h", ENTRY_TO_HIGHER_TF["1h"]]:
        start = resolve_backfill_start(ml_symbol, tf, ASSET_CLASS_CRYPTO, earliest, resume=True)
        print(f"\n{'='*70}\n{ml_symbol} {tf}: {start}..{END}\n{'='*70}")
        backfill_crypto(ml_symbol, binance_symbol, tf, start, END, dry_run=False)

print("\nSTAGE 4 COMPLETE (resumed)")
