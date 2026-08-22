"""
Stage 4 driver — throwaway, not part of the ml_macd module. Backfills
all 9 remaining crypto symbols (BTC/USDT was Stage 3), both entry and
4x-higher timeframe each, from their real earliest-available archive
month (probed directly against data.binance.vision, not guessed)
through 2026-07.
"""
import sys
sys.path.insert(0, "ml_macd")
from data import backfill_crypto, ENTRY_TO_HIGHER_TF

SYMBOLS = {
    # ml_macd symbol -> (binance symbol, earliest available 1h archive month)
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

for ml_symbol, (binance_symbol, start) in SYMBOLS.items():
    for tf in ["1h", ENTRY_TO_HIGHER_TF["1h"]]:
        print(f"\n{'='*70}\n{ml_symbol} {tf}: {start}..{END}\n{'='*70}")
        backfill_crypto(ml_symbol, binance_symbol, tf, start, END, dry_run=False)

print("\nSTAGE 4 COMPLETE")
