"""
Stage 5 driver — throwaway. Backfills EUR/USD and GBP/USD, all three
timeframes (15m, 1h, 4h — both the 15m+1h and 1h+4h pairings, per the
decision to keep the entry-timeframe choice open at modelling time),
from Twelve Data's real confirmed floor (2020-01-30, identical across
all three timeframes for EUR/USD, confirmed by direct probe) through
the present.
"""
import sys
from datetime import datetime, timezone
sys.path.insert(0, "ml_macd")
from data import backfill_forex

SYMBOLS = {
    "EUR/USD": "EUR/USD",
    "GBP/USD": "GBP/USD",
}
TIMEFRAMES = ["15m", "1h", "4h"]
START = datetime(2020, 1, 30, tzinfo=timezone.utc)
START_MS = int(START.timestamp() * 1000)

for ml_symbol, td_symbol in SYMBOLS.items():
    for tf in TIMEFRAMES:
        print(f"\n{'='*70}\n{ml_symbol} {tf}: {START.isoformat()}..now\n{'='*70}")
        backfill_forex(ml_symbol, td_symbol, tf, START_MS, dry_run=False)

print("\nSTAGE 5 COMPLETE")
