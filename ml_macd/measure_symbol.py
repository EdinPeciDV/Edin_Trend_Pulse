"""
ml_macd/measure_symbol.py — throwaway measurement harness, not part of
the module. Run once per symbol as a SEPARATE subprocess (peak working
set is process-lifetime, so a fresh process per symbol gives a true
per-symbol peak, not a cumulative one) — matching the committed
"process one symbol at a time and release it" design.

Peak working set measured via the native Windows API
(GetProcessMemoryInfo through ctypes) — no new dependency added;
psutil is not installed and was not added just for this measurement.
"""
import ctypes
import json
import sys
import time

sys.path.insert(0, "ml_macd")


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


_psapi = ctypes.WinDLL("psapi.dll")
_kernel32 = ctypes.WinDLL("kernel32.dll")
_psapi.GetProcessMemoryInfo.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_uint32]
_psapi.GetProcessMemoryInfo.restype = ctypes.c_int
_kernel32.GetCurrentProcess.restype = ctypes.c_void_p
# Explicit argtypes/restype are required here — without them ctypes
# silently misinterprets the pointer/return types on 64-bit Windows
# and GetProcessMemoryInfo "succeeds" while leaving every field 0.
# Found exactly that way: first version of this function returned
# peak_working_set_mb() == 0.0 for a real, memory-heavy run.


def peak_working_set_mb():
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    ok = _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        raise OSError(f"GetProcessMemoryInfo failed: {ctypes.get_last_error()}")
    return counters.PeakWorkingSetSize / (1024 * 1024)


def main():
    symbol, timeframe, htf_timeframe, asset_class = sys.argv[1:5]

    from data import load_candles
    from macd_features import build_features
    from labels import build_label_grid

    t0 = time.time()
    candles = load_candles(symbol, timeframe, asset_class)
    htf = load_candles(symbol, htf_timeframe, asset_class)
    t_load = time.time() - t0

    t0 = time.time()
    X, valid = build_features(candles, htf)
    t_features = time.time() - t0

    t0 = time.time()
    grid = build_label_grid(candles)
    t_labels = time.time() - t0

    result = {
        "symbol": symbol, "timeframe": timeframe,
        "n_bars": len(candles["close"]),
        "n_features": X.shape[1],
        "valid_rows": int(valid.sum()),
        "t_load_s": round(t_load, 2),
        "t_features_s": round(t_features, 2),
        "t_labels_s": round(t_labels, 2),
        "t_total_s": round(t_load + t_features + t_labels, 2),
        "peak_working_set_mb": round(peak_working_set_mb(), 1),
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
