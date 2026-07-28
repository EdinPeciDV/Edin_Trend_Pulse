/**
 * src/lib/api.js
 * -------------------------------------------------------------------
 * The frontend's only route to the backend: /api/* , which netlify.toml
 * rewrites to /.netlify/functions/*.
 *
 * The browser never talks to Binance, Twelve Data or Supabase's
 * service-role surface directly. Everything goes through our functions,
 * which keeps API keys server-side and gives us one place to add
 * caching or rate limiting later.
 * -------------------------------------------------------------------
 */

import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = '/api';

/* ------------------------------------------------------------------ */
/* Fetch wrappers                                                      */
/* ------------------------------------------------------------------ */

async function request(path, { signal } = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    signal,
    headers: { Accept: 'application/json' },
  });

  let payload;
  try {
    payload = await res.json();
  } catch {
    throw new Error(`Server returned a non-JSON response (HTTP ${res.status}).`);
  }

  if (!res.ok) {
    throw new Error(payload?.error || `Request failed with HTTP ${res.status}.`);
  }
  return payload;
}

/** Dashboard: every instrument, latest signal. */
export function fetchDashboard(strategy = 'conservative', options = {}) {
  return request(`/get-analysis?strategy=${encodeURIComponent(strategy)}`, options);
}

/** Detail: one symbol with candle series and the prediction log. */
export function fetchSymbol(symbol, strategy = 'conservative', options = {}) {
  return request(
    `/get-analysis?symbol=${encodeURIComponent(symbol)}` +
      `&strategy=${encodeURIComponent(strategy)}&history=1`,
    options
  );
}

/** Manual ingest trigger — mainly useful in development. */
export function triggerIngest() {
  return request('/fetch-market-data');
}

/* ------------------------------------------------------------------ */
/* Polling hook                                                        */
/* ------------------------------------------------------------------ */

/**
 * usePolling — the 60-second refresh loop the brief asks for, with the
 * details that matter in practice:
 *
 *   * aborts the in-flight request on unmount or dependency change, so
 *     React 18 StrictMode's double-mount can't leave a zombie fetch
 *   * pauses while the tab is hidden and refetches immediately on
 *     return, instead of hammering the API in background tabs
 *   * keeps the last good data visible during a failed refresh, so a
 *     transient upstream blip doesn't blank the dashboard
 *
 * @param {Function} fetcher  receives { signal }, returns a promise
 * @param {number} intervalMs
 * @param {Array} deps
 */
export function usePolling(fetcher, intervalMs = 60000, deps = []) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  const abortRef = useRef(null);
  const timerRef = useRef(null);
  const mountedRef = useRef(true);

  const load = useCallback(
    async (isBackground = false) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      if (isBackground) setIsRefreshing(true);
      else setLoading(true);

      try {
        const result = await fetcher({ signal: controller.signal });
        if (!mountedRef.current || controller.signal.aborted) return;
        setData(result);
        setError(null);
        setLastUpdated(new Date());
      } catch (err) {
        if (err.name === 'AbortError' || !mountedRef.current) return;
        // Surface the error but keep whatever we already had on screen.
        setError(err.message || 'Could not reach the analysis service.');
      } finally {
        if (mountedRef.current) {
          setLoading(false);
          setIsRefreshing(false);
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps
  );

  useEffect(() => {
    mountedRef.current = true;
    load(false);

    const schedule = () => {
      clearInterval(timerRef.current);
      timerRef.current = setInterval(() => {
        if (document.visibilityState === 'visible') load(true);
      }, intervalMs);
    };
    schedule();

    // Refetch the moment the tab becomes visible again.
    const onVisibility = () => {
      if (document.visibilityState === 'visible') {
        load(true);
        schedule();
      }
    };
    document.addEventListener('visibilitychange', onVisibility);

    return () => {
      mountedRef.current = false;
      clearInterval(timerRef.current);
      abortRef.current?.abort();
      document.removeEventListener('visibilitychange', onVisibility);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [load, intervalMs]);

  return {
    data,
    error,
    loading,
    isRefreshing,
    lastUpdated,
    refresh: () => load(true),
  };
}
