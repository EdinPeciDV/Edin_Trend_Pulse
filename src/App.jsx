/**
 * src/App.jsx
 * -------------------------------------------------------------------
 * Shell, routing, and the two pieces of state everything else reads:
 * the auth session and the selected strategy.
 *
 * Strategy resolution order:
 *   1. the signed-in user's profiles.risk_tolerance  (authoritative)
 *   2. localStorage                                   (anonymous users)
 *   3. 'conservative'                                 (default — the
 *      cautious option is the default on purpose)
 * -------------------------------------------------------------------
 */

import { useCallback, useEffect, useState } from 'react';
import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom';
import Dashboard from './components/Dashboard.jsx';
import DetailView from './components/DetailView.jsx';
import Settings from './components/Settings.jsx';
import {
  supabase,
  isSupabaseConfigured,
  getProfile,
  saveRiskTolerance,
  getWatchlist,
  toggleWatchlist,
} from './lib/supabaseClient.js';

const STRATEGY_KEY = 'trendpulse.strategy';

/* ------------------------------------------------------------------ */
/* Chrome                                                              */
/* ------------------------------------------------------------------ */

function Header({ session, strategy }) {
  const { pathname } = useLocation();

  const navItems = [
    { to: '/', label: 'Dashboard' },
    { to: '/settings', label: 'Settings' },
  ];

  return (
    <header className="sticky top-0 z-20 border-b border-divider bg-base/95 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-4 px-4 py-3 sm:px-6">
        {/* Wordmark. The pulse dot is the only decoration in the chrome. */}
        <Link to="/" className="flex shrink-0 items-center gap-2">
          <span
            className="h-1.5 w-1.5 bg-amber animate-pulse-tick"
            aria-hidden="true"
          />
          <span className="font-sans text-base font-bold tracking-tight text-ink">
            Trend<span className="text-amber">Pulse</span>
          </span>
        </Link>

        <span className="hidden text-micro uppercase text-ink-faint sm:inline">
          momentum terminal
        </span>

        <nav className="ml-auto flex items-center gap-1">
          {navItems.map((item) => {
            const active =
              item.to === '/' ? pathname === '/' : pathname.startsWith(item.to);
            return (
              <Link
                key={item.to}
                to={item.to}
                className={`px-2.5 py-1.5 text-tick uppercase tracking-wider transition-colors ${
                  active ? 'text-amber' : 'text-ink-faint hover:text-ink-muted'
                }`}
              >
                {item.label}
              </Link>
            );
          })}

          <span className="ml-2 hidden border border-hairline px-2 py-1 text-micro uppercase text-ink-muted sm:inline">
            {strategy}
          </span>

          {isSupabaseConfigured && (
            <span className="ml-1 hidden text-micro uppercase text-ink-faint md:inline">
              {session ? '● signed in' : '○ guest'}
            </span>
          )}
        </nav>
      </div>
    </header>
  );
}

function Footer() {
  return (
    <footer className="mt-10 border-t border-divider">
      <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6">
        <p className="max-w-3xl text-micro normal-case leading-relaxed text-ink-faint">
          <span className="text-amber">Not investment advice.</span> TrendPulse
          reports standard technical indicators on recent price data. The
          confidence figure measures internal agreement between its own rules,
          not the probability of any market outcome, and the heuristic has not
          been validated against historical returns. Check the prediction log on
          any detail page to see how its calls have actually performed. Trading
          crypto and leveraged forex carries substantial risk of loss.
        </p>
        <p className="mt-3 text-micro uppercase text-ink-faint">
          Crypto via Binance · Forex via Twelve Data · 5-minute candles
        </p>
      </div>
    </footer>
  );
}

function NotFound() {
  return (
    <div className="panel px-6 py-10 text-center">
      <p className="label-amber">404</p>
      <h1 className="mt-2 font-sans text-xl font-semibold text-ink">
        No such screen
      </h1>
      <Link to="/" className="btn-ghost mt-4">
        Back to dashboard
      </Link>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

export default function App() {
  const [session, setSession] = useState(null);
  const [profile, setProfile] = useState(null);
  const [watchlist, setWatchlist] = useState([]);
  const [strategy, setStrategy] = useState(
    () => localStorage.getItem(STRATEGY_KEY) || 'conservative'
  );
  const [isSaving, setIsSaving] = useState(false);

  /* ---- Auth session ---- */
  useEffect(() => {
    if (!supabase) return;

    supabase.auth.getSession().then(({ data }) => setSession(data.session ?? null));

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, newSession) => {
      setSession(newSession);
      if (!newSession) {
        setProfile(null);
        setWatchlist([]);
      }
    });

    return () => subscription.unsubscribe();
  }, []);

  /* ---- Load profile + watchlist on sign-in ---- */
  useEffect(() => {
    if (!session?.user?.id) return;
    let cancelled = false;

    (async () => {
      const [p, w] = await Promise.all([
        getProfile(session.user.id),
        getWatchlist(session.user.id),
      ]);
      if (cancelled) return;

      setProfile(p);
      setWatchlist(w);

      // The account's saved preference wins over whatever this browser had.
      if (p?.risk_tolerance && p.risk_tolerance !== strategy) {
        setStrategy(p.risk_tolerance);
        localStorage.setItem(STRATEGY_KEY, p.risk_tolerance);
      }
    })();

    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [session?.user?.id]);

  /* ---- Strategy change ---- */
  const handleStrategyChange = useCallback(
    async (next) => {
      // Update locally first so the UI never waits on the network.
      setStrategy(next);
      localStorage.setItem(STRATEGY_KEY, next);

      if (!session?.user?.id) return;
      setIsSaving(true);
      try {
        const saved = await saveRiskTolerance(session.user.id, next);
        setProfile((prev) => ({ ...prev, ...saved }));
      } catch (err) {
        console.error('[TrendPulse] could not save strategy:', err.message);
      } finally {
        setIsSaving(false);
      }
    },
    [session?.user?.id]
  );

  /* ---- Watchlist ---- */
  const handleToggleWatch = useCallback(
    async (symbol) => {
      if (!session?.user?.id) return;
      const isWatched = watchlist.includes(symbol);

      // Optimistic update, rolled back if the write fails.
      setWatchlist((prev) =>
        isWatched ? prev.filter((s) => s !== symbol) : [...prev, symbol]
      );

      try {
        await toggleWatchlist(session.user.id, symbol, isWatched);
      } catch (err) {
        console.error('[TrendPulse] watchlist write failed:', err.message);
        setWatchlist((prev) =>
          isWatched ? [...prev, symbol] : prev.filter((s) => s !== symbol)
        );
      }
    },
    [session?.user?.id, watchlist]
  );

  const plan = profile?.plan || 'free';

  return (
    <div className="flex min-h-screen flex-col">
      <Header session={session} strategy={strategy} />

      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-5 sm:px-6">
        <Routes>
          <Route
            path="/"
            element={
              <Dashboard
                strategy={strategy}
                watchlist={watchlist}
                onToggleWatch={handleToggleWatch}
                canWatch={Boolean(session)}
                plan={plan}
              />
            }
          />
          <Route path="/symbol/:symbol" element={<DetailView strategy={strategy} plan={plan} />} />
          <Route
            path="/settings"
            element={
              <Settings
                strategy={strategy}
                onStrategyChange={handleStrategyChange}
                session={session}
                profile={profile}
                isSaving={isSaving}
              />
            }
          />
          <Route path="/dashboard" element={<Navigate to="/" replace />} />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </main>

      <Footer />
    </div>
  );
}
