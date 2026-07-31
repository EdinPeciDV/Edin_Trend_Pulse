/**
 * src/components/Settings.jsx
 * -------------------------------------------------------------------
 * Strategy toggle plus account state.
 *
 * The toggle is not cosmetic: it changes SMA/RSI lookback lengths and
 * the confidence floor below which no call is made. The page spells out
 * exactly what each profile changes, because a setting whose effect you
 * can't see is a setting nobody trusts.
 * -------------------------------------------------------------------
 */

import { useState } from 'react';
import { STRATEGY_PROFILES } from '../lib/helpers.js';
import {
  signInWithEmail,
  signInWithGitHub,
  signInWithGoogle,
  signOut,
  isSupabaseConfigured,
} from '../lib/supabaseClient.js';
import { openCheckout, isPaddleConfigured } from '../lib/paddle.js';
import { PLAN_LIMITS } from '../lib/plans.js';

/* ------------------------------------------------------------------ */
/* Strategy comparison                                                 */
/* ------------------------------------------------------------------ */

function StrategyPanel({ strategy, onChange, isSaving }) {
  const rows = [
    ['SMA lookback', (p) => `${p.smaPeriod} periods`],
    ['RSI lookback', (p) => `${p.rsiPeriod} periods`],
    ['Bollinger lookback', (p) => `${p.bbPeriod} periods`],
    ['Confidence floor', (p) => `${p.minConfidence}%`],
    ['Stop loss · crypto', (p) => `−${p.stopLossPct.crypto}%`],
    ['Stop loss · forex', (p) => `−${p.stopLossPct.forex}%`],
    ['Sideways threshold', (p) => `${(p.neutralBandwidth * 100).toFixed(2)}% bandwidth`],
  ];

  return (
    <section className="panel edge-top">
      <header className="border-b border-divider px-4 py-3">
        <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
          Strategy
        </h2>
        <p className="mt-0.5 text-micro normal-case text-ink-faint">
          Changes how long the indicators look back and how much agreement a call needs.
        </p>
      </header>

      <div className="px-4 py-4">
        <div className="segment max-w-sm">
          {Object.entries(STRATEGY_PROFILES).map(([key, profile]) => (
            <button
              key={key}
              type="button"
              onClick={() => onChange(key)}
              disabled={isSaving}
              aria-pressed={strategy === key}
              className={`segment-item ${strategy === key ? 'segment-item-active' : ''}`}
            >
              {profile.label}
            </button>
          ))}
        </div>

        <p className="mt-3 max-w-prose text-tick normal-case leading-relaxed text-ink-muted">
          {STRATEGY_PROFILES[strategy]?.blurb}
        </p>

        {/* Side-by-side, so the tradeoff is visible rather than described. */}
        <div className="mt-5 overflow-x-auto">
          <table className="term-table">
            <thead>
              <tr>
                <th>Parameter</th>
                <th className={strategy === 'aggressive' ? 'text-amber' : ''}>
                  Aggressive
                </th>
                <th className={strategy === 'conservative' ? 'text-amber' : ''}>
                  Conservative
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map(([label, get]) => (
                <tr key={label}>
                  <td className="text-ink-faint">{label}</td>
                  <td
                    className={`tnum ${
                      strategy === 'aggressive' ? 'text-amber' : 'text-ink-muted'
                    }`}
                  >
                    {get(STRATEGY_PROFILES.aggressive)}
                  </td>
                  <td
                    className={`tnum ${
                      strategy === 'conservative' ? 'text-amber' : 'text-ink-muted'
                    }`}
                  >
                    {get(STRATEGY_PROFILES.conservative)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Account                                                             */
/* ------------------------------------------------------------------ */

function AccountPanel({ session, profile }) {
  const [email, setEmail] = useState('');
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);

  const handleMagicLink = async () => {
    if (!email.includes('@')) {
      setStatus({ type: 'error', text: 'Enter a valid email address.' });
      return;
    }
    setBusy(true);
    setStatus(null);
    try {
      await signInWithEmail(email);
      setStatus({ type: 'ok', text: `Sign-in link sent to ${email}. Check your inbox.` });
    } catch (err) {
      setStatus({ type: 'error', text: err.message });
    } finally {
      setBusy(false);
    }
  };

  const handleGitHub = async () => {
    setBusy(true);
    setStatus(null);
    try {
      await signInWithGitHub();
    } catch (err) {
      setStatus({ type: 'error', text: err.message });
      setBusy(false);
    }
  };

  const handleGoogle = async () => {
    setBusy(true);
    setStatus(null);
    try {
      await signInWithGoogle();
    } catch (err) {
      setStatus({ type: 'error', text: err.message });
      setBusy(false);
    }
  };

  if (!isSupabaseConfigured) {
    return (
      <section className="panel px-4 py-4">
        <h2 className="label-amber">Accounts are not configured</h2>
        <p className="mt-2 max-w-prose text-tick normal-case leading-relaxed text-ink-muted">
          Set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY to enable sign-in.
          Without them the terminal still works — your strategy choice is kept
          in this browser instead of your account.
        </p>
      </section>
    );
  }

  return (
    <section className="panel edge-top">
      <header className="border-b border-divider px-4 py-3">
        <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
          Account
        </h2>
        <p className="mt-0.5 text-micro normal-case text-ink-faint">
          Sign in to sync your strategy and watchlist across devices.
        </p>
      </header>

      <div className="px-4 py-4">
        {session ? (
          <div className="space-y-3">
            <div className="data-row">
              <span className="label">Signed in as</span>
              <span className="data-value">{session.user.email || profile?.username}</span>
            </div>
            <div className="data-row">
              <span className="label">Saved risk tolerance</span>
              <span className="data-value text-amber">
                {profile?.risk_tolerance || 'conservative'}
              </span>
            </div>
            <button type="button" onClick={signOut} className="btn-ghost mt-2">
              Sign out
            </button>
          </div>
        ) : (
          <div className="max-w-sm space-y-4">
            <div>
              <label htmlFor="email" className="label block">
                Email
              </label>
              <div className="mt-1.5 flex gap-2">
                <input
                  id="email"
                  type="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleMagicLink()}
                  placeholder="you@example.com"
                  autoComplete="email"
                  className="min-w-0 flex-1 border border-hairline bg-void px-3 py-2 text-tick text-ink placeholder:text-ink-faint focus:border-amber/60"
                />
                <button
                  type="button"
                  onClick={handleMagicLink}
                  disabled={busy}
                  className="btn-amber shrink-0"
                >
                  {busy ? 'Sending' : 'Send link'}
                </button>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <span className="h-px flex-1 bg-hairline" />
              <span className="label">or</span>
              <span className="h-px flex-1 bg-hairline" />
            </div>

            <button
              type="button"
              onClick={handleGoogle}
              disabled={busy}
              className="btn-ghost w-full"
            >
              Continue with Google
            </button>

            <button
              type="button"
              onClick={handleGitHub}
              disabled={busy}
              className="btn-ghost w-full"
            >
              Continue with GitHub
            </button>

            {status && (
              <p
                className={`text-tick normal-case ${
                  status.type === 'error' ? 'text-amber' : 'text-up'
                }`}
                role="status"
              >
                {status.text}
              </p>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Billing                                                             */
/* ------------------------------------------------------------------ */

function BillingPanel({ session, profile }) {
  const [busyPlan, setBusyPlan] = useState(null);
  const [error, setError] = useState(null);
  const currentPlan = profile?.plan || 'free';

  const handleSubscribe = async (planKey) => {
    const plan = PLAN_LIMITS[planKey];
    setError(null);
    setBusyPlan(planKey);
    try {
      await openCheckout(plan.paddlePriceId, session?.user?.id, session?.user?.email);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyPlan(null);
    }
  };

  return (
    <section className="panel edge-top">
      <header className="border-b border-divider px-4 py-3">
        <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
          Billing
        </h2>
        <p className="mt-0.5 text-micro normal-case text-ink-faint">
          Current plan: <span className="text-amber uppercase">{currentPlan}</span>
        </p>
      </header>

      <div className="grid grid-cols-1 gap-px bg-hairline sm:grid-cols-3">
        {Object.values(PLAN_LIMITS).map((plan) => {
          const isCurrent = plan.key === currentPlan;
          return (
            <div key={plan.key} className="bg-panel px-4 py-3.5">
              <p className="label">{plan.label}</p>
              <p className="tnum mt-1 text-lg text-ink">{plan.priceLabel}</p>
              <ul className="mt-2 space-y-0.5 text-tick normal-case text-ink-muted">
                <li>
                  {plan.maxInstruments} pair{plan.maxInstruments === 1 ? '' : 's'}
                </li>
                <li>{plan.historyDays}-day prediction history</li>
              </ul>

              {isCurrent ? (
                <p className="mt-3 text-micro uppercase text-up">current plan</p>
              ) : plan.key === 'free' ? (
                <p className="mt-3 text-micro uppercase text-ink-faint">default</p>
              ) : (
                <button
                  type="button"
                  onClick={() => handleSubscribe(plan.key)}
                  disabled={Boolean(busyPlan) || !isPaddleConfigured || !session}
                  className="btn-amber mt-3 w-full"
                >
                  {busyPlan === plan.key ? 'Opening checkout…' : 'Subscribe'}
                </button>
              )}
            </div>
          );
        })}
      </div>

      <div className="border-t border-hairline px-4 py-2.5">
        {!session && (
          <p className="text-tick normal-case text-ink-faint">
            Sign in above to subscribe — Paddle needs an account to attach the plan to.
          </p>
        )}
        {session && !isPaddleConfigured && (
          <p className="text-tick normal-case text-ink-faint">
            Billing isn't configured yet (VITE_PADDLE_CLIENT_TOKEN is not set).
          </p>
        )}
        {error && <p className="text-tick normal-case text-amber">{error}</p>}
        <p className="mt-1 text-micro normal-case text-ink-faint">
          Plan changes take effect once Paddle confirms the payment — usually a few
          seconds, sometimes up to a minute. Cancelling in Paddle's customer portal
          reverts you to Free at the end of the billing period.
        </p>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function Settings({ strategy, onStrategyChange, session, profile, isSaving }) {
  return (
    <div className="max-w-4xl space-y-3">
      <div>
        <h1 className="font-sans text-xl font-semibold text-ink">Settings</h1>
        <p className="mt-1 text-tick text-ink-faint">
          {session
            ? 'Changes are saved to your account.'
            : 'Changes are saved in this browser. Sign in to sync them.'}
        </p>
      </div>

      <StrategyPanel strategy={strategy} onChange={onStrategyChange} isSaving={isSaving} />
      <AccountPanel session={session} profile={profile} />
      <BillingPanel session={session} profile={profile} />

      {/* Scope + limits. Deliberately not buried in a footer. */}
      <section className="panel border-amber/30 px-4 py-4">
        <h2 className="label-amber">What this tool is and is not</h2>
        <div className="mt-2 max-w-prose space-y-2 text-tick normal-case leading-relaxed text-ink-muted">
          <p>
            TrendPulse applies textbook technical indicators — RSI, SMA,
            Bollinger Bands, VWAP — to recent price data and reports where
            price sits relative to them. That is all it does.
          </p>
          <p>
            The confidence percentage counts how many of its own conditions
            agree. It is not a probability that the market will move that way,
            and it has not been validated against historical returns. The
            thresholds (40–60, 65, 70/30) are conventional defaults, not values
            fitted to these instruments.
          </p>
          <p>
            The prediction log on each detail page is the only real evidence of
            whether the heuristic works. Read it before acting on anything here,
            and remember that direction accuracy ignores how far price moved,
            spreads, slippage, and funding costs — so even a good hit rate does
            not imply a profitable strategy.
          </p>
          <p className="text-ink-faint">
            Nothing here is investment advice, and none of it is personalised to
            your situation. Trading crypto and leveraged forex can lose you more
            than you put in. Speak to a licensed financial adviser before risking
            money you cannot afford to lose.
          </p>
        </div>
      </section>
    </div>
  );
}
