/**
 * src/components/Admin.jsx
 * -------------------------------------------------------------------
 * Manual plan management + system health, for the site owner only.
 *
 * This page is UX-only gating — it hides itself from anyone whose
 * email isn't in VITE_ADMIN_EMAILS, but that check happens in the
 * browser and proves nothing. The real gate is server-side: every
 * request below carries the signed-in user's Supabase access token,
 * and netlify/functions/admin.js re-checks it against the (private)
 * ADMIN_EMAILS env var before touching anything. A non-admin who
 * forces this route open just gets 403s from every fetch.
 * -------------------------------------------------------------------
 */

import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabaseClient.js';

const ADMIN_EMAILS = (import.meta.env.VITE_ADMIN_EMAILS || '')
  .split(',')
  .map((e) => e.trim().toLowerCase())
  .filter(Boolean);

const PLAN_KEYS = ['free', 'basic', 'pro'];

async function callAdmin(path, { method = 'GET', body } = {}) {
  const { data } = await supabase.auth.getSession();
  const token = data?.session?.access_token;
  if (!token) throw new Error('Not signed in.');

  const res = await fetch(`/api/admin${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${token}`,
      ...(body ? { 'Content-Type': 'application/json' } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  const payload = await res.json().catch(() => null);
  if (!res.ok) {
    throw new Error(payload?.error || `Request failed with HTTP ${res.status}.`);
  }
  return payload;
}

/* ------------------------------------------------------------------ */
/* Health                                                              */
/* ------------------------------------------------------------------ */

function EnvRow({ name, present }) {
  return (
    <div className="data-row">
      <span className="label">{name}</span>
      <span className={`text-tick uppercase ${present ? 'text-up' : 'text-down'}`}>
        {present ? 'set' : 'missing'}
      </span>
    </div>
  );
}

function HealthPanel({ health, onRefresh, isRefreshing }) {
  if (!health) return null;

  return (
    <section className="panel edge-top">
      <header className="flex items-center justify-between border-b border-divider px-4 py-3">
        <div>
          <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
            System health
          </h2>
          <p className="mt-0.5 text-micro normal-case text-ink-faint">
            Environment config and data freshness.
          </p>
        </div>
        <button type="button" onClick={onRefresh} className="btn-ghost" disabled={isRefreshing}>
          {isRefreshing ? 'Checking…' : 'Recheck'}
        </button>
      </header>

      <div className="px-4 py-4">
        <p className="label mb-2">Environment variables</p>
        <div className="max-w-md">
          {Object.entries(health.env).map(([name, present]) => (
            <EnvRow key={name} name={name} present={present} />
          ))}
        </div>

        <p className="label mb-2 mt-5">Snapshot freshness</p>
        {health.snapshotError && (
          <p className="text-tick normal-case text-amber">{health.snapshotError}</p>
        )}
        <div className="overflow-x-auto">
          <table className="term-table">
            <thead>
              <tr>
                <th>Symbol</th>
                <th>Last bucket</th>
                <th>Age</th>
              </tr>
            </thead>
            <tbody>
              {health.snapshotStaleness.map((row) => (
                <tr key={row.symbol}>
                  <td>{row.symbol}</td>
                  <td className="tnum">{new Date(row.bucketAt).toLocaleString()}</td>
                  <td className={`tnum ${row.ageMinutes > 30 ? 'text-amber' : 'text-up'}`}>
                    {row.ageMinutes}m
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {health.missingSymbols.length > 0 && (
          <p className="mt-3 text-tick normal-case text-amber">
            No snapshot at all yet for: {health.missingSymbols.join(', ')}
          </p>
        )}
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Users                                                               */
/* ------------------------------------------------------------------ */

function UserRow({ user, onSetPlan, busy }) {
  const isBusy = busy === user.id;

  return (
    <tr>
      <td>{user.email}</td>
      <td className="uppercase text-amber">
        {user.plan}
        {user.isLifetime && <span className="ml-1 text-ink-faint">(lifetime)</span>}
      </td>
      <td className="text-ink-faint">{new Date(user.createdAt).toLocaleDateString()}</td>
      <td>
        <div className="flex flex-wrap gap-1.5">
          {PLAN_KEYS.map((plan) => (
            <button
              key={plan}
              type="button"
              disabled={isBusy}
              onClick={() => onSetPlan(user.id, plan, user.isLifetime)}
              className={`segment-item border border-hairline ${
                user.plan === plan && !user.isLifetime ? 'segment-item-active' : ''
              }`}
            >
              {plan}
            </button>
          ))}
          <button
            type="button"
            disabled={isBusy}
            onClick={() => onSetPlan(user.id, user.plan === 'free' ? 'pro' : user.plan, !user.isLifetime)}
            className={`segment-item border border-hairline ${user.isLifetime ? 'segment-item-active' : ''}`}
          >
            lifetime
          </button>
        </div>
      </td>
    </tr>
  );
}

function UsersPanel({ users, onSetPlan, busy }) {
  return (
    <section className="panel edge-top">
      <header className="border-b border-divider px-4 py-3">
        <h2 className="font-sans text-sm font-semibold uppercase tracking-widest text-ink">
          Users
        </h2>
        <p className="mt-0.5 text-micro normal-case text-ink-faint">
          {users.length} account{users.length === 1 ? '' : 's'}. Plan changes take effect
          immediately — this bypasses Paddle entirely.
        </p>
      </header>

      <div className="overflow-x-auto px-4 py-4">
        <table className="term-table">
          <thead>
            <tr>
              <th>Email</th>
              <th>Plan</th>
              <th>Joined</th>
              <th>Set plan</th>
            </tr>
          </thead>
          <tbody>
            {users.map((u) => (
              <UserRow key={u.id} user={u} onSetPlan={onSetPlan} busy={busy} />
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Page                                                                */
/* ------------------------------------------------------------------ */

export default function Admin({ session }) {
  const [health, setHealth] = useState(null);
  const [users, setUsers] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [refreshingHealth, setRefreshingHealth] = useState(false);
  const [busyUserId, setBusyUserId] = useState(null);

  const isAllowed = Boolean(
    session?.user?.email && ADMIN_EMAILS.includes(session.user.email.toLowerCase())
  );

  const loadHealth = useCallback(async () => {
    setRefreshingHealth(true);
    try {
      const { health: h } = await callAdmin('?action=health');
      setHealth(h);
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshingHealth(false);
    }
  }, []);

  const loadUsers = useCallback(async () => {
    try {
      const { users: u } = await callAdmin('?action=list-users');
      setUsers(u);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (!isAllowed) {
      setLoading(false);
      return;
    }
    setLoading(true);
    Promise.all([loadHealth(), loadUsers()]).finally(() => setLoading(false));
  }, [isAllowed, loadHealth, loadUsers]);

  const handleSetPlan = async (userId, plan, isLifetime) => {
    setBusyUserId(userId);
    setError(null);
    try {
      const { profile } = await callAdmin('', {
        method: 'POST',
        body: { action: 'set-plan', userId, plan, isLifetime },
      });
      setUsers((prev) =>
        prev.map((u) =>
          u.id === userId ? { ...u, plan: profile.plan, isLifetime: profile.is_lifetime } : u
        )
      );
    } catch (err) {
      setError(err.message);
    } finally {
      setBusyUserId(null);
    }
  };

  if (!session) {
    return (
      <div className="panel px-6 py-10 text-center">
        <p className="label-amber">Admin</p>
        <h1 className="mt-2 font-sans text-xl font-semibold text-ink">Sign in required</h1>
        <p className="mt-2 text-tick normal-case text-ink-faint">
          Sign in from Settings with an admin account to see this page.
        </p>
      </div>
    );
  }

  if (!isAllowed) {
    return (
      <div className="panel border-amber/40 px-6 py-10 text-center">
        <p className="label-amber">403</p>
        <h1 className="mt-2 font-sans text-xl font-semibold text-ink">Not authorized</h1>
        <p className="mt-2 text-tick normal-case text-ink-faint">
          {session.user.email} is not on the admin list.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="font-sans text-xl font-semibold text-ink">Admin</h1>
        <p className="mt-1 text-tick text-ink-faint">Signed in as {session.user.email}.</p>
      </div>

      {error && (
        <div className="panel border-amber/40 px-4 py-2.5">
          <p className="text-tick normal-case text-amber">{error}</p>
        </div>
      )}

      {loading ? (
        <div className="panel h-32 animate-pulse-tick" />
      ) : (
        <>
          <HealthPanel health={health} onRefresh={loadHealth} isRefreshing={refreshingHealth} />
          {users && <UsersPanel users={users} onSetPlan={handleSetPlan} busy={busyUserId} />}
        </>
      )}
    </div>
  );
}
