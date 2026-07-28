/**
 * src/lib/supabaseClient.js
 * -------------------------------------------------------------------
 * Browser-side Supabase client.
 *
 * Uses the ANON key only. That key is designed to be public — it grants
 * nothing beyond what the RLS policies in supabase/migrations.sql allow.
 * The SERVICE ROLE key must never appear in this file or anywhere under
 * src/, because everything here is compiled into the public bundle.
 *
 * Auth is optional: with no credentials configured the app still runs
 * fully in read-only mode, because market data is public.
 * -------------------------------------------------------------------
 */

import { createClient } from '@supabase/supabase-js';

const url = import.meta.env.VITE_SUPABASE_URL;
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY;

export const isSupabaseConfigured = Boolean(url && anonKey);

if (!isSupabaseConfigured && import.meta.env.DEV) {
  console.warn(
    '[TrendPulse] VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY are not set. ' +
      'Sign-in and saved preferences are disabled; market data still works.'
  );
}

export const supabase = isSupabaseConfigured
  ? createClient(url, anonKey, {
      auth: {
        persistSession: true,
        autoRefreshToken: true,
        detectSessionInUrl: true, // needed for magic-link + OAuth returns
      },
    })
  : null;

/* ------------------------------------------------------------------ */
/* Auth                                                                */
/* ------------------------------------------------------------------ */

export async function signInWithEmail(email) {
  if (!supabase) throw new Error('Sign-in is unavailable: Supabase is not configured.');
  const { error } = await supabase.auth.signInWithOtp({
    email,
    options: { emailRedirectTo: window.location.origin },
  });
  if (error) throw error;
  return { sent: true };
}

export async function signInWithGitHub() {
  if (!supabase) throw new Error('Sign-in is unavailable: Supabase is not configured.');
  const { error } = await supabase.auth.signInWithOAuth({
    provider: 'github',
    options: { redirectTo: window.location.origin },
  });
  if (error) throw error;
}

export async function signOut() {
  if (!supabase) return;
  await supabase.auth.signOut();
}

/* ------------------------------------------------------------------ */
/* Profile + watchlist                                                 */
/* ------------------------------------------------------------------ */

export async function getProfile(userId) {
  if (!supabase || !userId) return null;
  const { data, error } = await supabase
    .from('profiles')
    .select('id, username, risk_tolerance')
    .eq('id', userId)
    .maybeSingle();
  if (error) {
    console.error('[TrendPulse] profile read failed:', error.message);
    return null;
  }
  return data;
}

export async function saveRiskTolerance(userId, riskTolerance) {
  if (!supabase || !userId) return null;
  const { data, error } = await supabase
    .from('profiles')
    .upsert(
      { id: userId, risk_tolerance: riskTolerance, updated_at: new Date().toISOString() },
      { onConflict: 'id' }
    )
    .select()
    .maybeSingle();
  if (error) throw error;
  return data;
}

export async function getWatchlist(userId) {
  if (!supabase || !userId) return [];
  const { data, error } = await supabase
    .from('watchlists')
    .select('symbol')
    .eq('user_id', userId);
  if (error) {
    console.error('[TrendPulse] watchlist read failed:', error.message);
    return [];
  }
  return (data || []).map((r) => r.symbol);
}

export async function toggleWatchlist(userId, symbol, isWatched) {
  if (!supabase || !userId) return;
  if (isWatched) {
    const { error } = await supabase
      .from('watchlists')
      .delete()
      .eq('user_id', userId)
      .eq('symbol', symbol);
    if (error) throw error;
  } else {
    const { error } = await supabase
      .from('watchlists')
      .insert({ user_id: userId, symbol });
    if (error) throw error;
  }
}
