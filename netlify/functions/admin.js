/**
 * netlify/functions/admin.js
 * -------------------------------------------------------------------
 * Manual plan management + system health, gated to a hard-coded admin
 * allowlist. Every request must carry a Supabase access token
 * (Authorization: Bearer <token>) for a user whose email is in
 * ADMIN_EMAILS (comma-separated, server-side only — never exposed to
 * the client). VITE_ADMIN_EMAILS is a separate, public copy used only
 * to decide whether the frontend renders the "Admin" nav link; it
 * grants nothing by itself — this function is the actual gate.
 *
 * Uses the SERVICE ROLE key for everything after the auth check, since
 * writing another user's `profiles.plan` / `is_lifetime` is exactly
 * what the DB trigger in supabase/migrations.sql restricts to the
 * service role.
 * -------------------------------------------------------------------
 */

import { createClient } from "@supabase/supabase-js";
import { INSTRUMENTS } from "../../shared/marketSources.js";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": process.env.ALLOWED_ORIGIN || "*",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Content-Type": "application/json",
};

function json(statusCode, body) {
  return { statusCode, headers: CORS_HEADERS, body: JSON.stringify(body) };
}

function getAdminClient() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

function adminEmails() {
  return (process.env.ADMIN_EMAILS || "")
    .split(",")
    .map((e) => e.trim().toLowerCase())
    .filter(Boolean);
}

/** Verifies the bearer token and that its owner is on the admin allowlist. */
async function requireAdmin(event, supabase) {
  const authHeader =
    event.headers?.authorization || event.headers?.Authorization;
  const token = authHeader?.startsWith("Bearer ") ? authHeader.slice(7) : null;
  if (!token) return null;

  const { data, error } = await supabase.auth.getUser(token);
  if (error || !data?.user?.email) return null;

  const allow = adminEmails();
  if (!allow.includes(data.user.email.toLowerCase())) return null;

  return data.user;
}

/* ---------------------------- Actions ----------------------------- */

async function listUsers(supabase) {
  const [
    { data: authData, error: authError },
    { data: profiles, error: profileError },
  ] = await Promise.all([
    supabase.auth.admin.listUsers({ perPage: 200 }),
    supabase
      .from("profiles")
      .select("id, username, plan, is_lifetime, risk_tolerance, created_at"),
  ]);

  if (authError)
    throw new Error(`auth.admin.listUsers failed: ${authError.message}`);
  if (profileError)
    throw new Error(`profiles query failed: ${profileError.message}`);

  const profileById = new Map((profiles || []).map((p) => [p.id, p]));

  return authData.users.map((u) => {
    const p = profileById.get(u.id);
    return {
      id: u.id,
      email: u.email,
      createdAt: u.created_at,
      lastSignInAt: u.last_sign_in_at,
      username: p?.username ?? null,
      plan: p?.plan ?? "free",
      isLifetime: p?.is_lifetime ?? false,
      riskTolerance: p?.risk_tolerance ?? "conservative",
    };
  });
}

async function setPlan(supabase, { userId, plan, isLifetime }) {
  if (!userId || !["free", "basic", "pro"].includes(plan)) {
    throw new Error("userId and a valid plan (free|basic|pro) are required.");
  }
  const { data, error } = await supabase
    .from("profiles")
    .update({
      plan,
      is_lifetime: Boolean(isLifetime),
      updated_at: new Date().toISOString(),
    })
    .eq("id", userId)
    .select()
    .maybeSingle();
  if (error) throw new Error(error.message);
  return data;
}

// Presence-only check — never returns the values, just whether Netlify
// has them configured. A missing one names exactly which feature is
// broken and where to look (matches the hints already in
// shared/marketSources.js and netlify/functions/webhook.js).
const REQUIRED_ENV = [
  "SUPABASE_URL",
  "SUPABASE_SERVICE_ROLE_KEY",
  "TWELVE_DATA_API_KEY",
  "PADDLE_WEBHOOK_SECRET",
  "PADDLE_PRICE_ID_BASIC",
  "PADDLE_PRICE_ID_PRO",
  "ADMIN_EMAILS",
];

async function health(supabase) {
  const env = Object.fromEntries(
    REQUIRED_ENV.map((k) => [k, Boolean(process.env[k])]),
  );

  const { data: latest, error: latestError } = await supabase
    .from("latest_snapshots")
    .select("symbol, bucket_at")
    .order("bucket_at", { ascending: false });

  const now = Date.now();
  const staleness = (latest || []).map((row) => ({
    symbol: row.symbol,
    bucketAt: row.bucket_at,
    ageMinutes: Math.round((now - new Date(row.bucket_at).getTime()) / 60000),
  }));

  const missingSymbols = INSTRUMENTS.map((i) => i.symbol).filter(
    (s) => !staleness.some((row) => row.symbol === s),
  );

  // forex_candle_cache is a separate table from market_snapshots (see
  // migrations.sql section 8) — get-analysis.js's read path depends on
  // it, but fetch-market-data.js writes to it silently and only logs on
  // failure, so a missing table or stale cache never shows up in
  // snapshotStaleness above even though ingest looks healthy. Checked
  // explicitly here so that failure mode is visible without digging
  // through function logs.
  const forexSymbols = INSTRUMENTS.filter((i) => i.kind === "forex").map(
    (i) => i.symbol,
  );
  const { data: forexCache, error: forexCacheError } = await supabase
    .from("forex_candle_cache")
    .select("symbol, updated_at");

  const forexCacheStaleness = (forexCache || []).map((row) => ({
    symbol: row.symbol,
    updatedAt: row.updated_at,
    ageMinutes: Math.round((now - new Date(row.updated_at).getTime()) / 60000),
  }));
  const forexCacheMissing = forexSymbols.filter(
    (s) => !forexCacheStaleness.some((row) => row.symbol === s),
  );

  return {
    env,
    instrumentsRegistered: INSTRUMENTS.map((i) => i.symbol),
    snapshotStaleness: staleness,
    missingSymbols,
    snapshotError: latestError?.message ?? null,
    forexCache: {
      // PGRST205 = table doesn't exist yet — the most common cause of
      // "no cached forex candles" on the dashboard despite ingest
      // otherwise looking fine. See supabase/migrations.sql section 8.
      tableExists: !forexCacheError,
      error: forexCacheError?.message ?? null,
      staleness: forexCacheStaleness,
      missingSymbols: forexCacheMissing,
    },
  };
}

/* ---------------------------- Handler ------------------------------ */

export async function handler(event) {
  if (event.httpMethod === "OPTIONS") {
    return { statusCode: 204, headers: CORS_HEADERS, body: "" };
  }

  const supabase = getAdminClient();
  if (!supabase) {
    return json(500, { error: "Storage not configured." });
  }

  const caller = await requireAdmin(event, supabase);
  if (!caller) {
    return json(403, { error: "Not authorized." });
  }

  try {
    if (event.httpMethod === "GET") {
      const action = event.queryStringParameters?.action;
      if (action === "health")
        return json(200, { ok: true, health: await health(supabase) });
      if (action === "list-users")
        return json(200, { ok: true, users: await listUsers(supabase) });
      return json(400, { error: `Unknown action "${action}".` });
    }

    if (event.httpMethod === "POST") {
      const body = JSON.parse(event.body || "{}");
      if (body.action === "set-plan") {
        const profile = await setPlan(supabase, body);
        console.log(
          `[admin] ${caller.email} set ${body.userId} -> plan '${body.plan}' (lifetime: ${Boolean(body.isLifetime)})`,
        );
        return json(200, { ok: true, profile });
      }
      return json(400, { error: `Unknown action "${body.action}".` });
    }

    return json(405, { error: "Method not allowed." });
  } catch (err) {
    console.error("[admin] error:", err.message);
    return json(500, { error: err.message });
  }
}
