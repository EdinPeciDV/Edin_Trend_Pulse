/**
 * netlify/functions/webhook.js
 * -------------------------------------------------------------------
 * Paddle Billing webhook — the ONLY place `profiles.plan` changes.
 * (supabase/migrations.sql additionally enforces this at the database
 * level: a trigger silently reverts any plan change that doesn't come
 * from the service role, so even a bug here can't let a client set its
 * own plan.)
 *
 * Every request is signature-verified before anything in it is
 * trusted — an unverified or malformed request is rejected with
 * 401/400 and never reaches Supabase.
 *
 * LINKING A PAYMENT BACK TO A TRENDPULSE ACCOUNT: relies on
 * `custom_data.supabase_user_id`, attached when the Subscribe button
 * opens Paddle's checkout overlay (src/lib/paddle.js). Paddle echoes
 * custom_data back on every webhook event for that subscription's
 * whole lifecycle. A webhook that arrives without it is logged and
 * skipped rather than guessed at (e.g. by matching on email) — a guess
 * that updates the wrong user's plan is worse than a missed update
 * someone can fix by hand from the Paddle dashboard.
 *
 * Uses the SERVICE ROLE key, so it bypasses RLS — same as every other
 * function in netlify/functions/ that writes to Supabase.
 *
 * LIFETIME ACCOUNTS: profiles.is_lifetime is set by hand from /admin
 * (netlify/functions/admin.js), never by this webhook. A cancel/pause
 * event here must not downgrade one of those accounts, so downgrades
 * check the flag first and skip if it's set.
 * -------------------------------------------------------------------
 */

import crypto from "node:crypto";
import { createClient } from "@supabase/supabase-js";

function json(statusCode, body) {
  return {
    statusCode,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  };
}

function getAdminClient() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return createClient(url, key, {
    auth: { persistSession: false, autoRefreshToken: false },
  });
}

/* ---------------------- Signature verification ------------------- */

/**
 * Paddle Billing signs every webhook with a `Paddle-Signature` header:
 *   ts=<unix-seconds>;h1=<hex hmac-sha256 of "ts:rawBody">
 * Verified with the notification destination's secret (Paddle dashboard
 * -> Developer Tools -> Notifications -> your destination -> secret
 * key, starts with `pdl_ntfset_`). MUST be computed over the exact raw
 * request body bytes — JSON.parse then re-stringify can reorder keys
 * or change whitespace and silently break every signature.
 */
function verifyPaddleSignature(rawBody, signatureHeader, secret) {
  if (!signatureHeader || !secret) return false;

  const parts = Object.fromEntries(
    signatureHeader.split(";").map((p) => p.split("=")),
  );
  const { ts, h1 } = parts;
  if (!ts || !h1) return false;

  // Reject stale signatures — bounds how long a captured request could
  // be replayed. 5 minutes is generous relative to normal delivery
  // latency; widen it if Paddle's retry behaviour ever needs it.
  const ageSeconds = Math.abs(Date.now() / 1000 - Number(ts));
  if (!Number.isFinite(ageSeconds) || ageSeconds > 300) return false;

  const expected = crypto
    .createHmac("sha256", secret)
    .update(`${ts}:${rawBody}`)
    .digest("hex");

  const a = Buffer.from(expected, "hex");
  const b = Buffer.from(h1, "hex");
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}

/* ---------------------------- Plan mapping ------------------------- */

function priceToPlan(priceId) {
  if (priceId && priceId === process.env.PADDLE_PRICE_ID_BASIC) return "basic";
  if (priceId && priceId === process.env.PADDLE_PRICE_ID_PRO) return "pro";
  return null;
}

// Subscriptions can in principle carry multiple items; TrendPulse's
// checkout only ever adds one (see openCheckout() in src/lib/paddle.js),
// so the first item's price is authoritative.
function planForSubscription(data) {
  return priceToPlan(data?.items?.[0]?.price?.id);
}

const UPGRADE_EVENTS = new Set([
  "subscription.created",
  "subscription.activated",
  "subscription.updated",
]);
const DOWNGRADE_EVENTS = new Set([
  "subscription.canceled",
  "subscription.paused",
]);

/* ------------------------------ Handler ---------------------------- */

export async function handler(event) {
  if (event.httpMethod !== "POST") {
    return json(405, { error: "Method not allowed. Paddle sends POST." });
  }

  const secret = process.env.PADDLE_WEBHOOK_SECRET;
  const rawBody = event.isBase64Encoded
    ? Buffer.from(event.body || "", "base64").toString("utf8")
    : event.body || "";

  const signatureHeader =
    event.headers?.["paddle-signature"] || event.headers?.["Paddle-Signature"];
  if (!verifyPaddleSignature(rawBody, signatureHeader, secret)) {
    console.error("[webhook] signature verification failed");
    return json(401, { error: "Invalid signature." });
  }

  let payload;
  try {
    payload = JSON.parse(rawBody);
  } catch {
    return json(400, { error: "Malformed JSON body." });
  }

  const eventType = payload.event_type;
  const data = payload.data || {};
  const userId = data.custom_data?.supabase_user_id;

  if (!userId) {
    console.warn(
      `[webhook] ${eventType}: no custom_data.supabase_user_id — skipped`,
      {
        subscription_id: data.id,
      },
    );
    // 200, not an error: Paddle retries non-2xx responses, and retrying
    // will never make a missing user id appear.
    return json(200, {
      ok: true,
      skipped: "no supabase_user_id in custom_data",
    });
  }

  let newPlan;
  if (UPGRADE_EVENTS.has(eventType)) {
    newPlan = planForSubscription(data);
    if (!newPlan) {
      console.warn(`[webhook] ${eventType}: unrecognised price id — skipped`, {
        price_id: data.items?.[0]?.price?.id,
        user_id: userId,
      });
      return json(200, { ok: true, skipped: "unrecognised price id" });
    }
  } else if (DOWNGRADE_EVENTS.has(eventType)) {
    newPlan = "free";
  } else {
    // transaction.*, customer.*, and everything else: acknowledged, but
    // plan changes only ever come from the subscription lifecycle
    // events above.
    return json(200, { ok: true, ignored: eventType });
  }

  const supabase = getAdminClient();
  if (!supabase) {
    console.error(
      "[webhook] SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY not configured",
    );
    return json(500, { error: "Storage not configured." });
  }

  // Lifetime accounts are granted by hand from /admin, never through
  // Paddle — a cancel/pause on some unrelated subscription must never
  // knock them back to free.
  if (DOWNGRADE_EVENTS.has(eventType)) {
    const { data: profile, error: profileError } = await supabase
      .from("profiles")
      .select("is_lifetime")
      .eq("id", userId)
      .maybeSingle();
    if (profileError) {
      console.error(
        `[webhook] lifetime check failed for ${userId}:`,
        profileError.message,
      );
      return json(500, { error: profileError.message });
    }
    if (profile?.is_lifetime) {
      console.log(
        `[webhook] ${eventType}: user ${userId} is lifetime — downgrade skipped`,
      );
      return json(200, { ok: true, skipped: "lifetime account" });
    }
  }

  const { error } = await supabase
    .from("profiles")
    .update({ plan: newPlan, updated_at: new Date().toISOString() })
    .eq("id", userId);

  if (error) {
    console.error(`[webhook] plan update failed for ${userId}:`, error.message);
    return json(500, { error: error.message });
  }

  console.log(`[webhook] ${eventType}: user ${userId} -> plan '${newPlan}'`);
  return json(200, { ok: true, userId, plan: newPlan });
}
