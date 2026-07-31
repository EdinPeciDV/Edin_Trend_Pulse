/**
 * src/lib/paddle.js
 * -------------------------------------------------------------------
 * Thin wrapper around Paddle.js (Billing) for the Subscribe buttons in
 * Settings.jsx. Loaded lazily on first use — most visitors never open
 * Settings, so there's no reason to ship Paddle's SDK on every page.
 *
 * VITE_PADDLE_CLIENT_TOKEN is public by design: Paddle's client-side
 * tokens are scoped to opening a checkout, nothing more — same trust
 * model as the Supabase anon key.
 * -------------------------------------------------------------------
 */

let loadPromise = null;

function loadPaddleScript() {
  if (loadPromise) return loadPromise;
  loadPromise = new Promise((resolve, reject) => {
    if (window.Paddle) {
      resolve(window.Paddle);
      return;
    }
    const script = document.createElement('script');
    script.src = 'https://cdn.paddle.com/paddle/v2/paddle.js';
    script.async = true;
    script.onload = () => resolve(window.Paddle);
    script.onerror = () => reject(new Error('Failed to load Paddle.js — check your network/adblocker.'));
    document.head.appendChild(script);
  });
  return loadPromise;
}

export const isPaddleConfigured = Boolean(import.meta.env.VITE_PADDLE_CLIENT_TOKEN);

/**
 * Open Paddle's checkout overlay for `priceId`, tagging the transaction
 * with the signed-in user's Supabase id as custom_data — this is what
 * netlify/functions/webhook.js reads back to know whose `profiles.plan`
 * to update once the payment completes. Without a signed-in user,
 * checkout is refused outright rather than opened with no way to ever
 * apply the purchase to an account.
 */
export async function openCheckout(priceId, userId, userEmail) {
  if (!isPaddleConfigured) {
    throw new Error('Billing is not configured yet (VITE_PADDLE_CLIENT_TOKEN is not set).');
  }
  if (!priceId) {
    throw new Error('This plan is missing its Paddle price id — check VITE_PADDLE_PRICE_ID_* in .env.');
  }
  if (!userId) {
    throw new Error('Sign in before subscribing — TrendPulse needs an account to attach the plan to.');
  }

  const Paddle = await loadPaddleScript();

  if (!Paddle._trendpulseInitialised) {
    Paddle.Environment.set(import.meta.env.VITE_PADDLE_ENVIRONMENT || 'production');
    Paddle.Initialize({ token: import.meta.env.VITE_PADDLE_CLIENT_TOKEN });
    Paddle._trendpulseInitialised = true;
  }

  Paddle.Checkout.open({
    items: [{ priceId, quantity: 1 }],
    customer: userEmail ? { email: userEmail } : undefined,
    customData: { supabase_user_id: userId },
  });
}
