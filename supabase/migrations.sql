-- ===================================================================
-- TrendPulse — Supabase schema
-- ===================================================================
-- Run this once in the Supabase dashboard: SQL Editor › New query ›
-- paste › Run. It is idempotent, so re-running is safe.
--
-- Design notes:
--   * market_snapshots is PUBLIC READ (it's market data, not user data)
--     but writable only by the service role.
--   * profiles and watchlists are per-user and locked down with RLS.
--   * predictions hold both system rows (user_id IS NULL, world-readable
--     so the accuracy table works for anonymous visitors) and per-user
--     rows (readable only by their owner).
-- ===================================================================

-- Needed for gen_random_uuid()
create extension if not exists "pgcrypto";

-- ===================================================================
-- 1. profiles — extends auth.users
-- ===================================================================

create table if not exists public.profiles (
  id            uuid primary key references auth.users (id) on delete cascade,
  username      text unique,
  risk_tolerance text not null default 'conservative'
                 check (risk_tolerance in ('aggressive', 'conservative')),
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now()
);

comment on table public.profiles is
  'User preferences. risk_tolerance drives the Aggressive/Conservative strategy toggle.';

alter table public.profiles enable row level security;

drop policy if exists "profiles: read own" on public.profiles;
create policy "profiles: read own"
  on public.profiles for select
  using (auth.uid() = id);

drop policy if exists "profiles: insert own" on public.profiles;
create policy "profiles: insert own"
  on public.profiles for insert
  with check (auth.uid() = id);

drop policy if exists "profiles: update own" on public.profiles;
create policy "profiles: update own"
  on public.profiles for update
  using (auth.uid() = id)
  with check (auth.uid() = id);

-- Auto-create a profile row whenever a user signs up.
create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, username)
  values (
    new.id,
    coalesce(
      new.raw_user_meta_data ->> 'user_name',   -- GitHub OAuth
      split_part(new.email, '@', 1)             -- magic link
    )
  )
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ===================================================================
-- 2. market_snapshots — the 5-minute time series
-- ===================================================================

create table if not exists public.market_snapshots (
  id                        bigserial primary key,
  symbol                    text not null,
  asset_class               text check (asset_class in ('crypto', 'forex')),
  price                     double precision not null,
  rsi                       double precision,
  sma_50                    double precision,
  bb_upper                  double precision,
  bb_lower                  double precision,
  bb_bandwidth              double precision,
  bb_breakout               text check (bb_breakout in ('UPPER', 'LOWER')),
  vwap                      double precision,
  vwap_is_volume_weighted   boolean default false,
  volatility                double precision,
  volume                    double precision,
  signal                    text not null default 'NEUTRAL'
                            check (signal in ('UP', 'DOWN', 'NEUTRAL')),
  confidence                integer not null default 0
                            check (confidence between 0 and 100),
  rule                      text,
  reasons                   jsonb,
  -- Timestamp rounded down to the 5-minute boundary. The upsert target.
  bucket_at                 timestamptz not null,
  created_at                timestamptz not null default now()
);

comment on column public.market_snapshots.bucket_at is
  'Ingest time floored to a 5-minute boundary. Upsert key, so a re-run inside the same window updates rather than duplicates.';
comment on column public.market_snapshots.vwap_is_volume_weighted is
  'False for spot FX, which has no consolidated volume — the value is then an unweighted mean of typical prices.';

-- The upsert conflict target used by fetch-market-data.js
create unique index if not exists market_snapshots_symbol_bucket_key
  on public.market_snapshots (symbol, bucket_at);

create index if not exists market_snapshots_symbol_created_idx
  on public.market_snapshots (symbol, created_at desc);

create index if not exists market_snapshots_created_idx
  on public.market_snapshots (created_at desc);

alter table public.market_snapshots enable row level security;

-- Market data is not user data — anyone may read it.
drop policy if exists "snapshots: public read" on public.market_snapshots;
create policy "snapshots: public read"
  on public.market_snapshots for select
  using (true);

-- No insert/update policy exists, so only the service role (which
-- bypasses RLS) can write. That's exactly what we want: writes come
-- only from the serverless functions.

-- ===================================================================
-- 3. predictions — the accuracy log
-- ===================================================================

create table if not exists public.predictions (
  id                    uuid primary key default gen_random_uuid(),
  user_id               uuid references auth.users (id) on delete cascade,
  symbol                text not null,
  predicted_direction   text not null
                        check (predicted_direction in ('UP', 'DOWN', 'NEUTRAL')),
  confidence            integer check (confidence between 0 and 100),
  price_at_prediction   double precision,
  rsi_at_prediction     double precision,
  actual_result         text not null default 'PENDING'
                        check (actual_result in ('PENDING', 'CORRECT', 'INCORRECT')),
  price_at_resolution   double precision,
  move_pct              double precision,
  resolved_at           timestamptz,
  created_at            timestamptz not null default now()
);

comment on table public.predictions is
  'Every signal we emit, later graded against what actually happened. user_id IS NULL means a system-wide prediction.';

create index if not exists predictions_symbol_created_idx
  on public.predictions (symbol, created_at desc);

create index if not exists predictions_pending_idx
  on public.predictions (actual_result, created_at)
  where actual_result = 'PENDING';

create index if not exists predictions_user_idx
  on public.predictions (user_id, created_at desc);

alter table public.predictions enable row level security;

-- System predictions are public (they power the trust/accuracy table);
-- user predictions are private to their owner.
drop policy if exists "predictions: read system or own" on public.predictions;
create policy "predictions: read system or own"
  on public.predictions for select
  using (user_id is null or auth.uid() = user_id);

drop policy if exists "predictions: insert own" on public.predictions;
create policy "predictions: insert own"
  on public.predictions for insert
  with check (auth.uid() = user_id);

-- ===================================================================
-- 4. watchlists — per-user followed symbols
-- ===================================================================

create table if not exists public.watchlists (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid not null references auth.users (id) on delete cascade,
  symbol      text not null,
  created_at  timestamptz not null default now(),
  unique (user_id, symbol)
);

alter table public.watchlists enable row level security;

drop policy if exists "watchlists: read own" on public.watchlists;
create policy "watchlists: read own"
  on public.watchlists for select
  using (auth.uid() = user_id);

drop policy if exists "watchlists: insert own" on public.watchlists;
create policy "watchlists: insert own"
  on public.watchlists for insert
  with check (auth.uid() = user_id);

drop policy if exists "watchlists: delete own" on public.watchlists;
create policy "watchlists: delete own"
  on public.watchlists for delete
  using (auth.uid() = user_id);

-- ===================================================================
-- 5. Convenience view: latest snapshot per symbol
-- ===================================================================

create or replace view public.latest_snapshots as
select distinct on (symbol) *
from public.market_snapshots
order by symbol, bucket_at desc;

comment on view public.latest_snapshots is
  'Most recent snapshot per symbol. Inherits RLS from market_snapshots.';

-- ===================================================================
-- 6. Accuracy rollup
-- ===================================================================

create or replace view public.prediction_accuracy as
select
  symbol,
  predicted_direction,
  count(*)                                                          as total,
  count(*) filter (where actual_result = 'CORRECT')                 as correct,
  round(
    100.0 * count(*) filter (where actual_result = 'CORRECT')
    / nullif(count(*) filter (where actual_result <> 'PENDING'), 0)
  , 1)                                                              as accuracy_pct
from public.predictions
where user_id is null
group by symbol, predicted_direction;

-- ===================================================================
-- 7. Retention: keep the snapshot table from growing without bound
-- ===================================================================
-- Every 5 minutes x 5 symbols is ~1,440 rows/day. Fine for months, but
-- this helper lets you prune. Call it from schedule-task.js, or enable
-- pg_cron in Supabase (Database › Extensions) and schedule it:
--
--   select cron.schedule('prune-snapshots', '0 3 * * *',
--     $$ select public.prune_old_snapshots(30) $$);

create or replace function public.prune_old_snapshots(keep_days integer default 30)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  removed integer;
begin
  delete from public.market_snapshots
  where created_at < now() - (keep_days || ' days')::interval;
  get diagnostics removed = row_count;
  return removed;
end;
$$;

-- ===================================================================
-- Done. Verify with:
--   select table_name from information_schema.tables
--   where table_schema = 'public';
-- ===================================================================
