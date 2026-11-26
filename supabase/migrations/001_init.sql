-- ============================================================================
-- GlyphCraft Initial Schema & Strict Row Level Security
-- ============================================================================

create extension if not exists pgcrypto;

-- 1. Profiles (mirrored from auth.users)
create table if not exists public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  created_at timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email)
  values (new.id, new.email)
  on conflict (id) do update
    set email = excluded.email;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- 2. Orders (written by Cloudflare Workers via service_role on Creem webhook)
create table if not exists public.orders (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  creem_checkout_id text unique,
  creem_order_id text unique,
  amount_cents integer not null check (amount_cents >= 0),
  currency text not null default 'USD',
  status text not null default 'pending' check (status in ('pending', 'paid', 'refunded')),
  created_at timestamptz not null default now()
);

create index if not exists idx_orders_user_id on public.orders(user_id);

-- 3. Font Generation Jobs
create table if not exists public.font_jobs (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references public.profiles(id) on delete cascade,
  order_id uuid unique references public.orders(id) on delete set null,
  draft_id text not null,
  family_name text not null default 'CustomFont',
  status text not null default 'queued' check (status in ('queued', 'running', 'completed', 'failed')),
  progress_message text not null default 'Queued for generation',
  r2_ttf_key text,
  r2_preview_key text,
  r2_glyphs_zip_key text,
  ttf_params jsonb not null default '{"threshold":160,"sidebearing":55,"curve_mode":"spline"}'::jsonb,
  error_message text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_font_jobs_user_created on public.font_jobs(user_id, created_at desc);
create index if not exists idx_font_jobs_status on public.font_jobs(status);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_font_jobs_updated_at on public.font_jobs;
create trigger trg_font_jobs_updated_at
  before update on public.font_jobs
  for each row execute function public.set_updated_at();

-- 4. Strict Row Level Security
-- Users can ONLY read their own records. All writes/updates go through Workers (service_role).
alter table public.profiles enable row level security;
alter table public.orders enable row level security;
alter table public.font_jobs enable row level security;

drop policy if exists "Users can view own profile" on public.profiles;
create policy "Users can view own profile"
  on public.profiles for select
  to authenticated
  using (auth.uid() = id);

drop policy if exists "Users can view own orders" on public.orders;
create policy "Users can view own orders"
  on public.orders for select
  to authenticated
  using (auth.uid() = user_id);

drop policy if exists "Users can view own font jobs" on public.font_jobs;
create policy "Users can view own font jobs"
  on public.font_jobs for select
  to authenticated
  using (auth.uid() = user_id);
