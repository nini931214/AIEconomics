-- 請在自己的 Supabase SQL Editor 執行；不需要把 service_role key 放到 GitHub。
create table if not exists public.ai_analyses (
  id bigint generated always as identity primary key,
  created_at timestamptz not null default now(),
  model text not null,
  signature text not null unique,
  sentiment_score integer not null check (sentiment_score between 0 and 100),
  hawkish_score integer not null check (hawkish_score between 0 and 100),
  payload jsonb not null
);
create index if not exists ai_analyses_created_idx on public.ai_analyses (created_at desc);
alter table public.ai_analyses enable row level security;
-- 不開放匿名或登入者直接讀寫：僅伺服器端 service_role 可操作。
