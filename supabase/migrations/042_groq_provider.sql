-- ============================================================
-- 042_groq_provider.sql — Groq as a fifth AI provider
--
-- 041_openrouter_provider.sql was the previous migration.
--
-- Widens the `provider` CHECK constraints to admit 'groq' alongside
-- 'openai' / 'anthropic' / 'gemini' / 'openrouter'. Same shape as 040 /
-- 041: drop the constraint by name (it carries Postgres' canonical
-- `<table>_<column>_check` name) and re-add with the full list.
-- Existing rows only hold earlier provider values, so revalidating is
-- instant. No column, type, or RLS change — keys stay AES-256-GCM
-- encrypted exactly as before regardless of provider.
--
-- Idempotent — safe to run multiple times.
-- ============================================================

ALTER TABLE ai_configs
  DROP CONSTRAINT IF EXISTS ai_configs_provider_check;
ALTER TABLE ai_configs
  ADD CONSTRAINT ai_configs_provider_check
    CHECK (provider IN ('openai', 'anthropic', 'gemini', 'openrouter', 'groq'));

-- Same widening for the usage log — auto-reply / draft calls made on a
-- Groq key would otherwise fail their INSERT here and silently drop
-- token-spend rows.
ALTER TABLE ai_usage_log
  DROP CONSTRAINT IF EXISTS ai_usage_log_provider_check;
ALTER TABLE ai_usage_log
  ADD CONSTRAINT ai_usage_log_provider_check
    CHECK (provider IN ('openai', 'anthropic', 'gemini', 'openrouter', 'groq'));
