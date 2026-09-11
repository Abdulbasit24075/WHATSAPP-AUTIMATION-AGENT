-- ============================================================
-- 041_openrouter_provider.sql — OpenRouter as a fourth AI provider
--
-- 040_gemini_provider.sql was the previous migration.
--
-- Widens the `provider` CHECK constraints to admit 'openrouter'
-- alongside 'openai' / 'anthropic' / 'gemini'. Same shape as 040: drop
-- the constraint by name (it carries Postgres' canonical
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
    CHECK (provider IN ('openai', 'anthropic', 'gemini', 'openrouter'));

-- Same widening for the usage log — auto-reply / draft calls made on an
-- OpenRouter key would otherwise fail their INSERT here and silently
-- drop token-spend rows.
ALTER TABLE ai_usage_log
  DROP CONSTRAINT IF EXISTS ai_usage_log_provider_check;
ALTER TABLE ai_usage_log
  ADD CONSTRAINT ai_usage_log_provider_check
    CHECK (provider IN ('openai', 'anthropic', 'gemini', 'openrouter'));
