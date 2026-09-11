-- ============================================================
-- 040_gemini_provider.sql — Google Gemini as a third AI provider
--
-- 039_inbound_media_mirror.sql was the previous migration.
--
-- Widens the `provider` CHECK constraints to admit 'gemini' alongside
-- 'openai' / 'anthropic'. Both constraints were created inline (so they
-- carry Postgres' canonical `<table>_<column>_check` names); dropping
-- them by name and re-adding with an explicit name keeps this runnable
-- whether or not 029 / 033 named theirs differently. Existing rows all
-- hold 'openai' or 'anthropic', so revalidating is instant. No column,
-- type, or RLS change — the app layer already writes `provider` as plain
-- text (see src/lib/ai/config.ts), and chat keys stay AES-256-GCM
-- encrypted exactly as before regardless of provider.
--
-- Idempotent — safe to run multiple times.
-- ============================================================

ALTER TABLE ai_configs
  DROP CONSTRAINT IF EXISTS ai_configs_provider_check;
ALTER TABLE ai_configs
  ADD CONSTRAINT ai_configs_provider_check
    CHECK (provider IN ('openai', 'anthropic', 'gemini'));

-- The usage log gets the same widening — auto-reply / draft calls made
-- on a Gemini key would otherwise fail their INSERT here and silently
-- drop token-spend rows.
ALTER TABLE ai_usage_log
  DROP CONSTRAINT IF EXISTS ai_usage_log_provider_check;
ALTER TABLE ai_usage_log
  ADD CONSTRAINT ai_usage_log_provider_check
    CHECK (provider IN ('openai', 'anthropic', 'gemini'));
