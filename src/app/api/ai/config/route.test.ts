import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { SupabaseClient } from '@supabase/supabase-js'
import { encrypt, decrypt } from '@/lib/whatsapp/encryption'

// Real encrypt()/decrypt() are used (vitest.config supplies a dummy
// ENCRYPTION_KEY) so the round-trip proves keys land encrypted.
const h = vi.hoisted(() => ({
  validateCalls: [] as Array<Record<string, unknown>>,
  db: null as unknown,
}))

vi.mock('next/server', () => ({
  NextResponse: {
    json: (body: unknown, init?: { status?: number }) => ({
      body,
      status: init?.status ?? 200,
    }),
  },
}))

vi.mock('@/lib/auth/account', () => ({
  requireRole: async () => ({
    supabase: h.db,
    accountId: 'acct-1',
    userId: 'user-1',
  }),
  getCurrentAccount: async () => ({ supabase: h.db, accountId: 'acct-1' }),
  toErrorResponse: (err: unknown) => ({
    body: { error: String(err) },
    status: 500,
  }),
}))

vi.mock('@/lib/rate-limit', () => ({
  checkRateLimit: () => ({ success: true }),
  rateLimitResponse: () => ({ body: { error: 'rate limited' }, status: 429 }),
  RATE_LIMITS: {
    adminAction: { name: 'admin_action', limit: 10, windowMs: 60_000 },
  },
}))

vi.mock('@/lib/ai/validate', () => ({
  validateAiCredentials: async (config: Record<string, unknown>) => {
    h.validateCalls.push(config)
  },
}))

vi.mock('@/lib/ai/embeddings', () => ({
  embedTexts: async () => [[0.01]],
}))

import { POST } from './route'

// The route handlers declare Response, but our next/server mock returns
// plain { body, status } objects — pin that here once.
type MockJson = { body: Record<string, unknown>; status: number }

function fakeDb(existingRow: Record<string, unknown> | null = null) {
  const calls = {
    updates: [] as Record<string, unknown>[],
    inserts: [] as Record<string, unknown>[],
  }
  const db = {
    calls,
    from(table: string) {
      if (table === 'profiles') {
        return {
          select: () => ({
            eq: () => ({
              eq: () => ({
                maybeSingle: async () => ({
                  data: { user_id: 'agent-1' },
                  error: null,
                }),
              }),
            }),
          }),
        }
      }
      if (table !== 'ai_configs') throw new Error(`unexpected table ${table}`)
      return {
        select: () => ({
          eq: () => ({
            maybeSingle: async () => ({ data: existingRow, error: null }),
          }),
        }),
        update: (payload: Record<string, unknown>) => {
          calls.updates.push(payload)
          return { eq: async () => ({ error: null }) }
        },
        insert: (payload: Record<string, unknown>) => {
          calls.inserts.push(payload)
          return Promise.resolve({ error: null })
        },
      }
    },
  }
  return db as unknown as SupabaseClient & { calls: typeof calls }
}

function post(body: Record<string, unknown>): Promise<MockJson> {
  return POST({
    json: async () => body,
  } as unknown as Request) as unknown as Promise<MockJson>
}

beforeEach(() => {
  h.validateCalls.length = 0
})

describe('POST /api/ai/config — validation & save', () => {
  it('rejects an unsupported provider, listing every accepted one', async () => {
    const res = await post({ provider: 'azure', model: 'x', api_key: 'k' })
    expect(res.status).toBe(400)
    expect(res.body.error).toContain('openrouter')
    expect(res.body.error).toContain('gemini')
    expect(res.body.error).toContain('groq')
  })

  it('validates then saves a new OpenRouter config with an encrypted key', async () => {
    const db = fakeDb(null)
    h.db = db

    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
      api_key: 'sk-or-v1-fake-not-a-real-key',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ success: true })
    // Validate-before-save, exactly like OpenAI/Anthropic/Gemini saves.
    expect(h.validateCalls).toHaveLength(1)
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
      apiKey: 'sk-or-v1-fake-not-a-real-key',
    })
    // Stored encrypted — ciphertext round-trips, plaintext never lands.
    expect(db.calls.inserts).toHaveLength(1)
    const row = db.calls.inserts[0]
    expect(row.provider).toBe('openrouter')
    expect(row.api_key).not.toBe('sk-or-v1-fake-not-a-real-key')
    expect(decrypt(row.api_key as string)).toBe('sk-or-v1-fake-not-a-real-key')
  })

  it('reuses the stored key when none is sent and leaves it untouched on update', async () => {
    const db = fakeDb({
      id: 'cfg-1',
      provider: 'anthropic',
      model: 'claude-haiku-4-5-20251001',
      api_key: encrypt('sk-or-v1-already-stored'),
    })
    h.db = db

    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
    })

    expect(res.status).toBe(200)
    // Provider changed → re-validated against OpenRouter using the
    // decrypted stored key.
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'openrouter',
      apiKey: 'sk-or-v1-already-stored',
    })
    expect(db.calls.updates).toHaveLength(1)
    expect(db.calls.updates[0].provider).toBe('openrouter')
    expect(db.calls.updates[0]).not.toHaveProperty('api_key')
  })

  it('requires an api_key when there is no stored one', async () => {
    h.db = fakeDb(null)
    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
    })
    expect(res.status).toBe(400)
    expect(res.body.error).toBe('api_key is required')
    expect(h.validateCalls).toHaveLength(0)
  })
})

describe('POST /api/ai/config — Groq save path', () => {
  it('validates then saves a new Groq config with an encrypted key and verified default model', async () => {
    const db = fakeDb(null)
    h.db = db

    const res = await post({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
      api_key: 'gsk_fake-placeholder-key',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ success: true })
    expect(h.validateCalls).toHaveLength(1)
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
      apiKey: 'gsk_fake-placeholder-key',
    })
    // Encrypted at rest; plaintext never lands in the DB payload.
    const row = db.calls.inserts[0]
    expect(row.provider).toBe('groq')
    expect(row.api_key).not.toBe('gsk_fake-placeholder-key')
    expect(decrypt(row.api_key as string)).toBe('gsk_fake-placeholder-key')
  })

  it('re-validates when switching from OpenRouter to Groq using the stored key', async () => {
    h.db = fakeDb({
      id: 'cfg-1',
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
      api_key: encrypt('gsk_switched-provider'),
    })

    const res = await post({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
    })

    expect(res.status).toBe(200)
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'groq',
      apiKey: 'gsk_switched-provider',
    })
  })
})