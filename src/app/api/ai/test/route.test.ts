import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { SupabaseClient } from '@supabase/supabase-js'
import { encrypt } from '@/lib/whatsapp/encryption'

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

import { POST } from './route'

// The route handlers declare Response, but our next/server mock returns
// plain { body, status } objects — pin that here once.
type MockJson = { body: Record<string, unknown>; status: number }

function fakeDb(existingRow: Record<string, unknown> | null) {
  const calls = { reads: 0 }
  const db = {
    calls,
    from(table: string) {
      calls.reads += 1
      if (table !== 'ai_configs') throw new Error(`unexpected table ${table}`)
      return {
        select: () => ({
          eq: () => ({
            maybeSingle: async () => ({ data: existingRow, error: null }),
          }),
        }),
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

describe('POST /api/ai/test — "Test key" behavior', () => {
  it('rejects an unsupported provider before touching anything', async () => {
    const res = await post({ provider: 'mystery', model: 'm', api_key: 'k' })
    expect(res.status).toBe(400)
    expect(res.body.error).toContain('openrouter')
    expect(res.body.error).toContain('groq')
    expect(h.validateCalls).toHaveLength(0)
  })

  it('tests an explicitly-sent OpenRouter key without any DB access', async () => {
    const db = fakeDb(null)
    h.db = db

    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
      api_key: 'sk-or-v1-candidate',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
    expect(db.calls.reads).toBe(0)
    expect(h.validateCalls).toHaveLength(1)
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
      apiKey: 'sk-or-v1-candidate',
    })
  })

  it('falls back to the stored encrypted key when api_key is omitted', async () => {
    h.db = fakeDb({ api_key: encrypt('sk-or-v1-stored') })

    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'openrouter',
      apiKey: 'sk-or-v1-stored',
    })
  })

  it('asks for a key when none was sent and none is stored', async () => {
    h.db = fakeDb(null)

    const res = await post({
      provider: 'openrouter',
      model: 'google/gemma-4-26b-a4b-it:free',
    })

    expect(res.status).toBe(400)
    expect(res.body.error).toBe('Enter an API key to test.')
    expect(h.validateCalls).toHaveLength(0)
  })
})

describe('POST /api/ai/test — Groq "Test key"', () => {
  it('tests an explicitly-sent Groq key against the verified default model', async () => {
    const db = fakeDb(null)
    h.db = db

    const res = await post({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
      api_key: 'gsk_candidate-not-real',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
    // Minimal request, no DB access, no WhatsApp involvement.
    expect(db.calls.reads).toBe(0)
    expect(h.validateCalls).toHaveLength(1)
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
      apiKey: 'gsk_candidate-not-real',
    })
  })

  it('falls back to the stored encrypted Groq key when api_key is omitted', async () => {
    h.db = fakeDb({ api_key: encrypt('gsk_stored-encrypted') })

    const res = await post({
      provider: 'groq',
      model: 'openai/gpt-oss-20b',
    })

    expect(res.status).toBe(200)
    expect(res.body).toEqual({ ok: true })
    expect(h.validateCalls[0]).toMatchObject({
      provider: 'groq',
      apiKey: 'gsk_stored-encrypted',
    })
  })
})