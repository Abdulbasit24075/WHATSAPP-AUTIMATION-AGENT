import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { generateReply, parseGeneration } from './generate'
import { AiError, type AiConfig, type AiProvider } from './types'
import { AI_PROVIDER_DEFAULT_MODEL, MAX_OUTPUT_TOKENS } from './defaults'

function config(overrides: Partial<AiConfig> = {}): AiConfig {
  return {
    provider: 'openai',
    model: 'gpt-test',
    apiKey: 'sk-test',
    systemPrompt: null,
    isActive: true,
    autoReplyEnabled: false,
    autoReplyMaxPerConversation: 3,
    handoffAgentId: null,
    embeddingsApiKey: null,
    ...overrides,
  }
}

function okResponse(json: unknown): Response {
  return {
    ok: true,
    status: 200,
    json: async () => json,
  } as unknown as Response
}

function errResponse(status: number, json: unknown): Response {
  return {
    ok: false,
    status,
    json: async () => json,
  } as unknown as Response
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})
afterEach(() => vi.unstubAllGlobals())

describe('parseGeneration', () => {
  it('returns text with no handoff', () => {
    expect(parseGeneration('Hello there')).toEqual({
      text: 'Hello there',
      handoff: false,
      usage: null,
    })
  })

  it('detects + strips the handoff sentinel', () => {
    expect(parseGeneration('[[HANDOFF]]')).toEqual({
      text: '',
      handoff: true,
      usage: null,
    })
    expect(parseGeneration('Let me get a human [[HANDOFF]]')).toEqual({
      text: 'Let me get a human',
      handoff: true,
      usage: null,
    })
  })

  it('passes usage straight through', () => {
    const usage = { promptTokens: 10, completionTokens: 5, totalTokens: 15 }
    expect(parseGeneration('Hi', usage)).toEqual({
      text: 'Hi',
      handoff: false,
      usage,
    })
  })
})

describe('generateReply — OpenAI', () => {
  it('calls the chat completions endpoint and returns the reply', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        choices: [{ message: { content: 'Sure — happy to help!' } }],
        usage: { prompt_tokens: 42, completion_tokens: 8, total_tokens: 50 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({ provider: 'openai' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Sure — happy to help!',
      handoff: false,
      usage: { promptTokens: 42, completionTokens: 8, totalTokens: 50 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toContain('api.openai.com')
    expect(opts.headers.Authorization).toBe('Bearer sk-test')
  })

  it('maps a 401 to an invalid_key AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, { error: { message: 'Incorrect API key' } }),
      ),
    )

    await expect(
      generateReply({
        config: config(),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('throws on an empty completion', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ choices: [{ message: { content: '' } }] })),
    )
    await expect(
      generateReply({
        config: config(),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toBeInstanceOf(AiError)
  })
})

describe('generateReply — Anthropic', () => {
  it('calls the messages endpoint with the version header and parses text blocks', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        content: [{ type: 'text', text: 'Hi there!' }],
        usage: { input_tokens: 30, output_tokens: 6 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({ provider: 'anthropic', apiKey: 'sk-ant-x' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hello' }],
    })

    // Anthropic reports input/output only — total is summed by normalizeUsage.
    expect(res).toEqual({
      text: 'Hi there!',
      handoff: false,
      usage: { promptTokens: 30, completionTokens: 6, totalTokens: 36 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toContain('api.anthropic.com')
    expect(opts.headers['x-api-key']).toBe('sk-ant-x')
    expect(opts.headers['anthropic-version']).toBeTruthy()
  })

  it('detects handoff in the model output', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({ content: [{ type: 'text', text: '[[HANDOFF]]' }] }),
      ),
    )
    const res = await generateReply({
      config: config({ provider: 'anthropic' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I want to speak to a person' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('')
  })

  it('drops a leading assistant turn so the payload starts on the customer', async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValue(okResponse({ content: [{ type: 'text', text: 'ok' }] }))
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'anthropic' }),
      systemPrompt: 'sys',
      messages: [
        { role: 'assistant', content: 'Welcome!' },
        { role: 'user', content: 'Hi' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.messages[0].role).toBe('user')
    expect(body.messages).toHaveLength(1)
  })
})

describe('generateReply — Gemini', () => {
  it('calls the OpenAI-compatible endpoint with Bearer auth and returns the reply', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        choices: [{ message: { content: 'Hello from Gemini!' } }],
        usage: { prompt_tokens: 11, completion_tokens: 4, total_tokens: 15 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({ provider: 'gemini', apiKey: 'AIza-test', model: 'gemini-x' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Hello from Gemini!',
      handoff: false,
      usage: { promptTokens: 11, completionTokens: 4, totalTokens: 15 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe(
      'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions',
    )
    expect(opts.headers.Authorization).toBe('Bearer AIza-test')
    // Chat Completions request shape: system prompt first, then turns.
    const body = JSON.parse(opts.body)
    expect(body.model).toBe('gemini-x')
    expect(body.messages[0]).toEqual({ role: 'system', content: 'sys' })
    expect(body.messages[1]).toEqual({ role: 'user', content: 'Hi' })
    expect(body.max_completion_tokens).toBe(MAX_OUTPUT_TOKENS)
  })

  it('maps a 401 to an invalid_key AiError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, { error: { message: 'API key not valid' } }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'gemini' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('detects the handoff sentinel in the model output', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          choices: [{ message: { content: '[[HANDOFF]]' } }],
          usage: { prompt_tokens: 5, completion_tokens: 1, total_tokens: 6 },
        }),
      ),
    )

    const res = await generateReply({
      config: config({ provider: 'gemini' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I want to speak to a person' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('')
    expect(res.usage?.totalTokens).toBe(6)
  })

  it('throws empty_response on malformed / missing text', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ choices: [] })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'gemini' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })
  })
})

describe('generateReply — dispatch', () => {
  it('rejects an unsupported provider', async () => {
    await expect(
      generateReply({
        config: config({ provider: 'palm' as AiProvider }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'unsupported_provider', status: 400 })
  })
})

describe('generateReply — OpenRouter', () => {
  it('pins the documented development default model', () => {
    expect(AI_PROVIDER_DEFAULT_MODEL.openrouter).toBe(
      'google/gemma-4-26b-a4b-it:free',
    )
  })

  it('calls the exact endpoint with Bearer auth, JSON content type, and the default model', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        choices: [{ message: { content: 'Hello from OpenRouter!' } }],
        usage: { prompt_tokens: 12, completion_tokens: 3, total_tokens: 15 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({
        provider: 'openrouter',
        apiKey: 'sk-or-v1-test',
        model: AI_PROVIDER_DEFAULT_MODEL.openrouter,
      }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Hello from OpenRouter!',
      handoff: false,
      usage: { promptTokens: 12, completionTokens: 3, totalTokens: 15 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe('https://openrouter.ai/api/v1/chat/completions')
    expect(opts.headers.Authorization).toBe('Bearer sk-or-v1-test')
    expect(opts.headers['Content-Type']).toBe('application/json')
    const body = JSON.parse(opts.body)
    expect(body.model).toBe('google/gemma-4-26b-a4b-it:free')
    expect(body.messages[0]).toEqual({ role: 'system', content: 'sys' })
    expect(body.messages[1]).toEqual({ role: 'user', content: 'Hi' })
    expect(body.max_tokens).toBe(MAX_OUTPUT_TOKENS)
  })

  it('passes any custom model slug through untouched (model stays free-text)', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({ choices: [{ message: { content: 'ok' } }] }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'openrouter', model: 'vendor/custom-slug-7b' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(JSON.parse(fetchMock.mock.calls[0][1].body).model).toBe(
      'vendor/custom-slug-7b',
    )
  })

  it('preserves the system prompt and merges consecutive turns', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({ choices: [{ message: { content: 'ok' } }] }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'openrouter' }),
      systemPrompt: 'be brief',
      messages: [
        { role: 'user', content: 'one' },
        { role: 'user', content: 'two' },
        { role: 'assistant', content: 'three' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.messages[0]).toEqual({ role: 'system', content: 'be brief' })
    expect(body.messages[1]).toEqual({ role: 'user', content: 'one\n\ntwo' })
    expect(body.messages[2]).toEqual({ role: 'assistant', content: 'three' })
  })

  it('returns null usage when the response omits it (never invents values)', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(okResponse({ choices: [{ message: { content: 'ok' } }] })),
    )

    const res = await generateReply({
      config: config({ provider: 'openrouter' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })
    expect(res.usage).toBeNull()
  })

  it('maps a 401 to invalid_key', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, {
          error: { code: 401, message: 'Invalid OpenRouter API key' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('maps a 402 to insufficient_credits with the provider detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(402, {
          error: { code: 402, message: 'Insufficient credits to run this model' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'insufficient_credits', status: 502 })
  })

  it('maps a 429 to rate_limited', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(429, {
          error: { code: 429, message: 'Free-model rate limit reached' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'rate_limited', status: 502 })
  })

  it('maps a 5xx to provider_error and surfaces only the safe detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(503, {
          error: { message: 'upstream temporarily unavailable' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({
      code: 'provider_error',
      status: 502,
      // Safe detail — never headers or the API key.
      message: expect.stringContaining('upstream temporarily unavailable'),
    })
  })

  it('treats malformed JSON as empty_response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.reject(new Error('Unexpected token < in JSON')),
      } as unknown as Response),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })
  })

  it('throws empty_response on missing / blank content', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ choices: [{ message: {} }] })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({ choices: [{ message: { content: '   ' } }] }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toBeInstanceOf(AiError)
  })

  it('maps a fetch rejection to network_error', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockRejectedValue(new Error('getaddrinfo ENOTFOUND openrouter.ai')),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'network_error', status: 502 })
  })

  it('maps a timeout rejection to timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockRejectedValue(
          new DOMException(
            'The operation was aborted due to timeout',
            'TimeoutError',
          ),
        ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'openrouter' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'timeout', status: 504 })
  })

  it('still parses + strips the handoff sentinel', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          choices: [
            { message: { content: 'Let me get a human [[HANDOFF]]' } },
          ],
          usage: { prompt_tokens: 5, completion_tokens: 2, total_tokens: 7 },
        }),
      ),
    )

    const res = await generateReply({
      config: config({ provider: 'openrouter' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I want to speak to a person' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('Let me get a human')
    expect(res.usage?.totalTokens).toBe(7)
  })
})

describe('generateReply — Groq', () => {
  it('pins the operator-verified development default model', () => {
    expect(AI_PROVIDER_DEFAULT_MODEL.groq).toBe('openai/gpt-oss-20b')
  })

  it('calls the exact endpoint with Bearer auth, JSON content type, and the default model', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({
        choices: [{ message: { content: 'Hello from Groq!' } }],
        usage: { prompt_tokens: 9, completion_tokens: 5, total_tokens: 14 },
      }),
    )
    vi.stubGlobal('fetch', fetchMock)

    const res = await generateReply({
      config: config({
        provider: 'groq',
        apiKey: 'gsk_test-not-real',
        model: AI_PROVIDER_DEFAULT_MODEL.groq,
      }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(res).toEqual({
      text: 'Hello from Groq!',
      handoff: false,
      usage: { promptTokens: 9, completionTokens: 5, totalTokens: 14 },
    })
    const [url, opts] = fetchMock.mock.calls[0]
    expect(url).toBe('https://api.groq.com/openai/v1/chat/completions')
    expect(opts.headers.Authorization).toBe('Bearer gsk_test-not-real')
    expect(opts.headers['Content-Type']).toBe('application/json')
    const body = JSON.parse(opts.body)
    expect(body.model).toBe('openai/gpt-oss-20b')
    expect(body.messages[0]).toEqual({ role: 'system', content: 'sys' })
    expect(body.messages[1]).toEqual({ role: 'user', content: 'Hi' })
    expect(body.max_tokens).toBe(MAX_OUTPUT_TOKENS)
  })

  it('passes any custom Groq model slug through untouched', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({ choices: [{ message: { content: 'ok' } }] }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'groq', model: 'llama-4-scout-17b' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })

    expect(JSON.parse(fetchMock.mock.calls[0][1].body).model).toBe(
      'llama-4-scout-17b',
    )
  })

  it('preserves the system prompt and merges consecutive turns', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      okResponse({ choices: [{ message: { content: 'ok' } }] }),
    )
    vi.stubGlobal('fetch', fetchMock)

    await generateReply({
      config: config({ provider: 'groq' }),
      systemPrompt: 'be brief',
      messages: [
        { role: 'assistant', content: 'hello' },
        { role: 'user', content: 'a' },
        { role: 'user', content: 'b' },
      ],
    })

    const body = JSON.parse(fetchMock.mock.calls[0][1].body)
    expect(body.messages[0]).toEqual({ role: 'system', content: 'be brief' })
    expect(body.messages[1]).toEqual({ role: 'assistant', content: 'hello' })
    expect(body.messages[2]).toEqual({ role: 'user', content: 'a\n\nb' })
  })

  it('returns null usage when the response omits it', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockResolvedValue(okResponse({ choices: [{ message: { content: 'ok' } }] })),
    )

    const res = await generateReply({
      config: config({ provider: 'groq' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'Hi' }],
    })
    expect(res.usage).toBeNull()
  })

  it('maps a 401 to invalid_key', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(401, { error: { message: 'Invalid API Key' } }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'invalid_key', status: 401 })
  })

  it('surfaces a 400 malformed-request failure as provider_error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(400, { error: { message: "'messages' is required" } }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({
      code: 'provider_error',
      status: 502,
      message: expect.stringContaining("'messages' is required"),
    })
  })

  it('surfaces a 404 unknown-model failure as provider_error', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(404, {
          error: { code: 'model_not_found', message: 'model not found' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq', model: 'does/not-exist' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'provider_error', status: 502 })
  })

  it('maps a 429 to rate_limited', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(429, {
          error: { code: 'rate_limit_exceeded', message: 'Too many requests' },
        }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'rate_limited', status: 502 })
  })

  it('maps a 5xx to provider_error and surfaces only the safe detail', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        errResponse(500, { error: { message: 'internal server thing' } }),
      ),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({
      code: 'provider_error',
      status: 502,
      // Safe detail — never headers or the API key.
      message: expect.stringContaining('internal server thing'),
    })
  })

  it('maps a fetch rejection to network_error and a timeout to timeout', async () => {
    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockRejectedValue(new Error('getaddrinfo ENOTFOUND api.groq.com')),
    )
    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'network_error', status: 502 })

    vi.stubGlobal(
      'fetch',
      vi
        .fn()
        .mockRejectedValue(
          new DOMException('The operation was aborted due to timeout', 'TimeoutError'),
        ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'timeout', status: 504 })
  })

  it('treats malformed JSON as empty_response', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: true,
        status: 200,
        json: () => Promise.reject(new Error('Unexpected token < in JSON')),
      } as unknown as Response),
    )

    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })
  })

  it('throws empty_response on missing / blank content', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(okResponse({ choices: [] })),
    )
    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toMatchObject({ code: 'empty_response' })

    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({ choices: [{ message: { content: '   ' } }] }),
      ),
    )
    await expect(
      generateReply({
        config: config({ provider: 'groq' }),
        systemPrompt: 'sys',
        messages: [{ role: 'user', content: 'Hi' }],
      }),
    ).rejects.toBeInstanceOf(AiError)
  })

  it('still parses + strips the handoff sentinel', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue(
        okResponse({
          choices: [{ message: { content: '[[HANDOFF]]' } }],
          usage: { prompt_tokens: 4, completion_tokens: 1, total_tokens: 5 },
        }),
      ),
    )

    const res = await generateReply({
      config: config({ provider: 'groq' }),
      systemPrompt: 'sys',
      messages: [{ role: 'user', content: 'I want to speak to a person' }],
    })
    expect(res.handoff).toBe(true)
    expect(res.text).toBe('')
    expect(res.usage?.totalTokens).toBe(5)
  })
})
