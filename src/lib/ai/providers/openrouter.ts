import { AiError, type ProviderResult } from '../types'
import { MAX_OUTPUT_TOKENS } from '../defaults'
import {
  mergeConsecutive,
  normalizeUsage,
  providerHttpError,
  toNetworkError,
  type ProviderArgs,
  type ProviderHttpErrorOverride,
} from './shared'

// OpenRouter exposes hundreds of models behind an OpenAI-compatible
// Chat Completions surface: standard request shape at their base URL,
// caller's own key as the bearer token. Same deal as the Gemini adapter —
// reuse the OpenAI-style request format rather than inventing a second
// one.
const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions'

interface OpenRouterResponse {
  choices?: { message?: { content?: string | null } }[]
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
    total_tokens?: number
  }
}

// OpenRouter's most actionable failure: 402 means the account is out of
// credit for the requested model (free-tier daily allowance spent, or a
// paid balance exhausted). A generic "API error (402)" would hide that —
// surface a dedicated code so "Test key"/draft UIs can say "top up".
const OPENROUTER_HTTP_ERRORS: Record<number, ProviderHttpErrorOverride> = {
  402: {
    code: 'insufficient_credits',
    base: 'OpenRouter account has insufficient credits for this model',
  },
}

/**
 * Call OpenRouter's Chat Completions endpoint with the caller's own key.
 * Returns the raw assistant text + token usage (handoff parsing happens
 * in `generateReply`).
 */
export async function generateOpenRouter(args: ProviderArgs): Promise<ProviderResult> {
  const { apiKey, model, systemPrompt, messages, timeoutMs } = args

  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    'Content-Type': 'application/json',
  }
  // Attribution headers OpenRouter advertises (shown on their site next
  // to your usage). Purely optional — include them only when the
  // deployment opts in via env, so the integration never depends on
  // configuration being present.
  if (process.env.OPENROUTER_PUBLIC_URL) {
    headers['HTTP-Referer'] = process.env.OPENROUTER_PUBLIC_URL
  }
  if (process.env.OPENROUTER_APP_TITLE) {
    headers['X-Title'] = process.env.OPENROUTER_APP_TITLE
  }

  let res: Response
  try {
    res = await fetch(OPENROUTER_URL, {
      method: 'POST',
      headers,
      body: JSON.stringify({
        model,
        messages: [
          { role: 'system', content: systemPrompt },
          ...mergeConsecutive(messages),
        ],
        // OpenRouter documents this cap parameter under its own name
        // (`max_tokens`) and maps it for every routed vendor — unlike
        // openai.ts, which can use OpenAI-native `max_completion_tokens`.
        max_tokens: MAX_OUTPUT_TOKENS,
      }),
      signal: AbortSignal.timeout(timeoutMs),
    })
  } catch (err) {
    throw toNetworkError(err)
  }

  if (!res.ok) {
    throw await providerHttpError('OpenRouter', res, OPENROUTER_HTTP_ERRORS)
  }

  const data = (await res.json().catch(() => null)) as OpenRouterResponse | null
  const text = data?.choices?.[0]?.message?.content
  if (!text || typeof text !== 'string' || !text.trim()) {
    throw new AiError('OpenRouter returned an empty response.', {
      code: 'empty_response',
    })
  }
  const usage = normalizeUsage({
    prompt: data?.usage?.prompt_tokens,
    completion: data?.usage?.completion_tokens,
    total: data?.usage?.total_tokens,
  })
  return { text, usage }
}
