"""Optional DeepSeek V4 client.

Mirrors the shape of core.call_gemini: returns a plain text string and appends
to core.USAGE_LOG. core.call_model dispatches here for the no-PDF clerical
stages when DEEPSEEK_API_KEY is set. Requires the ``openai`` package
(``pip install reviewer2[deepseek]``).

Model names: deepseek-v4-flash, deepseek-v4-pro.
"""

import os
import time
from datetime import datetime

from openai import OpenAI
from openai import APIError, APITimeoutError, RateLimitError, APIConnectionError


# Singleton client — lazy-init so import-time doesn't require the key.
_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY not in environment")
        _client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return _client


def _translate_thinking(thinking_budget, thinking_level):
    """Map Gemini thinking params to DeepSeek extra_body.

    DeepSeek V4 has thinking-on by default. Two critical behaviours:
      - When thinking is enabled, temperature/top_p are silently ignored.
      - reasoning_effort accepts only "high" (default) and "max"; low/medium
        alias up to "high".

    Rule: if the caller asked for thinking (budget != 0 or any level), enable
    it; otherwise disable it so temperature takes effect. Bureaucratic stages
    at temperature=0 must run with thinking disabled or they come back
    stochastic.
    """
    wants_thinking = False
    if thinking_budget is not None:
        try:
            wants_thinking = int(thinking_budget) != 0
        except (TypeError, ValueError):
            wants_thinking = bool(thinking_budget)
    elif thinking_level in ("low", "medium", "high", "max"):
        wants_thinking = True

    if not wants_thinking:
        return {"thinking": {"type": "disabled"}}

    effort = "max" if thinking_level == "max" else "high"
    return {"thinking": {"type": "enabled"}, "reasoning_effort": effort}


def call_deepseek(
    prompt,
    *,
    model,
    temperature=0.0,
    thinking_budget=None,
    thinking_level=None,
    system_instruction=None,
    step=None,
    max_output_tokens=None,
    max_retries=10,
):
    """Call DeepSeek and return text. Appends per-call usage to core.USAGE_LOG.

    Caller is responsible for having injected any needed text substrate into
    `prompt` already — DeepSeek has no PDF vision.
    """
    from reviewer2.core import USAGE_LOG, CURRENT_JOB_UUID  # local import avoids circular dependency

    client = _get_client()

    messages = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt or ""})

    extra_body = _translate_thinking(thinking_budget, thinking_level)

    kwargs = {"model": model, "messages": messages, "extra_body": extra_body}
    # Sampling controls are only respected when thinking is disabled.
    if extra_body.get("thinking", {}).get("type") == "disabled":
        kwargs["temperature"] = temperature
    if max_output_tokens is not None:
        kwargs["max_tokens"] = max_output_tokens

    attempt = 0
    base_delay = 5
    max_delay = 300

    while True:
        attempt += 1
        try:
            resp = client.chat.completions.create(**kwargs)
        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt >= max_retries:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"    ⚠  DeepSeek transient error ({type(e).__name__}); retry in {delay}s")
            time.sleep(delay)
            continue
        except APIError:
            raise

        choice = resp.choices[0] if resp.choices else None
        msg = choice.message if choice else None
        content = (msg.content if msg else "") or ""

        if not content.strip():
            if attempt >= max_retries:
                raise RuntimeError(
                    f"DeepSeek returned empty content after {attempt} attempts "
                    f"(step={step}, model={model})"
                )
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            print(f"    ⚠  DeepSeek empty content (step={step}); retry in {delay}s")
            time.sleep(delay)
            continue

        u = resp.usage
        cache_hit = getattr(u, "prompt_cache_hit_tokens", 0) or 0
        cache_miss = getattr(u, "prompt_cache_miss_tokens", 0) or 0
        prompt_tokens = u.prompt_tokens or 0
        completion_tokens = u.completion_tokens or 0
        reasoning_tokens = 0
        details = getattr(u, "completion_tokens_details", None)
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        USAGE_LOG.append({
            "provider": "deepseek",
            "model_name": model,
            "input_tokens": prompt_tokens,
            "cached_input_tokens": cache_hit,
            "uncached_input_tokens": cache_miss,
            "output_tokens": completion_tokens,
            "thoughts_tokens": reasoning_tokens,
            "timestamp": datetime.now().isoformat(),
            "step": step or "unknown",
            "job_uuid": CURRENT_JOB_UUID,
        })

        return content
