"""The ONLY place that talks to the chat LLMs (DeepSeek by default).

Every RAG gets its language model from here. To try a different LLM in the
"different LLM" experiment, change the llm: section of config.yaml. The judge can use a different chat model
(evaluation.judge_llm, default GPT-6 Luna on the OpenAI API; see README 14.12).
"""
import os
import threading

from lightrag.llm.openai import openai_complete_if_cache


class _Usage:
    """Counts the plain (non-LightRAG) LLM calls and tokens, for cost reporting."""

    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.calls = 0
        self.cache_hits = 0  # temperature-0 calls answered from the disk cache (no API call)
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def add(self, usage):
        with self.lock:
            self.calls += 1
            if usage is not None:
                self.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.completion_tokens += getattr(usage, "completion_tokens", 0) or 0

    def hit(self):
        with self.lock:
            self.cache_hits += 1

    def snapshot(self):
        with self.lock:
            return {
                "calls": self.calls,
                "cache_hits": self.cache_hits,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
            }


USAGE = _Usage()


def _settings(cfg):
    c = cfg["llm"]
    api_key = os.getenv(c["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"No API key found. Put {c['api_key_env']}=... in your .env file."
        )
    # DEEPSEEK_BASE_URL is optional; it lets us point at a test server. It only applies to DeepSeek itself, so a
    # judge on another provider (evaluation.judge_llm with its own base_url) is not redirected by it.
    base_url = c["base_url"]
    if "deepseek" in base_url:
        base_url = os.getenv("DEEPSEEK_BASE_URL", base_url)
    # "thinking" is a DeepSeek-only request field; `thinking: null` (set it in judge_llm for other providers) omits it.
    thinking = c.get("thinking", "disabled")
    extra_body = {"thinking": {"type": thinking}} if thinking else None
    return c, api_key, base_url, extra_body


def get_llm_func(cfg):
    """Async version, used by LightRAG."""
    c, api_key, base_url, extra_body = _settings(cfg)

    async def llm_func(
        prompt,
        system_prompt=None,
        history_messages=None,
        keyword_extraction=False,
        entity_extraction=False,
        **kwargs,
    ):
        kwargs.setdefault("temperature", c.get("temperature", 0.0))
        return await openai_complete_if_cache(
            c["model"],
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            keyword_extraction=keyword_extraction,
            entity_extraction=entity_extraction,
            base_url=base_url,
            api_key=api_key,
            extra_body=extra_body,
            **kwargs,
        )

    return llm_func


def get_chat_func(cfg):
    """Plain (non-async) version, used by RAPTOR and MedRAG.
    Usage: chat(system_text, user_text, max_tokens=None, temperature=None) -> answer text"""
    from openai import BadRequestError, OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    from src.llmcache import LLMCache, get_cache

    c, api_key, base_url, extra_body = _settings(cfg)
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = get_cache(cfg)

    # Settings for providers other than DeepSeek (all optional, set them in judge_llm; README 14.12):
    #   token_param        name of the output-limit parameter: max_tokens (default) | max_completion_tokens (OpenAI reasoning models)
    #   reasoning_effort   sent as `reasoning_effort` (e.g. none), so a reasoning model does not spend the output on thinking
    #   min_output_tokens  raise every output limit to at least this (room for reasoning tokens)
    #   send_temperature   false = never send `temperature` (models that only accept their default)
    # If the API rejects one of these parameters, the call is repeated without / with another name and the change is
    # remembered for the rest of the run, so a model with different rules still works.
    effort = c.get("reasoning_effort")
    state = {"token_param": c.get("token_param", "max_tokens"), "temperature": c.get("send_temperature", True),
             "effort": bool(effort)}

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def chat(system, user, max_tokens=None, temperature=None):
        # Only calls that ask for temperature 0 explicitly are cached (judge, mutation checks); see src/llmcache.py
        if max_tokens is not None and c.get("min_output_tokens"):
            max_tokens = max(max_tokens, int(c["min_output_tokens"]))
        key = None
        if cache is not None and temperature is not None and float(temperature) == 0:
            # the cache key only changes when a reasoning setting is in use, so existing DeepSeek entries stay valid
            key = LLMCache.key(c["model"], (extra_body, effort) if effort else extra_body, system, user, max_tokens, 0)
            hit = cache.get(key)
            if hit is not None:
                USAGE.hit()
                return hit
        temp = c.get("temperature", 0.0) if temperature is None else temperature
        for _ in range(4):  # each rejected parameter is changed once, then the call goes through
            kwargs = {}
            if max_tokens is not None:
                kwargs[state["token_param"]] = max_tokens
            if state["temperature"]:
                kwargs["temperature"] = temp
            if state["effort"]:
                kwargs["reasoning_effort"] = effort
            try:
                response = client.chat.completions.create(
                    model=c["model"],
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    extra_body=extra_body,
                    **kwargs,
                )
                break
            except BadRequestError as e:
                msg = str(e).lower()
                if "max_tokens" in msg and state["token_param"] == "max_tokens" and "max_completion_tokens" in msg:
                    state["token_param"] = "max_completion_tokens"
                elif "temperature" in msg and state["temperature"]:
                    state["temperature"] = False
                elif "reasoning_effort" in msg and state["effort"]:
                    state["effort"] = False
                else:
                    raise
        USAGE.add(getattr(response, "usage", None))
        text = (response.choices[0].message.content or "").strip()
        if key is not None and text:
            cache.put(key, text)
        return text

    return chat
