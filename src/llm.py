"""The ONLY place that talks to DeepSeek.

Every RAG gets its language model from here. To try a different LLM in the
"different LLM" experiment, change the llm: section of config.yaml.
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
    # DEEPSEEK_BASE_URL is optional; it lets us point at a test server.
    base_url = os.getenv("DEEPSEEK_BASE_URL", c["base_url"])
    extra_body = {"thinking": {"type": c.get("thinking", "disabled")}}
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
    from openai import OpenAI
    from tenacity import retry, stop_after_attempt, wait_random_exponential

    from src.llmcache import LLMCache, get_cache

    c, api_key, base_url, extra_body = _settings(cfg)
    client = OpenAI(api_key=api_key, base_url=base_url)
    cache = get_cache(cfg)

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def chat(system, user, max_tokens=None, temperature=None):
        # Only calls that ask for temperature 0 explicitly are cached (judge, mutation checks); see src/llmcache.py
        key = None
        if cache is not None and temperature is not None and float(temperature) == 0:
            key = LLMCache.key(c["model"], extra_body, system, user, max_tokens, 0)
            hit = cache.get(key)
            if hit is not None:
                USAGE.hit()
                return hit
        kwargs = {}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        response = client.chat.completions.create(
            model=c["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=c.get("temperature", 0.0) if temperature is None else temperature,
            extra_body=extra_body,
            **kwargs,
        )
        USAGE.add(getattr(response, "usage", None))
        text = (response.choices[0].message.content or "").strip()
        if key is not None and text:
            cache.put(key, text)
        return text

    return chat
